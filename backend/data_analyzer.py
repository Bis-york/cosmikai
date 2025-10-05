"""Utilities to load and run the trained exoplanet detector model."""
from __future__ import annotations

import argparse
import json
import sys
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    import batman
except ImportError:  # pragma: no cover
    batman = None  # type: ignore

try:
    from astropy.timeseries import BoxLeastSquares
except ImportError:  # pragma: no cover
    BoxLeastSquares = None  # type: ignore


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"
DEFAULT_CHECKPOINT = MODEL_DIR / "trained_exoplanet_detector.pth"
DEFAULT_CONFIDENCE_THRESHOLD = 0.5


class SmallCNN(nn.Module):
    """Tiny 1D CNN used during training."""

    def __init__(self, input_length: int) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (B, 1, L)
        features = self.feature(x)  # (B, 128, 1)
        return self.head(features).squeeze(1)  # (B,)


@dataclass
class ModelBundle:
    model: nn.Module
    metadata: Dict[str, Any]
    device: torch.device


def _resolve_device(device: Optional[str]) -> torch.device:
    if device is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available on this machine.")
    return resolved


def load_detector(
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    *,
    device: Optional[str] = None,
) -> ModelBundle:
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

    target_device = _resolve_device(device)
    payload = torch.load(checkpoint_file, map_location=target_device)

    if isinstance(payload, dict) and "model_state_dict" in payload:
        metadata = dict(payload)
        state_dict = metadata.pop("model_state_dict")
        input_size = int(metadata.get("input_size", 512))
    else:
        metadata = {}
        state_dict = payload
        input_size = 512

    model = SmallCNN(input_size)
    model.load_state_dict(state_dict)
    model.to(target_device)
    model.eval()

    return ModelBundle(model=model, metadata=metadata, device=target_device)


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(number):
        return None
    return number


def _normalize_flux_curve(flux: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
    curve = np.asarray(flux, dtype=np.float32)
    curve -= np.nanmedian(curve)
    curve /= np.nanstd(curve) + 1e-6
    np.nan_to_num(curve, nan=0.0, copy=False)
    return curve


def load_parameter_row(
    csv_path: Union[str, Path],
    *,
    target: Optional[Union[str, int]] = None,
    target_column: str = "kepoi_name",
) -> Dict[str, Any]:
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"Parameter CSV not found: {csv_file}")

    try:
        df = pd.read_csv(csv_file, comment="#", skip_blank_lines=True, low_memory=False)
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Failed to read parameter CSV '{csv_file}': {exc}")

    if df.empty:
        raise ValueError(f"Parameter CSV '{csv_file}' is empty.")

    if target is not None:
        if target_column not in df.columns:
            raise ValueError(
                f"Column '{target_column}' not found in parameter CSV. Available columns: {list(df.columns)}"
            )
        matches = df[df[target_column].astype(str) == str(target)]
        if matches.empty:
            raise ValueError(
                f"No row found for target '{target}' using column '{target_column}' in '{csv_file}'."
            )
        row = matches.iloc[0]
    else:
        if len(df) > 1:
            raise ValueError(
                f"Parameter CSV '{csv_file}' contains multiple rows. Specify a target to disambiguate."
            )
        row = df.iloc[0]

    return row.to_dict()


def _prepare_transit_inputs(raw: Dict[str, Any]) -> Dict[str, Any]:
    weights = {
        "koi_period": 0.25,
        "koi_time0bk": 0.15,
        "koi_duration": 0.20,
        "koi_depth": 0.20,
        "koi_dor": 0.10,
        "koi_impact": 0.05,
        "koi_limbdark": 0.05,
    }
    fidelity_components: Dict[str, Dict[str, Any]] = {}
    fidelity = 0.0

    def _estimate_quality(key: str, value: Optional[float]) -> float:
        if value is None:
            return 0.0
        err_keys = [f"{key}_err1", f"{key}_err2", f"{key}_err"]
        uncertainties: List[float] = []
        for err_key in err_keys:
            err_val = _coerce_float(raw.get(err_key))
            if err_val is not None:
                uncertainties.append(abs(err_val))
        if not uncertainties or value == 0:
            return 1.0
        rel_uncertainty = max(uncertainties) / (abs(value) + 1e-9)
        if rel_uncertainty <= 0.05:
            return 1.0
        if rel_uncertainty >= 0.5:
            return 0.0
        return float(np.clip(1.0 - (rel_uncertainty - 0.05) / (0.5 - 0.05), 0.0, 1.0))

    def _score(name: str, available: bool, quality: float = 1.0) -> None:
        nonlocal fidelity
        weight = weights[name]
        quality_clamped = float(np.clip(quality, 0.0, 1.0)) if available else 0.0
        score = weight * quality_clamped if available else 0.0
        fidelity += score
        fidelity_components[name] = {
            "weight": weight,
            "available": bool(available),
            "quality": quality_clamped,
            "score": score,
        }

    period = None
    period_source = None
    for key in ("koi_period", "period", "period_days"):
        value = _coerce_float(raw.get(key))
        if value is not None and value > 0:
            period = value
            period_source = key
            break
    if period is None or period <= 0:
        raise ValueError("Missing or invalid period in parameter CSV entry.")
    period_quality = _estimate_quality(period_source or "koi_period", period)
    _score("koi_period", True, period_quality)

    duration_days = _coerce_float(raw.get("duration_days"))
    duration_source = "duration_days" if duration_days is not None and duration_days > 0 else None
    original_duration = duration_days
    if duration_days is None or duration_days <= 0:
        duration_hours = _coerce_float(raw.get("koi_duration"))
        if duration_hours is not None and duration_hours > 0:
            original_duration = duration_hours
            duration_days = duration_hours / 24.0
            duration_source = "koi_duration"
    if duration_days is None or duration_days <= 0:
        duration_days = min(0.1 * period, 0.9 * period)
        duration_provided = False
    else:
        duration_days = min(duration_days, 0.9 * period)
        duration_provided = True
    duration_quality = _estimate_quality(duration_source or "koi_duration", original_duration) if duration_provided else 0.0
    _score("koi_duration", duration_provided, duration_quality)

    t0_value = _coerce_float(raw.get("koi_time0bk") or raw.get("t0"))
    t0_provided = t0_value is not None
    t0 = t0_value if t0_value is not None else 0.0
    t0_quality = _estimate_quality("koi_time0bk", t0_value) if t0_provided else 0.0
    _score("koi_time0bk", t0_provided, t0_quality)

    scaled_a_value = _coerce_float(raw.get("koi_dor") or raw.get("a_rs") or raw.get("scaled_semimajor"))
    scaled_a_provided = scaled_a_value is not None and scaled_a_value > 1.0
    scaled_a = scaled_a_value if scaled_a_provided else 10.0
    scaled_a = max(scaled_a, 1.001)
    scaled_a_quality = _estimate_quality("koi_dor", scaled_a_value) if scaled_a_provided else 0.0
    _score("koi_dor", scaled_a_provided, scaled_a_quality)

    impact_value = _coerce_float(raw.get("koi_impact") or raw.get("impact_parameter"))
    impact_provided = impact_value is not None
    impact = float(np.clip(impact_value if impact_value is not None else 0.0, 0.0, max(scaled_a - 1e-3, 0.0)))
    impact_quality = _estimate_quality("koi_impact", impact_value) if impact_provided else 0.0
    _score("koi_impact", impact_provided, impact_quality)

    ecc = _coerce_float(raw.get("koi_eccen") or raw.get("eccentricity"))
    if ecc is None:
        ecc = 0.0

    omega_value = _coerce_float(raw.get("koi_longp") or raw.get("omega"))
    omega = omega_value if omega_value is not None else 90.0

    limb_dark_raw = raw.get("koi_limbdark_mod")
    limb_dark_model = str(limb_dark_raw or "quadratic").strip().lower()
    if limb_dark_model not in {"linear", "quadratic"}:
        limb_dark_model = "quadratic"

    coeff_keys = [key for key in raw.keys() if key.startswith("koi_ldm_coeff")]
    coeffs: List[float] = []
    for key in sorted(coeff_keys):
        value = _coerce_float(raw.get(key))
        if value is not None:
            coeffs.append(value)
    coeffs_provided = bool(coeffs)

    if limb_dark_model == "linear":
        if not coeffs:
            coeffs = [0.3]
        else:
            coeffs = coeffs[:1]
    else:
        if len(coeffs) < 2:
            coeffs = (coeffs + [0.3, 0.2])[:2]
        else:
            coeffs = coeffs[:2]
    limb_dark_provided = bool(limb_dark_raw) and coeffs_provided
    _score("koi_limbdark", limb_dark_provided, 1.0)

    ror_value = _coerce_float(raw.get("koi_ror"))
    depth_key = "koi_ror" if ror_value is not None and ror_value > 0 else "koi_depth"
    if ror_value is not None and ror_value > 0:
        rp = float(max(ror_value, 1e-3))
        depth = rp ** 2
        depth_provided = True
        depth_quality = _estimate_quality("koi_ror", ror_value)
    else:
        depth_raw = _coerce_float(raw.get("koi_depth"))
        if depth_raw is None:
            depth = 1e-4
            depth_provided = False
            depth_quality = 0.0
        else:
            depth = depth_raw if depth_raw < 1 else depth_raw * 1e-6
            depth = max(depth, 1e-6)
            rp = math.sqrt(depth)
            rp = max(rp, 1e-3)
            depth_provided = True
            depth_quality = _estimate_quality("koi_depth", depth_raw)
    if ror_value is not None and ror_value > 0:
        rp = max(ror_value, 1e-3)
        depth = max(rp ** 2, 1e-6)
    else:
        rp = max(math.sqrt(depth), 1e-3)
    _score("koi_depth", depth_provided, depth_quality)

    if impact >= scaled_a:
        impact = max(0.0, scaled_a - 1e-3)

    ratio = impact / scaled_a if scaled_a > 0 else 0.0
    ratio = float(np.clip(ratio, -0.999999, 0.999999))
    inclination = math.degrees(math.acos(ratio))

    return {
        "period": float(period),
        "transit_time": float(t0),
        "duration": float(duration_days),
        "depth": float(depth),
        "rp": float(rp),
        "a": float(scaled_a),
        "impact_parameter": float(impact),
        "eccentricity": float(ecc if ecc is not None else 0.0),
        "omega": float(omega),
        "limb_dark_model": limb_dark_model,
        "limb_dark_coeffs": [float(c) for c in coeffs],
        "inclination": float(inclination),
        "raw_parameters": raw,
        "fidelity": float(round(fidelity, 6)),
        "fidelity_components": fidelity_components,
    }


def _generate_phase_curve_batman(transit: Dict[str, Any], nbins: int) -> np.ndarray:
    if batman is None:
        raise ImportError("batman package is not installed.")

    params = batman.TransitParams()
    params.t0 = transit["transit_time"]
    params.per = transit["period"]
    params.rp = transit["rp"]
    params.a = transit["a"]
    params.inc = transit["inclination"]
    params.ecc = transit["eccentricity"]
    params.w = transit["omega"]
    params.limb_dark = transit["limb_dark_model"]
    params.u = transit["limb_dark_coeffs"]

    phases = np.linspace(0.0, 1.0, nbins, endpoint=False, dtype=np.float32)
    times = transit["transit_time"] + (phases - 0.5) * transit["period"]

    model = batman.TransitModel(
        params,
        times,
        supersample_factor=7,
        exp_time=max(transit["period"] / nbins, 1e-4),
    )
    flux = model.light_curve(params)
    return _normalize_flux_curve(flux)


def _generate_phase_curve_trapezoid(transit: Dict[str, Any], nbins: int) -> np.ndarray:
    period = transit["period"]
    duration = transit["duration"]
    duration = max(min(duration, 0.9 * period), period * 0.01)
    half_duration = duration / 2.0

    impact = transit["impact_parameter"]
    depth = max(transit["depth"], 1e-6)

    phases = np.linspace(0.0, 1.0, nbins, endpoint=False, dtype=np.float32)
    centered = (phases - 0.5) * period

    ingress_fraction = float(np.clip(impact / max(transit["a"], 1.0), 0.05, 0.45))
    ingress_duration = max(min(half_duration, duration * ingress_fraction), duration * 0.05)
    ingress_duration = min(ingress_duration, half_duration - 1e-6) if half_duration > 1e-6 else half_duration

    flux = np.ones_like(centered, dtype=np.float32)
    flat_limit = max(0.0, half_duration - ingress_duration)

    abs_centered = np.abs(centered)
    flat_mask = abs_centered <= flat_limit
    flux[flat_mask] -= depth

    wedge_mask = (abs_centered > flat_limit) & (abs_centered <= half_duration) & (ingress_duration > 0)
    if np.any(wedge_mask):
        slope = (abs_centered[wedge_mask] - flat_limit) / ingress_duration
        flux[wedge_mask] -= depth * (1.0 - slope)

    return _normalize_flux_curve(flux)


def generate_phase_curve_from_parameters(
    raw_params: Dict[str, Any],
    *,
    nbins: int,
) -> Tuple[np.ndarray, Dict[str, Any], str, Optional[str]]:
    if nbins <= 0:
        raise ValueError("nbins must be positive when generating a phase curve.")

    transit = _prepare_transit_inputs(raw_params)

    model_used = "batman"
    warning: Optional[str] = None
    try:
        phase_curve = _generate_phase_curve_batman(transit, nbins)
    except Exception as exc:  # pragma: no cover - fallback path
        warning = str(exc)
        model_used = "trapezoid"
        phase_curve = _generate_phase_curve_trapezoid(transit, nbins)

    summary = {
        "period_days": transit["period"],
        "duration_days": transit["duration"],
        "transit_time": transit["transit_time"],
        "rp_rs": transit["rp"],
        "depth": transit["depth"],
        "scaled_semimajor_axis": transit["a"],
        "impact_parameter": transit["impact_parameter"],
        "eccentricity": transit["eccentricity"],
        "omega_deg": transit["omega"],
        "inclination_deg": transit["inclination"],
        "limb_dark_model": transit["limb_dark_model"],
        "limb_dark_coeffs": transit["limb_dark_coeffs"],
        "fidelity": transit["fidelity"],
        "fidelity_components": transit["fidelity_components"],
    }

    return phase_curve, summary, model_used, warning

def _slugify_target(label: str) -> str:
    if not label:
        return "target"
    cleaned = [c if c.isalnum() or c in {"-", "_"} else "_" for c in label]
    slug = "".join(cleaned).strip("_")
    return slug or "target"


def _export_phase_curve_csv(phase_curve: np.ndarray, output_path: Path) -> Path:
    nbins = len(phase_curve)
    phases = (np.arange(nbins, dtype=np.float32) + 0.5) / max(float(nbins), 1.0)
    df = pd.DataFrame(
        {
            "bin_index": np.arange(nbins, dtype=np.int32),
            "phase": phases.astype(np.float32),
            "normalized_flux": phase_curve.astype(np.float32),
        }
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def _run_detection_from_config(
    config: Dict[str, Any],
    *,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    device: Optional[str] = None,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> Dict[str, Any]:
    config_copy = dict(config)

    bundle = load_detector(checkpoint_path, device=device)

    threshold_value = _coerce_float(config_copy.get("threshold", threshold))
    if threshold_value is None:
        threshold_value = threshold

    nbins_value = config_copy.get("nbins")
    if nbins_value is not None:
        try:
            effective_nbins = int(nbins_value)
        except (TypeError, ValueError):
            raise ValueError("nbins must be an integer.")
    else:
        effective_nbins = int(bundle.metadata.get("input_size", 512))

    parameter_csv_path = Path(config_copy["parameter_csv"]) if config_copy.get("parameter_csv") else None
    csv_file_path = Path(config_copy["csv_path"]) if config_copy.get("csv_path") else None
    parameter_target_column = config_copy.get("parameter_target_column", "kepoi_name")
    target_override = config_copy.get("target_name")
    time_column = config_copy.get("time_column", "time")
    flux_column = config_copy.get("flux_column", "flux")
    phase_curve_output_value = config_copy.get("phase_curve_output")
    output_dir_value = config_copy.get("output_dir")

    phase_model = "folded_light_curve"
    parameter_warning: Optional[str] = None
    transit_parameters: Optional[Dict[str, Any]] = None
    data_source: str = ""
    target_label: Optional[str] = str(target_override) if target_override is not None else None
    phase_curve: Optional[np.ndarray] = None
    period = 0.0
    duration = 0.0
    transit_time = 0.0
    data_points = 0

    if parameter_csv_path:
        print(f"[INFO] Loading transit parameters from CSV: {parameter_csv_path}")
        raw_row = load_parameter_row(
            parameter_csv_path,
            target=target_override,
            target_column=parameter_target_column,
        )
        inferred_target = raw_row.get(parameter_target_column)
        target_label = str(target_override or inferred_target or parameter_csv_path.stem)
        phase_curve, transit_parameters, phase_model, parameter_warning = generate_phase_curve_from_parameters(
            raw_row,
            nbins=effective_nbins,
        )
        period = float(transit_parameters["period_days"])
        duration = float(transit_parameters["duration_days"])
        transit_time = float(transit_parameters["transit_time"])
        data_points = len(phase_curve)
        data_source = f"Parameter CSV: {parameter_csv_path}"
        print(f"[INFO] Generating synthetic transit using {phase_model} model with {effective_nbins} bins...")
        if parameter_warning:
            print(f"[WARN] {parameter_warning}")
    elif csv_file_path:
        print(f"[INFO] Loading light curve from CSV file: {csv_file_path}")
        time, flux = load_csv_light_curve(
            csv_file_path,
            time_column=time_column,
            flux_column=flux_column,
        )
        target_label = str(target_override or csv_file_path.stem)
        data_source = f"CSV file: {csv_file_path}"

        print(f"[INFO] Processing {len(time)} data points...")
        print(f"[INFO] Running Box Least Squares analysis...")
        period, duration, transit_time = estimate_period(time, flux)
        print(f"[INFO] Folding light curve with period {period:.4f} days...")
        phase_curve = fold_light_curve(time, flux, period, transit_time, nbins=effective_nbins)
        data_points = len(time)
        transit_parameters = {
            "period_days": period,
            "duration_days": duration,
            "transit_time": transit_time,
            "fidelity": 1.0,
            "fidelity_components": {},
        }
    else:
        raise ValueError("Configuration must provide either 'parameter_csv' or 'csv_path'.")

    if target_label is None:
        target_label = "unknown-target"

    if phase_curve is None:
        raise RuntimeError("Phase curve data was not generated.")

    if phase_curve_output_value:
        phase_output_path = Path(phase_curve_output_value)
    else:
        base_dir = Path(output_dir_value) if output_dir_value else ROOT_DIR / "outputs"
        slug = _slugify_target(str(target_label))
        phase_output_path = Path(base_dir) / f"{slug}_phase_curve.csv"

    exported_phase_path = _export_phase_curve_csv(phase_curve, phase_output_path)

    print("[INFO] Running neural network inference...")
    confidence, logit = score_phase_curve(phase_curve, bundle)
    has_candidate = confidence >= threshold_value

    fidelity = float(transit_parameters.get("fidelity", 1.0)) if transit_parameters else 1.0
    fidelity = float(np.clip(fidelity, 0.0, 1.0))
    confidence_adjusted = float((confidence - 0.5) * fidelity + 0.5)
    fidelity_components = transit_parameters.get("fidelity_components") if transit_parameters else None

    result = {
        "target": str(target_label),
        "data_source": data_source,
        "confidence": confidence,
        "confidence_adjusted": confidence_adjusted,
        "logit": logit,
        "period_days": period,
        "duration_days": duration,
        "transit_time": transit_time,
        "nbins": effective_nbins,
        "device": bundle.device.type,
        "threshold": threshold_value,
        "has_candidate": has_candidate,
        "assessment": "LIKELY PLANET CANDIDATE" if has_candidate else "NO CLEAR TRANSIT SIGNAL",
        "data_points": data_points,
        "config_used": config_copy,
        "metadata": bundle.metadata,
        "phase_model": phase_model,
        "fidelity": fidelity,
        "phase_curve_output": str(exported_phase_path),
        "phase_curve_points": len(phase_curve),
    }

    if fidelity_components is not None:
        result["fidelity_components"] = fidelity_components

    if transit_parameters is not None:
        result["transit_parameters"] = transit_parameters
    if parameter_warning:
        result["phase_model_warning"] = parameter_warning
    if parameter_csv_path:
        result["parameter_source"] = str(parameter_csv_path)
    if csv_file_path:
        result["light_curve_source"] = str(csv_file_path)

    return result

def load_csv_light_curve(
    csv_path: Union[str, Path],
    *,
    time_column: str = "time",
    flux_column: str = "flux",
    min_points: int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load light curve data from a CSV file."""
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file '{csv_file}': {e}")
    
    if time_column not in df.columns:
        raise ValueError(f"Time column '{time_column}' not found in CSV. Available columns: {list(df.columns)}")
    if flux_column not in df.columns:
        raise ValueError(f"Flux column '{flux_column}' not found in CSV. Available columns: {list(df.columns)}")
    
    # Extract and clean data
    time = df[time_column].values.astype(np.float32)
    flux = df[flux_column].values.astype(np.float32)
    
    # Remove NaN values
    mask = np.isfinite(time) & np.isfinite(flux)
    time = time[mask]
    flux = flux[mask]
    
    if len(flux) < min_points:
        raise ValueError(f"Insufficient data points in CSV: {len(flux)} < {min_points}")
    
    # Normalize flux
    baseline = np.nanmedian(flux)
    flux = flux / (baseline + 1e-6)
    flux = (flux - np.nanmedian(flux)) / (np.nanstd(flux) + 1e-6)
    
    return time, flux


def estimate_period(
    time: np.ndarray,
    flux: np.ndarray,
    *,
    period_bounds: Tuple[float, float] = (0.5, 20.0),
    duration_bounds: Tuple[float, float] = (0.0005, 0.1),
    period_samples: int = 5000,
    duration_samples: int = 20,
) -> Tuple[float, float, float]:
    _require_bls()
    if BoxLeastSquares is None:
        raise ImportError(
            "astropy.timeseries.BoxLeastSquares is required. Install astropy with 'pip install astropy'."
        )

    periods = np.linspace(period_bounds[0], period_bounds[1], period_samples).astype(np.float32)
    durations = np.linspace(duration_bounds[0], duration_bounds[1], duration_samples).astype(np.float32)

    bls = BoxLeastSquares(time, flux)
    power = bls.power(periods, durations)
    idx = int(np.nanargmax(power.power))

    return (
        float(power.period[idx]),
        float(power.duration[idx]),
        float(power.transit_time[idx]),
    )


def fold_light_curve(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    transit_time: float,
    *,
    nbins: int,
) -> np.ndarray:
    phase = ((time - transit_time + 0.5 * period) % period) / period
    bins = np.linspace(0.0, 1.0, nbins + 1)
    indices = np.digitize(phase, bins) - 1

    folded = np.empty(nbins, dtype=np.float32)
    for i in range(nbins):
        mask = indices == i
        if not np.any(mask):
            folded[i] = np.nan
        else:
            folded[i] = np.nanmedian(flux[mask])

    finite = folded[np.isfinite(folded)]
    if finite.size == 0:
        folded.fill(0.0)
    else:
        median = float(np.nanmedian(finite))
        np.nan_to_num(folded, nan=median, copy=False)
        folded -= np.nanmedian(folded)
        folded /= np.nanstd(folded) + 1e-6
        np.nan_to_num(folded, nan=0.0, copy=False)

    return folded.astype(np.float32)


def score_phase_curve(
    phase_curve: np.ndarray,
    bundle: ModelBundle,
    *,
    use_amp: Optional[bool] = None,
) -> Tuple[float, float]:
    tensor = torch.from_numpy(phase_curve).float().unsqueeze(0).to(bundle.device)

    should_amp = use_amp if use_amp is not None else bundle.device.type == "cuda"

    with torch.no_grad():
        if should_amp and bundle.device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = bundle.model(tensor)
        else:
            logits = bundle.model(tensor)

    logit = float(logits.squeeze().item())
    confidence = float(torch.sigmoid(logits).squeeze().item())
    return confidence, logit


def process_json_input(
    json_input: Union[str, Dict[str, Any]],
    *,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    device: Optional[str] = None,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> Dict[str, Any]:
    """Process JSON input to determine if a target is an exoplanet candidate.

    JSON format:
    {
        "target_name": "Kepler-10",              # Optional label used in outputs
        "parameter_csv": "params.csv",           # Transit parameter CSV (use instead of csv_path)
        "parameter_target_column": "kepoi_name", # Column used to match target_name in parameter CSV
        "csv_path": null,                        # Raw light curve CSV (mutually exclusive with parameter_csv)
        "time_column": "time",                   # Optional column name for raw light curve CSV
        "flux_column": "flux",                   # Optional column name for raw light curve CSV
        "phase_curve_output": "outputs/kepler10_phase_curve.csv",  # Optional explicit output path
        "output_dir": "outputs",                # Optional directory for auto-named exports
        "nbins": 512,                            # Optional phase bin count
        "threshold": 0.5                         # Optional decision threshold
    }
    """
    if isinstance(json_input, str):
        try:
            config = json.loads(json_input)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON input: {exc}")
    else:
        config = dict(json_input)

    return _run_detection_from_config(
        config,
        checkpoint_path=checkpoint_path,
        device=device,
        threshold=threshold,
    )


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate or fold light curves and score them with the trained detector.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''CSV-focused Usage Examples:\n\n1. Synthetic via KOI parameter CSV:\n   python csv.py K00752.01 --parameter-csv data/kepler_params.csv\n\n2. Raw light curve CSV:\n   python csv.py MyTarget --csv-path data/lightcurve.csv --time-column time --flux-column flux\n\n3. JSON configuration:\n   python csv.py --json config.json\n''',
    )

    parser.add_argument(
        "target",
        nargs="?",
        help="Target label used in outputs; optional when supplied via JSON.",
    )
    parser.add_argument(
        "--json",
        help="Path to JSON configuration file or JSON string directly.",
    )
    parser.add_argument(
        "--parameter-csv",
        dest="parameter_csv",
        help="Path to KOI-style parameter CSV for synthetic phase-curve generation.",
    )
    parser.add_argument(
        "--parameter-target-column",
        default="kepoi_name",
        help="Column name used to match the target inside the parameter CSV (default: kepoi_name).",
    )
    parser.add_argument(
        "--csv-path",
        dest="csv_path",
        help="Path to raw light curve CSV containing time and flux columns.",
    )
    parser.add_argument(
        "--time-column",
        default="time",
        help="Time column name for raw light curve CSVs (default: time).",
    )
    parser.add_argument(
        "--flux-column",
        default="flux",
        help="Flux column name for raw light curve CSVs (default: flux).",
    )
    parser.add_argument(
        "--phase-curve-output",
        dest="phase_curve_output",
        help="Optional explicit path for the exported phase-curve CSV.",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        help="Directory used when auto-generating the phase-curve CSV (default: <repo>/outputs).",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="Path to the trained model checkpoint (default: models/trained_exoplanet_detector.pth).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Force a torch device string, e.g. 'cpu' or 'cuda:0'.",
    )
    parser.add_argument(
        "--nbins",
        type=int,
        default=None,
        help="Override number of phase bins (defaults to training setting).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Decision threshold applied to the confidence score (default: 0.5).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_cli()
    args = parser.parse_args(argv)

    try:
        if args.json:
            json_path = Path(args.json)
            if json_path.exists():
                print(f"[INFO] Loading configuration from JSON file: {json_path}")
                json_input = json_path.read_text()
            else:
                print("[INFO] Using JSON string input")
                json_input = args.json

            result = process_json_input(
                json_input,
                checkpoint_path=Path(args.checkpoint),
                device=args.device,
                threshold=args.threshold,
            )
        else:
            if not args.parameter_csv and not args.csv_path:
                print("[ERROR] Provide --parameter-csv or --csv-path (or use --json).", file=sys.stderr)
                return 1

            config = {
                "target_name": args.target,
                "parameter_csv": args.parameter_csv,
                "parameter_target_column": args.parameter_target_column,
                "csv_path": args.csv_path,
                "time_column": args.time_column,
                "flux_column": args.flux_column,
                "phase_curve_output": args.phase_curve_output,
                "output_dir": args.output_dir,
                "nbins": args.nbins,
                "threshold": args.threshold,
            }
            config = {k: v for k, v in config.items() if v is not None}

            result = _run_detection_from_config(
                config,
                checkpoint_path=Path(args.checkpoint),
                device=args.device,
                threshold=args.threshold,
            )

    except Exception as exc:  # pragma: no cover
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print("\n" + "=" * 70)
    print("EXOPLANET DETECTION RESULTS")
    print("=" * 70)
    print(f"Target: {result['target']}")
    print(f"Data source: {result.get('data_source', 'N/A')}")
    print(f"Phase curve CSV: {result.get('phase_curve_output', 'N/A')}")
    print(f"Phase bins used: {result['nbins']}")
    print(f"Data points processed: {result.get('data_points', 'N/A')}")

    confidence = result['confidence']
    confidence_adjusted = result.get('confidence_adjusted', confidence)
    print(f"Confidence score: {confidence:.3f}")
    print(f"Adjusted confidence: {confidence_adjusted:.3f}")
    print(f"Decision threshold: {result['threshold']:.3f}")
    print(f"Fidelity: {result.get('fidelity', float('nan')):.3f}")
    print(f"Assessment: {result.get('assessment', 'LIKELY PLANET CANDIDATE' if result['has_candidate'] else 'NO CLEAR TRANSIT SIGNAL')}")

    effective_conf = confidence_adjusted
    if effective_conf >= 0.8:
        interpretation = "VERY HIGH confidence - strong exoplanet candidate"
    elif effective_conf >= 0.6:
        interpretation = "HIGH confidence - likely exoplanet candidate"
    elif effective_conf >= 0.4:
        interpretation = "MEDIUM confidence - possible candidate, needs further investigation"
    elif effective_conf >= 0.2:
        interpretation = "LOW confidence - weak signal, probably not a planet"
    else:
        interpretation = "VERY LOW confidence - no significant transit signal detected"
    print(f"Interpretation: {interpretation}")

    print()
    print("TECHNICAL DETAILS:")
    print("-" * 30)
    print(f"Raw logit score: {result['logit']:.3f}")
    print(f"Estimated period: {result['period_days']:.4f} days")
    print(f"Estimated duration: {result['duration_days']:.4f} days")
    print(f"Transit time: {result['transit_time']:.4f} days")
    print(f"Compute device: {result['device']}")

    if 'config_used' in result:
        print(f"Configuration: {json.dumps(result['config_used'], indent=2)}")

    print("=" * 70)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())  # pragma: no cover
    raise SystemExit(main())































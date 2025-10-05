from __future__ import annotations

import argparse
import json
import sys
import math
from time import perf_counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# FastAPI imports
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# MongoDB imports (assuming these exist from newmain.py)
try:
    from backend.newMongo import (
        database_status,
        get_cached_result,
        list_cached_targets,
        save_result,
    )
    MONGO_AVAILABLE = True
except ImportError:
    print("Warning: MongoDB functionality not available")
    MONGO_AVAILABLE = False

# Optional dependencies
try:
    import batman
except ImportError:
    batman = None

try:
    from astropy.timeseries import BoxLeastSquares
except ImportError:
    BoxLeastSquares = None

# Constants
ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_CHECKPOINT = MODEL_DIR / "trained_exoplanet_detector.pth"
DEFAULT_CONFIDENCE_THRESHOLD = 0.5

# Neural Network Model
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

# Target extraction keys
_TargetKeys = ("target", "target_name", "star_id", "object_id")

# Utility Functions
def _resolve_device(device: Optional[str]) -> torch.device:
    if device is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available on this machine.")
    return resolved

def _extract_target(config: Dict[str, Any]) -> str:
    """Extract target identifier from configuration."""
    for key in _TargetKeys:
        if key not in config:
            continue
        raw_value = config[key]
        if raw_value is None:
            continue
        if isinstance(raw_value, str):
            candidate = raw_value.strip()
            if candidate:
                return candidate
        else:
            candidate = str(raw_value).strip()
            if candidate:
                return candidate
    raise ValueError("Configuration must include a target identifier (e.g. 'target_name').")

def _coerce_float(value: Any) -> Optional[float]:
    """Convert value to float, handling various edge cases."""
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
    """Normalize flux curve for model input."""
    curve = np.asarray(flux, dtype=np.float32)
    curve -= np.nanmedian(curve)
    curve /= np.nanstd(curve) + 1e-6
    np.nan_to_num(curve, nan=0.0, copy=False)
    return curve

def load_detector(
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    *,
    device: Optional[str] = None,
) -> ModelBundle:
    """Load the exoplanet detection model."""
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

def load_parameter_row(
    csv_path: Union[str, Path],
    *,
    target: Optional[Union[str, int]] = None,
    target_column: str = "kepoi_name",
) -> Dict[str, Any]:
    """Load parameter data from CSV file."""
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"Parameter CSV not found: {csv_file}")

    try:
        df = pd.read_csv(csv_file, comment="#", skip_blank_lines=True, low_memory=False)
    except Exception as exc:
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
    """Prepare transit parameters for phase curve generation."""
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

    # Extract period
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

    # Extract duration
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

    # Extract transit time
    t0_value = _coerce_float(raw.get("koi_time0bk") or raw.get("t0"))
    t0_provided = t0_value is not None
    t0 = t0_value if t0_value is not None else 0.0
    t0_quality = _estimate_quality("koi_time0bk", t0_value) if t0_provided else 0.0
    _score("koi_time0bk", t0_provided, t0_quality)

    # Extract depth/radius ratio
    ror_value = _coerce_float(raw.get("koi_ror"))
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

    _score("koi_depth", depth_provided, depth_quality)

    # Extract scaled semi-major axis
    scaled_a_value = _coerce_float(raw.get("koi_dor") or raw.get("a_rs") or raw.get("scaled_semimajor"))
    scaled_a_provided = scaled_a_value is not None and scaled_a_value > 1.0
    scaled_a = scaled_a_value if scaled_a_provided else 10.0
    scaled_a = max(scaled_a, 1.001)
    scaled_a_quality = _estimate_quality("koi_dor", scaled_a_value) if scaled_a_provided else 0.0
    _score("koi_dor", scaled_a_provided, scaled_a_quality)

    # Extract impact parameter
    impact_value = _coerce_float(raw.get("koi_impact") or raw.get("impact_parameter"))
    impact_provided = impact_value is not None
    impact = float(np.clip(impact_value if impact_value is not None else 0.0, 0.0, max(scaled_a - 1e-3, 0.0)))
    impact_quality = _estimate_quality("koi_impact", impact_value) if impact_provided else 0.0
    _score("koi_impact", impact_provided, impact_quality)

    # Other parameters
    ecc = _coerce_float(raw.get("koi_eccen") or raw.get("eccentricity"))
    if ecc is None:
        ecc = 0.0

    omega_value = _coerce_float(raw.get("koi_longp") or raw.get("omega"))
    omega = omega_value if omega_value is not None else 90.0

    # Limb darkening
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

    # Calculate inclination
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

def _generate_phase_curve_trapezoid(transit: Dict[str, Any], nbins: int) -> np.ndarray:
    """Generate phase curve using trapezoid approximation."""
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
    """Generate phase curve from transit parameters."""
    if nbins <= 0:
        raise ValueError("nbins must be positive when generating a phase curve.")

    transit = _prepare_transit_inputs(raw_params)
    model_used = "trapezoid"
    warning: Optional[str] = None

    # Try batman first if available, fall back to trapezoid
    try:
        if batman is not None:
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
            phase_curve = _normalize_flux_curve(flux)
            model_used = "batman"
        else:
            raise ImportError("batman not available")

    except Exception as exc:
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

def _json_ready(value: Any) -> Any:
    """Convert values to JSON-serializable format."""
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(val) for val in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)

def _coerce_config(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Coerce result payload to expected format."""
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("Result payload did not contain any target entries.")

    formatted: Dict[str, Dict[str, Any]] = {}
    for entry in results:
        if not isinstance(entry, dict):
            continue
        target = str(entry.get("target", "unknown")).strip() or "unknown"
        formatted[target] = {k: v for k, v in entry.items() if k != "target"}

    if not formatted:
        raise ValueError("No valid target entries were found in the results list.")

    return formatted

def analyze_exoplanet_from_csv(
    config: Dict[str, Any],
    *,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    device: Optional[str] = None,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> Dict[str, Any]:
    """Analyze exoplanet data from CSV files and parameters."""

    start_time = perf_counter()

    # Load model
    bundle = load_detector(checkpoint_path, device=device)

    # Extract configuration
    parameter_csv_path = config.get("parameter_csv")
    if not parameter_csv_path:
        raise ValueError("parameter_csv path is required")

    target_name = config.get("target_name")
    parameter_target_column = config.get("parameter_target_column", "kepoi_name")

    # Load parameter data
    raw_row = load_parameter_row(
        parameter_csv_path,
        target=target_name,
        target_column=parameter_target_column
    )

    # Generate phase curve
    nbins = int(bundle.metadata.get("input_size", 512))
    phase_curve, transit_parameters, model_used, warning = generate_phase_curve_from_parameters(
        raw_row, nbins=nbins
    )

    # Run model prediction
    phase_tensor = torch.from_numpy(phase_curve).unsqueeze(0).to(bundle.device)

    with torch.no_grad():
        logit = bundle.model(phase_tensor).item()
        confidence = torch.sigmoid(torch.tensor(logit)).item()
        has_candidate = confidence >= threshold

    # Extract target name
    target_label = target_name or raw_row.get(parameter_target_column) or "unknown"

    # Build result
    elapsed_seconds = perf_counter() - start_time

    result = {
        "target": target_label,
        "confidence": confidence,
        "logit": logit,
        "has_candidate": has_candidate,
        "planet_probability": confidence,
        "planet_probability_percent": confidence * 100.0,
        "threshold": threshold,
        "period_days": transit_parameters["period_days"],
        "duration_days": transit_parameters["duration_days"],
        "transit_time": transit_parameters["transit_time"],
        "transit_parameters": transit_parameters,
        "model_used": model_used,
        "data_source": f"Parameter CSV: {parameter_csv_path}",
        "nbins": nbins,
        "elapsed_seconds": elapsed_seconds,
    }

    if warning:
        result["warning"] = warning

    # Format for caching
    wrapped_payload = {
        "checkpoint": str(checkpoint_path),
        "device": bundle.device.type,
        "elapsed_seconds": elapsed_seconds,
        "results": [result],
    }

    return wrapped_payload

# MongoDB Mock Functions (if MongoDB not available)
def mock_database_status():
    return {"ok": False, "error": "MongoDB not configured"}

def mock_get_cached_result(target: str):
    return None

def mock_list_cached_targets():
    return []

def mock_save_result(payload: Dict[str, Any]):
    print(f"Mock: Would save result for targets: {[r.get('target') for r in payload.get('results', [])]}")

# Use mock functions if MongoDB not available
if not MONGO_AVAILABLE:
    database_status = mock_database_status
    get_cached_result = mock_get_cached_result
    list_cached_targets = mock_list_cached_targets
    save_result = mock_save_result

# Main Analysis Function
def check_and_analyze_exoplanet(
    config: Dict[str, Any],
    *,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    device: Optional[str] = None,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> Dict[str, Any]:
    """
    Main function that:
    1. Checks if exoplanet data is cached in MongoDB
    2. If cached, returns cached data
    3. If not cached, analyzes CSV data and caches results
    """

    try:
        # Extract target name from config
        target_name = _extract_target(config)
        print(f"Checking for target: {target_name}")

        # Check if already cached in MongoDB
        cached = get_cached_result(target_name)
        if cached:
            print(f"Found cached result for target: {target_name}")
            return {
                "status": "cached",
                "target": target_name,
                "data": _coerce_config(cached),
                "message": f"Exoplanet {target_name} found in database cache"
            }

        # Not cached, analyze the CSV data
        print(f"No cached result found. Analyzing CSV data for target: {target_name}")

        # Analyze exoplanet from CSV
        wrapped_payload = analyze_exoplanet_from_csv(
            config,
            checkpoint_path=checkpoint_path,
            device=device,
            threshold=threshold
        )

        # Cache the results
        print(f"Caching results for target: {target_name}")
        save_result(wrapped_payload)

        return {
            "status": "analyzed",
            "target": target_name,
            "data": _coerce_config(wrapped_payload),
            "message": f"Analyzed and cached exoplanet data for {target_name}"
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": f"Error processing exoplanet analysis: {str(e)}"
        }

# FastAPI Application
app = FastAPI(title="Combined Exoplanet Detection and Caching System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ExoplanetAnalysisRequest(BaseModel):
    parameter_csv: str = Field(..., description="Path to the parameter CSV file")
    target_name: Optional[str] = Field(None, description="Target exoplanet name")
    parameter_target_column: str = Field("kepoi_name", description="Column name for target identification")
    checkpoint: Optional[str] = Field(None, description="Optional model checkpoint path")
    device: Optional[str] = Field(None, description="Optional device specification")
    threshold: float = Field(DEFAULT_CONFIDENCE_THRESHOLD, description="Decision threshold")

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "message": "Combined Exoplanet Detection and Caching System"}

@app.get("/db/status")
async def mongo_status():
    """Check MongoDB database status."""
    status = database_status()
    if status.get("ok"):
        return status
    raise HTTPException(status_code=503, detail=status.get("error", "MongoDB connection unavailable."))

@app.get("/db/targets")
async def list_targets():
    """List all cached targets."""
    return {"targets": list_cached_targets()}

@app.get("/db/targets/{target}")
async def get_cached_target(target: str):
    """Get cached result for a specific target."""
    cached = get_cached_result(target)
    if not cached:
        raise HTTPException(status_code=404, detail=f"No cached entry found for target '{target}'.")

    try:
        return _coerce_config(cached)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/analyze")
async def analyze_exoplanet(request: ExoplanetAnalysisRequest):
    """
    Main endpoint that checks cache and analyzes exoplanet data.
    """
    config = {
        "parameter_csv": request.parameter_csv,
        "target_name": request.target_name,
        "parameter_target_column": request.parameter_target_column,
    }

    checkpoint_path = Path(request.checkpoint) if request.checkpoint else DEFAULT_CHECKPOINT

    result = check_and_analyze_exoplanet(
        config,
        checkpoint_path=checkpoint_path,
        device=request.device,
        threshold=request.threshold
    )

    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])

    return result

# CLI Interface
def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description="Combined Exoplanet Detection and Caching System")
    parser.add_argument("--parameter-csv", required=True, help="Path to parameter CSV file")
    parser.add_argument("--target-name", help="Target exoplanet name")
    parser.add_argument("--target-column", default="kepoi_name", help="Target column name")
    parser.add_argument("--checkpoint", help="Model checkpoint path")
    parser.add_argument("--device", help="Device (cpu/cuda)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD, help="Decision threshold")

    args = parser.parse_args()

    config = {
        "parameter_csv": args.parameter_csv,
        "target_name": args.target_name,
        "parameter_target_column": args.target_column,
    }

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else DEFAULT_CHECKPOINT

    result = check_and_analyze_exoplanet(
        config,
        checkpoint_path=checkpoint_path,
        device=args.device,
        threshold=args.threshold
    )

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        # Run FastAPI server
        import uvicorn
        uvicorn.run("__main__:app", host="0.0.0.0", port=8000, reload=True)

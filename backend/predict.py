"""High-throughput lightkurve inference script for the CosmiKai exoplanet detector."""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

try:  # Local import so tests can run without lightkurve installed
    import lightkurve as lk  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime
    lk = None

try:
    from astropy.timeseries import BoxLeastSquares
except ImportError:  # pragma: no cover
    BoxLeastSquares = None  # type: ignore


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"
DEFAULT_CHECKPOINT = MODEL_DIR / "trained_exoplanet_detector.pth"
DEFAULT_CONFIDENCE_THRESHOLD = 0.5

SECONDS_PER_DAY = 86_400.0
R_SUN_METERS = 6.957e8
R_EARTH_METERS = 6.371e6
M_SUN_KG = 1.98847e30
G_NEWTON = 6.67430e-11
RSUN_TO_REARTH = R_SUN_METERS / R_EARTH_METERS

VERBOSE = True

def _log(message: str) -> None:
    if VERBOSE:
        print(f"[INFO] {message}")


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
        x = x.unsqueeze(1)
        features = self.feature(x)
        return self.head(features).squeeze(1)


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

    _log(f"Loading detector checkpoint from {checkpoint_file}")
    target_device = _resolve_device(device)
    _log(f"Using device: {target_device}")

    try:
        payload = torch.load(checkpoint_file, map_location=target_device, weights_only=True)
        _log("Loaded checkpoint with weights_only=True")
    except TypeError:
        _log("torch.load does not support weights_only; falling back to legacy load.")
        payload = torch.load(checkpoint_file, map_location=target_device)

    if isinstance(payload, dict) and "model_state_dict" in payload:
        metadata = dict(payload)
        state_dict = metadata.pop("model_state_dict")
        input_size = int(metadata.get("input_size", 512))
    else:
        metadata = {}
        state_dict = payload
        input_size = 512

    _log(f"Initializing SmallCNN with input_size={input_size}")
    model = SmallCNN(input_size)
    model.load_state_dict(state_dict)
    model.to(target_device)
    model.eval()
    _log("Detector model ready for inference")

    return ModelBundle(model=model, metadata=metadata, device=target_device)


def estimate_period(
    time: np.ndarray,
    flux: np.ndarray,
    *,
    period_bounds: Tuple[float, float] = (0.5, 20.0),
    duration_bounds: Tuple[float, float] = (0.0005, 0.1),
    period_samples: int = 5000,
    duration_samples: int = 20,
) -> Tuple[float, float, float]:
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


@dataclass
class TargetRequest:
    """Target metadata for pulling light curves from the archive."""

    target: str
    mission: str
    author: Optional[str] = None
    nbins: Optional[int] = None
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    label: Optional[str] = None

    @classmethod
    def from_config(cls, payload: Dict[str, Any]) -> "TargetRequest":
        if "target_name" not in payload:
            raise ValueError("'target_name' is required in lightkurve configuration")
        if "mission" not in payload:
            raise ValueError("'mission' is required in lightkurve configuration")

        return cls(
            target=str(payload["target_name"]),
            mission=str(payload["mission"]),
            author=payload.get("author"),
            nbins=payload.get("nbins"),
            threshold=float(payload.get("threshold", DEFAULT_CONFIDENCE_THRESHOLD)),
            label=payload.get("label"),
        )


@dataclass
class LightCurveDetails:
    time: np.ndarray
    flux_norm: np.ndarray
    flux_rel: np.ndarray
    statistics: Dict[str, Any]
    author_used: Optional[str]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run high-performance inference on one or more lightkurve targets.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to JSON config (single object or list under 'targets').",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Model checkpoint to load (defaults to packaged detector).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device string, e.g. 'cuda:0' or 'cpu'.",
    )
    parser.add_argument(
        "--mission",
        type=str,
        help="Mission name applied to CLI targets (ignored when using --config).",
    )
    parser.add_argument(
        "--author",
        type=str,
        help="Optional author filter applied to CLI targets (ignored when using --config).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of phase curves per forward pass when batching.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of times to repeat the forward pass for benchmarking.",
    )
    parser.add_argument(
        "--disable-amp",
        action="store_true",
        help="Disable automatic mixed precision even when CUDA is available.",
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Use torch.compile for potential speedups (requires PyTorch >= 2.0).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose step-by-step logging.",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        help="Optional path to write inference results as JSON.",
    )
    parser.add_argument(
        "targets",
        nargs="*",
        help="Optional explicit target names (overrides config when provided).",
    )
    return parser.parse_args(argv)


def load_requests(args: argparse.Namespace) -> List[TargetRequest]:
    requests: List[TargetRequest] = []

    if args.config:
        _log(f"Loading targets from config file: {args.config}")
        with args.config.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if isinstance(payload, dict) and "targets" in payload:
            entries: Iterable[Dict[str, Any]] = payload["targets"]  # type: ignore[assignment]
        elif isinstance(payload, list):
            entries = payload  # type: ignore[assignment]
        else:
            entries = [payload]  # type: ignore[list-item]

        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("Each target entry must be a JSON object.")
            requests.append(TargetRequest.from_config(entry))

    if args.targets:
        if not args.mission:
            raise ValueError("--mission is required when specifying targets via CLI.")
        _log(f"Using CLI-provided targets: {args.targets}")
        override_requests = [
            TargetRequest(
                target=str(raw_target),
                mission=args.mission,
                author=args.author,
            )
            for raw_target in args.targets
        ]
        requests = override_requests

    if not requests:
        raise ValueError(
            "No lightkurve targets provided. Use --config or specify target names explicitly."
        )

    _log(f"Prepared {len(requests)} target request(s).")
    return requests


def assign_default_nbins(requests: List[TargetRequest], default_nbins: int) -> None:
    for request in requests:
        if request.nbins is None:
            request.nbins = default_nbins
            _log(f"Setting nbins={default_nbins} for target {request.target}")


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    if hasattr(value, "value"):
        value = value.value  # type: ignore[assignment]
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _extract_star_parameters(lc: Any) -> Dict[str, Optional[float]]:
    meta = getattr(lc, "meta", {}) or {}

    def grab(keys: Sequence[str]) -> Optional[float]:
        for key in keys:
            if key in meta:
                value = _coerce_float(meta.get(key))
                if value is not None:
                    return value
        return None

    return {
        "star_radius_rsun": grab(["RADIUS", "R_STAR", "STAR_RADIUS", "RADIUS1", "RADIUS_PRED"]),
        "star_mass_msun": grab(["MASS", "M_STAR", "STAR_MASS", "MASS1", "MASS_PRED"]),
        "star_teff_k": grab(["TEFF", "TEFF_STAR", "STAR_TEFF", "TEFF1", "TEFF_PRED"]),
    }


def load_light_curve_details(request: TargetRequest, *, min_points: int = 2000) -> LightCurveDetails:
    if lk is None:  # pragma: no cover - handled during runtime execution
        raise ImportError(
            "lightkurve is required but not installed. Install it with 'pip install lightkurve'."
        )

    _log(f"Searching light curve for target {request.target} (mission={request.mission}, author={request.author})")

    def _candidate_authors(mission: str, preferred: Optional[str]) -> List[Optional[str]]:
        mission_key = mission.strip().upper() if mission else ""
        candidates: List[Optional[str]] = []
        if preferred not in (None, ""):
            candidates.append(preferred)
        if mission_key == "TESS":
            for cand in (None, "SPOC", "QLP", "TESS-SPOC"):
                if cand not in candidates:
                    candidates.append(cand)
        elif mission_key in {"KEPLER", "K2"}:
            for cand in (None, "Kepler", "K2"):
                if cand not in candidates:
                    candidates.append(cand)
        else:
            if None not in candidates:
                candidates.append(None)
        return candidates

    search = None
    effective_author: Optional[str] = None
    last_exception: Optional[Exception] = None

    for candidate in _candidate_authors(request.mission, request.author):
        try:
            _log(f"Attempting archive lookup with author={candidate}")
            trial = lk.search_lightcurve(str(request.target), mission=request.mission, author=candidate)
        except Exception as exc:  # pragma: no cover - network/archive issues
            last_exception = exc
            _log(f"Lookup failed for author={candidate}: {type(exc).__name__}: {exc}")
            continue

        if len(trial) == 0:
            _log(f"No files returned for author={candidate}")
            continue

        search = trial
        effective_author = candidate
        _log(f"Found {len(search)} light curve file(s) using author={candidate}")
        break

    if search is None:
        if last_exception is not None:
            raise ValueError(
                f"No light curves found for target '{request.target}' in mission '{request.mission}'. "
                f"Most recent archive error: {last_exception}"
            ) from last_exception
        raise ValueError(
            f"No light curves found for target '{request.target}' in mission '{request.mission}'."
        )

    _log("Downloading and processing light curves (this may take a few minutes)...")
    lc = search.download_all().stitch().remove_nans()  # type: ignore[assignment]
    _log("Download completed")

    _log("Flattening light curve to remove stellar variability...")
    lc = lc.flatten(window_length=401, polyorder=2)

    flux_raw = lc.flux.value.astype(np.float32)
    time = lc.time.value.astype(np.float32)

    if flux_raw.size < min_points:
        raise ValueError(
            f"Insufficient data points for '{request.target}': {flux_raw.size} < {min_points}."
        )

    baseline = float(np.nanmedian(flux_raw))
    if not math.isfinite(baseline) or baseline == 0.0:
        baseline = 1.0

    flux_rel = flux_raw / (baseline + 1e-6)
    rel_median = float(np.nanmedian(flux_rel))
    rel_std = float(np.nanstd(flux_rel) + 1e-6)
    flux_norm = (flux_rel - rel_median) / rel_std
    _log(f"Prepared light curve with {flux_raw.size} data points")

    stats = {
        "baseline_flux": baseline,
        "relative_median": rel_median,
        "relative_std": rel_std,
    }
    star_params = {k: v for k, v in _extract_star_parameters(lc).items() if v is not None}
    stats.update(**star_params)
    # Do not add effective_author to stats, as it is not a float

    return LightCurveDetails(
        time=time,
        flux_norm=flux_norm.astype(np.float32),
        flux_rel=flux_rel.astype(np.float32),
        statistics=stats,
        author_used=effective_author,
    )



def _safe_median(values: np.ndarray) -> Optional[float]:
    if values.size == 0:
        return None
    med = float(np.nanmedian(values))
    if not math.isfinite(med):
        return None
    return med


def compute_diagnostics(
    *,
    time: np.ndarray,
    flux_rel: np.ndarray,
    period: float,
    duration: float,
    transit_time: float,
    stats: Dict[str, Any],
) -> Dict[str, Any]:
    diagnostics: Dict[str, Any] = {}

    if period <= 0 or duration <= 0:
        diagnostics.update(
            {
                "transit_depth_fraction": None,
                "odd_even_match": None,
                "odd_even_depth_delta": None,
                "secondary_depth_fraction": None,
                "no_secondary": None,
                "planet_radius_earth": None,
                "a_over_r_star": None,
                "equilibrium_temp_k": None,
            }
        )
        diagnostics.update({
            "star_radius_rsun": stats.get("star_radius_rsun"),
            "star_mass_msun": stats.get("star_mass_msun"),
            "star_teff_k": stats.get("star_teff_k"),
        })
        return diagnostics

    period_days = float(period)
    duration_days = float(duration)
    transit_fraction = max(duration_days / period_days, 1e-3)
    half_window = min(0.25, transit_fraction / 2.0)

    phase_mod = ((time - transit_time) % period) / period
    cycle_index = np.floor((time - transit_time) / period).astype(int)

    primary_mask = (phase_mod <= half_window) | (phase_mod >= 1.0 - half_window)
    secondary_mask = np.abs(phase_mod - 0.5) <= half_window
    baseline_mask = (~primary_mask) & (np.abs(phase_mod - 0.5) > half_window)

    baseline_level = _safe_median(flux_rel[baseline_mask])
    if baseline_level is None:
        baseline_level = 1.0
    transit_level = _safe_median(flux_rel[primary_mask])
    if transit_level is None:
        transit_level = baseline_level

    transit_depth = max(0.0, baseline_level - transit_level)

    odd_mask = primary_mask & ((cycle_index % 2) != 0)
    even_mask = primary_mask & ((cycle_index % 2) == 0)
    odd_level = _safe_median(flux_rel[odd_mask])
    even_level = _safe_median(flux_rel[even_mask])
    if odd_level is None or even_level is None:
        odd_even_match = None
        odd_even_delta = None
    else:
        odd_even_delta = abs(odd_level - even_level)
        tolerance = max(0.0005, 0.2 * transit_depth)
        odd_even_match = bool(odd_even_delta <= tolerance)

    secondary_level = _safe_median(flux_rel[secondary_mask])
    if secondary_level is None:
        secondary_depth = None
        no_secondary = True
    else:
        secondary_depth = max(0.0, baseline_level - secondary_level)
        no_secondary = bool(secondary_depth < max(0.0005, 0.2 * transit_depth))

    star_radius_rsun = _coerce_float(stats.get("star_radius_rsun"))
    star_mass_msun = _coerce_float(stats.get("star_mass_msun"))
    star_teff_k = _coerce_float(stats.get("star_teff_k"))

    if star_radius_rsun is None or star_radius_rsun <= 0:
        star_radius_rsun = 1.0
    if star_mass_msun is None or star_mass_msun <= 0:
        star_mass_msun = 1.0
    if star_teff_k is None or star_teff_k <= 0:
        star_teff_k = 5778.0

    radius_ratio = math.sqrt(max(transit_depth, 0.0))
    planet_radius_earth = radius_ratio * star_radius_rsun * RSUN_TO_REARTH

    period_seconds = period_days * SECONDS_PER_DAY
    a_meters: Optional[float]
    try:
        a_meters = (
            (G_NEWTON * (star_mass_msun * M_SUN_KG) * period_seconds**2) / (4.0 * math.pi**2)
        ) ** (1.0 / 3.0)
    except OverflowError:  # pragma: no cover - extremely unlikely
        a_meters = None

    if a_meters is not None and star_radius_rsun > 0:
        a_over_r_star = a_meters / (star_radius_rsun * R_SUN_METERS)
    else:
        a_over_r_star = None

    if (a_over_r_star is None or not math.isfinite(a_over_r_star)) and duration_days > 0:
        a_over_r_star = period_days / (math.pi * duration_days)
        a_meters = a_over_r_star * star_radius_rsun * R_SUN_METERS

    if a_meters is not None and a_meters > 0 and star_radius_rsun > 0:
        t_eq = star_teff_k * math.sqrt(
            (star_radius_rsun * R_SUN_METERS) / (2.0 * a_meters)
        ) * (1.0 - 0.3) ** 0.25
    else:
        t_eq = None

    diagnostics.update(
        {
            "transit_depth_fraction": float(transit_depth),
            "odd_even_match": odd_even_match,
            "odd_even_depth_delta": None if odd_even_delta is None else float(odd_even_delta),
            "secondary_depth_fraction": None if secondary_depth is None else float(secondary_depth),
            "no_secondary": no_secondary,
            "planet_radius_earth": float(planet_radius_earth),
            "a_over_r_star": None if a_over_r_star is None else float(a_over_r_star),
            "equilibrium_temp_k": None if t_eq is None else float(t_eq),
            "star_radius_rsun": float(star_radius_rsun),
            "star_mass_msun": float(star_mass_msun),
            "star_teff_k": float(star_teff_k),
        }
    )
    return diagnostics


def prepare_phase_curve(
    request: TargetRequest,
    *,
    nbins: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], Dict[str, Any]]:
    _log(f"Preparing phase curve for target {request.target} with nbins={nbins}")
    details = load_light_curve_details(request)

    period, duration, transit_time = estimate_period(details.time, details.flux_norm)
    _log("Computed period, duration, and transit time estimates")
    phase_curve = fold_light_curve(details.time, details.flux_norm, period, transit_time, nbins=nbins)
    _log("Folded light curve into phase bins")

    bin_centers = (np.arange(nbins, dtype=np.float32) + 0.5) / float(nbins)
    phase_axis = bin_centers - 0.5

    diagnostics = compute_diagnostics(
        time=details.time,
        flux_rel=details.flux_rel,
        period=period,
        duration=duration,
        transit_time=transit_time,
        stats=details.statistics,
    )
    _log("Computed diagnostic metrics")

    data_source = f"{request.mission} archive"
    if details.author_used:
        data_source += f" ({details.author_used})"
    meta = {
        "target": request.target,
        "mission": request.mission,
        "data_source": data_source,
        "period_days": float(period),
        "duration_days": float(duration),
        "transit_time": float(transit_time),
        "nbins": nbins,
        "threshold": request.threshold,
        "label": request.label or request.target,
    }
    meta.update({
        "star_radius_rsun": diagnostics.get("star_radius_rsun"),
        "star_mass_msun": diagnostics.get("star_mass_msun"),
        "star_teff_k": diagnostics.get("star_teff_k"),
    })
    return phase_curve.astype(np.float32).reshape(-1), phase_axis.astype(np.float32).reshape(-1), meta, diagnostics


def batched_inference(
    model: torch.nn.Module,
    device: torch.device,
    phase_curves: np.ndarray,
    *,
    batch_size: int,
    repeats: int,
    use_amp: bool,
) -> Tuple[np.ndarray, np.ndarray, float]:
    tensor = torch.from_numpy(phase_curves).to(device)
    _log(f"Initial tensor shape from numpy: {list(tensor.shape)}")
    
    # Ensure we have a 2D tensor (batch_size, sequence_length)
    while tensor.ndim > 2:
        _log(f"Squeezing extra dimension from tensor shape {list(tensor.shape)}")
        tensor = tensor.squeeze(1)
    
    if tensor.ndim == 1:
        # Single sample, add batch dimension
        tensor = tensor.unsqueeze(0)
        _log(f"Added batch dimension: {list(tensor.shape)}")
    
    if tensor.ndim != 2:
        raise RuntimeError(f"Expected tensor to be 2D after processing, got shape {list(tensor.shape)}")
    
    _log(f"Final 2D tensor shape (batch, sequence): {list(tensor.shape)}")

    last_logits: List[torch.Tensor] = []
    last_confidences: List[torch.Tensor] = []

    start_time = time.perf_counter()
    with torch.inference_mode():
        for _ in range(max(repeats, 1)):
            current_logits: List[torch.Tensor] = []
            current_conf: List[torch.Tensor] = []
            for start in range(0, tensor.shape[0], batch_size):
                stop = min(start + batch_size, tensor.shape[0])
                batch = tensor[start:stop]
                _log(f"Processing batch {start} to {stop}, batch shape: {list(batch.shape)}")
                
                if use_amp and device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        batch_logits = model(batch)
                else:
                    batch_logits = model(batch)
                current_logits.append(batch_logits)
                current_conf.append(batch_logits.sigmoid())
            last_logits = current_logits
            last_confidences = current_conf
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    concat_logits = torch.cat(last_logits).cpu().numpy()
    concat_conf = torch.cat(last_confidences).cpu().numpy()
    
    # Ensure arrays are always 1D, even for single targets
    if concat_logits.ndim == 0:
        concat_logits = np.array([concat_logits.item()])
    elif concat_logits.ndim > 1:
        concat_logits = concat_logits.flatten()
    
    if concat_conf.ndim == 0:
        concat_conf = np.array([concat_conf.item()])
    elif concat_conf.ndim > 1:
        concat_conf = concat_conf.flatten()
    
    _log(f"Confidence array shape: {concat_conf.shape}, Logits array shape: {concat_logits.shape}")
    return concat_conf, concat_logits, elapsed


def run(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    global VERBOSE
    VERBOSE = not args.quiet
    if VERBOSE:
        _log("Verbose logging enabled")
    else:
        print("[INFO] Running in quiet mode")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        _log("Enabled cuDNN benchmark for performance")

    requests = load_requests(args)

    bundle = load_detector(args.checkpoint, device=args.device)
    model = bundle.model

    default_nbins = int(bundle.metadata.get("input_size", 512))
    _log(f"Default nbins derived from checkpoint: {default_nbins}")
    assign_default_nbins(requests, default_nbins)

    if args.torch_compile and hasattr(torch, "compile"):
        try:
            compiled_model = torch.compile(model, mode="reduce-overhead")
            _log("Compiled model with torch.compile")
            model = compiled_model
        except Exception as exc:  # pragma: no cover - depends on runtime support
            print(f"[WARN] torch.compile failed: {exc}. Using original model.")
            model = bundle.model

    if not isinstance(model, torch.nn.Module):
        class CompiledModelWrapper(torch.nn.Module):
            def __init__(self, fn):
                super().__init__()
                self.fn = fn
            def forward(self, *args, **kwargs):  # type: ignore[override]
                return self.fn(*args, **kwargs)
        model = CompiledModelWrapper(model)

    model.eval()
    _log("Model set to eval mode")

    phase_curves: List[np.ndarray] = []
    phase_axes: List[np.ndarray] = []
    metadata: List[Dict[str, Any]] = []
    diagnostics_list: List[Dict[str, Any]] = []
    for request in requests:
        resolved_nbins = int(request.nbins) if request.nbins is not None else default_nbins
        _log(f"Processing target {request.target} with resolved_nbins={resolved_nbins}")
        phase_curve, phase_axis, meta, diagnostics = prepare_phase_curve(
            request,
            nbins=resolved_nbins,
        )
        phase_curves.append(np.asarray(phase_curve, dtype=np.float32).reshape(-1))
        phase_axes.append(np.asarray(phase_axis, dtype=np.float32).reshape(-1))
        metadata.append(meta)
        diagnostics_list.append(diagnostics)

    stacked = np.stack(phase_curves, axis=0).astype(np.float32)
    stacked = stacked.reshape(stacked.shape[0], -1)
    _log(f"Stacked {len(phase_curves)} phase curve(s) for inference; tensor shape {stacked.shape}")

    use_amp = not args.disable_amp
    if use_amp:
        _log("Using automatic mixed precision where available")
    else:
        _log("AMP disabled; running in full precision")
    confidences, logits, elapsed = batched_inference(
        model,
        bundle.device,
        stacked,
        batch_size=max(1, args.batch_size),
        repeats=max(1, args.repeats),
        use_amp=use_amp,
    )
    _log(f"Inference complete in {elapsed:.3f}s")

    per_target_results: List[Dict[str, Any]] = []
    for idx, meta in enumerate(metadata):
        points = np.column_stack((phase_axes[idx], phase_curves[idx])).astype(np.float32)
        diag = diagnostics_list[idx]
        _log(f"Aggregating results for {meta['target']}")
        result = {
            **meta,
            "confidence": float(confidences[idx]),
            "logit": float(logits[idx]),
            "has_candidate": bool(confidences[idx] >= meta["threshold"]),
            "device": bundle.device.type,
            "planet_probability": float(confidences[idx]),
            "planet_probability_percent": float(confidences[idx] * 100.0),
            "odd_even_match": diag.get("odd_even_match"),
            "odd_even_depth_delta": diag.get("odd_even_depth_delta"),
            "no_secondary": diag.get("no_secondary"),
            "secondary_depth_fraction": diag.get("secondary_depth_fraction"),
            "transit_depth_fraction": diag.get("transit_depth_fraction"),
            "size_vs_earth": diag.get("planet_radius_earth"),
            "a_over_r_star": diag.get("a_over_r_star"),
            "equilibrium_temperature_k": diag.get("equilibrium_temp_k"),
            "star_radius_rsun": diag.get("star_radius_rsun"),
            "star_mass_msun": diag.get("star_mass_msun"),
            "star_teff_k": diag.get("star_teff_k"),
            "light_curve_points": points.round(6).tolist(),
        }
        per_target_results.append(result)

    print("=" * 80)
    print(f"Processed {len(per_target_results)} target(s) in {elapsed:.3f}s")
    print("=" * 80)
    for result in per_target_results:
        status = "CANDIDATE" if result["has_candidate"] else "NO CANDIDATE"
        print(
            f"{result['label']}: confidence={result['confidence']:.4f} "
            f"(threshold={result['threshold']:.2f}) -> {status}"
        )
    print("=" * 80)

    if args.save_json:
        _log(f"Writing results to {args.save_json}")
        payload = {
            "checkpoint": str(args.checkpoint),
            "device": bundle.device.type,
            "elapsed_seconds": elapsed,
            "results": per_target_results,
        }
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with args.save_json.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"[INFO] Wrote results to {args.save_json}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())


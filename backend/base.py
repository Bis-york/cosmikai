"""Utilities to load and run the trained exoplanet detector model."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    import lightkurve as lk
except ImportError:  # pragma: no cover
    lk = None  # type: ignore

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


def _require_lightkurve() -> None:
    if lk is None:
        raise ImportError(
            "lightkurve is required but not installed. Install it with 'pip install lightkurve'."
        )


def _require_bls() -> None:
    if BoxLeastSquares is None:
        raise ImportError(
            "astropy.timeseries.BoxLeastSquares is required. Install astropy with 'pip install astropy'."
        )


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


def fetch_light_curve(
    target: Union[str, int],
    *,
    mission: str = "Kepler",
    author: Optional[str] = None,
    min_points: int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    _require_lightkurve()
    if lk is None:
        raise ImportError("lightkurve is required but not installed. Install it with 'pip install lightkurve'.")

    search = lk.search_lightcurve(str(target), mission=mission, author=author)
    if len(search) == 0:
        raise ValueError(f"No light curves found for target '{target}' in mission '{mission}'.")

    lc = search.download_all().stitch().remove_nans() # type: ignore
    lc = lc.flatten(window_length=401, polyorder=2)

    flux = lc.flux.value.astype(np.float32)
    time = lc.time.value.astype(np.float32)

    if flux.size < min_points:
        raise ValueError(
            f"Insufficient data points for '{target}': {flux.size} < {min_points}."
        )

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
            "astropy.timeseries.BoxLeastSquares is required but not installed. Install astropy with 'pip install astropy'."
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
        "target_name": "Kepler-10",  # Required if using lightkurve
        "mission": "Kepler",         # Optional, default "Kepler"
        "author": null,              # Optional
        "use_lightkurve": true,      # true to fetch from archive, false to use CSV
        "csv_path": null,            # Required if use_lightkurve is false
        "time_column": "time",       # Optional, for CSV files
        "flux_column": "flux",       # Optional, for CSV files
        "nbins": 512,               # Optional
        "threshold": 0.5            # Optional
    }
    """
    # Parse JSON input if it's a string
    if isinstance(json_input, str):
        try:
            config = json.loads(json_input)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON input: {e}")
    else:
        config = json_input
    
    # Extract configuration
    use_lightkurve = config.get("use_lightkurve", True)
    nbins = config.get("nbins", None)
    threshold = config.get("threshold", threshold)
    
    # Load the model
    bundle = load_detector(checkpoint_path, device=device)
    effective_nbins = nbins or int(bundle.metadata.get("input_size", 512))
    
    # Get light curve data based on source
    if use_lightkurve:
        if "target_name" not in config:
            raise ValueError("'target_name' is required when use_lightkurve is true")
        
        target = config["target_name"]
        mission = config.get("mission", "Kepler")
        author = config.get("author", None)
        
        print(f"[INFO] Fetching light curve for '{target}' from {mission} mission...")
        time, flux = fetch_light_curve(target, mission=mission, author=author)
        data_source = f"{mission} archive"
        
    else:
        if "csv_path" not in config:
            raise ValueError("'csv_path' is required when use_lightkurve is false")
        
        csv_path = config["csv_path"]
        time_column = config.get("time_column", "time")
        flux_column = config.get("flux_column", "flux")
        
        print(f"[INFO] Loading light curve from CSV file: {csv_path}")
        time, flux = load_csv_light_curve(
            csv_path, 
            time_column=time_column, 
            flux_column=flux_column
        )
        data_source = f"CSV file: {csv_path}"
        target = config.get("target_name", Path(csv_path).stem)
    
    print(f"[INFO] Processing {len(time)} data points...")
    
    # Estimate period using BLS
    print(f"[INFO] Running Box Least Squares analysis...")
    period, duration, transit_time = estimate_period(time, flux)
    
    # Fold the light curve
    print(f"[INFO] Folding light curve with period {period:.4f} days...")
    phase_curve = fold_light_curve(time, flux, period, transit_time, nbins=effective_nbins)
    
    # Score with the model
    print(f"[INFO] Running neural network inference...")
    confidence, logit = score_phase_curve(phase_curve, bundle)
    
    has_candidate = confidence >= threshold
    
    result = {
        "target": str(target),
        "data_source": data_source,
        "confidence": confidence,
        "logit": logit,
        "period_days": period,
        "duration_days": duration,
        "transit_time": transit_time,
        "nbins": effective_nbins,
        "device": bundle.device.type,
        "threshold": threshold,
        "has_candidate": has_candidate,
        "assessment": "LIKELY PLANET CANDIDATE" if has_candidate else "NO CLEAR TRANSIT SIGNAL",
        "data_points": len(time),
        "config_used": config,
        "metadata": bundle.metadata,
    }
    
    return result


def score_target(
    target: Union[str, int],
    *,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    mission: str = "Kepler",
    author: Optional[str] = None,
    nbins: Optional[int] = None,
    device: Optional[str] = None,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> Dict[str, Any]:
    bundle = load_detector(checkpoint_path, device=device)
    effective_nbins = nbins or int(bundle.metadata.get("input_size", 512))

    time, flux = fetch_light_curve(target, mission=mission, author=author)
    period, duration, transit_time = estimate_period(time, flux)
    phase_curve = fold_light_curve(time, flux, period, transit_time, nbins=effective_nbins)
    confidence, logit = score_phase_curve(phase_curve, bundle)

    has_candidate = confidence >= threshold

    return {
        "target": str(target),
        "mission": mission,
        "confidence": confidence,
        "logit": logit,
        "period_days": period,
        "duration_days": duration,
        "transit_time": transit_time,
        "nbins": effective_nbins,
        "device": bundle.device.type,
        "threshold": threshold,
        "has_candidate": has_candidate,
        "metadata": bundle.metadata,
    }


def detect_planet(
    target: Union[str, int],
    *,
    mission: str = "Kepler",
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    author: Optional[str] = None,
    nbins: Optional[int] = None,
    device: Optional[str] = None,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> Tuple[bool, Dict[str, Any]]:
    """Return (has_candidate, details) for convenience in other modules."""

    details = score_target(
        target,
        checkpoint_path=checkpoint_path,
        mission=mission,
        author=author,
        nbins=nbins,
        device=device,
        threshold=threshold,
    )
    return details["has_candidate"], details


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run inference with the trained exoplanet detector.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
JSON Input Format Examples:

1. Using Lightkurve (fetch from archive):
{
    "target_name": "Kepler-10",
    "mission": "Kepler", 
    "use_lightkurve": true,
    "threshold": 0.5
}

2. Using CSV file:
{
    "target_name": "My Star",
    "use_lightkurve": false,
    "csv_path": "/path/to/lightcurve.csv",
    "time_column": "time",
    "flux_column": "flux",
    "threshold": 0.3
}

3. From JSON file:
python base.py --json config.json
        '''
    )
    
    # Make target optional when using JSON
    parser.add_argument(
        "target", 
        nargs="?", 
        help="Target name or identifier, e.g. 'Kepler-10' (not needed with --json)."
    )
    
    # Add JSON input options
    parser.add_argument(
        "--json",
        help="Path to JSON configuration file or JSON string directly.",
    )
    
    # Original arguments (for backward compatibility)
    parser.add_argument(
        "--mission",
        default="Kepler",
        help="Archive mission name passed to lightkurve search (default: Kepler).",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="Path to the trained model checkpoint (default: models/trained_exoplanet_detector.pth).",
    )
    parser.add_argument("--author", default=None, help="Optional data author passed to lightkurve search.")
    parser.add_argument("--device", default=None, help="Force a torch device string, e.g. 'cpu' or 'cuda:0'.")
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
        # Handle JSON input
        if args.json:
            # Check if it's a file path or JSON string
            json_path = Path(args.json)
            if json_path.exists():
                print(f"[INFO] Loading configuration from JSON file: {json_path}")
                with open(json_path, 'r') as f:
                    json_input = f.read()
            else:
                print(f"[INFO] Using JSON string input")
                json_input = args.json
            
            result = process_json_input(
                json_input,
                checkpoint_path=Path(args.checkpoint),
                device=args.device,
                threshold=args.threshold,
            )
            
        else:
            # Handle traditional command-line arguments
            if not args.target:
                print("[ERROR] Target name is required when not using --json", file=sys.stderr)
                return 1
                
            result = score_target(
                args.target,
                checkpoint_path=Path(args.checkpoint),
                mission=args.mission,
                author=args.author,
                nbins=args.nbins,
                device=args.device,
                threshold=args.threshold,
            )
            
    except Exception as exc:  # pragma: no cover
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    # Display results
    print("\n" + "=" * 70)
    print("EXOPLANET DETECTION RESULTS")
    print("=" * 70)
    print(f"Target: {result['target']}")
    
    if 'data_source' in result:
        print(f"Data source: {result['data_source']}")
    elif 'mission' in result:
        print(f"Mission: {result['mission']}")
    
    print(f"Data points processed: {result.get('data_points', 'N/A')}")
    print(f"Confidence score: {result['confidence']:.3f}")
    print(f"Decision threshold: {result['threshold']:.3f}")
    print(f"Assessment: {result.get('assessment', 'LIKELY PLANET CANDIDATE' if result['has_candidate'] else 'NO CLEAR TRANSIT')}")
    
    # Confidence interpretation
    confidence = result['confidence']
    if confidence >= 0.8:
        print("🟢 VERY HIGH confidence - strong exoplanet candidate!")
    elif confidence >= 0.6:
        print("🟢 HIGH confidence - likely exoplanet candidate")
    elif confidence >= 0.4:
        print("🟡 MEDIUM confidence - possible candidate, needs further investigation")
    elif confidence >= 0.2:
        print("🟠 LOW confidence - weak signal, probably not a planet")
    else:
        print("🔴 VERY LOW confidence - no significant transit signal detected")
    
    print()
    print("TECHNICAL DETAILS:")
    print("-" * 30)
    print(f"Raw logit score: {result['logit']:.3f}")
    print(f"Estimated period: {result['period_days']:.4f} days")
    print(f"Estimated duration: {result['duration_days']:.4f} days")
    print(f"Transit time: {result['transit_time']:.4f} days")
    print(f"Phase bins used: {result['nbins']}")
    print(f"Compute device: {result['device']}")
    
    if 'config_used' in result:
        print(f"Configuration: {json.dumps(result['config_used'], indent=2)}")
    
    print("=" * 70)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
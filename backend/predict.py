"""Simplified lightkurve-based inference helpers for CosmiKai."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import logging

import numpy as np

from backend.data_analyzer import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIDENCE_THRESHOLD,
    ModelBundle,
    estimate_period,
    fold_light_curve,
    load_detector,
    score_phase_curve,
)

try:  # lightkurve is an optional dependency for tests
    import lightkurve as lk  # type: ignore
except ImportError:  # pragma: no cover - surfaced at runtime
    lk = None

JsonPayload = Union[str, Path, Dict[str, Any]]


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


@dataclass
class TargetConfig:
    target: str
    mission: str
    author: Optional[str]
    nbins: int
    threshold: float


def _ensure_lightkurve() -> None:
    if lk is None:  # pragma: no cover - handled during execution
        raise ImportError(
            "lightkurve is required for fetching light curves. Install it with 'pip install lightkurve'."
        )


@lru_cache(maxsize=4)
def _load_model(checkpoint: str, device: Optional[str]) -> ModelBundle:
    checkpoint_path = Path(checkpoint)
    logger.info("Loading detector checkpoint from %s (device=%s)", checkpoint_path, device or "auto")
    return load_detector(checkpoint_path, device=device)


def _normalize_light_curve(light_curve: Any) -> Any:
    cleaned = light_curve.remove_nans()
    try:
        cleaned = cleaned.normalize()
    except Exception:  # pragma: no cover - best effort normalisation
        pass
    try:
        cleaned = cleaned.flatten(window_length=401)
    except Exception:  # pragma: no cover - flatten may fail for short curves
        pass
    return cleaned


def _download_light_curve(config: TargetConfig) -> Tuple[np.ndarray, np.ndarray]:
    _ensure_lightkurve()
    search = lk.search_lightcurve( # type: ignore
        config.target,
        mission=config.mission,
        author=config.author,
    )
    if search is None or len(search) == 0:
        raise ValueError(
            f"No light curve found for target='{config.target}' mission='{config.mission}' author='{config.author}'."
        )

    collection = search.download_all()
    if not collection:
        raise RuntimeError(
            f"Failed to download light curve data for target '{config.target}'."
        )

    stitched = collection.stitch() # type: ignore
    cleaned = _normalize_light_curve(stitched)
    time = np.asarray(cleaned.time.value, dtype=np.float32)
    flux = np.asarray(cleaned.flux.value, dtype=np.float32)

    if time.size == 0 or flux.size == 0:
        raise ValueError(
            f"Downloaded light curve for target '{config.target}' is empty."
        )

    return time, flux


def _resolve_config(payload: Dict[str, Any], *, default_threshold: float) -> TargetConfig:
    possible_target_keys = [
        "target_name",
        "target",
        "star_id",
        "object_id",
    ]
    target_value: Optional[str] = None
    for key in possible_target_keys:
        value = payload.get(key)
        if value is not None:
            target_value = str(value).strip()
            if target_value:
                break
    if not target_value:
        raise ValueError("JSON payload must include a target identifier (e.g. 'target_name').")

    mission = str(payload.get("mission", "")).strip()
    if not mission:
        raise ValueError("JSON payload must include a mission field (e.g. 'Kepler' or 'TESS').")

    author = payload.get("author")
    author_value = str(author).strip() if isinstance(author, str) else None

    nbins = payload.get("nbins")
    nbins_value = int(nbins) if nbins is not None else 512
    if nbins_value <= 0:
        raise ValueError("nbins must be a positive integer when provided.")

    threshold = float(payload.get("threshold", default_threshold))
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must be between 0 and 1.")

    return TargetConfig(
        target=target_value,
        mission=mission,
        author=author_value,
        nbins=nbins_value,
        threshold=threshold,
    )


def score_target(
    target: Union[str, int],
    *,
    mission: str,
    author: Optional[str] = None,
    nbins: Optional[int] = None,
    device: Optional[str] = None,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    checkpoint_path: Union[str, Path] = DEFAULT_CHECKPOINT,
) -> Dict[str, Any]:
    config = TargetConfig(
        target=str(target),
        mission=mission,
        author=author,
        nbins=nbins if nbins is not None else 512,
        threshold=float(threshold),
    )

    checkpoint_str = str(Path(checkpoint_path).expanduser().resolve())
    bundle = _load_model(checkpoint_str, device)

    logger.info(
        "Fetching light curve for target=%s mission=%s author=%s",
        config.target,
        config.mission,
        config.author or "(any)",
    )
    time, flux = _download_light_curve(config)
    logger.info("Retrieved %d light curve points", len(flux))
    period_days, duration_days, transit_time = estimate_period(time, flux)
    logger.info(
        "Estimated period=%.4f days, duration=%.4f days, transit_time=%.4f",
        period_days,
        duration_days,
        transit_time,
    )
    phase_curve = fold_light_curve(
        time,
        flux,
        period_days,
        transit_time,
        nbins=config.nbins,
    )
    logger.info("Folded light curve into %d bins", config.nbins)
    phase_axis = np.linspace(0.0, 1.0, config.nbins, endpoint=False, dtype=np.float32)

    confidence, logit = score_phase_curve(phase_curve, bundle)
    has_candidate = confidence >= config.threshold
    logger.info(
        "Inference complete for %s: confidence=%.4f (threshold=%.4f) -> %s",
        config.target,
        confidence,
        config.threshold,
        "candidate" if has_candidate else "no candidate",
    )

    light_curve_points = [
        [float(phase_axis[i]), float(phase_curve[i])] for i in range(config.nbins)
    ]

    return {
        "target": config.target,
        "mission": config.mission,
        "author": config.author,
        "nbins": config.nbins,
        "threshold": config.threshold,
        "confidence": float(confidence),
        "confidence_adjusted": float(confidence),
        "logit": float(logit),
        "has_candidate": bool(has_candidate),
        "period_days": float(period_days),
        "duration_days": float(duration_days),
        "transit_time": float(transit_time),
        "device": bundle.device.type,
        "checkpoint_path": checkpoint_str,
        "data_points": int(time.size),
        "light_curve_points": light_curve_points,
    }


def _coerce_payload(payload: JsonPayload) -> Dict[str, Any]:
    if isinstance(payload, dict):
        return dict(payload)
    if isinstance(payload, Path):
        return json.loads(payload.read_text(encoding="utf-8"))
    if isinstance(payload, str):
        candidate_path = Path(payload)
        if candidate_path.exists():
            return json.loads(candidate_path.read_text(encoding="utf-8"))
        return json.loads(payload)
    raise TypeError("Unsupported payload type for JSON configuration.")


def process_json_input(
    payload: JsonPayload,
    *,
    checkpoint_path: Union[str, Path] = DEFAULT_CHECKPOINT,
    device: Optional[str] = None,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> Dict[str, Any]:
    config_dict = _coerce_payload(payload)
    logger.info("Processing JSON payload for inference: keys=%s", sorted(config_dict.keys()))
    target_config = _resolve_config(config_dict, default_threshold=threshold)
    logger.info(
        "Resolved config -> target=%s mission=%s nbins=%s threshold=%.3f",
        target_config.target,
        target_config.mission,
        target_config.nbins,
        target_config.threshold,
    )

    result = score_target(
        target_config.target,
        mission=target_config.mission,
        author=target_config.author,
        nbins=target_config.nbins,
        device=device,
        threshold=target_config.threshold,
        checkpoint_path=checkpoint_path,
    )

    result["config_used"] = {
        "target": target_config.target,
        "mission": target_config.mission,
        "author": target_config.author,
        "nbins": target_config.nbins,
        "threshold": target_config.threshold,
    }
    return result


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CosmiKai inference for a single target described by JSON.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to a JSON file containing the target configuration.",
    )
    parser.add_argument(
        "--json",
        type=str,
        help="JSON string with the target configuration (alternative to --config).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Model checkpoint to use.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device string, e.g. 'cpu' or 'cuda:0'.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Override decision threshold when the JSON payload omits it.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    if args.json:
        payload: JsonPayload = args.json
    elif args.config:
        payload = args.config
    else:
        raw = sys.stdin.read()
        if not raw.strip():
            raise SystemExit("Provide --config, --json, or pipe JSON via stdin.")
        payload = raw

    result = process_json_input(
        payload,
        checkpoint_path=args.checkpoint,
        device=args.device,
        threshold=args.threshold,
    )

    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

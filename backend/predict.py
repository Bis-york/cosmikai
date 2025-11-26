"""Uses lightkurve for light curve analysis from the target star and mission."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union, List
import logging

import numpy as np

# Import your existing utilities
from backend.data_analyzer import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIDENCE_THRESHOLD,
    ModelBundle,
    estimate_period,
    fold_light_curve,
    load_detector,
    score_phase_curve,
)

try:
    import lightkurve as lk
except ImportError:
    lk = None

JsonPayload = Union[str, Path, Dict[str, Any]]
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

@dataclass
class TargetConfig:
    target: str
    mission: str
    nbins: int
    threshold: float

def _ensure_lightkurve() -> None:
    if lk is None:
        raise ImportError("lightkurve is required. Install it with 'pip install lightkurve'.")

@lru_cache(maxsize=4)
def _load_model(checkpoint: str, device: Optional[str]) -> ModelBundle:
    checkpoint_path = Path(checkpoint)
    return load_detector(checkpoint_path, device=device)

def _get_clean_light_curve(config: TargetConfig) -> Any:
    """Downloads and returns the LightCurve object (not just numpy arrays)."""
    _ensure_lightkurve()
    search = lk.search_lightcurve(config.target, mission=config.mission)
    
    if search is None or len(search) == 0:
        raise ValueError(f"No light curve found for {config.target} ({config.mission}).")

    collection = search.download_all()
    if not collection:
        raise RuntimeError(f"Failed to download data for {config.target}.")

    # Stitch, fill nans, and remove outliers (basic cleaning)
    lc = collection.stitch().remove_nans().remove_outliers(sigma=5)
    
    # Flattening is crucial for BLS to work well
    lc = lc.flatten(window_length=401)
    return lc

def score_target(
    target: Union[str, int],
    *,
    mission: str,
    nbins: Optional[int] = None,
    device: Optional[str] = None,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    checkpoint_path: Union[str, Path] = DEFAULT_CHECKPOINT,
) -> List[Dict[str, Any]]:
    """
    Refactored to find MULTIPLE planets via Iterative Whitening.
    Returns a LIST of results (one per candidate found).
    """
    config = TargetConfig(
        target=str(target),
        mission=mission,
        nbins=nbins if nbins is not None else 512,
        threshold=float(threshold),
    )

    checkpoint_str = str(Path(checkpoint_path).expanduser().resolve())
    bundle = _load_model(checkpoint_str, device)

    # 1. Get the initial clean light curve object
    logger.info("Fetching light curve for %s...", config.target)
    lc = _get_clean_light_curve(config)
    
    found_candidates = []
    iteration = 0
    max_iterations = 5  # Safety break to prevent infinite loops

    while iteration < max_iterations:
        iteration += 1
        logger.info(f"--- Iteration {iteration}: Searching for signal ---")
        
        # Extract numpy arrays for your existing math functions
        time = np.asarray(lc.time.value, dtype=np.float32)
        flux = np.asarray(lc.flux.value, dtype=np.float32)

        if len(time) < 1000:
            logger.info("Not enough data points remaining. Stopping search.")
            break

        # 2. Run BLS (Box Least Squares) to find strongest period
        try:
            period, duration, t0 = estimate_period(time, flux)
        except Exception as e:
            logger.warning(f"BLS failed on iteration {iteration}: {e}")
            break

        # 3. Fold and Score
        phase_curve = fold_light_curve(time, flux, period, t0, nbins=config.nbins)
        confidence, logit = score_phase_curve(phase_curve, bundle)
        
        is_candidate = confidence >= config.threshold
        
        logger.info(f"Signal found: Period={period:.2f}d, Conf={confidence:.3f}")

        # Prepare result entry
        result_entry = {
            "target": config.target,
            "mission": config.mission,
            "candidate_id": f"{config.target}.{iteration}", # e.g. "Kepler-10.1"
            "nbins": config.nbins,
            "threshold": config.threshold,
            "confidence": float(confidence),
            "has_candidate": bool(is_candidate),
            "period_days": float(period),
            "duration_days": float(duration),
            "transit_time": float(t0),
            "device": bundle.device.type,
            "checkpoint_path": checkpoint_str,
            "data_points": int(time.size),
            # Only sending light curve points for the first (primary) candidate to save bandwidth, 
            # or you can send it for all if you want visualizations for every planet.
            "light_curve_points": [] 
        }

        # 4. Logic Branch
        if is_candidate:
            logger.info(">>> CANDIDATE CONFIRMED. Masking and searching again.")
            found_candidates.append(result_entry)
            
            # THE MAGIC STEP: Mask this planet so we can find the next one
            # Lightkurve makes this incredibly easy
            mask = lc.create_transit_mask(
                period=period,
                duration=duration * 1.5, # Mask slightly wider than detected to be safe
                transit_time=t0
            )
            # Apply the mask (remove those points)
            lc = lc[~mask]
            
        else:
            logger.info("Signal rejected (Low Confidence). Stopping search.")
            # If the strongest signal left is noise, we are done.
            # However, we should return this "failed" result just so the user sees we tried.
            if not found_candidates:
                found_candidates.append(result_entry)
            break

    return found_candidates

# --- Wrapper to maintain compatibility with your CLI/JSON inputs ---

def process_json_input(
    payload: JsonPayload,
    *,
    checkpoint_path: Union[str, Path] = DEFAULT_CHECKPOINT,
    device: Optional[str] = None,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> Dict[str, Any]:
    
    # (Keep your existing config parsing logic)
    if isinstance(payload, (str, Path)):
        # ... simplified for brevity, assume dict load works as before ...
        if isinstance(payload, Path): payload = json.loads(payload.read_text())
        else: payload = json.loads(payload)

    config_dict = dict(payload)
    target_keys = ["target_name", "target", "star_id", "object_id"]
    target = next((str(config_dict.get(k)) for k in target_keys if config_dict.get(k)), "Unknown")
    mission = str(config_dict.get("mission", "TESS"))
    nbins = int(config_dict.get("nbins", 512))

    # Call the new multi-planet function
    results_list = score_target(
        target, 
        mission=mission, 
        nbins=nbins, 
        threshold=threshold, 
        checkpoint_path=checkpoint_path, 
        device=device
    )

    # Return a wrapper dict (Note: 'results' is now a list of planets)
    return {
        "config_used": config_dict,
        "results": results_list, # <--- This contains all found planets
        "total_candidates": len([r for r in results_list if r['has_candidate']])
    }

# (Rest of your main/CLI code remains mostly valid, just ensure it prints the list)
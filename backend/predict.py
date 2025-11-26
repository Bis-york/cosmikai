"""Uses lightkurve for light curve analysis from the target star and mission."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
import csv
import requests
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

def _normalize_star_name(name: str) -> str:
    return str(name or "").strip().lower()

def _try_local_kepler_catalog(name: str) -> Optional[str]:
    """Return a best-effort identifier from local Kepler CSVs.

    Prefers returning a KIC identifier string like "KIC 11442793" if found.
    """
    normalized = _normalize_star_name(name)
    candidate_files = [
        Path("data_test/kepler_cumulative.csv"),
        Path("model code/data/kepler_cumulative.csv"),
        Path("model code/data/kepler_cumulative.csv"),
    ]
    for fpath in candidate_files:
        if fpath.exists():
            try:
                with fpath.open("r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Columns vary by dataset; try a few common ones
                        kepler_name = (row.get("kepler_name") or row.get("kepoi_name") or row.get("koi_name") or "").strip()
                        kepid = (row.get("kepid") or row.get("kic_kepler_id") or "").strip()
                        if kepler_name and _normalize_star_name(kepler_name) == normalized and kepid:
                            return f"KIC {int(float(kepid))}"
            except Exception as e:
                logger.debug("Local Kepler catalog read failed from %s: %s", fpath, e)
    return None

def _try_exoplanet_archive(name: str) -> Optional[str]:
    """Query NASA Exoplanet Archive over HTTPS to resolve to KIC/TIC.

    Returns a string identifier like "KIC 11442793" or "TIC 123456789" if found.
    """
    try:
        url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
        n = _normalize_star_name(name).replace("'", "")
        # Try both Kepler name and host/star name
        query = (
            "select top 1 kepler_name, kepid, hostname, sy_ticid "
            "from pscomppars "
            f"where lower(kepler_name)='{n}' or lower(hostname)='{n}'"
        )
        params = {"query": query, "format": "json"}
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and data:
            row = data[0]
            kepid = row.get("kepid")
            ticid = row.get("sy_ticid")
            if kepid:
                return f"KIC {int(kepid)}"
            if ticid:
                # sy_ticid may be float or string
                try:
                    return f"TIC {int(float(ticid))}"
                except Exception:
                    return f"TIC {ticid}"
    except Exception as e:
        logger.debug("Exoplanet Archive lookup failed: %s", e)
    return None

def _resolve_target_identifier(name: str) -> str:
    """Resolve a user-provided star name to a numeric catalog ID if possible.

    This avoids the MAST HTTP name resolver (port 80) which may be blocked.
    Resolution order:
      1) If name already looks like KIC/TIC or is numeric, return as-is
      2) Local Kepler CSV lookup
      3) NASA Exoplanet Archive (HTTPS) lookup
      4) Fallback to original name
    """
    raw = str(name).strip()
    low = raw.lower()
    if low.startswith("kic ") or low.startswith("tic "):
        return raw
    if low.isdigit():
        # Assume KIC if plain digits (Kepler) — better than failing outright
        return f"KIC {raw}"

    ident = _try_local_kepler_catalog(raw)
    if ident:
        logger.info("Resolved '%s' via local catalog to '%s'", raw, ident)
        return ident

    ident = _try_exoplanet_archive(raw)
    if ident:
        logger.info("Resolved '%s' via Exoplanet Archive to '%s'", raw, ident)
        return ident

    logger.info("Using provided target name without resolution: '%s'", raw)
    return raw

def _get_clean_light_curve(config: TargetConfig) -> Any:
    """Downloads and returns the LightCurve object (not just numpy arrays)."""
    _ensure_lightkurve()
    # Resolve to an ID when possible to bypass MAST's HTTP name resolver
    resolved = _resolve_target_identifier(config.target)
    search = lk.search_lightcurve(resolved, mission=config.mission)
    
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
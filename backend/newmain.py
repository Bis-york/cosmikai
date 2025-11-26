from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

from backend.data_analyzer import process_json_input as run_data_analyzer
from backend.newMongo import (
    database_status,
    get_cached_result,
    list_cached_targets,
    prediction_stats,
    save_result,
)
from backend.predict import process_json_input as run_lightcurve

# Default CORS origins for local development and production
DEFAULT_ORIGINS = [
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:4173", "http://127.0.0.1:4173",
    "http://localhost:5180", "http://127.0.0.1:5180",
    "http://localhost:5280", "http://127.0.0.1:5280",
    "https://api.flyingwaffle.ca", "http://api.flyingwaffle.ca",
    "https://visuals.flyingwaffle.ca", "http://visuals.flyingwaffle.ca",
    "https://cosmikai.flyingwaffle.ca", "http://cosmikai.flyingwaffle.ca",
]

def _resolve_allowed_origins() -> list[str]:
    env_value = os.getenv("COSMIKAI_CORS_ORIGINS")
    if env_value:
        origins = [origin.strip() for origin in env_value.split(",") if origin.strip()]
        return origins if origins else ["*"]
    return DEFAULT_ORIGINS


app = FastAPI(title="CosmiKai Prediction Gateway")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_resolve_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


TARGET_KEYS = ("target", "target_name", "star_id", "object_id")


def _extract_target(config: Dict[str, Any]) -> str:
    """Extract and validate target identifier from config."""
    for key in TARGET_KEYS:
        value = config.get(key)
        if value:
            candidate = str(value).strip()
            if candidate:
                return candidate
    raise HTTPException(status_code=400, detail="Configuration must include a target identifier (e.g. 'target_name').")


def _coerce_config(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Transform prediction payload into frontend-compatible format."""
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("Result payload did not contain any target entries.")

    formatted: Dict[str, Dict[str, Any]] = {}
    for entry in results:
        if not isinstance(entry, dict):
            continue
        
        target = str(entry.get("target", "unknown")).strip() or "unknown"
        detail = {k: v for k, v in entry.items() if k != "target"}
        
        # Add metadata from payload if not already present
        metadata_mappings = [
            ("original_target", "original_target"),
            ("timestamp", "cached_timestamp"),
            ("target_aliases", "target_aliases"),
            ("checkpoint", "checkpoint_path"),
            ("device", "device"),
        ]
        for payload_key, detail_key in metadata_mappings:
            if payload.get(payload_key) and detail_key not in detail:
                detail[detail_key] = payload[payload_key]
        
        formatted[target] = detail
    
    if not formatted:
        raise ValueError("No valid target entries were found in the results list.")
    return formatted


class PredictionRequest(BaseModel):
    pipeline: Optional[Literal["lightcurve", "data_analyzer"]] = Field(
        default=None,
        description="Override automatic pipeline selection. Defaults to lightcurve unless csv/parameter data is supplied.",
    )
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration forwarded to the chosen inference pipeline.",
    )
    checkpoint: Optional[str] = Field(
        default=None,
        description="Optional checkpoint path override.",
    )
    device: Optional[str] = Field(
        default=None,
        description="Optional torch device string (e.g. 'cpu' or 'cuda:0').",
    )
    threshold: Optional[float] = Field(
        default=None,
        description="Optional decision threshold override.",
    )

    @validator("config", pre=True)
    def _ensure_dict(cls, value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        raise ValueError("config must be a JSON object")

    def resolved_pipeline(self) -> Literal["lightcurve", "data_analyzer"]:
        """Auto-detect pipeline based on config or use explicit override."""
        if self.pipeline:
            return self.pipeline
        has_csv = any(key in self.config for key in ("parameter_csv", "csv_path"))
        return "data_analyzer" if has_csv else "lightcurve"

    def common_kwargs(self) -> Dict[str, Any]:
        """Build kwargs dict for pipeline functions."""
        return {
            k: v for k, v in [
                ("checkpoint_path", self.checkpoint),
                ("device", self.device),
                ("threshold", self.threshold),
            ] if v is not None
        }


@app.get("/db/status")
async def mongo_status() -> Dict[str, Any]:
    status = database_status()
    if status.get("ok"):
        return status
    raise HTTPException(status_code=503, detail=status.get("error", "MongoDB connection unavailable."))


@app.get("/db/stars")
async def list_stars() -> Dict[str, List[str]]:
    return {"targets": list_cached_targets()}


@app.get("/db/stars/{target}")
async def get_star(target: str) -> Dict[str, Dict[str, Any]]:
    cached = get_cached_result(target)
    if not cached:
        raise HTTPException(status_code=404, detail=f"No cached entry found for target '{target}'.")
    try:
        return _coerce_config(cached)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/db/stats")
async def mongo_stats() -> Dict[str, Any]:
    return prediction_stats()


@app.post("/predict")
async def predict_star(request: PredictionRequest) -> Dict[str, Dict[str, Any]]:
    target_name = _extract_target(request.config)

    cached = get_cached_result(target_name)
    if cached:
        try:
            return _coerce_config(cached)
        except ValueError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    pipeline = request.resolved_pipeline()
    kwargs = request.common_kwargs()

    if pipeline == "lightcurve":
        raw_result = run_lightcurve(request.config, **kwargs)
        wrapped_payload = {
            "checkpoint": raw_result.get("checkpoint_path"),
            "device": raw_result.get("device"),
            "elapsed_seconds": raw_result.get("elapsed_seconds"),
            "results": [raw_result],
        }
    else:
        wrapped_payload = run_data_analyzer(request.config, **kwargs)

    save_result(wrapped_payload)
    try:
        return _coerce_config(wrapped_payload)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


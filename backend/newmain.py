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
    save_result,
)
from backend.predict import process_json_input as run_lightcurve

def _resolve_allowed_origins() -> list[str] | None:
    env_value = os.getenv("COSMIKAI_CORS_ORIGINS")
    if not env_value:
        return [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:4173",
            "http://127.0.0.1:4173",
            "http://localhost:5180",
            "http://127.0.0.1:5180",
            "http://localhost:5280",
            "http://127.0.0.1:5280",
        ]
    origins = [origin.strip() for origin in env_value.split(",") if origin.strip()]
    return origins or None


app = FastAPI(title="CosmiKai Prediction Gateway")

allowed_origins = _resolve_allowed_origins() or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_TargetKeys = ("target", "target_name", "star_id", "object_id")


def _extract_target(config: Dict[str, Any]) -> str:
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
    raise HTTPException(status_code=400, detail="Configuration must include a target identifier (e.g. 'target_name').")


def _coerce_config(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("Result payload did not contain any target entries.")

    formatted: Dict[str, Dict[str, Any]] = {}
    for entry in results:
        if not isinstance(entry, dict):
            continue
        target = str(entry.get("target", "unknown")).strip() or "unknown"
        detail = {k: v for k, v in entry.items() if k != "target"}
        if "original_target" not in detail and payload.get("original_target"):
            detail["original_target"] = payload.get("original_target")
        if payload.get("timestamp") and "cached_timestamp" not in detail:
            detail["cached_timestamp"] = payload.get("timestamp")
        if payload.get("target_aliases"):
            detail.setdefault("target_aliases", payload.get("target_aliases"))
        if payload.get("checkpoint"):
            detail.setdefault("checkpoint_path", payload.get("checkpoint"))
        if payload.get("device"):
            detail.setdefault("device", payload.get("device"))
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
    def _ensure_dict(cls, value: Any) -> Dict[str, Any]:  # noqa: N805 - pydantic validator signature
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        raise ValueError("config must be a JSON object")

    def resolved_pipeline(self) -> Literal["lightcurve", "data_analyzer"]:
        if self.pipeline is not None:
            return self.pipeline
        config_keys = self.config.keys()
        if any(key in config_keys for key in ("parameter_csv", "csv_path")):
            return "data_analyzer"
        return "lightcurve"

    def common_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if self.checkpoint:
            kwargs["checkpoint_path"] = self.checkpoint
        if self.device:
            kwargs["device"] = self.device
        if self.threshold is not None:
            kwargs["threshold"] = self.threshold
        return kwargs


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


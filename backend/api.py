"""FastAPI application exposing inference endpoints for the exoplanet detector."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, root_validator

from backend.predict import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIDENCE_THRESHOLD,
    process_json_input,
    score_target,
)

app = FastAPI(
    title="CosmiKai Exoplanet Detector API",
    description="REST API for running inference with the CosmiKai exoplanet model.",
    version="0.1.0",
)


def _resolve_checkpoint(checkpoint: Optional[str]) -> Path:
    """Resolve checkpoint string to a Path, defaulting to the packaged model."""
    if checkpoint:
        return Path(checkpoint).expanduser()
    return DEFAULT_CHECKPOINT


class ScoreTargetRequest(BaseModel):
    target: Union[str, int] = Field(..., description="Target name or numeric identifier.")
    mission: str = Field("Kepler", description="Archive mission name passed to lightkurve.")
    author: Optional[str] = Field(None, description="Optional data author filter for lightkurve.")
    nbins: Optional[int] = Field(
        None,
        description="Override number of phase bins (defaults to model training setting).",
    )
    device: Optional[str] = Field(None, description="Torch device string, e.g. 'cpu' or 'cuda:0'.")
    threshold: float = Field(
        DEFAULT_CONFIDENCE_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Decision threshold applied to the confidence score.",
    )
    checkpoint: Optional[str] = Field(
        None,
        description="Optional path to a custom model checkpoint.",
    )

    @root_validator() # type: ignore
    def _validate_target(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        target = values.get("target")
        if isinstance(target, str) and not target.strip():
            raise ValueError("target must not be blank")
        return values


class JsonInferenceRequest(BaseModel):
    payload: Union[str, Dict[str, Any]] = Field(
        ...,
        description="JSON string or object matching the CLI configuration format.",
    )
    checkpoint: Optional[str] = Field(
        None,
        description="Optional path to a custom model checkpoint.",
    )
    device: Optional[str] = Field(None, description="Torch device string, e.g. 'cpu' or 'cuda:0'.")
    threshold: float = Field(
        DEFAULT_CONFIDENCE_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Decision threshold applied to the confidence score.",
    )

    @root_validator() # type: ignore
    def _validate_payload(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        payload = values.get("payload")
        if isinstance(payload, str):
            if not payload.strip():
                raise ValueError("payload string must not be empty")
        elif not isinstance(payload, dict):
            raise ValueError("payload must be a JSON string or object")
        elif not payload:
            raise ValueError("payload dictionary must not be empty")
        return values


@app.get("/health")
async def health() -> Dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "ok"}


@app.post("/score-target")
async def api_score_target(request: ScoreTargetRequest) -> Dict[str, Any]:
    checkpoint_path = _resolve_checkpoint(request.checkpoint)

    try:
        result = await run_in_threadpool(
            score_target,
            request.target,
            checkpoint_path=checkpoint_path,
            mission=request.mission,
            author=request.author,
            nbins=request.nbins,
            device=request.device,
            threshold=request.threshold,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(status_code=424, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - unexpected issues bubble up
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    payload = jsonable_encoder(result)
    payload["checkpoint_path"] = str(checkpoint_path)
    return payload


@app.post("/process-config")
async def api_process_config(request: JsonInferenceRequest) -> Dict[str, Any]:
    checkpoint_path = _resolve_checkpoint(request.checkpoint)

    try:
        result = await run_in_threadpool(
            process_json_input,
            request.payload,
            checkpoint_path=checkpoint_path,
            device=request.device,
            threshold=request.threshold,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(status_code=424, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - unexpected issues bubble up
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    payload = jsonable_encoder(result)
    payload["checkpoint_path"] = str(checkpoint_path)
    return payload

if __name__ == "__main__":  # pragma: no cover - manual execution helper
    import uvicorn

    uvicorn.run("backend.api:app", host="0.0.0.0", port=8000, reload=True)

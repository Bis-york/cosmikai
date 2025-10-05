"""Tests for backend.predict simplified inference helpers."""
from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pytest

import predict


class DummyBundle:
    def __init__(self, device_type: str = "cpu") -> None:
        self.device = types.SimpleNamespace(type=device_type)


def test_score_target_with_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_load_model(checkpoint: str, device: str | None) -> DummyBundle:
        # ensure the checkpoint path reaches the loader
        assert checkpoint.endswith("dummy.ckpt")
        return DummyBundle()

    def fake_download_light_curve(config: predict.TargetConfig) -> tuple[np.ndarray, np.ndarray]:
        assert config.target == "Kepler-10"
        assert config.mission == "Kepler"
        assert config.nbins == 5
        time = np.linspace(0.0, 1.0, 10, dtype=np.float32)
        flux = np.ones_like(time)
        return time, flux

    def fake_estimate_period(time: np.ndarray, flux: np.ndarray) -> tuple[float, float, float]:
        assert time.size == flux.size == 10
        return 1.0, 0.2, 0.5

    def fake_fold_light_curve(
        time: np.ndarray,
        flux: np.ndarray,
        period: float,
        transit_time: float,
        *,
        nbins: int,
    ) -> np.ndarray:
        assert period == pytest.approx(1.0)
        assert transit_time == pytest.approx(0.5)
        return np.linspace(0.0, 1.0, nbins, dtype=np.float32)

    def fake_score_phase_curve(curve: np.ndarray, bundle: DummyBundle) -> tuple[float, float]:
        assert curve.shape[0] == 5
        assert bundle.device.type == "cpu"
        return 0.75, 1.5

    monkeypatch.setattr(predict, "_load_model", fake_load_model)
    monkeypatch.setattr(predict, "_download_light_curve", fake_download_light_curve)
    monkeypatch.setattr(predict, "estimate_period", fake_estimate_period)
    monkeypatch.setattr(predict, "fold_light_curve", fake_fold_light_curve)
    monkeypatch.setattr(predict, "score_phase_curve", fake_score_phase_curve)

    result = predict.score_target(
        "Kepler-10",
        mission="Kepler",
        nbins=5,
        checkpoint_path="dummy.ckpt",
    )

    assert result["confidence"] == pytest.approx(0.75)
    assert result["logit"] == pytest.approx(1.5)
    assert result["has_candidate"] is True
    assert result["light_curve_points"][0][0] == pytest.approx(0.0)
    assert len(result["light_curve_points"]) == 5


def test_process_json_input_uses_score_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pytest.TempPathFactory,
) -> None:
    config = {
        "target_name": "Kepler-22b",
        "mission": "Kepler",
        "nbins": 8,
        "threshold": 0.6,
    }
    configs_dir = tmp_path / "configs" # type: ignore
    configs_dir.mkdir()
    config_path = configs_dir / "target.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    def fake_score_target(
        target: str,
        *,
        mission: str,
        author: str | None,
        nbins: int,
        device: str | None,
        threshold: float,
        checkpoint_path: str | Path,
    ) -> dict[str, object]:
        assert target == "Kepler-22b"
        assert mission == "Kepler"
        assert nbins == 8
        assert threshold == pytest.approx(0.6)
        return {
            "target": target,
            "mission": mission,
            "author": author,
            "nbins": nbins,
            "threshold": threshold,
            "confidence": 0.9,
            "confidence_adjusted": 0.9,
            "logit": 2.0,
            "has_candidate": True,
            "period_days": 1.1,
            "duration_days": 0.2,
            "transit_time": 0.3,
            "device": "cpu",
            "checkpoint_path": str(checkpoint_path),
            "data_points": 0,
            "light_curve_points": [],
        }

    monkeypatch.setattr(predict, "score_target", fake_score_target)

    result = predict.process_json_input(
        config_path,
        checkpoint_path="custom.ckpt",
        device="cpu",
        threshold=0.5,
    )

    assert result["target"] == "Kepler-22b"
    assert result["mission"] == "Kepler"
    assert result["config_used"]["threshold"] == pytest.approx(0.6)
    assert result["config_used"]["nbins"] == 8
    assert result["config_used"]["target"] == "Kepler-22b"



"""M2 model and data endpoints: /api/model*, deletes with confirm + loopback + writer lock, cache, calibrator."""
import json

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from server import model_store, paths
from tests.conftest import BASE_URL


@pytest.fixture
def files(tmp_path, monkeypatch):
    """Model, metrics, dataset, Elo, cache and auto-backtest files in a temp dir."""
    import data.cache as cache
    f = {name: tmp_path / name for name in ("model.joblib", "model_metrics.json", "dataset.csv",
                                             "elo.json", "autobacktest_results.csv")}
    monkeypatch.setattr(paths, "MODEL_FILE", f["model.joblib"])
    monkeypatch.setattr(paths, "MODEL_METRICS_FILE", f["model_metrics.json"])
    monkeypatch.setattr(paths, "DATASET_FILE", f["dataset.csv"])
    monkeypatch.setattr(paths, "ELO_FILE", f["elo.json"])
    monkeypatch.setattr(paths, "AUTOBACKTEST_CSV", f["autobacktest_results.csv"])
    monkeypatch.setattr(cache, "CACHE_DB", str(tmp_path / "cache.db"))
    cache.init()
    model_store.reload()
    yield f
    model_store.reload()


METRICS = {
    "version": "v5", "trained_at": "2026-10-20T10:00:00Z", "rows": 3690, "seasons": ["2024-25"],
    "metrics": {"val_acc": 0.66, "val_loss": 0.62, "n_train": 3000, "n_val": 300, "best_iter": 400,
                "best_params": {"max_depth": 3}},
    "feature_importance": [{"feature": "ELO_DIFF", "importance": 0.3}],
    "calibration": {"bins": [{"lo": 0.5, "hi": 0.6, "n": 40, "mean_pred": 0.55, "actual": 0.52}]},
}


def test_model_without_a_model_file(client, files):
    d = client.get("/api/model").json()
    assert d["exists"] is False and d["metrics"] is None and d["trained_at"] is None
    assert d["feature_importance"] == [] and d["calibration"] == {"bins": []}
    assert len(d["train_stages"]) == 7 and d["train_eta_s"] == 594


def test_model_metrics_survive_a_refresh(client, files):
    joblib.dump({"stub": True}, files["model.joblib"])
    files["model_metrics.json"].write_text(json.dumps(METRICS))
    d = client.get("/api/model").json()
    assert d["exists"] is True and d["trained_at"] == "2026-10-20T10:00:00Z"
    assert d["metrics"]["val_acc"] == 0.66 and d["rows"] == 3690
    assert d["feature_importance"] == METRICS["feature_importance"]
    assert d["calibration"] == METRICS["calibration"]
    assert client.get("/api/status").json()["model"]["val_acc"] == 0.66


def test_reload_loads_without_refit(client, files, monkeypatch):
    assert client.post("/api/model/reload").status_code == 412
    joblib.dump({"stub": 1}, files["model.joblib"])
    files["model_metrics.json"].write_text(json.dumps(METRICS))
    r = client.post("/api/model/reload")
    assert r.status_code == 200
    assert r.json()["trained_at"] == METRICS["trained_at"] and r.json()["loaded_at"].endswith("Z")
    assert model_store.get_model() == {"stub": 1}


def test_delete_model_needs_confirm_and_loopback(app, client, files):
    joblib.dump({"stub": 1}, files["model.joblib"])
    files["model_metrics.json"].write_text(json.dumps(METRICS))
    r = client.delete("/api/model")
    assert r.status_code == 400 and r.json()["error"] == "confirm_required"
    remote = TestClient(app, base_url=BASE_URL, client=("10.0.0.7", 1))
    r = remote.delete("/api/model?confirm=true")
    assert r.status_code == 403 and r.json()["error"] == "loopback_only"
    assert files["model.joblib"].exists()
    r = client.delete("/api/model?confirm=true")
    assert r.json() == {"ok": True, "removed": True, "file": "model.joblib"}
    assert not files["model.joblib"].exists() and not files["model_metrics.json"].exists()
    assert model_store.get_model() is None
    assert client.get("/api/status").json()["capabilities"]["predict_mode"] == "elo_only"
    assert client.delete("/api/model?confirm=true").json()["removed"] is False


@pytest.mark.parametrize("path, key", [("/api/dataset", "dataset.csv"), ("/api/elo", "elo.json")])
def test_delete_dataset_and_elo(client, files, path, key):
    files[key].write_text("x")
    assert client.delete(path).status_code == 400
    assert files[key].exists()
    assert client.delete(path, params={"confirm": "true"}).json() == {"ok": True, "removed": True, "file": key}
    assert not files[key].exists()


def test_cache_stats_and_clear(app, client, files):
    from data import cache
    cache.set("k1", {"a": 1})
    cache.set("k2", [1, 2], ttl=-1)
    assert client.get("/api/cache").json() == {"total": 2, "live": 1, "expired": 1}
    remote = TestClient(app, base_url=BASE_URL, client=("10.0.0.7", 1))
    assert remote.delete("/api/cache").status_code == 403
    assert client.delete("/api/cache").json() == {"ok": True, "total": 0, "live": 0, "expired": 0}


def test_calibrator_and_player_impact(client):
    c = client.get("/api/calibrator").json()
    assert set(c["weights"]) == {"xgb", "mc", "elo"} and "bias" in c and "n_updates" in c
    p = client.get("/api/player-impact", params={"top": 5}).json()
    assert {"season", "updated", "count", "fallback", "top"} <= set(p)
    assert len(p["top"]) <= 5
    impacts = [row["impact"] for row in p["top"]]
    assert impacts == sorted(impacts, reverse=True)


def test_ensemble_elo_only_without_model(client, files):
    d = client.get("/api/model/ensemble").json()
    assert d["method"] == "elo_only"
    assert d["components"] == [{"key": "elo", "label": "Elo", "coef": 1.0,
                                "mean_abs_contribution": None, "oos_log_loss": None}]
    assert set(d["calibrator"]["weights"]) == {"xgb", "mc", "elo"}


def test_ensemble_blend_with_contributions(client, files):
    from engine.models import get_calibrator
    joblib.dump({"stub": 1}, files["model.joblib"])
    pd.DataFrame({"XGB_P": [0.7, 0.4], "MC_P": [0.6, 0.5], "ELO_DIFF": [80.0, -40.0]}).to_csv(
        files["autobacktest_results.csv"], index=False)
    d = client.get("/api/model/ensemble").json()
    assert d["method"] == "blend_v5"
    assert [c["key"] for c in d["components"]] == ["elo", "mc", "xgb"]
    w = get_calibrator().weights
    by_key = {c["key"]: c for c in d["components"]}
    assert by_key["xgb"]["coef"] == pytest.approx(w[0], abs=1e-4)
    assert by_key["xgb"]["mean_abs_contribution"] == pytest.approx(abs(w[0]) * (0.2 + 0.1) / 2, abs=1e-4)
    assert by_key["elo"]["mean_abs_contribution"] == pytest.approx(abs(w[2]) * (0.2 + 0.1) / 2, abs=1e-4)
    assert by_key["mc"]["oos_log_loss"] is None

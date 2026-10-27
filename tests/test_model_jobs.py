"""M2 model jobs (train / autobacktest / player_impact) with fake workers, the train worker's
JSON lines, and the engine changes E1 (atomic writes), E6 (N_JOBS, load_model), E8 (cwd=ROOT).

Nothing here trains a real model or touches the network."""
import inspect
import json
import os
import stat
import sys
import time
from datetime import date

import joblib
import numpy as np
import pandas as pd
import pytest

import config
from server import jobs, model_store, paths
from server.services import model_jobs
from server.workers import train as worker
from tests.test_jobs import wait_for

FAKE_WORKER = '''
import json, sys, time
def out(o): print(json.dumps(o), flush=True)
mode = sys.argv[1]
for i in range(1, 8):
    out({"type": "stage", "i": i, "n": 7, "msg": f"Stage {i}"})
    if mode == "slow":
        time.sleep(0.3)
print("a plain line from a library", flush=True)
if mode == "fail":
    out({"type": "error", "msg": "Training failed: no game data"})
    sys.exit(1)
out({"type": "result", "data": {"rows": 10, "metrics": {"val_acc": 0.61}}})
'''


@pytest.fixture
def fake_worker(tmp_path, monkeypatch):
    script = tmp_path / "fake_worker.py"
    script.write_text(FAKE_WORKER)
    mode = {"value": "ok"}
    monkeypatch.setattr(model_jobs, "train_command",
                        lambda params: [sys.executable, str(script), mode["value"]])
    reloads = []
    monkeypatch.setattr(model_store, "reload", lambda: reloads.append(1))
    return mode, reloads


# ── train job ─────────────────────────────────────────────────────────────────

def test_train_job_runs_the_worker_and_reloads_the_model(client, fake_worker):
    mode, reloads = fake_worker
    r = client.post("/api/jobs/train", json={"force_rebuild": True})
    assert r.status_code == 202
    rec = wait_for(client, r.json()["job_id"])
    assert rec["state"] == "done", rec
    assert rec["params"] == {"force_rebuild": True, "use_optuna": False}
    assert rec["result"] == {"rows": 10, "metrics": {"val_acc": 0.61}}
    assert rec["total"] == 7 and rec["progress"] == 1.0
    assert "a plain line from a library" in rec["log"]
    assert reloads == [1]


def test_train_job_failure_reports_the_worker_error(client, fake_worker):
    mode, reloads = fake_worker
    mode["value"] = "fail"
    rec = wait_for(client, client.post("/api/jobs/train", json={}).json()["job_id"])
    assert rec["state"] == "error" and rec["error"] == "Training failed: no game data"
    assert reloads == []


def test_train_job_progress_is_staged_and_cancel_works(client, fake_worker):
    mode, reloads = fake_worker
    mode["value"] = "slow"
    job_id = client.post("/api/jobs/train", json={}).json()["job_id"]
    rec = wait_for(client, job_id, states=("running",))
    deadline = time.monotonic() + 5
    while rec["done"] in (None, 0) and time.monotonic() < deadline:
        time.sleep(0.05)
        rec = client.get(f"/api/jobs/{job_id}").json()
    assert rec["total"] == 7 and rec["progress"] is None and rec["stage"].startswith("Stage")
    assert client.post("/api/jobs/train", json={}).status_code == 409
    client.delete(f"/api/jobs/{job_id}")
    assert wait_for(client, job_id)["state"] == "cancelled"
    assert reloads == []


def test_train_params(client, fake_worker, monkeypatch):
    assert client.post("/api/jobs/train", json={"force_rebuild": "nope"}).status_code == 422
    monkeypatch.setattr(model_jobs, "has_module", lambda name: False)
    r = client.post("/api/jobs/train", json={"use_optuna": True})
    assert r.status_code == 422 and "Optuna is not installed" in r.json()["detail"]


def test_train_command():
    assert model_jobs.train_command({"force_rebuild": True, "use_optuna": True})[1:] == [
        "-m", "server.workers.train", "--force-rebuild", "--optuna"]


# ── train worker (in-process, fake train_model) ───────────────────────────────

STATUS = ["Downloading 3 seasons of game data…", "Paired 3,690 matchup rows from 7,380 team rows",
          "Elo ratings replayed chronologically and saved (live ratings updated)",
          "Dataset built: 3,690 matchup rows saved to master_dataset_v5.csv",
          "Running RandomizedSearchCV (20 candidates × 3 time-series folds)…",
          "Best params — depth=3 lr=0.040 n_est=400 cv_loss=0.6300",
          "Training XGBoost with recency-weighted samples…", "Calibrating probabilities (Platt sigmoid)…",
          "Model saved — val_acc=0.660 val_loss=0.6200 n_train=3,000"]


def fake_train(force_rebuild, use_optuna, status_cb):
    for msg in STATUS:
        status_cb(msg)
    rng = np.random.default_rng(0)
    proba = rng.uniform(0.2, 0.8, 200)
    return object(), 3690, {
        "val_acc": 0.66, "val_loss": 0.62, "n_train": 3000, "n_val": 200, "best_iter": 400,
        "best_params": {"max_depth": 3}, "feat_imp": {"ELO_DIFF": 0.3, "NET5_DIFF": 0.1, "H_B2B": 0.02},
        "val_proba": proba.tolist(), "val_y": (rng.uniform(size=200) < proba).astype(int).tolist()}


def test_worker_emits_stages_and_writes_metrics(isolated, monkeypatch):
    monkeypatch.setattr(paths, "MODEL_METRICS_FILE", isolated / "model_metrics.json")
    lines = []
    assert worker.run(force_rebuild=True, train=fake_train, out=lines.append) == 0
    stages = [line["i"] for line in lines if line["type"] == "stage"]
    assert stages == [1, 2, 3, 4, 5, 6, 7]
    assert all(line["n"] == 7 for line in lines if line["type"] == "stage")
    assert lines[-1]["type"] == "result" and lines[-1]["data"]["rows"] == 3690
    json.dumps(lines)                                                    # all JSON-able
    saved = json.loads((isolated / "model_metrics.json").read_text())
    assert saved["metrics"]["val_acc"] == 0.66 and "val_proba" not in saved["metrics"]
    assert saved["trained_at"].endswith("Z") and saved["version"] == "v5"
    assert [f["feature"] for f in saved["feature_importance"]] == ["ELO_DIFF", "NET5_DIFF", "H_B2B"]
    bins = saved["calibration"]["bins"]
    assert sum(b["n"] for b in bins) == 200 and all(b["lo"] <= b["mean_pred"] <= b["hi"] for b in bins)


def test_worker_reports_failures(isolated, monkeypatch):
    monkeypatch.setattr(paths, "MODEL_METRICS_FILE", isolated / "model_metrics.json")
    lines = []
    assert worker.run(train=lambda **kw: (None, 0, None), out=lines.append) == 1
    assert lines[-1]["type"] == "error" and "no game data" in lines[-1]["msg"]

    def boom(**kw):
        raise ValueError("bad dataset")
    lines.clear()
    assert worker.run(train=boom, out=lines.append) == 1
    assert lines[-1] == {"type": "error", "msg": "Training failed: ValueError: bad dataset"}
    assert not (isolated / "model_metrics.json").exists()


@pytest.mark.skipif(not os.path.isdir(os.environ.get("NBA_HISTORY_DIR", "/home/user/nocturnebear/nba-data-2010-2024")),
                    reason="history files not available")
def test_worker_demo_mode_reads_history(monkeypatch):
    import engine.models as models
    monkeypatch.setattr(models, "fetch_game_logs", models.fetch_game_logs)
    monkeypatch.setattr(models, "TRAIN_SEASONS", models.TRAIN_SEASONS)
    seasons = worker.use_history_logs(date(2024, 3, 15))
    assert seasons == ["2021-22", "2022-23", "2023-24"]
    df = models.fetch_game_logs(None, report=lambda m: None)
    assert set(df["SEASON_YEAR"]) == set(seasons)
    assert df["GAME_DATE"].max() < pd.Timestamp("2024-03-15")


# ── autobacktest job ──────────────────────────────────────────────────────────

def test_autobacktest_job_maps_progress_and_result(client, tmp_path, monkeypatch):
    import data.autobacktest as ab
    monkeypatch.setattr(paths, "MODEL_FILE", tmp_path / "model.joblib")
    (tmp_path / "model.joblib").write_bytes(b"x")
    monkeypatch.setattr(model_store, "get_model", lambda: "MODEL")
    seen = {}

    def fake_run(model, status_cb=None, should_stop=None, progress_cb=None):
        seen["model"] = model
        status_cb("Found 4 new completed games not yet backtested.")
        for i in range(4):
            progress_cb(i, 4)
        seen["stop"] = should_stop()
        return {"new": 4, "correct": 3, "acc": 0.75, "stopped": False,
                "calibrator": {"weights": {"XGBoost": 2.0, "Monte Carlo": 1.5, "ELO diff": 0.3},
                               "bias": -1.7, "n_updates": 31}}

    monkeypatch.setattr(ab, "run_auto_backtest", fake_run)
    rec = wait_for(client, client.post("/api/jobs/autobacktest", json={}).json()["job_id"])
    assert rec["state"] == "done" and seen == {"model": "MODEL", "stop": False}
    assert rec["result"] == {"new": 4, "correct": 3, "acc": 0.75, "stopped": False,
                             "calibrator": {"weights": {"xgb": 2.0, "mc": 1.5, "elo": 0.3},
                                            "bias": -1.7, "n_updates": 31}}
    assert rec["done"] == 4 and rec["total"] == 4
    assert client.post("/api/jobs/autobacktest", json={"x": 1}).status_code == 422


def test_run_auto_backtest_stops_cleanly_and_saves_progress(tmp_path, monkeypatch):
    """Cancel between games: the graded games are logged, so a re-run never re-grades them."""
    import data.autobacktest as ab
    import data.elo as elo
    import engine.models as models
    monkeypatch.setattr(ab, "AUTOBACKTEST_LOG", str(tmp_path / "log.json"))
    monkeypatch.setattr(ab, "AUTOBACKTEST_CSV", str(tmp_path / "results.csv"))
    monkeypatch.setattr(elo, "ELO_FILE", str(tmp_path / "elo.json"))
    monkeypatch.setattr(models, "BLEND_FILE", str(tmp_path / "blend.json"))
    games = pd.DataFrame({"GAME_ID": ["0022300001", "0022300002", "0022300003"],
                          "GAME_DATE": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                          "HOME_TEAM_ID": [1610612747] * 3, "AWAY_TEAM_ID": [1610612744] * 3,
                          "HOME_WON": [1, 0, 1], "MARGIN": [5, -3, 8]})
    monkeypatch.setattr(ab, "_fetch_new_games", lambda status_cb=None: games)
    monkeypatch.setattr(ab, "get_team_advanced_stats", lambda name, target_date=None: {"ELO": 1500})
    monkeypatch.setattr(ab, "predict_hybrid",
                        lambda hd, ad, model: (0.6, -3, 220, 0.62, 0.58, False, {}, 1.0, 1.0))
    progress = []
    out = ab.run_auto_backtest("MODEL", should_stop=lambda: len(progress) >= 2,
                               progress_cb=lambda d, t: progress.append((d, t)))
    assert out["stopped"] is True and out["new"] == 2
    assert progress == [(0, 3), (1, 3)]
    log = json.loads((tmp_path / "log.json").read_text())
    assert sorted(log["processed_game_ids"]) == ["0022300001", "0022300002"] and log["total"] == 2
    assert len(pd.read_csv(tmp_path / "results.csv")) == 2
    assert json.loads((tmp_path / "blend.json").read_text())["n_updates"] == 2
    assert not [p for p in os.listdir(tmp_path) if p.endswith(".tmp")]


def test_demo_autobacktest_source_is_scoped(monkeypatch):
    import data.autobacktest as ab
    before = ab.leaguegamefinder, ab.SEASONS
    monkeypatch.setenv("NBA_QUANT_DEMO_DATE", "2024-03-15")
    with model_jobs.demo_autobacktest_source():
        assert ab.SEASONS == ["2022-23", "2023-24"]
        assert isinstance(ab.leaguegamefinder, model_jobs._HistoryFinder)
    assert (ab.leaguegamefinder, ab.SEASONS) == before


# ── player impact job (E8) ────────────────────────────────────────────────────

def _updater(tmp_path, monkeypatch, body: str):
    script = tmp_path / "fake_updater.py"
    script.write_text(body)
    monkeypatch.setattr(paths, "PLAYER_IMPACT_UPDATER", script)
    return script


def test_player_impact_job_runs_updater_from_project_root(client, tmp_path, monkeypatch):
    _updater(tmp_path, monkeypatch, "import os, sys\nprint('cwd', os.getcwd())\nprint('args', sys.argv[1:])\n")
    monkeypatch.chdir(tmp_path)                                         # the server's cwd must not matter
    rec = wait_for(client, client.post("/api/jobs/player_impact", json={"season": "2025-26"}).json()["job_id"])
    assert rec["state"] == "done", rec
    assert f"cwd {paths.ROOT}" in rec["log"] and "args ['2025-26']" in rec["log"]
    assert {"season", "updated", "count"} == set(rec["result"])


def test_player_impact_failures(client, tmp_path, monkeypatch):
    assert client.post("/api/jobs/player_impact", json={"season": "2025"}).status_code == 422
    _updater(tmp_path, monkeypatch, "import sys\nprint('ConnectionError: stats.nba.com timed out')\nsys.exit(1)\n")
    rec = wait_for(client, client.post("/api/jobs/player_impact", json={}).json()["job_id"])
    assert rec["state"] == "error" and rec["error"].endswith("ConnectionError: stats.nba.com timed out")

    _updater(tmp_path, monkeypatch, "import time\ntime.sleep(30)\n")
    monkeypatch.setattr(model_jobs, "PLAYER_IMPACT_TIMEOUT_S", 1)
    rec = wait_for(client, client.post("/api/jobs/player_impact", json={}).json()["job_id"], timeout=10)
    assert rec["state"] == "error" and rec["error"] == "Timed out after 1 s"

    monkeypatch.setenv("NBA_QUANT_DEMO_DATE", "2024-03-15")
    rec = wait_for(client, client.post("/api/jobs/player_impact", json={}).json()["job_id"])
    assert rec["state"] == "error" and "not available in demo mode" in rec["error"]


# ── E1 atomic writes ──────────────────────────────────────────────────────────

def test_atomic_write_replaces_whole_file_and_keeps_mode(tmp_path):
    from data.atomic import atomic_write_json
    path = tmp_path / "state.json"
    path.write_text('{"old": true}')
    os.chmod(path, 0o644)
    atomic_write_json(path, {"new": [1, 2]}, indent=2)
    assert json.loads(path.read_text()) == {"new": [1, 2]}
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o644
    assert os.listdir(tmp_path) == ["state.json"]


def test_atomic_path_failure_leaves_the_old_file(tmp_path):
    from data.atomic import atomic_path
    path = tmp_path / "model.joblib"
    path.write_bytes(b"old model")
    with pytest.raises(RuntimeError):
        with atomic_path(path) as tmp:
            with open(tmp, "wb") as f:
                f.write(b"half a mod")
            raise RuntimeError("cancelled mid-write")
    assert path.read_bytes() == b"old model"
    assert os.listdir(tmp_path) == ["model.joblib"]


def test_state_writers_use_atomic_writes(tmp_path, monkeypatch):
    import data.elo as elo
    from engine.models import BlendCalibrator
    monkeypatch.setattr(elo, "ELO_FILE", str(tmp_path / "elo.json"))
    elo.save_elo({"Boston Celtics": 1650.5})
    assert elo.load_elo() == {"Boston Celtics": 1650.5}
    bc = BlendCalibrator()
    bc.update(0.7, 0.6, 50, 1)
    bc.save(str(tmp_path / "blend.json"))
    again = BlendCalibrator.load(str(tmp_path / "blend.json"))
    assert again.n_updates == 1 and np.allclose(again.weights, bc.weights)
    assert sorted(os.listdir(tmp_path)) == ["blend.json", "elo.json"]
    for fn in (elo.save_elo, BlendCalibrator.save):
        assert "atomic_write_json" in inspect.getsource(fn)


def test_train_model_writes_model_and_dataset_atomically():
    import engine.models as models
    src = inspect.getsource(models.train_model) + inspect.getsource(models.build_master_dataset)
    assert "atomic_joblib_dump(model, MODEL_FILE)" in src and "atomic_path(DATASET_FILE)" in src


# ── E6 N_JOBS and load_model ──────────────────────────────────────────────────

def test_n_jobs_leaves_a_core_free():
    import engine.models as models
    assert config.N_JOBS == max(1, (os.cpu_count() or 2) - 1)
    src = inspect.getsource(models)
    assert "n_jobs=-1" not in src and src.count("n_jobs=N_JOBS") == 2


def test_load_model_has_no_refit(tmp_path):
    from engine.models import load_model
    path = tmp_path / "m.joblib"
    joblib.dump({"weights": [1, 2, 3]}, path)
    assert load_model(str(path)) == {"weights": [1, 2, 3]}
    assert load_model(str(tmp_path / "missing.joblib")) is None


def test_jobs_registered():
    assert {"train", "autobacktest", "player_impact"} <= set(jobs.KINDS)
    assert all(jobs.KINDS[k].writer for k in ("train", "autobacktest", "player_impact"))


def test_worker_demo_mode_always_rebuilds_from_history(isolated, monkeypatch):
    monkeypatch.setattr(paths, "MODEL_METRICS_FILE", isolated / "model_metrics.json")
    monkeypatch.setenv("NBA_QUANT_DEMO_DATE", "2024-03-15")
    monkeypatch.setattr(worker, "use_history_logs", lambda demo: ["2021-22", "2022-23", "2023-24"])
    seen, lines = {}, []

    def train(force_rebuild, use_optuna, status_cb):
        seen["force_rebuild"] = force_rebuild
        return fake_train(force_rebuild, use_optuna, status_cb)

    assert worker.run(force_rebuild=False, train=train, out=lines.append) == 0
    assert seen == {"force_rebuild": True}
    assert json.loads((isolated / "model_metrics.json").read_text())["seasons"] == ["2021-22", "2022-23", "2023-24"]

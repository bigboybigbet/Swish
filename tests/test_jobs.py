"""M2 job manager: fake fast job kinds through the API, SSE order and resume, the writer lock, cancel."""
import json
import os
import sys
import threading
import time

import pytest
from pydantic import BaseModel, ConfigDict, Field

from server import jobs, paths

SECRET = "sk-test-odds-9f8e7d6c5b4a"
RECORD_KEYS = {"id", "kind", "state", "progress", "stage", "done", "total", "log", "params",
               "created_at", "started_at", "finished_at", "result_ref", "error"}


class FakeParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n: int = Field(3, ge=1, le=50)


@pytest.fixture
def fake_kinds(monkeypatch):
    """fake (reader), fake_writer (blocks until gate.set()), fake_fail, fake_proc (subprocess)."""
    gate = threading.Event()

    def fake(ctx):
        n = ctx.params["n"]
        for i in range(n):
            ctx.check_cancel()
            ctx.progress(done=i, total=n, stage=f"Step {i + 1}")
            ctx.item(i, {"value": i * 1.5, "missing": float("nan")})
        ctx.log("all steps done")
        return {"n": n}

    def blocking(ctx):
        ctx.progress(stage="Waiting for the gate", progress=None)
        while not gate.wait(0.02):
            ctx.check_cancel()
        return {"released": True}

    def failing(ctx):
        ctx.log(f"calling https://api.example.com/odds?apiKey={SECRET}")
        raise RuntimeError(f"upstream said no for key {SECRET}")

    def proc(ctx):
        script = ("import os, sys, time\nprint('pid', os.getpid(), flush=True)\n"
                  "for i in range(600):\n    print('tick', i, flush=True)\n    time.sleep(0.05)\n")
        rc = ctx.run_subprocess([sys.executable, "-c", script])
        return {"rc": rc}

    for kind in (jobs.JobKind("fake", fake, params=FakeParams, label="Fake"),
                 jobs.JobKind("fake_writer", blocking, writer=True, label="Fake writer"),
                 jobs.JobKind("fake_fail", failing, label="Fake failure"),
                 jobs.JobKind("fake_proc", proc, writer=True, label="Fake process")):
        monkeypatch.setitem(jobs.KINDS, kind.name, kind)
    yield gate
    gate.set()


def wait_for(client, job_id, states=jobs.TERMINAL, timeout=10.0) -> dict:
    deadline = time.monotonic() + timeout
    rec = None
    while time.monotonic() < deadline:
        rec = client.get(f"/api/jobs/{job_id}").json()
        if rec["state"] in states:
            return rec
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never reached {states}: {rec}")


def read_sse(client, job_id, headers=None, params=None) -> list:
    """[(id, type, data)] from the SSE stream (which ends after `done`)."""
    with client.stream("GET", f"/api/jobs/{job_id}/events", headers=headers or {}, params=params) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        assert r.headers["cache-control"] == "no-cache"
        assert r.headers["x-accel-buffering"] == "no"
        text = "".join(r.iter_text())
    events = []
    for block in text.split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines()
                      if line and not line.startswith(":"))
        if "event" in fields:
            events.append((int(fields["id"]), fields["event"], json.loads(fields["data"])))
    return events


# ── lifecycle ─────────────────────────────────────────────────────────────────

def test_fake_job_runs_queued_to_done(client, fake_kinds):
    r = client.post("/api/jobs/fake", json={"n": 4})
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    rec = wait_for(client, job_id)
    assert RECORD_KEYS <= set(rec)
    assert rec["state"] == "done" and rec["kind"] == "fake" and rec["params"] == {"n": 4}
    assert rec["progress"] == 1.0 and rec["done"] == rec["total"] == 4
    assert rec["started_at"] and rec["finished_at"] and rec["error"] is None
    assert rec["log"] == ["all steps done"]
    assert rec["result"] == {"n": 4} and rec["result_url"] == f"/api/jobs/{job_id}"
    assert (paths.RESULTS_DIR / f"{job_id}.json").exists()


def test_sse_event_order_and_last_event_id_resume(client, fake_kinds):
    job_id = client.post("/api/jobs/fake", json={"n": 3}).json()["job_id"]
    wait_for(client, job_id)
    events = read_sse(client, job_id)
    ids = [e[0] for e in events]
    types = [e[1] for e in events]
    assert ids == list(range(1, len(ids) + 1))
    assert types[0] == "progress" and events[0][2]["stage"] == "Queued"
    assert types[-1] == "done" and types.count("done") == 1
    assert events[-1][2] == {"state": "done", "result_url": f"/api/jobs/{job_id}", "error": None}
    items = [e[2] for e in events if e[1] == "item"]
    assert [i["index"] for i in items] == [0, 1, 2]
    assert items[1]["data"] == {"value": 1.5, "missing": None}          # NaN → null
    assert types.index("log") > max(i for i, t in enumerate(types) if t == "item")
    progress = [e[2] for e in events if e[1] == "progress"]
    assert {"progress", "stage", "done", "total"} == set(progress[-1]) and progress[-1]["progress"] == 1.0

    resumed = read_sse(client, job_id, headers={"Last-Event-ID": str(ids[3])})
    assert resumed == events[4:]
    assert read_sse(client, job_id, params={"after": ids[-2]}) == events[-1:]


def test_sse_streams_a_live_job_until_done(client, fake_kinds):
    job_id = client.post("/api/jobs/fake_writer").json()["job_id"]
    threading.Timer(0.5, fake_kinds.set).start()
    events = read_sse(client, job_id)
    assert [t for _, t, _ in events][-1] == "done"
    assert any(d.get("stage") == "Waiting for the gate" for _, t, d in events if t == "progress")
    assert events[-1][2]["state"] == "done"


def test_sse_unknown_job_is_404(client):
    r = client.get("/api/jobs/nope/events")
    assert r.status_code == 404 and r.json()["error"] == "not_found"


# ── writer lock ───────────────────────────────────────────────────────────────

def test_second_writer_gets_409_and_destructive_endpoints_wait(client, fake_kinds, tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "MODEL_FILE", tmp_path / "model.joblib")
    writer = client.post("/api/jobs/fake_writer").json()["job_id"]
    wait_for(client, writer, states=("running",))

    r = client.post("/api/jobs/fake_writer")
    assert r.status_code == 409
    assert r.json()["error"] == "busy" and r.json()["job_id"] == writer
    r = client.post("/api/jobs/fake_proc")                       # any writer kind
    assert r.status_code == 409 and r.json()["job_id"] == writer
    for path in ("/api/model?confirm=true", "/api/dataset?confirm=true", "/api/elo?confirm=true", "/api/cache"):
        r = client.delete(path)
        assert r.status_code == 409 and r.json()["error"] == "busy", path

    reader = client.post("/api/jobs/fake", json={"n": 2}).json()["job_id"]   # readers still run
    assert wait_for(client, reader)["state"] == "done"

    status = client.get("/api/status").json()
    active = [j for j in status["active_jobs"] if j["id"] == writer]
    assert active and active[0]["kind"] == "fake_writer" and active[0]["writer"] is True

    assert client.delete(f"/api/jobs/{writer}").json() == {"state": "cancelling"}
    assert wait_for(client, writer)["state"] == "cancelled"
    assert client.delete(f"/api/jobs/{writer}").json() == {"state": "cancelled"}
    assert client.post("/api/jobs/fake_writer").status_code == 202


def test_writer_lock_blocks_a_writer_job_while_held(isolated, fake_kinds):
    m = jobs.JobManager(isolated / "jobs.json", isolated / "results")
    with m.writer_lock("Deleting the model"):
        with pytest.raises(jobs.ApiError) as e:
            m.submit("fake_writer")
        assert e.value.code == "busy"
        m.submit("fake", {"n": 1})                                    # readers don't need it
    job = m.submit("fake_writer")
    fake_kinds.set()
    _wait(job)
    assert job.state == "done"
    m.shutdown()


# ── cancel ────────────────────────────────────────────────────────────────────

def test_cancel_terminates_a_subprocess_job(client, fake_kinds):
    job_id = client.post("/api/jobs/fake_proc").json()["job_id"]
    deadline = time.monotonic() + 10
    rec = {}
    while time.monotonic() < deadline:
        rec = client.get(f"/api/jobs/{job_id}").json()
        if any(line.startswith("tick") for line in rec["log"]):
            break
        time.sleep(0.02)
    pid = int(next(line for line in rec["log"] if line.startswith("pid")).split()[1])
    t0 = time.monotonic()
    assert client.delete(f"/api/jobs/{job_id}").json() == {"state": "cancelling"}
    rec = wait_for(client, job_id, timeout=8)
    assert rec["state"] == "cancelled" and rec["result"] is None
    assert time.monotonic() - t0 < 6
    with pytest.raises(OSError):
        os.kill(pid, 0)                                               # the process is gone


def test_cancel_a_queued_reader_finishes_it_at_once(isolated, fake_kinds, monkeypatch):
    started = threading.Event()

    def slow(ctx):
        started.set()
        while not fake_kinds.wait(0.02):
            ctx.check_cancel()

    monkeypatch.setitem(jobs.KINDS, "slow_reader", jobs.JobKind("slow_reader", slow))
    m = jobs.JobManager(isolated / "jobs.json", isolated / "results", reader_workers=1)
    first = m.submit("slow_reader")
    started.wait(5)
    queued = m.submit("fake", {"n": 1})
    assert queued.state == "queued"
    assert m.cancel(queued.id) == "cancelled" and queued.state == "cancelled"
    fake_kinds.set()
    _wait(first)
    time.sleep(0.1)
    assert queued.state == "cancelled" and queued.started_at is None
    m.shutdown()


# ── listing, validation, errors ───────────────────────────────────────────────

def test_list_filters(client, fake_kinds):
    a = client.post("/api/jobs/fake", json={"n": 1}).json()["job_id"]
    wait_for(client, a)
    w = client.post("/api/jobs/fake_writer").json()["job_id"]
    wait_for(client, w, states=("running",))
    assert [j["id"] for j in client.get("/api/jobs", params={"active": True}).json()] == [w]
    assert [j["id"] for j in client.get("/api/jobs", params={"kind": "fake"}).json()] == [a]
    assert [j["id"] for j in client.get("/api/jobs", params={"limit": 1}).json()] == [w]
    assert client.get("/api/jobs", params={"limit": 0}).status_code == 422


def test_unknown_kind_bad_params_and_no_model(client, fake_kinds, tmp_path, monkeypatch):
    r = client.post("/api/jobs/nope")
    assert r.status_code == 404 and r.json()["error"] == "not_found"
    r = client.post("/api/jobs/fake", json={"n": 0})
    assert r.status_code == 422 and r.json()["error"] == "validation"
    r = client.post("/api/jobs/fake", json={"n": 2, "extra": True})
    assert r.status_code == 422
    monkeypatch.setattr(paths, "MODEL_FILE", tmp_path / "missing.joblib")
    r = client.post("/api/jobs/autobacktest", json={})
    assert r.status_code == 412 and r.json()["error"] == "no_model"
    assert client.get("/api/jobs/doesnotexist").status_code == 404
    assert client.delete("/api/jobs/doesnotexist").status_code == 404


def test_job_errors_and_logs_are_redacted(client, fake_kinds, monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", SECRET)
    job_id = client.post("/api/jobs/fake_fail").json()["job_id"]
    rec = wait_for(client, job_id)
    assert rec["state"] == "error"
    assert "upstream said no" in rec["error"] and "•••" in rec["error"]
    raw = json.dumps(rec) + json.dumps(read_sse(client, job_id))
    time.sleep(0.2)                                   # jobs.json is written after finishing
    raw += paths.JOBS_FILE.read_text(encoding="utf-8")
    assert SECRET not in raw
    assert any("apiKey=•••" in line for line in rec["log"])


# ── persistence and restart ───────────────────────────────────────────────────

def _wait(job, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not job.terminal and time.monotonic() < deadline:
        time.sleep(0.02)
    assert job.terminal, job


def test_jobs_json_and_restart_marks_running_interrupted(isolated, fake_kinds):
    jobs_file = isolated / "runs" / "jobs.json"
    m = jobs.JobManager(jobs_file, isolated / "results")
    done = m.submit("fake", {"n": 1})
    _wait(done)
    running = m.submit("fake_writer")
    time.sleep(0.2)
    saved = json.loads(jobs_file.read_text())["jobs"]
    assert {r["id"]: r["state"] for r in saved} == {done.id: "done", running.id: "running"}

    m2 = jobs.JobManager(jobs_file, isolated / "results").load()      # as after a crash
    again = m2.get(running.id)
    assert again.state == "interrupted" and again.finished_at and "stopped" in again.error
    assert m2.get(done.id).state == "done"
    states = {r["id"]: r["state"] for r in json.loads(jobs_file.read_text())["jobs"]}
    assert states[running.id] == "interrupted"
    # a restarted server still ends an SSE stream for it (even with a stale Last-Event-ID)
    events = m2.events_after(again, 0)
    assert [e.type for e in events] == ["progress", "done"] and events[-1].data["state"] == "interrupted"

    fake_kinds.set()
    _wait(running)
    m.shutdown()


def test_stale_last_event_id_replays_from_start(client, fake_kinds):
    job_id = client.post("/api/jobs/fake", json={"n": 1}).json()["job_id"]
    wait_for(client, job_id)
    events = read_sse(client, job_id, headers={"Last-Event-ID": "9999"})
    assert events[0][0] == 1 and events[-1][1] == "done"


def test_shutdown_interrupts_running_jobs(isolated, fake_kinds):
    m = jobs.JobManager(isolated / "jobs.json", isolated / "results")
    job = m.submit("fake_writer")
    time.sleep(0.1)
    m.shutdown(timeout=3)
    assert job.state == "interrupted"
    with pytest.raises(jobs.ApiError):
        m.submit("fake", {"n": 1})


def test_records_and_events_are_bounded(isolated, fake_kinds, monkeypatch):
    monkeypatch.setattr(jobs, "MAX_JOBS", 3)
    monkeypatch.setattr(jobs, "MAX_EVENTS", 10)
    m = jobs.JobManager(isolated / "jobs.json", isolated / "results")
    last = None
    for _ in range(5):
        last = m.submit("fake", {"n": 20})
        _wait(last)
    time.sleep(0.2)
    assert len(m.jobs) <= 4                              # trimmed when the next job is added
    assert len(json.loads((isolated / "jobs.json").read_text())["jobs"]) == 3
    assert len(last.events) == 10
    kinds = [e.type for e in last.events]
    assert kinds.count("item") == 8 or "item" in kinds   # items survive; old progress events go first
    assert kinds[-1] == "done"
    m.shutdown()

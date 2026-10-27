"""
Background jobs: JobManager, the writer lock, persistence and the SSE stream.

A job kind is registered once with register_kind() and started with
manager().submit(kind, params). Every job runs on a thread: reader kinds
(scan, backtest) share a 2-thread pool; writer kinds (train, autobacktest,
player_impact, ratings) run one at a time, each on its own thread. CPU-heavy
kinds start a subprocess from that thread with ctx.run_subprocess().

A runner gets a JobContext and returns a JSON-able result (saved to
runs/results/<id>.json) or None:

    def run(ctx):
        for i, game in enumerate(games):
            ctx.check_cancel()                        # raises JobCancelled
            ctx.progress(done=i, total=len(games), stage=game["label"])
            ctx.item(i, card)                         # SSE `item` event
        return {"games": cards}

    register_kind(JobKind("scan", run, params=ScanParams, label="Slate scan"))

Events (progress, log, item, done) carry incrementing ids, so an SSE client
resumes with Last-Event-ID. Job records are mirrored to runs/jobs.json (last
100); a job still running when the server stopped is marked "interrupted"
at the next start.
"""
import asyncio
import json
import logging
import os
import signal
import subprocess
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from data.atomic import atomic_write_json
from server import clock, jsonutil, paths
from server.errors import ApiError
from server.security import redact

log = logging.getLogger("server.jobs")

ACTIVE = ("queued", "running")
TERMINAL = ("done", "error", "cancelled", "interrupted")
MAX_JOBS = 100                     # records kept in memory and in runs/jobs.json
MAX_LOG = 200                      # log lines kept per record
MAX_EVENTS = 5000                  # events kept per job (oldest progress events go first)
INLINE_RESULT_MAX = 2 * 1024 * 1024
KILL_GRACE_S = 5.0
READER_WORKERS = 2

_AUTO = object()                   # progress(): compute from done/total


class JobCancelled(Exception):
    """Raised inside a runner (by ctx.check_cancel) to stop a cancelled job."""


@dataclass
class JobKind:
    """A kind of job. `params` (a pydantic model) validates the POST body."""
    name: str
    run: Callable[["JobContext"], Any]
    writer: bool = False
    params: type[BaseModel] | None = None
    requires_model: bool | Callable[[dict], bool] = False
    label: str = ""


KINDS: dict[str, JobKind] = {}


def register_kind(kind: JobKind) -> JobKind:
    """Add (or replace) a job kind. Later milestones call this for scan, backtest, ratings."""
    KINDS[kind.name] = kind
    return kind


@dataclass
class Event:
    id: int
    type: str
    data: dict


@dataclass
class Job:
    id: str
    kind: str
    params: dict
    writer: bool = False
    state: str = "queued"
    progress: float | None = None
    stage: str = ""
    done: int | None = None
    total: int | None = None
    log: list = field(default_factory=list)
    created_at: str = ""
    started_at: str | None = None
    finished_at: str | None = None
    result_ref: str | None = None
    result_url: str | None = None
    error: str | None = None
    # runtime only (not persisted)
    seq: int = 0                       # submission order (ties within the same second)
    events: list = field(default_factory=list, repr=False)
    last_event_id: int = 0
    cancel_requested: bool = False
    interrupted: bool = False
    proc: Any = field(default=None, repr=False)
    thread: Any = field(default=None, repr=False)

    RECORD_KEYS = ("id", "kind", "state", "progress", "stage", "done", "total", "log", "params",
                   "created_at", "started_at", "finished_at", "result_ref", "result_url",
                   "error", "writer")

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL

    def record(self) -> dict:
        rec = {k: getattr(self, k) for k in self.RECORD_KEYS}
        rec["log"] = list(self.log)
        return jsonutil.clean(rec)

    @classmethod
    def from_record(cls, rec: dict) -> "Job":
        known = {k: rec[k] for k in cls.RECORD_KEYS if k in rec}
        known.setdefault("params", {})
        return cls(**known)

    def summary(self) -> dict:
        return {"id": self.id, "kind": self.kind, "state": self.state, "progress": self.progress,
                "stage": self.stage, "done": self.done, "total": self.total,
                "writer": self.writer, "started_at": self.started_at or self.created_at}


class JobContext:
    """What a runner sees: progress/log/item reporting, cancel checks, subprocesses."""

    def __init__(self, manager: "JobManager", job: Job):
        self._m = manager
        self.job = job
        self.params = job.params
        self.result_ref: str | None = None      # override where the result lives (e.g. a scan file)
        self.result_url: str | None = None      # override the done event's result_url

    @property
    def cancelled(self) -> bool:
        return self.job.cancel_requested

    def check_cancel(self):
        if self.job.cancel_requested:
            raise JobCancelled()

    def progress(self, done=None, total=None, stage=None, progress=_AUTO):
        """Update progress. progress=None means indeterminate (e.g. within a training stage)."""
        job = self.job
        with self._m.lock:
            if done is not None:
                job.done = int(done)
            if total is not None:
                job.total = int(total)
            if stage is not None:
                job.stage = redact(str(stage))[:300]
            if progress is _AUTO:
                if job.done is not None and job.total:
                    job.progress = max(0.0, min(1.0, job.done / job.total))
            else:
                job.progress = None if progress is None else max(0.0, min(1.0, float(progress)))
            self._m._emit(job, "progress", _progress_data(job))

    def log(self, msg):
        line = redact(str(msg)).rstrip()
        if not line:
            return
        job = self.job
        with self._m.lock:
            job.log.append(line)
            del job.log[:-MAX_LOG]
            self._m._emit(job, "log", {"msg": line})

    def item(self, index: int, data):
        with self._m.lock:
            self._m._emit(self.job, "item", {"index": int(index), "data": jsonutil.clean(data)})

    def run_subprocess(self, cmd: list, on_line: Callable[[str], None] | None = None,
                       timeout: float | None = None, cwd=None, env: dict | None = None) -> int:
        """Run `cmd` (cwd defaults to the project root), feeding each output line to
        on_line (default: ctx.log). Cancel terminates it, then kills it after 5 s.
        Raises JobCancelled when cancelled and TimeoutError after `timeout` seconds."""
        kw = dict(stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                  text=True, encoding="utf-8", errors="replace", bufsize=1,
                  cwd=str(cwd or paths.ROOT), env={**os.environ, "PYTHONUNBUFFERED": "1", **(env or {})})
        if os.name == "posix":
            kw["start_new_session"] = True          # own process group: cancel reaches grandchildren
        else:
            kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        proc = subprocess.Popen([str(c) for c in cmd], **kw)
        with self._m.lock:
            self.job.proc = proc
            cancelled_already = self.job.cancel_requested
        if cancelled_already:
            terminate_process(proc)

        timed_out = threading.Event()
        timer = None
        if timeout:
            def _on_timeout():
                timed_out.set()
                terminate_process(proc)
            timer = threading.Timer(timeout, _on_timeout)
            timer.daemon = True
            timer.start()
        handle = on_line or self.log
        try:
            for line in proc.stdout:
                line = line.rstrip("\r\n")
                if line:
                    handle(line)
            rc = proc.wait()
        except BaseException:
            terminate_process(proc)
            raise
        finally:
            if timer:
                timer.cancel()
            proc.stdout.close()
            with self._m.lock:
                self.job.proc = None
        if self.job.cancel_requested:
            raise JobCancelled()
        if timed_out.is_set():
            raise TimeoutError(f"Timed out after {timeout:.0f} s")
        return rc


def terminate_process(proc, grace: float = KILL_GRACE_S):
    """terminate() (SIGTERM to the process group), then kill() after `grace` seconds.

    Runs in the background so the caller (e.g. a cancel request) returns at once."""

    def _signal(term: bool):
        if proc.poll() is not None and os.name != "posix":
            return
        try:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGTERM if term else signal.SIGKILL)
            elif term:
                proc.terminate()
            else:
                proc.kill()
        except (ProcessLookupError, PermissionError, OSError):
            pass

    def _run():
        _signal(True)
        try:
            proc.wait(grace)
        except subprocess.TimeoutExpired:
            _signal(False)

    threading.Thread(target=_run, name=f"kill-{proc.pid}", daemon=True).start()


def _progress_data(job: Job) -> dict:
    return {"progress": job.progress, "stage": job.stage, "done": job.done, "total": job.total}


def _done_data(job: Job) -> dict:
    return {"state": job.state, "result_url": job.result_url, "error": job.error}


def _validation_detail(err: ValidationError) -> str:
    parts = []
    for e in err.errors():
        loc = ".".join(str(x) for x in e.get("loc", ()))
        parts.append(f"{loc}: {e.get('msg')}" if loc else str(e.get("msg")))
    return "; ".join(parts) or "Invalid parameters"


class JobManager:
    """In-process job registry. One per app; see start()/manager() below."""

    def __init__(self, jobs_file: Path | None = None, results_dir: Path | None = None,
                 reader_workers: int = READER_WORKERS):
        self.jobs_file = Path(jobs_file or paths.JOBS_FILE)
        self.results_dir = Path(results_dir or paths.RESULTS_DIR)
        self.lock = threading.RLock()
        self._io_lock = threading.Lock()
        self.jobs: dict[str, Job] = {}
        self._readers = ThreadPoolExecutor(max_workers=reader_workers, thread_name_prefix="job-reader")
        self._sync_writer: str | None = None      # label of a running synchronous destructive action
        self._listeners: list[Callable[[Job], None]] = []
        self._closed = False

    # ── persistence ───────────────────────────────────────────────────────

    def load(self):
        """Read runs/jobs.json; jobs that were running when the server stopped become interrupted."""
        data = None
        try:
            with open(self.jobs_file, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            pass
        changed = False
        records = (data or {}).get("jobs", [])
        for i, rec in enumerate(records):              # saved newest first
            try:
                job = Job.from_record(rec)
            except TypeError:
                continue
            job.seq = len(records) - i
            if job.state in ACTIVE:
                job.state = "interrupted"
                job.finished_at = job.finished_at or clock.iso_utc()
                job.error = job.error or "The server stopped before this job finished."
                changed = True
            self.jobs[job.id] = job
        if changed:
            self._save()
        return self

    def _save(self):
        # Snapshot and write under one lock, so an older snapshot never overwrites a newer one.
        with self._io_lock:
            with self.lock:
                records = [j.record() for j in self._ordered()[:MAX_JOBS]]
            try:
                self.jobs_file.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_json(self.jobs_file, {"jobs": records}, ensure_ascii=False)
            except OSError:
                log.exception("Could not write %s", self.jobs_file)

    def _ordered(self) -> list:
        return sorted(self.jobs.values(), key=lambda j: (j.created_at, j.seq), reverse=True)

    def _trim(self):
        """Keep at most MAX_JOBS records in memory (dropping the oldest finished ones)."""
        old = [j for j in self._ordered()[MAX_JOBS:] if j.terminal]
        for j in old:
            self.jobs.pop(j.id, None)

    # ── events ────────────────────────────────────────────────────────────

    def _emit(self, job: Job, type_: str, data: dict):
        """Append an event (caller holds the lock)."""
        job.last_event_id += 1
        job.events.append(Event(job.last_event_id, type_, data))
        if len(job.events) > MAX_EVENTS:
            idx = next((i for i, e in enumerate(job.events) if e.type == "progress"), 0)
            del job.events[idx]

    def events_after(self, job: Job, cursor: int) -> list:
        with self.lock:
            if not job.events and job.terminal:
                # loaded from runs/jobs.json: no live events, so describe the final state
                self._emit(job, "progress", _progress_data(job))
                self._emit(job, "done", _done_data(job))
            return [e for e in job.events if e.id > cursor]

    # ── queries ───────────────────────────────────────────────────────────

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def get_or_404(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job is None:
            raise ApiError("not_found", f"No job with id {job_id!r}.")
        return job

    def list(self, active: bool | None = None, kind: str | None = None, limit: int = 20) -> list:
        with self.lock:
            out = self._ordered()
        if active is not None:
            out = [j for j in out if (j.state in ACTIVE) == active]
        if kind:
            out = [j for j in out if j.kind == kind]
        return out[:max(0, limit)]

    def active_writer(self) -> Job | None:
        with self.lock:
            return next((j for j in self.jobs.values() if j.writer and j.state in ACTIVE), None)

    def active_summary(self) -> list:
        with self.lock:
            active = [j for j in self.jobs.values() if j.state in ACTIVE]
        return [j.summary() for j in sorted(active, key=lambda j: (j.created_at, j.seq))]

    def result(self, job: Job):
        """The saved result if it is small enough to inline (≤ 2 MB), else None."""
        path = self.results_dir / f"{job.id}.json"
        try:
            if path.stat().st_size > INLINE_RESULT_MAX:
                return None
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def add_listener(self, fn: Callable[[Job], None]):
        """Call fn(job) on a worker thread whenever a job finishes (any terminal state)."""
        self._listeners.append(fn)

    # ── writer lock ───────────────────────────────────────────────────────

    def _busy_error(self) -> ApiError | None:
        writer = self.active_writer()
        if writer:
            label = (KINDS.get(writer.kind).label if writer.kind in KINDS else "") or writer.kind
            return ApiError("busy", f"{label} is running. Wait for it to finish or cancel it.",
                            job_id=writer.id)
        if self._sync_writer:
            return ApiError("busy", f"{self._sync_writer} is in progress. Try again in a moment.")
        return None

    @contextmanager
    def writer_lock(self, action: str):
        """Hold the writer slot for a synchronous destructive action (409 busy if taken)."""
        with self.lock:
            err = self._busy_error()
            if err:
                raise err
            self._sync_writer = action
        try:
            yield
        finally:
            with self.lock:
                self._sync_writer = None

    # ── lifecycle ─────────────────────────────────────────────────────────

    def submit(self, kind_name: str, params: dict | None = None) -> Job:
        """Validate and start a job; raises ApiError not_found / validation / no_model / busy."""
        kind = KINDS.get(kind_name)
        if kind is None:
            raise ApiError("not_found", f"Unknown job kind {kind_name!r}.",
                           hint="Known kinds: " + ", ".join(sorted(KINDS)))
        params = dict(params or {})
        if kind.params is not None:
            try:
                params = kind.params.model_validate(params).model_dump()
            except ValidationError as e:
                raise ApiError("validation", _validation_detail(e)) from None
        needs_model = kind.requires_model(params) if callable(kind.requires_model) else kind.requires_model
        if needs_model and not paths.MODEL_FILE.exists():
            raise ApiError("no_model", "This job needs a trained model.",
                           hint="Train a model on the Model page first.")
        with self.lock:
            if self._closed:
                raise ApiError("busy", "The server is shutting down.")
            if kind.writer:
                err = self._busy_error()
                if err:
                    raise err
            job = Job(id=uuid.uuid4().hex[:12], kind=kind.name, params=params, writer=kind.writer,
                      created_at=clock.iso_utc(), stage="Queued",
                      seq=max((j.seq for j in self.jobs.values()), default=0) + 1)
            self.jobs[job.id] = job
            self._emit(job, "progress", _progress_data(job))
            self._trim()
            if kind.writer:
                job.thread = threading.Thread(target=self._execute, args=(job, kind),
                                              name=f"job-{kind.name}-{job.id}", daemon=True)
                job.thread.start()
            else:
                self._readers.submit(self._execute, job, kind)
        self._save()
        return job

    def cancel(self, job_id: str) -> str:
        """Request cancellation; returns the state to report ("cancelling" or the final state)."""
        job = self.get_or_404(job_id)
        with self.lock:
            if job.terminal:
                return job.state
            job.cancel_requested = True
            proc = job.proc
            if job.state == "queued":           # never started: finish now, the pool will skip it
                self._finish(job, "cancelled")
                return job.state
            job.stage = "Cancelling…"
            self._emit(job, "progress", _progress_data(job))
        if proc is not None:
            terminate_process(proc)
        self._save()
        return "cancelling"

    def _execute(self, job: Job, kind: JobKind):
        with self.lock:
            if job.terminal:
                return
            if job.cancel_requested:
                self._finish(job, "cancelled")
                return
            job.state = "running"
            job.started_at = clock.iso_utc()
            if job.stage == "Queued":
                job.stage = "Starting…"
            self._emit(job, "progress", _progress_data(job))
        self._save()

        ctx = JobContext(self, job)
        result, state, error = None, "done", None
        try:
            result = kind.run(ctx)
            if job.cancel_requested:
                state = "cancelled"
        except JobCancelled:
            state = "cancelled"
        except Exception as e:                         # noqa: BLE001  (any runner failure)
            if job.cancel_requested:
                state = "cancelled"
            else:
                log.exception("Job %s (%s) failed", job.id, job.kind)
                state, error = "error", redact(str(e).strip() or type(e).__name__)
        if job.interrupted:
            state = "interrupted"
            error = error or "The server stopped before this job finished."

        if result is not None:
            try:
                self.results_dir.mkdir(parents=True, exist_ok=True)
                atomic_write_json(self.results_dir / f"{job.id}.json", jsonutil.clean(result),
                                  ensure_ascii=False, allow_nan=False)
                job.result_ref = ctx.result_ref or f"runs/results/{job.id}.json"
                job.result_url = ctx.result_url or f"/api/jobs/{job.id}"
            except (OSError, TypeError, ValueError) as e:
                log.exception("Could not save the result of job %s", job.id)
                if state == "done":
                    state, error = "error", f"Could not save the result: {e}"
        with self.lock:
            self._finish(job, state, error)

    def _finish(self, job: Job, state: str, error: str | None = None):
        """Mark a job finished (caller holds the lock), persist, notify listeners."""
        job.state = state
        job.error = redact(error) if error else None
        job.finished_at = clock.iso_utc()
        if state == "done":
            job.progress = 1.0
            if job.total:
                job.done = job.total
        job.stage = {"done": "Finished", "cancelled": "Cancelled",
                     "interrupted": "Interrupted"}.get(state, "Failed")
        self._emit(job, "progress", _progress_data(job))
        self._emit(job, "done", _done_data(job))
        threading.Thread(target=self._after_finish, args=(job,), daemon=True).start()

    def _after_finish(self, job: Job):
        self._save()
        for fn in list(self._listeners):
            try:
                fn(job)
            except Exception:                          # noqa: BLE001
                log.exception("Job listener failed for %s", job.id)

    def shutdown(self, timeout: float = KILL_GRACE_S + 1):
        """Stop accepting jobs; interrupt running ones (subprocesses are terminated)."""
        with self.lock:
            self._closed = True
            running = [j for j in self.jobs.values() if j.state in ACTIVE]
            for job in running:
                job.cancel_requested = True
                job.interrupted = True
                if job.proc is not None:
                    terminate_process(job.proc, grace=min(timeout, KILL_GRACE_S))
                if job.state == "queued":
                    self._finish(job, "interrupted", "The server stopped before this job started.")
        self._readers.shutdown(wait=False, cancel_futures=True)
        for job in running:
            if job.thread is not None:
                job.thread.join(timeout)
        self._save()


# ── the app's manager ─────────────────────────────────────────────────────────

_manager: JobManager | None = None
_closing = threading.Event()        # set on Ctrl+C / SIGTERM: SSE streams end at once


def start() -> JobManager:
    """Create the app's manager (called at startup) and load runs/jobs.json."""
    global _manager
    _closing.clear()
    _manager = JobManager(paths.JOBS_FILE, paths.RESULTS_DIR).load()
    return _manager


def end_streams_on_shutdown_signal():
    """Chain the server's SIGINT/SIGTERM handlers so open SSE streams end immediately.

    Uvicorn's graceful shutdown waits for open requests, and a job stream stays open
    until its job ends (training: minutes). Uvicorn restores the original handlers
    when it exits. Outside the main thread (e.g. TestClient) this does nothing.
    """
    if threading.current_thread() is not threading.main_thread():
        return
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous = signal.getsignal(sig)
        if not callable(previous):
            continue

        def handler(signum, frame, previous=previous):
            _closing.set()
            previous(signum, frame)

        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            pass


def stop():
    """Interrupt running jobs (called at shutdown)."""
    global _manager
    if _manager is not None:
        _manager.shutdown()
        _manager = None


def manager() -> JobManager:
    """The current manager (started lazily outside the app, e.g. in scripts)."""
    return _manager or start()


def active_summary() -> list:
    """[{id, kind, progress, stage, …}] for /api/status; [] when no manager is running."""
    return _manager.active_summary() if _manager is not None else []


async def event_stream(job: Job, last_event_id: int | None = None, poll: float = 0.4):
    """SSE events for a job, resuming after `last_event_id`; ends after the done event."""
    from fastapi.sse import ServerSentEvent
    m = manager()
    m.events_after(job, 0)               # jobs loaded from disk get their final events here
    cursor = last_event_id or 0
    if cursor > job.last_event_id:
        cursor = 0                       # an id from an earlier server run: replay from the start
    while True:
        for ev in m.events_after(job, cursor):
            cursor = ev.id
            yield ServerSentEvent(event=ev.type, id=str(ev.id), raw_data=jsonutil.dumps(ev.data))
        if _closing.is_set() or (job.terminal and not m.events_after(job, cursor)):
            return
        await asyncio.sleep(poll)

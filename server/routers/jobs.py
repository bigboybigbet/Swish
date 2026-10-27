"""Background jobs API: start, list, inspect, stream (SSE) and cancel."""
from fastapi import APIRouter, Body, Depends, Header, Query
from fastapi.sse import EventSourceResponse

from server import jobs

router = APIRouter(prefix="/api/jobs", tags=["jobs"])
ORDER = 20


def on_startup(app):
    app.state.jobs = jobs.start()
    jobs.end_streams_on_shutdown_signal()


def on_shutdown(app):
    jobs.stop()


def _job(job_id: str) -> jobs.Job:
    """Path dependency: the job or 404 (resolved before an SSE stream starts)."""
    return jobs.manager().get_or_404(job_id)


@router.post("/{kind}", status_code=202)
def start_job(kind: str, params: dict | None = Body(default=None)):
    """Start a job: 202 {job_id}, or 409 busy / 412 no_model / 422 validation / 404 unknown kind."""
    job = jobs.manager().submit(kind, params or {})
    return {"job_id": job.id, "kind": job.kind, "state": job.state, "writer": job.writer}


@router.get("")
def list_jobs(active: bool | None = None, kind: str | None = None,
              limit: int = Query(20, ge=1, le=100)):
    return [j.record() for j in jobs.manager().list(active=active, kind=kind, limit=limit)]


@router.get("/{job_id}")
def get_job(job_id: str):
    m = jobs.manager()
    job = m.get_or_404(job_id)
    rec = job.record()
    rec["result"] = m.result(job) if job.result_ref else None
    return rec


@router.get("/{job_id}/events", response_class=EventSourceResponse)
async def job_events(job: jobs.Job = Depends(_job), last_event_id: str | None = Header(None),
                     after: int | None = Query(None, ge=0, description="Resume after this event id")):
    """SSE: progress / log / item / done events; resumes after Last-Event-ID (or ?after=)."""
    try:
        cursor = int(last_event_id) if last_event_id else (after or 0)
    except ValueError:
        cursor = after or 0
    async for event in jobs.event_stream(job, cursor):
        yield event


@router.delete("/{job_id}")
def cancel_job(job_id: str):
    return {"state": jobs.manager().cancel(job_id)}

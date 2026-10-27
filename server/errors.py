"""
Error envelope: {error: code, detail: str, hint?: str, job_id?: str, ...}.

Raise ApiError(code, detail, hint=..., **extra) anywhere in a route or
service; the handlers below turn it (and FastAPI/Starlette errors) into the
envelope. HTML page requests that 404 get the 404 page instead.
"""
import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from server.jsonutil import SafeJSONResponse

log = logging.getLogger("server")

STATUS = {
    "validation":           422,
    "team_not_found":       404,
    "not_found":            404,
    "no_model":             412,
    "busy":                 409,
    "confirm_required":     400,
    "bad_request":          400,
    "upstream_unavailable": 503,
    "forbidden_host":       403,
    "forbidden_origin":     403,
    "loopback_only":        403,
    "method_not_allowed":   405,
    "internal":             500,
}


class ApiError(Exception):
    """An API error with a stable code; extra kwargs (job_id, source…) join the envelope."""

    def __init__(self, code: str, detail: str = "", status: int | None = None,
                 hint: str | None = None, **extra):
        super().__init__(detail or code)
        self.code = code
        self.detail = detail
        self.status = status or STATUS.get(code, 400)
        self.hint = hint
        self.extra = extra


def error_body(code: str, detail: str = "", hint: str | None = None, **extra) -> dict:
    body = {"error": code, "detail": detail}
    if hint:
        body["hint"] = hint
    body.update({k: v for k, v in extra.items() if v is not None})
    return body


def _redact(text) -> str:
    from server.security import redact      # late import: security imports this module
    return redact(text)


def _wants_html(request: Request) -> bool:
    path = request.url.path
    return (request.method in ("GET", "HEAD")
            and not path.startswith(("/api/", "/static/"))
            and "text/html" in request.headers.get("accept", ""))


def _validation_text(errors) -> str:
    parts = []
    for e in errors:
        loc = ".".join(str(x) for x in e.get("loc", ()) if x not in ("body", "query", "path"))
        parts.append(f"{loc}: {e.get('msg')}" if loc else str(e.get("msg")))
    return "; ".join(parts) or "Invalid request"


def install(app: FastAPI, templates=None):
    """Register the envelope handlers on the app."""

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError):
        body = error_body(exc.code, _redact(exc.detail),
                          _redact(exc.hint) if exc.hint else None, **exc.extra)
        return SafeJSONResponse(body, status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        errors = [{"loc": list(e.get("loc", ())), "msg": e.get("msg"), "type": e.get("type")}
                  for e in exc.errors()]
        body = error_body("validation", _redact(_validation_text(errors)), errors=errors)
        return SafeJSONResponse(body, status_code=422)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        if exc.status_code == 404 and templates is not None and _wants_html(request):
            from server import __version__
            return templates.TemplateResponse(
                request, "404.html", {"page": "404", "version": __version__},
                status_code=404)
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        return SafeJSONResponse(error_body(code, _redact(detail)),
                                status_code=exc.status_code, headers=exc.headers)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        log.exception("Unhandled error on %s %s", request.method, request.url.path)
        detail = _redact(f"{type(exc).__name__}: {exc}")
        return SafeJSONResponse(error_body("internal", detail), status_code=500)

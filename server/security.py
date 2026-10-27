"""
Request guard, loopback check and secret redaction.

HostOriginGuard is pure ASGI middleware (not @app.middleware("http")) so SSE
streams pass through unbuffered. It rejects:
  - any request whose Host is not one of the allowed loopback names with the
    server's port (403 forbidden_host; DNS-rebinding defence);
  - any non-GET/HEAD request whose Origin is present and not same-origin
    (403 forbidden_origin; CSRF defence for a localhost API).
"""
import ipaddress
import logging
from urllib.parse import quote

from fastapi import Request

from server import config_store
from server.errors import ApiError, error_body
from server.jsonutil import dumps

LOOPBACK_CLIENTS = {"127.0.0.1", "::1"}
SAFE_METHODS = {"GET", "HEAD"}
MASK = "•••"


def is_loopback_host(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def allowed_hosts(bind_host: str, port: int) -> set:
    """Host header values the server answers to."""
    names = {"127.0.0.1", "localhost", "[::1]"}
    if bind_host and bind_host not in ("0.0.0.0", "::") and not is_loopback_host(bind_host):
        names.add(f"[{bind_host}]" if ":" in bind_host else bind_host)
    hosts = {f"{n}:{port}" for n in names}
    if port == 80:
        hosts |= names
    return hosts


class HostOriginGuard:
    """Pure ASGI middleware enforcing the Host allow-list and same-origin writes."""

    def __init__(self, app, allowed: set):
        self.app = app
        self.allowed = {h.lower() for h in allowed}

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)

        headers = {k.decode("latin-1").lower(): v.decode("latin-1")
                   for k, v in scope.get("headers", [])}
        host = headers.get("host", "").strip().lower()
        if host not in self.allowed:
            return await _reject(send, "forbidden_host",
                                 "Requests must use http://127.0.0.1 or http://localhost with the server's port.")

        method = scope.get("method", "GET").upper()
        origin = headers.get("origin")
        if method not in SAFE_METHODS and origin is not None:
            if origin.strip().lower() != f"http://{host}":
                return await _reject(send, "forbidden_origin",
                                     "Cross-origin requests are not allowed.")
        return await self.app(scope, receive, send)


async def _reject(send, code: str, detail: str):
    body = dumps(error_body(code, detail)).encode("utf-8")
    await send({"type": "http.response.start", "status": 403,
                "headers": [(b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})


def is_loopback_client(request: Request) -> bool:
    return bool(request.client) and request.client.host in LOOPBACK_CLIENTS


def require_loopback(request: Request):
    """FastAPI dependency for secrets and destructive endpoints."""
    if not is_loopback_client(request):
        raise ApiError("loopback_only", "This action is only allowed from this computer.")


# ── redaction ─────────────────────────────────────────────────────────────────

def redact(text) -> str:
    """Replace every configured secret value (and its URL-encoded form) with •••."""
    if text is None:
        return text
    s = str(text)
    values = set()
    for v in config_store.secret_values():
        values.add(v)
        values.add(quote(v, safe=""))
    for v in sorted(values, key=len, reverse=True):
        if v:
            s = s.replace(v, MASK)
    return s


class RedactFilter(logging.Filter):
    """Logging filter that redacts secrets from the message and any traceback."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        clean = redact(msg)
        if clean != msg:
            record.msg, record.args = clean, ()
        if record.exc_info and not record.exc_text:
            # Formatter.format() reuses exc_text when it is already set.
            record.exc_text = redact(logging.Formatter().formatException(record.exc_info))
        return True


_FILTER = RedactFilter()


def install_log_redaction():
    """Attach the redaction filter to the root and uvicorn handlers; quieten urllib3.

    Filters on a logger only see records logged on that logger, so the filter
    goes on every handler. The root logger gets a stderr handler if it has
    none, so app warnings don't fall through to logging's unfiltered lastResort.
    """
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s: %(message)s"))
        root.addHandler(handler)
    for name in ("", "uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        if _FILTER not in logger.filters:
            logger.addFilter(_FILTER)
        for handler in logger.handlers:
            if _FILTER not in handler.filters:
                handler.addFilter(_FILTER)

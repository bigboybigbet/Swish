"""
create_app(): the FastAPI application.

    uvicorn server.app:create_app --factory      (or: python -m server)

Wires the Host/Origin guard, the error envelope, /static (vendor files
cached as immutable, app files no-cache), Jinja page shells, and every
router module in server/routers/ (see server/routers/__init__.py).
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from server import __version__, clock, config_store, errors, paths, routers, security
from server.jsonutil import SafeJSONResponse

log = logging.getLogger("server")

VENDOR_CACHE = "public, max-age=31536000, immutable"


class AppStaticFiles(StaticFiles):
    """Static files: versioned vendor paths are immutable, app JS/CSS revalidate."""

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        vendored = path.replace("\\", "/").startswith("vendor/")
        response.headers["Cache-Control"] = VENDOR_CACHE if vendored else "no-cache"
        return response


def create_app(host: str | None = None, port: int | None = None) -> FastAPI:
    """Build the app. host/port default to NBA_HOST/NBA_PORT (set by python -m server)."""
    paths.chdir_root()
    demo = clock.demo_date()                       # fail fast on a malformed demo date
    env_host, env_port = config_store.server_address()
    host, port = host or env_host, port or env_port

    modules = routers.discover()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        security.install_log_redaction()
        if demo:
            log.warning("Demo mode: replaying %s (NBA_QUANT_DEMO_DATE)", demo.isoformat())
        for mod in modules:
            if hasattr(mod, "on_startup"):
                mod.on_startup(app)
        yield
        for mod in reversed(modules):
            if hasattr(mod, "on_shutdown"):
                mod.on_shutdown(app)

    app = FastAPI(
        title="Swish", version=__version__, lifespan=lifespan,
        default_response_class=SafeJSONResponse,
        docs_url="/api/docs", redoc_url=None, openapi_url="/api/openapi.json",
    )
    templates = Jinja2Templates(directory=str(paths.TEMPLATES_DIR))
    app.state.templates = templates
    app.state.bind = (host, port)

    errors.install(app, templates)
    app.mount("/static", AppStaticFiles(directory=str(paths.STATIC_DIR)), name="static")
    for mod in modules:
        app.include_router(mod.router)

    # No GZip and no CORS middleware (same-origin only; SSE must not be buffered).
    app.add_middleware(security.HostOriginGuard, allowed=security.allowed_hosts(host, port))
    return app

"""
Demo mode hook and GET /api/demo.

With NBA_QUANT_DEMO_DATE set, on_startup installs the historical data
sources (server/services/demo.py) before any request is served, and
on_shutdown restores the live ones. Live mode is untouched.
"""
from fastapi import APIRouter

from server import clock
from server.services import demo

router = APIRouter(prefix="/api", tags=["meta"])
ORDER = 5                     # before every other router's startup hook


def on_startup(app):
    if clock.is_demo():
        demo.install()


def on_shutdown(app):
    demo.uninstall()


@router.get("/demo")
def demo_info():
    """{date, history_dir, first, last} in demo mode, null in live mode."""
    return demo.info() if demo.active() else None

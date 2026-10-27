"""GET /api/schedule, /api/scans and /api/scans/{scan_id} (the scan job lives in services/scan)."""
from datetime import date as Date

from fastapi import APIRouter

from server import clock
from server.errors import ApiError
from server.services import scan

router = APIRouter(prefix="/api", tags=["slate"])


@router.get("/schedule")
def get_schedule(date: Date | None = None):
    return scan.schedule(date or clock.today_et())


@router.get("/scans")
def latest_scan(date: Date | None = None):
    found = scan.latest(date or clock.today_et())
    if found is None:
        raise ApiError("not_found", "No scan for this date yet.")
    return found


@router.get("/scans/{scan_id}")
def get_scan(scan_id: str):
    return scan.load(scan_id)

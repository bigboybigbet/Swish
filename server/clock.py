"""
The server's idea of "today".

Game dates are US Eastern. In demo mode (env NBA_QUANT_DEMO_DATE=YYYY-MM-DD)
the server behaves as if today were that date: today_et() and now_et()
return the demo date (with the real ET time of day). Data sources are
swapped for the historical CSVs by server/services/demo.py; live mode is
never changed by this flag. Always ask this module for dates, never
date.today().
"""
import os
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
DEMO_ENV = "NBA_QUANT_DEMO_DATE"


def demo_date() -> date | None:
    """The demo date, or None in live mode. Raises ValueError on a bad value."""
    raw = os.environ.get(DEMO_ENV, "").strip()
    if not raw:
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError:
        raise ValueError(f"{DEMO_ENV} must be YYYY-MM-DD, got {raw!r}") from None


def is_demo() -> bool:
    return demo_date() is not None


def now_utc() -> datetime:
    """Real wall-clock time (UTC). Not shifted in demo mode."""
    return datetime.now(timezone.utc)


def now_et() -> datetime:
    """Current ET time; in demo mode the date is the demo date."""
    now = datetime.now(ET)
    d = demo_date()
    return now.replace(year=d.year, month=d.month, day=d.day) if d else now


def today_et() -> date:
    return demo_date() or datetime.now(ET).date()


def iso_utc(dt: datetime | None = None) -> str:
    """ISO-8601 UTC timestamp, second precision, 'Z' suffix."""
    dt = (dt or now_utc()).astimezone(timezone.utc)
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def iso_mtime(path) -> str | None:
    """A file's mtime as an ISO UTC timestamp, or None if it doesn't exist."""
    try:
        return iso_utc(datetime.fromtimestamp(os.path.getmtime(path), timezone.utc))
    except OSError:
        return None

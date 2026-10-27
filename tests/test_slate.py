"""Slate: schedule, the scan job and saved scans (demo mode, offline)."""
import os

import pytest

from server.services import demo
from tests.test_demo import DEMO, PHX, demo_client  # noqa: F401  (fixture)
from tests.test_jobs import wait_for

pytestmark = pytest.mark.skipif(not os.path.isdir(demo.history_dir()),
                                reason="historical CSVs not available (NBA_HISTORY_DIR)")


def test_schedule_lists_the_days_games(demo_client):
    s = demo_client.get("/api/schedule").json()
    assert s["date"] == DEMO and len(s["games"]) == 6
    g = s["games"][0]
    assert g["home"]["id"] and g["away"]["id"] and g["status"] == "pre"
    assert demo_client.get("/api/scans").status_code == 404


def test_scan_streams_cards_and_saves_them(demo_client):
    job = demo_client.post("/api/jobs/scan", json={}).json()
    rec = wait_for(demo_client, job["job_id"], timeout=60)
    assert rec["state"] == "done", rec
    scan = demo_client.get("/api/scans").json()
    assert scan["date"] == DEMO and len(scan["games"]) == 6
    cards = {c["away_id"]: c for c in scan["games"]}
    phx = cards[PHX]                                  # PHX @ CHA: Phoenix the clear favourite
    assert phx["mode"] == "team_strength" and phx["win_prob"]["away"] > 0.6
    assert phx["home_spread"] > 0 and 190 < phx["total"] < 250
    assert phx["ratings"]["away"]["rating"] > phx["ratings"]["home"]["rating"]
    assert demo_client.get(f"/api/scans/{scan['scan_id']}").json() == scan


def test_scan_rejects_unknown_params_and_ids(demo_client):
    assert demo_client.post("/api/jobs/scan", json={"foo": 1}).status_code == 422
    assert demo_client.get("/api/scans/../../etc").status_code == 404
    assert demo_client.get("/api/scans/nothere123").status_code == 404

import json
import os
import unicodedata

import requests

from config import STATUS_WEIGHT, PLAYER_IMPACT_FILE, PLAYER_IMPACT_FALLBACK


# ── helpers ──────────────────────────────────────────────────────────────────

def _ascii(name: str) -> str:
    """Strip accents so 'Jokić' matches 'Jokic' across sources."""
    return unicodedata.normalize("NFD", name).encode("ascii", "ignore").decode()


def _team_matches(team_name: str, espn_name: str) -> bool:
    """Non-empty substring match in either direction."""
    if not team_name or not espn_name:
        return False
    a = team_name.strip().lower()
    b = espn_name.strip().lower()
    return a in b or b in a


# ── player impact ─────────────────────────────────────────────────────────────

def load_player_impact() -> dict:
    """Load from player_impact.json; fall back to hardcoded dict."""
    if os.path.exists(PLAYER_IMPACT_FILE):
        with open(PLAYER_IMPACT_FILE) as f:
            data = json.load(f)
        return data.get("players", data)
    return PLAYER_IMPACT_FALLBACK


def load_player_impact_meta() -> dict:
    """Return metadata (season, updated, count) from the JSON file."""
    if os.path.exists(PLAYER_IMPACT_FILE):
        try:
            with open(PLAYER_IMPACT_FILE) as f:
                data = json.load(f)
            return {
                "season":  data.get("season", "—"),
                "updated": data.get("updated", "—"),
                "count":   data.get("count", len(data.get("players", data))),
            }
        except Exception:
            pass
    return {"season": "—", "updated": "—", "count": 0}


# ── injury report ─────────────────────────────────────────────────────────────

def get_injury_report(team_name: str) -> list:
    try:
        data = requests.get(
            "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries",
            timeout=8,
        ).json()
        injured, seen = [], set()
        for entry in data.get("injuries", []):
            tn = entry.get("displayName", "")
            if not _team_matches(team_name, tn):
                continue
            for p in entry.get("injuries", []):
                player_name = (p.get("athlete") or {}).get("displayName", "")
                if not player_name or player_name in seen:
                    continue
                status = p.get("status", "").lower()
                key = next((s for s in STATUS_WEIGHT if s in status), None)
                if key:
                    seen.add(player_name)
                    injured.append({
                        "name":   player_name,
                        "status": key,
                        "detail": p.get("shortComment", ""),
                    })
        return injured
    except Exception:
        return []


# ── strength factor ───────────────────────────────────────────────────────────

def calc_strength_factor(injured: list) -> float:
    impact = load_player_impact()
    loss = sum(
        impact.get(_ascii(p["name"]), 0.05) * STATUS_WEIGHT.get(p["status"], 0.2)
        for p in injured
    )
    return max(0.5, 1.0 - loss)


def apply_injury_adjustment(stats: dict, factor: float) -> dict:
    """
    Translate injury severity into stat penalties.

    `factor` is 1.0 (healthy) → 0.5 (decimated).  Penalties are additive
    for signed stats (net rating, Elo) — multiplying a negative
    PLUS_MINUS by factor < 1 would wrongly make an injured team look
    BETTER — and gently multiplicative for rate/volume stats.
    """
    adj = stats.copy()
    if factor >= 1.0:
        return adj
    pen = 1.0 - factor

    # Scoring volume: a star out (~factor 0.7) ≈ −7.5 projected points
    for k in ["PTS", "PTS10"]:
        if k in adj:
            adj[k] = adj[k] * (1.0 - pen * 0.25)
    # Efficiency rates: mild haircut
    for k in ["eFG_PCT", "EFG10", "TS_PCT", "OREB_PCT", "ORB10",
              "AST_TOV", "REB"]:
        if k in adj:
            adj[k] = adj[k] * (1.0 - pen * 0.15)
    # Signed net-rating stats: subtract points, never scale
    for k in ["PLUS_MINUS", "NET5", "NET15"]:
        if k in adj:
            adj[k] = adj[k] - pen * 15.0
    # Elo: a star out ≈ −75 Elo (~10% win probability)
    if "ELO" in adj:
        adj["ELO"] = adj["ELO"] - pen * 250.0
    return adj

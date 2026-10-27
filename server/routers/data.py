"""Data files and caches: dataset / Elo deletes, API cache, calibrator, player impact.

Deletes and the cache clear are loopback-only and take the jobs writer lock
(409 busy while a writer job runs). File deletes need ?confirm=true.
"""
from fastapi import APIRouter, Depends, Query

from server import clock, jobs, paths
from server.routers import meta
from server.routers.model import _unlink, require_confirm
from server.security import require_loopback

router = APIRouter(prefix="/api", tags=["data"])
ORDER = 35


@router.delete("/dataset", dependencies=[Depends(require_loopback), Depends(require_confirm)])
def delete_dataset():
    """Remove master_dataset_v5.csv (the next training run downloads it again)."""
    with jobs.manager().writer_lock("Deleting the dataset"):
        removed = _unlink(paths.DATASET_FILE)
    return {"ok": True, "removed": removed, "file": paths.DATASET_FILE.name}


@router.delete("/elo", dependencies=[Depends(require_loopback), Depends(require_confirm)])
def delete_elo():
    """Remove elo_ratings.json: every team falls back to 1500 until the next training run."""
    with jobs.manager().writer_lock("Resetting Elo"):
        removed = _unlink(paths.ELO_FILE)
    return {"ok": True, "removed": removed, "file": paths.ELO_FILE.name}


@router.get("/cache")
def cache_stats():
    return meta.cache_info()


@router.delete("/cache", dependencies=[Depends(require_loopback)])
def clear_cache():
    """Empty the NBA API response cache (nba_cache.db)."""
    from data import cache
    with jobs.manager().writer_lock("Clearing the cache"):
        cache.clear_all()
    return {"ok": True, **cache.stats()}


@router.get("/calibrator")
def calibrator():
    """Blend calibrator weights {xgb, mc, elo}, bias and number of online updates."""
    return {**meta.calibrator_info(), "exists": paths.BLEND_FILE.exists(),
            "updated_at": clock.iso_mtime(paths.BLEND_FILE)}


@router.get("/player-impact")
def player_impact(top: int = Query(20, ge=0, le=500)):
    from data.injuries import load_player_impact
    info = meta.player_impact_info()
    players = load_player_impact() or {}
    ranked = sorted(players.items(), key=lambda kv: kv[1], reverse=True)[:top]
    return {**info, "top": [{"name": name, "impact": round(float(v), 4)} for name, v in ranked]}

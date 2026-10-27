import os
from datetime import datetime

MODEL_FILE         = "nba_brain_v5.joblib"
DATASET_FILE       = "master_dataset_v5.csv"
ELO_FILE           = "elo_ratings.json"
HISTORY_FILE       = "prediction_history.csv"
OPTUNA_PARAMS_FILE = "optuna_best_params.json"
PLAYER_IMPACT_FILE = "player_impact.json"
CACHE_DB           = "nba_cache.db"
BLEND_FILE         = "nba_blend_calibrator.json"
TEAM_STRENGTH_FILE = "team_strength.json"

N_SIMS    = 50_000
HOME_ADV  = 2.8     # points of home-court advantage (split ± in simulations)
MC_CORR   = 0.30    # correlation between the two teams' scores (shared pace)

# Training parallelism: leave one core for the web server and the OS.
N_JOBS    = max(1, (os.cpu_count() or 2) - 1)

def season_string(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def current_season(today=None) -> str:
    """NBA season for a date: a season tips off in October, so Oct–Dec
    belong to the season starting that year, Jan–Sep to the previous one."""
    today = today or datetime.today()
    return season_string(today.year if today.month >= 10 else today.year - 1)


def recent_seasons(n: int, today=None) -> list:
    """The n most recent seasons, oldest first, ending with the current one."""
    end = int(current_season(today)[:4])
    return [season_string(y) for y in range(end - n + 1, end + 1)]


# Seasons used to build the training dataset, rolled forward automatically
# (on 2026-10-27 → 2024-25, 2025-26, 2026-27).  Recency weighting
# (RECENCY_HALFLIFE_DAYS) discounts older seasons.
N_TRAIN_SEASONS        = 3
TRAIN_SEASONS          = recent_seasons(N_TRAIN_SEASONS)
CURRENT_SEASON         = TRAIN_SEASONS[-1]
RECENCY_HALFLIFE_DAYS  = 150

# v5 — matchup-level features: every row is ONE GAME (home vs away),
# expressed mostly as home-minus-away differentials so the model always
# sees both teams at once.
FEATURE_COLS = [
    "ELO_DIFF",        # pre-game MOV-adjusted Elo difference
    "NET5_DIFF",       # net rating (plus/minus) diff, last 5 games
    "NET15_DIFF",      # net rating diff, last 15 games
    "PTS10_DIFF",      # scoring diff, last 10
    "PA10_DIFF",       # points-allowed diff, last 10
    "EFG10_DIFF",      # effective FG% diff, last 10
    "TOV10_DIFF",      # turnover rate diff, last 10
    "ORB10_DIFF",      # offensive rebound rate diff, last 10
    "FTR10_DIFF",      # free-throw rate diff, last 10
    "PACE10_MEAN",     # combined pace context
    "WPCT10_DIFF",     # win% diff, last 10
    "STREAK_DIFF",     # win/lose streak diff
    "REST_DIFF",       # rest days diff
    "H_B2B",           # home on back-to-back
    "A_B2B",           # away on back-to-back
    "H_HOME_WPCT",     # home team's win% in its last 10 HOME games
    "A_AWAY_WPCT",     # away team's win% in its last 10 AWAY games
]

ELO_K        = 20
ELO_DEFAULT  = 1500
ELO_HOME_ADV = 70   # Elo points of home advantage in expected-score calc

STATUS_WEIGHT = {
    "out": 1.0, "doubtful": 0.80,
    "questionable": 0.30, "day-to-day": 0.20,
}

PLAYER_IMPACT_FALLBACK = {
    "LeBron James": 0.28, "Stephen Curry": 0.30, "Nikola Jokic": 0.32,
    "Giannis Antetokounmpo": 0.31, "Luka Doncic": 0.33, "Joel Embiid": 0.29,
    "Kevin Durant": 0.27, "Jayson Tatum": 0.26, "Anthony Davis": 0.27,
    "Shai Gilgeous-Alexander": 0.29, "Donovan Mitchell": 0.25, "Ja Morant": 0.26,
    "Damian Lillard": 0.25, "Kawhi Leonard": 0.26, "Paul George": 0.22,
    "Jimmy Butler": 0.24, "Bam Adebayo": 0.22, "Trae Young": 0.26,
    "Devin Booker": 0.25, "Karl-Anthony Towns": 0.24,
}

import json
import os

from config import ELO_FILE, ELO_K, ELO_DEFAULT, ELO_HOME_ADV


def load_elo() -> dict:
    if os.path.exists(ELO_FILE):
        with open(ELO_FILE) as f:
            return json.load(f)
    return {}


def save_elo(d: dict):
    with open(ELO_FILE, "w") as f:
        json.dump(d, f)


def elo_expected(ra: float, rb: float) -> float:
    return 1 / (1 + 10 ** ((rb - ra) / 400))


def update_elo_game(home: str, away: str, home_won: bool,
                    margin: float, elo_dict: dict) -> dict:
    """
    MOV-adjusted Elo update (FiveThirtyEight style).

    - Home court is worth ELO_HOME_ADV points in the expected-score calc.
    - K is scaled by margin of victory, dampened when the favourite wins
      big (autocorrelation correction) so blowouts by strong teams don't
      inflate ratings.
    """
    rh = elo_dict.get(home, ELO_DEFAULT)
    ra = elo_dict.get(away, ELO_DEFAULT)
    eh = elo_expected(rh + ELO_HOME_ADV, ra)

    winner_diff = (rh + ELO_HOME_ADV - ra) if home_won else (ra - rh - ELO_HOME_ADV)
    mov  = max(abs(margin), 1.0)
    mult = ((mov + 3.0) ** 0.8) / (7.5 + 0.006 * winner_diff)
    k    = ELO_K * mult

    sh = 1.0 if home_won else 0.0
    elo_dict[home] = round(rh + k * (sh - eh), 2)
    elo_dict[away] = round(ra + k * ((1.0 - sh) - (1.0 - eh)), 2)
    return elo_dict


def update_elo(winner: str, loser: str, elo_dict: dict) -> dict:
    """Legacy simple update — kept for backward compatibility."""
    rw = elo_dict.get(winner, ELO_DEFAULT)
    rl = elo_dict.get(loser,  ELO_DEFAULT)
    ew = elo_expected(rw, rl)
    elo_dict[winner] = round(rw + ELO_K * (1 - ew), 2)
    elo_dict[loser]  = round(rl + ELO_K * (0 - (1 - ew)), 2)
    return elo_dict


def get_team_elo(team_name: str) -> float:
    elo = load_elo()
    for key in elo:
        if team_name.lower() in key.lower() or key.lower() in team_name.lower():
            return elo[key]
    return ELO_DEFAULT

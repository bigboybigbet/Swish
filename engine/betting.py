def american_to_decimal(odds: float) -> float:
    return (odds / 100) + 1 if odds >= 0 else (100 / abs(odds)) + 1


def kelly_criterion(p_win: float, decimal_odds: float, fraction: float = 0.25) -> tuple:
    """Returns (full_kelly_pct, fractional_kelly_pct)."""
    b      = decimal_odds - 1
    full_k = max(0.0, (b * p_win - (1 - p_win)) / b)
    return round(full_k * 100, 2), round(full_k * fraction * 100, 2)

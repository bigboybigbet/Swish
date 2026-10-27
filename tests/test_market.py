"""Market maths: odds conversion, de-vig, fair line, edge, cover/over, Kelly."""
import numpy as np
import pytest

from server.services import market


@pytest.mark.parametrize("ml,dec", [(150, 2.5), (-200, 1.5), (100, 2.0), (-100, 2.0), (-110, 1 + 100 / 110)])
def test_american_decimal_round_trip(ml, dec):
    assert market.american_to_decimal(ml) == pytest.approx(dec)
    assert market.decimal_to_american(dec) == pytest.approx(ml if ml != -100 else 100)


@pytest.mark.parametrize("bad", [0, 50, -99.5])
def test_american_rejects_impossible_prices(bad):
    with pytest.raises(ValueError):
        market.american_to_decimal(bad)


def test_devig_is_proportional():
    dv = market.devig(-110, -110)
    assert dv["implied"]["home"] == pytest.approx(110 / 210)
    assert dv["overround"] == pytest.approx(2 * 110 / 210 - 1)
    assert dv["fair"]["home"] == pytest.approx(0.5)

    dv = market.devig(-180, 150)
    ih, ia = 180 / 280, 100 / 250
    assert dv["fair"]["home"] == pytest.approx(ih / (ih + ia))
    assert dv["fair"]["home"] + dv["fair"]["away"] == pytest.approx(1)
    assert dv["overround"] > 0


@pytest.mark.parametrize("p,ml", [(0.5, -100), (0.6, -150), (0.75, -300), (0.4, 150), (0.2, 400)])
def test_fair_moneyline(p, ml):
    assert market.fair_moneyline(p) == pytest.approx(ml)


def test_fair_moneyline_inverts_to_probability():
    for p in (0.31, 0.5, 0.64, 0.9):
        assert 1 / market.american_to_decimal(market.fair_moneyline(p)) == pytest.approx(p)
    assert market.fair_moneyline(0) is None and market.fair_moneyline(1) is None


def test_edge():
    assert market.edge(0.61, 0.57) == pytest.approx(0.04)
    assert market.edge(0.39, 0.43) == pytest.approx(-0.04)


def test_cover_and_over_probabilities():
    margins = np.array([-10, -3, 2, 4, 5, 8, 12, 20], dtype=float)
    # home −4.5: home covers when margin > 4.5
    assert market.cover_prob(margins, -4.5) == pytest.approx(4 / 8)
    # home +3.5 (underdog): covers when margin > −3.5
    assert market.cover_prob(margins, 3.5) == pytest.approx(7 / 8)
    totals = np.array([200, 210, 220, 230, 240], dtype=float)
    assert market.over_prob(totals, 225.5) == pytest.approx(2 / 5)


def test_kelly_uses_real_price_and_caps():
    k = market.kelly_stake(0.6, 2.0, fraction=0.25, cap_pct=5.0)
    assert k["full_pct"] == pytest.approx(20.0)        # (1*0.6 - 0.4)/1
    assert k["frac_pct"] == pytest.approx(5.0) and k["capped"] is False
    k = market.kelly_stake(0.7, 2.0, fraction=0.25, cap_pct=5.0)
    assert k["full_pct"] == pytest.approx(40.0) and k["frac_pct"] == 5.0 and k["capped"] is True
    k = market.kelly_stake(0.4, 2.0, fraction=0.25, cap_pct=5.0)
    assert k["full_pct"] == 0 and k["frac_pct"] == 0


def test_market_block_value_side_and_probs():
    margins = np.linspace(-20, 30, 1001)
    totals = np.linspace(200, 250, 1001)
    found = {"status": "ok", "source": "DraftKings via ESPN", "fetched_at": None,
             "home_ml": -110, "away_ml": -110, "spread": -4.5, "total": 225.5}
    block = market.market_block(found, 0.60, margins, totals, value_edge=0.04)
    assert block["fair"] == {"home": 0.5, "away": 0.5}
    assert block["edge"]["home"] == pytest.approx(0.10) and block["edge"]["away"] == pytest.approx(-0.10)
    assert block["value_side"] == "home"
    assert block["cover_prob"] == pytest.approx(np.mean(margins > 4.5), abs=1e-4)
    assert block["over_prob"] == pytest.approx(np.mean(totals > 225.5), abs=1e-4)

    block = market.market_block(found, 0.52, margins, totals, value_edge=0.04)
    assert block["value_side"] is None


def test_market_block_without_prices():
    found = {"status": "no_odds_yet", "source": "ESPN", "home_ml": None, "away_ml": None,
             "spread": None, "total": None}
    block = market.market_block(found, 0.6, np.zeros(10), np.zeros(10), 0.04)
    assert block["status"] == "no_odds_yet"
    assert block["fair"] is None and block["edge"] is None and block["value_side"] is None
    assert block["cover_prob"] is None and block["over_prob"] is None
    assert market.kelly_block(0.6, block, {"fraction": 0.25, "cap_pct": 5}) is None


def test_kelly_block_price_source():
    mk = {"status": "manual", "home_ml": 120, "away_ml": -140}
    k = market.kelly_block(0.55, mk, {"fraction": 0.25, "cap_pct": 5.0})
    assert k["price_source"] == "manual"
    assert k["home"]["decimal"] == pytest.approx(2.2)
    assert k["home"]["frac_pct"] > 0 and k["away"]["frac_pct"] == 0
    mk["status"] = "ok"
    assert market.kelly_block(0.55, mk, {"fraction": 0.25, "cap_pct": 5.0})["price_source"] == "market"


def test_fair_line_block():
    fl = market.fair_line(0.6)
    assert fl == {"home_ml": -150.0, "away_ml": 150.0}

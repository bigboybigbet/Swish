"""Recap timing helpers, rotation and news tagging (offline)."""
from server.services import news, recap


def test_elapsed_game_clock():
    assert recap._elapsed(1, "12:00") == 0
    assert recap._elapsed(2, "7:19") == 720 + 281
    assert recap._elapsed(4, "0.0") == 2880
    assert recap._elapsed(5, "5:00") == 2880
    assert recap._elapsed(5, "45.3") == 2880 + 254.7


def test_rotation_from_substitutions():
    players = {"home": [{"id": "a", "starter": True}, {"id": "b", "starter": False}],
               "away": [{"id": "c", "starter": True}]}
    flow = [{"type": "Substitution", "actors": ["b", "a"], "t": 600.0, "period": 1},
            {"type": "Substitution", "actors": ["a", "b"], "t": 1500.0, "period": 3},
            {"type": "End Game", "actors": [], "t": 2880.0, "period": 4}]
    rot = recap._rotation(flow, players)
    assert rot["a"] == [[0.0, 600.0], [1500.0, 2880.0]]
    assert rot["b"] == [[600.0, 1500.0]]
    assert rot["c"] == [[0.0, 2880.0]]


def test_news_tags_flag_availability():
    assert news.classify("Jalen Duren to miss at least two more weeks with hamstring injury") == ["injury"]
    assert "streak" in news.classify("Utah faces Miami, seeks to halt 8-game slide")
    assert news.classify("Lakers sign guard to two-way contract") == ["roster"]
    a = news.article({"headline": "Star ruled out vs. Celtics", "description": "", "categories": [
        {"type": "team", "description": "Boston Celtics", "team": {"description": "Boston Celtics"}}]})
    assert a["availability"] and a["teams"][0]["abbr"] == "BOS"

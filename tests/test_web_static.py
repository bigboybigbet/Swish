"""Templates, vendored assets, static headers and teams.json."""
import importlib.util
import json
import re

import pytest

from server import paths
from server.routers.pages import PAGES

_spec = importlib.util.spec_from_file_location("vendor_script", paths.ROOT / "scripts" / "vendor.py")
vendor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vendor)

LOCK = vendor.load_lock()
VENDOR_FILES = [(pkg, dest) for pkg in LOCK for dest in pkg["files"]]


# ── templates ─────────────────────────────────────────────────────────────────

def _page_templates():
    return sorted(p for p in paths.TEMPLATES_DIR.glob("*.html")
                  if not p.name.startswith("_") and p.name != "base.html")


@pytest.mark.parametrize("tpl", _page_templates(), ids=lambda p: p.name)
def test_every_template_extends_base(tpl):
    assert re.search(r"{%-?\s*extends\s+[\"']base\.html[\"']", tpl.read_text(encoding="utf-8"))


@pytest.mark.parametrize("tpl", _page_templates(), ids=lambda p: p.name)
def test_page_module_exists(tpl):
    m = re.search(r"{%\s*block page_module\s*%}(.*?){%\s*endblock", tpl.read_text(encoding="utf-8"))
    if m and m.group(1).strip():
        rel = m.group(1).strip().removeprefix("/static/")
        assert (paths.STATIC_DIR / rel).exists()


def test_nav_matches_routes(client):
    html = client.get("/").text
    for page, href in PAGES.items():
        assert f'href="{href}"' in html
    labels = re.findall(r'<span class="nav-link-title">([^<]+)</span>', html)
    assert labels == ["Slate", "Game", "Ratings", "Accuracy", "Bets", "Model", "Settings"]


@pytest.mark.parametrize("page,href", PAGES.items())
def test_pages_render(client, page, href):
    r = client.get(href)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert f'data-page="{page}"' in r.text
    assert '<script type="module" src="/static/js/main.js">' in r.text
    if not (paths.TEMPLATES_DIR / f"{page}.html").exists():
        assert "is coming soon" in r.text


def test_shell_makes_no_external_requests(client):
    html = client.get("/").text
    refs = re.findall(r'(?:href|src)="([^"]+)"', html)
    refs += re.findall(r'"(/static/[^"]+)"', html)            # import map
    assert not [r for r in refs if r.startswith(("http:", "https:", "//"))]
    for ref in refs:
        if ref.startswith("/static/"):
            assert (paths.STATIC_DIR / ref.removeprefix("/static/")).exists(), ref


def test_macros_render():
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    env = Environment(loader=FileSystemLoader(paths.TEMPLATES_DIR), autoescape=select_autoescape())
    src = """{% from "_macros.html" import page_header, empty_state, error_state, skeleton, skeleton_rows,
        glossary_tip, stat_tile, team_badge, prob_split_bar, confidence_badge, edge_badge, freshness,
        confirm_modal %}
        {{ page_header("Game") }}{{ empty_state("vs", "Pick two teams") }}{{ error_state("err", "load()") }}
        {{ skeleton(2) }}{{ skeleton_rows(2) }}{{ glossary_tip("elo") }}
        {{ stat_tile("Spread", "$fmt.num(x)", "'model'", "edge") }}{{ team_badge("game.home") }}
        {{ prob_split_bar("g.home", "g.away", "g.p") }}{{ confidence_badge("pick.tier") }}
        {{ edge_badge("m.edge", "'Value: ' + m.abbr") }}{{ freshness("as_of", "stale", "Scoreboard") }}
        {% call confirm_modal("del", "Delete?", "Delete", "remove()") %}Gone for good.{% endcall %}"""
    html = env.from_string(src).render()
    for needle in ('class="empty"', "placeholder-glow", "team-badge", "prob-split",
                   'id="del"', "Gone for good.", "glossary-tip"):
        assert needle in html


# ── vendored libraries ────────────────────────────────────────────────────────

@pytest.mark.parametrize("pkg,dest", VENDOR_FILES, ids=[f"{p['dir']}/{d}" for p, d in VENDOR_FILES])
def test_vendor_file_matches_lock(pkg, dest):
    path = vendor.target_path(pkg, dest)
    assert path.exists(), f"missing {path}; run python scripts/vendor.py"
    assert vendor.sri(path.read_bytes()) == pkg["files"][dest]["sha384"]


def test_vendor_check_passes():
    assert vendor.main(["--check"]) == 0


def test_no_tabler_libs_vendored():
    # dist/libs contains apexcharts (not MIT from 5.3.6): never vendor from it.
    assert not any("dist/libs" in spec.get("src", "") for pkg in LOCK for spec in pkg["files"].values())


def test_vendor_paths_cannot_escape():
    with pytest.raises(ValueError):
        vendor.safe_rel("../../etc/passwd")
    with pytest.raises(ValueError):
        vendor.safe_rel("/abs")


# ── static serving ────────────────────────────────────────────────────────────

def test_mjs_served_as_javascript(client):
    r = client.get("/static/vendor/echarts@6.1.0/echarts.esm.min.mjs")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/javascript")


def test_cache_headers(client):
    vend = client.get("/static/vendor/alpinejs@3.17.4/module.esm.min.js")
    assert vend.headers["cache-control"] == "public, max-age=31536000, immutable"
    assert vend.headers["content-type"].startswith("text/javascript")
    for app_file in ("/static/js/main.js", "/static/css/app.css"):
        assert client.get(app_file).headers["cache-control"] == "no-cache"


def test_font_served(client):
    r = client.get("/static/vendor/@tabler/icons-webfont@3.49.0/fonts/tabler-icons.woff2")
    assert r.status_code == 200 and r.headers["content-type"] == "font/woff2"


# ── teams.json ────────────────────────────────────────────────────────────────

def test_teams_json():
    from nba_api.stats.static import teams as nba_static
    rows = json.loads(paths.TEAMS_JSON.read_text(encoding="utf-8"))
    assert len(rows) == 30
    static = {t["id"]: t for t in nba_static.get_teams()}
    assert {r["id"] for r in rows} == set(static)
    assert len({r["abbr"] for r in rows}) == 30 and len({r["espn_abbr"] for r in rows}) == 30
    for r in rows:
        assert r["abbr"] == static[r["id"]]["abbreviation"]
        assert r["name"] == static[r["id"]]["full_name"]
        assert re.fullmatch(r"#[0-9A-F]{6}", r["primary"]) and re.fullmatch(r"#[0-9A-F]{6}", r["secondary"])
        assert not any(k for k in r if "logo" in k)


def test_glossary_json():
    items = json.loads(paths.GLOSSARY_JSON.read_text(encoding="utf-8"))
    for key in ("elo", "kalman", "sd", "adjusted_net", "healthy_current", "fair_line", "overround",
                "edge", "kelly", "brier", "log_loss", "calibration_gap", "within_3"):
        assert items[key]

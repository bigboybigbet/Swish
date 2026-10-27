# Spec: replace the Streamlit app with FastAPI and an HTML frontend (NBA Quant web v1)

Status: ready to build. Date: 2026-10-27. Repo: `/home/user/nba_app_quant`. Nothing in the repo was changed while writing this.

## Tóm tắt (VI)

- **Bỏ Streamlit.** Thay bằng một server **FastAPI** chạy cục bộ. Server dùng lại nguyên `engine/` và `data/`, trả JSON qua `/api/*` và phục vụ các trang HTML tĩnh (Tabler + Alpine.js + ECharts + Tabulator).
- **Không cần Node hay bước build.** Mọi thư viện được tải sẵn vào `web/static/vendor/` nên app chạy được cả khi offline.
- **Một lệnh để chạy:** `python -m server`, sau đó mở http://127.0.0.1:8000.
- **Tác vụ chậm chạy nền.** Huấn luyện (khoảng 10 phút), backtest, quét lịch và cập nhật cầu thủ đều là background job. Tiến độ hiện theo thời gian thực qua SSE.
- **7 màn hình:** Slate (lịch hôm nay), Game (phân tích một trận), Ratings (đội và cầu thủ), Accuracy (độ chính xác, backtest), Bets (ROI), Model, Settings.
- **Bảo mật khóa API.** Khóa Odds API và Telegram nằm trong `.env`, không bao giờ trả về trình duyệt và được che trong log.
- **Thứ tự triển khai:** M0 đến M5 phải đạt đủ tính năng như app cũ, xong thì xóa `nba_app.py` và Streamlit. Sau đó M6 đến M8 thêm ratings (Elo, Kalman ± độ bất định, cầu thủ), sức mạnh đội có tính chấn thương, và trọng số ensemble.

---

## 0. Decisions at a glance

| Topic | Decision |
|---|---|
| Backend | FastAPI (native SSE from `fastapi.sse`), uvicorn, **1 process, 1 worker**, bound to `127.0.0.1:8000` |
| Page delivery | Jinja2 is used **only for page shells**: a shared `base.html` with nav and vendor tags. Templates get no data. Every number comes from `/api` through `fetch()`. |
| Frontend | Tabler 1.6.1 (CSS and Bootstrap JS), Alpine.js 3.17.4 (ESM with an import map), ECharts 6.1.0, Tabulator 6.6.1, Tabler Icons 3.49.0 |
| Build step | **None.** Libraries are vendored and committed, with sha384 checks. |
| Jobs | In-process `JobManager`. CPU-heavy jobs (training) run as a **subprocess** that prints JSON-line progress. I/O jobs run on a 2-thread pool. Progress uses SSE, with polling as a fallback. |
| Persistence | Existing files stay where they are. New state goes under `runs/` (gitignored) as JSON. |
| Secrets | `.env` at the project root (already gitignored). Env vars override it. Secrets are never returned and are redacted everywhere. |
| Team identity | nba_api team IDs everywhere in the API. `web/static/data/teams.json` holds colours and ESPN name aliases. **No logos.** |
| Dates | Game dates are **US Eastern (America/New_York)**. The UI shows tip-off in local time and in ET. This matters for a user in Vietnam (UTC+7): a 7 pm ET game starts at 7 am the next day locally. |
| No-model mode | If `nba_brain_v5.joblib` is missing (true in this checkout), predictions still work in `mode:"elo_only"`. The README benchmark shows Elo beating v5 XGBoost on log loss (0.634 vs 0.652). |

---

## 1. Architecture

### 1.1 Runtime picture

```
browser ──HTML/JS/CSS──▶ FastAPI (server/app.py) ──▶ web/templates/*.html, web/static/**
   │                         │
   ├─ fetch /api/* (JSON) ──▶├─ routers/* ──▶ services/* ──▶ engine/*  data/*   (imported, reused)
   └─ EventSource /api/jobs/{id}/events ◀── JobManager ──▶ ThreadPool(2) : scan, backtest, autobacktest
                                                       └─▶ subprocess   : train, player_impact, ratings
files: nba_brain_v5.joblib, master_dataset_v5.csv, elo_ratings.json, nba_blend_calibrator.json,
       autobacktest_*.{json,csv}, player_impact.json, nba_cache.db, prediction_history.csv   (unchanged)
new:   .env, settings.json, model_metrics.json, runs/{jobs.json, results/, scans/, ratings_snapshots/}
```

### 1.2 File layout

```
nba_app_quant/
  server/
    __init__.py            # __version__ = "6.0.0-web"
    __main__.py            # `python -m server` entry point (see 1.3)
    app.py                 # create_app(): guard middleware, routers, /static, templates, startup hooks
    paths.py               # ROOT (absolute), all file paths, RUNS_DIR; mimetypes fix for .mjs
    config_store.py        # .env load/write (python-dotenv), settings.json (thresholds, kelly), defaults
    security.py            # pure-ASGI Host/Origin guard, loopback check, redact()
    jsonutil.py            # numpy → python, NaN/inf → None, round helpers
    errors.py              # ApiError(code, status, detail, hint) + exception handlers
    jobs.py                # Job dataclass, JobManager, writer lock, persistence, SSE stream generator
    model_store.py         # XGB model singleton, reload on mtime change; cached SHAP explainer
    services/
      teams.py             # teams.json + nba_api.stats.static.teams; id↔name↔espn_name
      espn.py              # cached scoreboard (schedule+odds+results) and injury feed, stale fallback
      market.py            # american↔decimal, de-vig, fair line, edge, cover/over probabilities
      predict.py           # predict_core(): wraps predict_hybrid + MC extras + market + kelly + ensemble
      ratings.py           # Elo now; Kalman/adjusted/player via engine.ratings when present
      standings.py         # nba_api leaguestandingsv3 (record, L10, streak), cached 1 h
      scan.py  backtest.py  accuracy.py  bets.py  telegram.py  metrics.py
    routers/
      meta.py teams.py predict.py ratings.py jobs.py model.py data.py accuracy.py bets.py settings.py pages.py
    workers/
      train.py             # subprocess: runs train_model, prints JSON lines, writes model_metrics.json
  web/
    templates/
      base.html  _macros.html
      slate.html game.html ratings.html accuracy.html bets.html model.html settings.html 404.html
    static/
      css/app.css          # tokens, Tabulator↔Tabler var mapping, small components
      js/main.js           # imports Alpine + persist, registers stores and page components, Alpine.start()
      js/api.js js/jobs.js js/format.js js/charts.js js/tables.js js/teams.js
      js/pages/{slate,game,ratings,accuracy,bets,model,settings}.js
      data/teams.json      # 30 rows: id, abbr, name, espn_name, primary, secondary
      data/glossary.json   # column/metric tooltips
      vendor/…             # committed, see §4
    vendor.lock.json       # package, version, file list, sha384 per file
  scripts/vendor.py        # downloads npm tarballs, verifies, copies listed files + LICENSE/NOTICE
  tests/test_api.py tests/test_jobs.py tests/test_security.py tests/test_market.py tests/test_web_static.py
  .env.example
```

`.gitignore` additions: `runs/`, `settings.json`, `model_metrics.json`. `.env`, `*.joblib`, `*.csv` and `*.db` are already ignored.

### 1.3 How it starts

```bash
pip install -r requirements.txt
python -m server                 # opens http://127.0.0.1:8000
python -m server --reload        # dev; still 1 worker
python -m server --port 8010 --no-browser
```

`server/__main__.py` does the following:

1. Sets `ROOT = Path(__file__).resolve().parent.parent` and calls `os.chdir(ROOT)`. This fixes the relative paths in `config.py`, `autobacktest.py` and `player_impact_updater.py`.
2. Calls `sys.path.insert(0, str(ROOT))`.
3. Loads `.env`.
4. Calls `mimetypes.add_type("text/javascript", ".mjs")`. Module scripts fail if `.mjs` is served as octet-stream, and CI runs Python 3.11 as well as 3.13.
5. Calls `uvicorn.run("server.app:create_app", factory=True, host, port, reload, workers=1)`.
6. Opens the browser after one second, unless `--no-browser` is passed.

`NBA_HOST` and `NBA_PORT` in `.env` override the defaults. Binding to anything other than loopback prints a warning, and the secrets endpoint then still accepts loopback clients only.

### 1.4 Reusing `engine/` and `data/`

The server imports these modules directly. `nba_app.py` code that is not Streamlit moves to `server/services`:

- `send_telegram` and `format_picks_telegram` move to `telegram.py`.
- The data behind `fig_*` moves to `metrics.py`, `predict.py` and `accuracy.py`.

Required small changes to engine and data. Each one ships with a test.

| # | Change | Why |
|---|---|---|
| E1 | Add `data/atomic.py`: `atomic_write_json(path, obj)` and `atomic_write_text`, which write a temp file in the same directory and then `os.replace`. Use it in `elo.save_elo`, `BlendCalibrator.save`, `autobacktest._save_log/_append_to_csv`, the bets CSV and `model_metrics.json`. `joblib.dump` writes to `*.tmp` and is then replaced. | Auto-backtest rewrites Elo once per game while scans read it. A cancelled training run must never leave half a model. |
| E2 | `data/injuries.py`: split into `fetch_injury_feed()` (raises on error), `parse_injury_report(feed, team_name)`, and `get_injury_report(team_name)` (old wrapper). | The server caches the feed for 10 minutes; today a 12-game scan makes 24 identical requests. The server also needs to tell "no injuries" apart from "feed down". |
| E3 | `data/schedule.py`: add `fetch_scoreboard(d) -> dict` (raises on error). Rebuild `get_schedule_by_date`, `get_past_results_espn` and the parsing in `fetch_odds_espn` on top of it, and also keep `ev["id"]`, `ev["date"]`, status, scores, and any `odds[].spread` / `odds[].overUnder`. | One scoreboard call serves schedule, results and market odds. The ESPN odds field names could not be checked from the sandbox, so check one real response and parse them as nullable. |
| E4 | `data/odds.py`: add `fetch_odds_api_all(api_key) -> list` (one call returns every NBA h2h line). The per-game lookup then filters the cached list. | The Odds API quota is 500 requests a month. |
| E5 | `data/cache.py`: add `get_with_age(key, allow_stale=False) -> (value, stored_at) or None`. | Offline fallback to expired entries. |
| E6 | `engine/models.py`: replace `n_jobs=-1` (lines 194 and 198) with `config.N_JOBS = max(1, os.cpu_count() - 1)`. Add `load_model(path)` with no refit. | Training took 594 s on 4 cores and oversubscribed the CPU. The new Reload button needs a load-only path. |
| E7 | Do the injury adjustment once in `predict_hybrid` and use the same adjusted rows for SHAP: return `hs` and `as_` or expose a helper. | Today SHAP explains the unadjusted row while `xgb_p` uses the adjusted one. |
| E8 | `player_impact_updater.py` is run by the job with `cwd=ROOT`. | Today it writes into whatever the working directory is. |

The model upgrade (v6) plugs in through one interface module, `engine/ratings.py`. The server imports it lazily, and every field it feeds is nullable until it exists:

```python
def available() -> dict            # {"kalman": bool, "adjusted": bool, "players": bool, "stacked": bool}
def team_table(system: str, as_of=None) -> list[dict]          # TEAM_ID, RATING, SD?, OFF?, DEF?  (KalmanRatings.ratings())
def team_history(team_id: int, system: str, days: int) -> list[dict]   # date, rating, sd
def player_table(min_games=20, active_days=90) -> list[dict]   # PlayerElo.leaderboard() rows
def availability(team_id, injuries: list) -> dict  # healthy, current, absence, players[{name, play_prob, margin_impact}]
def predict(home_id, away_id, date, injuries) -> dict  # {components[{key,label,prob|margin,weight}], intercept,
                                                       #  margin:{mean,sd}, total:{mean,sd}}
```

The state lives in `ratings_state.joblib` and is built by the `ratings` job. The return shapes come from `research/variants/kalman_engine.py` (`predict` returns μ, sd, sd_mean, total; `ratings()` returns RATING, SD, OFF, DEF), `player_elo_engine.py` (`leaderboard`) and `player_elo.py` (`live_prediction`, `STATUS_PLAY_PROB`).

### 1.5 Background jobs

**Job record** (in memory, mirrored to `runs/jobs.json`, which keeps the last 100):

```
{id, kind, state:"queued|running|done|error|cancelled|interrupted", progress: 0..1|null, stage, done, total,
 log:[str] (last 200, redacted), params, created_at, started_at, finished_at, result_ref|null, error|null}
```

**Kinds**

| kind | writer | params | runs in | progress source | result |
|---|---|---|---|---|---|
| `train` | yes | `force_rebuild`, `use_optuna` | subprocess `python -m server.workers.train`, `os.nice(10)` | JSON lines `{"type":"stage","i":k,"n":7,"msg":…}`; the bar is indeterminate within a stage. Show "typically about 10 min on 4 cores" (594 s measured). | `{rows, metrics:{val_acc,val_loss,n_train,n_val,best_iter,best_params}}`; writes `model_metrics.json`; server reloads the model |
| `autobacktest` | yes | — | thread | `status_cb` strings, regex `(\d+)/(\d+)` | `{new, correct, acc, calibrator:{weights,bias,n_updates}}` |
| `backtest` | no | `start`, `end` (end before today ET, span ≤ 60 days) | thread | days done / days | `{summary, rows, caveats}` |
| `scan` | no | `date`, `use_live_odds` (default true) | thread | games done / games; emits `item` per game | `{scan_id, date, games[]}` saved under `runs/scans/{date}/{scan_id}.json` |
| `player_impact` | yes | `season?` | subprocess `player_impact_updater.py`, `cwd=ROOT`, timeout 120 s | indeterminate | `{season, updated, count}` |
| `ratings` (v6) | yes | — | subprocess | stage lines | `{systems:[…], built_at}` |

**Rules**

- **One writer at a time.** A second writer gets `409 {error:"busy", job_id}`. The synchronous destructive endpoints (deletes, Elo reset, cache clear) take the same lock and also return 409.
- **Readers** (scan, backtest) share a `ThreadPoolExecutor(max_workers=2)`. They may run during a writer because file writes are atomic (E1).
- **Model required.** `412 {error:"no_model"}` is returned only where XGB is required: `autobacktest`, and `backtest` while v5 is the active model. Scan and predict fall back to `elo_only`.
- **Cancel.** `DELETE /api/jobs/{id}` sets `cancel_requested`. Thread loops check it between games or days. Subprocesses get `terminate()`, then `kill()` after 5 s, and their state becomes `cancelled`.
- **Startup.** Any job left `running` in `runs/jobs.json` is marked `interrupted`.
- **SSE.** `GET /api/jobs/{id}/events` uses `fastapi.sse.EventSourceResponse` (added in FastAPI 0.135.0; built-in 15 s keep-alive, `Cache-Control: no-cache`, `X-Accel-Buffering: no`). Events have incrementing `id`s so a reconnect sends `Last-Event-ID` and the stream resumes. Event types:
  - `progress` `{progress, stage, done, total}`
  - `log` `{msg}`
  - `item` `{index, data}`
  - `done` `{state, result_url, error}`
- **Fallback.** After 2 EventSource errors the client polls `GET /api/jobs/{id}` every 800 ms.
- **No GZip middleware** anywhere: the routes are local and SSE must not be buffered.

```python
# routers/jobs.py (sketch)
from fastapi.sse import EventSourceResponse, ServerSentEvent

@router.get("/api/jobs/{job_id}/events", response_class=EventSourceResponse)
async def job_events(job_id: str, last_event_id: int | None = Header(None)):
    job = jobs.get_or_404(job_id)
    cursor = last_event_id or 0
    while True:
        for ev in job.events_after(cursor):          # list of (id, type, data)
            cursor = ev.id
            yield ServerSentEvent(event=ev.type, id=str(ev.id), data=ev.data)
        if job.terminal and not job.events_after(cursor):
            return
        await asyncio.sleep(0.4)
```

### 1.6 Secrets and security

- **`.env` keys:** `ODDS_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, plus optional `NBA_HOST` and `NBA_PORT`. Commit a `.env.example`.
- **Writing secrets.** `PUT /api/settings/secrets` writes with `dotenv.set_key`, then `chmod 0600` on POSIX. An empty string removes a key. Process env vars take precedence, and the API reports `source:"env"|"file"`.
- **Never echoed.** GET responses return only `configured: bool`, `source`, and `chat_id_masked` (`•••4821`).
- **`redact(s)`.** Replaces every configured secret value with `•••`. It is applied to job log lines, error `detail`, Telegram and odds error strings, and a logging filter on the root logger. This is needed because `requests` exception text includes the full URL: the Odds API key is a query parameter (odds.py:79) and the Telegram token is in the URL path. Set the `urllib3` logger to WARNING.
- **Host/Origin guard.** Implement it as **pure ASGI middleware**, not `@app.middleware("http")`, so SSE streams pass through. It rejects:
  - any request whose `Host` is not `127.0.0.1:PORT` or `localhost:PORT` (`403 forbidden_host`, a DNS-rebinding defence);
  - any non-GET request whose `Origin` is present and not same-origin (`403 forbidden_origin`, a CSRF defence for a localhost API).
- **Loopback only.** `PUT /api/settings/secrets` and the destructive data endpoints also require `request.client.host` in `{127.0.0.1, ::1}`.
- **No CORS middleware.** Same-origin only.
- **Pickle cache.** `nba_cache.db` stores pickles, so no endpoint ever accepts cache content from clients.

### 1.7 Offline behaviour and caching

| Source | TTL | Offline fallback |
|---|---|---|
| ESPN scoreboard (schedule, market odds, finals) | 5 min for today and future dates; 24 h for past dates | Any age, with `stale:true, as_of` and an "Offline: cached at 14:05" banner |
| ESPN injury feed | 10 min | Up to 24 h, marked stale; older than that gives `availability.status:"unavailable"` and predictions use no injuries plus a caveat |
| The Odds API (all games in one call) | 10 min | Fall back to ESPN odds |
| nba_api team logs | existing 6 h | `get_with_age(..., allow_stale=True)` |
| nba_api standings | 1 h | stale allowed |
| Predictions | not cached (about 30 ms warm; MC 0.011 s) | — |

- **No external requests from the pages.** Assets are vendored. Tabler CSS has no remote `@import`; I checked, so it uses the system font stack. The team list comes from `nba_api.stats.static.teams`, which works offline.
- **FastAPI's `/docs` page** loads Swagger UI from a CDN. Move it to `/api/docs` and treat it as a developer page that needs internet.

### 1.8 Conventions

- JSON keys are `snake_case`. Probabilities are floats from 0 to 1, rounded to 4 dp. Points are rounded to 1 dp. Unavailable values are `null`, never `NaN`; `jsonutil` converts numpy types and non-finite floats.
- **Margin sign:** `margin` is home minus away. `home_spread = -margin_mean`. The UI renders the favourite form, for example "LAL −4.5".
- Dates are `YYYY-MM-DD` in ET. Timestamps are ISO-8601 UTC (`as_of`, `fetched_at`).
- **Error envelope:** `{error: code, detail: str, hint?: str, job_id?: str}`.

| Code | Status |
|---|---|
| `validation` | 422 (FastAPI's details go in `detail`) |
| `team_not_found` | 404 |
| `not_found` | 404 |
| `no_model` | 412 |
| `busy` | 409 |
| `confirm_required` | 400 |
| `upstream_unavailable` | 503, with `source` |
| `forbidden_host` / `forbidden_origin` / `loopback_only` | 403 |

- **Business thresholds live only in the backend** and are returned in `/api/status`:

| Threshold | Default |
|---|---|
| `pick` | 0.55 |
| `safe` | 0.65 |
| `vip` | 0.65 and no conflict |
| `value_edge` | 0.04 |
| Kelly fraction | 0.25 |
| `kelly_cap_pct` | 5.0 |

- Confidence tiers: `low` below 0.55, `medium` from 0.55 to below 0.65, `high` at 0.65 or above.

---

## 2. JSON API (prefix `/api`)

### 2.1 Meta and status

| Method | Path | Request | Response |
|---|---|---|---|
| GET | `/api/health` | — | `{ok:true, version}` |
| GET | `/api/status` | — | See below |

`/api/status` returns:

```
{version, server_time_utc, today_et,
 model:{exists, version:"v5", trained_at, val_acc, val_loss, n_train, age_days},
 dataset:{exists, rows, mtime}, elo:{exists, n_teams, mtime},
 calibrator:{weights:{xgb, mc, elo}, bias, n_updates},
 autobacktest:{total}, player_impact:{season, updated, count, fallback},
 cache:{total, live, expired},
 capabilities:{predict_mode:"blend_v5"|"elo_only", shap, optuna,
               ratings:{elo:true, kalman, adjusted, players}, stacked_ensemble},
 secrets:{odds_api, telegram},                          // booleans only
 thresholds:{pick, safe, vip, value_edge}, kelly:{fraction, cap_pct},
 seasons:{train:[…], current}, active_jobs:[{id, kind, progress, stage}]}
```

`/api/status` replaces sidebar items S1, S5, S7, S10, S12 and the packages list.

### 2.2 Teams, schedule, odds, availability

| Method | Path | Request | Response |
|---|---|---|---|
| GET | `/api/teams` | — | `[{id, abbr, name, nickname, city, espn_name, colors:{primary, secondary}}]` |
| GET | `/api/teams/{team_id}/form` | `?date=` | The `get_team_advanced_stats` dict with keys lower-snake-cased (`elo`, `net5`, `net15`, `efg10`, `days_rest`, `is_b2b`, `pts`, `pts_allowed`, …), plus `as_of` |
| GET | `/api/teams/{team_id}/availability` | `?date=` | `{team_id, status:"ok"\|"stale"\|"unavailable", fetched_at, strength, healthy, current, deficit, players:[{name, status, detail, impact, status_weight, play_prob\|null, margin_impact\|null, matched}], unmatched}` |
| GET | `/api/schedule` | `?date=` (default today ET) | `{date, as_of, stale, games:[{event_id, tipoff_utc, status:"pre"\|"in"\|"post", period, home:{id, abbr, score\|null}, away:{…}, market:{home_ml, away_ml, spread\|null, total\|null, source}\|null}]}` |
| GET | `/api/odds` | `?home_id=&away_id=&date=` | `{status:"ok"\|"no_odds_yet"\|"not_found", source, fetched_at, home_ml, away_ml, spread, total}`. Odds API first if a key is set, else ESPN. |

In `/api/teams/{team_id}/availability`, `healthy`, `current` and `deficit` are in strength-factor units for v5 and in points for v6.

### 2.3 Prediction

**`POST /api/predict`**. Synchronous, 200 ms or less when warm.

Request:

```
{home_id, away_id, date?: "YYYY-MM-DD",                         // ET; past date ⇒ stats as of that date
 odds?: {home_ml:int, away_ml:int} | null,                       // manual; null ⇒ cached market lookup
 injuries?: "live" | "none" | {home:[{name, status}], away:[{name, status}]},   // default "live"
 include?: ["hist", "shap", "poisson"]}                          // default ["hist"]
```

Response:

```
{as_of, mode:"blend_v5"|"elo_only"|"stacked_v6",
 model:{version, trained_at, age_days} | null,
 game:{date, event_id|null, tipoff_utc|null, home:{id, abbr, name}, away:{…}},
 win_prob:{home, away},
 pick:{side:"home"|"away", team_id, prob, tier:"low"|"medium"|"high"},
 flags:{conflict, safe, show_pick, vip, value},
 engines:{xgb|null, mc, elo, poisson|null},                // elo = elo_expected(home_elo + ELO_HOME_ADV, away_elo)
 ensemble:{method, n_updates, order:["elo","mc","xgb"],
           base:{logit, prob},                             // neutral game x0 = (0.5, 0.5, 0)
           components:[{key, label, prob, coef, contribution_logit, step_pp, share}],
           bias, final:{logit, prob}},
 margin:{mean, sd, home_spread, p_within_3,
         intervals:{p50:[lo, hi], p80:[…], p95:[…]},
         hist?:{edges:[61], counts:[60]}},
 total:{mean, median, sd, intervals:{p50, p80, p95}, hist?:{edges, counts}},
 scores:{home:{mean, p25, p75, hist?}, away:{…}},
 market:{status:"ok"|"no_odds_yet"|"not_found"|"manual", source, fetched_at,
         home_ml, away_ml, decimal:{home, away}, implied:{home, away}, overround,
         fair:{home, away}, spread|null, total|null,
         edge:{home, away}, value_side:"home"|"away"|null,
         cover_prob|null, over_prob|null} | null,
 fair_line:{home_ml, away_ml},
 kelly:{fraction, cap_pct, price_source:"market"|"manual",
        home:{decimal, full_pct, frac_pct}, away:{…}} | null,  // null when there is no price
 teams:{home:{…form keys}, away:{…}},                      // same keys as /teams/{id}/form; injury-adjusted values in teams.*.adjusted
 ratings:{home:{elo, elo_rank, kalman:{rating, sd, off, def}|null, adjusted:{net, off, def}|null,
                player_sum:{strength, impact, box, var}|null}, away:{…}},
 availability:{home:{…same as /availability}, away:{…}},
 radar:{axes:[6 labels], home:[6], away:[6]},
 poisson?:{lambda_h, lambda_a, mean_h, mean_a, median_total, ou_80, ou_20,
           quarters:[{q, home, away, h_cum, a_cum}]},
 shap?:{base_value, features:[{name, value, shap}]} | null,  // null if shap is not installed; uses adjusted rows (E7)
 feature_row:{ELO_DIFF:…, …17 FEATURE_COLS},
 caveats:["elo_only_no_model"|"injuries_unavailable"|"injuries_stale"|"odds_stale"|"past_date_current_elo"]}
```

Computation rules:

- **Ensemble (`blend_v5`).**
  - Inputs and weights: `x = [xgb, mc, elo_diff/400]`, `w` = calibrator weights, `x0 = [0.5, 0.5, 0]`.
  - `base.logit = w·x0 + bias` and `contribution_i = w_i(x_i − x0_i)`.
  - `step_pp` is the sequential probability step in the fixed order Elo, then MC, then XGB. Order matters (from the parallax-nba ladder).
  - `share_i = |c_i| / Σ|c|`.
  - `elo_only` has a single Elo component with weight 1.
  - `stacked_v6` fills the same schema from `engine.ratings.predict`.
- **Margin and total.** Percentiles of the MC `_margins` and `_totals` arrays (p25/p75, p10/p90, p2.5/p97.5) and `np.histogram(…, bins=60)`. Raw arrays are never serialised. `p_within_3` replaces the old "OT %" label.
- **Market.**
  - De-vig uses the proportional method: `fair_i = implied_i / Σ implied`.
  - `edge_i = model_i − fair_i`.
  - `value_side` is set when an edge is at least `value_edge`.
  - `cover_prob = P(margin > −spread)` and `over_prob = P(total > line)`, both from the MC samples.
- **Fair moneyline from probability p.** If p ≥ 0.5 it is `−100p/(1−p)`, otherwise `100(1−p)/p`.
- **Kelly** uses `kelly_criterion` at the market or manual decimal price, capped at `cap_pct`. It is never computed at a fixed −110.
- **Poisson** returns λ-based means as well as the sampled quarters. The single random draw is no longer the headline number.

### 2.4 Ratings

**`GET /api/ratings/teams?system=elo|kalman|adjusted&injuries=live|none&date=`**

```
{system, as_of, available:{elo:true, kalman, adjusted},
 teams:[{rank, team:{id, abbr, name},
         power,                                 // value of the selected system
         elo, elo_rank, elo_delta_7d|null,
         kalman:{rating, sd, off, def}|null,
         adjusted:{net, off, def}|null,
         healthy, current, deficit, missing_count,
         record, l10, streak,                   // from standings
         spark:[{date, value}]}]}               // last 30 snapshot points
```

- `elo_delta_7d` and `spark` come from `runs/ratings_snapshots/YYYY-MM-DD.json`. The server writes a snapshot on the first ratings call of each ET day and after every auto-backtest. These fields stay null until history builds up.
- Standings use `leaguestandingsv3`. Check the column names (`WINS`, `LOSSES`, `L10`, `strCurrentStreak`) once locally.

**`GET /api/ratings/teams/{team_id}/history?system=elo|kalman&days=120`**

```
{system, points:[{date, rating, sd|null}]}
```

**`GET /api/ratings/players?team_id=&min_games=20&limit=200&q=`**

```
{available, as_of, players:[{rank, player_id, name, team:{id, abbr}, rating, impact, box, sd, exp_min, games,
                             injury:{status, detail}|null}]}
```

When the engine is absent, these endpoints return `available:false, reason:"engine_not_built"`.

**`GET /api/model/ensemble`** returns the global component weights:

```
{method, components:[{key, label, coef, mean_abs_contribution|null, oos_log_loss|null}], bias, n_updates, updated_at}
```

### 2.5 Jobs

| Method | Path | Request | Response |
|---|---|---|---|
| POST | `/api/jobs/{kind}` | Params per §1.5 | `202 {job_id}`, or `409 busy` / `412 no_model` |
| GET | `/api/jobs` | `?active=true&kind=&limit=20` | `[job record]` |
| GET | `/api/jobs/{id}` | — | Job record plus `result` inline when small (≤ 2 MB) |
| GET | `/api/jobs/{id}/events` | header `Last-Event-ID` | SSE stream (§1.5) |
| DELETE | `/api/jobs/{id}` | — | `{state:"cancelling"}` |

Job results:

- **`backtest`:** `{summary:{games, correct, acc, wrong, vip_games, vip_correct, vip_acc, brier, log_loss}, rows:[{date, home:{id, abbr}, away, h_score, a_score, pred:"home"|"away", prob, conflict, correct, reason}], caveats:["current_elo","no_injuries"]}`
- **`scan`:** `{scan_id, date, created_at, games:[card]}`. Each `card` is a subset of the predict response: `event_id, tipoff_utc, status, home, away, win_prob, pick, flags, engines{xgb, mc, elo}, margin{mean, home_spread, p50}, total{median, p50}, scores{home:[p25, p75], away:[p25, p75]}, market{home_ml, away_ml, spread, total, fair, edge, value_side, source}|null, fair_line, kelly{side, frac_pct}|null, availability{home:{strength, out_count}, away}, result{home_score, away_score, correct}|null, error|null`.

### 2.6 Scans, model and data

| Method | Path | Notes |
|---|---|---|
| GET | `/api/scans?date=` | Latest scan for the date, or 404 |
| GET | `/api/scans/{scan_id}` | Saved scan |
| GET | `/api/model` | `{exists, trained_at, version, metrics, feature_importance:[{feature, importance}], calibration:{bins:[{lo, hi, n, mean_pred, actual}]}}` from `model_metrics.json` |
| POST | `/api/model/reload` | Load the joblib file with no refit; returns `{trained_at}` |
| DELETE | `/api/model?confirm=true` | Removes `MODEL_FILE`. Without `confirm` it returns `400 confirm_required`. Takes the writer lock. |
| DELETE | `/api/dataset?confirm=true` | Removes `DATASET_FILE` |
| DELETE | `/api/elo?confirm=true` | Removes `ELO_FILE` (the old "Reset Elo") |
| GET | `/api/cache` | `cache.stats()` |
| DELETE | `/api/cache` | `cache.clear_all()` |
| GET | `/api/calibrator` | `BlendCalibrator.summary()` with keys `xgb`, `mc`, `elo` |
| GET | `/api/player-impact?top=20` | `{season, updated, count, fallback, top:[{name, impact}]}` |

### 2.7 Accuracy

**`GET /api/accuracy?source=autobacktest`** (Phase 3 adds `source=issued`):

```
{source, n, summary:{acc, brier, log_loss, base_home_acc, base_coin_log_loss:0.693,
                     vip:{n, acc}, last_graded_date},
 calibration:{bins:[{lo, hi, n, mean_pred, actual, gap, ci95:[lo, hi]}]},   // Wilson interval
 rolling:[{date, acc20}],
 tiers:[{label:"50–55%", n, acc}],
 caveats:["replay_current_model"]}
```

This is the data that `fig_history_chart` and the reliability curve drew. It comes from `autobacktest_results.csv` columns `HOME_WIN_PROB`, `ACTUAL`, `CORRECT`, `CONFIDENCE`.

**`GET /api/accuracy/rows?limit=50&offset=0`**

```
{rows:[{game_id, date, home, away, predicted, actual, correct, confidence, xgb_p, mc_p, elo_diff}], total}
```

### 2.8 Bets (ROI)

| Method | Path | Request | Response |
|---|---|---|---|
| GET | `/api/history` | `?period=all\|season\|month\|week` | `{rows:[bet], summary:{bets, settled, wins, losses, pushes, win_rate, breakeven:0.524, pnl, roi}, series:{cum_pnl:[{date, value}], rolling_win20:[{i, value}]}}` |
| POST | `/api/history` | `{date?, matchup, pick, confidence (50–100), odds (american, −1000…1000, not in −99…99), stake (0–10), result:"win"\|"loss"\|"push"\|"pending", prediction_ref?:{date, home_id, away_id, prob}}` | `bet` |
| PATCH | `/api/history/{id}` | `{result}` (settle a pending bet) | `bet` |
| DELETE | `/api/history/{id}` | — | `{ok}` |
| GET | `/api/history.csv` | — | CSV download |

- `bet = {id, date, matchup, pick, confidence, odds, stake, result, pnl|null, prediction_ref|null, created_at}`.
- PnL is `stake*(dec−1)` on a win, `−stake` on a loss, 0 on a push, and null while pending.
- The file is still `prediction_history.csv`. On first load, rows without an `id` get an 8-hex id, and the file is rewritten atomically. A missing file is treated as empty.

### 2.9 Settings and Telegram

| Method | Path | Request | Response |
|---|---|---|---|
| GET | `/api/settings` | — | `{secrets:{odds_api:{configured, source}, telegram:{configured, source, chat_id_masked}}, thresholds, kelly, server:{host, port, root, python, packages:{fastapi, xgboost, sklearn, nba_api, shap, optuna: "x.y"\|null}}}` |
| PUT | `/api/settings` | `{thresholds?:{pick, safe, vip, value_edge}, kelly?:{fraction, cap_pct}}` | Same as GET. Validated (thresholds 0.5–0.9, value_edge 0–0.2, fraction 0–1, cap 0–25) and written to `settings.json`. |
| PUT | `/api/settings/secrets` | `{odds_api_key?, telegram_token?, telegram_chat_id?}`; an empty string clears a value | Same as GET. Loopback only. |
| POST | `/api/telegram/test` | `{}` | `{ok, detail}` (redacted) |
| POST | `/api/telegram/send-picks` | `{scan_id, only_vip:true}` | `{ok, sent, detail}`. The message is built on the server from the saved scan with `parse_mode:"HTML"` and escaped text. This fixes the old button that never sent anything. |

### 2.10 HTML routes

`GET /` (Slate), `/game`, `/ratings`, `/accuracy`, `/bets`, `/model` and `/settings` each render a template with `{request, page, version}` only. Static files are served at `/static`:

- `/static/vendor/*` gets `Cache-Control: public, max-age=31536000, immutable`, because the paths contain the version.
- App JS and CSS get `no-cache`.

---

## 3. Screens

### 3.0 Global shell (`base.html`)

- **Navbar** (Tabler horizontal `navbar`):
  - Brand: `ti-ball-basketball` and "NBA Quant".
  - Links: Slate · Game · Ratings · Accuracy · Bets · Model · Settings.
  - Model status dot: green when trained, amber when older than 7 days, grey when there is no model and the app is in Elo-only mode.
  - **Jobs tray**: a dropdown with a count badge and one Tabler `.progress` per active job, fed by `Alpine.store('jobs')`. On page load it reads `/api/status.active_jobs` and reattaches EventSources.
  - Theme toggle: `data-bs-theme`, remembered by `tabler-theme.min.js` under the `tabler-theme` key.
  - Odds format toggle US/DEC, kept with `@alpinejs/persist`.
- **Banner region:**
  - No-model banner: "No trained model: showing Elo + Monte Carlo. Train (≈10 min) →".
  - Offline/stale banner: "Offline: showing data cached at 14:05 ET".
- **Toasts** (Alpine store, Tabler `.toast.show` markup): job finished or failed, with a link to the result.
- **Footer:** version, data sources with fetch times, "Not affiliated with the NBA. Team colours only, no logos.", and a responsible-gambling note.
- **Mobile:** the navbar collapses into an offcanvas menu. The jobs icon and status dot stay visible. Toasts dock at the bottom.

**Shared components** (Jinja macros in `_macros.html` plus Alpine helpers):

| Component | What it does | Source |
|---|---|---|
| `team_badge` | Coloured circle with the abbreviation; white or secondary text chosen by a WCAG contrast check | C19 |
| `prob_split_bar` | Two-colour stacked `.progress`. Team colours come from a contrasting-pair picker ported from DeepShot `get_best_color_pair` (MIT, credited in a comment). `aria-label` reads "LAL 61%, GSW 39%". | — |
| `stat_tile` | Label, value and sub-line | BelNaruto 4 tiles |
| `confidence_badge` | Low / Medium / High | — |
| `edge_badge` | Green "+4.3% edge" | — |
| `rank_chip` | "Ranked 14th of 30" tooltip, colour lerped green to red | sRating idea, written from scratch |
| `freshness` | `.status-dot` plus "3 min ago" | linedrift |
| `job_progress` | Bar, stage, elapsed time, Cancel | — |
| `empty_state` | Tabler `.empty` | — |
| `error_state` | Alert with Retry | — |
| `skeleton(n)` | `.placeholder-glow` | — |
| `confirm_modal` | Confirmation dialog | — |
| `glossary_tip` | Header tooltip from `glossary.json` | C20 |

Glossary entries: Elo, Kalman rating, ±SD, Adjusted net, Healthy/Current, Fair line, Overround, Edge, Kelly, Brier, Log loss, Calibration gap, Within 3.

**Rules for every page:**

- Never use colour alone: signs (+/−) and icons go with green and red.
- Numbers use `font-variant-numeric: tabular-nums`.
- ECharts has `aria.enabled: true`.

### 3.1 Slate (`/`): replaces Radar Scanner

**Top bar**
- Date navigator: ‹ date ›, a "Today (ET)" button, and ← → keys (NBA_AI).
- Freshness line: "Scoreboard 3 min · Injuries 7 min" (linedrift).
- Primary button "Run predictions" starts the `scan` job.

**KPI strip** (BelNaruto): Games · VIP picks · Value spots · for past or finished dates, Picks correct x/y.

**Filter and sort**
- Chips: All / VIP / Value / Conflicts.
- Sort: Tip-off (default) / Confidence / Edge / Interest. Interest is a 0–100 game-quality score: the harmonic mean of the two teams' Elo percentiles, inspired by FiveThirtyEight's game quality (C17, per search).

**Game cards** (`col-12 col-md-6 col-xl-4`), built from BelNaruto's card anatomy (C1, C2):

1. Status line: tip-off in local time and ET, Scheduled/Live/Final, confidence badge, "Engines disagree" badge, and a VIP `.ribbon`.
2. Two team rows: badge, name, record, injury icon with an "out" count (ZenGM pattern), and a large win %.
3. Split probability bar.
4. Four tiles:
   - **Projected score**, from the MC p25–p75 ranges, for example "112–118 / 106–113".
   - **Spread**: model "LAL −4.1" against market "−4.5".
   - **Total**: model against market.
   - **Fair ML** against market ML, in US or decimal format (C3).
5. Market line: "Market (fair): LAL 61% · −4.5 · 226.5 · DraftKings via ESPN". It adds `edge_badge` "Value: GSW +4.3%" when an edge is at least `value_edge`. With no market, the empty variant reads "No market yet: model only".
6. Finals: the actual score, and a Correct/Incorrect chip that recolours the prediction (NBA_AI, parallax-nba; C14).
7. Clicking a card opens `/game?home_id=&away_id=&date=`.

**Progressive fill.** The page renders the cards from `/api/schedule` straight away, with market odds. If `/api/scans?date=` exists, its predictions are merged in. Otherwise each card shows "–" placeholders, and SSE `item` events fill the cards one game at a time while the scan runs.

**Footer actions:** "Send VIP picks to Telegram" is disabled, with a tooltip, unless Telegram is configured and the scan has VIP picks. There is also a CSV export.

**States**

| State | What the user sees |
|---|---|
| Loading | 6 skeleton cards |
| Empty | "No NBA games on Tue 28 Oct (ET)" with the ‹ › buttons; in the offseason, "Season starts …" |
| Error | Cached scoreboard: stale banner. No cache: an error card with Retry. Per-game error (team not found): the card shows the error chip and the other cards still render. |
| No model | Global banner; cards show Elo + MC and mark the tier "uncalibrated" |

**Mobile:** one column; KPI strip in a 2×2 grid; chips scroll horizontally; a sticky bottom bar holds "Run predictions".

### 3.2 Game (`/game`): replaces Single Game

**Input bar**
- Home and Away `<select>` from `/api/teams` (replaces free-text substring matching), plus a swap button and a date field.
- Odds inputs for home and away ML, with an "Auto-fetch" button and a source caption. Statuses: ok / no odds yet / not found.
- Injuries segmented control: Live / None / Custom. Custom opens an offcanvas with per-player status selects, seeded from the live report.
- Primary button "Analyze". The page auto-runs when it is opened with query parameters. The URL updates, so a link reproduces the view.

**Hero** (tennis-crystal-ball two-sided header)
- Away @ Home, a large win % on each side, the split bar, and the pick line "Pick: Lakers 61% · High · engines agree".
- Or a warning: "XGBoost and Monte Carlo disagree: no pick".
- Badges: SAFE / VIP / VALUE.

**Tiles:** Projected score (mean and 50% range) · Spread (model against market, plus `cover_prob`) · Total (plus `over_prob`) · Fair ML · Kelly (fractional %, capped, with a responsible-use tooltip; hidden with "add odds to size a stake" when there is no price).

**Sections** (cards; accordions on mobile):

1. **Why this number: ensemble ladder** (C5; tennis-crystal-ball "Show details", parallax-nba explanation ladder).
   - A horizontal ECharts waterfall: base 50% → +Elo → +Monte Carlo → +XGBoost → final.
   - A table with columns Component · Its own win% · Weight (coefficient) · Contribution (pp) · Share.
   - In `stacked_v6` the same widget shows Ratings → Home court → Rest/travel → Injuries → Calibration.
2. **Margin and total** (C6).
   - Tabs: Margin / Total / Scores.
   - The margin histogram has markLines at 0, the model mean and the market spread, plus a shaded p80 band.
   - Text under it: "P(LAL covers −4.5) 48% · P(within 3) 19% · 80% range −11 to +19".
3. **Team comparison.**
   - A mirrored table (tennis-crystal-ball) with rows Elo (with rank chip), Kalman ±sd (v6), Adjusted net (v6), Net 5/15, eFG, TOV, ORB, FTR, Pace, Win% L10, Streak, Rest, B2B.
   - Better values are bold. A radar chart sits beside it on desktop and below it on mobile.
4. **Availability** (C8, ZenGM Current vs Healthy).
   - For each team: Healthy against Current strength, with the deficit in red (for example "−3.1 pts" in v6, "94%" in v5).
   - The player list shows a status badge (Out red / Doubtful orange / Questionable yellow / DTD grey), comment, impact, and play_prob and margin_impact in v6.
   - Reliability line (C15, parallax-nba): "Report fetched 12:03 ET · 1 player unmatched".
5. **Model internals** (collapsed): SHAP waterfall (adjusted rows), the 17-feature row table, and Poisson quarter bars labelled "legacy engine".

**Actions:** "Log this bet" opens the bets form prefilled, with `prediction_ref`. Also "Copy link".

**States**

| State | What the user sees |
|---|---|
| Before run | Empty state "Pick two teams" |
| Same team on both sides | Inline validation |
| Loading | Skeleton hero and the text "Fetching team logs (first run can take ~10 s)" |
| `team_not_found` | Inline error |
| Upstream 503 with cache | Stale banner |
| Upstream 503 without cache | Error card with Retry |
| No model | Elo-only banner; the ensemble shows one component |

**Mobile:** the hero stacks vertically; tiles in a 2-column grid; sections become accordions with only "Why this number" open; charts 240 px high.

### 3.3 Ratings (`/ratings`): new. Tabs: Teams | Players

**Teams tab** (Tabulator)

- Columns: Rank · Team · Power · Elo · ΔElo 7d (± with colour and sign) · Kalman ±SD · Off · Def · Adj. net · Healthy · Current · Record · L10 · Streak · Trend.
  - Kalman ±SD has an inline CSS band bar for ±1 SD.
  - Current turns red when it is below Healthy (ZenGM).
  - Trend is an inline SVG sparkline written as a small formatter, not an ECharts instance per row.
- Controls: system select Elo / Kalman / Adjusted (sets Power and the sort), a "With injuries / Healthy" toggle (ZenGM), a glossary tooltip on every header (C20), and CSV download.
- Columns for systems that are not built yet are greyed, with the tooltip "Available after the ratings engine is built (Model → Build ratings)".
- **Row click** opens an offcanvas team drawer:
  - Rating history with a ±1σ band (C7, from the sRating BandLine idea).
  - A rank strip per metric, "better →" (C10).
  - Roster with player ratings and injury icons.
  - Next 3 games.

Patterns used: C8, C9, C10, C7, C20.

**Players tab:** Rank · Player · Team · Rating (pts/48 vs average) · Impact · Box · ±SD · Exp. min · Games · Status icon. Filters: team, minimum games (default 20), name search. CSV download.

**States**

| State | What the user sees |
|---|---|
| Loading | Skeleton rows |
| Elo file missing | Empty state "No Elo ratings yet: train the model or run auto-backtest" |
| Players unavailable | Empty state with a "Build ratings" button (`ratings` job) or "coming with model v6" |
| Standings fetch failed | Record, L10 and Streak show "—" with a stale dot |

**Mobile:** Tabulator `responsiveLayout:"collapse"` with Rank and Team frozen; the drawer opens full-screen.

### 3.4 Accuracy (`/accuracy`): replaces the Backtest tab and the sidebar calibrator/auto-backtest

1. **Scoreboard tiles** (C12, from BelNaruto, NBA_AI and NBA_Betting): Accuracy vs "always home" · Brier · Log loss vs coin 0.693 · VIP accuracy · n games. A source toggle switches between "Replay (auto-backtest)" and "Issued forecasts" (Phase 3; parallax-nba keeps the two separate).
2. **Calibration** (C11, idea from sRating `Calibration.tsx`): an ECharts scatter with dot area proportional to n, plus the diagonal. Table: band · n · mean forecast · actual · gap · 95% CI.
3. **Online-learning history:** rolling 20-game accuracy line, and accuracy by confidence tier bars. This revives the unused `fig_history_chart`.
4. **Blend calibrator card:** the weights as a 100% stacked bar from `/api/model/ensemble`, the bias, n_updates, and a "Grade new games (auto-backtest)" button that starts a job.
5. **Range backtest:**
   - A date-range picker; validation blocks spans over 60 days or ending today or later.
   - Run starts a job with day progress.
   - Results: summary tiles, then a Tabulator table with columns Date · Matchup · Score · Pick (prob) · Engines · Result chip · Reason.
   - A **caveat banner** stays until the Elo-as-of-date fix ships: "Optimistic: uses today's Elo and no injuries".

**States**

| State | What the user sees |
|---|---|
| No graded games | "No graded games yet" with a "Run auto-backtest" button |
| No model | `412` turns into "Train a model first" with a link |
| Running job | Inline `job_progress` that survives navigation (jobs tray) |

**Mobile:** tiles in a 2×2 grid; charts stacked; table collapses.

### 3.5 Bets (`/bets`): replaces ROI Dashboard

- **Header** (NBA_Betting): Bets · Win rate (against 52.4% breakeven) · P&L in units · ROI. Period chips: All / Season / Month / Week.
- **Charts** (C13):
  - Cumulative P&L area, green above 0 and red below, using ECharts `visualMap` pieces at 0, with dataZoom.
  - Rolling 20-bet win rate with a markLine at 52.4%.
- **Log bet form:** a collapsible card on desktop and a floating button that opens a bottom offcanvas on mobile.
  - Fields: matchup (select from today's slate, or free text), pick, confidence, odds in the user's format, stake %, result Win / Loss / Push / Pending.
  - Submits to `POST /api/history`.
- **Table:** Date · Matchup · Pick · Conf · Odds · Stake · Result chip · P&L · link to the prediction. Pending rows have Settle buttons that call PATCH. Delete opens a confirm modal. CSV download.
- **States:** empty shows "No bets logged yet" with a "Log your first bet" button; validation errors appear inline on fields; the responsible-use note is always visible.

### 3.6 Model (`/model`): replaces the sidebar

- **Status cards:**
  - Model: trained_at, age, val_acc, val_loss, n_train.
  - Dataset: rows, mtime.
  - Elo: teams, mtime.
  - Player impact: season, updated, count, and a fallback warning.
  - Cache: live/expired, with a Clear button.
- **Actions:**
  - **Train**, with toggles "Force full rebuild" and "Optuna HPO" (disabled with a tooltip if optuna is missing). It starts a job card with a stage stepper (Tabler `.steps`: download → paired → Elo → search → fit → calibrate → saved), elapsed time, an ETA hint, a log tail and Cancel.
  - **Reload model** (no refit).
  - **Update player impact.**
  - **Build ratings** (v6).
- **Charts** from `/api/model`: feature importance bars and the validation reliability curve. Both now survive a refresh.
- **Ensemble weights:** global components with their coefficients and out-of-sample log loss when v6 provides it.
- **Job history** table: kind · state · started · duration · result link.
- **Empty state when there is no model:** "No trained model yet. Training downloads 3 seasons and takes about 10 minutes on a 4-core machine. Until then predictions use Elo + Monte Carlo."
- **Mobile:** cards in one column; the job card is sticky at the top while a job runs.

### 3.7 Settings (`/settings`)

- **API keys:**
  - Odds API: shows "Configured (from .env)" or "Not set", a replace input, and Clear.
  - Telegram: token and chat ID (chat ID masked), plus "Send test message".
  - Help text explains that the keys are stored in `.env` on this computer and are never sent back to the browser.
- **Thresholds and stake sizing:** pick / safe / vip / value edge, Kelly fraction and cap, with the note "applies to new predictions". Saves to `PUT /api/settings`.
- **Display:** odds format and theme. These are client-side only.
- **Danger zone** (red card): Clear model · Clear dataset · Reset Elo · Clear cache. Each opens a confirm modal that names the file and the consequence. The buttons are disabled while a writer job runs.
- **Environment:** real package versions, Python version, project root and bind address.
- **Mobile:** one column; the danger zone is last.

---

## 4. Libraries

### 4.1 Frontend: vendored and committed under `web/static/vendor/<pkg>@<ver>/`

| Package @ version | License | Files to vendor (sha384) | jsDelivr fallback URL |
|---|---|---|---|
| `@tabler/core@1.6.1` (2026-09-28) | MIT | `dist/css/tabler.min.css` (`tT2UAGE9hxG/p5d0iGIvZ/s8El3nWWG3tfG02i8iOY5Pbf8cZZRVmcrrVs+JG5Vw`)<br>`dist/js/tabler.min.js` (`1yCHfhyU8+V33urXkAlLLpNYd9jdzXkxdafCg0+auIIEtYFnOuM/DwJQgk5sXY98`)<br>`dist/js/tabler-theme.min.js` (`gPM8vc3koNjMZtzPmhqasm5L6/Zak4DcWmRrPt4GZJBKSEKQfwyFIehZzq2TH1AZ`) | `https://cdn.jsdelivr.net/npm/@tabler/core@1.6.1/dist/css/tabler.min.css` (same prefix for the js files) |
| `@tabler/icons-webfont@3.49.0` (2026-10-05) | MIT | `dist/tabler-icons.min.css` (`IcAOG/1tqv9Qnam+j3mZYPAFcNQMnQpMcafc8Nsvd44AUeTCzhUzdQiJkrROmqW6`)<br>`dist/fonts/tabler-icons.woff2` (`yBEWz226ooaXrXHlWwyPfgx8Ek8aHmIzCAU71izjTxVK0GSmFyrnX08fW8S7yAxm`). Keep the `./fonts/` relative path. | `https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@3.49.0/dist/tabler-icons.min.css` |
| `alpinejs@3.17.4` (2026-09-21) | MIT | `dist/module.esm.min.js` (`1Ke8PPO06VpcxfoPiRe4Jjq31uQTfwsebhSRsFKmzNNFeZY91GCD3tFaalnLaU30`) | `https://cdn.jsdelivr.net/npm/alpinejs@3.17.4/dist/module.esm.min.js` |
| `@alpinejs/persist@3.17.4` | MIT | `dist/module.esm.min.js` (`GVzTX289e7FrOjB0boLAoC72YktIkFmIt8pB1ZVjPttHeeQYTXFyW6yjI70EMtgO`) | `https://cdn.jsdelivr.net/npm/@alpinejs/persist@3.17.4/dist/module.esm.min.js` |
| `echarts@6.1.0` (2026-05-19) | Apache-2.0 (vendor LICENSE and NOTICE) | `dist/echarts.esm.min.mjs` (`2svH5QplTeBKhRXWxXOJE2P6pRpgnqVq8l8U1dYXUBXvc2zJ/FMMardtnPnnCXnt`) | `https://cdn.jsdelivr.net/npm/echarts@6.1.0/dist/echarts.esm.min.mjs` |
| `tabulator-tables@6.6.1` (2026-09-30) | MIT | `dist/js/tabulator_esm.min.mjs` (`BWRvD85P1JRHxztvZTpRvryElVps0W2/kgg3atlHAcR6zK70eaoV+IwUzLbn6Cao`)<br>`dist/css/tabulator_bootstrap5.min.css` (`G3nHFKuWvqfHxFfuwr4pyZK3g3pTse3g3RtcK/gJ7picizkfM3KyRHxBXqcgRGzh`) | `https://cdn.jsdelivr.net/npm/tabulator-tables@6.6.1/dist/js/tabulator_esm.min.mjs` |

- I recomputed every sha384 above from the npm tarball bytes in `/tmp/claude-0/ui-research/tarballs`, and every path exists in its tarball.
- The vendored set is about 3.2 MB.
- Every pinned version is at least 14 days old. Bump later to Tabler 1.7.1 (fixes dark-mode `.text-muted` contrast), Alpine 3.17.5 and Tabulator 6.7.1.

**Vendoring (`scripts/vendor.py`)**
1. Reads `web/vendor.lock.json`.
2. Downloads `https://registry.npmjs.org/<pkg>/-/<name>-<ver>.tgz`.
3. Verifies the tarball's npm `dist.integrity`.
4. Extracts only the listed files plus `LICENSE` and `NOTICE` and checks each sha384.
5. Writes `web/static/vendor/…`.

The vendored files are committed, so a fresh clone runs offline with no script. The CDN tags with SRI stay as a commented-out block in `base.html`.

**Never copy from Tabler `dist/libs/`.** It contains apexcharts, which is no longer MIT from 5.3.6 onward. Only layout and card markup ideas come from Tabler's demo pages; all charts are ECharts.

**`base.html` head:**

```html
<html lang="en" data-bs-theme="light">
<link rel="stylesheet" href="/static/vendor/@tabler/core@1.6.1/tabler.min.css">
<link rel="stylesheet" href="/static/vendor/@tabler/icons-webfont@3.49.0/tabler-icons.min.css">
<link rel="stylesheet" href="/static/vendor/tabulator-tables@6.6.1/tabulator_bootstrap5.min.css">
<link rel="stylesheet" href="/static/css/app.css">
<script src="/static/vendor/@tabler/core@1.6.1/tabler-theme.min.js"></script>
<script defer src="/static/vendor/@tabler/core@1.6.1/tabler.min.js"></script>
<script type="importmap">{"imports":{
 "alpinejs":"/static/vendor/alpinejs@3.17.4/module.esm.min.js",
 "@alpinejs/persist":"/static/vendor/@alpinejs/persist@3.17.4/module.esm.min.js",
 "echarts":"/static/vendor/echarts@6.1.0/echarts.esm.min.mjs",
 "tabulator-tables":"/static/vendor/tabulator-tables@6.6.1/tabulator_esm.min.mjs"}}</script>
<script type="module" src="/static/js/main.js"></script>
```

**Integration rules**

1. **Keep chart and table instances out of Alpine's reactive data.** Hold them in a closure created in `init()`; Alpine's Proxy breaks chart objects.
   ```js
   Alpine.data('marginChart', () => { let chart; return {
     init() { chart = charts.mount(this.$refs.el); this.$watch('result', r => chart.update(charts.histogram(r.margin))); },
     destroy() { chart?.dispose(); } }; });
   ```
2. **Resize on reveal.** `charts.mount` attaches a `ResizeObserver`. Tables call `redraw()` on `shown.bs.tab`.
3. **Map Tabler's variables for Tabulator** in `app.css`: `:root,[data-bs-theme]{--bs-body-bg:var(--tblr-body-bg);--bs-body-color:var(--tblr-body-color);--bs-border-color:var(--tblr-border-color);--bs-emphasis-color:var(--tblr-emphasis-color);--bs-secondary-bg:var(--tblr-secondary-bg);--bs-tertiary-bg:var(--tblr-tertiary-bg)}`.
4. **Theme sync.** A `MutationObserver` on `data-bs-theme` calls `chart.setTheme('dark'|'default')` for every mounted chart.
5. **Bootstrap classes.** Tabler's UMD bundle exposes Modal, Tab, Toast and Tooltip on the `tabler` global. Prefer `data-bs-*` attributes; use the global only for programmatic tooltips.

### 4.2 Python (append to `requirements.txt`)

```
fastapi>=0.142.4        # MIT; native SSE (fastapi.sse) needs >=0.135; 0.142.4 released 2026-10-07
uvicorn>=0.54.0         # BSD-3-Clause; 2026-09-25
jinja2>=3.1.6           # BSD-3-Clause; 2025-03-05
python-dotenv>=1.2.4    # BSD-3-Clause; 2026-10-01
```

Tests and CI also install `httpx>=0.28.1` (BSD-3-Clause) for `TestClient`. After the switch-off, `streamlit` and `matplotlib` are removed. `shap` and `optuna` stay optional.

**Why no build step.** Seven screens, a JSON API and the components above fit Alpine plus vendored ES modules. A Vite/Preact or TypeScript build would pay off only if you later want (a) TypeScript types generated from FastAPI's OpenAPI to catch API drift, (b) more than about 10 screens with shared stateful components, or (c) a custom Tailwind brand look. The API stays the same either way.

---

## 5. GitHub repos to look at (all seen and checked by the researchers; licence and last commit re-checked on the clones)

**Code you may reuse (permissive licences, keep the notice)**

| Repo | License | Last commit | Stars | What to take |
|---|---|---|---|---|
| https://github.com/BelNaruto/nba-predictions-dashboard | MIT | 2026-10-16 | 0 | Closest to the target. Take the game-card anatomy (status line, confidence badge, two team rows, split probability bar, 4 tiles), the "Market … / Value: X (4.3% edge)" line with its "model only" empty state, the slate KPI strip, the power table with ΔElo and the model-accuracy tab. File: `public/index.html` (vanilla JS). |
| https://github.com/NBA-Betting/NBA_AI | MIT | 2026-10-12 | 126 | Date picker with ‹ ›, the "Closing line / Predicted / Actual" stack, green/red recolouring after finals, and KPI tiles with baselines ("49.2% ATS, breakeven 52.4%, always home 51.2%"). Files: `src/web_app/templates/dashboard.html`, `static/js/dashboard.js` (Flask, Jinja, Bootstrap, the same no-build approach). |
| https://github.com/NBA-Betting/NBA_Betting | MIT | 2026-01-31 | 208 | Bets page: cumulative ROI area split at 0, the "X% · 52.4% needed to break even" record chart, period selector, actual vs simulated toggle. |
| https://github.com/eshangark/parallax-nba | MIT | 2026-10-27 | 1 | The explanation-ladder idea (`nba_forecast/dashboard_data.py`), the "Inspect forecast reliability" panel (injury coverage, unmatched players, timestamps), calibration bands with n and gap, and issued vs replay records. It is Streamlit, so take the concepts rather than the code. |
| https://github.com/souliN02/linedrift | MIT | 2026-07-05 | 0 | `src/lib/odds-math.ts` (de-vig, fair odds; port it to `server/services/market.py` with tests), the "impl x% · fair y%" book table, the freshness dot, and line movement and CLV (Phase 3). |
| https://github.com/saccofrancesco/deepshot | MIT | 2026-10-26 | 174 | `get_best_color_pair()` in `main.py` for contrasting two-team colours, and a "biggest stat differences" highlight. Highcharts there has its own licence; don't take it. |
| https://github.com/lobsterbush/colorr | MIT | 2026-10-01 | — | One-off seed for `teams.json` colours (`R/nbacolors.r`), with attribution. |
| https://github.com/mcekovic/tennis-crystal-ball | Apache-2.0 code; the Elo/prediction algorithms are CC BY-NC-SA 4.0 | **2022-02-22 (stale)** | 295 | `h2hHypotheticalMatchup.html`: two-sided probability header, mirrored ratings table, and a "Show details" expansion of per-factor probabilities. That is the model for the ensemble breakdown. Take UI only, not algorithms. |

**Ideas only (do not copy code)**

| Repo | License | Last commit | Why look |
|---|---|---|---|
| https://github.com/srating-io/gui | GPL-3.0 | 2026-10-25 | Best chart reasoning: `components/generic/Chart/Calibration.tsx` (dot area proportional to n), `BandLine.tsx` (rating ± σ band), `RankStrip.tsx`, `RankSpan` (rank-chip tooltip). Re-implement in ECharts and CSS. |
| https://github.com/zengm-games/zengm | Source-available (not open source) | 2026-10-25 | `src/ui/views/PowerRankings.tsx`: Current vs Healthy columns, the "with/without injuries" toggle, and the red deficit. |
| https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting | **No LICENSE** | 2026-01-08 | About 1.8k stars. Per-book columns with model probability and EV colouring, and "Kelly" chips. |

**Building blocks (§4):** https://github.com/tabler/tabler (MIT, 2026-10-24) · https://github.com/tabler/tabler-icons (MIT, 2026-10-05) · https://github.com/alpinejs/alpine (MIT, 2026-10-05) · https://github.com/apache/echarts (Apache-2.0, 2026-10-04) · https://github.com/olifolkerd/tabulator (MIT, 2026-10-22) · https://github.com/fastapi/fastapi (MIT, 2026-10-26; read `docs/en/docs/tutorial/server-sent-events.md`).

---

## 6. Migration order

Each milestone ends with green tests. Streamlit keeps working until the switch-off gate.

| # | Ships | Done when |
|---|---|---|
| **M0 Scaffold** | `server/` skeleton, `paths.py` with chdir, guard middleware, error envelope, `jsonutil`, `/api/health`, `/api/status`, `/api/teams`, vendored assets, `base.html`, nav, theme, `teams.json` | `python -m server` serves a styled empty shell offline. Tests cover Host/Origin rejection, NaN→null and `.mjs` MIME. |
| **M1 Game page** | E2, E3, E5, E7; `services/espn`, `market`, `predict`; `/api/predict` (`blend_v5` and `elo_only`), `/api/odds`, `/api/teams/{id}/form` and `/availability`; `game.html` with hero, tiles, ensemble ladder, margin/total histograms, comparison, availability, internals | Same numbers as the Streamlit Single Game tab for the same inputs, apart from MC noise. Works with no model file. Tests cover market maths, ensemble contributions summing to the final logit, and no secret in responses. |
| **M2 Jobs, Model and Settings** | E1, E6, E8; `JobManager`, SSE, train worker, autobacktest and player_impact jobs, `/api/model*`, `/api/calibrator`, `/api/cache`, `model.html`, `config_store`, secrets endpoints, Telegram test, `settings.html` with the danger zone | Training runs as a subprocess while the UI stays responsive. Progress survives a page reload, Cancel works, a second writer gets 409, and metrics survive a refresh. `.env` is written with 0600 and GET never returns values. |
| **M3 Slate** | `/api/schedule`, the scan job with `item` events, `/api/scans*`, `slate.html`, `/api/telegram/send-picks` | Parity with Radar Scanner. Kelly uses market prices, cards fill progressively, and the Telegram send actually sends. |
| **M4 Accuracy and Bets** | Backtest job, `/api/accuracy*`, `accuracy.html`; bets CRUD with ids, pending/push, CSV, `bets.html` | Parity with the Backtest and ROI tabs, plus the online-learning history panel. |
| **Switch-off gate** | Tick every row of the parity checklist below, then tag `streamlit-final`. Delete `nba_app.py`. Remove `streamlit` and `matplotlib` from requirements. Change the CI compileall line to `config.py engine data research server optuna_tuner.py player_impact_updater.py` and add `httpx`. Update the README run section to `python -m server`. | Streamlit can be uninstalled. |
| **M5 Ratings (Elo)** | Daily snapshots, standings, `/api/ratings/teams?system=elo`, history, `ratings.html` Teams tab | Power table with ΔElo, sparklines and Healthy/Current (v5 strength factor). |
| **M6 Model v6 outputs** | `engine/ratings.py`, the `ratings` job; Kalman ±SD and adjusted columns, players tab, v6 availability (play_prob, margin_impact), `stacked_v6` ensemble and margin model intervals in `/api/predict`; backtest Elo as of date (removes the "optimistic" caveat) | Fields that were null are now filled; the UI needs no change because it already renders them. |
| **M7 Polish** | `runs/forecasts.jsonl` (issued-forecast ledger, "Issued vs Replay" accuracy), `runs/odds_snapshots.jsonl` (line-movement chart, CLV on bets), Interest sort, glossary on every header, CSV exports everywhere | — |

**Parity checklist (Streamlit → new)**

| Streamlit | New location |
|---|---|
| S2–S4 Train (force rebuild, Optuna, status, metrics) | Model, Train job |
| S5–S6 auto-backtest count and button | Accuracy (calibrator card) and Model |
| S7 calibrator panel | Accuracy and Model (ensemble weights) |
| S8 feature importance, S9 calibration | Model (persisted) |
| S10–S11 player impact caption and update | Model, player_impact job |
| S12–S13 cache stats and clear | Model and Settings |
| Tab 1 Single Game (all outputs incl. SHAP, Poisson, Kelly, radar) | Game |
| Tab 2 Radar Scanner, VIP filter, Telegram | Slate |
| Tab 3 Backtest | Accuracy (range backtest) |
| Tab 4 ROI log and charts | Bets |
| Tab 5 API key, Telegram test, packages, data deletes | Settings |

**Dropped or changed**

- **Removed:**
  - All `fig_*` matplotlib functions.
  - Streamlit CSS and HTML.
  - The static "engine stack" card.
  - The mismatched "PRO v4 / v5" strings; one version comes from `server.__version__` and `model.version`.
  - Unused session keys and imports.
  - The hard-coded packages list; real import checks replace it.
- **Behaviour changes:**
  - "OT %" becomes "Within 3 pts".
  - The Poisson projected score (a single random draw) becomes λ means, and Poisson moves to "Model internals" as a legacy engine. Drop it entirely once the v6 margin model ships.
  - The fixed −110 scanner Kelly is replaced by the market price or no stake.
  - Free-text team input becomes a select.
  - "Train / Reload" becomes two buttons, Train and Reload.
  - The session-only API key moves to `.env`.
  - Data deletes now need a confirm step.

**Tests to add** (`tests/`, run in CI on Python 3.11–3.13; network functions are monkeypatched):

- `test_api.py`: status without a model; predict in `elo_only` with stubbed team stats; schema keys present; no `NaN` in JSON.
- `test_market.py`: de-vig, fair line, edge, cover probability.
- `test_jobs.py`: fake job kind through queued → running → done; SSE event order and `Last-Event-ID` resume; 409 for a second writer; cancel.
- `test_security.py`: bad Host or Origin gives 403; the secret value never appears in `/api/settings`, job logs or error details.
- `test_web_static.py`: every template extends `base.html`; every vendor file in `vendor.lock.json` exists and matches its sha384.
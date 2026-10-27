"""
Replay engine for the availability variant.

Walks the player box scores day by day.  For every team-game it first reads
the pre-game state (built from strictly earlier days only), then feeds that
day's box scores in.  The box score of the game itself is used for one bit
per player: did he take the floor or dress (available) or not (inactive,
injured, traded away, ...).  Minutes, points and plus-minus of the game only
enter the state after the game.

Per player (season-to-date sums plus the previous season's, the same numbers
leaguedashplayerstats returns for Season=current / previous):
    MIN, GS (Hollinger Game Score), PM (plus-minus), REL (plus-minus minus the
    team margin pro-rated to his minutes), GP
    em   expected minutes when he plays (EWMA over his games played)
Per team:
    p_i  participation: EWMA over the team's games of "player i played for
         us" (halflife part_hl games) — the lineup the team's Elo was earned
         with.

Ratings per 48 minutes, shrunk toward a replacement prior over k minutes:
    R_gs  = Game Score per 48 above the league average
    R_pm  = plus-minus per 48
    R_rel = plus-minus per 48 above the team margin

Team-game features (a_i = 1 if available for this game):
    NEWS_X = sum_i (a_i - p_i) * em_i / 48 * X_i      X in {1, R_gs, R_pm, R_rel}
             (lineup strength today minus the recent typical lineup; negative
             when regulars sit, positive when they return)
    MISS_X = same over unavailable players only, RET_X over available only
    STR_X  = sum over available of X_i * share of 240 minutes (absolute)
    CONT   = share of last season's team minutes played by players on the
             roster this season (played for us this season or available now)
"""
import glob
import os

import numpy as np
import pandas as pd

from research.benchmark import CACHE_DIR, DEFAULT_DATA_DIR

BOX_CACHE = os.path.join(CACHE_DIR, "availability_box.parquet")

PARAMS = dict(
    part_hl=8.0,      # participation halflife (team games)
    min_hl=8.0,       # expected-minutes halflife (games played)
    new_min=10.0,     # expected minutes of a player with no history
    w_prev=0.6,       # weight of last season's sums next to this season's
    k_gs=400.0,       # shrinkage minutes, Game Score rating
    k_pm=1000.0,      # shrinkage minutes, plus-minus rating
    k_rel=1500.0,     # shrinkage minutes, relative plus-minus rating
    b_gs=-6.0,        # prior (replacement) Game Score per 48 vs league average
    b_pm=-2.0,        # prior plus-minus per 48
    b_rel=-1.0,       # prior relative plus-minus per 48
)

# index of each sum inside a player's season vector
MIN, GS, PM, REL, GP = range(5)


# ── data ──────────────────────────────────────────────────────────────────────

def _minutes(s: pd.Series) -> np.ndarray:
    """'MM:SS' (or empty for DNP) → float minutes."""
    s = s.fillna("").astype(str)
    mm = s.str.extract(r"^(\d+)(?:\.\d+)?:(\d+)")
    out = mm[0].astype(float) + mm[1].astype(float) / 60
    plain = pd.to_numeric(s.where(mm[0].isna()), errors="coerce")
    return out.fillna(plain).fillna(0.0).values


def load_box(data_dir: str = DEFAULT_DATA_DIR, use_cache: bool = True) -> pd.DataFrame:
    """Player box scores, one row per listed player-game, with STATUS
    2 = played, 1 = dressed but unused (DNP - Coach's Decision), 0 = listed
    out (DND / NWT / DNP - Injury ...).  Cached as parquet."""
    if use_cache and os.path.exists(BOX_CACHE):
        return pd.read_parquet(BOX_CACHE)
    files = sorted(glob.glob(os.path.join(
        data_dir, "regular_season_box_scores_2010_2024_part_*.csv")))
    files.append(os.path.join(data_dir, "play_off_box_scores_2010_2024.csv"))
    df = pd.concat([pd.read_csv(f) for f in files if os.path.exists(f)],
                   ignore_index=True)
    df["GAME_ID"] = df["gameId"].astype(str).str.zfill(10)
    df = df.drop_duplicates(["GAME_ID", "personId"])
    mins = _minutes(df["minutes"])
    comment = df["comment"].fillna("").astype(str).str.strip()
    status = np.where(mins > 0, 2,
                      np.where((comment == "") | comment.str.startswith("DNP - Coach"), 1, 0))
    n = lambda c: df[c].fillna(0).astype(float).values  # noqa: E731
    gs = (n("points") + 0.4 * n("fieldGoalsMade") - 0.7 * n("fieldGoalsAttempted")
          - 0.4 * (n("freeThrowsAttempted") - n("freeThrowsMade"))
          + 0.7 * n("reboundsOffensive") + 0.3 * n("reboundsDefensive")
          + n("steals") + 0.7 * n("assists") + 0.7 * n("blocks")
          - 0.4 * n("foulsPersonal") - n("turnovers"))
    out = pd.DataFrame({
        "GAME_ID": df["GAME_ID"].values,
        "TEAM_ID": df["teamId"].astype(int).values,
        "PLAYER_ID": df["personId"].astype(int).values,
        "STATUS": status.astype(np.int8),
        "MIN": mins,
        "GS": np.where(mins > 0, gs, 0.0),
        "PM": np.where(mins > 0, n("plusMinusPoints"), 0.0),
    })
    os.makedirs(CACHE_DIR, exist_ok=True)
    out.to_parquet(BOX_CACHE, index=False)
    return out


def game_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Team game logs → one row per game (home vs away), date-sorted."""
    df = raw.copy()
    df["GAME_ID"] = df["GAME_ID"].astype(str).str.zfill(10)
    df = df[df["GAME_ID"].str[:3].isin(("002", "004", "005"))]
    df = df.dropna(subset=["WL", "PLUS_MINUS"])
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.normalize()
    is_home = df["MATCHUP"].str.contains("vs.", regex=False)
    h = df[is_home][["GAME_ID", "GAME_DATE", "SEASON_YEAR", "TEAM_ID", "WL", "PLUS_MINUS"]]
    a = df[~is_home][["GAME_ID", "TEAM_ID"]]
    g = h.merge(a, on="GAME_ID", suffixes=("_H", "_A"))
    g = g.rename(columns={"SEASON_YEAR": "SEASON"})
    g["RESULT"] = (g["WL"] == "W").astype(int)
    g["MARGIN"] = g["PLUS_MINUS"].astype(float)
    g = g.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    return g[["GAME_ID", "GAME_DATE", "SEASON", "TEAM_ID_H", "TEAM_ID_A",
              "RESULT", "MARGIN"]]


# ── replay ────────────────────────────────────────────────────────────────────

FEATS = ["NEWS_MIN", "NEWS_GS", "NEWS_PM", "NEWS_REL",
         "MISS_MIN", "MISS_GS", "MISS_PM", "MISS_REL",
         "RET_MIN", "RET_GS", "RET_PM", "RET_REL",
         "STR_GS", "STR_PM", "STR_REL", "CONT", "TEAM_GP"]


def _vec():
    return [0.0, 0.0, 0.0, 0.0, 0.0]


class Engine:
    """Player and team state; all reads see only games already fed in."""

    def __init__(self, **params):
        self.p = {**PARAMS, **params}
        self.alpha_part = 1 - 0.5 ** (1 / self.p["part_hl"])
        self.alpha_min = 1 - 0.5 ** (1 / self.p["min_hl"])
        self.pl = {}          # pid -> [season, cur vec, prev vec, em]
        self.part = {}        # tid -> {pid: participation}
        self.lg = {}          # season -> league vec (MIN, GS)
        self.team_min = {}    # (tid, season) -> {pid: minutes}
        self.team_gp = {}     # (tid, season) -> games played

    # season vectors as seen from season s
    def _sums(self, pid, s):
        st = self.pl.get(pid)
        if st is None:
            return None, None
        if st[0] == s:
            return st[1], st[2]
        if st[0] == s - 1:
            return None, st[1]
        return None, None

    def _league_gs48(self, s):
        for k, w in ((s, 1.0), (s - 1, 1.0)):
            v = self.lg.get(k)
            if v and v[0] > 5000:
                return 48 * v[1] / v[0]
        return 22.0

    def ratings(self, pid, s, lg48):
        """(em, R_gs, R_pm, R_rel) of a player before a game in season s."""
        p = self.p
        cur, prev = self._sums(pid, s)
        m = [0.0] * 5
        for vec, w in ((cur, 1.0), (prev, p["w_prev"])):
            if vec is not None:
                for j in range(5):
                    m[j] += w * vec[j]
        st = self.pl.get(pid)
        em = st[3] if st is not None else p["new_min"]
        r_gs = (48 * m[GS] + p["k_gs"] * (lg48 + p["b_gs"])) / (m[MIN] + p["k_gs"]) - lg48
        r_pm = (48 * m[PM] + p["k_pm"] * p["b_pm"]) / (m[MIN] + p["k_pm"])
        r_rel = (48 * m[REL] + p["k_rel"] * p["b_rel"]) / (m[MIN] + p["k_rel"])
        return em, r_gs, r_pm, r_rel

    def features(self, tid, s, avail: set) -> list:
        """Pre-game availability features of team tid (see module doc)."""
        lg48 = self._league_gs48(s)
        part = self.part.get(tid, {})
        news = [0.0] * 4
        miss = [0.0] * 4
        ret = [0.0] * 4
        strength = [0.0] * 3
        tot_m = 0.0
        for pid in set(part) | avail:
            em, r_gs, r_pm, r_rel = self.ratings(pid, s, lg48)
            a = 1.0 if pid in avail else 0.0
            d = (a - part.get(pid, 0.0)) * em / 48
            x = (d, d * r_gs, d * r_pm, d * r_rel)
            tgt = ret if a else miss
            for j in range(4):
                news[j] += x[j]
                tgt[j] += x[j]
            if a:
                tot_m += em
                strength[0] += em * r_gs
                strength[1] += em * r_pm
                strength[2] += em * r_rel
        scale = 5.0 / tot_m if tot_m > 0 else 0.0
        strength = [v * scale for v in strength]
        last = self.team_min.get((tid, s - 1), {})
        tot_last = sum(last.values())
        roster = avail | set(self.team_min.get((tid, s), {}))
        cont = (sum(v for k, v in last.items() if k in roster) / tot_last
                if tot_last > 0 else np.nan)
        return news + miss + ret + strength + [cont, self.team_gp.get((tid, s), 0)]

    def update(self, tid, s, margin, rows):
        """Feed one finished team-game: rows of (pid, status, min, gs, pm)."""
        a_p, a_m = self.alpha_part, self.alpha_min
        part = self.part.setdefault(tid, {})
        for k in list(part):
            part[k] *= 1 - a_p
            if part[k] < 0.01:
                del part[k]
        tm = self.team_min.setdefault((tid, s), {})
        self.team_gp[(tid, s)] = self.team_gp.get((tid, s), 0) + 1
        lg = self.lg.setdefault(s, [0.0, 0.0])
        for pid, status, mins, gs, pm in rows:
            if status != 2:
                continue
            st = self.pl.get(pid)
            if st is None:
                st = self.pl[pid] = [s, _vec(), None, self.p["new_min"]]
            if st[0] != s:
                st[2] = st[1] if st[0] == s - 1 else None
                st[1] = _vec()
                st[0] = s
            v = st[1]
            v[MIN] += mins
            v[GS] += gs
            v[PM] += pm
            v[REL] += pm - margin * mins / 48
            v[GP] += 1
            st[3] += a_m * (mins - st[3])
            part[pid] = part.get(pid, 0.0) + a_p
            tm[pid] = tm.get(pid, 0.0) + mins
            lg[0] += mins
            lg[1] += gs


def replay(games: pd.DataFrame, box: pd.DataFrame, **params) -> pd.DataFrame:
    """Pre-game features H_*, A_* for every row of games (same order)."""
    eng = Engine(**params)
    box = box[box["GAME_ID"].isin(set(games["GAME_ID"]))]
    by_tg = {k: list(zip(g["PLAYER_ID"], g["STATUS"], g["MIN"], g["GS"], g["PM"]))
             for k, g in box.groupby(["GAME_ID", "TEAM_ID"], sort=False)}
    season = games["SEASON"].str[:4].astype(int).values
    out = np.full((len(games), 2, len(FEATS)), np.nan)
    gid, hid, aid = games["GAME_ID"].values, games["TEAM_ID_H"].values, games["TEAM_ID_A"].values
    margin = games["MARGIN"].values
    day = games["GAME_DATE"].values
    i, n = 0, len(games)
    while i < n:
        j = i
        while j < n and day[j] == day[i]:
            j += 1
        for k in range(i, j):            # pre-game reads for the whole day
            for side, t in ((0, hid[k]), (1, aid[k])):
                rows = by_tg.get((gid[k], t), [])
                avail = {r[0] for r in rows if r[1] >= 1}
                out[k, side] = eng.features(t, season[k], avail)
        for k in range(i, j):            # then the day's results
            for t, mg in ((hid[k], margin[k]), (aid[k], -margin[k])):
                eng.update(t, season[k], mg, by_tg.get((gid[k], t), []))
        i = j
    cols = {}
    for side, pre in ((0, "H_"), (1, "A_")):
        for f, name in enumerate(FEATS):
            cols[pre + name] = out[:, side, f]
    return pd.DataFrame(cols, index=games.index)

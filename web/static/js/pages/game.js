// Game page: one matchup in depth (POST /api/predict).
//
// gamePage holds the form and the response; buildView() turns the response
// into display strings once (and again when the odds format changes).
// gameChart(kind) mounts an ECharts instance lazily, when its box becomes
// visible, and keeps it in a closure (never in Alpine's reactive data).

import { api } from '../api.js';
import { fmt } from '../format.js';
import * as charts from '../charts.js';
import { pairColors } from '../teams.js';

const { pct, signed, num, odds, spread, tipoff, day, ago } = fmt;
const DASH = fmt.DASH;
const STATUSES = ['out', 'doubtful', 'questionable', 'day-to-day'];
const STATUS_LABEL = { out: 'Out', doubtful: 'Doubtful', questionable: 'Questionable', 'day-to-day': 'DTD' };
const STATUS_CLASS = {
  out: 'bg-red-lt', doubtful: 'bg-orange-lt', questionable: 'bg-yellow-lt', 'day-to-day': 'bg-secondary-lt',
};
const CAVEATS = {
  elo_only_no_model: ['info', 'No trained model: the win probability comes from Elo; Monte Carlo gives the score ranges.'],
  injuries_unavailable: ['warning', 'No injury report used: the feed is unavailable or the date is in the past.'],
  injuries_stale: ['warning', 'The injury report is from an older fetch (feed offline).'],
  odds_stale: ['warning', 'Market odds come from a cached scoreboard (ESPN offline).'],
  stats_stale: ['warning', 'Team stats come from an older download (stats.nba.com offline).'],
  past_date_current_elo: ['info', 'Past date: team form is as of that date, but Elo ratings are current.'],
  demo_mode: ['info', 'Demo replay: historical team logs, no injury report and no market odds.'],
};
const ENGINE_LABEL = { xgb: 'XGBoost', mc: 'Monte Carlo', elo: 'Elo' };

// ── odds input: the user's format (US or decimal) ↔ American ─────────────

/** Parse a price typed in the user's format to an American moneyline (or null). */
export function parseOdds(text, mode) {
  const s = String(text ?? '').trim().replace('−', '-').replace(/^\+/, '');
  if (!s) return null;
  const x = Number(s);
  if (!Number.isFinite(x)) return NaN;
  const looksDecimal = mode === 'dec' || (Math.abs(x) < 100 && x > 1 && s.includes('.'));
  if (looksDecimal) {
    if (x <= 1.001) return NaN;
    return Math.round(x >= 2 ? (x - 1) * 100 : -100 / (x - 1));
  }
  if (Math.abs(x) < 100) return NaN;
  return Math.round(x);
}

function oddsText(ml, mode) {
  if (ml == null) return '';
  return mode === 'dec' ? fmt.americanToDecimal(ml).toFixed(2) : (ml > 0 ? `+${ml}` : String(ml));
}

// ── view model ───────────────────────────────────────────────────────────

const signedSpread = (x) => (Math.abs(x) < 0.05 ? 'PK' : signed(x, 1));
const pct1 = (x) => pct(x, 1);
const streak = (s) => (s > 0 ? `W${s}` : s < 0 ? `L${-s}` : DASH);
const ordinal = (n) => {
  const s = ['th', 'st', 'nd', 'rd'];
  const v = n % 100;
  return n + (s[(v - 20) % 10] || s[v] || s[0]);
};

const COMPARE = [
  { label: 'Net rating, last 5', key: 'net5', f: (v) => signed(v, 1), better: 'high' },
  { label: 'Net rating, last 15', key: 'net15', f: (v) => signed(v, 1), better: 'high' },
  { label: 'Points, last 10', key: 'pts10', f: (v) => num(v, 1), better: 'high' },
  { label: 'Allowed, last 10', key: 'pa10', f: (v) => num(v, 1), better: 'low' },
  { label: 'eFG%', key: 'efg10', f: pct1, better: 'high' },
  { label: 'TS%', key: 'ts_pct', f: pct1, better: 'high' },
  { label: 'Turnover rate', key: 'tov10', f: pct1, better: 'low' },
  { label: 'Off. rebound rate', key: 'orb10', f: pct1, better: 'high' },
  { label: 'Free-throw rate', key: 'ftr10', f: (v) => num(v, 3), better: 'high' },
  { label: 'AST/TOV', key: 'ast_tov', f: (v) => num(v, 2), better: 'high' },
  { label: 'Pace', key: 'pace10', f: (v) => num(v, 1), better: null },
  { label: 'Win%, last 10', key: 'wpct10', f: (v) => pct(v, 0), better: 'high' },
  { label: 'Streak', key: 'win_streak', f: streak, better: 'high' },
  { label: 'Rest days', key: 'days_rest', f: (v) => num(v, 0), better: 'high' },
  { label: 'Back-to-back', key: 'is_b2b', f: (v) => (v ? 'Yes' : 'No'), better: 'low' },
];

function betterSide(h, a, better) {
  if (!better || h == null || a == null || h === a) return null;
  return (better === 'high') === (h > a) ? 'home' : 'away';
}

function teamInfo(ref, byId) {
  const t = byId[ref.id] || {};
  return { ...ref, nickname: t.nickname || ref.name.split(' ').pop(), colors: t.colors || null };
}

function availabilityView(av, team) {
  const deficit = av.deficit ?? 0;
  const units = av.units === 'points' ? 'pts' : null;
  const val = (x) => (units ? `${num(x, 1)} pts` : pct(x, 0));
  let empty = 'No injuries reported.';
  if (av.source === 'none') empty = 'Injuries ignored (None selected).';
  else if (av.source === 'custom') empty = 'No players marked out in the custom list.';
  else if (av.status === 'unavailable') empty = 'Injury report unavailable for this date.';
  const reliability = [];
  if (av.fetched_at) reliability.push(`Report fetched ${fmt.time(av.fetched_at, 'America/New_York')} ET (${ago(av.fetched_at)})`);
  if (av.status === 'stale') reliability.push('cached copy (feed offline)');
  if (av.unmatched) reliability.push(`${av.unmatched} player${av.unmatched > 1 ? 's' : ''} unmatched`);
  return {
    team,
    healthy: val(av.healthy),
    current: val(av.current),
    deficit: Math.abs(deficit) < 1e-4 ? null : (units ? `${signed(deficit, 1)} pts` : `${signed(deficit * 100, 0)}%`),
    hurt: deficit < -1e-4,
    players: av.players.map((p) => ({
      ...p,
      label: STATUS_LABEL[p.status] || p.status,
      cls: STATUS_CLASS[p.status] || 'bg-secondary-lt',
      impactText: num(p.impact, 3),
      playText: p.play_prob == null ? DASH : pct(p.play_prob, 0),
      marginText: p.margin_impact == null ? DASH : signed(p.margin_impact, 1),
    })),
    v6: av.players.some((p) => p.play_prob != null || p.margin_impact != null),
    empty,
    reliability: reliability.join(' · '),
    status: av.status,
  };
}

export function buildView(r, mode, byId) {
  const home = teamInfo(r.game.home, byId);
  const away = teamInfo(r.game.away, byId);
  const colors = pairColors(home, away);
  const pickTeam = r.pick.side === 'home' ? home : away;
  const m = r.market;
  const hasPrice = m && m.home_ml != null && m.away_ml != null;

  // pick line
  let pickLine; let pickKind;
  const agree = r.mode === 'elo_only' ? 'Elo and Monte Carlo agree' : 'engines agree';
  if (r.flags.conflict) {
    pickKind = 'conflict';
    pickLine = r.mode === 'elo_only'
      ? 'Elo and Monte Carlo disagree: no pick'
      : 'XGBoost and Monte Carlo disagree: no pick';
  } else if (r.flags.show_pick) {
    pickKind = 'pick';
    pickLine = `Pick: ${pickTeam.nickname} ${pct(r.pick.prob)} · ${agree}`;
  } else {
    pickKind = 'lean';
    pickLine = `Lean ${pickTeam.nickname} ${pct(r.pick.prob)}: below the pick threshold`;
  }

  // game status line
  const t = tipoff(r.game.tipoff_utc);
  const status = { pre: 'Scheduled', in: 'Live', post: 'Final' }[r.game.status] || 'Not on the schedule';
  const when = r.game.tipoff_utc ? `${t.localDay} ${t.local} local · ${t.et}` : day(r.game.date, true);
  let final = null;
  if (r.game.result) {
    const { home_score: hs, away_score: as } = r.game.result;
    const winner = hs > as ? 'home' : 'away';
    final = {
      text: `Final: ${away.abbr} ${as} – ${hs} ${home.abbr}`,
      correct: winner === r.pick.side,
    };
  }

  // market line
  let marketLine = 'No market yet: model only';
  let value = null;
  if (m && hasPrice) {
    const fav = m.fair.home >= m.fair.away ? home : away;
    const parts = [`Market (fair): ${fav.abbr} ${pct(Math.max(m.fair.home, m.fair.away))}`];
    if (m.spread != null) parts.push(spread(home.abbr, away.abbr, m.spread));
    if (m.total != null) parts.push(num(m.total, 1));
    parts.push(m.status === 'manual' ? 'manual prices' : m.source);
    marketLine = parts.join(' · ');
    if (m.value_side) {
      value = { label: `Value: ${(m.value_side === 'home' ? home : away).abbr}`, edge: m.edge[m.value_side] };
    }
  } else if (m && m.status === 'no_odds_yet') {
    marketLine = 'No odds posted yet: model only';
  }

  // tiles
  const s = r.scores;
  const tiles = {
    score: {
      value: `${away.abbr} ${num(s.away.mean, 0)} – ${num(s.home.mean, 0)} ${home.abbr}`,
      sub: `50% range ${num(s.away.p25, 0)}–${num(s.away.p75, 0)} / ${num(s.home.p25, 0)}–${num(s.home.p75, 0)}`,
    },
    spread: {
      value: spread(home.abbr, away.abbr, r.margin.home_spread),
      sub: m && m.spread != null
        ? `Market ${spread(home.abbr, away.abbr, m.spread)} · P(${home.abbr} covers ${signedSpread(m.spread)}) ${pct(m.cover_prob)}`
        : 'No market spread',
    },
    total: {
      value: num(r.total.median, 1),
      sub: m && m.total != null
        ? `Market ${num(m.total, 1)} · P(over) ${pct(m.over_prob)}`
        : `50% range ${num(r.total.intervals.p50[0], 0)}–${num(r.total.intervals.p50[1], 0)}`,
    },
    fair: {
      value: `${away.abbr} ${odds(r.fair_line.away_ml, mode)} / ${home.abbr} ${odds(r.fair_line.home_ml, mode)}`,
      sub: hasPrice
        ? `${m.status === 'manual' ? 'Manual' : 'Market'} ${odds(m.away_ml, mode)} / ${odds(m.home_ml, mode)}`
        : 'No market price',
    },
    kelly: { value: DASH, sub: 'Add odds to size a stake', none: true },
  };
  if (r.kelly) {
    const k = r.kelly;
    const side = k.home.frac_pct >= k.away.frac_pct ? 'home' : 'away';
    const ks = k[side];
    const team = side === 'home' ? home : away;
    const share = k.fraction === 0.25 ? '¼' : `${num(k.fraction, 2)}×`;
    tiles.kelly = ks.frac_pct > 0
      ? { value: `${num(ks.frac_pct, 1)}%`,
          sub: `${team.abbr} at ${odds(side === 'home' ? m.home_ml : m.away_ml, mode)} · ${share} Kelly${ks.capped ? `, capped at ${num(k.cap_pct, 0)}%` : ''}` }
      : { value: '0%', sub: 'No edge at this price' };
  }

  // ensemble ladder table
  const e = r.ensemble;
  const ladder = {
    method: e.method === 'elo' ? 'Elo only (no trained model)'
      : `Blend calibrator · ${e.n_updates ?? 0} online updates`,
    base: pct1(e.base.prob),
    final: pct1(e.final.prob),
    rows: e.components.map((c) => ({
      key: c.key,
      label: c.label,
      prob: pct1(c.prob),
      coef: num(c.coef, 3),
      step: `${signed(c.step_pp, 1)} pp`,
      up: c.step_pp > 0,
      share: c.share == null ? DASH : pct(c.share, 0),
    })),
  };

  // distribution captions
  const mg = r.margin;
  const distText = {
    margin: [
      m && m.spread != null ? `P(${home.abbr} covers ${signedSpread(m.spread)}) ${pct(m.cover_prob)}` : null,
      `P(within 3) ${pct(mg.p_within_3)}`,
      `80% range ${signed(mg.intervals.p80[0], 0)} to ${signed(mg.intervals.p80[1], 0)} (${home.abbr} margin)`,
    ].filter(Boolean).join(' · '),
    total: [
      m && m.total != null ? `P(over ${num(m.total, 1)}) ${pct(m.over_prob)}` : null,
      `Median ${num(r.total.median, 1)}`,
      `80% range ${num(r.total.intervals.p80[0], 0)}–${num(r.total.intervals.p80[1], 0)}`,
    ].filter(Boolean).join(' · '),
    scores: `${away.abbr} ${num(s.away.mean, 1)} (50%: ${num(s.away.p25, 0)}–${num(s.away.p75, 0)}) · `
      + `${home.abbr} ${num(s.home.mean, 1)} (50%: ${num(s.home.p25, 0)}–${num(s.home.p75, 0)})`,
  };

  // team comparison
  const rh = r.ratings.home; const ra = r.ratings.away;
  const compare = [{
    label: 'Elo', glossary: 'elo',
    home: num(rh.elo, 0), away: num(ra.elo, 0),
    homeRank: rh.elo_rank, awayRank: ra.elo_rank,
    better: betterSide(rh.elo, ra.elo, 'high'),
  }, {
    label: 'Kalman ±SD', glossary: 'kalman', v6: !rh.kalman,
    home: rh.kalman ? `${signed(rh.kalman.rating, 1)} ±${num(rh.kalman.sd, 1)}` : DASH,
    away: ra.kalman ? `${signed(ra.kalman.rating, 1)} ±${num(ra.kalman.sd, 1)}` : DASH,
    better: rh.kalman && ra.kalman ? betterSide(rh.kalman.rating, ra.kalman.rating, 'high') : null,
  }, {
    label: 'Adjusted net', glossary: 'adjusted_net', v6: !rh.adjusted,
    home: rh.adjusted ? signed(rh.adjusted.net, 1) : DASH,
    away: ra.adjusted ? signed(ra.adjusted.net, 1) : DASH,
    better: rh.adjusted && ra.adjusted ? betterSide(rh.adjusted.net, ra.adjusted.net, 'high') : null,
  }];
  for (const row of COMPARE) {
    const h = r.teams.home[row.key]; const a = r.teams.away[row.key];
    compare.push({ label: row.label, home: row.f(h), away: row.f(a), better: betterSide(h, a, row.better) });
  }

  const caveats = (r.caveats || []).filter((c) => CAVEATS[c]).map((c) => ({ code: c, kind: CAVEATS[c][0], text: CAVEATS[c][1] }));

  const engines = ['xgb', 'mc', 'elo'].filter((k) => r.engines[k] != null)
    .map((k) => `${ENGINE_LABEL[k]} ${pct(r.engines[k])}`);

  return {
    home, away, colors, pickLine, pickKind, status, when, final, marketLine, value, tiles,
    ladder, distText, compare, caveats,
    engines: `P(${home.abbr} win): ${engines.join(' · ')}`,
    availability: {
      home: availabilityView(r.availability.home, home),
      away: availabilityView(r.availability.away, away),
    },
    features: Object.entries(r.feature_row || {}).map(([name, v]) => ({ name, value: num(v, 3) })),
    poisson: r.poisson ? {
      line: `λ means ${away.abbr} ${num(r.poisson.mean_a, 1)} – ${num(r.poisson.mean_h, 1)} ${home.abbr} · `
        + `median total ${num(r.poisson.median_total, 1)} · 20–80%: ${num(r.poisson.ou_20, 0)}–${num(r.poisson.ou_80, 0)} · `
        + `P(${home.abbr}) ${pct(r.poisson.home_prob)}`,
    } : null,
    shapNote: r.shap ? `Base value ${num(r.shap.base_value, 3)} (log-odds). Bars push toward ${home.abbr} (right) or ${away.abbr} (left).` : null,
    modelText: r.model ? `Model ${r.model.version}, trained ${r.model.age_days != null ? `${num(r.model.age_days, 0)} days ago` : ''}` : 'Elo + Monte Carlo',
  };
}

// ── chart options ────────────────────────────────────────────────────────

// Tabler 1.6 defines colours with oklch() / light-dark(), which ECharts can't
// parse: resolve a CSS colour variable to rgba() through a 1-pixel canvas.
let probe = null;
function cssVar(name, fallback) {
  try {
    const span = document.createElement('span');
    span.style.color = `var(${name}, ${fallback})`;
    document.body.appendChild(span);
    const color = getComputedStyle(span).color;
    span.remove();
    probe ??= Object.assign(document.createElement('canvas'), { width: 1, height: 1 }).getContext('2d', { willReadFrequently: true });
    probe.clearRect(0, 0, 1, 1);
    probe.fillStyle = fallback;
    probe.fillStyle = color;
    probe.fillRect(0, 0, 1, 1);
    const [r, g, b, a] = probe.getImageData(0, 0, 1, 1).data;
    return `rgba(${r}, ${g}, ${b}, ${(a / 255).toFixed(3)})`;
  } catch {
    return fallback;
  }
}

function ladderOption(r, v) {
  const e = r.ensemble;
  const rows = [{ label: 'Neutral game', start: null, end: e.base.prob * 100 }];
  let prev = e.base.prob * 100;
  for (const c of e.components) {
    const next = prev + c.step_pp;
    rows.push({ label: `+ ${c.label}`, start: prev, end: next, step: c.step_pp });
    prev = next;
  }
  rows.push({ label: 'Final', start: null, end: e.final.prob * 100 });
  const values = rows.flatMap((x) => (x.start == null ? [x.end] : [x.start, x.end])).concat([50]);
  const lo = Math.max(0, Math.floor((Math.min(...values) - 5) / 5) * 5);
  const hi = Math.min(100, Math.ceil((Math.max(...values) + 5) / 5) * 5);
  const neutral = cssVar('--tblr-secondary', '#6c7a91');
  const primary = cssVar('--tblr-primary', '#066fd1');
  const labels = rows.map((x) => (x.start == null ? `${x.end.toFixed(1)}%` : `${signed(x.step, 1)} pp`));
  return {
    grid: { left: 8, right: 56, top: 8, bottom: 24, containLabel: true },
    tooltip: {
      trigger: 'axis', axisPointer: { type: 'shadow' },
      formatter: (ps) => {
        const i = ps[0].dataIndex; const x = rows[i];
        return x.start == null ? `${x.label}: ${x.end.toFixed(1)}%`
          : `${x.label}: ${x.start.toFixed(1)}% → ${x.end.toFixed(1)}% (${signed(x.step, 1)} pp)`;
      },
    },
    xAxis: { type: 'value', min: lo, max: hi, axisLabel: { formatter: '{value}%' }, splitLine: { show: true } },
    yAxis: { type: 'category', inverse: true, data: rows.map((x) => x.label), axisTick: { show: false } },
    series: [
      {
        type: 'bar', stack: 'w', silent: true, itemStyle: { color: 'transparent' },
        data: rows.map((x) => (x.start == null ? lo : Math.min(x.start, x.end))),
      },
      {
        type: 'bar', stack: 'w', barMaxWidth: 26, name: 'P(home win)',
        data: rows.map((x) => ({
          value: x.start == null ? x.end - lo : Math.max(Math.abs(x.end - x.start), 0.15),
          itemStyle: { color: x.start == null ? (x.label === 'Final' ? primary : neutral) : (x.step >= 0 ? v.colors.home : v.colors.away) },
        })),
        label: { show: true, position: 'right', formatter: (p) => labels[p.dataIndex] },
        markLine: {
          silent: true, symbol: 'none', data: [{ xAxis: 50 }],
          lineStyle: { type: 'dashed', color: neutral }, label: { show: false },
        },
      },
    ],
  };
}

function histSeries(name, hist, colorOf, opacity = 0.9) {
  const total = hist.counts.reduce((a, b) => a + b, 0) || 1;
  const data = hist.counts.map((c, i) => [hist.edges[i], hist.edges[i + 1], (c / total) * 100]);
  return {
    type: 'custom', name,
    encode: { x: [0, 1], y: 2 },
    dimensions: ['from', 'to', 'share'],
    renderItem(params, api) {
      const x0 = api.value(0); const x1 = api.value(1); const y = api.value(2);
      const a = api.coord([x0, y]); const b = api.coord([x1, 0]);
      return {
        type: 'rect',
        shape: { x: a[0], y: a[1], width: Math.max(b[0] - a[0] - 1, 1), height: b[1] - a[1] },
        style: { fill: colorOf((x0 + x1) / 2), opacity },
      };
    },
    data,
  };
}

function markers(lines, band) {
  return {
    type: 'line', data: [], silent: true, symbol: 'none',
    markLine: {
      silent: true, symbol: 'none',
      lineStyle: { type: 'dashed' },
      label: { show: false },
      data: lines.filter((l) => l.x != null).map((l) => ({
        name: l.label, xAxis: l.x, lineStyle: { color: l.color, type: l.type || 'dashed', width: 2 },
      })),
    },
    markArea: band ? {
      silent: true, itemStyle: { color: 'rgba(120,130,150,0.16)' },
      label: { show: false },
      data: [[{ xAxis: band.lo }, { xAxis: band.hi }]],
    } : undefined,
  };
}

function distOption(r, v, tab) {
  const neutral = cssVar('--tblr-secondary', '#6c7a91');
  const text = cssVar('--tblr-body-color', '#182433');
  const tooltip = {
    trigger: 'item',
    formatter: (p) => (p.seriesType === 'custom'
      ? `${p.seriesName}: ${num(p.value[0], 0)} to ${num(p.value[1], 0)} — ${num(p.value[2], 1)}% of sims`
      : ''),
  };
  const yAxis = { type: 'value', name: '% of sims', nameTextStyle: { align: 'left' }, axisLabel: { formatter: '{value}%' } };
  const grid = { left: 8, right: 16, top: 32, bottom: 30, containLabel: true };
  const lo = (h) => Math.floor(h.edges[0] / 5) * 5;
  const hi = (h) => Math.ceil(h.edges[h.edges.length - 1] / 5) * 5;
  const m = r.market;
  if (tab === 'total') {
    const h = r.total.hist;
    return {
      grid, tooltip, yAxis,
      xAxis: { type: 'value', min: lo(h), max: hi(h), name: 'Total points', nameLocation: 'middle', nameGap: 28 },
      series: [
        histSeries('Total', h, () => neutral, 0.75),
        markers([
          { x: r.total.median, label: `Model ${num(r.total.median, 1)}`, color: text, type: 'solid' },
          { x: m?.total, label: `Market ${num(m?.total, 1)}`, color: cssVar('--tblr-orange', '#f76707') },
        ], { lo: r.total.intervals.p80[0], hi: r.total.intervals.p80[1], label: '80%' }),
      ],
    };
  }
  if (tab === 'scores') {
    const ha = r.scores.away.hist; const hh = r.scores.home.hist;
    return {
      grid, yAxis,
      tooltip,
      legend: { top: 0, data: [v.away.abbr, v.home.abbr] },
      color: [v.colors.away, v.colors.home],
      xAxis: { type: 'value', min: Math.min(lo(ha), lo(hh)), max: Math.max(hi(ha), hi(hh)), name: 'Points', nameLocation: 'middle', nameGap: 28 },
      series: [
        histSeries(v.away.abbr, ha, () => v.colors.away, 0.55),
        histSeries(v.home.abbr, hh, () => v.colors.home, 0.55),
      ],
    };
  }
  const h = r.margin.hist;
  return {
    grid, tooltip, yAxis,
    xAxis: {
      type: 'value', min: lo(h), max: hi(h),
      name: `← ${v.away.abbr} wins · ${v.home.abbr} margin · ${v.home.abbr} wins →`, nameLocation: 'middle', nameGap: 28,
    },
    series: [
      histSeries('Margin', h, (x) => (x >= 0 ? v.colors.home : v.colors.away), 0.8),
      markers([
        { x: 0, label: 'Even', color: neutral, type: 'solid' },
        { x: r.margin.mean, label: `Model ${signed(r.margin.mean, 1)}`, color: text, type: 'solid' },
        { x: m?.spread != null ? -m.spread : null, label: `Market ${signed(-(m?.spread ?? 0), 1)}`, color: cssVar('--tblr-orange', '#f76707') },
      ], { lo: r.margin.intervals.p80[0], hi: r.margin.intervals.p80[1], label: '80%' }),
    ],
  };
}

function radarOption(r, v) {
  return {
    legend: { bottom: 0, data: [v.away.abbr, v.home.abbr] },
    tooltip: { trigger: 'item' },
    color: [v.colors.away, v.colors.home],
    radar: {
      radius: '60%', center: ['50%', '47%'],
      indicator: r.radar.axes.map((name) => ({ name, max: 1, min: 0 })),
      axisName: { color: cssVar('--tblr-secondary', '#6c7a91') },
    },
    series: [{
      type: 'radar',
      data: [
        { name: v.away.abbr, value: r.radar.away, areaStyle: { opacity: 0.18 } },
        { name: v.home.abbr, value: r.radar.home, areaStyle: { opacity: 0.18 } },
      ],
    }],
  };
}

function shapOption(r, v) {
  if (!r.shap) return null;
  const feats = [...r.shap.features].reverse();          // largest |shap| on top
  return {
    grid: { left: 8, right: 48, top: 8, bottom: 30, containLabel: true },
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    xAxis: { type: 'value', name: 'SHAP (log-odds)', nameLocation: 'middle', nameGap: 28 },
    yAxis: { type: 'category', data: feats.map((f) => f.name), axisTick: { show: false } },
    series: [{
      type: 'bar', name: 'SHAP', barMaxWidth: 16,
      data: feats.map((f) => ({ value: f.shap, itemStyle: { color: f.shap >= 0 ? v.colors.home : v.colors.away } })),
      label: { show: true, position: 'right', formatter: (p) => signed(p.value, 3) },
    }],
  };
}

function poissonOption(r, v) {
  if (!r.poisson) return null;
  const q = r.poisson.quarters;
  return {
    grid: { left: 8, right: 8, top: 32, bottom: 8, containLabel: true },
    legend: { top: 0, data: [v.away.abbr, v.home.abbr] },
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    color: [v.colors.away, v.colors.home],
    xAxis: { type: 'category', data: q.map((x) => `Q${x.q}`) },
    yAxis: { type: 'value', name: 'Points' },
    series: [
      { type: 'bar', name: v.away.abbr, data: q.map((x) => x.away), label: { show: true, position: 'top' } },
      { type: 'bar', name: v.home.abbr, data: q.map((x) => x.home), label: { show: true, position: 'top' } },
    ],
  };
}

const OPTIONS = {
  ladder: (r, v) => ladderOption(r, v),
  dist: (r, v, page) => distOption(r, v, page.distTab),
  radar: (r, v) => radarOption(r, v),
  shap: (r, v) => shapOption(r, v),
  poisson: (r, v) => poissonOption(r, v),
};

// ── Alpine components ────────────────────────────────────────────────────

function waitForStatus(Alpine) {
  if (Alpine.store('status').data) return Promise.resolve();
  return new Promise((resolve) => {
    document.addEventListener('nq:status', () => resolve(), { once: true });
    setTimeout(resolve, 4000);               // fall back to the browser's ET date
  });
}

export default function register(Alpine) {
  Alpine.data('gameChart', (kind) => {
    let chart = null;
    let ro = null;
    return {
      init() {
        const el = this.$refs.chart;
        const render = () => {
          if (!this.result || !this.v || !el.isConnected) return;
          if (!el.clientWidth || !el.clientHeight) return;          // hidden: mount on reveal
          const opt = OPTIONS[kind](this.result, this.v, this);
          if (!opt) return;
          if (chart) chart.update(opt);
          else chart = charts.mount(el, opt);
        };
        ro = new ResizeObserver(() => { if (!chart) render(); });
        ro.observe(el);
        this.$watch('v', render);
        this.$watch('$store.prefs.theme', () => setTimeout(render, 0));   // re-resolve CSS colours
        if (kind === 'dist') this.$watch('distTab', render);
        this.$nextTick(render);
      },
      destroy() {
        ro?.disconnect();
        chart?.dispose();
        chart = null;
      },
    };
  });

  Alpine.data('gamePage', () => ({
    // form
    homeId: '',
    awayId: '',
    date: '',
    injMode: 'live',                         // live | none | custom
    custom: { home: [], away: [] },
    customFor: '',                           // "homeId-awayId" the custom list belongs to
    customBusy: false,
    newPlayer: { home: { name: '', status: 'out' }, away: { name: '', status: 'out' } },
    oddsText: { home: '', away: '' },
    oddsMl: { home: null, away: null },
    oddsSource: null,                        // 'market' | 'manual' | null
    oddsInfo: null,                          // {kind, text}
    oddsBusy: false,
    formError: '',
    // result
    result: null,
    v: null,
    loading: false,
    slow: false,
    error: null,
    distTab: 'margin',
    open: { why: true, dist: true, compare: true, avail: true, internals: false },
    statuses: STATUSES,
    statusLabel: STATUS_LABEL,

    async init() {
      const mobile = window.matchMedia('(max-width: 767.98px)').matches;
      if (mobile) this.open = { why: true, dist: false, compare: false, avail: false, internals: false };
      this.$watch('$store.prefs.odds', () => this.refreshView());
      this.$watch('$store.teams.list', () => this.refreshView());

      const q = new URLSearchParams(window.location.search);
      await waitForStatus(Alpine);
      this.date = q.get('date') || this.$store.status.todayEt;
      if (q.get('home_id')) this.homeId = q.get('home_id');
      if (q.get('away_id')) this.awayId = q.get('away_id');
      const hml = Number(q.get('home_ml')); const aml = Number(q.get('away_ml'));
      if (q.get('home_ml') && q.get('away_ml') && Number.isFinite(hml) && Number.isFinite(aml)) {
        this.oddsMl = { home: hml, away: aml };
        this.oddsSource = 'manual';
        this.renderOddsText();
      }
      const inj = q.get('injuries');
      if (inj === 'none' || inj === 'live') this.injMode = inj;
      else if (inj) {
        try {
          const c = JSON.parse(inj);
          this.custom = { home: c.home || [], away: c.away || [] };
          this.injMode = 'custom';
          this.customFor = this.matchKey;
        } catch { /* ignore a malformed link */ }
      }
      if (this.homeId && this.awayId) this.analyze();
    },

    // ── form helpers ──
    get teamList() { return this.$store.teams.list; },
    get homeTeam() { return this.$store.teams.get(Number(this.homeId)); },
    get awayTeam() { return this.$store.teams.get(Number(this.awayId)); },
    get sameTeam() { return this.homeId && this.homeId === this.awayId; },
    get matchKey() { return `${this.homeId}-${this.awayId}`; },
    get oddsMode() { return this.$store.prefs.odds; },
    get oddsPlaceholder() { return this.oddsMode === 'dec' ? 'e.g. 1.91' : 'e.g. −110'; },

    swap() {
      const customOk = this.customFor === this.matchKey;
      [this.homeId, this.awayId] = [this.awayId, this.homeId];
      [this.oddsMl.home, this.oddsMl.away] = [this.oddsMl.away, this.oddsMl.home];
      this.custom = { home: this.custom.away, away: this.custom.home };
      if (customOk) this.customFor = this.matchKey;
      this.renderOddsText();
      this.oddsInfo = null;
    },

    teamsChanged() {
      // a different matchup invalidates fetched market odds and the custom list
      if (this.oddsSource === 'market') this.clearOdds();
      this.oddsInfo = null;
      this.formError = '';
    },

    renderOddsText() {
      this.oddsText = { home: oddsText(this.oddsMl.home, this.oddsMode), away: oddsText(this.oddsMl.away, this.oddsMode) };
    },

    oddsInput(side) {
      const ml = parseOdds(this.oddsText[side], this.oddsMode);
      this.oddsMl[side] = Number.isNaN(ml) ? NaN : ml;
      this.oddsSource = this.oddsMl.home == null && this.oddsMl.away == null ? null : 'manual';
      this.oddsInfo = null;
    },

    oddsInvalid(side) { return Number.isNaN(this.oddsMl[side]); },

    clearOdds() {
      this.oddsMl = { home: null, away: null };
      this.oddsText = { home: '', away: '' };
      this.oddsSource = null;
    },

    async fetchOdds() {
      if (!this.validate()) return;
      this.oddsBusy = true;
      this.oddsInfo = null;
      try {
        const o = await api.get('/api/odds', { home_id: this.homeId, away_id: this.awayId, date: this.date });
        if (o.status === 'ok') {
          this.oddsMl = { home: o.home_ml, away: o.away_ml };
          this.oddsSource = 'market';
          this.renderOddsText();
          this.oddsInfo = { kind: o.stale ? 'warning' : 'ok', text: `${o.source} · ${ago(o.fetched_at)}${o.stale ? ' (cached, offline)' : ''}` };
        } else if (o.status === 'no_odds_yet') {
          this.oddsInfo = { kind: 'muted', text: 'Game found, but no odds are posted yet. Try closer to tip-off.' };
        } else {
          this.oddsInfo = { kind: 'warning', text: `No ${this.awayTeam?.abbr} @ ${this.homeTeam?.abbr} game on the schedule for ${day(this.date)}.` };
        }
      } catch (e) {
        this.oddsInfo = { kind: 'danger', text: e.detail || e.message };
      } finally {
        this.oddsBusy = false;
      }
    },

    // ── custom injuries ──
    async setInjMode(mode) {
      this.injMode = mode;
      if (mode === 'custom' && this.customFor !== this.matchKey) await this.seedCustom();
      if (mode === 'custom') this.$nextTick(() => this.$refs.injEdit?.click());
    },

    async seedCustom() {
      if (!this.homeId || !this.awayId || this.sameTeam) return;
      this.customBusy = true;
      try {
        const [h, a] = await Promise.all([
          api.get(`/api/teams/${this.homeId}/availability`, { date: this.date }),
          api.get(`/api/teams/${this.awayId}/availability`, { date: this.date }),
        ]);
        const pick = (av) => av.players.map((p) => ({ name: p.name, status: p.status, detail: p.detail || '' }));
        this.custom = { home: pick(h), away: pick(a) };
        this.customFor = this.matchKey;
      } catch (e) {
        this.$store.toasts.push({ title: 'Could not load the injury report', body: e.detail || e.message, kind: 'warning' });
        this.custom = { home: [], away: [] };
        this.customFor = this.matchKey;
      } finally {
        this.customBusy = false;
      }
    },

    addPlayer(side) {
      const p = this.newPlayer[side];
      const name = p.name.trim();
      if (!name) return;
      this.custom[side] = [...this.custom[side].filter((x) => x.name !== name), { name, status: p.status, detail: '' }];
      this.newPlayer[side] = { name: '', status: 'out' };
    },

    removePlayer(side, i) { this.custom[side] = this.custom[side].filter((_, j) => j !== i); },

    // ── analyze ──
    validate() {
      this.formError = '';
      if (!this.homeId || !this.awayId) this.formError = 'Pick both teams.';
      else if (this.sameTeam) this.formError = 'Home and away must be different teams.';
      else if (!/^\d{4}-\d{2}-\d{2}$/.test(this.date || '')) this.formError = 'Pick a date.';
      return !this.formError;
    },

    requestBody() {
      const body = { home_id: Number(this.homeId), away_id: Number(this.awayId), date: this.date, include: ['hist', 'shap', 'poisson'] };
      if (this.oddsSource === 'manual') body.odds = { home_ml: this.oddsMl.home, away_ml: this.oddsMl.away };
      body.injuries = this.injMode === 'custom'
        ? { home: this.custom.home.map(({ name, status }) => ({ name, status })), away: this.custom.away.map(({ name, status }) => ({ name, status })) }
        : this.injMode;
      return body;
    },

    updateUrl(body) {
      const q = new URLSearchParams({ home_id: body.home_id, away_id: body.away_id, date: body.date });
      if (body.odds) { q.set('home_ml', body.odds.home_ml); q.set('away_ml', body.odds.away_ml); }
      if (this.injMode !== 'live') q.set('injuries', this.injMode === 'custom' ? JSON.stringify(body.injuries) : this.injMode);
      window.history.replaceState(null, '', `${window.location.pathname}?${q}`);
    },

    async analyze() {
      if (!this.validate()) return;
      if (this.oddsSource === 'manual') {
        const { home, away } = this.oddsMl;
        if (Number.isNaN(home) || Number.isNaN(away)) { this.formError = 'Odds must be American (≤ −100 or ≥ +100) or decimal (above 1.0).'; return; }
        if ((home == null) !== (away == null)) { this.formError = 'Enter both prices, or clear them to use the market.'; return; }
        if (home == null) this.oddsSource = null;
      }
      if (this.injMode === 'custom' && this.customFor !== this.matchKey) await this.seedCustom();
      const body = this.requestBody();
      this.updateUrl(body);
      this.loading = true;
      this.error = null;
      this.slow = false;
      const timer = setTimeout(() => { this.slow = true; }, 1500);
      try {
        const r = await api.post('/api/predict', body);
        this.result = r;
        this.refreshView();
        this.reportFreshness(r);
        if (!this.oddsSource && r.market && r.market.status === 'ok') {
          this.oddsMl = { home: r.market.home_ml, away: r.market.away_ml };
          this.oddsSource = 'market';
          this.renderOddsText();
          this.oddsInfo = { kind: r.market.stale ? 'warning' : 'ok', text: `${r.market.source} · ${ago(r.market.fetched_at)}` };
        }
      } catch (e) {
        this.result = null;
        this.v = null;
        this.error = e;
      } finally {
        clearTimeout(timer);
        this.loading = false;
        this.slow = false;
      }
    },

    refreshView() {
      if (!this.result) return;
      try {
        this.v = buildView(this.result, this.oddsMode, this.$store.teams.byId);
      } catch (e) {
        console.error('Could not render the prediction', e);
        this.error = { code: 'render', detail: 'Could not display this prediction.', message: String(e) };
      }
    },

    reportFreshness(r) {
      const f = this.$store.freshness;
      const stale = r.teams.home.stale || r.teams.away.stale;
      f.report('stats', 'Team stats', stale ? (r.teams.home.as_of < r.teams.away.as_of ? r.teams.home.as_of : r.teams.away.as_of) : r.teams.home.as_of, stale);
      if (r.market && r.market.status !== 'manual' && r.market.fetched_at) f.report('scoreboard', 'Scoreboard', r.market.fetched_at, r.market.stale);
      const av = r.availability.home;
      if (av.source === 'live' && av.fetched_at) f.report('injuries', 'Injuries', av.fetched_at, av.status === 'stale');
    },

    // ── result helpers ──
    get errorKind() {
      const c = this.error?.code;
      if (c === 'team_not_found' || c === 'validation' || c === 'not_found') return 'inline';
      return 'card';
    },

    toggle(key) { this.open[key] = !this.open[key]; },

    rankTip(rank) { return rank ? `Ranked ${ordinal(rank)} of 30 by Elo` : ''; },

    get betHref() {
      const r = this.result; const v = this.v;
      if (!r || !v) return '/bets';
      const side = r.pick.side;
      const m = r.market;
      const ml = m && m.home_ml != null ? (side === 'home' ? m.home_ml : m.away_ml) : null;
      const prefill = {
        date: r.game.date,
        matchup: `${v.away.abbr} @ ${v.home.abbr}`,
        pick: (side === 'home' ? v.home : v.away).abbr,
        confidence: Math.round(r.pick.prob * 100),
        odds: ml,
        prediction_ref: { date: r.game.date, home_id: r.game.home.id, away_id: r.game.away.id, prob: r.pick.prob },
      };
      return `/bets?${new URLSearchParams({ prefill: JSON.stringify(prefill) })}`;
    },

    async copyLink() {
      const url = window.location.href;
      try {
        await navigator.clipboard.writeText(url);
        this.$store.toasts.push({ title: 'Link copied', body: 'Opening it reruns this analysis.', kind: 'success', timeout: 3000 });
      } catch {
        window.prompt('Copy this link:', url);
      }
    },
  }));
}

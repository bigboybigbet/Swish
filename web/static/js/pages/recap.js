// Recap: one game in depth from ESPN's play-by-play — game flow with win
// probability, rotations, player impact, key moments, box score and news.

import { api } from '../api.js';
import { fmt } from '../format.js';
import * as charts from '../charts.js';

const css = (name, fallback) => getComputedStyle(document.body).getPropertyValue(name).trim() || fallback;
const REG = 2880;   // regulation seconds

function periodLabel(p) { return p <= 4 ? `Q${p}` : `OT${p > 5 ? p - 4 : ''}`; }
function gameClock(t) {
  const p = t < REG ? Math.floor(t / 720) + 1 : 5 + Math.floor((t - REG) / 300);
  const left = t < REG ? 720 - (t % 720) : 300 - ((t - REG) % 300);
  return `${periodLabel(p)} ${Math.floor(left / 60)}:${String(Math.floor(left % 60)).padStart(2, '0')}`;
}
function periodTicks(periods) {
  const out = [];
  for (let p = 1; p <= periods; p += 1) out.push(p <= 4 ? (p - 1) * 720 : REG + (p - 5) * 300);
  return out;
}
const endOfGame = (periods) => (periods <= 4 ? REG : REG + (periods - 4) * 300);

const OPTIONS = {
  flow(r, v) {
    const home = v.colors.home, away = v.colors.away;
    const pts = r.flow;
    const end = endOfGame(r.periods);
    return {
      grid: { left: 48, right: 52, top: 36, bottom: 36 },
      legend: { top: 0, data: [`${r.home.abbr} ahead`, `${r.away.abbr} ahead`, `${r.home.abbr} win probability`] },
      tooltip: {
        trigger: 'axis', axisPointer: { type: 'line' },
        formatter: (ps) => {
          const f = pts[ps[0].dataIndex];
          return `<b>${gameClock(f.t)}</b> · ${r.away.abbr} ${f.away} – ${f.home} ${r.home.abbr}<br>`
            + `${r.home.abbr} win chance ${f.wp == null ? '–' : fmt.pct(f.wp)}<br><span style="opacity:.75">${f.text || ''}</span>`;
        },
      },
      xAxis: { type: 'value', min: 0, max: end, interval: 720, axisLabel: { formatter: (t) => (t >= end ? 'End' : periodLabel(t < REG ? t / 720 + 1 : 5 + (t - REG) / 300)) },
               splitLine: { show: true, lineStyle: { type: 'dashed', opacity: 0.4 } } },
      yAxis: [
        { type: 'value', name: 'Lead', axisLabel: { formatter: (m) => (m === 0 ? 'Tie' : `${m > 0 ? r.home.abbr : r.away.abbr} +${Math.abs(m)}`) } },
        { type: 'value', min: 0, max: 1, position: 'right', axisLabel: { formatter: (p) => `${Math.round(p * 100)}%` }, splitLine: { show: false } },
      ],
      series: [
        { name: `${r.home.abbr} ahead`, type: 'line', step: 'end', symbol: 'none', data: pts.map((f) => [f.t, Math.max(f.margin, 0)]),
          lineStyle: { width: 1.5, color: home }, itemStyle: { color: home }, areaStyle: { color: home, opacity: 0.22 } },
        { name: `${r.away.abbr} ahead`, type: 'line', step: 'end', symbol: 'none', data: pts.map((f) => [f.t, Math.min(f.margin, 0)]),
          lineStyle: { width: 1.5, color: away }, itemStyle: { color: away }, areaStyle: { color: away, opacity: 0.22 } },
        { name: 'Big swings', type: 'scatter', symbolSize: 9, data: r.moments.swings.map((m) => [m.t, m.home - m.away]),
          itemStyle: { color: css('--tblr-orange', '#f97316') }, tooltip: { show: false }, z: 5 },
        { name: `${r.home.abbr} win probability`, type: 'line', yAxisIndex: 1, symbol: 'none', smooth: false,
          data: pts.filter((f) => f.wp != null).map((f) => [f.t, f.wp]),
          lineStyle: { width: 2, color: css('--tblr-body-color', '#334155'), opacity: 0.7 } },
      ],
    };
  },
  rotation(r, v) {
    const side = v.rotationSide;
    const team = r[side];
    const color = v.colors[side];
    const ps = r.players[side].filter((p) => !p.dnp && p.stints.length)
      .sort((a, b) => (b.starter - a.starter) || (b.min - a.min));
    const names = ps.map((p) => p.short || p.name);
    const data = [];
    ps.forEach((p, i) => p.stints.forEach(([a, b]) => data.push({ value: [i, a, b, p.pm], itemStyle: { color } })));
    const end = endOfGame(r.periods);
    return {
      grid: { left: 110, right: 16, top: 12, bottom: 32 },
      tooltip: { formatter: (x) => `<b>${names[x.value[0]]}</b><br>${gameClock(x.value[1])} → ${gameClock(Math.min(x.value[2], end - 0.1))}<br>${fmt.num((x.value[2] - x.value[1]) / 60)} min` },
      xAxis: { type: 'value', min: 0, max: end, interval: 720, axisLabel: { formatter: (t) => (t >= end ? 'End' : periodLabel(t < REG ? t / 720 + 1 : 5 + (t - REG) / 300)) },
               splitLine: { show: true, lineStyle: { type: 'dashed', opacity: 0.4 } } },
      yAxis: { type: 'category', data: names, inverse: true, axisTick: { show: false } },
      series: [{
        type: 'custom', encode: { x: [1, 2], y: 0 }, data,
        renderItem: (params, api) => {
          const y = api.value(0); const s = api.coord([api.value(1), y]); const e = api.coord([api.value(2), y]);
          const h = api.size([0, 1])[1] * 0.6;
          return { type: 'rect', shape: { x: s[0], y: s[1] - h / 2, width: Math.max(1, e[0] - s[0]), height: h, r: 3 },
                   style: api.style() };
        },
      }],
      title: { show: false, text: `${team.abbr} rotation` },
    };
  },
  impact(r, v) {
    const rows = ['away', 'home'].flatMap((s) => r.players[s].filter((p) => !p.dnp).map((p) => ({ ...p, side: s })))
      .sort((a, b) => b.wpa - a.wpa).filter((p, i, all) => i < 8 || i >= all.length - 4);
    return {
      grid: { left: 120, right: 40, top: 8, bottom: 28 },
      tooltip: { formatter: (x) => { const p = rows[x.dataIndex]; return `<b>${p.name}</b> (${r[p.side].abbr})<br>Win chance added ${fmt.signed(p.wpa * 100, 1)} pts<br>${p.pts} pts · ${p.reb} reb · ${p.ast} ast · Game Score ${p.game_score}`; } },
      xAxis: { type: 'value', axisLabel: { formatter: (x) => `${x > 0 ? '+' : ''}${Math.round(x * 100)}` } },
      yAxis: { type: 'category', inverse: true, data: rows.map((p) => `${p.short || p.name} · ${r[p.side].abbr}`), axisTick: { show: false } },
      series: [{ type: 'bar', data: rows.map((p) => ({ value: p.wpa, itemStyle: { color: v.colors[p.side], opacity: p.wpa >= 0 ? 1 : 0.5 } })), barMaxWidth: 16 }],
    };
  },
};

export default function register(Alpine) {
  Alpine.data('recapChart', (kind) => {
    let chart = null;
    return {
      init() {
        const render = () => {
          if (!this.r || !this.$el.isConnected) return;
          const opt = OPTIONS[kind](this.r, this);
          if (chart) chart.update(opt); else chart = charts.mount(this.$el, opt);
        };
        this.$watch('r', render);
        if (kind === 'rotation') this.$watch('rotationSide', render);
        this.$watch('$store.prefs.theme', () => setTimeout(render, 0));
        this.$nextTick(render);
      },
      destroy() { chart?.dispose(); chart = null; },
    };
  });

  Alpine.data('recapPage', () => ({
    r: null,
    error: null,
    rotationSide: 'home',
    boxSide: 'home',
    momentFilter: 'all',
    timer: null,

    init() {
      this.eventId = new URLSearchParams(location.search).get('event');
      if (!this.eventId) { this.error = { message: 'No game selected. Open a game from the Slate.' }; return; }
      this.load();
      // live games: refresh every 30 s (the server caches ESPN for 30 s)
      this.timer = setInterval(() => { if (this.r && !this.r.status.completed && !document.hidden) this.load(); }, 30_000);
    },
    destroy() { clearInterval(this.timer); },

    async load() {
      try {
        const r = await api.get(`/api/recap/${encodeURIComponent(this.eventId)}`);
        for (const s of ['home', 'away']) r[s].id = r[s].id || null;
        this.r = r;
        document.title = `${r.away.abbr} @ ${r.home.abbr} · Swish`;
      } catch (e) {
        this.error = e;
      }
    },

    get colors() {
      const store = Alpine.store('teams').byId;
      const pick = (t, fallback) => store[t?.id]?.colors?.primary || t?.color || fallback;
      let home = pick(this.r?.home, '#206bc4');
      let away = pick(this.r?.away, '#d63939');
      if (home.toLowerCase() === away.toLowerCase()) away = store[this.r?.away?.id]?.colors?.secondary || '#f59f00';
      return { home, away };
    },
    get winner() { return this.r ? (this.r.home.score >= this.r.away.score ? 'home' : 'away') : null; },
    get quarters() { return this.r ? this.r.home.linescores.map((_, i) => i + 1) : []; },

    get moments() {
      if (!this.r) return [];
      const m = this.r.moments;
      const out = [
        ...m.swings.map((x) => ({ ...x, kind: 'swing', title: `Win chance ${x.delta > 0 ? this.r.home.abbr : this.r.away.abbr} ${fmt.signed(Math.abs(x.delta) * 100, 1)} pts`, detail: x.text })),
        ...m.runs.map((x) => ({ ...x.to, kind: 'run', title: `${this.r[x.side].abbr} ${x.points}-0 run`, detail: `From ${periodLabel(x.from.period)} ${x.from.clock} to ${periodLabel(x.to.period)} ${x.to.clock}` })),
        ...m.clutch.map((x) => ({ ...x, kind: 'clutch', title: 'Clutch basket', detail: x.text })),
      ];
      return out.filter((x) => this.momentFilter === 'all' || x.kind === this.momentFilter).sort((a, b) => a.t - b.t);
    },
    momentIcon(k) { return { swing: 'ti-bolt', run: 'ti-flame', clutch: 'ti-clock-exclamation' }[k]; },
    periodLabel,

    teamStat(key) { return this.r?.team_stats?.[this.side]?.[key]; },
    statRows() {
      const ts = this.r?.team_stats || {};
      const rows = [
        ['Field goals', 'fieldGoalsMade-fieldGoalsAttempted', 'fieldGoalPct'],
        ['Three pointers', 'threePointFieldGoalsMade-threePointFieldGoalsAttempted', 'threePointFieldGoalPct'],
        ['Free throws', 'freeThrowsMade-freeThrowsAttempted', 'freeThrowPct'],
        ['Rebounds', 'totalRebounds'], ['Offensive rebounds', 'offensiveRebounds'], ['Assists', 'assists'],
        ['Turnovers', 'totalTurnovers', null, true], ['Steals', 'steals'], ['Blocks', 'blocks'],
        ['Points in the paint', 'pointsInPaint'], ['Fast break points', 'fastBreakPoints'], ['Points off turnovers', 'turnoverPoints'],
      ];
      return rows.filter(([, k]) => ts.home?.[k] != null).map(([label, k, pct, lowerBetter]) => {
        const val = (s) => parseFloat(pct ? ts[s][pct] : ts[s][k]) || 0;
        const a = val('away'), h = val('home');
        const show = (s) => (pct ? `${ts[s][k]} (${ts[s][pct]}%)` : ts[s][k]);
        const total = a + h || 1;
        const better = a === h ? null : ((h > a) !== !!lowerBetter ? 'home' : 'away');
        return { label, away: show('away'), home: show('home'), awayPct: (a / total) * 100, homePct: (h / total) * 100, better };
      });
    },
    get box() {
      return (this.r?.players[this.boxSide] || []).slice().sort((a, b) => (a.dnp - b.dnp) || (b.starter - a.starter) || ((b.min || 0) - (a.min || 0)));
    },
    tipoff() { return this.r?.date ? fmt.day(this.r.date, true) : ''; },
  }));
}

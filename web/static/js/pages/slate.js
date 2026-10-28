// Slate: the day's games as cards. The schedule draws at once; "Run predictions"
// starts the scan job and each card fills in as its `item` event arrives. The
// latest saved scan for the date is merged in on load.

import { api } from '../api.js';
import { fmt } from '../format.js';

const FILTERS = ['all', 'vip', 'value', 'conflict'];
const POLL_MS = 60_000;                                    // the server caches live scoreboards for 30 s
const RATING_JOBS = ['ratings', 'train', 'autobacktest'];

export default function register(Alpine) {
  Alpine.data('slatePage', () => ({
    date: null,
    games: [],
    cards: {},           // event_id → scan card
    asOf: null,
    stale: false,
    scanAt: null,
    loading: true,
    error: null,
    jobId: null,
    filter: 'all',
    sort: 'tipoff',
    filters: FILTERS,
    polling: false,
    refreshedAt: null,
    ratingsUpdated: false,

    init() {
      const fromUrl = new URLSearchParams(location.search).get('date');
      const start = () => { this.date = fromUrl || Alpine.store('status').todayEt; this.load(); };
      // "today" is the server's ET date (the replayed date in demo mode): wait for /api/status
      if (fromUrl || Alpine.store('status').data) start();
      else document.addEventListener('nq:status', start, { once: true });
      document.addEventListener('nq:job', (e) => this.onJob(e.detail));
      document.addEventListener('visibilitychange', () => { if (!document.hidden && this.polling) this.poll(); });
      setInterval(() => { if (this.polling && !document.hidden) this.poll(); }, POLL_MS);
      window.addEventListener('keydown', (e) => {
        if (e.target.closest('input, select, textarea')) return;
        if (e.key === 'ArrowLeft') this.shift(-1);
        if (e.key === 'ArrowRight') this.shift(1);
      });
      // a scan started before a reload keeps streaming into this page
      document.addEventListener('nq:status', () => {
        const running = Alpine.store('jobs').active.find((j) => j.kind === 'scan');
        if (running && !this.jobId) this.jobId = running.id;
      });
    },

    async load() {
      this.loading = true;
      this.error = null;
      this.cards = {};
      this.scanAt = null;
      try {
        const s = await api.get('/api/schedule', { date: this.date });
        this.games = s.games;
        this.asOf = s.as_of;
        this.stale = s.stale;
        this.refreshedAt = new Date().toISOString();
        this.updatePolling();
      } catch (e) {
        this.games = [];
        this.error = e.message || 'Could not load the schedule.';
      } finally {
        this.loading = false;
      }
      try {
        const scan = await api.get('/api/scans', { date: this.date });
        for (const c of scan.games) this.cards[c.event_id] = c;
        this.scanAt = scan.created_at;
      } catch { /* no scan yet for this date */ }
    },

    /** Poll while the date is today or yesterday (ET) and a game is not final yet. */
    updatePolling() {
      const today = Alpine.store('status').todayEt;
      const recent = this.date >= fmt.addDays(today, -1) && this.date <= today;
      this.polling = recent && this.games.some((g) => !g.completed);
    },
    get liveCount() { return this.games.filter((g) => g.status === 'in').length; },

    /** Refresh scores and status in place; predictions stay as they are. */
    async poll() {
      try {
        const s = await api.get('/api/schedule', { date: this.date });
        if (s.date !== this.date) return;
        const byId = Object.fromEntries(s.games.map((g) => [g.event_id, g]));
        this.games = this.games.map((g) => (byId[g.event_id] ? { ...g, ...byId[g.event_id] } : g));
        this.asOf = s.as_of;
        this.stale = s.stale;
        this.refreshedAt = new Date().toISOString();
        this.updatePolling();
      } catch { /* keep the last scores; the next poll retries */ }
    },

    shift(n) { this.go(fmt.addDays(this.date, n)); },
    go(d) {
      this.date = d;
      history.replaceState(null, '', d === Alpine.store('status').todayEt ? '/' : `/?date=${d}`);
      this.load();
    },
    get isToday() { return this.date === Alpine.store('status').todayEt; },

    async run() {
      try {
        const job = await Alpine.store('jobs').start('scan', { date: this.date });
        this.jobId = job.id;
      } catch (e) {
        Alpine.store('toasts')?.push?.({ kind: 'danger', title: 'Scan failed to start', body: e.message });
      }
    },
    get job() { return this.jobId ? Alpine.store('jobs').get(this.jobId) : null; },

    onJob({ id, kind, type, data }) {
      if (RATING_JOBS.includes(kind) && type === 'done' && data.state === 'done') this.ratingsUpdated = true;
      if (kind !== 'scan') return;
      if (!this.jobId) this.jobId = id;
      if (id !== this.jobId) return;
      if (type === 'item' && data?.data?.event_id) this.cards[data.data.event_id] = data.data;
      if (type === 'done') {
        this.jobId = null;
        this.ratingsUpdated = false;
        if (data.state === 'done') this.scanAt = new Date().toISOString();
      }
    },

    // ── view model ──────────────────────────────────────────────────────

    view(g) {
      const c = this.cards[g.event_id] || null;
      const pHome = c?.win_prob?.home ?? null;
      const final = g.completed && g.home.score != null && g.away.score != null;
      const homeWon = final ? g.home.score > g.away.score : null;
      const pickedHome = c?.pick ? c.pick.side === 'home' : null;
      const edge = c?.market?.value_side ? c.market.edge?.[c.market.value_side] ?? null : null;
      return {
        g, c, pHome, final,
        correct: final && pickedHome != null ? pickedHome === homeWon : null,
        vip: !!c?.flags?.vip,
        value: !!c?.flags?.value,
        conflict: !!c?.flags?.conflict,
        valueTeam: c?.market?.value_side ? g[c.market.value_side] : null,
        edge,
        favourite: pHome == null ? null : (pHome >= 0.5 ? g.home : g.away),
      };
    },

    get rows() {
      let rows = this.games.map((g) => this.view(g));
      if (this.filter !== 'all') rows = rows.filter((r) => r[this.filter]);
      const conf = (r) => (r.pHome == null ? -1 : Math.max(r.pHome, 1 - r.pHome));
      const by = {
        tipoff: (a, b) => (a.g.tipoff_utc || '').localeCompare(b.g.tipoff_utc || ''),
        confidence: (a, b) => conf(b) - conf(a),
        edge: (a, b) => (b.edge ?? -1) - (a.edge ?? -1),
      }[this.sort];
      return rows.sort(by);
    },

    get kpi() {
      const all = this.games.map((g) => this.view(g));
      const graded = all.filter((r) => r.correct != null);
      return {
        games: all.length,
        predicted: all.filter((r) => r.c && !r.c.error).length,
        vip: all.filter((r) => r.vip).length,
        value: all.filter((r) => r.value).length,
        correct: graded.filter((r) => r.correct).length,
        graded: graded.length,
      };
    },

    count(f) { return f === 'all' ? this.games.length : this.games.map((g) => this.view(g)).filter((r) => r[f]).length; },

    spreadText(r) {
      if (r.c?.home_spread == null) return fmt.DASH;
      return fmt.spread(r.g.home.abbr, r.g.away.abbr, r.c.home_spread);
    },
    marketSpread(r) {
      const s = r.c?.market?.spread ?? r.g.market?.spread;
      return s == null ? null : fmt.spread(r.g.home.abbr, r.g.away.abbr, s);
    },
    /** "07:00 Sat 16 Mar · 19:00 ET": local time first (the user may be in Asia). */
    statusText(r) {
      if (r.final) return 'Final';
      if (r.g.status === 'in') return r.g.period > 4 ? `OT${r.g.period > 5 ? r.g.period - 4 : ''}` : `Q${r.g.period || 1}`;
      return this.tipoffText(r.g);
    },
    tipoffText(g) {
      const t = fmt.tipoff(g.tipoff_utc);
      return t.local === fmt.DASH ? 'Time TBD' : `${t.local} ${t.localDay} · ${t.et}`;
    },
    /** Whole percentages that add up to 100. */
    probText(r, side) {
      if (r.pHome == null) return fmt.DASH;
      const home = Math.round(r.pHome * 100);
      return `${side === 'home' ? home : 100 - home}%`;
    },
    gameHref(g) {
      return `/game?home_id=${g.home.id}&away_id=${g.away.id}&date=${this.date}`;
    },
  }));
}

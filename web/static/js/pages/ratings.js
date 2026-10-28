// Ratings: Kalman-Elo team table (rating ± sd, 7-day change, season path as an
// inline SVG sparkline) and the player impact weights used for injuries.

import { api } from '../api.js';
import { fmt } from '../format.js';

/** SVG path for a sparkline of [date, value] points in a w×h box. */
function sparkPath(points, w = 110, h = 28) {
  if (!points || points.length < 2) return '';
  const ys = points.map((p) => p[1]);
  const lo = Math.min(...ys), hi = Math.max(...ys);
  const span = hi - lo || 1;
  return points.map((p, i) => {
    const x = (i / (points.length - 1)) * (w - 2) + 1;
    const y = h - 1 - ((p[1] - lo) / span) * (h - 2);
    return `${i ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
}

export default function register(Alpine) {
  Alpine.data('ratingsPage', () => ({
    tab: 'teams',
    teams: null,
    players: null,
    error: null,
    query: '',
    maxAbs: 1,
    jobId: null,

    async init() {
      const hash = location.hash.replace('#', '');
      if (hash === 'players') this.tab = 'players';
      await this.loadTeams();
      this.$watch('tab', (t) => {
        history.replaceState(null, '', t === 'players' ? '#players' : location.pathname);
        if (t === 'players' && !this.players) this.loadPlayers();
      });
      if (this.tab === 'players') this.loadPlayers();
      document.addEventListener('nq:job', ({ detail: d }) => {
        if (d.kind === 'ratings' && !this.jobId) this.jobId = d.id;
        if (['ratings', 'train', 'autobacktest'].includes(d.kind) && d.type === 'done') {
          if (d.id === this.jobId) this.jobId = null;
          if (d.data.state === 'done') this.loadTeams();
        }
      });
    },

    async update() {
      try {
        const job = await Alpine.store('jobs').start('ratings', {});
        this.jobId = job.id;
      } catch (e) {
        Alpine.store('toasts')?.push?.({ kind: 'danger', title: 'Could not start the update', body: e.message });
      }
    },
    get job() { return this.jobId ? Alpine.store('jobs').get(this.jobId) : null; },

    async loadTeams() {
      this.error = null;
      try {
        const data = await api.get('/api/ratings/teams');
        for (const t of data.teams) t.id = t.team_id;          // team_badge / $teamStyle read .id
        this.teams = data;
        this.maxAbs = Math.max(1, ...this.teams.teams.map((t) => Math.abs(t.rating)));
      } catch (e) {
        this.error = e;
      }
    },
    async loadPlayers() {
      try { this.players = await api.get('/api/ratings/players'); } catch (e) { this.error = e; }
    },

    get filteredPlayers() {
      const q = this.query.trim().toLowerCase();
      const rows = this.players?.players || [];
      return q ? rows.filter((p) => p.name.toLowerCase().includes(q)) : rows;
    },

    spark: sparkPath,
    /** Bar from the centre: positive ratings to the right, negative to the left. */
    barStyle(r) {
      const pct = (Math.abs(r) / this.maxAbs) * 50;
      return r >= 0 ? `left:50%;width:${pct}%` : `left:${50 - pct}%;width:${pct}%`;
    },
    changeText(c) { return c == null ? fmt.DASH : (Math.abs(c) < 0.05 ? '±0.0' : fmt.signed(c)); },
  }));
}

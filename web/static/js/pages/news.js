// News: ESPN's NBA feed tagged for decisions. Articles that change who plays
// are flagged, and teams with a game today are marked so the news that can
// still move a prediction comes first.

import { api } from '../api.js';

const TAGS = [
  ['all', 'All'], ['availability', 'Affects who plays'], ['injury', 'Injuries'], ['rest', 'Rest'],
  ['roster', 'Roster moves'], ['streak', 'Streaks'],
];
const TAG_LABEL = { injury: 'Injury', rest: 'Rest', suspension: 'Suspension', roster: 'Roster', streak: 'Streak' };

export default function register(Alpine) {
  Alpine.data('newsPage', () => ({
    tags: TAGS,
    only: 'all',
    teamId: '',
    data: null,
    error: null,
    playingToday: {},     // team id → "AWAY @ HOME" for today's games
    todayFirst: true,

    init() {
      const q = new URLSearchParams(location.search);
      this.only = q.get('only') || 'all';
      this.teamId = q.get('team_id') || '';
      this.load();
      this.loadToday();
      this.$watch('only', () => this.sync());
      this.$watch('teamId', () => this.sync());
      setInterval(() => { if (!document.hidden) this.load(); }, 10 * 60_000);
    },
    sync() {
      const q = new URLSearchParams();
      if (this.only !== 'all') q.set('only', this.only);
      if (this.teamId) q.set('team_id', this.teamId);
      history.replaceState(null, '', q.toString() ? `/news?${q}` : '/news');
      this.load();
    },
    async load() {
      this.error = null;
      try {
        this.data = await api.get('/api/news', {
          only: this.only === 'all' ? null : this.only, team_id: this.teamId || null, limit: 60,
        });
      } catch (e) {
        this.error = e;
      }
    },
    async loadToday() {
      try {
        const s = await api.get('/api/schedule');
        const map = {};
        for (const g of s.games) {
          if (g.completed) continue;
          const label = `${g.away.abbr} @ ${g.home.abbr}`;
          if (g.home.id) map[g.home.id] = label;
          if (g.away.id) map[g.away.id] = label;
        }
        this.playingToday = map;
      } catch { /* schedule offline: no "plays today" marks */ }
    },
    todayGame(a) { return a.teams.map((t) => this.playingToday[t.id]).find(Boolean) || null; },
    get articles() {
      const list = this.data?.articles || [];
      if (!this.todayFirst) return list;
      // news about tonight's teams that changes who plays goes first
      const score = (a) => (this.todayGame(a) && a.availability ? 2 : (this.todayGame(a) ? 1 : 0));
      return list.slice().sort((a, b) => score(b) - score(a));
    },
    get teams() { return Alpine.store('teams').list || []; },
    tagLabel(t) { return TAG_LABEL[t] || t; },
  }));
}

// App entry: registers the shell stores and magics, loads the page module, starts Alpine.
//
// Page modules: a template sets {% block page_module %}/static/js/pages/<page>.js{% endblock %};
// the module's default export receives Alpine before start:
//     export default function register(Alpine) { Alpine.data('gamePage', () => ({ ... })); }

import Alpine from 'alpinejs';
import persist from '@alpinejs/persist';
import { api } from './api.js';
import { fmt } from './format.js';
import { loadTeams, badgeStyle, pairColors } from './teams.js';
import { registerJobs } from './jobs.js';

Alpine.plugin(persist);

function storageGet(store, key) {
  try { return window[store].getItem(key); } catch { return null; }
}
function storageSet(store, key, value) {
  try { window[store].setItem(key, value); } catch { /* private mode */ }
}
const themeNow = () => (document.documentElement.getAttribute('data-bs-theme') === 'dark' ? 'dark' : 'light');

// ── stores ───────────────────────────────────────────────────────────────

Alpine.store('status', {
  data: null,        // /api/status payload
  error: null,
  loading: false,

  async load() {
    this.loading = true;
    try {
      this.data = await api.get('/api/status');
      this.error = null;
      Alpine.store('jobs').sync(this.data.active_jobs);
      document.dispatchEvent(new CustomEvent('nq:status', { detail: this.data }));
    } catch (e) {
      this.error = e;
    } finally {
      this.loading = false;
    }
  },

  /** 'ok' | 'old' (> 7 days) | 'none' (Elo-only mode) | null (not loaded) */
  get modelState() {
    const m = this.data?.model;
    if (!m) return null;
    if (!m.exists) return 'none';
    return m.age_days != null && m.age_days > 7 ? 'old' : 'ok';
  },
  get modelTitle() {
    const m = this.data?.model;
    if (!m) return 'Model status loading';
    if (!m.exists) return 'No trained model: Elo + Monte Carlo only';
    const age = m.age_days != null ? `${fmt.num(m.age_days, 0)} days ago` : 'date unknown';
    return `Model ${m.version} trained ${age}`;
  },
  get demoDate() { return this.data?.demo?.date || null; },
  /** Today's ET date as the server sees it (the demo date in demo mode). */
  get todayEt() { return this.data?.today_et || fmt.todayEt(); },
});

Alpine.store('prefs', {
  odds: Alpine.$persist('us').as('nbaq-odds-format'),     // 'us' | 'dec'
  theme: themeNow(),

  setOdds(v) { this.odds = v === 'dec' ? 'dec' : 'us'; },
  toggleTheme() {
    const next = themeNow() === 'dark' ? 'light' : 'dark';
    storageSet('localStorage', 'tabler-theme', next);      // read by tabler-theme.min.js
    document.documentElement.setAttribute('data-bs-theme', next);
    this.theme = next;
  },
});

Alpine.store('toasts', {
  items: [],
  seq: 0,
  /** push({title, body, kind: 'success'|'danger'|'info'|'warning', href, linkText, timeout}) */
  push({ title, body = '', kind = 'info', href = null, linkText = 'View', timeout = 6000 } = {}) {
    const id = ++this.seq;
    this.items.push({ id, title, body, kind, href, linkText });
    if (timeout) setTimeout(() => this.dismiss(id), timeout);
    return id;
  },
  dismiss(id) { this.items = this.items.filter((t) => t.id !== id); },
});

// Data freshness for the footer and the offline banner. Pages report each
// source they load: report('scoreboard', 'Scoreboard', as_of, stale).
Alpine.store('freshness', {
  sources: {},
  report(key, label, asOf, stale = false) {
    this.sources = { ...this.sources, [key]: { label, as_of: asOf, stale: !!stale } };
  },
  clear(key) {
    const { [key]: _drop, ...rest } = this.sources;
    this.sources = rest;
  },
  get list() { return Object.values(this.sources); },
  get stale() { return this.list.find((s) => s.stale) || null; },
});

Alpine.store('banners', {
  dismissed: (() => {
    try { return JSON.parse(storageGet('sessionStorage', 'nbaq-banners') || '{}'); } catch { return {}; }
  })(),
  isDismissed(key) { return !!this.dismissed[key]; },
  dismiss(key) {
    this.dismissed = { ...this.dismissed, [key]: true };
    storageSet('sessionStorage', 'nbaq-banners', JSON.stringify(this.dismissed));
  },
});

Alpine.store('teams', {
  list: [],
  byId: {},
  error: null,
  async load() {
    try {
      const list = await loadTeams();
      this.list = list;
      this.byId = Object.fromEntries(list.map((t) => [t.id, t]));
    } catch (e) {
      this.error = e;
    }
  },
  get(id) { return this.byId[id] || null; },
});

Alpine.store('glossary', {
  items: {},
  async load() {
    try {
      const res = await fetch('/static/data/glossary.json');
      if (res.ok) this.items = await res.json();
    } catch { /* tooltips are optional */ }
  },
  text(key) { return this.items[key] || ''; },
});

registerJobs(Alpine);

// ── magics: $fmt.pct(p), $teamStyle(team), $teamColors(home, away) ───────

Alpine.magic('fmt', () => fmt);
Alpine.magic('teamStyle', () => (team) => badgeStyle(Alpine.store('teams').byId[team?.id] || team));
Alpine.magic('teamColors', () => (home, away) => {
  const byId = Alpine.store('teams').byId;
  return pairColors(byId[home?.id] || home, byId[away?.id] || away);
});

// ── page module, then start ──────────────────────────────────────────────

const pageModule = document.body.dataset.module;
if (pageModule) {
  try {
    const mod = await import(pageModule);
    mod.default?.(Alpine);
  } catch (e) {
    console.error(`Failed to load page module ${pageModule}`, e);
  }
}

window.Alpine = Alpine;
Alpine.start();

Alpine.store('status').load();
Alpine.store('teams').load();
Alpine.store('glossary').load();

setInterval(() => { if (!document.hidden) Alpine.store('status').load(); }, 60_000);
document.addEventListener('visibilitychange', () => {
  if (!document.hidden) Alpine.store('status').load();
});

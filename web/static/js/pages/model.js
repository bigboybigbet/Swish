// Model page: status cards, Train / reload / auto-backtest / player impact jobs,
// feature importance and the validation reliability curve (from /api/model, so
// they survive a refresh), ensemble weights and the job history.
//
// Running jobs come from $store.jobs (SSE); this page listens to `nq:job` to
// refresh itself when one finishes. Charts live in closures, not Alpine data.

import { api } from '../api.js';
import { fmt } from '../format.js';
import * as charts from '../charts.js';

const TERMINAL = ['done', 'error', 'cancelled', 'interrupted'];
const STAGES = ['Download', 'Pair games', 'Elo', 'Search', 'Fit', 'Calibrate', 'Save'];
const TYPICAL_TRAIN_S = 594;

const FEATURES = {
  ELO_DIFF: 'Elo difference', NET5_DIFF: 'Net rating, last 5', NET15_DIFF: 'Net rating, last 15',
  PTS10_DIFF: 'Points, last 10', PA10_DIFF: 'Points allowed, last 10', EFG10_DIFF: 'eFG%, last 10',
  TOV10_DIFF: 'Turnover rate, last 10', ORB10_DIFF: 'Off. rebound rate, last 10',
  FTR10_DIFF: 'Free-throw rate, last 10', PACE10_MEAN: 'Pace (both teams)', WPCT10_DIFF: 'Win %, last 10',
  STREAK_DIFF: 'Win streak', REST_DIFF: 'Rest days', H_B2B: 'Home back-to-back',
  A_B2B: 'Away back-to-back', H_HOME_WPCT: 'Home win %, at home', A_AWAY_WPCT: 'Away win %, on the road',
};
const STATE_BADGE = {
  done: 'bg-green-lt', error: 'bg-red-lt', cancelled: 'bg-yellow-lt', interrupted: 'bg-yellow-lt',
  running: 'bg-blue-lt', queued: 'bg-secondary-lt',
};
const STATE_LABEL = {
  done: 'Done', error: 'Failed', cancelled: 'Cancelled', interrupted: 'Interrupted', running: 'Running', queued: 'Queued',
};

/** A Tabler colour token as a concrete colour. Tabler 1.6 defines tokens with
 *  light-dark()/oklch(), which ECharts can't parse, so resolve them through an element. */
function token(name, fallback) {
  const el = document.createElement('span');
  el.style.display = 'none';
  el.style.color = `var(${name}, ${fallback})`;
  document.body.appendChild(el);
  const color = getComputedStyle(el).color;
  el.remove();
  return color || fallback;
}
const pctAxis = (v) => `${Math.round(v * 100)}%`;
const count = (n) => (n == null ? fmt.DASH : Number(n).toLocaleString('en-US'));

function importanceOption(rows) {
  const asc = [...rows].reverse();                      // category axes draw bottom-up
  const primary = token('--tblr-primary', '#066fd1');
  const ink = token('--tblr-secondary', '#667382');
  return {
    grid: { left: 16, right: 48, top: 4, bottom: 4, containLabel: true },
    tooltip: {
      trigger: 'axis', axisPointer: { type: 'shadow' },
      formatter: (ps) => `${ps[0].name}<br><b>${fmt.pct(ps[0].value, 1)}</b> of total importance`,
    },
    xAxis: { type: 'value', axisLabel: { formatter: pctAxis }, splitLine: { lineStyle: { type: 'dashed', opacity: 0.5 } } },
    yAxis: {
      type: 'category', data: asc.map((r) => FEATURES[r.feature] || r.feature),
      axisTick: { show: false }, axisLine: { show: false },
    },
    series: [{
      type: 'bar', name: 'Importance', data: asc.map((r) => r.importance), barMaxWidth: 12,
      itemStyle: { color: primary, borderRadius: [0, 4, 4, 0] },
      label: {                                           // direct labels on the top 3 only
        show: true, position: 'right', fontSize: 11, color: ink, textBorderWidth: 0,
        formatter: (p) => (p.dataIndex >= asc.length - 3 ? fmt.pct(p.value, 1) : ''),
      },
    }],
  };
}

function reliabilityOption(bins) {
  const primary = token('--tblr-primary', '#066fd1');
  const surface = token('--tblr-bg-surface', '#ffffff');
  const muted = token('--tblr-secondary', '#8a94a6');
  const maxN = Math.max(...bins.map((b) => b.n), 1);
  const size = (n) => 8 + 26 * Math.sqrt(n / maxN);     // area grows with the number of games
  const axis = (name, gap) => ({
    type: 'value', min: 0, max: 1, interval: 0.2, name, nameLocation: 'middle', nameGap: gap,
    axisLabel: { formatter: pctAxis }, splitLine: { lineStyle: { type: 'dashed', opacity: 0.5 } },
  });
  return {
    grid: { left: 56, right: 16, top: 16, bottom: 48 },
    tooltip: {
      trigger: 'item',
      formatter: (p) => {
        const b = p.data?.[3];
        if (!b) return '';
        return `Forecasts ${fmt.pct(b.lo)}–${fmt.pct(b.hi)}<br>${b.n} games<br>`
          + `Mean forecast <b>${fmt.pct(b.mean_pred, 1)}</b><br>Home won <b>${fmt.pct(b.actual, 1)}</b>`;
      },
    },
    xAxis: axis('Forecast home win probability', 30),
    yAxis: axis('Home win rate', 42),
    series: [
      {
        type: 'line', name: 'Perfect calibration', data: [[0, 0], [1, 1]], symbol: 'none', silent: true,
        lineStyle: { type: 'dashed', width: 1, color: muted }, tooltip: { show: false }, z: 1,
      },
      {
        type: 'line', name: 'Validation games', data: bins.map((b) => [b.mean_pred, b.actual, b.n, b]),
        symbol: 'circle', symbolSize: (d) => size(d[2]), lineStyle: { width: 2, color: primary },
        itemStyle: { color: primary, borderColor: surface, borderWidth: 2 }, z: 2,
      },
    ],
  };
}

export default function register(Alpine) {
  Alpine.data('modelPage', () => {
    let importanceChart = null;
    let reliabilityChart = null;
    let themeObserver = null;
    let onJob = null;
    let refreshTimer = null;

    return {
      model: null,
      ensemble: null,
      history: [],
      historyLoaded: false,
      loaded: false,
      loadError: null,
      form: { force_rebuild: false, use_optuna: false },
      starting: null,
      reloading: false,
      selected: null,
      stages: STAGES,
      isPhone: window.matchMedia('(max-width: 767.98px)').matches,

      init() {
        this.loadAll();
        onJob = (e) => {
          const { type } = e.detail;
          if (type === 'log') this.$nextTick(() => this.scrollLog());
          if (type === 'done') {
            clearTimeout(refreshTimer);
            refreshTimer = setTimeout(() => this.loadAll(), 300);
          }
        };
        document.addEventListener('nq:job', onJob);
        themeObserver = new MutationObserver(() => this.drawCharts());
        themeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ['data-bs-theme'] });
      },

      destroy() {
        document.removeEventListener('nq:job', onJob);
        themeObserver?.disconnect();
        importanceChart?.dispose();
        reliabilityChart?.dispose();
      },

      // ── data ──────────────────────────────────────────────────────────
      async loadAll() {
        await Promise.all([this.loadModel(), this.loadEnsemble(), this.loadHistory()]);
      },

      async loadModel() {
        try {
          this.model = await api.get('/api/model');
          this.loadError = null;
        } catch (e) {
          this.loadError = `Could not load the model details: ${e.message}`;
        } finally {
          this.loaded = true;
        }
        this.$nextTick(() => this.drawCharts());
      },

      async loadEnsemble() {
        try { this.ensemble = await api.get('/api/model/ensemble'); } catch { /* card shows a skeleton */ }
      },

      async loadHistory() {
        try {
          this.history = await api.get('/api/jobs', { limit: 20 });
        } catch { /* keep the old list */ } finally {
          this.historyLoaded = true;
        }
      },

      // ── charts ────────────────────────────────────────────────────────
      get importance() { return this.model?.feature_importance || []; },
      get bins() { return this.model?.calibration?.bins || []; },

      drawCharts() {
        if (this.importance.length && this.$refs.importance) {
          const opt = importanceOption(this.importance);
          if (importanceChart) importanceChart.update(opt); else importanceChart = charts.mount(this.$refs.importance, opt);
        }
        if (this.bins.length && this.$refs.reliability) {
          const opt = reliabilityOption(this.bins);
          if (reliabilityChart) reliabilityChart.update(opt); else reliabilityChart = charts.mount(this.$refs.reliability, opt);
        }
      },

      featureLabel(key) { return FEATURES[key] || key; },

      /** Blend components (Elo, MC, XGBoost); without a model, the learned weights that will apply. */
      get blendRows() {
        const e = this.ensemble;
        if (!e) return [];
        if (e.method === 'blend_v5') return e.components;
        const w = e.calibrator?.weights || {};
        return [['elo', 'Elo'], ['mc', 'Monte Carlo'], ['xgb', 'XGBoost']]
          .map(([key, label]) => ({ key, label, coef: w[key], mean_abs_contribution: null }));
      },

      // ── status (from $store.status) ───────────────────────────────────
      get s() { return this.$store.status.data; },
      get demoDate() { return this.$store.status.demoDate; },
      get optunaAvailable() { return !!this.s?.capabilities?.optuna; },
      get seasonsText() {
        const t = this.s?.seasons?.train || [];
        return t.length ? `${t.length} seasons (${t[0]} to ${t[t.length - 1]})` : '3 seasons';
      },
      get modelBadge() {
        const state = this.$store.status.modelState;
        if (state === 'ok') return { cls: 'bg-green-lt', text: 'Ready' };
        if (state === 'old') return { cls: 'bg-yellow-lt', text: 'Over 7 days old' };
        if (state === 'none') return { cls: 'bg-secondary-lt', text: 'Elo only' };
        return { cls: 'bg-secondary-lt', text: '…' };
      },
      get trainedText() {
        const iso = this.s?.model?.trained_at;
        if (!iso) return fmt.DASH;
        const d = new Date(iso).toLocaleString('en-GB', { dateStyle: 'medium', timeStyle: 'short' });
        return `${d} (${fmt.ago(iso)})`;
      },
      count,

      // ── jobs ──────────────────────────────────────────────────────────
      get trainJob() { return this.$store.jobs.byKind('train'); },
      get focusJob() {
        const j = this.$store.jobs;
        return j.byKind('train') || j.byKind('autobacktest') || j.byKind('player_impact') || j.byKind('ratings');
      },
      get stepIndex() { return this.focusJob?.kind === 'train' ? (this.focusJob.done || 0) : 0; },
      get etaText() {
        const job = this.focusJob;
        const t = Date.parse(job?.started_at || '');
        if (!Number.isFinite(t)) return '';
        const elapsed = (this.$store.jobs.now - t) / 1000;
        if (elapsed < TYPICAL_TRAIN_S) return `About ${fmt.duration(TYPICAL_TRAIN_S - elapsed)} left for a typical run.`;
        return 'Taking longer than a typical run (a full download or Optuna adds time).';
      },
      get busyText() {
        const w = this.$store.jobs.writerBusy;
        return w ? `${this.$store.jobs.label(w)} is running.` : '';
      },
      get lastTrain() { return this.history.find((j) => j.kind === 'train' && TERMINAL.includes(j.state)) || null; },
      get lastTrainAlert() {
        const j = this.lastTrain;
        if (!j) return {};
        const when = fmt.ago(j.finished_at);
        if (j.state === 'done') {
          const m = this.s?.model;
          const body = m?.exists
            ? `Validation accuracy ${fmt.pct(m.val_acc, 1)} · log loss ${fmt.num(m.val_loss, 3)} · ${count(m.n_train)} training games`
            : 'The model file has since been removed.';
          return { cls: 'alert-success', icon: 'ti-circle-check', title: `Last training finished ${when}`, body };
        }
        if (j.state === 'error') {
          return { cls: 'alert-danger', icon: 'ti-alert-triangle', title: `Last training failed ${when}`, body: j.error || '' };
        }
        return { cls: 'alert-warning', icon: 'ti-player-stop', title: `Last training ${j.state} ${when}`, body: j.error || '' };
      },

      async startTrain() {
        await this.startJob('train', { ...this.form });
      },

      async startJob(kind, params = {}) {
        this.starting = kind;
        try {
          await this.$store.jobs.start(kind, params);
          this.loadHistory();
          if (this.isPhone) window.scrollTo({ top: 0, behavior: 'smooth' });
        } catch (e) {
          this.$store.toasts.push({
            title: `Could not start: ${this.$store.jobs.label({ kind })}`,
            body: [e.message, e.hint].filter(Boolean).join(' '), kind: 'danger', timeout: 10000,
          });
        } finally {
          this.starting = null;
        }
      },

      async reloadModel() {
        this.reloading = true;
        try {
          const r = await api.post('/api/model/reload');
          this.$store.toasts.push({
            title: 'Model reloaded', kind: 'success',
            body: r.trained_at ? `Trained ${new Date(r.trained_at).toLocaleString('en-GB', { dateStyle: 'medium', timeStyle: 'short' })}.` : '',
          });
          this.$store.status.load();
          this.loadModel();
        } catch (e) {
          this.$store.toasts.push({ title: 'Could not reload the model', body: e.message, kind: 'danger' });
        } finally {
          this.reloading = false;
        }
      },

      async clearCache() {
        try {
          await api.del('/api/cache');
          this.$store.toasts.push({ title: 'API cache cleared', kind: 'success' });
          this.$store.status.load();
        } catch (e) {
          this.$store.toasts.push({ title: 'Could not clear the cache', body: e.message, kind: 'danger' });
        }
      },

      scrollLog() {
        const el = this.$refs.log;
        if (el) el.scrollTop = el.scrollHeight;
      },

      // ── job history ───────────────────────────────────────────────────
      stateBadge(state) { return STATE_BADGE[state] || 'bg-secondary-lt'; },
      stateLabel(state) { return STATE_LABEL[state] || state || ''; },
      startedText(j) {
        const iso = j?.started_at || j?.created_at;
        return iso ? new Date(iso).toLocaleString('en-GB', { dateStyle: 'short', timeStyle: 'short' }) : fmt.DASH;
      },
      durationText(j) {
        const a = Date.parse(j?.started_at || '');
        const b = Date.parse(j?.finished_at || '') || (j && !TERMINAL.includes(j.state) ? this.$store.jobs.now : NaN);
        return Number.isFinite(a) && Number.isFinite(b) ? fmt.duration((b - a) / 1000) : fmt.DASH;
      },
      paramsText(j) {
        const p = j?.params || {};
        const on = Object.entries(p).filter(([, v]) => v !== false && v !== null && v !== undefined);
        if (!on.length) return 'Defaults';
        return on.map(([k, v]) => (v === true ? k.replace(/_/g, ' ') : `${k.replace(/_/g, ' ')} ${v}`)).join(', ');
      },
      async showJob(j) {
        this.selected = j;
        try { this.selected = await api.get(`/api/jobs/${encodeURIComponent(j.id)}`); } catch { /* keep the list row */ }
      },
    };
  });
}

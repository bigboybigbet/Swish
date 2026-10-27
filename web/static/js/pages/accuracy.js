// Accuracy: how the pre-game Kalman-Elo predictions did — KPIs, a calibration
// chart, accuracy by confidence tier and week by week.

import { api } from '../api.js';
import { fmt } from '../format.js';
import * as charts from '../charts.js';

const css = (name) => getComputedStyle(document.body).getPropertyValue(name).trim();

const OPTIONS = {
  calibration(a) {
    const pts = a.calibration.map((c) => [c.p, c.actual, c.n]);
    const maxN = Math.max(...a.calibration.map((c) => c.n), 1);
    return {
      grid: { left: 48, right: 16, top: 16, bottom: 40 },
      tooltip: { formatter: (p) => p.seriesIndex === 1
        ? `Predicted ${fmt.pct(p.value[0])}<br>Home team won ${fmt.pct(p.value[1])}<br>${p.value[2]} games` : '' },
      xAxis: { type: 'value', min: 0, max: 1, name: 'Predicted home win %', nameLocation: 'middle', nameGap: 26,
               axisLabel: { formatter: (v) => `${Math.round(v * 100)}` } },
      yAxis: { type: 'value', min: 0, max: 1, name: 'Actual', axisLabel: { formatter: (v) => `${Math.round(v * 100)}` } },
      series: [
        { type: 'line', data: [[0, 0], [1, 1]], symbol: 'none', silent: true,
          lineStyle: { type: 'dashed', color: css('--tblr-secondary') || '#888', width: 1 } },
        { type: 'scatter', data: pts, itemStyle: { color: css('--tblr-primary') || '#206bc4' },
          symbolSize: (v) => 6 + 18 * Math.sqrt(v[2] / maxN) },
      ],
    };
  },
  trend(a) {
    return {
      grid: { left: 48, right: 16, top: 24, bottom: 40 },
      legend: { top: 0, right: 0 },
      tooltip: { trigger: 'axis', valueFormatter: (v) => fmt.pct(v, 1) },
      xAxis: { type: 'category', data: a.trend.map((t) => t.week), axisLabel: { formatter: (v) => fmt.day(v).slice(4) } },
      yAxis: { type: 'value', min: 0.3, max: 1, axisLabel: { formatter: (v) => `${Math.round(v * 100)}%` } },
      series: [
        { name: 'Week', type: 'bar', data: a.trend.map((t) => t.acc), itemStyle: { color: css('--tblr-azure') || '#4299e1', opacity: 0.55 } },
        { name: 'Running', type: 'line', data: a.trend.map((t) => t.cum_acc), symbol: 'none', smooth: true,
          lineStyle: { width: 2, color: css('--tblr-primary') || '#206bc4' } },
        { name: 'Always home', type: 'line', data: a.trend.map(() => a.baseline_home), symbol: 'none',
          lineStyle: { type: 'dashed', width: 1, color: css('--tblr-secondary') || '#888' } },
      ],
    };
  },
};

export default function register(Alpine) {
  Alpine.data('accuracyChart', (kind) => {
    let chart = null;
    return {
      init() {
        const render = () => {
          if (!this.data?.n || !this.$el.isConnected) return;
          const opt = OPTIONS[kind](this.data);
          if (chart) chart.update(opt); else chart = charts.mount(this.$el, opt);
        };
        this.$watch('data', render);
        this.$watch('$store.prefs.theme', () => setTimeout(render, 0));
        this.$nextTick(render);
      },
      destroy() { chart?.dispose(); chart = null; },
    };
  });

  Alpine.data('accuracyPage', () => ({
    window: 'season',
    data: null,
    error: null,
    loading: false,

    init() { this.load(); this.$watch('window', () => this.load()); },

    async load() {
      this.loading = true;
      this.error = null;
      try {
        this.data = await api.get('/api/accuracy', { window: this.window });
      } catch (e) {
        this.error = e;
        this.data = null;
      } finally {
        this.loading = false;
      }
    },

    get lift() { return this.data?.n ? this.data.acc - this.data.baseline_home : null; },
    tierLabel(t) { return { high: 'High (≥ 65%)', medium: 'Medium (55–65%)', low: 'Low (< 55%)' }[t]; },
  }));
}

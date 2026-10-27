// ECharts helpers. Keep chart instances OUT of Alpine reactive data: hold the
// handle returned by mount() in a closure created in init() (Alpine's Proxy
// breaks chart objects), and call handle.dispose() in destroy().
//
//   Alpine.data('marginChart', () => { let chart; return {
//     init() { chart = charts.mount(this.$refs.el); this.$watch('result', r => chart.update(opt(r))); },
//     destroy() { chart?.dispose(); } }; });

import * as echarts from 'echarts';

const mounted = new Set();

export function currentTheme() {
  return document.documentElement.getAttribute('data-bs-theme') === 'dark' ? 'dark' : 'default';
}

function bodyFont() {
  return getComputedStyle(document.body).fontFamily || 'sans-serif';
}

/** Defaults every chart gets: accessible, transparent (sits on Tabler cards), body font. */
export function withDefaults(option) {
  return {
    backgroundColor: 'transparent',
    aria: { enabled: true },
    animationDuration: 300,
    ...option,
    textStyle: { fontFamily: bodyFont(), ...(option?.textStyle || {}) },
  };
}

/** Create a chart in `el`. Resizes with its container and follows the theme. */
export function mount(el, option) {
  let chart = echarts.init(el, currentTheme() === 'dark' ? 'dark' : null);
  let last = null;
  const ro = new ResizeObserver(() => chart.resize());
  ro.observe(el);

  const handle = {
    get instance() { return chart; },
    update(opt, notMerge = true) {
      last = withDefaults(opt);
      chart.setOption(last, { notMerge });
    },
    setTheme(theme) {
      if (typeof chart.setTheme === 'function') {
        chart.setTheme(theme);
      } else {                         // older ECharts: rebuild with the new theme
        chart.dispose();
        chart = echarts.init(el, theme === 'dark' ? 'dark' : null);
        if (last) chart.setOption(last);
      }
    },
    dispose() {
      ro.disconnect();
      mounted.delete(handle);
      chart.dispose();
    },
  };
  mounted.add(handle);
  if (option) handle.update(option);
  return handle;
}

// Theme sync: Tabler's toggle flips data-bs-theme on <html>.
new MutationObserver(() => {
  const theme = currentTheme();
  mounted.forEach((h) => h.setTheme(theme));
}).observe(document.documentElement, { attributes: true, attributeFilter: ['data-bs-theme'] });

export { echarts };

// Tabulator helpers. Like charts, keep the table instance out of Alpine's
// reactive data (hold it in a closure from init()).

import { TabulatorFull as Tabulator } from 'tabulator-tables';

const DEFAULTS = {
  layout: 'fitColumns',
  responsiveLayout: 'collapse',
  placeholder: 'No rows',
  reactiveData: false,
};

/** Create a table; it redraws itself when revealed by a Bootstrap tab. */
export function create(el, options) {
  const table = new Tabulator(el, { ...DEFAULTS, ...options });
  const onShown = () => { if (el.offsetParent !== null) table.redraw(); };
  document.addEventListener('shown.bs.tab', onShown);
  const destroy = table.destroy.bind(table);
  table.destroy = () => {
    document.removeEventListener('shown.bs.tab', onShown);
    destroy();
  };
  return table;
}

/** Download the table as CSV (visible columns, current filter/sort). */
export function downloadCsv(table, filename) {
  table.download('csv', filename, { bom: true });
}

export { Tabulator };

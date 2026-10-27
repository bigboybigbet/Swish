// Jobs tray store. M0: shows the active jobs reported by /api/status.
// M2 replaces the body with EventSource attach/reattach, the 800 ms polling
// fallback, cancel and toasts; keep the store name and `active`/`count`.

export function registerJobs(Alpine) {
  Alpine.store('jobs', {
    active: [],          // [{id, kind, progress, stage}]

    get count() { return this.active.length; },

    /** Replace the list (called with /api/status.active_jobs on each status load). */
    sync(list) { this.active = Array.isArray(list) ? list : []; },

    label(job) {
      const names = {
        train: 'Training', autobacktest: 'Auto-backtest', backtest: 'Backtest',
        scan: 'Slate scan', player_impact: 'Player impact', ratings: 'Ratings build',
      };
      return names[job?.kind] || job?.kind || 'Job';
    },
  });
}

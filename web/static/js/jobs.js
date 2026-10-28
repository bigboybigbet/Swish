// Jobs store: the running background jobs, with live progress over SSE.
//
//   const job = await Alpine.store('jobs').start('train', { force_rebuild: true });
//   Alpine.store('jobs').cancel(job.id);
//   document.addEventListener('nq:job', (e) => { const { id, kind, type, data } = e.detail; });
//
// Each job gets an EventSource on /api/jobs/{id}/events (the browser resends
// Last-Event-ID when it reconnects, so the stream resumes). After 2 stream
// errors the store polls GET /api/jobs/{id} every 800 ms instead. Every event
// is re-dispatched on document as `nq:job` ({id, kind, type, data, job});
// type is progress | log | item | done. A finished job leaves `active`,
// shows a toast and refreshes /api/status. On page load, /api/status
// active_jobs feeds sync(), which reattaches to jobs started elsewhere.

import { api } from './api.js';
import { fmt } from './format.js';

const POLL_MS = 800;
const POLL_MAX_MS = 5000;
const MAX_STREAM_ERRORS = 2;
const TAIL_LINES = 40;
const EVENT_TYPES = ['progress', 'log', 'item', 'done'];

const LABELS = {
  train: 'Training', autobacktest: 'Auto-backtest', backtest: 'Backtest',
  scan: 'Slate scan', player_impact: 'Player impact update', ratings: 'Ratings update',
};
const RESULT_PAGES = {
  train: '/model', autobacktest: '/accuracy', backtest: '/accuracy', scan: '/',
  player_impact: '/model', ratings: '/ratings',
};

// Connections live outside Alpine's reactive data (EventSource objects don't proxy well).
const conns = new Map();   // id → { es, errors, timer, delay, logSeen }

function entryFrom(job) {
  return {
    id: job.id, kind: job.kind, state: job.state || 'queued', writer: !!job.writer,
    progress: job.progress ?? null, stage: job.stage || '', done: job.done ?? null, total: job.total ?? null,
    started_at: job.started_at || job.created_at || new Date().toISOString(),
    error: null, tail: [], cancelling: false,
  };
}

export function registerJobs(Alpine) {
  const toasts = () => Alpine.store('toasts');

  function dispatch(store, id, type, data) {
    const entry = store.get(id);
    const job = entry ? { ...entry, tail: [...entry.tail] } : { id };
    document.dispatchEvent(new CustomEvent('nq:job', { detail: { id, kind: job.kind, type, data, job } }));
  }

  function handle(store, id, type, data) {
    const entry = store.get(id);
    if (!entry) return;
    if (type === 'progress') {
      Object.assign(entry, {
        progress: data.progress ?? null, stage: data.stage || '', done: data.done ?? null, total: data.total ?? null,
      });
      if (entry.state === 'queued' && data.stage && data.stage !== 'Queued') entry.state = 'running';
    } else if (type === 'log') {
      entry.tail.push(data.msg);
      if (entry.tail.length > TAIL_LINES) entry.tail.splice(0, entry.tail.length - TAIL_LINES);
    } else if (type === 'done') {
      entry.state = data.state;
      entry.error = data.error || null;
    }
    dispatch(store, id, type, data);
    if (type === 'done') finish(store, id, data);
  }

  function finish(store, id, data) {
    detach(id);
    const entry = store.get(id);
    if (!entry) return;
    store.active = store.active.filter((j) => j.id !== id);
    const label = store.label(entry);
    const href = RESULT_PAGES[entry.kind] || null;
    if (data.state === 'done') {
      toasts().push({ title: `${label} finished`, kind: 'success', href, linkText: 'View result' });
    } else if (data.state === 'error') {
      toasts().push({ title: `${label} failed`, body: data.error || '', kind: 'danger', href, linkText: 'Details', timeout: 15000 });
    } else {
      toasts().push({ title: `${label} ${data.state}`, body: data.error || '', kind: 'warning' });
    }
    Alpine.store('status').load();
  }

  function detach(id) {
    const c = conns.get(id);
    if (!c) return;
    c.es?.close();
    clearTimeout(c.timer);
    conns.delete(id);
  }

  function openStream(store, id) {
    const c = conns.get(id);
    const es = new EventSource(`/api/jobs/${encodeURIComponent(id)}/events`);
    c.es = es;
    for (const type of EVENT_TYPES) {
      es.addEventListener(type, (ev) => {
        c.errors = 0;
        let data;
        try { data = JSON.parse(ev.data); } catch { return; }
        if (type === 'done') es.close();          // before the browser tries to reconnect
        handle(store, id, type, data);
      });
    }
    es.onerror = () => {
      if (conns.get(id) !== c) return;
      c.errors += 1;
      if (c.errors >= MAX_STREAM_ERRORS || es.readyState === EventSource.CLOSED) startPolling(store, id);
    };
  }

  function startPolling(store, id) {
    const c = conns.get(id);
    if (!c) return;
    c.es?.close();
    c.es = null;
    const tick = async () => {
      if (conns.get(id) !== c) return;
      try {
        const rec = await api.get(`/api/jobs/${encodeURIComponent(id)}`);
        c.delay = POLL_MS;
        const lines = rec.log || [];
        for (const msg of lines.slice(Math.min(c.logSeen, lines.length))) handle(store, id, 'log', { msg });
        c.logSeen = lines.length;
        handle(store, id, 'progress', { progress: rec.progress, stage: rec.stage, done: rec.done, total: rec.total });
        if (['done', 'error', 'cancelled', 'interrupted'].includes(rec.state)) {
          handle(store, id, 'done', { state: rec.state, result_url: rec.result_url, error: rec.error });
          return;
        }
      } catch (e) {
        if (e.status === 404) {                    // forgotten by a restarted server
          detach(id);
          store.active = store.active.filter((j) => j.id !== id);
          return;
        }
        c.delay = Math.min(POLL_MAX_MS, c.delay * 2);   // server down: back off
      }
      if (conns.get(id) === c) c.timer = setTimeout(tick, c.delay);
    };
    c.delay = POLL_MS;
    tick();
  }

  function attach(store, id) {
    if (conns.has(id)) return;
    conns.set(id, { es: null, errors: 0, timer: null, delay: POLL_MS, logSeen: 0 });
    if (typeof EventSource === 'function') openStream(store, id);
    else startPolling(store, id);
  }

  Alpine.store('jobs', {
    active: [],          // [{id, kind, state, writer, progress, stage, done, total, started_at, tail, ...}]
    now: Date.now(),     // ticks every second while jobs run (elapsed times)

    get count() { return this.active.length; },
    /** The running writer job (train, autobacktest, player_impact, ratings), if any. */
    get writerBusy() { return this.active.find((j) => j.writer) || null; },
    get(id) { return this.active.find((j) => j.id === id) || null; },
    byKind(kind) { return this.active.find((j) => j.kind === kind) || null; },

    /** Add jobs reported by /api/status.active_jobs and attach to their streams. */
    sync(list) {
      if (!Array.isArray(list)) return;
      for (const job of list) {
        if (!this.get(job.id)) {
          this.active.push(entryFrom(job));
          attach(this, job.id);
        }
      }
    },

    /** POST /api/jobs/{kind}; throws ApiError (busy / no_model / validation). */
    async start(kind, params = {}) {
      const r = await api.post(`/api/jobs/${encodeURIComponent(kind)}`, params);
      if (!this.get(r.job_id)) {
        this.active.push(entryFrom({ id: r.job_id, kind, state: r.state, writer: r.writer, stage: 'Queued' }));
      }
      attach(this, r.job_id);
      return this.get(r.job_id);
    },

    async cancel(id) {
      const entry = this.get(id);
      if (entry) entry.cancelling = true;
      try {
        await api.del(`/api/jobs/${encodeURIComponent(id)}`);
        if (entry) entry.stage = 'Cancelling…';
      } catch (e) {
        if (entry) entry.cancelling = false;
        toasts().push({ title: 'Could not cancel', body: e.message, kind: 'danger' });
      }
    },

    label(job) { return LABELS[job?.kind] || job?.kind || 'Job'; },

    /** "Step 3 of 7 · Replaying Elo ratings" for staged jobs, else the stage text. */
    stageText(job) {
      if (!job) return '';
      if (job.progress == null && job.total && job.done) return `Step ${job.done} of ${job.total} · ${job.stage || ''}`;
      if (job.progress != null && job.total) return `${job.stage || ''}`;
      return job.stage || '';
    },

    /** Width for a determinate bar, null when indeterminate. */
    pct(job) { return job?.progress != null ? Math.round(job.progress * 1000) / 10 : null; },

    elapsed(job) {
      const t = Date.parse(job?.started_at || '');
      return Number.isFinite(t) ? fmt.duration(Math.max(0, (this.now - t) / 1000)) : '';
    },
  });

  setInterval(() => {
    const store = Alpine.store('jobs');
    if (store.count) store.now = Date.now();
  }, 1000);
}

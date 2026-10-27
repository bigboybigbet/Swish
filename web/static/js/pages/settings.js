// Settings page: write-only API keys (.env), Telegram test, thresholds and
// Kelly sizing, display preferences, environment info and the danger zone.
//
// Secrets are never read back: the API only says whether each one is set and
// where it comes from (.env or an environment variable).

import { api } from '../api.js';

const PCT_FIELDS = ['pick', 'safe', 'vip', 'value_edge'];
const LIMITS = {                                  // in form units (percent except fraction)
  pick: [50, 90], safe: [50, 90], vip: [50, 90], value_edge: [0, 20], fraction: [0, 1], cap_pct: [0, 25],
};
const LABELS = {
  pick: 'Pick', safe: 'Safe', vip: 'VIP', value_edge: 'Value edge', fraction: 'Kelly fraction', cap_pct: 'Stake cap',
};
const round = (x, dp) => Math.round(x * 10 ** dp) / 10 ** dp;

function toForm(s) {
  const t = s.thresholds;
  const k = s.kelly;
  return {
    pick: round(t.pick * 100, 2), safe: round(t.safe * 100, 2), vip: round(t.vip * 100, 2),
    value_edge: round(t.value_edge * 100, 2), fraction: round(k.fraction, 4), cap_pct: round(k.cap_pct, 2),
  };
}

const DANGER = [
  { key: 'model', title: 'Clear model', button: 'Clear model', path: '/api/model', confirm: true,
    effect: 'predictions fall back to Elo + Monte Carlo', done: 'Model deleted' },
  { key: 'dataset', title: 'Clear dataset', button: 'Clear dataset', path: '/api/dataset', confirm: true,
    effect: 'the next training run downloads it again', done: 'Dataset deleted' },
  { key: 'elo', title: 'Reset Elo', button: 'Reset Elo', path: '/api/elo', confirm: true,
    effect: 'every team goes back to 1500', done: 'Elo ratings reset' },
  { key: 'cache', title: 'Clear cache', button: 'Clear cache', path: '/api/cache', confirm: false,
    effect: 'cached NBA.com responses are downloaded again', done: 'API cache cleared' },
];

export default function register(Alpine) {
  Alpine.data('settingsPage', () => ({
    settings: null,
    loadError: null,
    keys: { odds_api_key: '', telegram_token: '', telegram_chat_id: '' },
    savingKey: false,
    tgTesting: false,
    tgResult: null,
    form: { pick: 55, safe: 65, vip: 65, value_edge: 4, fraction: 0.25, cap_pct: 5 },
    errors: {},
    formError: null,
    saving: false,
    savedAt: null,
    dangerBusy: null,

    init() { this.load(); },

    async load() {
      try {
        this.settings = await api.get('/api/settings');
        this.form = toForm(this.settings);
        this.loadError = null;
      } catch (e) {
        this.loadError = `Could not load the settings: ${e.message}`;
      }
    },

    get secrets() { return this.settings?.secrets || {}; },
    get server() { return this.settings?.server || {}; },
    get files() { return this.settings?.files || {}; },
    get packages() {
      return Object.entries(this.server.packages || {}).map(([name, version]) => ({ name, version }));
    },
    get telegramStatus() {
      const t = this.secrets.telegram || {};
      if (t.configured) return `Configured · chat ${t.chat_id_masked || ''}`;
      if (t.token_configured) return 'Token set, chat ID missing';
      if (t.chat_id_configured) return 'Chat ID set, token missing';
      return 'Not set';
    },
    sourceText(source) {
      return source === 'env' ? 'from an environment variable (it overrides .env)' : 'from .env';
    },

    // ── keys ─────────────────────────────────────────────────────────────
    async putSecrets(body, message) {
      this.savingKey = true;
      try {
        this.settings = await api.put('/api/settings/secrets', body);
        this.$store.toasts.push({ title: message, kind: 'success' });
        this.$store.status.load();
        this.tgResult = null;
        return true;
      } catch (e) {
        this.$store.toasts.push({ title: 'Could not save the key', body: e.message, kind: 'danger', timeout: 10000 });
        return false;
      } finally {
        this.savingKey = false;
      }
    },

    async saveKeys(fields) {
      const body = {};
      for (const f of fields) if (this.keys[f].trim()) body[f] = this.keys[f].trim();
      if (!Object.keys(body).length) return;
      if (await this.putSecrets(body, 'Saved to .env')) for (const f of fields) this.keys[f] = '';
    },

    async clearKeys(fields) {
      const body = Object.fromEntries(fields.map((f) => [f, '']));
      await this.putSecrets(body, 'Removed from .env');
    },

    async testTelegram() {
      this.tgTesting = true;
      this.tgResult = null;
      try {
        this.tgResult = await api.post('/api/telegram/test', {});
      } catch (e) {
        this.tgResult = { ok: false, detail: e.message };
      } finally {
        this.tgTesting = false;
      }
    },

    // ── thresholds ───────────────────────────────────────────────────────
    validate() {
      const errors = {};
      for (const [key, [lo, hi]] of Object.entries(LIMITS)) {
        const v = this.form[key];
        if (typeof v !== 'number' || !Number.isFinite(v) || v < lo || v > hi) {
          errors[key] = `${LABELS[key]} must be from ${lo} to ${hi}${key === 'fraction' ? '' : '%'}.`;
        }
      }
      if (!errors.safe && !errors.pick && this.form.safe < this.form.pick) errors.safe = 'Safe must be at least Pick.';
      if (!errors.vip && !errors.pick && this.form.vip < this.form.pick) errors.vip = 'VIP must be at least Pick.';
      this.errors = errors;
      return Object.values(errors)[0] || null;
    },

    get dirty() {
      if (!this.settings) return false;
      const saved = toForm(this.settings);
      return Object.keys(saved).some((k) => saved[k] !== this.form[k]);
    },
    get isDefault() {
      if (!this.settings?.defaults) return true;
      const d = toForm(this.settings.defaults);
      return Object.keys(d).every((k) => d[k] === this.form[k]);
    },

    resetDefaults() {
      if (this.settings?.defaults) this.form = toForm(this.settings.defaults);
      this.errors = {};
      this.formError = null;
    },

    async saveThresholds() {
      this.formError = this.validate();
      if (this.formError) return;
      const f = this.form;
      const thresholds = Object.fromEntries(PCT_FIELDS.map((k) => [k, round(f[k] / 100, 4)]));
      this.saving = true;
      try {
        this.settings = await api.put('/api/settings', {
          thresholds, kelly: { fraction: f.fraction, cap_pct: f.cap_pct },
        });
        this.form = toForm(this.settings);
        this.savedAt = Date.now();
        this.$store.toasts.push({ title: 'Settings saved', body: 'They apply to new predictions.', kind: 'success' });
        this.$store.status.load();
      } catch (e) {
        this.formError = e.message;
      } finally {
        this.saving = false;
      }
    },

    // ── danger zone ──────────────────────────────────────────────────────
    get dangerActions() {
      return DANGER.map((d) => ({ ...d, file: this.files[d.key] || '' }));
    },
    get busy() { return this.$store.jobs.writerBusy; },
    get busyText() {
      return this.busy ? `Disabled while ${this.$store.jobs.label(this.busy).toLowerCase()} is running.` : '';
    },

    async danger(key) {
      const d = DANGER.find((x) => x.key === key);
      if (!d) return;
      this.dangerBusy = key;
      try {
        const r = await api.del(d.path, d.confirm ? { confirm: true } : undefined);
        const nothing = r && r.removed === false;
        this.$store.toasts.push({
          title: nothing ? 'Nothing to delete' : d.done, kind: nothing ? 'info' : 'success',
          body: nothing ? `${this.files[key] || 'The file'} did not exist.` : '',
        });
        this.$store.status.load();
      } catch (e) {
        this.$store.toasts.push({ title: `${d.title} failed`, body: e.message, kind: 'danger', timeout: 10000 });
      } finally {
        this.dangerBusy = null;
      }
    },
  }));
}

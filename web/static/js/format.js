// Display formatting. Null/undefined always renders as an en dash.
// Dates from the API are ET calendar days ("YYYY-MM-DD"); timestamps are ISO UTC.

const MINUS = '−';
const DASH = '–';
const isNum = (x) => typeof x === 'number' && Number.isFinite(x);

export function num(x, dp = 1) {
  return isNum(x) ? x.toFixed(dp).replace('-', MINUS) : DASH;
}

/** Signed number with a real minus sign: +4.3 / −4.3 / 0.0 */
export function signed(x, dp = 1) {
  if (!isNum(x)) return DASH;
  const s = Math.abs(x).toFixed(dp);
  if (Number(s) === 0) return (0).toFixed(dp);
  return (x > 0 ? '+' : MINUS) + s;
}

/** Probability 0..1 → "61%" */
export function pct(p, dp = 0) {
  return isNum(p) ? `${(p * 100).toFixed(dp)}%` : DASH;
}

/** Probability difference → "+4.3 pp" */
export function pp(d, dp = 1) {
  return isNum(d) ? `${signed(d * 100, dp)} pp` : DASH;
}

export function americanToDecimal(ml) {
  if (!isNum(ml) || ml === 0) return null;
  return ml > 0 ? 1 + ml / 100 : 1 + 100 / -ml;
}

/** Moneyline in the user's format: mode "us" → +150 / −180, "dec" → 2.50 */
export function odds(ml, mode = 'us') {
  if (!isNum(ml)) return DASH;
  if (mode === 'dec') return americanToDecimal(ml).toFixed(2);
  return ml > 0 ? `+${Math.round(ml)}` : `${MINUS}${Math.abs(Math.round(ml))}`;
}

/** Favourite form of a spread: homeSpread −4.5 → "LAL −4.5"; 0 → "PK". */
export function spread(homeAbbr, awayAbbr, homeSpread, dp = 1) {
  if (!isNum(homeSpread)) return DASH;
  if (Math.abs(homeSpread) < 0.05) return 'PK';
  return homeSpread < 0
    ? `${homeAbbr} ${MINUS}${Math.abs(homeSpread).toFixed(dp)}`
    : `${awayAbbr} ${MINUS}${homeSpread.toFixed(dp)}`;
}

// ── dates ────────────────────────────────────────────────────────────────

/** "2026-10-28" → "Tue 28 Oct" (ET calendar day; no timezone shift). */
export function day(ymd, withYear = false) {
  if (!ymd) return DASH;
  const d = new Date(`${ymd}T12:00:00Z`);
  const opts = { weekday: 'short', day: 'numeric', month: 'short', timeZone: 'UTC' };
  if (withYear) opts.year = 'numeric';
  return d.toLocaleDateString('en-GB', opts).replace(',', '');
}

/** Shift a "YYYY-MM-DD" by n days. */
export function addDays(ymd, n) {
  const d = new Date(`${ymd}T12:00:00Z`);
  d.setUTCDate(d.getUTCDate() + n);
  return d.toISOString().slice(0, 10);
}

/** Today in ET from the browser clock. Prefer $store.status.data.today_et (demo mode). */
export function todayEt() {
  return new Intl.DateTimeFormat('en-CA', { timeZone: 'America/New_York' }).format(new Date());
}

/** Clock time of an ISO timestamp, in local time or a given zone. */
export function time(iso, timeZone) {
  if (!iso) return DASH;
  return new Date(iso).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', timeZone });
}

/** Tip-off in local time and ET: {local: "07:00", et: "19:00 ET", localDay: "Wed 29 Oct"} */
export function tipoff(iso) {
  if (!iso) return { local: DASH, et: DASH, localDay: DASH };
  const d = new Date(iso);
  return {
    local: time(iso),
    et: `${time(iso, 'America/New_York')} ET`,
    localDay: d.toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short' }).replace(',', ''),
  };
}

/** "just now", "3 min ago", "2 h ago", "5 d ago" */
export function ago(iso, now = Date.now()) {
  if (!iso) return DASH;
  const s = Math.max(0, (now - new Date(iso).getTime()) / 1000);
  if (s < 45) return 'just now';
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return `${Math.round(s / 86400)} d ago`;
}

/** Duration in seconds → "42 s", "3 min 05 s", "1 h 02 min" */
export function duration(sec) {
  if (!isNum(sec)) return DASH;
  const s = Math.round(sec);
  if (s < 60) return `${s} s`;
  if (s < 3600) return `${Math.floor(s / 60)} min ${String(s % 60).padStart(2, '0')} s`;
  return `${Math.floor(s / 3600)} h ${String(Math.floor((s % 3600) / 60)).padStart(2, '0')} min`;
}

/** Confidence tier label for a pick probability (thresholds from /api/status). */
export function tier(p, thresholds = { pick: 0.55, safe: 0.65 }) {
  if (!isNum(p)) return null;
  if (p >= thresholds.safe) return 'high';
  if (p >= thresholds.pick) return 'medium';
  return 'low';
}

export const fmt = { num, signed, pct, pp, americanToDecimal, odds, spread, day, addDays, todayEt, time, tipoff, ago, duration, tier, DASH, MINUS };
export default fmt;

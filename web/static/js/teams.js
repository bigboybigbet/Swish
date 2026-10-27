// Team list (from /api/teams, cached) and colour helpers. Team colours only,
// no logos.

import { api } from './api.js';

let teamsPromise = null;

/** All 30 teams, fetched once per page load. */
export function loadTeams() {
  teamsPromise ??= api.get('/api/teams').catch((e) => { teamsPromise = null; throw e; });
  return teamsPromise;
}

// ── colour maths (WCAG 2.x) ──────────────────────────────────────────────

function rgb(hex) {
  const h = String(hex || '').replace('#', '');
  const v = h.length === 3 ? h.split('').map((c) => c + c).join('') : h;
  const n = parseInt(v, 16);
  return Number.isNaN(n) ? [128, 128, 128] : [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function luminance(hex) {
  const [r, g, b] = rgb(hex).map((c) => {
    const s = c / 255;
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

export function contrastRatio(a, b) {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}

/** Badge colours: primary background; white or the secondary colour as text,
 *  whichever contrasts more (black if neither reaches 4.5:1). */
export function badgeColors(team) {
  const c = team?.colors;
  if (!c) return null;
  const bg = c.primary;
  let fg = contrastRatio(bg, '#FFFFFF') >= contrastRatio(bg, c.secondary) ? '#FFFFFF' : c.secondary;
  if (contrastRatio(bg, fg) < 4.5 && contrastRatio(bg, '#000000') > contrastRatio(bg, fg)) fg = '#000000';
  return { bg, fg };
}

export function badgeStyle(team) {
  const c = badgeColors(team);
  return c ? { backgroundColor: c.bg, color: c.fg } : {};
}

// Perceptual-ish distance between two colours (redmean approximation).
function distance(a, b) {
  const [r1, g1, b1] = rgb(a);
  const [r2, g2, b2] = rgb(b);
  const rm = (r1 + r2) / 2;
  const dr = r1 - r2; const dg = g1 - g2; const db = b1 - b2;
  return Math.sqrt((2 + rm / 256) * dr * dr + 4 * dg * dg + (2 + (255 - rm) / 256) * db * db);
}

/** Two distinguishable colours for a matchup (e.g. the split probability bar).
 *  Keeps both primaries unless they clash, then tries secondaries.
 *  Idea from DeepShot's get_best_color_pair (MIT,
 *  github.com/saccofrancesco/deepshot); written from scratch. */
export function pairColors(home, away, minDistance = 150) {
  const h = home?.colors; const a = away?.colors;
  if (!h || !a) return { home: 'var(--tblr-primary)', away: 'var(--tblr-secondary)' };
  const options = [
    [h.primary, a.primary], [h.primary, a.secondary],
    [h.secondary, a.primary], [h.secondary, a.secondary],
  ];
  const best = options.find(([x, y]) => distance(x, y) >= minDistance)
    ?? options.reduce((p, q) => (distance(q[0], q[1]) > distance(p[0], p[1]) ? q : p));
  return { home: best[0], away: best[1] };
}

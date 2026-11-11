// Predicts Roblox Up-and-Coming game rankings using a probabilistic diffusion model with drift.
// A probabilistic diffusion model predicting Roblox experience trajectory and ranking.
// roblox-up-and-coming-forecast-01 entry (Node 22, standard library only).
// A diffusion-with-drift forecaster for Roblox's Up-and-Coming chart.
//
// For each game on the chart, its rank at horizon h hours after the history
// ends is modelled as Normal(current rank + drift(h), sigma(h)):
//   - drift: the band mean rank change per 24 h, estimated from every pair of
//     reads ~24 h apart in the history, plus the game's own recent rank
//     velocity (shrunk, averaged over two consecutive ~24 h steps) and its
//     recent log-playing growth (playing growth predicts rank climbs:
//     forward R^2 0.069 alone, 0.075 with rank velocity; coefficient -3,
//     shrunk well below the OLS -7.4 because the log score punishes
//     overconfident drift);
//   - sigma: the band standard deviation of 24 h rank changes, scaled by
//     sqrt(h / 24) and by SIGMA_SCALE (the horizon average already spreads
//     the forecast, so the per-horizon width is narrower than the raw sd).
// The per-place probabilities are the time-average over horizons 0..72 h
// (trapezoid weights = uniform over time), so the first hours stay close to
// the chart as it stands and later hours spread out as the chart churns.
// Each place leaves UNNAMED of its mass unnamed: the scorer shares it over
// the games on a read's chart the forecast does not name there, which is
// exactly the cover for brand-new games (in the practice window a game never
// seen in the history reached #1 in 3 of 10 reads).
// The top 10 is ordered by expected rank.
// When a read carries the games' details, a game's `created` date is used
// too: the chart only lists games published in the last 28 days, so a game
// is given no mass at horizons past its 28th day (that mass flows to the
// unnamed share, i.e. to the games that replace it).
//
// Backtest (walk-forward, 6 folds each, mean log score): this model
// -36.94 (spread folds) / -31.87 (late folds); practice window -31.80
// vs the previous version's -38.57 walk-forward / -33.71 practice and the
// persistence baseline -49.68 / -43.51. Tuned offline (grid search +
// walk-forward guard); the recommended config clears the top-10 practice
// bar (-32.0) and improves on 10 of 11 walk-forward folds.
// Deterministic: no clock, no randomness; the same history gives the same
// forecast byte for byte.
//
// Run locally: mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Momentum: how much of a game's recent 24 h rank change carries forward. */
const VEL_SHRINK = 0.1;
/** Per-horizon width as a fraction of the estimated 24 h rank-change sd. */
const SIGMA_SCALE = 0.6;
/** Mass left unnamed at every place, for games the forecast does not name. */
const UNNAMED = 0.15;
/** Rank bands for drift/sigma estimation. */
const BANDS: Array<[number, number]> = [[1, 3], [4, 10], [11, 20], [21, 40], [41, 200]];
/** Fallbacks when a band has too few read pairs to estimate from. */
const DEFAULT_DRIFT = [1.95, 3.94, 6.05, 4.95, -3.22];
const DEFAULT_SIGMA = [4.12, 8.6, 12.12, 15.69, 19.45];
/** Horizon grid step, in hours (the window comes from task.json). */
const HORIZON_STEP = 3;
/** A band needs this many pairs before its own estimate replaces the default. */
const MIN_PAIRS = 100;
/** Games are only named at a place when their probability clears this. */
const MIN_P = 0.005;
/** At most this many rows per place (the rules allow 100). */
const MAX_ROWS_PER_PLACE = 100;
/** Log-playing growth per 24 h, in ranks/day (negative: growth climbs). */
const PLAY_COEF = -3;
/** The chart only lists games published in the last 28 days. */
const CHART_MAX_AGE_MS = 28 * 24 * 3600 * 1000;

interface ChartRow {
  rank: number;
  universe_id: number;
  playing?: number | null;
}

interface HistoryRead {
  at: string;
  up_and_coming: ChartRow[];
  details?: unknown;
}

function bandOf(rank: number): number {
  for (let i = 0; i < BANDS.length; i++) {
    if (rank >= BANDS[i]![0] && rank <= BANDS[i]![1]) return i;
  }
  return BANDS.length - 1;
}
// Helper function to calculate the arithmetic mean of an array of numbers.
function mean(xs: number[]): number {
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}


// Computes arithmetic mean across numerical arrays for statistical estimators.

function pstdev(xs: number[]): number {
  const m = mean(xs);
  return Math.sqrt(xs.reduce((a, b) => a + (b - m) * (b - m), 0) / xs.length);
}

/** erf via the Abramowitz-Stegun approximation (|error| < 1.5e-7). */
function erf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const y =
    1 -
    (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-ax * ax));
  return sign * y;
}

/** Standard normal CDF. */
function normCdf(x: number): number {
  return 0.5 * (1 + erf(x / Math.SQRT2));
}

const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8')) as {
  places: number;
  output: string;
  window: { from: string; until: string };
};
const places = task.places;
const lines = readFileSync(join(INPUT_DIR, 'history.ndjson'), 'utf8')
  .split('\n')
  .filter((l) => l !== '');
const reads = lines.map((l) => JSON.parse(l) as HistoryRead);
const times = reads.map((r) => Date.parse(r.at));
const rankOf = reads.map((r) => {
  const m = new Map<number, number>();
  for (const g of r.up_and_coming) m.set(g.universe_id, g.rank);
  return m;
});

// Band drift/sigma per 24 h, from every pair of reads 16-32 h apart.
const diffs: number[][] = BANDS.map(() => []);
for (let i = 0; i < reads.length; i++) {
  for (let j = i + 1; j < reads.length; j++) {
    const dtH = (times[j]! - times[i]!) / 3600000;
    if (dtH < 16 || dtH > 32) continue;
    for (const g of reads[i]!.up_and_coming) {
      const rj = rankOf[j]!.get(g.universe_id);
      if (rj === undefined) continue;
      diffs[bandOf(g.rank)]!.push(rj - g.rank);
    }
  }
}
const drift = diffs.map((d, b) => (d.length >= MIN_PAIRS ? mean(d) : DEFAULT_DRIFT[b]!));
const sigma = diffs.map((d, b) => (d.length >= MIN_PAIRS ? pstdev(d) : DEFAULT_SIGMA[b]!));

/** A game's rank change per 24 h over its recent past (0 when unknown):
    the mean of two consecutive ~24 h steps, which halves the noise of a
    single step. */
function velocity(gid: number): number {
  const idxs: number[] = [];
  for (let i = 0; i < reads.length; i++) if (rankOf[i]!.has(gid)) idxs.push(i);
  if (idxs.length < 2) return 0;
  const last = idxs[idxs.length - 1]!;
  let best = -1;
  for (const p of idxs.slice(0, -1)) {
    const dt = (times[last]! - times[p]!) / 3600000;
    if (dt >= 12 && dt <= 36) {
      if (best === -1 || Math.abs(dt - 24) < Math.abs((times[last]! - times[best]!) / 3600000 - 24)) best = p;
    }
  }
  if (best === -1) best = idxs[idxs.length - 2]!;
  const dt = (times[last]! - times[best]!) / 3600000;
  if (dt <= 0) return 0;
  const v1 = ((rankOf[last]!.get(gid)! - rankOf[best]!.get(gid)!) / dt) * 24;
  if (idxs.length >= 3) {
    let prev = -1;
    for (const p of idxs) {
      if (p >= best) break;
      const dt2 = (times[best]! - times[p]!) / 3600000;
      if (dt2 >= 12 && dt2 <= 36) {
        if (prev === -1 || Math.abs(dt2 - 24) < Math.abs((times[best]! - times[prev]!) / 3600000 - 24)) prev = p;
      }
    }
    if (prev !== -1) {
      const dt2 = (times[best]! - times[prev]!) / 3600000;
      if (dt2 > 0) {
        const v2 = ((rankOf[best]!.get(gid)! - rankOf[prev]!.get(gid)!) / dt2) * 24;
        return (v1 + v2) / 2;
      }
    }
  }
  return v1;
}

/** A game's log-playing change per 24 h over its recent past (0 when
    unknown): same pairing as velocity. Playing growth is the strongest
    single predictor of the next day's rank change found in the history. */
function playVelocity(gid: number): number {
  const idxs: number[] = [];
  for (let i = 0; i < reads.length; i++) {
    for (const g of reads[i]!.up_and_coming) {
      if (g.universe_id === gid && g.playing) {
        idxs.push(i);
        break;
      }
    }
  }
  if (idxs.length < 2) return 0;
  const last = idxs[idxs.length - 1]!;
  let best = -1;
  for (const p of idxs.slice(0, -1)) {
    const dt = (times[last]! - times[p]!) / 3600000;
    if (dt >= 12 && dt <= 36) {
      if (best === -1 || Math.abs(dt - 24) < Math.abs((times[last]! - times[best]!) / 3600000 - 24)) best = p;
    }
  }
  if (best === -1) best = idxs[idxs.length - 2]!;
  const dt = (times[last]! - times[best]!) / 3600000;
  if (dt <= 0) return 0;
  const pLast = reads[last]!.up_and_coming.find((g) => g.universe_id === gid)!.playing!;
  const pBest = reads[best]!.up_and_coming.find((g) => g.universe_id === gid)!.playing!;
  if (!pLast || !pBest || pLast <= 0 || pBest <= 0) return 0;
  return (Math.log(pLast / pBest) / dt) * 24;
}

// Candidate games: the last chart, plus games seen in the last 2 days that
// have since left the chart (they can re-enter; their rank is their last seen).
const last = reads[reads.length - 1]!;
const t0 = Date.parse(last.at);
const candRank = new Map<number, number>();
for (const g of last.up_and_coming) candRank.set(g.universe_id, g.rank);
for (const r of reads) {
  if (t0 - Date.parse(r.at) <= 2 * 86400000) {
    for (const g of r.up_and_coming) {
      if (!candRank.has(g.universe_id)) candRank.set(g.universe_id, g.rank);
    }
  }
}

// Latest known `created` date per game, from any read's details (the archive
// reads carry none, so this stays empty on the practice history; the house
// recorder's reads do). The details may be an object keyed by universe id or
// an array of detail objects; anything else is ignored.
const created = new Map<number, number>();
for (const r of reads) {
  const det = r.details;
  if (!det || typeof det !== 'object') continue;
  const put = (id: unknown, c: unknown): void => {
    if (typeof id !== 'number' || typeof c !== 'string') return;
    const ms = Date.parse(c);
    if (Number.isFinite(ms)) created.set(id, ms);
  };
  if (Array.isArray(det)) {
    for (const d of det as Array<{ id?: unknown; universe_id?: unknown; created?: unknown }>) {
      put(d.id ?? d.universe_id, d.created);
    }
  } else {
    for (const [key, d] of Object.entries(det as Record<string, { created?: unknown }>)) {
      put(Number(key), d && typeof d === 'object' ? d.created : undefined);
    }
  }
}

// Horizon grid over the scored window, trapezoid weights = uniform over time.
const windowHours = Math.max(1, Math.round((Date.parse(task.window.until) - Date.parse(task.window.from)) / 3600000));
const horizons: number[] = [];
for (let h = 0; h <= windowHours; h += HORIZON_STEP) horizons.push(h);
if (horizons[horizons.length - 1] !== windowHours) horizons.push(windowHours);
const weights = horizons.map((h, i) => {
  const prev = i === 0 ? 0 : h - horizons[i - 1]!;
  const next = i === horizons.length - 1 ? 0 : horizons[i + 1]! - h;
  return (prev + next) / 2;
});

const placeP: Array<Map<number, number>> = Array.from({ length: places }, () => new Map());
const expRank = new Map<number, number>();
for (const [gid, r0] of candRank) {
  const b = bandOf(r0);
  const d24 = drift[b]! + velocity(gid) * VEL_SHRINK + playVelocity(gid) * PLAY_COEF;
  const s24 = Math.max(2, sigma[b]! * SIGMA_SCALE);
  const dist = new Array<number>(places).fill(0);
  let meanSum = 0;
  let wSum = 0;
  for (let hi = 0; hi < horizons.length; hi++) {
    const h = horizons[hi]!;
    const wh = weights[hi]! / windowHours;
    const createdMs = created.get(gid);
    if (createdMs !== undefined && h > 0 && createdMs + CHART_MAX_AGE_MS < t0 + h * 3600000) {
      continue; // aged out of the chart by this horizon (h = 0 is always kept: it is on the chart now)
    }
    wSum += wh;
    if (h === 0) {
      if (r0 >= 1 && r0 <= places) dist[r0 - 1]! += wh;
      meanSum += r0 * wh;
      continue;
    }
    const m = r0 + d24 * (h / 24);
    const s = s24 * Math.sqrt(h / 24);
    meanSum += m * wh;
    for (let k = 1; k <= places; k++) {
      dist[k - 1]! += (normCdf((k + 0.5 - m) / s) - normCdf((k - 0.5 - m) / s)) * wh;
    }
  }
  expRank.set(gid, wSum > 0 ? meanSum / wSum : r0);
  for (let k = 1; k <= places; k++) {
    if (dist[k - 1]! > 0) placeP[k - 1]!.set(gid, dist[k - 1]!);
  }
}

const probabilities: Array<{ universe_id: number; place: number; p: number }> = [];
for (let k = 1; k <= places; k++) {
  const items = [...placeP[k - 1]!.entries()]
    .filter(([, p]) => p >= MIN_P)
    .sort((a, b) => b[1] - a[1] || a[0] - b[0])
    .slice(0, MAX_ROWS_PER_PLACE);
  const tot = items.reduce((a, [, p]) => a + p, 0);
  let scale = tot > 0 ? (1 - UNNAMED) / tot : 0;
  if (scale > 1) scale = 1;
  for (const [gid, p] of items) {
    probabilities.push({ universe_id: gid, place: k, p: p * scale });
  }
}
const top10 = [...expRank.entries()]
  .sort((a, b) => a[1] - b[1] || a[0] - b[0])
  .slice(0, places)
  .map(([gid]) => gid);

mkdirSync(OUTPUT_DIR, { recursive: true });
writeFileSync(join(OUTPUT_DIR, task.output), `${JSON.stringify({ probabilities, top10 })}\n`);

/**
 * Widgets specific to lesson-4.html (expected free energy: scoring policies
 * before their observations have happened). Shared infrastructure and the
 * reusable EFE arithmetic live in widget.js and aif.js respectively.
 */

import {
  mount,
  mountAll,
  ctxOf,
  label,
  drawGrid,
  drawRow,
  drawStateMarks,
  drawStateStrip,
  COLOUR,
  COLS,
  N,
} from "./widget.js";
import { transitionModel, scorePolicy, sophisticatedValues, predict, expectedObs, softmax } from "./aif.js";
import { AGENT_WALKS } from "./agent-comparison-data.js";
import "./agent-comparison.js";
import "./figure6-curves.js";

// Canonical lesson-4 world from content/lesson-4.md, "Watching it run".
// Prose coordinates are 1-indexed; state indices are the zero-indexed row-major
// representation used by A and B throughout the code.
const LAMPS = [9, 13]; // (2, 5) and (3, 4)
const HOME = 4; // (1, 5)
const TRUE_START = 16; // (4, 2), hidden from the agent
const LIT_MASK = Array.from({ length: N }, (_, s) => LAMPS.includes(s));
const C = [0.05, 0.1, 0.85];
const REWARD = Array.from({ length: N }, (_, s) => (s === HOME ? 10 : 0));

// The lamp placement is not decorative. With the lamps further off, at (5,3)
// and (3,5), the epistemic term is an order of magnitude below the pragmatic
// one, no policy ever routes via a lamp, and the active inference agent
// becomes indistinguishable from a reward maximiser. Verified in
// docs/learning-records/0008; do not move them without re-running it.

// The sensor reports a KIND of cell, not which cell: the two lamps are
// identical, so reaching one leaves the agent choosing between them. Mirrors
// aif.generative_model.kind_observation_model(mark_goal=True).
const P_CORRECT = 0.9;
const OBS_NAMES = ["dark", "lit", "home"];
const A_LESSON4 = [0, 1, 2].map((o) =>
  Array.from({ length: N }, (_, s) => {
    const kind = s === HOME ? 2 : LAMPS.includes(s) ? 1 : 0;
    return o === kind ? P_CORRECT : (1 - P_CORRECT) / 2;
  }),
);

// This B encodes the agent's uncertainty about where a command may land.
// The comparison walk below uses a deterministic physical world while its
// agents retain an uncertain B. At reliability 1 the belief merely shifts;
// below 1 its movement prediction spreads. See learning record 0008 for the
// earlier stochastic-world calculation used by the static figures.
const RELIABILITY = 0.7;
const B_LESSON4 = transitionModel({ rows: 5, cols: COLS, reliability: RELIABILITY });

// The agent knows its column, not its row: (2,4), (3,3), (4,2) zero-indexed.
const START_CELLS = [8, 12, 16];
const START_CELL = 12; // (3,3), the cell the agent is really in
const START_BELIEF = Array.from({ length: N }, (_, s) =>
  START_CELLS.includes(s) ? 1 / START_CELLS.length : 0,
);

// --- c-not-reward ---------------------------------------------------------
// Figure: C is a three-entry distribution indexed by observations, whereas a
// reward vector has one unconstrained real number per state. Their visibly
// different lengths and totals carry the distinction made in "What the agent
// is after"; adding preferences over states here would blur that contrast.
mount("c-not-reward", (el) => {
  const W = 760;
  const H = 306;
  const ctx = ctxOf(el, W, H);
  if (!ctx) return;

  ctx.canvas.setAttribute(
    "aria-label",
    "A five by five world with two lamps and a home marking, beside a three-entry preference distribution over observations and a twenty-five-entry reward vector over states.",
  );

  const WORLD_X = 24;
  const WORLD_Y = 58;
  const WORLD_CELL = 34;
  const PREF_X = 242;
  const PREF_Y = 92;
  const PREF_CELL = 44;
  const REWARD_X = 430;
  const REWARD_Y = 112;
  const REWARD_CELL = 12;

  function drawWorld() {
    label(ctx, "the world", WORLD_X, 24, COLOUR.ink, 17);
    label(ctx, "states are cells", WORLD_X, 43, COLOUR.dim, 13);
    // The red dot is the agent's actual cell, (3,3), shown only to the reader:
    // its prior spans cells 8, 12 and 16 and it cannot tell which it is in.
    drawGrid(ctx, WORLD_X, WORLD_Y, WORLD_CELL,
      { lit: LIT_MASK, mark: HOME, agent: START_CELL });
    ctx.font = "bold 15px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = "#fff";
    ctx.fillText(
      "H",
      WORLD_X + (HOME % COLS) * WORLD_CELL + (WORLD_CELL - 2) / 2,
      WORLD_Y + Math.floor(HOME / COLS) * WORLD_CELL + (WORLD_CELL - 2) / 2,
    );
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    label(ctx, "H  home", WORLD_X, 248, COLOUR.dim, 13);
    // Second line: the legend must not run into the preference column at x=242.
    label(ctx, "\u25cf  where the agent actually is", WORLD_X, 266, COLOUR.dim, 13);
  }

  function drawPreferences() {
    label(ctx, "preference  p(o | C)", PREF_X, 24, COLOUR.ink, 17);
    label(ctx, "one entry per observation", PREF_X, 43, COLOUR.dim, 13);
    const names = ["dark", "lit", "home"];
    const maxBar = 64;
    for (let o = 0; o < C.length; o++) {
      const x = PREF_X + o * PREF_CELL;
      const height = C[o] * maxBar;
      ctx.fillStyle = "#eef1f4";
      ctx.fillRect(x, PREF_Y, PREF_CELL - 7, maxBar);
      ctx.fillStyle = o === 1 ? COLOUR.lit : COLOUR.neutral;
      ctx.fillRect(x, PREF_Y + maxBar - height, PREF_CELL - 7, height);
      ctx.strokeStyle = COLOUR.rule;
      ctx.strokeRect(x + 0.5, PREF_Y + 0.5, PREF_CELL - 8, maxBar - 1);
      ctx.font = "12px system-ui, sans-serif";
      ctx.fillStyle = COLOUR.dim;
      ctx.textAlign = "center";
      ctx.fillText(names[o], x + (PREF_CELL - 7) / 2, PREF_Y + maxBar + 18);
      ctx.fillText(C[o].toFixed(2), x + (PREF_CELL - 7) / 2, PREF_Y + maxBar + 36);
    }
    ctx.textAlign = "left";
    ctx.strokeStyle = COLOUR.agent;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(PREF_X, 226);
    ctx.lineTo(PREF_X + C.length * PREF_CELL - 7, 226);
    ctx.stroke();
    label(ctx, "total = 1.00", PREF_X + 13, 248, COLOUR.agent, 14);
    label(ctx, "must sum to one", PREF_X + 2, 270, COLOUR.dim, 13);
  }

  function drawReward() {
    label(ctx, "example reward  r(s)", REWARD_X, 24, COLOUR.ink, 17);
    label(ctx, "comparison only: one value per state", REWARD_X, 43, COLOUR.dim, 13);
    label(ctx, "0", REWARD_X + 2, 94, COLOUR.dim, 12);
    label(ctx, "1", REWARD_X + REWARD_CELL + 2, 94, COLOUR.dim, 12);
    label(ctx, "2", REWARD_X + 2 * REWARD_CELL + 2, 94, COLOUR.dim, 12);
    label(ctx, "4", REWARD_X + 4 * REWARD_CELL + 2, 94, COLOUR.agent, 12);
    label(ctx, "…", REWARD_X + 10 * REWARD_CELL - 1, 94, COLOUR.dim, 12);
    label(ctx, "24", REWARD_X + 23.5 * REWARD_CELL, 94, COLOUR.dim, 12);
    for (let s = 0; s < N; s++) {
      const x = REWARD_X + s * REWARD_CELL;
      ctx.fillStyle = REWARD[s] === 0 ? "#fff" : COLOUR.agent;
      ctx.fillRect(x, REWARD_Y, REWARD_CELL - 2, 44);
      ctx.strokeStyle = COLOUR.rule;
      ctx.lineWidth = 1;
      ctx.strokeRect(x + 0.5, REWARD_Y + 0.5, REWARD_CELL - 3, 43);
    }
    label(ctx, "value 10 at state 4 (home)", REWARD_X + 34, 178, COLOUR.agent, 13);
    ctx.strokeStyle = COLOUR.rule;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(REWARD_X, 226);
    ctx.lineTo(REWARD_X + N * REWARD_CELL - 2, 226);
    ctx.stroke();
    label(ctx, "real-valued scores", REWARD_X, 248, COLOUR.ink, 14);
    label(ctx, "not normalised", REWARD_X, 270, COLOUR.agent, 13);
  }

  drawWorld();
  drawPreferences();
  drawReward();

  const caption = document.createElement("p");
  caption.style.margin = "0.75rem 0 0";
  caption.style.color = COLOUR.dim;
  caption.textContent =
    "C has three competing probabilities, one per possible observation. " +
    "The reward vector is only a comparison, not part of this active-inference model: " +
    "here the agent's goal is C over observations.";
  el.appendChild(caption);
});

// Shared by the two figures below: the marks under every state vector, so a
// state keeps the same label in the same place wherever it is drawn.
const STATE_MARKS = [
  { index: HOME, label: "home" },
  { index: LAMPS[0], label: "lamp" },
  { index: LAMPS[1], label: "lamp" },
];

// One step east from the starting belief, the quantity both figures below
// are about. Computed once here so the two figures cannot disagree.
const EAST = 3;
const STEP = scorePolicy(START_BELIEF, B_LESSON4, A_LESSON4, C, [EAST]).trace[0];

// --- pragmatic-term -------------------------------------------------------
// Figure: where q(o | pi) comes from, and how it is scored against C.
// Serves the pragmatic half of "What the score is made of".
//
// Every state is drawn, all twenty-five, including the fourteen carrying no
// belief. The sum runs over all of them, and a row showing only the eleven
// that happen to be occupied would teach the wrong summation range: those
// terms are zero because q is zero there, not because the states are absent.
// Lesson 2 settled this convention deliberately (see its belief-sum row).
//
// Numbers are not printed under the bars: twenty-five columns leave no room.
// They appear in the readout on hover, in index order, every term shown.
mount("pragmatic-term", (el) => {
  const W = 760;
  const H = 598;
  const ctx = ctxOf(el, W, H);
  if (!ctx) return;
  ctx.canvas.setAttribute(
    "aria-label",
    "How the observations expected under a policy are computed from the " +
      "belief and the observation model A, and scored against the preference C.",
  );

  const CELL = 21;
  const X = 186;
  const BELIEF_Y = 104;
  const A_Y = 224;
  const A_CELL_H = 26;
  const OBS_Y = 430;
  const OBS_W = 58;
  const OBS_H = 42;

  const predicted = STEP.belief;
  const outcomes = STEP.outcomes;

  const readout = document.createElement("p");
  readout.style.margin = "0.75rem 0 0";
  readout.style.minHeight = "2.8em";
  readout.style.fontSize = "0.95rem";
  readout.style.lineHeight = "1.5";

  // hover is {o, s}: s null means the whole sum for observation o,
  // an index means one cell of A and the one belief bar above it.
  let hover = null;

  function setReadout(parts) {
    readout.textContent = "";
    for (const [text, kind] of parts) {
      const span = document.createElement("span");
      span.textContent = text;
      if (kind === "table") span.style.color = COLOUR.neutral;
      else if (kind === "dim") span.style.color = COLOUR.dim;
      else if (kind === "obs") span.style.color = COLOUR.litEdge;
      readout.appendChild(span);
    }
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    label(ctx, "what the agent expects to see", 22, 30, COLOUR.ink, 17);
    label(ctx, "and how much it wants it", 22, 50, COLOUR.dim, 13);

    // The belief over all 25 states.
    label(ctx, "q(s\u2081 | \u03c0)", 22, BELIEF_Y + 20, COLOUR.ink, 17);
    label(ctx, "after one step east", 22, BELIEF_Y + 38, COLOUR.dim, 12);
    drawStateStrip(ctx, X, BELIEF_Y, predicted, {
      cellW: CELL, barH: 40, colour: COLOUR.neutral,
    });
    drawStateMarks(ctx, X, BELIEF_Y + 40, CELL, STATE_MARKS);

    // A sparse index row: every fifth state, so a highlighted column can be
    // named. All twenty-five will not fit at this cell width.
    ctx.font = "11px system-ui, sans-serif";
    ctx.fillStyle = COLOUR.dim;
    ctx.textAlign = "center";
    for (let s = 0; s < N; s += 5) {
      ctx.fillText(String(s), X + s * CELL + (CELL - 4) / 2, BELIEF_Y - 6);
    }
    ctx.textAlign = "left";
    label(ctx, "s =", X - 26, BELIEF_Y - 6, COLOUR.dim, 11);

    // Ring the belief bars being read. One bar when a single cell of A is
    // hovered, every contributing bar when a whole sum is.
    if (hover !== null) {
      const ringed = hover.s !== null
        ? [hover.s]
        : [...Array(N).keys()].filter((s) => predicted[s] > 1e-9);
      ctx.strokeStyle = COLOUR.agent;
      ctx.lineWidth = hover.s !== null ? 2 : 1;
      for (const s of ringed) {
        ctx.strokeRect(X + s * CELL - 1.5, BELIEF_Y - 1.5, CELL - 4 + 3, 40 + 3);
      }
      if (hover.s !== null) {
        label(ctx, `s = ${hover.s}`, X + hover.s * CELL - 8, BELIEF_Y - 26, COLOUR.agent, 13);
        // The value goes beside the bar: `s = N` already occupies the space
        // above it. Near the right edge it flips to the left of the bar so
        // that it stays on the canvas.
        const text = predicted[hover.s].toFixed(3);
        ctx.font = "13px system-ui, sans-serif";
        const tw = ctx.measureText(text).width;
        const barRight = X + hover.s * CELL + CELL - 4;
        const flip = barRight + 6 + tw > X + N * CELL;
        label(ctx, text,
          flip ? X + hover.s * CELL - tw - 6 : barRight + 6,
          BELIEF_Y + 26, COLOUR.agent, 13);
      }
    }

    // A, three rows over the same 25 columns, so a column lines up with the
    // belief entry above it.
    label(ctx, "A[o][s]", 22, A_Y + 16, COLOUR.ink, 17);
    label(ctx, "what each state", 22, A_Y + 36, COLOUR.dim, 12);
    label(ctx, "would show", 22, A_Y + 52, COLOUR.dim, 12);
    for (let o = 0; o < 3; o++) {
      const y = A_Y + o * A_CELL_H;
      const rowOn = hover !== null && hover.o === o;
      label(ctx, OBS_NAMES[o], X - 48, y + 17, rowOn ? COLOUR.agent : COLOUR.dim, 13);
      for (let s = 0; s < N; s++) {
        const v = A_LESSON4[o][s];
        ctx.fillStyle = COLOUR.neutral;
        ctx.globalAlpha = 0.12 + 0.68 * v;
        ctx.fillRect(X + s * CELL, y, CELL - 4, A_CELL_H - 3);
        ctx.globalAlpha = 1;
      }
      if (rowOn && hover.s === null) {
        ctx.strokeStyle = COLOUR.agent;
        ctx.lineWidth = 2;
        ctx.strokeRect(X - 2, y - 2, N * CELL - 4 + 4, A_CELL_H - 3 + 4);
      }
    }
    // One cell, with its value printed: at this width the grid cannot carry
    // numbers, so the hovered one is written out where it sits.
    if (hover !== null && hover.s !== null) {
      const y = A_Y + hover.o * A_CELL_H;
      ctx.strokeStyle = COLOUR.agent;
      ctx.lineWidth = 2;
      ctx.strokeRect(X + hover.s * CELL - 1.5, y - 1.5, CELL - 4 + 3, A_CELL_H - 3 + 3);
      // Printed under the grid, not beside the cell: the columns touch, so a
      // value set next to one would overprint its neighbour.
      const text = A_LESSON4[hover.o][hover.s].toFixed(2);
      ctx.font = "13px system-ui, sans-serif";
      const tw = ctx.measureText(text).width;
      const centre = X + hover.s * CELL + (CELL - 4) / 2;
      const vx = Math.min(Math.max(centre - tw / 2, X), X + N * CELL - tw - 4);
      label(ctx, `A[${OBS_NAMES[hover.o]}][${hover.s}] = ${text}`,
        Math.min(vx - 34, X + N * CELL - tw - 90), A_Y + 3 * A_CELL_H + 34, COLOUR.agent, 13);
    }
    label(ctx, "darker = more likely.  each COLUMN sums to one: one reading is sampled per state",
      X, A_Y + 3 * A_CELL_H + 15, COLOUR.dim, 12);
    drawStateMarks(ctx, X, A_Y + 3 * A_CELL_H + 44, CELL, STATE_MARKS);

    // The expected observation, and the preference beside it.
    label(ctx, "q(o\u2081 | \u03c0) = \u2211\u209b A[o][s] \u00b7 q(s\u2081 | \u03c0)",
      22, OBS_Y - 26, COLOUR.ink, 15);
    for (let o = 0; o < 3; o++) {
      const x = X + o * OBS_W;
      const w = OBS_W - 8;
      const h = (outcomes[o] / Math.max(...outcomes, ...C)) * OBS_H;
      ctx.fillStyle = "#eef1f4";
      ctx.fillRect(x, OBS_Y, w, OBS_H);
      ctx.fillStyle = o === 1 ? COLOUR.lit : COLOUR.neutral;
      ctx.fillRect(x, OBS_Y + OBS_H - h, w, h);
      const barOn = hover !== null && hover.o === o;
      ctx.strokeStyle = barOn ? COLOUR.agent : COLOUR.rule;
      ctx.lineWidth = barOn ? 2 : 1;
      ctx.strokeRect(x + 0.5, OBS_Y + 0.5, w - 1, OBS_H - 1);
      ctx.textAlign = "center";
      ctx.font = "13px system-ui, sans-serif";
      ctx.fillStyle = COLOUR.dim;
      ctx.fillText(OBS_NAMES[o], x + w / 2, OBS_Y + OBS_H + 20);
      ctx.font = "14px system-ui, sans-serif";
      ctx.fillStyle = COLOUR.ink;
      ctx.fillText(outcomes[o].toFixed(4), x + w / 2, OBS_Y + OBS_H + 40);
      ctx.textAlign = "left";
    }

    // The pragmatic sum, written out with the three terms.
    const PX = 440;
    label(ctx, "pragmatic  =  \u2212 \u2211\u2092 q(o) \u00b7 ln C(o)", PX, OBS_Y - 26, COLOUR.ink, 15);
    let ty = OBS_Y + 12;
    for (let o = 0; o < 3; o++) {
      const term = -outcomes[o] * Math.log(C[o]);
      const colour = hover !== null && hover.o === o ? COLOUR.agent : COLOUR.dim;
      label(ctx, `\u2212 ${outcomes[o].toFixed(4)} \u00d7 ln ${C[o].toFixed(2)}`, PX, ty, colour, 13);
      label(ctx, `= +${term.toFixed(4)}`, PX + 160, ty, colour, 13);
      ty += 22;
    }
    ctx.strokeStyle = COLOUR.rule;
    ctx.beginPath();
    ctx.moveTo(PX, ty - 14);
    ctx.lineTo(PX + 230, ty - 14);
    ctx.stroke();
    label(ctx, "pragmatic", PX, ty + 6, COLOUR.ink, 15);
    label(ctx, `+${STEP.pragmatic.toFixed(4)}`, PX + 160, ty + 6, COLOUR.agent, 17);
    label(ctx, "C is the preference from figure 1", PX, ty + 30, COLOUR.dim, 12);
  }

  function defaultReadout() {
    setReadout([
      ["Hover a cell of A to trace one term, or an observation bar below to see the whole sum.", "dim"],
    ]);
  }

  // One cell: the A entry, the belief bar above it, and their product. This
  // is the smallest verifiable unit, and all three are highlighted at once.
  function traceCell(o, s) {
    setReadout([
      [`A[${OBS_NAMES[o]}][${s}] \u00b7 q(s\u2081=${s})`, "dim"],
      [":  ", "dim"],
      [A_LESSON4[o][s].toFixed(2), "table"],
      [" \u00d7 ", "dim"],
      [predicted[s].toFixed(5), null],
      ["  ≈  ", "dim"],
      [(A_LESSON4[o][s] * predicted[s]).toFixed(5), "obs"],
      [`     one of the 25 terms of q(o = ${OBS_NAMES[o]})`, "dim"],
    ]);
  }

  // Every term, in index order, including the zeros: the sum runs over all
  // twenty-five states and the readout should not suggest otherwise.
  function traceObservation(o) {
    const parts = [[`q(o = ${OBS_NAMES[o]}) = `, "dim"]];
    let shown = 0;
    for (let s = 0; s < N; s++) {
      if (predicted[s] <= 1e-9) continue;
      if (shown > 0) parts.push([" + ", "dim"]);
      parts.push([`s${s}:`, "dim"]);
      parts.push([A_LESSON4[o][s].toFixed(2), "table"]);
      parts.push(["\u00d7", "dim"]);
      parts.push([predicted[s].toFixed(5), null]);
      shown += 1;
    }
    const zeros = N - shown;
    parts.push([`  (+ ${zeros} terms where q(s) = 0)`, "dim"]);
    parts.push([`  ≈  ${outcomes[o].toFixed(5)}`, "obs"]);
    setReadout(parts);
  }

  ctx.canvas.addEventListener("mousemove", (event) => {
    const rect = ctx.canvas.getBoundingClientRect();
    const scale = ctx.canvas.width / rect.width;
    const mx = (event.clientX - rect.left) * scale;
    const my = (event.clientY - rect.top) * scale;
    let next = null;
    if (my >= A_Y && my < A_Y + 3 * A_CELL_H && mx >= X) {
      const o = Math.floor((my - A_Y) / A_CELL_H);
      const s = Math.floor((mx - X) / CELL);
      if (s >= 0 && s < N) next = { o, s };
    } else if (my >= OBS_Y && my <= OBS_Y + OBS_H && mx >= X) {
      const o = Math.floor((mx - X) / OBS_W);
      if (o >= 0 && o < 3) next = { o, s: null };
    }
    const same = JSON.stringify(next) === JSON.stringify(hover);
    if (!same) {
      hover = next;
      if (hover === null) defaultReadout();
      else if (hover.s === null) traceObservation(hover.o);
      else traceCell(hover.o, hover.s);
      draw();
    }
  });
  ctx.canvas.addEventListener("mouseleave", () => {
    if (hover !== null) { hover = null; defaultReadout(); draw(); }
  });

  el.appendChild(readout);
  defaultReadout();
  draw();
});

// --- epistemic-term -------------------------------------------------------
// Figure: what each observation would do to the belief, and why that is worth
// something. Serves the epistemic half of "What the score is made of".
//
// This is the half the pragmatic figure cannot show, because it needs a
// quantity that figure never draws: the posterior the agent would hold AFTER
// each observation. Three posteriors, one per observation, against the belief
// they all start from.
//
// H is shown before the divergence because a reader meeting "how far the
// belief moved" first has nothing to measure it against. Entropy is the
// spread of one belief; the divergence is the gap between two.
mount("epistemic-term", (el) => {
  const W = 760;
  const H = 580;
  const ctx = ctxOf(el, W, H);
  if (!ctx) return;
  ctx.canvas.setAttribute(
    "aria-label",
    "The belief before observing, and what it would become under each of the " +
      "three possible observations, with the entropy and divergence of each.",
  );

  const CELL = 20;
  const X = 150;
  const BEFORE_Y = 104;
  const AFTER_Y = 250;
  const ROW_GAP = 78;
  const BAR_H = 34;

  const predicted = STEP.belief;
  const outcomes = STEP.outcomes;

  // The posterior under each observation, by Bayes: q(s|o) proportional to
  // A[o][s] q(s), renormalised. The same update lesson 2 taught.
  const posteriors = [0, 1, 2].map((o) => {
    const un = predicted.map((q, s) => A_LESSON4[o][s] * q);
    const total = un.reduce((a, b) => a + b, 0);
    return un.map((v) => v / total);
  });
  const entropyOf = (p) => p.reduce((h, x) => h - (x > 1e-300 ? x * Math.log(x) : 0), 0);
  const klOf = (q, p) =>
    q.reduce((d, x, i) => d + (x > 1e-300 ? x * Math.log(x / p[i]) : 0), 0);

  const beforeH = entropyOf(predicted);
  const rows = posteriors.map((post, o) => ({
    o, post, h: entropyOf(post), kl: klOf(post, predicted), p: outcomes[o],
  }));
  // All four beliefs share one scale, so the lit posterior's collapse is a
  // real comparison and not an artefact of each row rescaling itself.
  const scale = Math.max(...predicted, ...posteriors.flat());

  const readout = document.createElement("p");
  readout.style.margin = "0.75rem 0 0";
  readout.style.minHeight = "2.8em";
  readout.style.fontSize = "0.95rem";
  readout.style.lineHeight = "1.5";
  let hover = null;

  function setReadout(parts) {
    readout.textContent = "";
    for (const [text, kind] of parts) {
      const span = document.createElement("span");
      span.textContent = text;
      if (kind === "table") span.style.color = COLOUR.neutral;
      else if (kind === "dim") span.style.color = COLOUR.dim;
      else if (kind === "obs") span.style.color = COLOUR.litEdge;
      readout.appendChild(span);
    }
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    label(ctx, "what each observation would do to the belief", 22, 30, COLOUR.ink, 17);
    label(ctx, "the agent cannot say which will arrive, so it averages over all three",
      22, 50, COLOUR.dim, 13);

    // Before.
    label(ctx, "before", 22, BEFORE_Y + 18, COLOUR.ink, 17);
    label(ctx, "q(s\u2081 | \u03c0)", 22, BEFORE_Y + 36, COLOUR.dim, 13);
    drawStateStrip(ctx, X, BEFORE_Y, predicted, {
      cellW: CELL, barH: BAR_H, scale, colour: COLOUR.neutral,
    });
    drawStateMarks(ctx, X, BEFORE_Y + BAR_H, CELL, STATE_MARKS);
    label(ctx, `H = ${beforeH.toFixed(3)}`, X + N * CELL + 8, BEFORE_Y + 22, COLOUR.ink, 14);

    label(ctx, "H is how spread a belief is:  H = \u2212 \u2211\u209b q(s) ln q(s)",
      22, BEFORE_Y + 80, COLOUR.dim, 12);
    label(ctx, "Three equal tallest bars: lamps 9, 13, and dark cell 17 (4,3).",
      22, BEFORE_Y + 98, COLOUR.dim, 12);
    label(ctx, "q(o) = \u2211\u209b A[o][s]q(s): the predicted probability of this observation.",
      22, AFTER_Y - 34, COLOUR.dim, 12);

    // After, one row per observation.
    label(ctx, "after", 22, AFTER_Y - 14, COLOUR.ink, 17);
    for (const row of rows) {
      const y = AFTER_Y + row.o * ROW_GAP;
      const on = hover === row.o;
      label(ctx, `see ${OBS_NAMES[row.o]}`, 22, y + 18, on ? COLOUR.agent : COLOUR.ink, 15);
      label(ctx, `q(o) = ${row.p.toFixed(4)}`, 22, y + 36, COLOUR.dim, 12);
      drawStateStrip(ctx, X, y, row.post, {
        cellW: CELL, barH: BAR_H, scale,
        colour: row.o === 1 ? COLOUR.lit : COLOUR.neutral,
      });
      drawStateMarks(ctx, X, y + BAR_H, CELL, STATE_MARKS);
      if (on) {
        ctx.strokeStyle = COLOUR.agent;
        ctx.lineWidth = 2;
        ctx.strokeRect(X - 3, y - 3, N * CELL - 4 + 6, BAR_H + 6);
      }
      label(ctx, `H = ${row.h.toFixed(3)}`, X + N * CELL + 8, y + 16, COLOUR.ink, 13);
      label(ctx, `moved ${row.kl.toFixed(4)}`, X + N * CELL + 8, y + 34,
        on ? COLOUR.agent : COLOUR.dim, 13);
    }

    // The sum.
    const SY = AFTER_Y + 3 * ROW_GAP + 10;
    label(ctx, "epistemic  =  \u2212 \u2211\u2092 q(o) \u00b7 D(o)", 22, SY, COLOUR.ink, 15);
    const terms = rows.map((r) => `\u2212 ${r.p.toFixed(3)}\u00d7${r.kl.toFixed(3)}`);
    label(ctx, terms.join("   "), 250, SY, COLOUR.dim, 13);
    label(ctx, `≈  ${STEP.epistemic.toFixed(4)}`, 250, SY + 24, COLOUR.agent, 17);
    label(ctx, "D(o) = KL[posterior || before]; q(o) weights each branch and the three q(o) sum to one.",
      22, SY + 48, COLOUR.dim, 12);
    label(ctx, "A divergence is never negative, so this term is never positive: learning can only lower G.",
      22, SY + 66, COLOUR.dim, 12);
  }

  function defaultReadout() {
    setReadout([["Hover an observation to see how its posterior is built.", "dim"]]);
  }

  // The posterior, by Bayes, in index order over every state that carries
  // belief. Same rule as lesson 2: multiply by the likelihood, renormalise.
  function tracePosterior(o) {
    const parts = [
      [`For observation ${OBS_NAMES[o]}: q(o) = `, "dim"],
      [outcomes[o].toFixed(5), "obs"],
      [" from \u03a3\u209b A[o][s]q(s).  Bayes: q(s|o) = A[o][s]q(s)/q(o).  Examples:  ", "dim"],
    ];
    let shown = 0;
    for (let s = 0; s < N && shown < 4; s++) {
      if (predicted[s] <= 1e-9) continue;
      if (shown > 0) parts.push(["   ", "dim"]);
      parts.push([`s=${s}: `, "dim"]);
      parts.push([A_LESSON4[o][s].toFixed(2), "table"]);
      parts.push(["\u00d7", "dim"]);
      parts.push([predicted[s].toFixed(5), null]);
      parts.push([`/${outcomes[o].toFixed(5)} ≈ `, "dim"]);
      parts.push([posteriors[o][s].toFixed(5), "obs"]);
      shown += 1;
    }
    setReadout(parts);
  }

  ctx.canvas.addEventListener("mousemove", (event) => {
    const rect = ctx.canvas.getBoundingClientRect();
    const sc = ctx.canvas.width / rect.width;
    const my = (event.clientY - rect.top) * sc;
    let next = null;
    for (const row of rows) {
      const y = AFTER_Y + row.o * ROW_GAP;
      if (my >= y - 4 && my <= y + BAR_H + 4) next = row.o;
    }
    if (next !== hover) {
      hover = next;
      if (hover === null) defaultReadout();
      else tracePosterior(hover);
      draw();
    }
  });
  ctx.canvas.addEventListener("mouseleave", () => {
    if (hover !== null) { hover = null; defaultReadout(); draw(); }
  });

  el.appendChild(readout);
  defaultReadout();
  draw();
});

// One source for the fixed sequences used by Figures 3, 4 and 5: for each
// possible first action, enumerate complete policies and keep the lowest G.
function bestFixedPoliciesByFirstAction(prior, B, A, preferences, horizon) {
  const best = Array.from({ length: B[0][0].length }, (_, action) => ({ action, G: Infinity }));
  function enumerate(policy) {
    if (policy.length === horizon) {
      const score = scorePolicy(prior, B, A, preferences, policy);
      const row = best[policy[0]];
      if (score.G < row.G) Object.assign(row, { ...score, policy: policy.slice() });
      return;
    }
    for (let action = 0; action < best.length; action++) enumerate([...policy, action]);
  }
  enumerate([]);
  return best;
}

// --- policies-compared ----------------------------------------------------
// Figure 3 uses the finalized Figure 6 world and its "See home marker" task:
// same A, B=.6, prior and C. The earlier one-step-crossover sketch was for a
// different world and does not hold here: east is best even at horizon one.
//
// Layout: three policies as COLUMNS, the horizon running DOWN the page. A
// column is one policy, so reading down it is watching that policy's belief
// spread step by step, and each belief gets a third of the page rather than
// an eighth. Only east, north and west are shown; five rows of 25-bar strips
// was too much to read at once.
//
// In fixed-sequence mode, each strip shows the next prediction and a running
// total G for that selected sequence. Changing the horizon selects a new
// complete sequence for each first action; the displayed score at a prefix
// is not necessarily the best score at that shorter horizon.
mount("policies-compared", (el) => {
  const W = 760;
  const H = 560;
  const ctx = ctxOf(el, W, H);
  if (!ctx) return;
  ctx.canvas.setAttribute("aria-label", "Three policies compared as columns, with the planning horizon running down the page.");

  const protocol = AGENT_WALKS.protocol;
  const prior = protocol.prior;
  const B = transitionModel({ rows: 5, cols: COLS, reliability: protocol.assumed_B_reliability });
  const preferences = AGENT_WALKS.modes.observations.C;
  const ACTIONS = ["north", "south", "west", "east", "stay"];
  const SHORT = ["N", "S", "W", "E", "STAY"];
  const OBS_SHORT = ["dark", "lit", "home"];
  // East is best here, north is the best route that does not start toward a
  // lamp, and west is the worst. Together they span the range without
  // showing five near-identical strips.
  const SHOWN = [3, 0, 2];
  const COL_X = [150, 355, 560];
  const COL_W = 185;
  const fixedByHorizon = new Map();
  const branchingByHorizon = new Map();

  function fixedRows(horizon) {
    const key = horizon;
    if (fixedByHorizon.has(key)) return fixedByHorizon.get(key);
    const rows = bestFixedPoliciesByFirstAction(prior, B, A_LESSON4, preferences, horizon);
    fixedByHorizon.set(key, rows);
    return rows;
  }

  function branchingRows(horizon) {
    if (branchingByHorizon.has(horizon)) return branchingByHorizon.get(horizon);
    const scores = sophisticatedValues(prior, B, A_LESSON4, preferences, horizon);
    const rows = scores.map((G, action) => {
      const firstBelief = predict(prior, B, action);
      // The observation-contingent continuation, one action per possible
      // observation rather than a single fixed sequence.
      let next = null;
      if (horizon > 1) {
        const q = expectedObs(A_LESSON4, firstBelief);
        next = q.map((po, o) => {
          // A reading the policy cannot produce has no continuation to show.
          if (!(po > 1e-9)) return null;
          const posterior = firstBelief.map((v, st) => A_LESSON4[o][st] * v / po);
          const values = sophisticatedValues(posterior, B, A_LESSON4, preferences, horizon - 1);
          const best = values.indexOf(Math.min(...values));
          return best >= 0 ? best : null;
        });
      }
      // The three beliefs the planner actually evaluates at step 2: one per
      // observation it might receive, each weighted by how likely it is. G is
      // their probability-weighted score, which is why it changes with the
      // horizon even though no single belief can be drawn for later steps.
      let branches = null;
      if (horizon > 1) {
        const q = expectedObs(A_LESSON4, firstBelief);
        branches = q.map((po, o) => po > 1e-9
          ? { p: po, belief: firstBelief.map((v, st) => A_LESSON4[o][st] * v / po) }
          : null);
      }
      return { action, G, firstBelief, next, branches };
    });
    branchingByHorizon.set(horizon, rows);
    return rows;
  }

  // Rank among ALL five actions, so hiding two does not silently change what
  // "best" means. A column showing rank 1 is best overall, not best of three.
  function rankOf(rows, action) {
    const sorted = [...rows].sort((a, b) => a.G - b.G || a.action - b.action);
    return sorted.findIndex((r) => r.action === action) + 1;
  }

  const controls = document.createElement("div");
  controls.style.display = "flex";
  controls.style.flexWrap = "wrap";
  controls.style.gap = "0.5rem 1.5rem";
  controls.style.marginBottom = "0.75rem";

  const modeLabel = document.createElement("label");
  modeLabel.textContent = "Planning rule: ";
  const modeSelect = document.createElement("select");
  modeSelect.setAttribute("aria-label", "Planning rule");
  for (const [value, title] of [
    ["fixed", "AIF: fixed sequence"],
    ["branching", "Sophisticated AIF: branch on observations"],
  ]) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = title;
    modeSelect.appendChild(option);
  }
  modeLabel.appendChild(modeSelect);

  const horizonLabel = document.createElement("label");
  horizonLabel.textContent = "Horizon: ";
  const horizonSlider = document.createElement("input");
  horizonSlider.type = "range";
  horizonSlider.min = "1";
  horizonSlider.max = "4";
  horizonSlider.step = "1";
  horizonSlider.value = "3";
  const horizonValue = document.createElement("output");
  horizonValue.textContent = "3";
  horizonLabel.append(horizonSlider, horizonValue);
  controls.append(modeLabel, horizonLabel);
  el.insertBefore(controls, ctx.canvas);

  const note = document.createElement("p");
  note.style.margin = "0.75rem 0 0";
  note.style.minHeight = "2.8em";
  el.appendChild(note);

  // A wide strip: 25 cells across a third of the page, with the lamp and home
  // cells marked underneath so a tall bar can be located.
  function beliefStrip(belief, x, y, width, height) {
    const gap = 1;
    const cell = (width - (N - 1) * gap) / N;
    ctx.fillStyle = "#eef1f4";
    ctx.fillRect(x, y, width, height);
    for (let s = 0; s < N; s++) {
      // Fixed scale across every strip: the prior's three cells hold 1/3 each.
      const h = Math.min(belief[s] * 3, 1) * (height - 3);
      ctx.fillStyle = LAMPS.includes(s) ? COLOUR.lit : s === HOME ? COLOUR.agent : COLOUR.neutral;
      ctx.fillRect(x + s * (cell + gap), y + height - h, cell, h);
    }
    ctx.strokeStyle = COLOUR.rule;
    ctx.strokeRect(x + 0.5, y + 0.5, width - 1, height - 1);
    // Tick the two lamps and home so the coloured bars are locatable.
    for (const s of [...LAMPS, HOME]) {
      ctx.fillStyle = s === HOME ? COLOUR.agent : COLOUR.lit;
      ctx.fillRect(x + s * (cell + gap), y + height + 1, cell, 2);
    }
  }

  function draw() {
    const horizon = Number(horizonSlider.value);
    const branching = modeSelect.value === "branching";
    const stripH = 46;
    const rowGap = 34;
    const top = branching && horizon > 1 ? 184 : horizon > 3 ? 167 : 152;
    const scoreY = top + horizon * (stripH + rowGap) + 16;
    const verdictY = scoreY + (branching && horizon > 1 ? 74 : 51);
    const canvasHeight = Math.max(360, verdictY + 54);
    if (ctx.canvas.height !== canvasHeight) ctx.canvas.height = canvasHeight;
    ctx.clearRect(0, 0, W, canvasHeight);
    horizonValue.textContent = String(horizon);
    const rows = branching ? branchingRows(horizon) : fixedRows(horizon);
    const oneStep = branching ? branchingRows(1) : fixedRows(1);

    label(ctx, "Home-marker task \u00b7 same A, B, C and starting belief as Figure 1", 22, 26, COLOUR.ink, 16);
    label(ctx, branching
      ? "Later actions branch on possible observations; numbers in parentheses are their probabilities"
      : "Each column is a complete policy chosen for this horizon; strips are predictions, not observations",
      22, 48, COLOUR.dim, 13);

    // The panel ends just below the score rows, so a short horizon does not
    // leave a tall empty box.
    const boxH = scoreY + (branching && horizon > 1 ? 56 : 30) - 62;

    // Column headers: the policy, and its rank among all five first actions.
    SHOWN.forEach((action, c) => {
      const row = rows.find((r) => r.action === action);
      const rank = rankOf(rows, action);
      const x = COL_X[c];
      ctx.fillStyle = rank === 1 ? "#eef5fb" : "#fff";
      ctx.fillRect(x - 12, 62, COL_W + 24, boxH);
      ctx.strokeStyle = COLOUR.rule;
      ctx.strokeRect(x - 11.5, 62.5, COL_W + 23, boxH - 1);
      const OBS = ["dark", "lit", "home"];
      label(ctx, ACTIONS[action], x, 86, rank === 1 ? COLOUR.agent : COLOUR.ink, 17);
      if (branching) {
        const q = expectedObs(A_LESSON4, row.firstBelief);
        // One line per reading: these are alternatives, not a sequence.
        (row.next || []).forEach((a, o) => {
          if (a === null) return;
          label(ctx, `if ${OBS[o]} (${q[o].toFixed(2)})  \u2192  ${SHORT[a]}`,
            x, 120 + o * 15, COLOUR.dim, 12);
        });
      } else {
        label(ctx, `policy: ${row.policy.map((a) => SHORT[a]).join(" → ")}`,
          x, 124, COLOUR.dim, 12);
      }
      label(ctx, rank === 1 ? "best of all 5 first actions" : `rank ${rank} of 5 first actions`,
        x, 101, rank === 1 ? COLOUR.agent : COLOUR.dim, 11);
    });

    // Rows are steps in the selected complete policy. Running G is the prefix
    // sum for that sequence, not a re-optimised shorter-horizon policy.
    label(ctx, branching ? "planning" : "after", 24, top - 18, COLOUR.dim, 13);
    label(ctx, branching ? "stage" : "step", 24, top - 4, COLOUR.dim, 13);
    for (let t = 0; t < horizon; t++) {
      const y = top + t * (stripH + rowGap);
      label(ctx, branching ? (t === 0 ? "move 1" : t === 1 ? "observe 1" : "later") : String(t + 1),
        118, y + stripH / 2 + 5, COLOUR.ink, branching ? 12 : 15, "right");
      SHOWN.forEach((action, c) => {
        const row = rows.find((r) => r.action === action);
        if (branching) {
          if (t === 0) {
            beliefStrip(row.firstBelief, COL_X[c], y, COL_W, stripH);
          } else if (t === 1 && row.branches) {
            // One strip per possible reading, labelled with its probability.
            const live = row.branches.map((b, o) => b && { ...b, o }).filter(Boolean);
            const bw = (COL_W - (live.length - 1) * 6) / live.length;
            live.forEach((br, k) => {
              const bx = COL_X[c] + k * (bw + 6);
              beliefStrip(br.belief, bx, y, bw, stripH);
              label(ctx, `${OBS_SHORT[br.o]} ${br.p.toFixed(2)}`, bx, y + stripH + 14, COLOUR.dim, 11);
            });
          } else if (t > 1) {
            label(ctx, "branches again", COL_X[c], y + stripH / 2 + 4, COLOUR.dim, 12);
          }
        } else {
          beliefStrip(row.trace[t].belief, COL_X[c], y, COL_W, stripH);
          const runningG = row.trace.slice(0, t + 1).reduce((sum, step) => sum + step.G, 0);
          label(ctx, `after ${SHORT[row.policy[t]]} · running G ${runningG.toFixed(4)}`,
            COL_X[c], y + stripH + 14, COLOUR.dim, 11);
        }
      });
    }

    // Fixed mode already shows the one-step prefix above. Branching mode has
    // no single later prefix, so retains the horizon-one comparison here.
    if (branching && horizon > 1) label(ctx, "G at horizon 1", 24, scoreY + 16, COLOUR.dim, 13);
    label(ctx, `total G at horizon ${horizon}`, 24,
      scoreY + (branching && horizon > 1 ? 42 : 20), COLOUR.ink, 13);
    SHOWN.forEach((action, c) => {
      const one = oneStep.find((r) => r.action === action);
      const now = rows.find((r) => r.action === action);
      if (branching && horizon > 1) {
        label(ctx, one.G.toFixed(4), COL_X[c], scoreY + 16,
          rankOf(oneStep, action) === 1 ? COLOUR.agent : COLOUR.dim, 14);
      }
      label(ctx, now.G.toFixed(4), COL_X[c],
        scoreY + (branching && horizon > 1 ? 42 : 20),
        rankOf(rows, action) === 1 ? COLOUR.agent : COLOUR.ink, 15);
    });

    const same = rankOf(oneStep, SHOWN[0]) === 1 && rankOf(rows, SHOWN[0]) === 1;
    label(ctx, horizon === 1
      ? "One step only. Raise the horizon to see whether later steps change this ordering."
      : same
        ? "East remains the best first action at this horizon."
        : "The ordering changed between horizon 1 and this horizon.",
      24, verdictY, same || horizon === 1 ? COLOUR.ink : COLOUR.agent, 13);
    label(ctx, "bars: 25 cells yellow ticks mark the two lamps and red home.",
      24, verdictY + 30, COLOUR.dim, 12);

    note.textContent = branching
      ? "Lower G is better. Each column starts with one action, then branches on its possible observations. The first strip is the belief predicted after moving; the next row shows alternative posteriors after observing, not step-two predictions. Later actions depend on the observation, and their costs are probability-weighted in G."
      : "Lower G is better. Each column is one fixed policy and each row is one further prediction. Running G adds each step's score for the complete sequence shown. Moving the horizon slider can select a different complete sequence; no imagined observation updates these belief strips.";
  }

  modeSelect.addEventListener("change", draw);
  horizonSlider.addEventListener("input", draw);
  draw();
});

// --- efe-signs ------------------------------------------------------------
// Figure 3 bis, between the algebra and Figure 4: use the same Figure 3
// observation task and the same three first actions, but score only one step.
// This isolates the signs. A more negative epistemic contribution helps;
// a smaller non-negative pragmatic cost also helps. Figure 4 will handle
// whole-policy totals and changing C, so this figure has no extra controls.
mount("efe-signs", (el) => {
  const W = 760;
  const H = 334;
  const ctx = ctxOf(el, W, H);
  if (!ctx) return;
  ctx.canvas.setAttribute("aria-label",
    "One-step expected free energy for east, north and west in Figure 3. " +
    "Epistemic contributions are zero or negative; pragmatic costs are positive. " +
    "East has the lowest total score.");

  const protocol = AGENT_WALKS.protocol;
  const B = transitionModel({ rows: 5, cols: COLS, reliability: protocol.assumed_B_reliability });
  const preferences = AGENT_WALKS.modes.observations.C;
  const rows = [
    { name: "east", action: 3 },
    { name: "north", action: 0 },
    { name: "west", action: 2 },
  ].map((item) => ({
    ...item,
    ...scorePolicy(protocol.prior, B, A_LESSON4, preferences, [item.action]),
  }));
  const scale = 95; // pixels per nat, shared by both signed terms
  const epiZero = 316;
  const praZero = 414;
  const rowTop = [104, 169, 234];

  function signedValue(value) {
    return Math.abs(value) < 0.00005 ? "0.0000" : `${value > 0 ? "+" : ""}${value.toFixed(4)}`;
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    label(ctx, "One step from Figure 3's starting belief", 22, 28, COLOUR.ink, 17);
    label(ctx, "Same A, B and observation preference C · lower G is better", 22, 50, COLOUR.dim, 13);
    label(ctx, "action", 31, 86, COLOUR.dim, 13);
    label(ctx, "epistemic  −information gain", 154, 86, COLOUR.dim, 13);
    label(ctx, "pragmatic  expected preference cost", 403, 86, COLOUR.dim, 13);
    label(ctx, "G", 675, 86, COLOUR.dim, 13);

    rows.forEach((row, i) => {
      const y = rowTop[i];
      ctx.fillStyle = i === 0 ? "#eef5fb" : i === 2 ? "#f7f9fb" : "#fff";
      ctx.fillRect(22, y, 716, 57);
      label(ctx, row.name, 34, y + 28, i === 0 ? COLOUR.agent : COLOUR.ink, 16);

      const epiWidth = Math.abs(row.epistemic) * scale;
      const praWidth = row.pragmatic * scale;
      ctx.fillStyle = COLOUR.agent;
      ctx.fillRect(epiZero - epiWidth, y + 11, epiWidth, 15);
      ctx.fillStyle = COLOUR.neutral;
      ctx.fillRect(praZero, y + 11, praWidth, 15);
      label(ctx, signedValue(row.epistemic), epiZero - 4, y + 46, COLOUR.ink, 13, "right");
      label(ctx, signedValue(row.pragmatic), praZero + 4, y + 46, COLOUR.ink, 13);
      label(ctx, row.G.toFixed(4), 676, y + 30,
        i === 0 ? COLOUR.agent : COLOUR.ink, 16);
    });
    for (const zero of [epiZero, praZero]) {
      ctx.strokeStyle = COLOUR.dim;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(zero + 0.5, 98);
      ctx.lineTo(zero + 0.5, 286);
      ctx.stroke();
    }
    label(ctx, "← more information lowers G", 157, 307, COLOUR.dim, 13);
    label(ctx, "← shorter cost lowers G", 414, 307, COLOUR.dim, 13);
  }
  draw();

  const caption = document.createElement("p");
  caption.style.margin = "0.75rem 0 0";
  caption.style.color = COLOUR.dim;
  caption.textContent = "Both bars use the same scale and zero lines. " +
    "The epistemic term is non-positive; the pragmatic cost is non-negative. " +
    "G is their sum, computed before rounding the displayed terms.";
  el.appendChild(caption);
});

// --- preference-counterfactual --------------------------------------------
// Figure 4 holds Figure 3's best fixed sequence for each of its three shown
// first actions constant at each chosen horizon. The slider changes the
// horizon and thus the sequences; only the C toggle holds them constant.
// With uniform preferences over the
// three possible observations, pragmatic = horizon * log(3) for every fixed
// policy, while the epistemic term and predicted beliefs are unchanged.
mount("preference-counterfactual", (el) => {
  el.classList.add("efe-counterfactual");
  const protocol = AGENT_WALKS.protocol;
  const B = transitionModel({ rows: 5, cols: COLS, reliability: protocol.assumed_B_reliability });
  const originalC = AGENT_WALKS.modes.observations.C;
  const uniformC = Array.from({ length: originalC.length }, () => 1 / originalC.length);
  const shown = [
    { action: 3, name: "east" },
    { action: 0, name: "north" },
    { action: 2, name: "west" },
  ];
  const short = ["N", "S", "W", "E", "stay"];
  const cache = new Map();

  // Same exhaustive selection as Figure 3's fixedRows. The selected policy
  // depends on the ORIGINAL C, never on the flatten-C toggle.
  function policies(horizon) {
    if (cache.has(horizon)) return cache.get(horizon);
    const best = bestFixedPoliciesByFirstAction(protocol.prior, B, A_LESSON4, originalC, horizon);
    const selected = shown.map(({ action, name }) => {
      const original = best[action];
      const flat = scorePolicy(protocol.prior, B, A_LESSON4, uniformC, original.policy);
      if (Math.abs(original.epistemic - flat.epistemic) > 1e-10 ||
          Math.abs(flat.pragmatic - horizon * Math.log(originalC.length)) > 1e-10) {
        throw new Error("flattening C changed predictions or failed the uniform-C identity");
      }
      return { name, policy: original.policy, original, flat };
    });
    cache.set(horizon, selected);
    return selected;
  }

  const title = document.createElement("h4");
  title.textContent = "Compare preferences at a fixed horizon";
  el.appendChild(title);

  const controls = document.createElement("div");
  controls.className = "efe-counterfactual-controls";
  const horizonLabel = document.createElement("label");
  horizonLabel.textContent = "Choose horizon ";
  const horizon = document.createElement("input");
  horizon.type = "range";
  horizon.min = "1";
  horizon.max = "4";
  horizon.step = "1";
  horizon.value = "3";
  horizon.setAttribute("aria-label", "Policy horizon");
  const horizonValue = document.createElement("output");
  horizonLabel.append(horizon, horizonValue);
  const flatLabel = document.createElement("label");
  const flatToggle = document.createElement("input");
  flatToggle.type = "checkbox";
  flatLabel.append(flatToggle, " Make C uniform");
  controls.append(horizonLabel, flatLabel);
  el.appendChild(controls);

  const preference = document.createElement("p");
  preference.className = "efe-counterfactual-preference";
  el.appendChild(preference);

  const scroll = document.createElement("div");
  scroll.className = "efe-counterfactual-scroll";
  const table = document.createElement("table");
  table.className = "efe-counterfactual-table";
  const head = document.createElement("thead");
  const headerRow = document.createElement("tr");
  for (const heading of ["Figure 3 fixed sequence", "epistemic ≤ 0", "pragmatic ≥ 0", "G ↓"]) {
    const th = document.createElement("th");
    th.textContent = heading;
    headerRow.appendChild(th);
  }
  head.appendChild(headerRow);
  const body = document.createElement("tbody");
  table.append(head, body);
  scroll.appendChild(table);
  el.appendChild(scroll);

  const explanation = document.createElement("p");
  explanation.className = "efe-counterfactual-explanation";
  el.appendChild(explanation);

  function termCell(value, maxMagnitude, negative) {
    const td = document.createElement("td");
    const track = document.createElement("span");
    track.className = `efe-counterfactual-track ${negative ? "negative" : "positive"}`;
    const bar = document.createElement("span");
    bar.className = "efe-counterfactual-bar";
    bar.style.width = `${100 * Math.abs(value) / maxMagnitude}%`;
    track.appendChild(bar);
    const number = document.createElement("span");
    number.className = "efe-counterfactual-number";
    number.textContent = Math.abs(value) < 0.00005 ? "0.0000" :
      `${value > 0 ? "+" : ""}${value.toFixed(4)}`;
    td.append(track, number);
    return td;
  }

  function draw() {
    const h = Number(horizon.value);
    const flattened = flatToggle.checked;
    horizonValue.value = String(h);
    horizonValue.textContent = String(h);
    const selected = policies(h);
    const scores = selected.map((row) => flattened ? row.flat : row.original);
    // Use one bar scale for both terms AND both C settings at this horizon.
    // Toggling C cannot visually rescale an unchanged epistemic contribution.
    const maxMagnitude = Math.max(...selected.flatMap((row) => [
      Math.abs(row.original.epistemic), row.original.pragmatic,
      Math.abs(row.flat.epistemic), row.flat.pragmatic,
    ]));
    preference.textContent = flattened
      ? `Uniform C: dark = lit = home marker = 1/3. Every policy's pragmatic cost is ${h} × ln 3 = ${(h * Math.log(3)).toFixed(4)}.`
      : `Figure 3 C: dark ${originalC[0].toFixed(3)}, lit ${originalC[1].toFixed(3)}, home marker ${originalC[2].toFixed(3)}.`;
    body.replaceChildren();
    selected.forEach((row, i) => {
      const score = scores[i];
      const tr = document.createElement("tr");
      if (i === 0) tr.className = "best";
      const policyCell = document.createElement("td");
      const name = document.createElement("strong");
      name.textContent = row.name;
      const sequence = document.createElement("small");
      sequence.textContent = row.policy.map((a) => short[a]).join(" → ");
      policyCell.append(name, sequence);
      const total = document.createElement("td");
      total.className = "efe-counterfactual-total";
      total.textContent = score.G.toFixed(4);
      tr.append(policyCell,
        termCell(score.epistemic, maxMagnitude, true),
        termCell(score.pragmatic, maxMagnitude, false), total);
      body.appendChild(tr);
    });
    explanation.textContent = flattened
      ? "At this horizon, the C toggle keeps the action sequences and epistemic values fixed. Uniform C gives every row the same pragmatic cost, so only expected information gain separates their G scores."
      : "The horizon slider selects Figure 3's best fixed sequences for east, north and west at that length. The C toggle then rescores those same sequences without choosing new ones.";
  }
  horizon.addEventListener("input", draw);
  flatToggle.addEventListener("change", draw);
  draw();
});

// --- scores-to-probabilities ----------------------------------------------
// Figure 5 uses the three fixed sequences shown in Figure 4 at its default
// horizon 3 and original C. The softmax here is conditional on these three
// candidates; Figure 6's replay agents instead rank all policies and act
// greedily. Gamma changes this display's probabilities, never its G scores.
mount("scores-to-probabilities", (el) => {
  el.classList.add("policy-precision");
  const protocol = AGENT_WALKS.protocol;
  const B = transitionModel({ rows: 5, cols: COLS, reliability: protocol.assumed_B_reliability });
  const preferences = AGENT_WALKS.modes.observations.C;
  const best = bestFixedPoliciesByFirstAction(protocol.prior, B, A_LESSON4, preferences, 3);
  const short = ["N", "S", "W", "E", "stay"];
  const candidates = [
    { name: "east", ...best[3] },
    { name: "north", ...best[0] },
    { name: "west", ...best[2] },
  ];
  const scores = candidates.map((row) => row.G);

  const title = document.createElement("h4");
  title.textContent = "The scores stay still; the probabilities move";
  el.appendChild(title);

  const control = document.createElement("label");
  control.className = "policy-precision-control";
  const gammaName = document.createElement("span");
  gammaName.textContent = "Precision γ";
  const gamma = document.createElement("input");
  gamma.type = "range";
  gamma.min = "0.1";
  gamma.max = "8";
  gamma.step = "0.1";
  gamma.value = "0.2";
  gamma.setAttribute("aria-label", "Policy precision gamma");
  const gammaValue = document.createElement("output");
  control.append(gammaName, gamma, gammaValue);
  el.appendChild(control);

  const context = document.createElement("p");
  context.className = "policy-precision-context";
  context.textContent = "Figure 4's three original-C sequences at horizon 3. " +
    "The softmax is over these three displayed candidates only.";
  el.appendChild(context);

  const scroll = document.createElement("div");
  scroll.className = "policy-precision-scroll";
  const table = document.createElement("table");
  table.className = "policy-precision-table";
  const header = document.createElement("thead");
  const headerRow = document.createElement("tr");
  for (const heading of ["fixed policy", "G (unchanged)", "q(π) among these three"]) {
    const th = document.createElement("th");
    th.scope = "col";
    th.textContent = heading;
    headerRow.appendChild(th);
  }
  header.appendChild(headerRow);
  const body = document.createElement("tbody");
  const bars = [];
  const probabilities = [];
  candidates.forEach((candidate, i) => {
    const row = document.createElement("tr");
    if (i === 0) row.className = "best";
    const policyCell = document.createElement("td");
    const name = document.createElement("strong");
    name.textContent = candidate.name;
    const sequence = document.createElement("small");
    sequence.textContent = candidate.policy.map((a) => short[a]).join(" → ");
    policyCell.append(name, sequence);
    const scoreCell = document.createElement("td");
    scoreCell.className = "policy-precision-score";
    scoreCell.textContent = candidate.G.toFixed(4);
    const probabilityCell = document.createElement("td");
    const track = document.createElement("span");
    track.className = "policy-precision-track";
    track.setAttribute("aria-hidden", "true");
    const bar = document.createElement("span");
    bar.className = "policy-precision-bar";
    track.appendChild(bar);
    const number = document.createElement("span");
    number.className = "policy-precision-number";
    probabilityCell.append(track, number);
    row.append(policyCell, scoreCell, probabilityCell);
    body.appendChild(row);
    bars.push(bar);
    probabilities.push(number);
  });
  table.append(header, body);
  scroll.appendChild(table);
  el.appendChild(scroll);

  const readout = document.createElement("p");
  readout.className = "policy-precision-readout";
  el.appendChild(readout);

  function draw() {
    const precision = Number(gamma.value);
    gammaValue.value = precision.toFixed(1);
    gammaValue.textContent = precision.toFixed(1);
    const q = softmax(scores, precision);
    q.forEach((probability, i) => {
      bars[i].style.width = `${100 * probability}%`;
      probabilities[i].textContent = probability.toFixed(5);
    });
    const sum = q.reduce((total, probability) => total + probability, 0);
    readout.textContent = `Before display rounding, these three probabilities sum to ${sum.toFixed(5)}. ` +
      (precision < 1
        ? "Low positive precision gives the policies similar weight; they approach equal probabilities as γ approaches zero."
        : "Higher precision concentrates probability on the lowest-G policy, without changing any G score.");
  }
  gamma.addEventListener("input", draw);
  draw();
});

mountAll();

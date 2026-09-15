/**
 * Shared infrastructure for the lesson widgets.
 *
 * Each <widget id="NAME"> in a lesson becomes a <div data-widget="NAME">;
 * a page's own {slug}.js registers a function per id via mount(), and
 * mountAll() below runs them on load. Rendering is plain Canvas 2D. The
 * belief math comes from aif.js, the same code the Python is checked
 * against.
 */

import { uniformBelief, update, lampObservationModel } from "./aif.js";

export const registry = new Map();
export function mount(id, fn) {
  registry.set(id, fn);
}

export function ctxOf(el, width, height) {
  let canvas = el.querySelector("canvas");
  if (!canvas) {
    canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    canvas.style.maxWidth = "100%";
    el.appendChild(canvas);
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) console.error("no 2d context for widget");
  return ctx;
}

export function makeSlider(el, label, value) {
  const wrap = document.createElement("label");
  wrap.style.display = "block";
  wrap.style.margin = "0 0 0.75rem";
  wrap.textContent = `${label}: `;
  const input = document.createElement("input");
  input.type = "range";
  input.min = "0";
  input.max = "0.6";
  input.step = "0.05";
  input.value = String(value);
  const out = document.createElement("span");
  out.textContent = value.toFixed(2);
  wrap.append(input, out);
  el.appendChild(wrap);
  input.addEventListener("input", () => {
    out.textContent = Number(input.value).toFixed(2);
  });
  return input;
}

// ===========================================================================
// Lamp gridworld: every tile is lit or dark, the agent senses only which, and
// it never moves. Two observations, DARK and LIT. Shared across every lesson
// that reuses this scenario, not specific to any one page. The constants are
// fixed because lesson prose quotes numbers that depend on them: three lit
// tiles, a uniform prior, and this exact reading sequence.
// ===========================================================================

export const ROWS = 5;
export const COLS = 5;
export const N = ROWS * COLS;
export const DARK = 0;
export const LIT = 1;
export const LIT_TILES = [10, 19, 22];
export const AGENT_TILE = 10; // a side (edge, non-corner) lit tile
export const SEQUENCE = [LIT, LIT, DARK, LIT, DARK, LIT, LIT];
export const LIT_MASK = Array.from({ length: N }, (_, s) => LIT_TILES.includes(s));
export const A_LAMP = lampObservationModel(LIT_MASK);

// The featureless world: no tile is lit, so every column of A is identical and
// a reading carries no information about which tile the agent is on. This is
// perceptual aliasing in its purest form, and it reuses the lamp machinery
// rather than adding a second observation model.
export const DARK_MASK = Array.from({ length: N }, () => false);
export const A_FEATURELESS = lampObservationModel(DARK_MASK);

// One palette for every lamp widget, so a colour always means the same thing.
// Anything carrying an observation takes that observation's colour; everything
// else stays neutral, so the coloured row reads as the evidence coming in.
export const COLOUR = {
  lit: "#FFDB58",
  dark: "#111",
  litEdge: "#8a6d0b",
  agent: "#e8563f",
  neutral: "#4c72b0",
  ink: "#1a1a1a",
  dim: "#5a5a5a",
  rule: "#d8d8d8",
};
export const FONT = "14px system-ui, sans-serif"; // one size up from the original 12px
export const obsColour = (o) => (o === LIT ? COLOUR.lit : COLOUR.dark);
export const obsName = (o) => (o === LIT ? "lit" : "dark");

// The beliefs for the whole fixed sequence, folded once from the uniform prior.
// Stepping backwards is then free, and every widget shows the same numbers.
export function lampBeliefs() {
  const out = [uniformBelief(N)];
  let b = out[0];
  for (const o of SEQUENCE) {
    b = update(b, A_LAMP, o).posterior;
    out.push(b);
  }
  return out;
}

// A flat 5x5 grid. `opts.agent` draws the red circle, `opts.observed` overrides
// what the agent's own tile is showing this step, `opts.mark` rings one tile.
// `opts.showIds` prints the tile index (0..24) in each cell, matching the
// column index printed above the A matrix, so the two can be read together.
export function drawGrid(ctx, x, y, cell, opts = {}) {
  const { agent = null, observed = null, mark = null, lit = LIT_MASK, showIds = false } = opts;
  for (let s = 0; s < N; s++) {
    const r = Math.floor(s / COLS);
    const c = s % COLS;
    const isLit = s === agent && observed !== null ? observed === LIT : lit[s];
    ctx.fillStyle = isLit ? COLOUR.lit : COLOUR.dark;
    ctx.fillRect(x + c * cell, y + r * cell, cell - 2, cell - 2);
    if (isLit) {
      ctx.strokeStyle = COLOUR.litEdge;
      ctx.lineWidth = 1;
      ctx.strokeRect(x + c * cell + 0.5, y + r * cell + 0.5, cell - 3, cell - 3);
    }
    if (showIds) {
      // Bold, white, centred, with a dark outline: readable on both the
      // near-black dark tiles and the pale-yellow lit ones, matching the
      // treatment of the probability numbers in the A matrix.
      const cx = x + c * cell + (cell - 2) / 2;
      const cy = y + r * cell + (cell - 2) / 2;
      ctx.font = `bold ${Math.round(cell * 0.4)}px system-ui, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.lineWidth = 3;
      ctx.strokeStyle = "rgba(0,0,0,0.55)";
      ctx.strokeText(String(s), cx, cy);
      ctx.fillStyle = "#fff";
      ctx.fillText(String(s), cx, cy);
      ctx.textBaseline = "alphabetic";
      ctx.textAlign = "left";
    }
  }
  if (mark !== null) {
    const r = Math.floor(mark / COLS);
    const c = mark % COLS;
    ctx.strokeStyle = COLOUR.agent;
    ctx.lineWidth = 2;
    ctx.strokeRect(x + c * cell - 1, y + r * cell - 1, cell, cell);
  }
  if (agent !== null) {
    const r = Math.floor(agent / COLS);
    const c = agent % COLS;
    ctx.fillStyle = COLOUR.agent;
    ctx.beginPath();
    ctx.arc(x + c * cell + (cell - 2) / 2, y + r * cell + (cell - 2) / 2, cell / 4, 0, 2 * Math.PI);
    ctx.fill();
  }
}

// A belief drawn as one bar per cell, scaled so the tallest bar in this
// specific belief fills the band: the two callers (a real posterior, and a
// one-tile spike) have very different peak heights and both need to read as
// "this fills the picture", not "this is a sliver next to empty space".
export function drawBeliefBar(ctx, x, y, w, belief) {
  const h = 64;
  const bw = w / N;
  const maxV = Math.max(...belief, 1e-9);
  for (let s = 0; s < N; s++) {
    const bh = (belief[s] / maxV) * h;
    ctx.fillStyle = LIT_MASK[s] ? COLOUR.lit : COLOUR.neutral;
    ctx.fillRect(x + s * bw, y + h - bh, Math.max(1, bw - 2), bh);
    if (LIT_MASK[s]) {
      ctx.strokeStyle = COLOUR.litEdge;
      ctx.lineWidth = 1;
      ctx.strokeRect(x + s * bw + 0.5, y + h - bh + 0.5, Math.max(1, bw - 2) - 1, Math.max(0, bh - 1));
    }
  }
  ctx.strokeStyle = COLOUR.rule;
  ctx.beginPath();
  ctx.moveTo(x, y + h + 0.5);
  ctx.lineTo(x + w, y + h + 0.5);
  ctx.stroke();
}

// A one-hot belief: certainty on a single tile, 0 everywhere else. Passed to
// drawBeliefBar the same way a real posterior is, so committing to one answer
// renders on the same footing as the ordinary belief: same bar chart, same
// scale rule, just a spike instead of a smooth shape.
export function oneHot(tile) {
  const dist = new Array(N).fill(0);
  dist[tile] = 1;
  return dist;
}

// Isometric projection of the 5x5 plane. Returns the screen point for a grid
// coordinate at height h (0..1). Tiles are drawn as diamonds, back to front.
function iso(row, col, h, ox, oy, cell, lift) {
  return [ox + (col - row) * cell * 0.9, oy + (col + row) * cell * 0.5 - h * lift];
}

export function drawIsoPlane(ctx, ox, oy, cell, opts = {}) {
  const { heights = null, agent = null, observed = null, lit = LIT_MASK } = opts;
  // Back to front so nearer tiles and taller bumps overdraw what is behind.
  const order = [];
  for (let r = 0; r < ROWS; r++) for (let c = 0; c < COLS; c++) order.push([r, c]);
  order.sort((a, b) => a[0] + a[1] - (b[0] + b[1]));
  const lift = cell * 2.2;

  for (const [r, c] of order) {
    const s = r * COLS + c;
    const isLit = s === agent && observed !== null ? observed === LIT : lit[s];
    const p = [
      iso(r, c, 0, ox, oy, cell, lift),
      iso(r, c + 1, 0, ox, oy, cell, lift),
      iso(r + 1, c + 1, 0, ox, oy, cell, lift),
      iso(r + 1, c, 0, ox, oy, cell, lift),
    ];
    ctx.beginPath();
    ctx.moveTo(p[0][0], p[0][1]);
    for (let i = 1; i < 4; i++) ctx.lineTo(p[i][0], p[i][1]);
    ctx.closePath();
    ctx.fillStyle = heights ? "#efefec" : isLit ? COLOUR.lit : COLOUR.dark;
    ctx.fill();
    ctx.strokeStyle = COLOUR.rule;
    ctx.lineWidth = 1;
    ctx.stroke();

    if (agent === s && !heights) {
      const [ax, ay] = iso(r + 0.5, c + 0.5, 0, ox, oy, cell, lift);
      ctx.fillStyle = COLOUR.agent;
      ctx.beginPath();
      ctx.ellipse(ax, ay, cell * 0.3, cell * 0.2, 0, 0, 2 * Math.PI);
      ctx.fill();
    }

    // A gaussian bump. A stack of closely spaced ellipses from the tile
    // upwards, so it reads as one surface. Both the height AND the footprint
    // scale with the probability: a 0.1 bump must look small next to a 0.8
    // one, not merely shorter, or the picture contradicts the numbers.
    if (heights) {
      const h = heights[s];
      if (h > 1e-3) {
        const [bx, by] = iso(r + 0.5, c + 0.5, 0, ox, oy, cell, lift);
        const peak = h * lift;
        const foot = cell * 0.9 * (0.35 + 0.65 * h);
        const steps = 20;
        for (let k = 0; k <= steps; k++) {
          const t = k / steps;           // 0 at the base, 1 at the peak
          const y = by - peak * t;
          const wide = Math.sqrt(Math.max(0, -Math.log(Math.max(t, 1e-3))) / 3);
          const rx = foot * Math.min(1, wide);
          if (rx < 0.4) continue;
          ctx.beginPath();
          ctx.ellipse(bx, y, rx, rx * 0.5, 0, 0, 2 * Math.PI);
          ctx.fillStyle = `rgba(58, 94, 160, ${0.18 + 0.5 * t})`;
          ctx.fill();
        }
      }
    }
  }
}

// Run stepFn(i) on a timer, then leave a replay button behind.
export function autoplay(el, stepFn, nSteps, ms) {
  const button = document.createElement("button");
  button.textContent = "replay";
  el.appendChild(button);
  let timer = null;
  function run() {
    if (timer) clearInterval(timer);
    let i = 0;
    stepFn(i);
    timer = setInterval(() => {
      i += 1;
      if (i >= nSteps) {
        clearInterval(timer);
        timer = null;
        stepFn(nSteps - 1, true);
        return;
      }
      stepFn(i);
    }, ms);
  }
  button.addEventListener("click", run);
  run();
  return button;
}

export function label(ctx, text, x, y, colour = COLOUR.dim, size = 16) {
  ctx.fillStyle = colour;
  ctx.font = `${size}px system-ui, sans-serif`;
  ctx.textAlign = "left";
  ctx.fillText(text, x, y);
}

// Shared matrix drawing for a-columns and a-rows. Rows are observations in the
// order [dark, lit]; columns are tiles. Cells take their observation's colour,
// with opacity standing for the probability. MX/MY are the default origin;
// a-columns overrides y to sit the matrix under its own grid.
export const MX = 70;
export const MY = 90;
export const MCELL = 40;

// The A matrix, drawn with axes: a tile index above every column, "dark"/"lit"
// row labels on the left, and the probability printed on each cell, large and
// white with a dark outline so it reads against both the dark and lit fills.
export function drawMatrix(ctx, { markCol = null, markRow = null, upTo = null, x = MX, y = MY } = {}) {
  ctx.font = "14px system-ui, sans-serif";
  ctx.textAlign = "center";
  for (let s = 0; s < N; s++) {
    ctx.fillStyle = COLOUR.dim;
    ctx.fillText(String(s), x + s * MCELL + MCELL / 2, y - 16);
  }
  for (let o = 0; o < 2; o++) {
    for (let s = 0; s < N; s++) {
      const v = A_LAMP[o][s];
      const dim = upTo !== null && markRow === o && s > upTo;
      ctx.globalAlpha = dim ? 0.12 : 0.25 + 0.75 * v;
      ctx.fillStyle = obsColour(o);
      ctx.fillRect(x + s * MCELL, y + o * MCELL, MCELL - 1, MCELL - 1);
      ctx.globalAlpha = 1;
      if (o === LIT) {
        ctx.strokeStyle = COLOUR.litEdge;
        ctx.globalAlpha = dim ? 0.2 : 0.6;
        ctx.lineWidth = 1;
        ctx.strokeRect(x + s * MCELL + 0.5, y + o * MCELL + 0.5, MCELL - 2, MCELL - 2);
        ctx.globalAlpha = 1;
      }
      if (!dim) {
        const cx = x + s * MCELL + MCELL / 2;
        const cy = y + o * MCELL + MCELL / 2;
        ctx.font = "bold 15px system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        // A dark outline under white fill reads on every cell colour, from
        // near-black (p=0.9 dark) through pale cream (p=0.1 dark).
        ctx.lineWidth = 3;
        ctx.strokeStyle = "rgba(0,0,0,0.55)";
        ctx.strokeText(v.toFixed(1), cx, cy);
        ctx.fillStyle = "#fff";
        ctx.fillText(v.toFixed(1), cx, cy);
        ctx.textBaseline = "alphabetic";
      }
    }
  }
  ctx.textAlign = "left";
  label(ctx, "dark", x - 46, y + MCELL / 2 + 4, COLOUR.ink);
  label(ctx, "lit", x - 46, y + MCELL + MCELL / 2 + 4, COLOUR.ink);
  label(ctx, "tile (column) index, 0 to 24", x, y + 2 * MCELL + 22);
  if (markCol !== null) {
    ctx.strokeStyle = COLOUR.agent;
    ctx.lineWidth = 2;
    ctx.strokeRect(x + markCol * MCELL - 1, y - 1, MCELL + 1, 2 * MCELL + 1);
  }
  if (markRow !== null) {
    ctx.strokeStyle = COLOUR.agent;
    ctx.lineWidth = 2;
    ctx.strokeRect(x - 1, y + markRow * MCELL - 1, N * MCELL + 1, MCELL + 1);
  }
}

// --- dispatch ----------------------------------------------------------
// Call this from a page's own {slug}.js, after its mount() calls, so every
// widget is registered before the DOM gets scanned. A page module script
// runs after widget.js has finished evaluating, so registering here and
// calling from there is the ordering the split guarantees, not readyState.
export function mountAll() {
  for (const el of document.querySelectorAll("[data-widget]")) {
    const id = el.dataset.widget;
    const fn = registry.get(id);
    if (!fn) {
      console.error(`no widget registered for "${id}"`);
      continue;
    }
    try {
      fn(el);
    } catch (err) {
      console.error(`widget "${id}" failed:`, err);
    }
  }
}

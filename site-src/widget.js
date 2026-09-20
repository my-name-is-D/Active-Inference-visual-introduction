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
// Lamp gridworld: every cell is lit or dark, the agent senses only which, and
// it never moves. Two observations, DARK and LIT. Shared across every lesson
// that reuses this scenario, not specific to any one page. The constants are
// fixed because lesson prose quotes numbers that depend on them: three lit
// cells, a uniform prior, and this exact reading sequence.
// ===========================================================================

export const ROWS = 5;
export const COLS = 5;
export const N = ROWS * COLS;
export const DARK = 0;
export const LIT = 1;
export const LIT_TILES = [10, 19, 22];
export const AGENT_TILE = 10; // a side (edge, non-corner) lit cell
export const SEQUENCE = [LIT, LIT, DARK, LIT, DARK, LIT, LIT];
export const LIT_MASK = Array.from({ length: N }, (_, s) => LIT_TILES.includes(s));
export const A_LAMP = lampObservationModel(LIT_MASK);

// The featureless world: no cell is lit, so every column of A is identical and
// a reading carries no information about which cell the agent is on. This is
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
// what the agent's own cell is showing this step, `opts.mark` rings one cell.
// `opts.showIds` prints the cell index (0..24) in each cell, matching the
// column index printed above the A matrix, so the two can be read together.
export function drawGrid(ctx, x, y, cell, opts = {}) {
  const { agent = null, observed = null, mark = null, lit = LIT_MASK,
          showIds = false, home = null, homeHidden = false } = opts;
  for (let s = 0; s < N; s++) {
    const r = Math.floor(s / COLS);
    const c = s % COLS;
    const isLit = s === agent && observed !== null ? observed === LIT : lit[s];
    ctx.fillStyle = s === home ? (homeHidden ? '#334139' : '#32738a')
      : isLit ? COLOUR.lit : COLOUR.dark;
    ctx.fillRect(x + c * cell, y + r * cell, cell - 2, cell - 2);
    if (s === home) {
      ctx.fillStyle = '#fff';
      ctx.font = `bold ${Math.round(cell * 0.32)}px system-ui, sans-serif`;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText('H', x + c * cell + 3, y + r * cell + 2);
      ctx.textBaseline = 'alphabetic';
    }
    if (isLit) {
      ctx.strokeStyle = COLOUR.litEdge;
      ctx.lineWidth = 1;
      ctx.strokeRect(x + c * cell + 0.5, y + r * cell + 0.5, cell - 3, cell - 3);
    }
    if (showIds) {
      // Bold, white, centred, with a dark outline: readable on both the
      // near-black dark cells and the pale-yellow lit ones, matching the
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
// one-cell spike) have very different peak heights and both need to read as
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

// A one-hot belief: certainty on a single cell, 0 everywhere else. Passed to
// drawBeliefBar the same way a real posterior is, so committing to one answer
// renders on the same footing as the ordinary belief: same bar chart, same
// scale rule, just a spike instead of a smooth shape.
export function oneHot(cell) {
  const dist = new Array(N).fill(0);
  dist[cell] = 1;
  return dist;
}

// Isometric projection of the 5x5 plane. Returns the screen point for a grid
// coordinate at height h (0..1). Cells are drawn as diamonds, back to front.
function iso(row, col, h, ox, oy, cell, lift) {
  return [ox + (col - row) * cell * 0.9, oy + (col + row) * cell * 0.5 - h * lift];
}

export function drawIsoPlane(ctx, ox, oy, cell, opts = {}) {
  const { heights = null, agent = null, observed = null, lit = LIT_MASK,
          home = null, homeHidden = false } = opts;
  // Back to front so nearer cells and taller bumps overdraw what is behind.
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
    ctx.fillStyle = heights ? "#efefec" : s === home && homeHidden ? "#334139"
      : isLit ? COLOUR.lit : COLOUR.dark;
    ctx.fill();
    ctx.strokeStyle = s === home && !heights ? COLOUR.agent : COLOUR.rule;
    ctx.lineWidth = s === home && !heights ? 2.5 : 1;
    ctx.stroke();

    if (s === home && !heights) {
      const [hx, hy] = iso(r + 0.5, c + 0.5, 0, ox, oy, cell, lift);
      ctx.fillStyle = '#fff';
      ctx.font = `bold ${Math.round(cell * 0.5)}px system-ui, sans-serif`;
      ctx.textAlign = 'center';
      ctx.fillText('H', hx, hy - cell * 0.22);
      ctx.textAlign = 'left';
    }

    if (agent === s && !heights) {
      const [ax, ay] = iso(r + 0.5, c + 0.5, 0, ox, oy, cell, lift);
      ctx.fillStyle = COLOUR.agent;
      ctx.beginPath();
      ctx.ellipse(ax, ay, cell * 0.3, cell * 0.2, 0, 0, 2 * Math.PI);
      ctx.fill();
    }

    // A gaussian bump. A stack of closely spaced ellipses from the cell
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

export function label(ctx, text, x, y, colour = COLOUR.dim, size = 16, align = "left") {
  ctx.fillStyle = colour;
  ctx.font = `${size}px system-ui, sans-serif`;
  ctx.textAlign = align;
  ctx.fillText(text, x, y);
  ctx.textAlign = "left";   // restore: callers below assume the default
}

// Shared matrix drawing for a-columns and a-rows. Rows are observations in the
// order [dark, lit]; columns are cells. Cells take their observation's colour,
// with opacity standing for the probability. MX/MY are the default origin;
// a-columns overrides y to sit the matrix under its own grid.
export const MX = 70;
export const MY = 90;
export const MCELL = 40;

// The A matrix, drawn with axes: a cell index above every column, "dark"/"lit"
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
  label(ctx, "cell (column) index, 0 to 24", x, y + 2 * MCELL + 22);
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

// --- row stack and operand grids ------------------------------------------
// Promoted out of lesson-2's conserves-vs-not, which is still the shape both
// use: a labelled row of numbered bars with its totals set apart on the right,
// and a numeric grid with a totals strip. Lesson 3 draws four of these rows
// per figure and two of these grids, so the geometry that was closure state
// there is explicit parameters here. Lesson 2's rendering is unchanged.

export const ROW = {
  leftX: 24,
  barX: 250,
  barW: 56,
  barH: 26,
  totalGap: 26,
};

// `values` are drawn as bars scaled to the largest in the row, so a row always
// fills its band whatever its peak. `totals` is a list of {label, value, note,
// emphasis} drawn to the right, one column each: lesson 2 passes one, lesson 3
// passes F, the divergence and their difference side by side.
//
// `signed` puts the zero line through the middle of the band and lets bars run
// both ways, for the log-ratio rows whose terms are negative wherever q is
// less confident than the posterior. Each bar then prints its value with an
// explicit sign, because the sign is what the reader is meant to read; the
// direction of the bar only carries the magnitude.
export function drawRow(ctx, y, opts) {
  const {
    title,
    subtitle = "",
    values,
    totals = [],
    note = "",
    signed = false,
    scale = null,
    n = values.length,
    x = ROW.barX,
    barW = ROW.barW,
    barH = ROW.barH,
    leftX = ROW.leftX,
    colour = COLOUR.neutral,
    dimmed = false,
    textSize = 13,      // titles, subtitles, notes: the written text
    valueSize = 13,     // the number printed under each bar
    totalSize = 20,
  } = opts;

  label(ctx, title, leftX, y - 4, dimmed ? COLOUR.dim : COLOUR.ink, Math.max(16, textSize));
  // A signed row prints its values where an ordinary row's subtitle goes, so
  // its subtitle moves above the title rather than colliding with the numbers.
  if (subtitle) {
    if (signed) label(ctx, subtitle, leftX, y - barH - 8, COLOUR.dim, textSize);
    else label(ctx, subtitle, leftX, y + textSize + 2, COLOUR.dim, textSize);
  }

  // A shared scale is passed in when two rows must be read against each other
  // (the log ratio and the same ratio reweighted by q): computing each row's
  // own peak would rescale the second one back up and hide the reweighting,
  // which is the whole claim of that figure.
  const peak = scale ?? Math.max(...values.map((v) => Math.abs(v)), 1e-9);
  const zero = signed ? y - barH / 2 : y;

  for (let i = 0; i < n; i++) {
    const bx = x + i * barW;
    const w = barW - 8;
    ctx.fillStyle = "#eef1f4";
    ctx.fillRect(bx, y - barH, w, barH);
    const h = (Math.abs(values[i]) / peak) * (signed ? barH / 2 : barH);
    ctx.fillStyle = signed && values[i] < 0 ? COLOUR.agent : colour;
    ctx.fillRect(bx, values[i] < 0 && signed ? zero : zero - h, w, h);
    if (signed) {
      ctx.strokeStyle = COLOUR.dim;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(bx, zero + 0.5);
      ctx.lineTo(bx + w, zero + 0.5);
      ctx.stroke();
    }
    ctx.font = `${valueSize}px system-ui, sans-serif`;
    ctx.fillStyle = COLOUR.dim;
    ctx.textAlign = "center";
    const text = signed
      ? (values[i] >= 0 ? "+" : "") + values[i].toFixed(2)
      : values[i].toFixed(2);
    ctx.fillText(text, bx + w / 2, y + 15);
    ctx.textAlign = "left";
  }

  let tx = x + n * barW + ROW.totalGap;
  for (const t of totals) {
    if (t.label) label(ctx, t.label, tx, y - barH + 2, COLOUR.dim, textSize);
    ctx.font = `bold ${totalSize}px system-ui, sans-serif`;
    ctx.fillStyle = t.emphasis ? COLOUR.agent : COLOUR.ink;
    ctx.fillText(t.value, tx, y + 2);
    if (t.note) label(ctx, t.note, tx, y + 20, COLOUR.dim, 13);
    tx += t.width ?? 90;
  }
  if (note) label(ctx, note, tx, y + 2, COLOUR.dim, 13);
  return tx;
}

export const OP_CELL = 42;

// A grid of numbers with a totals strip. `totalsAlong` is "cols" when each
// column is a distribution, "row" for a single row whose total is the point,
// and null for a grid whose totals say nothing worth printing.
export function drawOperand(ctx, x, y, values, opts = {}) {
  const {
    rows = values.length,
    cols = values[0].length,
    totalsAlong = null,
    title = "",
    colAxis = "",
    rowAxis = "",
    cell = OP_CELL,
    colour = COLOUR.neutral,
  } = opts;

  if (title) label(ctx, title, x, y - 44, COLOUR.ink);
  if (colAxis) label(ctx, colAxis, x, y - 26, COLOUR.dim, 13);
  if (rowAxis) {
    ctx.save();
    ctx.translate(x - 30, y + (rows * cell) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.fillStyle = COLOUR.dim;
    ctx.font = "13px system-ui, sans-serif";
    ctx.fillText(rowAxis, 0, 0);
    ctx.textAlign = "left";
    ctx.restore();
  }

  ctx.font = "12px system-ui, sans-serif";
  ctx.fillStyle = COLOUR.dim;
  ctx.textAlign = "center";
  for (let j = 0; j < cols; j++) {
    ctx.fillText(String(j), x + j * cell + (cell - 2) / 2, y - 5);
  }
  if (rows > 1) {
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let i = 0; i < rows; i++) {
      ctx.fillText(String(i), x - 6, y + i * cell + (cell - 2) / 2);
    }
    ctx.textBaseline = "alphabetic";
  }
  ctx.textAlign = "left";

  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      const v = values[i][j];
      const cx = x + j * cell;
      const cy = y + i * cell;
      ctx.fillStyle = "#fff";
      ctx.fillRect(cx, cy, cell - 2, cell - 2);
      if (v > 1e-9) {
        ctx.globalAlpha = 0.12 + 0.6 * Math.min(1, v);
        ctx.fillStyle = colour;
        ctx.fillRect(cx, cy, cell - 2, cell - 2);
        ctx.globalAlpha = 1;
      }
      ctx.strokeStyle = COLOUR.rule;
      ctx.lineWidth = 1;
      ctx.strokeRect(cx + 0.5, cy + 0.5, cell - 3, cell - 3);
      ctx.font = "12px system-ui, sans-serif";
      ctx.fillStyle = v > 1e-9 ? COLOUR.ink : COLOUR.rule;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(v > 1e-9 ? v.toFixed(2) : ".", cx + (cell - 2) / 2, cy + (cell - 2) / 2);
      ctx.textAlign = "left";
      ctx.textBaseline = "alphabetic";
    }
  }

  const ty = y + rows * cell + 16;
  if (totalsAlong === "cols") {
    for (let j = 0; j < cols; j++) {
      let t = 0;
      for (let i = 0; i < rows; i++) t += values[i][j];
      ctx.font = "bold 13px system-ui, sans-serif";
      ctx.fillStyle = COLOUR.ink;
      ctx.textAlign = "center";
      ctx.fillText(t.toFixed(2), x + j * cell + (cell - 2) / 2, ty);
      ctx.textAlign = "left";
    }
  } else if (totalsAlong === "row") {
    let t = 0;
    for (let j = 0; j < cols; j++) t += values[0][j];
    ctx.font = "bold 13px system-ui, sans-serif";
    ctx.fillStyle = COLOUR.agent;
    ctx.fillText(t.toFixed(2), x + cols * cell + 10, y + (cell - 2) / 2 + 4);
  }
}

// ===========================================================================
// State-vector annotation
// ===========================================================================

// Marks under a state vector: a small arrow at a column, its name beneath.
// Used wherever a lesson draws a distribution over states, so that the same
// cell always carries the same label in the same place and the reader can
// follow one state across rows and figures without reading any numbers.
//
// `columns` is the state index shown in each column, for a row that shows a
// subset; pass null when the row shows every state in index order, which is
// the usual case. Whatever the row was drawn with must be passed here too, or
// the marks drift away from the bars they annotate.
//
// Marks whose labels would overlap are staggered onto a second line rather
// than overprinted. A mark naming a state the row does not show is a caller
// error and throws: a figure with a mislabelled state should fail loudly
// rather than quietly point at the wrong cell.
export function drawStateMarks(ctx, x, y, cellW, marks, columns = null) {
  const placed = [];
  for (const mark of marks) {
    const column = columns === null ? mark.index : columns.indexOf(mark.index);
    if (column < 0) {
      throw new Error(`drawStateMarks: state ${mark.index} is not in this row`);
    }
    const centre = x + column * cellW + (cellW - 2) / 2;
    const colour = mark.colour ?? COLOUR.dim;

    ctx.font = "12px system-ui, sans-serif";
    const halfWidth = ctx.measureText(mark.label).width / 2;
    // Second line when this label would touch the previous one.
    const clash = placed.some(
      (p) => Math.abs(p.centre - centre) < p.halfWidth + halfWidth + 6,
    );
    const row = clash ? 1 : 0;
    placed.push({ centre, halfWidth });

    ctx.strokeStyle = colour;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(centre, y + 2);
    ctx.lineTo(centre, y + 9);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(centre - 3, y + 4);
    ctx.lineTo(centre, y + 1);
    ctx.lineTo(centre + 3, y + 4);
    ctx.stroke();

    ctx.fillStyle = colour;
    ctx.textAlign = "center";
    ctx.fillText(mark.label, centre, y + 22 + row * 14);
    ctx.textAlign = "left";
  }
}

// A distribution over states as a row of bars, one column per state, in index
// order. Unlike `drawRow` this prints no per-bar values: at twenty-five
// columns there is no room for them, and the numbers belong in a readout
// where a whole line is available. What the row carries is shape and
// position, which is what makes two beliefs comparable at a glance.
//
// `scale` fixes the bar height against a peak shared with another row, so two
// rows drawn with the same scale can be read against each other.
export function drawStateStrip(ctx, x, y, values, opts = {}) {
  const {
    cellW = 24,
    barH = 34,
    scale = null,
    colour = COLOUR.neutral,
    highlight = [],
  } = opts;
  const peak = scale ?? Math.max(...values, 1e-9);

  for (let s = 0; s < values.length; s++) {
    const bx = x + s * cellW;
    const w = cellW - 4;
    ctx.fillStyle = "#eef1f4";
    ctx.fillRect(bx, y, w, barH);
    const h = (values[s] / peak) * barH;
    ctx.fillStyle = highlight.includes(s) ? COLOUR.agent : colour;
    ctx.fillRect(bx, y + barH - h, w, h);
  }
  ctx.strokeStyle = COLOUR.rule;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x, y + barH + 0.5);
  ctx.lineTo(x + values.length * cellW - 4, y + barH + 0.5);
  ctx.stroke();
}

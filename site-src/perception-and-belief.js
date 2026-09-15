/**
 * Widgets specific to perception-and-belief.html (the 5x5 walking-route
 * gridworld). Shared infrastructure (mount, ctxOf, makeSlider, walk) lives
 * in widget.js.
 */

import { mount, mountAll, ctxOf, makeSlider } from "./widget.js";
import {
  uniformBelief,
  update,
  predict,
  observationModel,
  transitionModel,
  mulberry32,
} from "./aif.js";

const ROUTE = [3, 3, 1, 1, 3, 1]; // right, right, down, down, right, down
const SEED = 4; // matches the Python seed so "wrong on N readings" is stable
const ROWS = 5;
const COLS = 5;
const N = ROWS * COLS;
const GOAL = [4, 4];
const MOVES = [
  [-1, 0],
  [1, 0],
  [0, -1],
  [0, 1],
  [0, 0],
];

// One noise value drives both sensor-noise-walk and a-matrix.
const noiseChannel = new EventTarget();
let currentNoise = 0.3;

function stepsToGoal(tile) {
  return Math.abs(Math.floor(tile / COLS) - GOAL[0]) + Math.abs((tile % COLS) - GOAL[1]);
}

// The fixed walk: true tiles, and what a seeded noisy sensor reported.
function walk(noise) {
  const rand = mulberry32(SEED);
  const trueTiles = [0];
  let row = 0;
  let col = 0;
  for (const action of ROUTE) {
    const [dr, dc] = MOVES[action];
    const r = row + dr;
    const c = col + dc;
    if (r >= 0 && r < ROWS && c >= 0 && c < COLS) {
      row = r;
      col = c;
    }
    trueTiles.push(row * COLS + col);
  }
  const reported = trueTiles.map((tile) => {
    if (noise <= 0 || rand() >= noise) return tile;
    const tr = Math.floor(tile / COLS);
    const tc = tile % COLS;
    const nb = [];
    for (const [dr, dc] of MOVES) {
      if (dr === 0 && dc === 0) continue;
      const r = tr + dr;
      const c = tc + dc;
      if (r >= 0 && r < ROWS && c >= 0 && c < COLS) nb.push(r * COLS + c);
    }
    return nb.length ? nb[Math.floor(rand() * nb.length)] : tile;
  });
  return { trueTiles, reported };
}

// --- sensor-noise-walk -----------------------------------------------------
// Figure: agent walks a fixed route with adjustable sensor noise
// (content/perception-and-belief.md, "Set the sensor noise and watch...").
mount("sensor-noise-walk", (el) => {
  const CANVAS_W = 480;
  const CANVAS_H = 240;
  const PAD_L = 34;
  const PAD_B = 34;
  const PAD_TOP = 28;
  const PAD_RIGHT = 12;
  const MAX_STEPS_FROM_GOAL = 8; // longest possible steps-from-goal distance on this route

  const slider = makeSlider(el, "sensor noise", currentNoise);
  const ctx = ctxOf(el, CANVAS_W, CANVAS_H);
  if (!ctx) return;

  function draw(noise) {
    currentNoise = noise;
    const { trueTiles, reported } = walk(noise);
    const wrong = trueTiles.reduce((n, t, i) => n + (t !== reported[i] ? 1 : 0), 0);
    const truthDist = trueTiles.map(stepsToGoal);
    const sensorDist = reported.map(stepsToGoal);

    const W = ctx.canvas.width;
    const H = ctx.canvas.height;
    ctx.clearRect(0, 0, W, H);
    const x = (i) => PAD_L + (i * (W - PAD_L - PAD_RIGHT)) / (truthDist.length - 1);
    const y = (d) => PAD_TOP + ((MAX_STEPS_FROM_GOAL - d) * (H - PAD_B - PAD_TOP)) / MAX_STEPS_FROM_GOAL;

    ctx.strokeStyle = "#bbb";
    ctx.beginPath();
    ctx.moveTo(PAD_L, y(0));
    ctx.lineTo(W - PAD_RIGHT, y(0));
    ctx.stroke();

    const line = (data, color, dash) => {
      ctx.strokeStyle = color;
      ctx.setLineDash(dash);
      ctx.lineWidth = 2;
      ctx.beginPath();
      data.forEach((d, i) => (i ? ctx.lineTo(x(i), y(d)) : ctx.moveTo(x(i), y(d))));
      ctx.stroke();
      ctx.setLineDash([]);
      data.forEach((d, i) => {
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(x(i), y(d), 3, 0, 2 * Math.PI);
        ctx.fill();
      });
    };
    line(truthDist, "#1f6f5c", []);
    line(sensorDist, "#c25a3f", [4, 3]);

    ctx.fillStyle = "#666";
    ctx.font = "16px system-ui, sans-serif";
    ctx.fillText("steps from the goal", PAD_L, 14);
    ctx.fillText(
      "solid: where it actually was     dashed: where the sensor implied",
      PAD_L,
      H - 6,
    );
    ctx.fillStyle = "#333";
    ctx.fillText(`the two part company on ${wrong} of ${truthDist.length} readings`, PAD_L, H - 20);

    noiseChannel.dispatchEvent(new CustomEvent("change", { detail: { noise } }));
  }

  slider.addEventListener("input", () => draw(Number(slider.value)));
  draw(currentNoise);
});

// --- a-matrix ------------------------------------------------------------
// Figure: A-matrix column view, "standing here, what might I see?"
// (content/perception-and-belief.md, before <widget a-matrix>).
mount("a-matrix", (el) => {
  const CANVAS_W = 360;
  const CANVAS_H = 380;
  const MARGIN_X = 24;
  const MARGIN_Y = 44;
  const GRID_ORIGIN_X = 16;
  const GRID_ORIGIN_Y = 8;

  const ctx = ctxOf(el, CANVAS_W, CANVAS_H);
  if (!ctx) return;
  const readout = document.createElement("p");
  readout.style.font = "13px system-ui, sans-serif";
  readout.style.margin = "0.5rem 0 0";
  el.appendChild(readout);

  const hasSharedSlider = document.querySelector('[data-widget="sensor-noise-walk"]') !== null;
  if (!hasSharedSlider) {
    const own = makeSlider(el, "sensor noise", currentNoise);
    own.addEventListener("input", () => draw(Number(own.value)));
  }

  function draw(noise) {
    const A = observationModel({ rows: ROWS, cols: COLS, sensorNoise: noise });
    const W = ctx.canvas.width;
    const H = ctx.canvas.height;
    ctx.clearRect(0, 0, W, H);
    const cell = Math.min(W - MARGIN_X, H - MARGIN_Y) / (N + 1);
    for (let o = 0; o <= N; o++) {
      for (let s = 0; s < N; s++) {
        const v = A[o][s];
        const g = Math.round(255 * (1 - v));
        ctx.fillStyle = `rgb(${g}, ${g}, ${Math.round(120 + 135 * v)})`;
        ctx.fillRect(s * cell + GRID_ORIGIN_X, o * cell + GRID_ORIGIN_Y, cell, cell);
      }
    }
    ctx.fillStyle = "#666";
    ctx.font = "11px system-ui, sans-serif";
    ctx.fillText("observation (row) against tile the agent is on (column)", 8, H - 26);

    const col7 = A.map((row, o) => [o, row[7]]).filter(([, p]) => p > 0);
    readout.textContent =
      "Standing on tile 7, the agent might see: " +
      col7.map(([o, p]) => `tile ${o} (${p.toFixed(2)})`).join(", ");
  }

  noiseChannel.addEventListener("change", (e) => draw(e.detail.noise));
  draw(currentNoise);
});

// --- watch-it-move -----------------------------------------------------
// Figure: full step-through combining movement + observation updates
// (content/perception-and-belief.md, "Now put the two halves together...").
mount("watch-it-move", (el) => {
  const controls = document.createElement("div");
  controls.style.margin = "0 0 0.75rem";
  const prev = document.createElement("button");
  prev.textContent = "previous step";
  const next = document.createElement("button");
  next.textContent = "next step";
  const restart = document.createElement("button");
  restart.textContent = "back to the start";
  controls.append(prev, next, restart);
  el.appendChild(controls);

  const CANVAS_W = 560;
  const CANVAS_H = 280;
  const GRID_CELL = 44;
  const GRID_ORIGIN_X = 8;
  const GRID_ORIGIN_Y = 20;
  const BAR_GAP = 24; // gap between the grid and the bar chart
  const BOTTOM_MARGIN = 44;

  const ctx = ctxOf(el, CANVAS_W, CANVAS_H);
  if (!ctx) return;

  const noise = currentNoise;
  const A = observationModel({ rows: ROWS, cols: COLS, sensorNoise: noise });
  const B = transitionModel({ rows: ROWS, cols: COLS });
  const { trueTiles, reported } = walk(noise);

  const beliefs = [];
  let belief = uniformBelief(N);
  belief = update(belief, A, reported[0]).posterior;
  beliefs.push(belief);
  for (let i = 0; i < ROUTE.length; i++) {
    belief = predict(belief, B, ROUTE[i]);
    belief = update(belief, A, reported[i + 1]).posterior;
    beliefs.push(belief);
  }

  let step = 0;
  function draw() {
    const b = beliefs[step];
    const W = ctx.canvas.width;
    const H = ctx.canvas.height;
    ctx.clearRect(0, 0, W, H);

    const maxB = Math.max(...b, 1e-9);
    for (let s = 0; s < N; s++) {
      const r = Math.floor(s / COLS);
      const c = s % COLS;
      const v = b[s] / maxB;
      const g = Math.round(255 * (1 - v));
      ctx.fillStyle = `rgb(${g},${g},${Math.round(120 + 135 * v)})`;
      ctx.fillRect(GRID_ORIGIN_X + c * GRID_CELL, GRID_ORIGIN_Y + r * GRID_CELL, GRID_CELL - 1, GRID_CELL - 1);
    }
    const tr = Math.floor(trueTiles[step] / COLS);
    const tc = trueTiles[step] % COLS;
    ctx.strokeStyle = "#e8563f";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(GRID_ORIGIN_X + tc * GRID_CELL + GRID_CELL / 2, GRID_ORIGIN_Y + tr * GRID_CELL + GRID_CELL / 2, GRID_CELL / 3, 0, 2 * Math.PI);
    ctx.stroke();

    const bx = GRID_ORIGIN_X + COLS * GRID_CELL + BAR_GAP;
    const bw = (W - bx - GRID_ORIGIN_X) / N;
    for (let s = 0; s < N; s++) {
      ctx.fillStyle = s === reported[step] ? "#e8563f" : "#4c72b0";
      const h = b[s] * (H - BOTTOM_MARGIN);
      ctx.fillRect(bx + s * bw, H - 24 - h, Math.max(1, bw - 1), h);
    }
    ctx.fillStyle = "#333";
    ctx.font = "12px system-ui, sans-serif";
    ctx.fillText("belief on the grid", 8, 14);
    ctx.fillText("one bar per tile", bx, 14);
    ctx.fillText(
      `step ${step}: the sensor said tile ${reported[step]}, the ring is the true position`,
      8,
      H - 6,
    );
  }

  prev.addEventListener("click", () => {
    step = Math.max(0, step - 1);
    draw();
  });
  next.addEventListener("click", () => {
    step = Math.min(beliefs.length - 1, step + 1);
    draw();
  });
  restart.addEventListener("click", () => {
    step = 0;
    draw();
  });
  draw();
});

mountAll();

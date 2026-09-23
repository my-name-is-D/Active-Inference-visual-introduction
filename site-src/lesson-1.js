/**
 * Widgets specific to lesson-1.html (the lamp gridworld walkthrough).
 * Shared infrastructure (mount, drawing helpers, the lamp world constants)
 * lives in widget.js.
 */

import {
  mount,
  mountAll,
  ctxOf,
  label,
  drawGrid,
  drawBeliefBar,
  drawIsoPlane,
  drawMatrix,
  oneHot,
  lampBeliefs,
  obsColour,
  obsName,
  COLOUR,
  N,
  DARK,
  LIT,
  LIT_TILES,
  LIT_MASK,
  AGENT_TILE,
  SEQUENCE,
  A_LAMP,
} from "./widget.js";
import { uniformBelief, update } from "./aif.js";

// --- legend ----------------------------------------------------------------
// Figure: the lamp/lit-vs-dark cell legend (content/lesson-1.md, before "the
// map is what makes the readings interpretable at all").
mount("legend", (el) => {
  const CANVAS_W = 480;
  const CANVAS_H = 96;
  const CELL = 44;
  const ITEM_GAP = 150; // horizontal spacing between legend entries
  const ORIGIN_X = 18;
  const ORIGIN_Y = 14;
  const LABEL_Y_OFFSET = 18; // gap between the swatch and its text label

  const ctx = ctxOf(el, CANVAS_W, CANVAS_H);
  if (!ctx) return;
  const items = [
    ["a lit cell", () => {
      ctx.fillStyle = COLOUR.lit;
      ctx.fillRect(0, 0, CELL, CELL);
      ctx.strokeStyle = COLOUR.litEdge;
      ctx.lineWidth = 1;
      ctx.strokeRect(0.5, 0.5, CELL - 1, CELL - 1);
    }],
    ["a dark cell", () => {
      ctx.fillStyle = COLOUR.dark;
      ctx.fillRect(0, 0, CELL, CELL);
    }],
    ["the agent", () => {
      ctx.fillStyle = "#efefec";
      ctx.fillRect(0, 0, CELL, CELL);
      ctx.fillStyle = COLOUR.agent;
      ctx.beginPath();
      ctx.arc(CELL / 2, CELL / 2, CELL / 3, 0, 2 * Math.PI);
      ctx.fill();
    }],
  ];
  items.forEach(([text, paint], i) => {
    ctx.save();
    ctx.translate(ORIGIN_X + i * ITEM_GAP, ORIGIN_Y);
    paint();
    ctx.restore();
    label(ctx, text, ORIGIN_X + i * ITEM_GAP, ORIGIN_Y + CELL + LABEL_Y_OFFSET, COLOUR.ink);
  });
});

// --- state-vs-belief -------------------------------------------------------
// Figure: hidden state vs the agent's belief, side by side (content/lesson-1.md,
// the "gap between them" section).
// Left: the world, one hidden state, the agent on it. Right: what the world
// would show, a bump per cell at p(lit | s). Several cells carry the same
// bump, which is exactly why a lit reading cannot name one cell.
mount("state-vs-belief", (el) => {
  const CANVAS_W = 700;
  const CANVAS_H = 320;
  const LEFT_X = 170;
  const LEFT_Y = 120;
  const RIGHT_X = 510;
  const RIGHT_Y = 150;
  const PLANE_CELL = 24;
  const LEFT_LABEL_X = 60;
  const RIGHT_LABEL_X = 400;
  const TITLE_Y = 20;
  const SUBTITLE_Y = 38;
  const CAPTION_Y = 306;

  const ctx = ctxOf(el, CANVAS_W, CANVAS_H);
  if (!ctx) return;
  ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);
  drawIsoPlane(ctx, LEFT_X, LEFT_Y, PLANE_CELL, { agent: AGENT_TILE });
  drawIsoPlane(ctx, RIGHT_X, RIGHT_Y, PLANE_CELL, { heights: A_LAMP[LIT] });
  label(ctx, "the world: one hidden state", LEFT_LABEL_X, TITLE_Y, COLOUR.ink);
  label(ctx, "the agent is on one of these cells", LEFT_LABEL_X, SUBTITLE_Y);
  label(ctx, "what it would show: p(lit | cell)", RIGHT_LABEL_X, TITLE_Y, COLOUR.ink);
  label(ctx, "tall on every lit cell, low on every dark one", RIGHT_LABEL_X, SUBTITLE_Y);
  label(ctx, "three cells share the tall bump: a lit reading fits all three", LEFT_LABEL_X, CAPTION_Y, COLOUR.ink);
});

// --- commit-vs-distribute --------------------------------------------------
// Figure: cost of committing to one cell vs keeping a distribution
// (content/lesson-1.md, "This is what committing fully to one answer costs.").
// The lesson's argument, stacked on one stepper. The agent never moves: it
// sits on cell 10 for the whole demo, exactly like every other widget on this
// page. The two panels differ only in what they DO with that fixed position.
// The top one commits to "I am on cell 10" the moment the first lit reading
// arrives, and holds that as a certainty (probability 1 on cell 10, 0
// everywhere else) until a reading it cannot explain forces it to give the
// whole thing up and start from nothing. The bottom one keeps the ordinary
// posterior. Same readings, same true cell, two different objects held.
mount("commit-vs-distribute", (el) => {
  const controls = document.createElement("div");
  controls.style.margin = "0 0 0.75rem";
  const next = document.createElement("button");
  next.textContent = "next observation";
  const restart = document.createElement("button");
  restart.textContent = "restart";
  controls.append(next, restart);
  el.appendChild(controls);

  const W = 720;
  const H = 640;
  const GRID_CELL = 30;
  const BAR_H = 64;

  const ctx = ctxOf(el, W, H);
  if (!ctx) return;
  const beliefs = lampBeliefs();

  // The committing agent, replayed from the start. It commits to its own
  // cell (10) on the first lit reading and holds that as certain; a dark
  // reading contradicts it, so it discards everything and starts over, with
  // no memory of having been right before.
  function committed(step) {
    let held = false; // true once committed to cell 10
    let restarts = 0;
    let justCommitted = false;
    for (let i = 0; i < step; i++) {
      const o = SEQUENCE[i];
      justCommitted = false;
      if (!held) {
        if (o === LIT) {
          held = true;
          justCommitted = true;
        }
      } else if (o === DARK) {
        held = false; // cell 10 is lit, so a dark reading cannot be it
        restarts += 1;
      }
    }
    return { held, restarts, justCommitted };
  }

  let step = 0;
  function draw() {
    ctx.clearRect(0, 0, W, H);
    const o = step === 0 ? null : SEQUENCE[step - 1];
    const b = beliefs[step];

    label(ctx, step === 0
      ? "no readings yet"
      : `reading ${step} of ${SEQUENCE.length}:  ${obsName(o)}`, 16, 16, COLOUR.ink);
    if (o !== null) {
      ctx.fillStyle = obsColour(o);
      ctx.fillRect(196, 6, 14, 12);
      ctx.strokeStyle = o === LIT ? COLOUR.litEdge : COLOUR.dark;
      ctx.strokeRect(196.5, 6.5, 13, 11);
    }

    // --- The cell itself: real position, never moves ------------------
    const { held, restarts, justCommitted } = committed(step);
    const gridTitleY = 46;
    label(ctx, "the cell the agent is on", 16, gridTitleY, COLOUR.ink);
    const gridY = gridTitleY + 14;
    // The agent is always on cell 10, never anywhere else: the red circle is
    // fixed there for the whole demo, on from the very first frame. What the
    // cell SHOWS (lit or dark) follows the reading, because that is what a
    // false reading is: the true cell does not change, only what it reports.
    drawGrid(ctx, 16, gridY, GRID_CELL, {
      agent: AGENT_TILE,
      observed: o,
      showIds: true,
    });

    // --- A centred section title over both distributions ---------------
    const sectionTitleY = gridY + 5 * GRID_CELL + 38;
    ctx.font = "bold 20px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillStyle = COLOUR.ink;
    ctx.fillText("Probability distributions", W / 2, sectionTitleY);
    ctx.textAlign = "left";

    // --- Top distribution: commit to one answer ------------------------
    // The committed belief AS a distribution: probability 1 on cell 10 once
    // held, uniform before the first commitment and again after every
    // contradiction (an agent starting over truly knows nothing, same as at
    // the very start), never a hybrid of the two.
    const barTitleY = sectionTitleY + 30;
    label(ctx, "When we commit to a belief:", 16, barTitleY, COLOUR.ink);
    const committedDist = held ? oneHot(AGENT_TILE) : uniformBelief(N);
    drawBeliefBar(ctx, 16, barTitleY + 14, W - 32, committedDist);

    // The explanation of what happened this step sits under this
    // distribution, because it is this distribution's story: what the
    // committed belief is doing, not a property of the grid above.
    const expY = barTitleY + 14 + BAR_H + 24;
    if (held) {
      label(ctx, `"I am on cell ${AGENT_TILE}."`, 16, expY, COLOUR.ink);
      if (justCommitted) {
        label(ctx, "The reading says lit, so it commits fully:", 16, expY + 20);
        label(ctx, `certain it is on cell ${AGENT_TILE}, nothing else possible.`, 16, expY + 36);
      } else {
        label(ctx, "Still held from before: no new lit reading has", 16, expY + 20);
        label(ctx, "arrived to re-confirm it, but nothing challenged it either.", 16, expY + 36);
      }
    } else if (step === 0) {
      label(ctx, "nothing committed yet", 16, expY);
    } else {
      label(ctx, "contradicted: start again", 16, expY, COLOUR.agent);
      label(ctx, `Cell ${AGENT_TILE} is lit, so a dark reading cannot`, 16, expY + 20);
      label(ctx, "come from it. The commitment is thrown away.", 16, expY + 36);
    }
    if (restarts > 0) {
      label(ctx, `thrown away and restarted ${restarts}x`, 16, expY + 60, COLOUR.agent);
    }

    // --- Bottom distribution: keep uncertainty --------------------------
    // expY's text block is at most 3 lines plus an optional restart-count
    // line; always leave room for all of it before the next heading.
    const botY = expY + (restarts > 0 ? 100 : 76);
    label(ctx, "When we keep uncertainty over our position:", 16, botY, COLOUR.ink);
    drawBeliefBar(ctx, 16, botY + 14, W - 32, b);
    const noteY = botY + 14 + BAR_H + 24;
    label(ctx, `p(each lit cell) = ${b[LIT_TILES[0]].toFixed(3)}   (all 25 cells sum to ${b.reduce((x, y) => x + y, 0).toFixed(2)})`, 16, noteY, COLOUR.ink);
    if (o === DARK) {
      label(ctx, "Dipped by the dark reading, not destroyed: the ranking survives.", 16, noteY + 20, COLOUR.agent);
    } else if (step > 0) {
      label(ctx, "The lit cells are still ranked against each other.", 16, noteY + 20);
    } else {
      label(ctx, "Every cell is equally likely.", 16, noteY + 20);
    }
  }

  next.addEventListener("click", () => {
    step = Math.min(SEQUENCE.length, step + 1);
    draw();
  });
  restart.addEventListener("click", () => {
    step = 0;
    draw();
  });
  draw();
});

// --- belief-bars -----------------------------------------------------------
// Figure: belief bar chart update after one observation (content/lesson-1.md,
// the "worked example" showing prior x likelihood = unnormalised / evidence).
// The update with its working shown. update() already hands back the
// likelihood and the unnormalised product, so all four rows are the same
// arithmetic the prose does by hand: multiply, then divide by the total.
mount("belief-bars", (el) => {
  const controls = document.createElement("div");
  controls.style.margin = "0 0 0.75rem";
  const seeLit = document.createElement("button");
  seeLit.textContent = "observe lit";
  seeLit.style.background = COLOUR.lit;
  seeLit.style.border = `1px solid ${COLOUR.litEdge}`;
  const seeDark = document.createElement("button");
  seeDark.textContent = "observe dark";
  seeDark.style.background = COLOUR.dark;
  seeDark.style.color = "#fff";
  seeDark.style.border = `1px solid ${COLOUR.dark}`;
  const reset = document.createElement("button");
  reset.textContent = "back to a flat belief";
  controls.append(seeLit, seeDark, reset);
  el.appendChild(controls);

  const CANVAS_W = 700;
  const CANVAS_H = 290;
  const ROW_H = 64;
  const BX = 210;   // wide enough for the two-line row labels
  const RIGHT_MARGIN = 86;
  const BW = (CANVAS_W - BX - RIGHT_MARGIN) / N;

  const ctx = ctxOf(el, CANVAS_W, CANVAS_H);
  if (!ctx) return;
  const readout = document.createElement("p");
  readout.style.font = "13px system-ui, sans-serif";
  readout.style.margin = "0.5rem 0 0";
  readout.style.minHeight = "2.4em";
  el.appendChild(readout);

  let prior = uniformBelief(N);
  let last = null;
  let hover = null;

  function bars(values, y, scaleMax, colourFn) {
    const h = ROW_H - 22;
    for (let s = 0; s < N; s++) {
      const v = scaleMax > 0 ? (values[s] / scaleMax) * h : 0;
      ctx.fillStyle = colourFn(s);
      ctx.fillRect(BX + s * BW, y + h - v, Math.max(1, BW - 2), v);
      if (colourFn(s) === COLOUR.lit) {
        ctx.strokeStyle = COLOUR.litEdge;
        ctx.lineWidth = 1;
        ctx.strokeRect(BX + s * BW + 0.5, y + h - v + 0.5, Math.max(1, BW - 2) - 1, Math.max(0, v - 1));
      }
      if (hover === s) {
        ctx.strokeStyle = COLOUR.agent;
        ctx.lineWidth = 1;
        ctx.strokeRect(BX + s * BW - 1, y - 1, BW, h + 2);
      }
    }
    ctx.strokeStyle = COLOUR.rule;
    ctx.beginPath();
    ctx.moveTo(BX, y + h + 0.5);
    ctx.lineTo(CANVAS_W - RIGHT_MARGIN, y + h + 0.5);
    ctx.stroke();
  }

  function draw() {
    ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);
    const neutral = () => COLOUR.neutral;

    // The belief this frame is about, and the three rows that only exist once
    // an observation has arrived. Before the first click the prior row is
    // drawn by the same code, in the same slot, on the same ruler: pressing a
    // button adds rows below it and never moves the row already on screen.
    const shown = last ? last.prior : prior;
    const darkCell = LIT_MASK.findIndex((l) => !l);
    // prior, unnormalised, and posterior are all real distributions over the
    // same cells, so they share one scale: that is what makes the dip below
    // one and the snap back to one visible as an actual height change,
    // instead of needing a caption to say so. With no observation yet, the
    // ruler is what one lit reading would stretch the peak to, so the flat
    // belief starts short with visible room above it.
    const distScale = last
      ? Math.max(...last.posterior)
      : Math.max(...update(shown, A_LAMP, LIT).posterior);

    const rows = [
      ["prior", shown, distScale, neutral, ["what it held before", "total 1.000"]],
    ];
    if (last) {
      const { obs, likelihood, unnormalised, posterior, evidence } = last;
      const oc = () => obsColour(obs);
      rows.push(
        // Read the two rates off A itself rather than writing them in: the lit
        // row is [0.8 lit, 0.1 dark] and the dark row [0.2 lit, 0.9 dark], and
        // a hardcoded pair silently mislabels one of the two observations.
        // This row alone is drawn against 0..1: it is a likelihood, not a
        // distribution over cells, so it shares no mass with the other three.
        // Saying so stops the reader reading equal bar heights as equal values.
        ["o  likelihood", likelihood, 1, oc, [
          `row A[${obsName(obs)}]:`,
          `${A_LAMP[obs][LIT_TILES[0]].toFixed(2)} on lit, ${A_LAMP[obs][darkCell].toFixed(2)} on dark`,
        ]],
        ["=  unnormalised", unnormalised, distScale, neutral, [`total ${evidence.toFixed(3)}`]],
        ["posterior", posterior, distScale, neutral, ["total 1.000"]],
      );
    }
    rows.forEach(([name, values, scale, colourFn, notes], i) => {
      const y = 20 + i * ROW_H;
      label(ctx, name, 8, y + 12, COLOUR.ink);
      notes.forEach((note, j) => label(ctx, note, 8, y + 28 + j * 15, COLOUR.dim, 14));
      bars(values, y, scale, colourFn);
    });

    if (!last) {
      label(ctx, "press a button to observe. The agent does not move,", 8, 130, COLOUR.ink);
      label(ctx, "every reading comes from the same cell.", 8, 148, COLOUR.ink);
      readout.textContent = "";
      return;
    }

    const { likelihood, unnormalised, posterior, evidence } = last;
    const s = hover === null ? LIT_TILES[0] : hover;
    readout.textContent =
      `cell ${s} (${LIT_MASK[s] ? "lit" : "dark"}): ` +
      `${shown[s].toFixed(4)} x ${likelihood[s].toFixed(2)} = ${unnormalised[s].toFixed(4)}, ` +
      `then / ${evidence.toFixed(4)} = ${posterior[s].toFixed(4)}. ` +
      (hover === null ? "Hover a bar to follow another cell." : "");
  }

  function observe(o) {
    const r = update(prior, A_LAMP, o);
    // Snapshot the belief this step started from. `prior` advances below to
    // feed the next update, so a redraw (hover, mouseleave) would otherwise
    // show this step's posterior in the prior row and the two would match.
    last = {
      obs: o,
      prior,
      likelihood: r.likelihood,
      unnormalised: r.unnormalised,
      posterior: r.posterior,
      evidence: r.unnormalised.reduce((a, b) => a + b, 0),
    };
    prior = r.posterior; // the posterior of one step is the prior of the next
    draw();
  }

  ctx.canvas.addEventListener("mousemove", (e) => {
    const rect = ctx.canvas.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * CANVAS_W;
    const s = Math.floor((x - BX) / BW);
    const next = s >= 0 && s < N ? s : null;
    if (next !== hover) {
      hover = next;
      draw();
    }
  });
  ctx.canvas.addEventListener("mouseleave", () => {
    hover = null;
    draw();
  });
  seeLit.addEventListener("click", () => observe(LIT));
  seeDark.addEventListener("click", () => observe(DARK));
  reset.addEventListener("click", () => {
    prior = uniformBelief(N);
    last = null;
    draw();
  });
  draw();
});

// --- a-columns -------------------------------------------------------------
// Figure: A-matrix columns, p(o | s) for a fixed state (content/lesson-1.md,
// "Read a column..." / the false-reading-rate paragraph before <widget a-columns>).
// Hovering a column connects it to the same cell in the world. Each column is
// a distribution over the two observations and fills its bar to exactly one.
mount("a-columns", (el) => {
  const W = 700;
  const H = 550;
  const ctx = ctxOf(el, W, H);
  if (!ctx) return;
  const GRID_CELL = 40;
  const GRID_Y = 40;
  const GRID_X = 16;
  const MATRIX_CELL = 25;
  const MATRIX_X = 60;
  const matrixTop = GRID_Y + 5 * GRID_CELL + 66;
  let hover = null;

  function draw() {
    ctx.clearRect(0, 0, W, H);
    label(ctx, "the grid, with every cell's index", GRID_X, GRID_Y - 12, COLOUR.ink, 20);
    drawGrid(ctx, GRID_X, GRID_Y, GRID_CELL, { mark: hover, showIds: true });

    label(ctx, "A: what the world would show, one column per cell", 16, matrixTop - 40, COLOUR.ink, 20);
    drawMatrix(ctx, { markCol: hover, x: MATRIX_X, y: matrixTop, cell: MATRIX_CELL });

    const ty = matrixTop + 2 * MATRIX_CELL + 50;
    const RSIZE = 18;
    if (hover === null) {
      label(ctx, "Hover a matrix column or a numbered grid cell to connect the two.", 16, ty, COLOUR.dim, RSIZE);
      label(ctx, "Every column contains the two readings possible at that cell.", 16, ty + 34, COLOUR.dim, RSIZE);
      return;
    }
    const col = hover;
    const isLit = LIT_MASK[col];
    label(ctx, `column ${col}: a ${isLit ? "lit" : "dark"} cell`, 16, ty, COLOUR.ink, RSIZE);
    label(ctx, `p(dark | cell ${col}) = ${A_LAMP[DARK][col].toFixed(2)}`, 16, ty + 34, COLOUR.ink, RSIZE);
    label(ctx, `p(lit  | cell ${col}) = ${A_LAMP[LIT][col].toFixed(2)}`, 16, ty + 64, COLOUR.ink, RSIZE);
    label(ctx, "dark + lit, every column, always one", 16, ty + 100, COLOUR.ink, RSIZE);
  }

  ctx.canvas.addEventListener("mousemove", (e) => {
    const rect = ctx.canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * ctx.canvas.width / rect.width;
    const y = (e.clientY - rect.top) * ctx.canvas.height / rect.height;
    let next = null;
    if (x >= MATRIX_X && x < MATRIX_X + N * MATRIX_CELL && y >= matrixTop && y < matrixTop + 2 * MATRIX_CELL) {
      next = Math.floor((x - MATRIX_X) / MATRIX_CELL);
    } else if (x >= GRID_X && x < GRID_X + 5 * GRID_CELL && y >= GRID_Y && y < GRID_Y + 5 * GRID_CELL) {
      next = Math.floor((y - GRID_Y) / GRID_CELL) * 5 + Math.floor((x - GRID_X) / GRID_CELL);
    }
    if (next !== hover) { hover = next; draw(); }
  });
  ctx.canvas.addEventListener("mouseleave", () => { hover = null; draw(); });
  draw();
});

// --- a-rows ----------------------------------------------------------------
// Figure: A-matrix rows, p(o | s) for a fixed observation (content/lesson-1.md,
// "Now it reads A in the other direction..." before <widget a-rows>).
// Hovering across the lit row exposes its running total. It passes one and
// keeps going, which is the whole point: rows are scores, not distributions.
mount("a-rows", (el) => {
  const W = 700;
  const H = 350;
  const ctx = ctxOf(el, W, H);
  if (!ctx) return;
  const MATRIX_CELL = 25;
  const MATRIX_X = 60;
  const matrixTop = 90;
  let hover = null;

  function draw() {
    ctx.clearRect(0, 0, W, H);
    label(ctx, "the same A, read across the lit row", 16, matrixTop - 42, COLOUR.ink, 20);
    drawMatrix(ctx, { markRow: LIT, markCol: hover, x: MATRIX_X, y: matrixTop, cell: MATRIX_CELL });

    // Same readout size as a-columns (RSIZE = 22), so the two images read as
    // one matched pair rather than one full-size and one shrunk.
    const RSIZE = 18;
    const ty = matrixTop + 2 * MATRIX_CELL + 50;
    if (hover !== null) {
      const running = A_LAMP[LIT].slice(0, hover + 1).reduce((a, b) => a + b, 0);
      label(ctx, `through cell ${hover}: add ${A_LAMP[LIT][hover].toFixed(2)}`, 16, ty, COLOUR.ink, RSIZE);
      label(ctx, `running total ${running.toFixed(1)}`, 16, ty + 32, running > 1 ? COLOUR.agent : COLOUR.ink, RSIZE);
      // The total as a bar, with the one mark it sails past.
      const sx = 16;
      const sw = W - 32;
      const full = sw / 5; // one unit of probability is a fifth of the bar
      ctx.fillStyle = COLOUR.lit;
      ctx.fillRect(sx, ty + 50, Math.min(running, 5) * full, 26);
      ctx.strokeStyle = COLOUR.litEdge;
      ctx.strokeRect(sx + 0.5, ty + 50.5, Math.min(running, 5) * full, 25);
      ctx.strokeStyle = COLOUR.agent;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(sx + full, ty + 44);
      ctx.lineTo(sx + full, ty + 82);
      ctx.stroke();
      label(ctx, "one", sx + full - 14, ty + 100, COLOUR.agent, RSIZE);
      if (running > 1) label(ctx, "already past one, and still adding", sx + full + 40, ty + 100, COLOUR.agent, RSIZE);
    } else {
      label(ctx, `the lit row adds to ${A_LAMP[LIT].reduce((a, b) => a + b, 0).toFixed(1)}`, 16, ty, COLOUR.ink, RSIZE);
      label(ctx, `the dark row adds to ${A_LAMP[DARK].reduce((a, b) => a + b, 0).toFixed(1)}`, 16, ty + 32, COLOUR.ink, RSIZE);
      label(ctx, "Hover a cell in the lit row to total the row up to that point.", 16, ty + 68, COLOUR.dim, RSIZE);
    }
  }

  ctx.canvas.addEventListener("mousemove", (e) => {
    const rect = ctx.canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * ctx.canvas.width / rect.width;
    const y = (e.clientY - rect.top) * ctx.canvas.height / rect.height;
    const next = x >= MATRIX_X && x < MATRIX_X + N * MATRIX_CELL &&
      y >= matrixTop + LIT * MATRIX_CELL && y < matrixTop + (LIT + 1) * MATRIX_CELL
      ? Math.floor((x - MATRIX_X) / MATRIX_CELL) : null;
    if (next !== hover) { hover = next; draw(); }
  });
  ctx.canvas.addEventListener("mouseleave", () => { hover = null; draw(); });
  draw();
});

// --- sequence-stepper ------------------------------------------------------
// Figure: belief converging over a sequence of observations (content/lesson-1.md,
// the payoff section ending "the agent is most of the way to it.").
// The payoff. The agent stands still on a lit cell for seven readings, two of
// which are false, and the belief is folded over the fixed sequence so that
// stepping backwards shows exactly the same numbers.
mount("sequence-stepper", (el) => {
  const controls = document.createElement("div");
  controls.style.margin = "0 0 0.75rem";
  const prev = document.createElement("button");
  prev.textContent = "previous observation";
  const next = document.createElement("button");
  next.textContent = "next observation";
  const restart = document.createElement("button");
  restart.textContent = "back to the start";
  controls.append(prev, next, restart);
  el.appendChild(controls);

  const CANVAS_W = 700;
  const CANVAS_H = 360;
  const PLANE_CELL = 23;
  const LEFT_X = 170;
  const LEFT_Y = 140;
  const RIGHT_X = 510;
  const RIGHT_Y = 165;

  const ctx = ctxOf(el, CANVAS_W, CANVAS_H);
  if (!ctx) return;
  const beliefs = lampBeliefs();

  let step = 0;
  function draw() {
    ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);
    const o = step === 0 ? null : SEQUENCE[step - 1];
    const b = beliefs[step];

    // The world's layout is fixed: which cells are lit never changes, so only
    // the agent's own cell recolours with the reading (via `observed`). Every
    // other cell holding still is what makes the false readings legible as
    // the sensor being wrong rather than the world changing.
    drawIsoPlane(ctx, LEFT_X, LEFT_Y, PLANE_CELL, {
      agent: AGENT_TILE,
      observed: o,
      lit: LIT_MASK,
    });
    drawIsoPlane(ctx, RIGHT_X, RIGHT_Y, PLANE_CELL, { heights: b.map((x) => x / Math.max(...b)) });

    label(ctx, "the world", 60, 22, COLOUR.ink);
    label(ctx, "the agent has not moved", 60, 40);
    label(ctx, "the belief", 400, 22, COLOUR.ink);
    label(ctx, "bump height is the probability of that cell", 400, 40);

    if (o === null) {
      label(ctx, "no readings yet: every cell equally likely", 16, 330, COLOUR.ink);
    } else {
      ctx.fillStyle = obsColour(o);
      ctx.fillRect(16, 320, 14, 12);
      ctx.strokeStyle = o === LIT ? COLOUR.litEdge : COLOUR.dark;
      ctx.strokeRect(16.5, 320.5, 13, 11);
      const truth = LIT_MASK[AGENT_TILE] ? LIT : DARK;
      label(
        ctx,
        `reading ${step} of ${SEQUENCE.length}: ${obsName(o)}` +
          (o === truth ? "" : "   (a false reading: the cell is lit)"),
        38,
        330,
        o === truth ? COLOUR.ink : COLOUR.agent,
      );
    }
    const darkCell = LIT_MASK.findIndex((l) => !l);
    label(ctx, `p(each lit cell) = ${b[LIT_TILES[0]].toFixed(3)}`, 400, 330, COLOUR.ink);
    label(ctx, `p(each dark cell) = ${b[darkCell].toFixed(4)}`, 400, 348);
    label(ctx, `total ${b.reduce((x, y) => x + y, 0).toFixed(2)}`, 16, 348);
  }

  prev.addEventListener("click", () => {
    step = Math.max(0, step - 1);
    draw();
  });
  next.addEventListener("click", () => {
    step = Math.min(SEQUENCE.length, step + 1);
    draw();
  });
  restart.addEventListener("click", () => {
    step = 0;
    draw();
  });
  draw();
});

mountAll();

/**
 * Widgets specific to lesson-2.html (moving, and what it costs the belief).
 * Shared infrastructure (mount, drawing helpers, the lamp world constants)
 * lives in widget.js.
 */

import {
  mount,
  mountAll,
  ctxOf,
  label,
  drawGrid,
  drawIsoPlane,
  oneHot,
  COLOUR,
  ROWS,
  COLS,
  N,
  DARK,
  LIT,
  LIT_MASK,
  DARK_MASK,
  A_LAMP,
  A_FEATURELESS,
  AGENT_TILE,
} from "./widget.js";
import { transitionModel, predict, update } from "./aif.js";

// --- permutation-vs-spread -------------------------------------------------
// Figure: the two sources of uncertainty, each with its own control, so the
// reader removes one at a time and watches what changes (content/lesson-2.md,
// "Why the table is not a permutation").
//
// The slider is the agent's confidence in its own step, NOT the world's
// behaviour: the true position always lands where it aimed. That is the
// section's actual claim, since the agent has no way of telling whether a step
// carried, and it means the deterministic world never needs explaining away.
// At 1.00 B is a permutation and the belief slides without changing shape.
//
// The toggle is the other uncertainty. In the featureless world every tile
// reports the same thing, so the look button visibly does nothing: aliasing is
// not undone by looking, only by going somewhere that reports something else.
mount("permutation-vs-spread", (el) => {
  // Actions, in the order aif.js MOVES uses them.
  const UP = 0;
  const DOWN = 1;
  const LEFT = 2;
  const RIGHT = 3;
  const STAY = 4;

  const W = 760;
  const H = 390; // the caption lives in the DOM below, not on the canvas
  const CELL = 44;
  const GRID_W = COLS * CELL;
  const LEFT_X = 40;
  const GRID_Y = 44;
  const TITLE_Y = 22;
  const GRID_BOTTOM = GRID_Y + ROWS * CELL;

  // The belief is drawn as an isometric bump field: height reads as magnitude
  // at a glance, which the flat opacity version did not. The world stays flat,
  // because it is a map and lit/dark is categorical, so height would say
  // nothing there. ISO_X/Y is the projection's origin, not its left edge.
  const ISO_X = 555;
  const ISO_Y = 120;
  const ISO_CELL = 26;

  // The spread meter and the step counter sit under the belief panel.
  const PANEL_X = 420;
  const METER_Y = GRID_BOTTOM + 28;
  const BAR_H = 14;
  const COUNTER_Y = METER_Y + 62;

  const START_TILE = AGENT_TILE;
  const STEPS_CALLOUT = 4; // steps without looking before the caption says so
  // How much the spread has to fall for the caption to claim the belief
  // tightened. It has to match what the reader can actually SEE move on the
  // bar: a dark reading in a mostly-dark world drops entropy by about 0.03,
  // which is real evidence but invisible, and calling that "tightened" would
  // have the caption assert something the picture does not show.
  const TIGHTENED = 0.08;
  const BARELY = 0.01; // below this, nothing moved at all

  // --- controls ------------------------------------------------------------
  const controls = document.createElement("div");
  controls.style.margin = "0 0 0.75rem";

  const sliderWrap = document.createElement("label");
  sliderWrap.style.display = "block";
  sliderWrap.style.margin = "0 0 0.5rem";
  sliderWrap.textContent = "how sure the agent is that its step landed where it aimed: ";
  const slider = document.createElement("input");
  slider.type = "range";
  slider.min = "0.6";
  slider.max = "1";
  slider.step = "0.05";
  slider.value = "1";
  const sliderOut = document.createElement("span");
  sliderOut.textContent = "1.00";
  sliderWrap.append(slider, sliderOut);

  const worldWrap = document.createElement("div");
  worldWrap.style.margin = "0 0 0.5rem";
  worldWrap.textContent = "the tiles report: ";
  const lampsBtn = document.createElement("button");
  lampsBtn.textContent = "lamps";
  const featurelessBtn = document.createElement("button");
  featurelessBtn.textContent = "featureless";
  worldWrap.append(lampsBtn, featurelessBtn);

  // A d-pad, not a row: the directions are directions, so they are laid out as
  // a cross and read as one without anybody having to parse five words.
  const moves = document.createElement("div");
  moves.style.display = "grid";
  moves.style.gridTemplateColumns = "repeat(3, 4.2rem)";
  moves.style.gap = "0.25rem";
  moves.style.margin = "0 0 0.6rem";

  const actions = document.createElement("div");
  const lookBtn = document.createElement("button");
  lookBtn.textContent = "look";
  const resetBtn = document.createElement("button");
  resetBtn.textContent = "reset";
  actions.append(lookBtn, resetBtn);

  controls.append(sliderWrap, worldWrap);
  el.appendChild(controls);

  const ctx = ctxOf(el, W, H);
  if (!ctx) return;

  // The movement controls go BELOW the canvas, under the left grid: the d-pad
  // belongs to the world panel, not to the settings above it.
  const pad = document.createElement("div");
  pad.style.margin = "0.5rem 0 0";
  pad.style.paddingLeft = `${LEFT_X}px`;
  pad.append(moves, actions);
  el.appendChild(pad);

  // The caption is DOM rather than canvas text: it wraps on its own, and it
  // stays the last thing in the widget however tall the controls get.
  const captionEl = document.createElement("p");
  captionEl.style.margin = "0.75rem 0 0";
  captionEl.style.paddingLeft = `${LEFT_X}px`;
  el.appendChild(captionEl);

  // --- state ---------------------------------------------------------------
  let reliability = 1;
  let world = "lamps";
  let truePos = START_TILE;
  let belief = oneHot(START_TILE);
  let sinceLook = 0;
  let caption = "A sharp belief, and nothing has happened to it yet.";
  let B = transitionModel({ rows: ROWS, cols: COLS, reliability });

  const litMask = () => (world === "lamps" ? LIT_MASK : DARK_MASK);
  const obsModel = () => (world === "lamps" ? A_LAMP : A_FEATURELESS);

  // Normalised entropy, 0 on a single cell and 1 when the belief is even across
  // every tile. Explained in the "If you want the number behind the bar" aside.
  function entropy(b) {
    let h = 0;
    for (const p of b) if (p > 0) h -= p * Math.log(p);
    return h / Math.log(N);
  }

  function reset() {
    truePos = START_TILE;
    belief = oneHot(START_TILE);
    sinceLook = 0;
    caption = "A sharp belief, and nothing has happened to it yet.";
    draw();
  }

  function move(action) {
    const row = Math.floor(truePos / COLS);
    const col = truePos % COLS;
    // The true step always lands where it aimed; only the belief hedges. The
    // grid wraps, exactly as B does, so the agent and its own model never
    // disagree about where a step off the edge leads.
    const dr = action === UP ? -1 : action === DOWN ? 1 : 0;
    const dc = action === LEFT ? -1 : action === RIGHT ? 1 : 0;
    const nr = (((row + dr) % ROWS) + ROWS) % ROWS;
    const nc = (((col + dc) % COLS) + COLS) % COLS;
    const wrapped = nr !== row + dr || nc !== col + dc;
    truePos = nr * COLS + nc;

    belief = predict(belief, B, action);
    sinceLook += 1;

    if (reliability >= 1) {
      caption =
        "Certain of its own step, the belief moved with the agent and kept its shape.";
    } else if (sinceLook >= STEPS_CALLOUT) {
      caption = `${sinceLook} steps and no looking. Each step's spread is still in the belief when the next one adds to it.`;
    } else {
      caption =
        "The belief spread. The agent cannot tell whether the step carried, so it has to allow for both.";
    }
    draw();
  }

  function look() {
    const before = entropy(belief);
    const obs = litMask()[truePos] ? LIT : DARK;
    belief = update(belief, obsModel(), obs).posterior;
    sinceLook = 0;
    const after = entropy(belief);

    const drop = before - after;
    if (drop > TIGHTENED) {
      caption = "One reading, and the belief tightened. Looking is what undoes the spread from moving.";
    } else if (world === "featureless") {
      caption =
        "Every tile reports the same thing, so this reading said nothing about which one the agent is on.";
    } else if (drop > BARELY) {
      // A dark reading where most tiles are dark: real evidence, but weak, and
      // the bar hardly moves. Say so rather than claim a tightening.
      caption =
        "Dark, like most of the grid. That narrows things barely at all: a reading is only worth as much as the tiles it rules out.";
    } else {
      caption =
        "Certain it is on a lit tile, and it cannot get more certain. Three tiles report the same thing; only moving somewhere different will separate them.";
    }
    draw();
  }

  // The spread meter: one bar, filled to `value` in 0..1, ends named.
  function drawMeter(value) {
    label(ctx, "how spread out that belief is (entropy)", PANEL_X, METER_Y, COLOUR.dim, 14);
    const barY = METER_Y + 10;
    ctx.fillStyle = "#e8e8e6";
    ctx.fillRect(PANEL_X, barY, GRID_W, BAR_H);
    ctx.fillStyle = COLOUR.neutral;
    ctx.fillRect(PANEL_X, barY, GRID_W * value, BAR_H);
    ctx.strokeStyle = COLOUR.rule;
    ctx.lineWidth = 1;
    ctx.strokeRect(PANEL_X + 0.5, barY + 0.5, GRID_W - 1, BAR_H - 1);
    label(ctx, "sharp", PANEL_X, barY + BAR_H + 15, COLOUR.dim, 13);
    ctx.textAlign = "right";
    ctx.fillStyle = COLOUR.dim;
    ctx.font = "13px system-ui, sans-serif";
    ctx.fillText("flat", PANEL_X + GRID_W, barY + BAR_H + 15);
    ctx.textAlign = "left";
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);

    label(ctx, "where the agent is", LEFT_X, TITLE_Y, COLOUR.ink);
    label(ctx, "where the agent thinks it is", PANEL_X, TITLE_Y, COLOUR.ink);

    drawGrid(ctx, LEFT_X, GRID_Y, CELL, { agent: truePos, lit: litMask() });

    // Scale the bumps against this belief's own peak, the same rule
    // drawBeliefBar uses: a belief spread over a dozen cells peaks near 0.1
    // and would otherwise render as an almost flat plane, hiding exactly the
    // shape the figure is about.
    const peak = Math.max(...belief, 1e-9);
    drawIsoPlane(ctx, ISO_X, ISO_Y, ISO_CELL, {
      heights: belief.map((p) => p / peak),
    });

    drawMeter(entropy(belief));

    label(ctx, "steps taken since the last observation", PANEL_X, COUNTER_Y, COLOUR.dim, 14);
    ctx.fillStyle = COLOUR.ink;
    ctx.font = "22px system-ui, sans-serif";
    ctx.fillText(String(sinceLook), PANEL_X, COUNTER_Y + 28);

    const hint =
      world === "featureless"
        ? "Try looking. Then try looking again."
        : reliability >= 1
          ? "Moving costs nothing here. Try lowering it."
          : "Move a few times without looking, then look.";
    label(ctx, hint, LEFT_X, GRID_BOTTOM + 30, COLOUR.dim, 14);
    captionEl.textContent = caption;

  }

  // --- wiring --------------------------------------------------------------
  // Row 1: north. Row 2: west, stay, east. Row 3: south. Blank cells hold the
  // cross together.
  for (const [col, name, action] of [
    [2, "north", UP],
    [1, "west", LEFT],
    [2, "stay", STAY],
    [3, "east", RIGHT],
    [2, "south", DOWN],
  ]) {
    const b = document.createElement("button");
    b.textContent = name;
    b.style.gridColumn = String(col);
    b.style.margin = "0";
    b.addEventListener("click", () => move(action));
    moves.appendChild(b);
  }
  lookBtn.addEventListener("click", look);
  resetBtn.addEventListener("click", reset);

  slider.addEventListener("input", () => {
    reliability = Number(slider.value);
    sliderOut.textContent = reliability.toFixed(2);
    // Rebuild B but leave the belief alone: changing the slider mid-run and
    // watching the next step behave differently is the point of the control.
    B = transitionModel({ rows: ROWS, cols: COLS, reliability });
    draw();
  });

  lampsBtn.addEventListener("click", () => {
    world = "lamps";
    reset();
  });
  featurelessBtn.addEventListener("click", () => {
    world = "featureless";
    reset();
  });

  draw();
});

mountAll();

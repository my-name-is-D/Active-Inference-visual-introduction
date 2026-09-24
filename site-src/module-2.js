/**
 * Widgets specific to module-2.html (moving, and what it costs the belief).
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
  drawRow,
  drawOperand,
  OP_CELL,
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
import { transitionModel, predict, update, lampObservationModel } from "./aif.js";

// --- permutation-vs-spread -------------------------------------------------
// Figure: the two sources of uncertainty, each with its own control, so the
// reader removes one at a time and watches what changes (content/module-2.md,
// "Why the table is not a permutation").
//
// The slider is the agent's confidence in its own step, NOT the world's
// behaviour: the true position always lands where it aimed. That is the
// section's actual claim, since the agent has no way of telling whether a step
// carried, and it means the deterministic world never needs explaining away.
// At 1.00 B is a permutation and the belief slides without changing shape.
//
// The toggle is the other uncertainty. In the featureless world every cell
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
  worldWrap.textContent = "the cells report: ";
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
  // every cell. Explained in the "If you want the number behind the bar" aside.
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
        "Every cell reports the same thing, so this reading said nothing about which one the agent is on.";
    } else if (drop > BARELY) {
      // A dark reading where most cells are dark: real evidence, but weak, and
      // the bar hardly moves. Say so rather than claim a tightening.
      caption =
        "Dark, like most of the grid. That narrows things barely at all: a reading is only worth as much as the cells it rules out.";
    } else {
      caption =
        "Certain it is on a lit cell, and it cannot get more certain. Three cells report the same thing; only moving somewhere different will separate them.";
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

// --- b-columns -------------------------------------------------------------
// Figure: what a row and a column of B actually mean (content/module-2.md,
// "Each column sums to one, because wherever the agent starts, it ends up
// somewhere").
//
// The matrix carries no index labels, because index notation is the thing the
// reader does not have yet. Instead, hovering a cell lights the world on the
// left: that cell's COLUMN is where the agent is, its ROW is where it could end
// up. The reader learns the convention by watching two cells light up in a
// world they can see.
//
// A 3x3 world, so the whole 9x9 fits and every column is complete and visibly
// sums to one. The lesson's 5x5 world has the same structure at 25x25, where no
// crop would contain a whole column.
mount("b-columns", (el) => {
  const R = 3;
  const C = 3;
  const NS = R * C;
  const EAST = 3;
  const RELIABILITY = 0.6;

  const W = 700;
  const H = 400;
  const MCELL = 34; // one matrix cell
  const MAT_X = 320;
  const MAT_Y = 86;
  const WORLD_CELL = 54;
  const WORLD_X = 56;
  const WORLD_Y = 120;
  // The index gutters sit between the matrix and its axis titles, so the
  // titles have to clear them.
  const INDEX_GAP = 5; // index labels to the matrix edge
  const AXIS_GAP = 26; // axis rule and title to the matrix edge

  const B = transitionModel({ rows: R, cols: C, reliability: RELIABILITY });

  const ctx = ctxOf(el, W, H);
  if (!ctx) return;

  const captionEl = document.createElement("p");
  captionEl.style.margin = "0.75rem 0 0";
  captionEl.style.minHeight = "1.4em";
  el.appendChild(captionEl);

  let hover = null; // {row, col}

  // The readout states what the matrix entry IS, not what the agent did. At
  // this point in the lesson B is a table of probabilities and no motion has
  // happened, so language like "the step did not carry" would invent an event
  // the figure is not showing.
  function readout(origin, dest, p) {
    return (
      p.toFixed(2) +
      ": probability to reach cell " + dest +
      " from cell " + origin +
      " in one step."
    );
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);

    // --- the world -----------------------------------------------------
    // Hovering any cell in column j shows the WHOLE column on the map: the
    // origin, and every destination that one step east could reach, each
    // shaded by its probability. The reader sees the complete spread of a
    // single step rather than one origin-destination pair.
    label(ctx, "the world", WORLD_X, WORLD_Y - 22, COLOUR.ink);
    const origin = hover ? hover.col : null;
    for (let s = 0; s < NS; s++) {
      const r = Math.floor(s / C);
      const c = s % C;
      const x = WORLD_X + c * WORLD_CELL;
      const y = WORLD_Y + r * WORLD_CELL;
      ctx.fillStyle = "#f4f4f2";
      ctx.fillRect(x, y, WORLD_CELL - 3, WORLD_CELL - 3);

      if (origin !== null) {
        const p = B[s][origin][EAST];
        if (p > 1e-9) {
          ctx.globalAlpha = 0.2 + 0.8 * p;
          ctx.fillStyle = COLOUR.neutral;
          ctx.fillRect(x, y, WORLD_CELL - 3, WORLD_CELL - 3);
          ctx.globalAlpha = 1;
        }
        if (s === origin) {
          // The cell the agent is on: a red ring, so it reads as a position
          // rather than as another probability.
          ctx.strokeStyle = COLOUR.agent;
          ctx.lineWidth = 3;
          ctx.strokeRect(x + 1.5, y + 1.5, WORLD_CELL - 6, WORLD_CELL - 6);
        }
        if (hover && s === hover.row) {
          // The one cell the hovered entry is about.
          ctx.strokeStyle = COLOUR.ink;
          ctx.lineWidth = 2;
          ctx.setLineDash([4, 3]);
          ctx.strokeRect(x + 4.5, y + 4.5, WORLD_CELL - 12, WORLD_CELL - 12);
          ctx.setLineDash([]);
        }
      }

      ctx.strokeStyle = COLOUR.rule;
      ctx.lineWidth = 1;
      ctx.strokeRect(x + 0.5, y + 0.5, WORLD_CELL - 4, WORLD_CELL - 4);

      // The cell index, which is what ties the world to the matrix axes.
      const p = origin !== null ? B[s][origin][EAST] : 0;
      ctx.font = "13px system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillStyle = p > 0.45 ? "#fff" : COLOUR.dim;
      ctx.fillText(String(s), x + (WORLD_CELL - 3) / 2, y + (WORLD_CELL - 3) / 2);
      ctx.textAlign = "left";
      ctx.textBaseline = "alphabetic";
    }

    if (hover) {
      const ly = WORLD_Y + R * WORLD_CELL + 16;
      ctx.strokeStyle = COLOUR.agent;
      ctx.lineWidth = 3;
      ctx.strokeRect(WORLD_X + 1.5, ly + 1.5, 11, 11);
      label(ctx, "where you are", WORLD_X + 22, ly + 11, COLOUR.dim, 14);
      ctx.globalAlpha = 0.7;
      ctx.fillStyle = COLOUR.neutral;
      ctx.fillRect(WORLD_X, ly + 20, 13, 13);
      ctx.globalAlpha = 1;
      label(ctx, "where you could end up", WORLD_X + 22, ly + 31, COLOUR.dim, 14);
    }

    // --- the matrix ----------------------------------------------------
    label(ctx, "where you are", MAT_X, MAT_Y - AXIS_GAP - 8, COLOUR.ink);
    // Arrow along the top, so "where you are" reads as the horizontal axis.
    ctx.strokeStyle = COLOUR.dim;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(MAT_X, MAT_Y - AXIS_GAP);
    ctx.lineTo(MAT_X + C * R * MCELL - 4, MAT_Y - AXIS_GAP);
    ctx.stroke();

    ctx.save();
    ctx.translate(MAT_X - AXIS_GAP - 8, MAT_Y + (NS * MCELL) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.fillStyle = COLOUR.ink;
    ctx.font = "16px system-ui, sans-serif";
    ctx.fillText("where you end up", 0, 0);
    ctx.textAlign = "left";
    ctx.restore();
    ctx.beginPath();
    ctx.moveTo(MAT_X - AXIS_GAP, MAT_Y);
    ctx.lineTo(MAT_X - AXIS_GAP, MAT_Y + NS * MCELL - 4);
    ctx.stroke();

    // Index labels: these are what let the reader match a matrix position to a
    // cell in the world, which is the whole point of the figure.
    ctx.font = "12px system-ui, sans-serif";
    ctx.fillStyle = COLOUR.dim;
    ctx.textAlign = "center";
    for (let j = 0; j < NS; j++) {
      ctx.fillStyle = hover && hover.col === j ? COLOUR.agent : COLOUR.dim;
      ctx.fillText(String(j), MAT_X + j * MCELL + (MCELL - 2) / 2, MAT_Y - INDEX_GAP);
    }
    ctx.textAlign = "right";
    for (let i = 0; i < NS; i++) {
      ctx.fillStyle = hover && hover.row === i ? COLOUR.ink : COLOUR.dim;
      ctx.textBaseline = "middle";
      ctx.fillText(String(i), MAT_X - INDEX_GAP, MAT_Y + i * MCELL + (MCELL - 2) / 2);
    }
    ctx.textBaseline = "alphabetic";
    ctx.textAlign = "left";

    for (let i = 0; i < NS; i++) {
      for (let j = 0; j < NS; j++) {
        const x = MAT_X + j * MCELL;
        const y = MAT_Y + i * MCELL;
        const p = B[i][j][EAST];
        ctx.fillStyle = "#fff";
        ctx.fillRect(x, y, MCELL - 2, MCELL - 2);
        if (p > 1e-9) {
          ctx.globalAlpha = 0.15 + 0.85 * p;
          ctx.fillStyle = COLOUR.neutral;
          ctx.fillRect(x, y, MCELL - 2, MCELL - 2);
          ctx.globalAlpha = 1;
        }
        ctx.strokeStyle = COLOUR.rule;
        ctx.lineWidth = 1;
        ctx.strokeRect(x + 0.5, y + 0.5, MCELL - 3, MCELL - 3);
      }
    }

    if (hover) {
      // Outline the column only, never a fill: a translucent band washes out
      // the probability colours underneath, which are the data. The row is not
      // marked, since the column is what the world panel is showing.
      ctx.strokeStyle = COLOUR.agent;
      ctx.lineWidth = 2;
      ctx.strokeRect(
        MAT_X + hover.col * MCELL - 1.5,
        MAT_Y - 1.5,
        MCELL,
        NS * MCELL - 1,
      );
      // And a thin ring on the one entry under the cursor.
      ctx.strokeStyle = COLOUR.ink;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(
        MAT_X + hover.col * MCELL - 0.5,
        MAT_Y + hover.row * MCELL - 0.5,
        MCELL - 1,
        MCELL - 1,
      );
    }

    if (hover) {
      const p = B[hover.row][hover.col][EAST];
      captionEl.textContent = readout(hover.col, hover.row, p);
    } else {
      captionEl.textContent =
        "Hover any cell. Its column is where the agent is; its row is where that step could put it.";
    }
  }

  ctx.canvas.addEventListener("mousemove", (e) => {
    const rect = ctx.canvas.getBoundingClientRect();
    const scale = ctx.canvas.width / rect.width;
    const mx = (e.clientX - rect.left) * scale;
    const my = (e.clientY - rect.top) * scale;
    const col = Math.floor((mx - MAT_X) / MCELL);
    const row = Math.floor((my - MAT_Y) / MCELL);
    const next =
      col >= 0 && col < NS && row >= 0 && row < NS ? { row, col } : null;
    if (JSON.stringify(next) !== JSON.stringify(hover)) {
      hover = next;
      draw();
    }
  });
  ctx.canvas.addEventListener("mouseleave", () => {
    if (hover) {
      hover = null;
      draw();
    }
  });

  draw();
});

// --- b-stack ---------------------------------------------------------------
// Figure: B is a stack of matrices, one per action (content/module-2.md, under
// B_ijk = p(s' = i | s = j, a = k)).
//
// Static, not interactive: the only claim is "there are several of these", and
// a control would invite the reader to go looking for a difference between the
// tables that this figure is not making. The front table carries the real
// numbers for `stay`; the ones behind are blank, with a gap marked "..." so the
// stack reads as "as many as there are actions" rather than exactly three.
mount("b-stack", (el) => {
  const R = 3;
  const C = 3;
  const NS = R * C;
  const STAY = 4;
  const RELIABILITY = 0.6;

  const W = 700;
  const H = 340;
  const MCELL = 22;
  const PLANE_W = NS * MCELL;
  const PLANE_H = NS * MCELL;
  const SHEAR = 0.16;
  // Tight overlap, as in a stack of sheets: a wide gap turns the figure into
  // three separate matrices side by side instead of one stack seen at an angle.
  const STEP_X = 34;
  const STEP_Y = -22;
  const GAP_X = 62; // the extra room the "..." needs
  const FRONT_X = 190;
  const FRONT_Y = 286; // bottom-left corner of the front plane
  const GAP_AFTER = 1; // the "..." sits after this many planes behind the front
  const PLANES = 3; // front, one behind, then the last after the gap

  const B = transitionModel({ rows: R, cols: C, reliability: RELIABILITY });

  const ctx = ctxOf(el, W, H);
  if (!ctx) return;

  const px = (x, y, ox, oy) => [
    FRONT_X + ox + x + (PLANE_H - y) * SHEAR,
    FRONT_Y + oy - PLANE_H + y,
  ];

  function planePath(ox, oy) {
    const q = [
      px(0, 0, ox, oy),
      px(PLANE_W, 0, ox, oy),
      px(PLANE_W, PLANE_H, ox, oy),
      px(0, PLANE_H, ox, oy),
    ];
    ctx.beginPath();
    ctx.moveTo(q[0][0], q[0][1]);
    for (let i = 1; i < 4; i++) ctx.lineTo(q[i][0], q[i][1]);
    ctx.closePath();
  }

  // Plane k sits k steps behind the front, with an extra gap after GAP_AFTER
  // to make room for the "...".
  const offsetOf = (k) => {
    const past = k > GAP_AFTER ? 1 : 0;
    return [k * STEP_X + past * GAP_X, k * STEP_Y + past * STEP_Y];
  };

  function draw() {
    ctx.clearRect(0, 0, W, H);

    for (let k = PLANES - 1; k >= 0; k--) {
      const [ox, oy] = offsetOf(k);
      planePath(ox, oy);
      ctx.fillStyle = "#cfe0ee";
      ctx.fill();
      ctx.strokeStyle = "#222";
      ctx.lineWidth = 1;
      ctx.stroke();

      // Only the front plane carries the numbers.
      if (k === 0) {
        ctx.save();
        planePath(0, 0);
        ctx.clip();
        for (let i = 0; i < NS; i++) {
          for (let j = 0; j < NS; j++) {
            const v = B[i][j][STAY];
            const y0 = i * MCELL;
            const q = [
              px(j * MCELL, y0, 0, 0),
              px((j + 1) * MCELL, y0, 0, 0),
              px((j + 1) * MCELL, y0 + MCELL, 0, 0),
              px(j * MCELL, y0 + MCELL, 0, 0),
            ];
            ctx.beginPath();
            ctx.moveTo(q[0][0], q[0][1]);
            for (let m = 1; m < 4; m++) ctx.lineTo(q[m][0], q[m][1]);
            ctx.closePath();
            if (v > 1e-9) {
              ctx.globalAlpha = 0.2 + 0.8 * v;
              ctx.fillStyle = COLOUR.neutral;
              ctx.fill();
              ctx.globalAlpha = 1;
            }
            ctx.strokeStyle = "rgba(0,0,0,0.13)";
            ctx.lineWidth = 1;
            ctx.stroke();
          }
        }
        ctx.restore();
      }
    }

    // Labels along the top, one per plane, with the "..." in the label row
    // rather than between the planes: the shear makes the planes overlap, so
    // any point "in the gap" lands on top of a matrix and reads as a smudge.
    const names = ["stay", "south", "east"];
    for (let k = 0; k < PLANES; k++) {
      const [ox, oy] = offsetOf(k);
      const [lx, ly] = px(0, 0, ox, oy);
      label(ctx, names[k], lx - 2, ly - 12, k === 0 ? COLOUR.ink : COLOUR.dim, 15);
      // Between the last two labels, mark that the stack continues.
      if (k === GAP_AFTER) {
        const [nx, ny] = offsetOf(k + 1);
        const [mx, my] = px(0, 0, nx, ny);
        ctx.fillStyle = COLOUR.dim;
        ctx.font = "17px system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("...", (lx + mx) / 2 + 12, (ly + my) / 2 - 12);
        ctx.textAlign = "left";
      }
    }
  }

  draw();
});
// --- conserves-vs-not ------------------------------------------------------
// Figure: prediction conserves total probability, the update does not
// (content/module-2.md, "Why one multiplication needs fixing and the other
// does not").
//
// No world panel. This is the one section of lesson 2 that is not about where
// the agent is, it is about what happens to a sum, and a grid would invite
// "which cell is it on?" when the answer is irrelevant.
//
// The re-roll button is the argument, not decoration: over 200 random beliefs
// the predicted total never leaves 1.0 by more than 4e-16, while the
// unnormalised update total ranges from 0.15 to 0.67.
mount("conserves-vs-not", (el) => {
  const NCELL = 5;
  const EAST = 3;
  const RELIABILITY = 0.6;
  const LIT = [true, false, false, true, false];

  const W = 760;
  const H = 600;
  const LEFT_X = 24;
  const BAR_X = 250; // where the row of values starts
  const BAR_W = 56; // one value column
  const TOTAL_X = BAR_X + NCELL * BAR_W + 26;
  const ROW_Y = [346, 416, 486, 556]; // belief, predict, update, normalised
  const BAR_H = 26;

  // The two operands, shown above the rows that use them. Without these the
  // reader is asked to trust two multiplications whose inputs are invisible,
  // and the asymmetry between them (every B column totals 1.00, the A row
  // totals 1.90) is the cause the paragraph below the figure explains.
  const B_X = 92;
  const B_Y = 64;
  const A_X = 470;
  const A_Y = B_Y + 2 * OP_CELL;

  const B = transitionModel({ rows: 1, cols: NCELL, reliability: RELIABILITY });
  const A = lampObservationModel(LIT);

  const controls = document.createElement("div");
  controls.style.margin = "0 0 0.75rem";
  const rollBtn = document.createElement("button");
  rollBtn.textContent = "new belief";
  controls.appendChild(rollBtn);
  el.appendChild(controls);

  const ctx = ctxOf(el, W, H);
  if (!ctx) return;

  // The traced arithmetic, in the DOM so it wraps on its own. Blue for numbers
  // read out of a table (B or the likelihood), black for the belief, so the
  // reader can see which factor came from where without a legend.
  const readoutEl = document.createElement("p");
  readoutEl.style.margin = "0.75rem 0 0";
  readoutEl.style.minHeight = "1.4em";
  el.appendChild(readoutEl);

  const TABLE_COLOUR = COLOUR.neutral;
  function setReadout(parts) {
    readoutEl.replaceChildren();
    for (const [text, kind] of parts) {
      const span = document.createElement("span");
      span.textContent = text;
      if (kind === "table") span.style.color = TABLE_COLOUR;
      else if (kind === "dim") span.style.color = COLOUR.dim;
      readoutEl.appendChild(span);
    }
  }

  let belief = [0.3, 0.45, 0.1, 0.1, 0.05];
  let hover = null; // {row: 1 for predict or 2 for update, cell}

  function roll() {
    const r = Array.from({ length: NCELL }, () => 0.05 + Math.random());
    const t = r.reduce((a, b) => a + b, 0);
    belief = r.map((x) => x / t);
    draw();
  }

  const sum = (v) => v.reduce((a, b) => a + b, 0);

  function draw() {
    ctx.clearRect(0, 0, W, H);

    const predicted = predict(belief, B, EAST);
    const u = update(belief, A, 1);

    // The two operands, with their totals. B's columns each read 1.00; A's one
    // row reads 1.90, and that difference is the whole of the section.
    const Bgrid = Array.from({ length: NCELL }, (_, i) =>
      Array.from({ length: NCELL }, (_, j) => B[i][j][EAST]),
    );
    drawOperand(ctx, B_X, B_Y, Bgrid, {
      totalsAlong: "cols",
      title: "Transition matrix B",
      colAxis: "where you are",
      rowAxis: "where you end up",
    });

    // Tracing: predict pulls a whole ROW of B and adds the products; update
    // pulls ONE likelihood entry and multiplies. Showing which entries are in
    // play is the contrast the section is about.
    if (hover && hover.row === 1) {
      ctx.strokeStyle = COLOUR.agent;
      ctx.lineWidth = 2;
      ctx.strokeRect(B_X - 1.5, B_Y + hover.cell * OP_CELL - 1.5, NCELL * OP_CELL, OP_CELL);
    }
    if (hover && hover.row === 2) {
      ctx.strokeStyle = COLOUR.agent;
      ctx.lineWidth = 2;
      ctx.strokeRect(A_X + hover.cell * OP_CELL - 1.5, A_Y - 1.5, OP_CELL, OP_CELL);
    }
    drawOperand(ctx, A_X, A_Y, [A[1]], {
      totalsAlong: "row",
      title: "the likelihood A, for a lit reading",
    });
    // The expression the row is an instance of: the observation is held fixed
    // at lit, and the cell varies across the five entries.
    ctx.font = "15px system-ui, sans-serif";
    ctx.fillStyle = COLOUR.ink;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText("p(o=lit | s)  =", A_X - 12, A_Y + (OP_CELL - 2) / 2);
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";

    ctx.font = "14px system-ui, sans-serif";
    ctx.fillStyle = COLOUR.dim;
    ctx.textAlign = "center";
    for (let j = 0; j < NCELL; j++) {
      ctx.fillText(String(j), BAR_X + j * BAR_W + (BAR_W - 8) / 2, ROW_Y[0] - BAR_H - 8);
    }
    ctx.textAlign = "left";
    label(ctx, "state", BAR_X - 38, ROW_Y[0] - BAR_H - 4, COLOUR.dim, 12);

    drawRow(ctx, ROW_Y[0], {
      title: "prior",
      subtitle: "",
      values: belief,
      n: NCELL,
      totals: [{ label: "total", value: sum(belief).toFixed(2), note: "", emphasis: false }],
    });

    // Which belief entries feed the hovered result cell.
    if (hover) {
      ctx.strokeStyle = COLOUR.ink;
      ctx.lineWidth = 2;
      for (let j = 0; j < NCELL; j++) {
        if (hover.row === 1) {
          // Every origin is in the sum. The ones whose B entry is zero
          // contribute nothing, but they are still summed over, so they are
          // marked faintly rather than left out: omitting them would suggest
          // the sum skips states, which is exactly the wrong idea.
          const contributes = B[hover.cell][j][EAST] > 1e-9;
          ctx.setLineDash(contributes ? [] : [3, 3]);
          ctx.globalAlpha = contributes ? 1 : 0.45;
        } else if (j !== hover.cell) {
          continue;
        }
        ctx.strokeRect(BAR_X + j * BAR_W - 1.5, ROW_Y[0] - BAR_H - 1.5, BAR_W - 5, BAR_H + 3);
        ctx.setLineDash([]);
        ctx.globalAlpha = 1;
      }
      const ry = ROW_Y[hover.row];
      ctx.strokeStyle = COLOUR.agent;
      ctx.strokeRect(BAR_X + hover.cell * BAR_W - 1.5, ry - BAR_H - 1.5, BAR_W - 5, BAR_H + 3);
    }
    drawRow(ctx, ROW_Y[1], {
      title: "predict transition",
      subtitle: "B @ belief",
      values: predicted,
      n: NCELL,
      totals: [{ label: "total", value: sum(predicted).toFixed(2), note: "already a belief", emphasis: false }],
    });
    drawRow(ctx, ROW_Y[2], {
      title: "explain the reading",
      subtitle: "likelihood x belief",
      values: u.unnormalised,
      n: NCELL,
      totals: [{ label: "total", value: sum(u.unnormalised).toFixed(2), note: "not a belief yet", emphasis: true }],
    });
    drawRow(ctx, ROW_Y[3], {
      title: "posterior",
      subtitle: `divide by ${sum(u.unnormalised).toFixed(2)}`,
      values: u.posterior,
      n: NCELL,
      totals: [{ label: "total", value: sum(u.posterior).toFixed(2), note: "a belief again", emphasis: false }],
    });
    // The arrow tying the update row to its normalised result.
    ctx.strokeStyle = COLOUR.dim;
    ctx.lineWidth = 1.5;
    // Over the values, not the totals column: normalising divides the bars.
    // Running it down the totals column crossed the posterior row's "total" label.
    const ax = BAR_X + (NCELL * BAR_W) / 2;
    const aTop = ROW_Y[2] + 26;
    const aBot = ROW_Y[3] - BAR_H - 10;
    ctx.beginPath();
    ctx.moveTo(ax, aTop);
    ctx.lineTo(ax, aBot);
    ctx.moveTo(ax - 4, aBot - 6);
    ctx.lineTo(ax, aBot);
    ctx.lineTo(ax + 4, aBot - 6);
    ctx.stroke();

    traceReadout();
  }

  // The arithmetic behind the hovered cell, written out with the table factors
  // in blue and the belief factors in black.
  function traceReadout() {
    if (!hover) {
      setReadout([["We see a lit lamp where we stand. Hover a number in the predict or update row to see where it came from.", "dim"]]);
      return;
    }
    const parts = [];
    if (hover.row === 1) {
      // All five origins, including the ones multiplied by zero: the sum runs
      // over every state, and showing only the survivors would misrepresent it.
      let total = 0;
      for (let j = 0; j < NCELL; j++) {
        const b = B[hover.cell][j][EAST];
        const kind = b > 1e-9 ? "table" : "dim";
        if (j > 0) parts.push([" + ", "dim"]);
        parts.push([b.toFixed(2), kind]);
        parts.push([" x ", "dim"]);
        parts.push([belief[j].toFixed(2), b > 1e-9 ? "ink" : "dim"]);
        total += b * belief[j];
      }
      parts.push([" = " + total.toFixed(3), "ink"]);
      parts.push(["    the sum runs over every state, and the zeros drop out", "dim"]);
    } else {
      const l = A[1][hover.cell];
      const q = belief[hover.cell];
      parts.push([l.toFixed(2), "table"]);
      parts.push([" x ", "dim"]);
      parts.push([q.toFixed(2), "ink"]);
      parts.push([" = " + (l * q).toFixed(3), "ink"]);
      parts.push(["    one state, one product", "dim"]);
    }
    setReadout(parts);
  }

  ctx.canvas.addEventListener("mousemove", (e) => {
    const rect = ctx.canvas.getBoundingClientRect();
    const scale = ctx.canvas.width / rect.width;
    const mx = (e.clientX - rect.left) * scale;
    const my = (e.clientY - rect.top) * scale;
    const cell = Math.floor((mx - BAR_X) / BAR_W);
    let row = null;
    for (const r of [1, 2]) {
      if (my >= ROW_Y[r] - BAR_H - 4 && my <= ROW_Y[r] + 18) row = r;
    }
    const next = row !== null && cell >= 0 && cell < NCELL ? { row, cell } : null;
    if (JSON.stringify(next) !== JSON.stringify(hover)) {
      hover = next;
      draw();
    }
  });
  ctx.canvas.addEventListener("mouseleave", () => {
    if (hover) {
      hover = null;
      draw();
    }
  });

  rollBtn.addEventListener("click", roll);
  draw();
});

mountAll();

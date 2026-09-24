/**
 * Widgets specific to module-3.html (free energy: scoring a belief you cannot
 * check against the answer). Shared infrastructure lives in widget.js.
 *
 * Every figure here uses the same five-cell world as lesson 2's
 * conserves-vs-not, with cells 0 and 3 lit, so a reader arriving from that
 * page is scoring beliefs over a world they have already seen the arithmetic
 * for. The prior is fixed across all three figures and deliberately not flat:
 * F depends on it, and a flat prior would make the posterior a rescaled
 * likelihood and hide that dependence.
 */

import {
  mount,
  mountAll,
  ctxOf,
  label,
  drawRow,
  drawOperand,
  ROW,
  autoplay,
  COLOUR,
} from "./widget.js";
import { lampObservationModel } from "./aif.js";

const NCELL = 5;
const LIT_ROW = [true, false, false, true, false];
const A = lampObservationModel(LIT_ROW);
const OBS = 1; // lit
const PRIOR = [0.1, 0.25, 0.3, 0.25, 0.1];

// The exact posterior, and the joint p(o,s) that free energy is scored
// against. The joint is what the agent actually holds: the likelihood column
// read at the observation, times its own prior. The posterior needs the total
// as well, which is the sum the lesson opens by calling expensive.
const LIKE = A[OBS];
const JOINT = PRIOR.map((p, s) => LIKE[s] * p);
const EVIDENCE = JOINT.reduce((a, b) => a + b, 0);
const POSTERIOR = JOINT.map((j) => j / EVIDENCE);

// Where a number came from, kept apart by colour: the prior the agent brought
// with it, and the likelihood it read out of A. Every product in this lesson
// has one factor of each, so the two are never the same colour.
const PRIOR_COLOUR = "#3f8f5c";
const LIKE_COLOUR = COLOUR.neutral;

const sum = (v) => v.reduce((a, b) => a + b, 0);
const freeEnergy = (q) => sum(q.map((x, s) => (x < 1e-12 ? 0 : x * Math.log(x / JOINT[s]))));
const divergence = (q) => sum(q.map((x, s) => (x < 1e-12 ? 0 : x * Math.log(x / POSTERIOR[s]))));

// The named situations the prose describes, in the order it describes them.
const NAMED = [
  ["flat", [0.2, 0.2, 0.2, 0.2, 0.2]],
  ["confident right", [0.05, 0.05, 0.05, 0.8, 0.05]],
  ["confident wrong", [0.05, 0.8, 0.05, 0.05, 0.05]],
  ["partly right", [0.1, 0.2, 0.2, 0.4, 0.1]],
  ["exact posterior", POSTERIOR],
];

function randomQ() {
  const r = Array.from({ length: NCELL }, () => 0.05 + Math.random());
  const t = sum(r);
  return r.map((x) => x / t);
}

// --- kl-never-negative -----------------------------------------------------
// Figure: the reader has just been told the divergence is never negative, and
// has been given a reason to doubt it, since the prose admits individual terms
// go negative (content/module-3.md, "It is never negative").
//
// The argument is entirely in the last two rows. The log ratio goes negative
// wherever q is less confident than the posterior; reweighting by q shrinks
// exactly those terms, because q being small there is what made the ratio
// small. The two rows therefore MUST share one scale, or the reweighting
// rescales back to full height and the figure argues the opposite of the text.
mount("kl-never-negative", (el) => {
  const TEXT = 18; // floor for every written label in this figure
  const W = 760;
  const H = 470;
  const ROW_Y = [126, 222, 330, 428]; // q, posterior, log ratio, weighted term

  const controls = document.createElement("div");
  controls.style.margin = "0 0 0.75rem";
  const buttons = [];
  for (const [name] of NAMED) {
    const b = document.createElement("button");
    b.textContent = name;
    controls.appendChild(b);
    buttons.push([b, name]);
  }
  const rollBtn = document.createElement("button");
  rollBtn.textContent = "random";
  controls.appendChild(rollBtn);
  el.appendChild(controls);

  const counterEl = document.createElement("p");
  counterEl.style.margin = "0.5rem 0 0";
  counterEl.style.fontSize = `${TEXT}px`;
  counterEl.style.color = COLOUR.dim;
  el.appendChild(counterEl);

  const ctx = ctxOf(el, W, H);
  if (!ctx) return;

  const readoutEl = document.createElement("p");
  readoutEl.style.margin = "0.75rem 0 0";
  readoutEl.style.minHeight = "1.4em";
  readoutEl.style.fontSize = `${TEXT}px`;
  el.appendChild(readoutEl);

  let q = NAMED[0][1];
  let tried = 0;
  let lowest = Infinity;
  let hover = null;

  function setReadout(parts) {
    readoutEl.replaceChildren();
    for (const [text, kind] of parts) {
      const span = document.createElement("span");
      span.textContent = text;
      if (kind === "table") span.style.color = COLOUR.neutral;
      else if (kind === "dim") span.style.color = COLOUR.dim;
      readoutEl.appendChild(span);
    }
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);

    const ratio = q.map((x, s) => (x < 1e-12 ? 0 : Math.log(x / POSTERIOR[s])));
    const term = q.map((x, s) => (x < 1e-12 ? 0 : x * ratio[s]));
    const total = sum(term);

    ctx.font = "12px system-ui, sans-serif";
    ctx.fillStyle = COLOUR.dim;
    ctx.textAlign = "center";
    for (let j = 0; j < NCELL; j++) {
      ctx.fillText(String(j), 250 + j * 56 + 24, ROW_Y[0] - 26 - 8);
    }
    ctx.textAlign = "left";
    label(ctx, "state", 176, ROW_Y[0] - 32, COLOUR.dim, TEXT);

    drawRow(ctx, ROW_Y[0], {
      title: "q(s)",
      subtitle: "the belief being scored",
      values: q,
      n: NCELL,
      textSize: TEXT,
      totals: [{ label: "total", value: sum(q).toFixed(2), note: "a belief" }],
    });
    drawRow(ctx, ROW_Y[1], {
      title: "p(s | o)",
      subtitle: "the true posterior",
      values: POSTERIOR,
      n: NCELL,
      textSize: TEXT,
      dimmed: true,
      totals: [{ label: "total", value: "1.00", note: "fixed" }],
    });

    // Each signed row takes its own scale. A shared one flattens the weighted
    // row into slivers, because the log ratio is several times larger, and the
    // reweighting then reads as "row 4 is small" rather than "the negative
    // terms shrank". The numbers carry the comparison; the bars carry the sign.
    drawRow(ctx, ROW_Y[2], {
      title: "log q(s) / p(s | o)",
      subtitle: "negative where q is the less confident",
      values: ratio,
      n: NCELL,
      textSize: TEXT,
      signed: true,
    });
    drawRow(ctx, ROW_Y[3], {
      title: "q(s) x log q(s) / p(s | o)",
      subtitle: "the same, weighted by q: the negative terms shrink",
      values: term,
      n: NCELL,
      textSize: TEXT,
      signed: true,
      totals: [{ label: "total", value: (total < 0 ? "" : "+") + total.toFixed(2), emphasis: true }],
    });

    if (hover !== null) {
      ctx.strokeStyle = COLOUR.agent;
      ctx.lineWidth = 2;
      for (const r of [0, 1, 2, 3]) {
        ctx.strokeRect(250 + hover * 56 - 1.5, ROW_Y[r] - 26 - 1.5, 51, 29);
      }
    }

    counterEl.textContent =
      `tried: ${tried} distributions      lowest total seen: ${lowest.toFixed(2)}` +
      (lowest < 1e-9 ? "  (the exact posterior)" : "");
    traceReadout(ratio, term);
  }

  function traceReadout(ratio, term) {
    if (hover === null) {
      setReadout([["Pick a belief, or roll a random one. Hover a state to see its term.", "dim"]]);
      return;
    }
    const s = hover;
    // Two decimals throughout, matching the numbers printed under the bars.
    // Quoting more precision here forces the reader to reconcile the 0.20 they
    // can see with a 0.2319 they cannot, which reads as an inconsistency.
    const f2 = (v) => v.toFixed(2);
    const sgn = (v) => (v >= 0 ? "+" : "") + v.toFixed(2);
    setReadout([
      [f2(q[s]), "ink"],
      [" x log( ", "dim"],
      [f2(q[s]), "ink"],
      [" / ", "dim"],
      [f2(POSTERIOR[s]), "table"],
      [" ) = ", "dim"],
      [f2(q[s]) + " x " + sgn(ratio[s]), "ink"],
      [" = " + sgn(term[s]), "ink"],
      [
        // A term goes negative when q is the less confident of the two, and
        // the weight that multiplies it is q itself, so the terms that could
        // drag the total down are exactly the ones held small. Saying "q is
        // small" would be wrong: what matters is q relative to the posterior.
        term[s] < 0
          ? `    negative: q is below the posterior here, and q is what weights it`
          : "    positive: q is at or above the posterior here",
        "dim",
      ],
    ]);
  }

  function pick(next) {
    q = next;
    tried += 1;
    lowest = Math.min(lowest, sum(q.map((x, s) => (x < 1e-12 ? 0 : x * Math.log(x / POSTERIOR[s])))));
    draw();
  }

  for (const [b, name] of buttons) {
    b.addEventListener("click", () => pick(NAMED.find(([n]) => n === name)[1]));
  }
  rollBtn.addEventListener("click", () => pick(randomQ()));

  ctx.canvas.addEventListener("mousemove", (e) => {
    const rect = ctx.canvas.getBoundingClientRect();
    const scale = ctx.canvas.width / rect.width;
    const mx = (e.clientX - rect.left) * scale;
    const my = (e.clientY - rect.top) * scale;
    const cell = Math.floor((mx - 250) / 56);
    const inRows = my > ROW_Y[0] - 40 && my < ROW_Y[3] + 24;
    const next = inRows && cell >= 0 && cell < NCELL ? cell : null;
    if (next !== hover) {
      hover = next;
      draw();
    }
  });
  ctx.canvas.addEventListener("mouseleave", () => {
    if (hover !== null) {
      hover = null;
      draw();
    }
  });

  pick(NAMED[0][1]);
});

// --- q-lives-in-the-simplex ------------------------------------------------
// Figure: the lesson has just said the agent's task changes from arithmetic
// with one answer to a search over the probability simplex (content/module-3.md,
// "The second is that the agent's task changes shape").
//
// That sentence is the lesson's central reframing and it is the one claim here
// that is purely geometric, so it gets the one figure in the lesson that draws
// a space rather than a distribution. The bars in the other figures show a
// belief's SHAPE; this shows its POSITION among all the beliefs there are.
//
// Three cells, because the 3-simplex is a filled triangle and can be drawn
// honestly. Four or more cannot, and the caption says so rather than pretending
// otherwise: the reader's inability to picture the five-cell case is the
// dimensionality point, not a defect of the figure.
//
// The constraint that the three numbers sum to one is not annotated here, it
// IS the picture: a point inside the triangle cannot violate it.
const SX_LIT = [true, false, true];
const SX_A = lampObservationModel(SX_LIT);
const SX_PRIOR = [0.2, 0.5, 0.3];
const SX_LIKE = SX_A[1];
const SX_JOINT = SX_PRIOR.map((p, s) => SX_LIKE[s] * p);
const SX_EVIDENCE = SX_JOINT.reduce((a, b) => a + b, 0);
const SX_POST = SX_JOINT.map((j) => j / SX_EVIDENCE);
const sxF = (q) => sum(q.map((x, s) => (x < 1e-12 ? 0 : x * Math.log(x / SX_JOINT[s]))));
const sxKL = (q) => sum(q.map((x, s) => (x < 1e-12 ? 0 : x * Math.log(x / SX_POST[s]))));

mount("q-lives-in-the-simplex", (el) => {
  const TEXT = 18;
  const W = 860;
  const H = 430;
  const TRI_CX = 210;   // centre of the triangle
  const TRI_TOP = 60;   // apex
  const TRI_H = 260;    // apex to base
  const TRI_HALF = 150; // half the base width
  const BAR_X = 470;
  const BAR_Y = 170;

  // The three corners, in the order the probabilities are indexed.
  const APEX = [TRI_CX, TRI_TOP];
  const LEFT = [TRI_CX - TRI_HALF, TRI_TOP + TRI_H];
  const RIGHT = [TRI_CX + TRI_HALF, TRI_TOP + TRI_H];
  const CORNERS = [APEX, LEFT, RIGHT];

  // Barycentric both ways. A point's weights ARE its probabilities, which is
  // the whole reason this projection is the honest one to use here.
  const toXY = (q) => [
    q[0] * APEX[0] + q[1] * LEFT[0] + q[2] * RIGHT[0],
    q[0] * APEX[1] + q[1] * LEFT[1] + q[2] * RIGHT[1],
  ];
  function toQ(x, y) {
    const d = (LEFT[1] - RIGHT[1]) * (APEX[0] - RIGHT[0]) + (RIGHT[0] - LEFT[0]) * (APEX[1] - RIGHT[1]);
    let a = ((LEFT[1] - RIGHT[1]) * (x - RIGHT[0]) + (RIGHT[0] - LEFT[0]) * (y - RIGHT[1])) / d;
    let b = ((RIGHT[1] - APEX[1]) * (x - RIGHT[0]) + (APEX[0] - RIGHT[0]) * (y - RIGHT[1])) / d;
    let c = 1 - a - b;
    // Dragging outside the triangle clamps back in, so q is always a valid
    // distribution: the reader cannot produce a belief that breaks the rule.
    a = Math.max(0, a); b = Math.max(0, b); c = Math.max(0, c);
    const t = a + b + c;
    return [a / t, b / t, c / t];
  }

  const controls = document.createElement("div");
  controls.style.margin = "0 0 0.75rem";
  const toPost = document.createElement("button");
  toPost.textContent = "snap to the posterior";
  const toFlat = document.createElement("button");
  toFlat.textContent = "snap to flat";
  controls.append(toPost, toFlat);
  el.appendChild(controls);

  const ctx = ctxOf(el, W, H);
  if (!ctx) return;
  ctx.canvas.style.cursor = "pointer";

  const note = document.createElement("p");
  note.style.margin = "0.75rem 0 0";
  note.style.fontSize = `${TEXT}px`;
  note.style.color = COLOUR.dim;
  note.textContent =
    "Drag the blue point anywhere inside the triangle. Three cells, so the space can be drawn in 2D";
  el.appendChild(note);

  let q = [0.55, 0.3, 0.15];
  let dragging = false;

  function draw() {
    ctx.clearRect(0, 0, W, H);

    label(ctx, "the space of all beliefs", 24, 30, COLOUR.ink, TEXT);
    label(ctx, "this belief", BAR_X, 30, COLOUR.ink, TEXT);

    // The triangle. Its interior is every list of three probabilities that
    // sums to one, which is what the prose has just named the simplex.
    ctx.beginPath();
    ctx.moveTo(APEX[0], APEX[1]);
    ctx.lineTo(LEFT[0], LEFT[1]);
    ctx.lineTo(RIGHT[0], RIGHT[1]);
    ctx.closePath();
    ctx.fillStyle = "#f4f6f8";
    ctx.fill();
    ctx.strokeStyle = COLOUR.rule;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Corners are certainty: all the probability on one cell.
    ctx.font = `13px system-ui, sans-serif`;
    ctx.fillStyle = COLOUR.dim;
    ctx.textAlign = "center";
    ctx.fillText("cell 0 certain", APEX[0], APEX[1] - 10);
    ctx.fillText("cell 1 certain", LEFT[0], LEFT[1] + 20);
    ctx.fillText("cell 2 certain", RIGHT[0], RIGHT[1] + 20);
    ctx.textAlign = "left";

    // Flat sits at the centroid: the belief that commits to nothing.
    const flat = toXY([1 / 3, 1 / 3, 1 / 3]);
    ctx.fillStyle = COLOUR.dim;
    ctx.beginPath();
    ctx.arc(flat[0], flat[1], 3, 0, 2 * Math.PI);
    ctx.fill();
    label(ctx, "flat", flat[0] + 8, flat[1] + 4, COLOUR.dim, 13);

    // The posterior: one point among all of them, and the one the agent is
    // looking for without being able to see where it is.
    const pp = toXY(SX_POST);
    ctx.strokeStyle = COLOUR.agent;
    ctx.lineWidth = 2;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.arc(pp[0], pp[1], 11, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = COLOUR.agent;
    ctx.beginPath();
    ctx.arc(pp[0], pp[1], 4, 0, 2 * Math.PI);
    ctx.fill();
    label(ctx, "the posterior", pp[0] - 34, pp[1] + 30, COLOUR.agent, 13);

    // q itself, the point the reader moves.
    const qp = toXY(q);
    ctx.fillStyle = COLOUR.neutral;
    ctx.beginPath();
    ctx.arc(qp[0], qp[1], 8, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.stroke();
    label(ctx, "q", qp[0] + 13, qp[1] - 10, COLOUR.neutral, TEXT);

    // The same belief as bars, so the point and the shape are read together.
    drawRow(ctx, BAR_Y, {
      title: "",
      values: q,
      n: 3,
      x: BAR_X,
      leftX: BAR_X,
      textSize: TEXT,
      totals: [{ label: "total", value: sum(q).toFixed(2), note: "always a belief" }],
    });

    ctx.font = "12px system-ui, sans-serif";
    ctx.fillStyle = COLOUR.dim;
    ctx.textAlign = "center";
    for (let i = 0; i < 3; i++) ctx.fillText(`cell ${i}`, BAR_X + i * ROW.barW + 24, BAR_Y - 34);
    ctx.textAlign = "left";

    const f = sxF(q);
    label(ctx, `F = ${f.toFixed(3)}`, BAR_X, 250, COLOUR.ink, TEXT);
    label(ctx, `lowest possible F = ${sxF(SX_POST).toFixed(3)}`, BAR_X, 276, COLOUR.dim, 14);
    label(ctx, `D_KL[q || p(s|o)] = ${sxKL(q).toFixed(3)}`, BAR_X, 308, COLOUR.dim, 14);
    label(ctx, "For visualisation only. Needs the posterior", BAR_X, 330, COLOUR.dim, 14);
    label(ctx, "to actually be computed", BAR_X, 350, COLOUR.dim, 14);

    label(ctx, "every point inside is a belief: the three numbers cannot help summing to one", 24, 380, COLOUR.dim, 14);
    label(ctx, "the corners are certainty, the middle is knowing nothing", 24, 402, COLOUR.dim, 14);
  }

  function pointerQ(e) {
    const rect = ctx.canvas.getBoundingClientRect();
    const scale = ctx.canvas.width / rect.width;
    return toQ((e.clientX - rect.left) * scale, (e.clientY - rect.top) * scale);
  }
  ctx.canvas.addEventListener("mousedown", (e) => {
    dragging = true;
    q = pointerQ(e);
    draw();
  });
  ctx.canvas.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    q = pointerQ(e);
    draw();
  });
  window.addEventListener("mouseup", () => {
    dragging = false;
  });
  toPost.addEventListener("click", () => {
    q = [...SX_POST];
    draw();
  });
  toFlat.addEventListener("click", () => {
    q = [1 / 3, 1 / 3, 1 / 3];
    draw();
  });

  draw();
});

// --- score-five-q ----------------------------------------------------------
// Figure: five beliefs scored against one observation, so the reader can see
// that F ranks them the same way the divergence does, and that the two differ
// by one fixed number (content/module-3.md, "Compare situations over one
// step"). The prior and the observation are drawn once at the top because the
// section insists the starting point is fixed: only q changes down the page.
mount("score-five-q", (el) => {
  const W = 1060;
  const H = 660;
  const HEAD_Y = 96;
  const ROW_Y = [270, 340, 410, 480, 550];

  const ctx = ctxOf(el, W, H);
  if (!ctx) return;

  const readoutEl = document.createElement("p");
  readoutEl.style.margin = "0.75rem 0 0";
  readoutEl.style.minHeight = "1.4em";
  el.appendChild(readoutEl);

  let hover = null;

  function setReadout(parts) {
    readoutEl.replaceChildren();
    for (const [text, kind] of parts) {
      if (kind === "break") {
        readoutEl.appendChild(document.createElement("br"));
        continue;
      }
      const span = document.createElement("span");
      span.textContent = text;
      if (kind === "table") span.style.color = LIKE_COLOUR;
      else if (kind === "prior") span.style.color = PRIOR_COLOUR;
      else if (kind === "dim") span.style.color = COLOUR.dim;
      readoutEl.appendChild(span);
    }
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);

    // The fixed starting point: the prior the last prediction step produced,
    // and the column of A read at the observation that arrived. Everything
    // below is scored against these two and nothing else.
    drawOperand(ctx, 92, HEAD_Y, [PRIOR], {
      totalsAlong: "row",
      title: "the prior p(s), from the last prediction step",
      colour: PRIOR_COLOUR,
    });
    drawOperand(ctx, 480, HEAD_Y, [LIKE], {
      totalsAlong: null,
      title: "the likelihood p(o = lit | s)",
      colour: LIKE_COLOUR,
    });
    label(ctx, "observation: lit", 480, HEAD_Y + 62, COLOUR.ink);
    ctx.fillStyle = COLOUR.lit;
    ctx.fillRect(608, HEAD_Y + 50, 14, 12);
    ctx.strokeStyle = COLOUR.litEdge;
    ctx.strokeRect(608.5, HEAD_Y + 50.5, 13, 11);

    label(ctx, "five beliefs, one observation, the same prior throughout", 24, 218, COLOUR.ink);
    ctx.font = "18px system-ui, sans-serif";
    ctx.fillStyle = COLOUR.ink;
    ctx.fillText("F", 250 + NCELL * 56 + 26, 240);
    ctx.fillText("D_KL[q || p(s|o)]", 250 + NCELL * 56 + 116, 240);
    ctx.fillText("F - D_KL", 250 + NCELL * 56 + 286, 240);

    NAMED.forEach(([name, q], i) => {
      const F = freeEnergy(q);
      const kl = divergence(q);
      drawRow(ctx, ROW_Y[i], {
        title: name,
        values: q,
        n: NCELL,
        dimmed: hover !== null && hover !== i,
        totals: [
          { label: "", value: F.toFixed(3), emphasis: i === 4, width: 90 },
          { label: "", value: kl.toFixed(3), emphasis: i === 4, width: 170 },
          { label: "", value: (F - kl).toFixed(3), width: 90 },
        ],
      });
    });

    // The difference column is the figure: it reads the same in every row,
    // and that number is -log p(o), which no comparison between rows can see.
    ctx.strokeStyle = COLOUR.rule;
    ctx.lineWidth = 1;
    const dx = 250 + NCELL * 56 + 280;
    ctx.strokeRect(dx, ROW_Y[0] - 34, 86, ROW_Y[4] - ROW_Y[0] + 44);
    // The constant column is the figure: what separates F from the divergence
    // does not depend on q, so it is the same in every row and cancels out of
    // every comparison between rows. That is what lets the agent use F at all.
    label(ctx, `the same in every row: -log p(o) = ${(-Math.log(EVIDENCE)).toFixed(3)}`, 24, ROW_Y[4] + 42, COLOUR.ink, 20);
    label(ctx, "it does not depend on q, so it cancels out of every comparison between rows", 24, ROW_Y[4] + 64, COLOUR.dim, 20);

    traceReadout();
  }

  function traceReadout() {
    if (hover === null) {
      setReadout([
        ["Hover a row to see how its score was built.      ", "dim"],
        ["green", "prior"],
        [": from the prior      ", "dim"],
        ["blue", "table"],
        [": read from A", "dim"],
      ]);
      return;
    }
    const [name, q] = NAMED[hover];
    const parts = [
      [`${name}:   F = `, "dim"],
    ];
    for (let s = 0; s < NCELL; s++) {
      if (s > 0) parts.push([" + ", "dim"]);
      parts.push([q[s].toFixed(2), "ink"]);
      parts.push(["·log(", "dim"]);
      parts.push([q[s].toFixed(2), "ink"]);
      parts.push([" / [", "dim"]);
      parts.push([PRIOR[s].toFixed(2), "prior"]);
      parts.push(["x", "dim"]);
      parts.push([LIKE[s].toFixed(2), "table"]);
      parts.push(["])", "dim"]);
    }
    parts.push([" = " + freeEnergy(q).toFixed(3), "ink"]);
    setReadout(parts);
  }

  ctx.canvas.addEventListener("mousemove", (e) => {
    const rect = ctx.canvas.getBoundingClientRect();
    const scale = ctx.canvas.width / rect.width;
    const my = (e.clientY - rect.top) * scale;
    let next = null;
    ROW_Y.forEach((y, i) => {
      if (my >= y - 30 && my <= y + 18) next = i;
    });
    if (next !== hover) {
      hover = next;
      draw();
    }
  });
  ctx.canvas.addEventListener("mouseleave", () => {
    if (hover !== null) {
      hover = null;
      draw();
    }
  });

  draw();
});

// --- descend-on-f ----------------------------------------------------------
// Figure: the claim being used rather than checked. An optimiser is handed one
// number, F, and lowers it; q ends up on the posterior it was never shown
// (content/module-3.md, "Step by step correction of one situation").
//
// The descent is on the softmax parameters of q rather than on q itself, so
// every step lands on a distribution without clamping or renormalising by
// hand. The gradient of F in those coordinates is q * (log q/joint - F), and
// the fixed point is the posterior: verified against the exact posterior in
// the check at the foot of this file.
const STEPS = 50;
// F falls steeply and then flattens, and no choice of rate changes that: it is
// the shape of the descent, not a plotting artefact. The rate is therefore set
// for convergence (9e-4 from the posterior by the last step) and the curve is
// left as steep as it honestly is. Stretching it to fill the panel is what the
// earlier log axis did, and that plotted the divergence instead of F.
const RATE = 1.5;

function descend() {
  let z = NAMED[2][1].map(Math.log); // start from "confident wrong"
  const trace = [];
  for (let i = 0; i <= STEPS; i++) {
    const m = Math.max(...z);
    const e = z.map((v) => Math.exp(v - m));
    const t = sum(e);
    const q = e.map((v) => v / t);
    trace.push(q);
    const F = freeEnergy(q);
    const g = q.map((x, s) => x * (Math.log(x / JOINT[s]) - F));
    z = z.map((v, s) => v - RATE * g[s]);
  }
  return trace;
}

mount("descend-on-f", (el) => {
  const W = 760;
  const H = 460;
  const Q_Y = 176;
  const PLOT_X = 250;
  const PLOT_Y = 262;
  const PLOT_W = NCELL * 56 - 8;
  const PLOT_H = 110;

  const trace = descend();
  const ctx = ctxOf(el, W, H);
  if (!ctx) return;

  let step = 0;

  function draw() {
    ctx.clearRect(0, 0, W, H);
    const q = trace[step];

    ctx.font = "12px system-ui, sans-serif";
    ctx.fillStyle = COLOUR.dim;
    ctx.textAlign = "center";
    for (let j = 0; j < NCELL; j++) ctx.fillText(String(j), 250 + j * 56 + 24, Q_Y - 34);
    ctx.textAlign = "left";

    // One fixed scale for the bars AND the outline. drawRow's default is to
    // scale each row by its own peak, which would draw q and the posterior at
    // the same height whatever they actually held, and the figure would show
    // them matching from the first frame.
    const SCALE = Math.max(...POSTERIOR, ...NAMED[2][1]);
    drawRow(ctx, Q_Y, {
      title: "q(s)",
      subtitle: `step ${step} of ${STEPS}`,
      values: q,
      n: NCELL,
      scale: SCALE,
      barH: 52,
    });

    // The posterior as a fixed outline over q: the target the optimiser is
    // never told about. q arriving on it is the whole claim.
    //
    // Both the scale and the width must match what drawRow just drew, or the
    // outline is a different quantity wearing the same picture: drawRow scales
    // each row by that row's own peak and draws bars barW-8 wide, so the
    // outline does the same. While q is still wrong its peak differs from the
    // posterior's, and the outline is genuinely taller or shorter than the
    // bars; they coincide only once q has converged, which is the event the
    // figure exists to show.
    ctx.strokeStyle = COLOUR.agent;
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 3]);
    for (let s = 0; s < NCELL; s++) {
      const h = (POSTERIOR[s] / SCALE) * 52;
      ctx.strokeRect(250 + s * 56, Q_Y - h, 48, h);
    }
    ctx.setLineDash([]);
    label(ctx, "dashed: the posterior, which the optimiser is never shown", 250, Q_Y + 36, COLOUR.agent, 13);

    // The descent is geometric: F falls 2.79 -> 1.09 in five steps and then
    // crawls, so a linear axis of F shows one cliff and a flat line. What is
    // plotted is therefore how far F still sits ABOVE its floor, on a log
    // axis, which turns that decay into a readable near-straight line.
    //
    // This quantity is not F, and the panel says so. It happens to equal the
    // divergence, since F and D_KL differ by exactly this floor; labelling it
    // "F" is what made an earlier version of this panel draw the one number
    // the figure states the agent cannot see.
    const floor = freeEnergy(POSTERIOR);
    const gaps = trace.map((t) => Math.max(freeEnergy(t) - floor, 1e-12));
    const lo = Math.log(Math.min(...gaps));
    const hi = Math.log(Math.max(...gaps));
    ctx.strokeStyle = COLOUR.rule;
    ctx.lineWidth = 1;
    ctx.strokeRect(PLOT_X + 0.5, PLOT_Y + 0.5, PLOT_W, PLOT_H);
    ctx.strokeStyle = COLOUR.neutral;
    ctx.lineWidth = 2;
    ctx.beginPath();
    gaps.forEach((g, i) => {
      const x = PLOT_X + (i / STEPS) * PLOT_W;
      const y = PLOT_Y + PLOT_H - ((Math.log(g) - lo) / (hi - lo || 1)) * PLOT_H;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
    const cx = PLOT_X + (step / STEPS) * PLOT_W;
    const cy = PLOT_Y + PLOT_H - ((Math.log(gaps[step]) - lo) / (hi - lo || 1)) * PLOT_H;
    ctx.fillStyle = COLOUR.agent;
    ctx.beginPath();
    ctx.arc(cx, cy, 4, 0, 2 * Math.PI);
    ctx.fill();
    label(ctx, "F, falling", 24, PLOT_Y + 16, COLOUR.ink);
    label(ctx, "plotted as how far it", 24, PLOT_Y + 38, COLOUR.dim, 13);
    label(ctx, "still sits above its floor,", 24, PLOT_Y + 55, COLOUR.dim, 13);
    label(ctx, "on a log scale", 24, PLOT_Y + 72, COLOUR.dim, 13);
    label(ctx, `floor = -log p(o)`, 24, PLOT_Y + 96, COLOUR.dim, 13);
    label(ctx, `= ${floor.toFixed(3)}`, 24, PLOT_Y + 113, COLOUR.dim, 13);

    label(ctx, `F = ${freeEnergy(q).toFixed(4)}`, 250, PLOT_Y + PLOT_H + 30, COLOUR.ink);
    label(
      ctx,
      `D_KL[q || p(s|o)] = ${divergence(q).toFixed(4)}`,
      420,
      PLOT_Y + PLOT_H + 30,
      COLOUR.dim,
    );
    label(ctx, "For visualisation only. Needs the posterior", 420, PLOT_Y + PLOT_H + 50, COLOUR.dim, 13);
    label(ctx, "to actually be computed", 420, PLOT_Y + PLOT_H + 68, COLOUR.dim, 13);
  }

  autoplay(el, (i) => {
    step = i;
    draw();
  }, STEPS + 1, 250);
});

mountAll();

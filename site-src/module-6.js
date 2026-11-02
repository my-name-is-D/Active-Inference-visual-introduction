// Module 6 widgets. Every number comes from verification/export_module6_data.py
// (module-6-data.js) or is recomputed live with the aif.js twins that the
// Python cross-checks through aif/belief_cases.json.
import { mount, mountAll, ctxOf, label, COLOUR } from "./widget.js";
import { softmax, accumulatedFreeEnergy, updatePrecision } from "./aif.js";
import { MODULE6_DATA as DATA } from "./module-6-data.js";

const SCORE = COLOUR.neutral;   // the score term, blue throughout this page
const HABIT = "#e67e50";        // the habit term, orange throughout this page
const OBS = [
  { name: "dark", colour: COLOUR.dark, text: "#fff" },
  { name: "lit", colour: COLOUR.lit, text: COLOUR.ink },
  { name: "home", colour: COLOUR.agent, text: "#fff" },
];

function bar(ctx, x, y, width, height, colour) {
  ctx.fillStyle = colour;
  ctx.fillRect(width >= 0 ? x : x + width, y, Math.abs(width), height);
}

function button(parent, text, onClick) {
  const b = document.createElement("button");
  b.type = "button";
  b.textContent = text;
  b.addEventListener("click", onClick);
  parent.appendChild(b);
  return b;
}

function row(el) {
  const div = document.createElement("div");
  div.style.margin = "0.4rem 0";
  el.appendChild(div);
  return div;
}

function radios(parent, name, options, onChange) {
  const inputs = options.map(([value, text], i) => {
    const wrap = document.createElement("label");
    wrap.style.marginRight = "1.2rem";
    const input = document.createElement("input");
    input.type = "radio";
    input.name = name;
    input.value = value;
    input.checked = i === 0;
    input.addEventListener("change", onChange);
    wrap.append(input, ` ${text}`);
    parent.appendChild(wrap);
    return input;
  });
  return () => inputs.find((input) => input.checked).value;
}

// --- Widget 1 ------------------------------------------------------------------
// "The greedy rule reads only the ordering, so the gap between the best policy
// and the next counts for nothing. Sampling reads the gaps."
mount("greedy-vs-sampling", (el) => {
  const W = 720, H = 230;
  const COL_SCORES = 20, COL_GREEDY = 270, COL_SAMPLE = 500, BAR_MAX = 150;
  const ROW_Y = [80, 120, 160], BAR_H = 22, SCORE_SCALE = 3.0, GAMMA = 1;
  const { names, G, tiedG } = DATA.scores;
  const ctx = ctxOf(el, W, H);

  const control = row(el);
  control.append("Gap between the scores: ");
  const slider = document.createElement("input");
  slider.type = "range";
  slider.min = "0";
  slider.max = "100";
  slider.step = "1";
  slider.value = "100";
  slider.setAttribute("aria-label", "Gap between the scores, percent of module 4's");
  const out = document.createElement("output");
  control.append(slider, " ", out);

  function draw() {
    const s = Number(slider.value) / 100;
    out.textContent = `${slider.value}%` + (s === 0 ? " (all equal: start unknown)" : "");
    const scores = G.map((g) => tiedG + s * (g - tiedG));
    const lowest = Math.min(...scores);
    const tied = scores.filter((g) => g - lowest < 1e-9).length;
    const greedy = scores.map((g) => (g - lowest < 1e-9 ? 1 / tied : 0));
    const q = softmax(scores, GAMMA);

    ctx.clearRect(0, 0, W, H);
    label(ctx, "Scores G (module 4, one step)", COL_SCORES, 30, COLOUR.ink, 15);
    label(ctx, "Take the lowest", COL_GREEDY, 30, COLOUR.ink, 15);
    label(ctx, `Sample from q(π), γ = ${GAMMA}`, COL_SAMPLE, 30, COLOUR.ink, 15);
    names.forEach((name, i) => {
      const y = ROW_Y[i];
      label(ctx, name, COL_SCORES, y + 16, COLOUR.ink, 14);
      bar(ctx, COL_SCORES + 50, y, (BAR_MAX * scores[i]) / SCORE_SCALE, BAR_H, SCORE);
      label(ctx, scores[i].toFixed(3), COL_SCORES + 56 + (BAR_MAX * scores[i]) / SCORE_SCALE, y + 16, COLOUR.dim, 13);
      bar(ctx, COL_GREEDY, y, BAR_MAX * greedy[i], BAR_H, COLOUR.agent);
      label(ctx, greedy[i] === 0 ? "0" : greedy[i].toFixed(2), COL_GREEDY + BAR_MAX * greedy[i] + 6, y + 16, COLOUR.dim, 13);
      bar(ctx, COL_SAMPLE, y, BAR_MAX * q[i], BAR_H, COLOUR.agent);
      label(ctx, q[i].toFixed(2), COL_SAMPLE + BAR_MAX * q[i] + 6, y + 16, COLOUR.dim, 13);
    });
    const gap = scores[1] - scores[0];
    label(ctx, tied > 1
      ? "All three scores are equal: taking the lowest now needs a tie-break; sampling gives 1/3 each."
      : `Gap east → north: ${gap.toFixed(2)}, compared with 1/γ = ${(1 / GAMMA).toFixed(0)}. ` +
        "Taking the lowest still picks east with certainty.",
      COL_SCORES, 210, COLOUR.dim, 14);
  }
  slider.addEventListener("input", draw);
  draw();
});

// --- Widget 2 ------------------------------------------------------------------
// "Comparing two policies, the habit favours i by ln(E_i/E_j) and the score
// favours j by γ(G_i − G_j), so the habit wins when ln(E_i/E_j) > γ(G_i − G_j)."
mount("habit-vs-score", (el) => {
  const W = 800, H = 250, GAMMA = 1;
  const AXIS_X0 = 150, AXIS_X1 = 650, AXIS_Y = 42, RATIO_MIN = 1, RATIO_MAX = 4;
  const ROW_Y = [110, 150, 190], BAR_H = 22;
  const HABIT_ZERO = 110, HABIT_PX = 60, SCORE_ZERO = 420, SCORE_PX = 45, Q_X = 470, Q_MAX = 180;
  const { names, G, tiedG } = DATA.scores;

  const controls = row(el);
  const mode = radios(controls, "habit-scores", [["module4", "start roughly known (module 4 : east best option)"], ["flat", "start unknown (all scores equal)"]], draw);
  const sliderRow = row(el);
  sliderRow.append("Habit strength for north, E_north / E_east: ");
  const slider = document.createElement("input");
  slider.type = "range";
  slider.min = String(RATIO_MIN);
  slider.max = String(RATIO_MAX);
  slider.step = "0.01";
  slider.value = "1.5";
  slider.setAttribute("aria-label", "Ratio of north's habit to east's");
  const out = document.createElement("output");
  sliderRow.append(slider, " ", out);
  const ctx = ctxOf(el, W, H);

  const xOfRatio = (r) => AXIS_X0 + ((r - RATIO_MIN) / (RATIO_MAX - RATIO_MIN)) * (AXIS_X1 - AXIS_X0);

  function draw() {
    const ratio = Number(slider.value);
    out.textContent = ratio.toFixed(2);
    const scores = mode() === "flat" ? G.map(() => tiedG) : G;
    const E = [1, ratio, 1].map((e) => e / (2 + ratio));
    const logits = scores.map((g, i) => Math.log(E[i]) - GAMMA * g);
    const largest = Math.max(...logits);
    const weights = logits.map((l) => Math.exp(l - largest));
    const total = weights.reduce((a, b) => a + b, 0);
    const probs = weights.map((w) => w / total);
    const best = probs.indexOf(Math.max(...probs));
    // i = north beats j = east when ln(E_i/E_j) > γ(G_i − G_j).
    const threshold = Math.exp(GAMMA * (scores[1] - scores[0]));

    ctx.clearRect(0, 0, W, H);
    label(ctx, "E_north / E_east", 20, AXIS_Y + 20, COLOUR.dim, 13);
    ctx.strokeStyle = COLOUR.rule;
    ctx.beginPath();
    ctx.moveTo(AXIS_X0, AXIS_Y);
    ctx.lineTo(AXIS_X1, AXIS_Y);
    ctx.stroke();
    for (let r = RATIO_MIN; r <= RATIO_MAX; r++) label(ctx, String(r), xOfRatio(r), AXIS_Y + 20, COLOUR.dim, 12, "center");
    ctx.fillStyle = COLOUR.ink;
    ctx.fillRect(xOfRatio(threshold) - 1, AXIS_Y - 10, 2, 20);
    label(ctx, `north overtakes east: ${threshold.toFixed(2)}`, xOfRatio(threshold), AXIS_Y - 14, COLOUR.ink, 12, "center");
    ctx.fillStyle = HABIT;
    ctx.beginPath();
    ctx.arc(xOfRatio(ratio), AXIS_Y, 6, 0, 2 * Math.PI);
    ctx.fill();

    label(ctx, "habit ln(E_i / E_east)", HABIT_ZERO - 60, 88, HABIT, 13);
    label(ctx, "score −γG", SCORE_ZERO - 110, 88, SCORE, 13);
    label(ctx, "q(π)", Q_X, 88, COLOUR.ink, 13);
    names.forEach((name, i) => {
      const y = ROW_Y[i];
      label(ctx, name, 20, y + 16, COLOUR.ink, 14);
      const habit = Math.log(E[i] / E[0]);
      bar(ctx, HABIT_ZERO, y, HABIT_PX * habit, BAR_H, HABIT);
      label(ctx, (habit >= 0 ? "+" : "") + habit.toFixed(2), HABIT_ZERO + HABIT_PX * habit + 6, y + 16, COLOUR.dim, 13);
      bar(ctx, SCORE_ZERO, y, -SCORE_PX * GAMMA * scores[i], BAR_H, SCORE);
      label(ctx, (-GAMMA * scores[i]).toFixed(2), SCORE_ZERO + 6, y + 16, COLOUR.dim, 13);
      bar(ctx, Q_X, y, Q_MAX * probs[i], BAR_H, COLOUR.agent);
      label(ctx, probs[i].toFixed(3), Q_X + Q_MAX * probs[i] + 6, y + 16, COLOUR.dim, 13);
      if (i === best) label(ctx, "◀ most probable", Q_X + Q_MAX + 40, y + 16, COLOUR.ink, 13);
    });
    label(ctx, mode() === "flat"
      ? "The scores are equal, so any habit decides."
      : `North needs ln(E_north/E_east) > γ(G_north − G_east) = ${(GAMMA * (scores[1] - scores[0])).toFixed(3)}.`,
      20, 232, COLOUR.dim, 14);
  }
  slider.addEventListener("input", draw);
  draw();
});

// --- Widget 2b -----------------------------------------------------------------
// "Treat γ as a quantity the agent holds beliefs about, distributed according to
// a Gamma distribution with shape 1. Write β > 0 for its rate; the expected
// precision is then γ = 1/β."
mount("precision-belief", (el) => {
  const W = 720, H = 360;
  const PLOT = { left: 70, right: 680, top: 100, bottom: 280, gammaMax: 4, densityMax: 4 };
  const BETA0 = DATA.precision.beta0;
  const PRIOR = "#888", DASH = [6, 5];

  const sliderRow = row(el);
  sliderRow.append("Rate β (doubt about the scores): ");
  const slider = document.createElement("input");
  slider.type = "range";
  slider.min = "0.25";
  slider.max = "4";
  slider.step = "0.05";
  slider.value = String(BETA0);
  slider.setAttribute("aria-label", "Rate β of the belief about γ");
  const out = document.createElement("output");
  sliderRow.append(slider, " ", out);
  const ctx = ctxOf(el, W, H);
  const xOf = (gamma) => PLOT.left + (gamma / PLOT.gammaMax) * (PLOT.right - PLOT.left);
  const yOf = (density) => PLOT.bottom - (Math.min(density, PLOT.densityMax) / PLOT.densityMax) * (PLOT.bottom - PLOT.top);

  // Gamma with shape 1: density β·exp(−βγ), mean 1/β.
  function curve(beta, colour, width, dash = []) {
    ctx.strokeStyle = colour;
    ctx.lineWidth = width;
    ctx.setLineDash(dash);
    ctx.beginPath();
    for (let px = 0; px <= PLOT.right - PLOT.left; px++) {
      const gamma = (px / (PLOT.right - PLOT.left)) * PLOT.gammaMax;
      const y = yOf(beta * Math.exp(-beta * gamma));
      px ? ctx.lineTo(PLOT.left + px, y) : ctx.moveTo(PLOT.left, y);
    }
    ctx.stroke();
    ctx.lineWidth = 1;
    ctx.setLineDash([]);
  }

  function legend(x, colour, width, dash, text) {
    ctx.strokeStyle = colour;
    ctx.lineWidth = width;
    ctx.setLineDash(dash);
    ctx.beginPath();
    ctx.moveTo(x, 17);
    ctx.lineTo(x + 30, 17);
    ctx.stroke();
    ctx.lineWidth = 1;
    ctx.setLineDash([]);
    label(ctx, text, x + 38, 22, COLOUR.ink, 15);
  }

  function draw() {
    const beta = Number(slider.value);
    out.textContent = beta.toFixed(2);
    const mean = 1 / beta;

    ctx.clearRect(0, 0, W, H);
    // Text sits outside the plot area so the curves never run through it.
    legend(PLOT.left, SCORE, 3, [], "belief for the selected rate β");
    legend(PLOT.left + 280, PRIOR, 2, DASH, `belief before any observation (β₀ = ${BETA0})`);
    label(ctx, "↑ how strongly the agent believes each value of γ", PLOT.left, 52, COLOUR.dim, 15);
    ctx.strokeStyle = COLOUR.rule;
    ctx.beginPath();
    ctx.moveTo(PLOT.left, PLOT.top);
    ctx.lineTo(PLOT.left, PLOT.bottom);
    ctx.lineTo(PLOT.right, PLOT.bottom);
    ctx.stroke();
    for (let g = 0; g <= PLOT.gammaMax; g++) label(ctx, String(g), xOf(g), PLOT.bottom + 20, COLOUR.dim, 14, "center");
    label(ctx, "precision γ →", (PLOT.left + PLOT.right) / 2, PLOT.bottom + 44, COLOUR.dim, 15, "center");

    curve(beta, SCORE, 3);
    curve(BETA0, PRIOR, 2, DASH);   // on top, dashed, so it stays visible when β = β₀

    const lineX = xOf(Math.min(mean, PLOT.gammaMax));
    ctx.fillStyle = COLOUR.ink;
    ctx.fillRect(lineX - 1, PLOT.top - 8, 2, PLOT.bottom - PLOT.top + 8);
    const text = `average used by the agent: γ = 1/β = ${mean.toFixed(2)}` + (mean > PLOT.gammaMax ? " (off the axis)" : "");
    const right = lineX > (PLOT.left + PLOT.right) / 2;
    label(ctx, text, right ? lineX - 8 : lineX + 8, PLOT.top - 14, COLOUR.ink, 15, right ? "right" : "left");

    label(ctx, beta === BETA0
      ? "The two curves coincide. Move β to compare."
      : beta > BETA0 ? "Larger β, lower average precision." : "Smaller β, higher average precision.",
      PLOT.left, H - 8, COLOUR.dim, 15);
  }
  slider.addEventListener("input", draw);
  draw();
});

// --- Widget 3 ------------------------------------------------------------------
// "With shallow replanning, the line is exactly flat: the observation cannot
// distinguish policies that did not exist when it arrived. With trial-spanning
// policies, informative evidence moves precision up when it supports the
// ranking by G and down when it opposes it."
mount("precision-update", (el) => {
  const W = 720, H = 360;
  const COLS = { policy: 20, G: 170, pi0Initial: 240, F: 370, pi0: 480, pi: 590 };
  const ROW_Y = [62, 92];
  const PLOT = { x: [140, 380, 620], top: 230, bottom: 330, low: 0.6, high: 1.4 };
  const P = DATA.precision;
  const readings = [];

  const controls = row(el);
  const policyMode = radios(controls, "precision-policies",
    [["span", "policies span the trial"], ["shallow", "policies re-created after each step"]], draw);
  const stepRows = [1, 2].map((step) => {
    const r = row(el);
    r.append(`Reading after step ${step}: `);
    return OBS.map((obs, o) => button(r, obs.name, () => {
      readings.length = step - 1;
      readings.push(o);
      draw();
    }));
  });
  button(row(el), "Start again", () => {
    readings.length = 0;
    draw();
  });
  const ctx = ctxOf(el, W, H);
  const yOf = (gamma) => PLOT.bottom - ((gamma - PLOT.low) / (PLOT.high - PLOT.low)) * (PLOT.bottom - PLOT.top);

  function gammaAfter(observations, shallow) {
    if (observations.length === 0) return { gamma: 1 / P.beta0 };
    let F = accumulatedFreeEnergy(P.D, P.B, P.A, observations, P.policies);
    if (shallow) F = F.map(() => (F[0] + F[1]) / 2);   // new policies share one past
    const { beta, pi0, pi } = updatePrecision(P.beta0, P.G, F);
    return { gamma: 1 / beta, F, pi0, pi, beta };
  }

  function draw() {
    const shallow = policyMode() === "shallow";
    stepRows[1].forEach((b) => (b.disabled = readings.length < 1));
    const now = gammaAfter(readings, shallow);
    const initial = softmax(P.G, 1 / P.beta0);

    ctx.clearRect(0, 0, W, H);
    [["policy", COLS.policy], ["G", COLS.G], ["initial q₀ at γ = 1", COLS.pi0Initial],
     ["F so far", COLS.F], ["q₀ (converged)", COLS.pi0], ["q (converged)", COLS.pi]]
      .forEach(([text, x]) => label(ctx, text, x, 36, COLOUR.dim, 13));
    P.names.forEach((name, k) => {
      const y = ROW_Y[k];
      label(ctx, `π${k + 1}  ${name}`, COLS.policy, y, COLOUR.ink, 14);
      label(ctx, P.G[k].toFixed(3), COLS.G, y, SCORE, 14);
      label(ctx, initial[k].toFixed(3), COLS.pi0Initial, y, COLOUR.ink, 14);
      const shared = shallow && readings.length > 0 ? " (shared)" : "";
      label(ctx, readings.length ? now.F[k].toFixed(3) + shared : "–", COLS.F, y, COLOUR.ink, 14);
      label(ctx, readings.length ? now.pi0[k].toFixed(3) : "–", COLS.pi0, y, COLOUR.ink, 14);
      label(ctx, readings.length ? now.pi[k].toFixed(3) : "–", COLS.pi, y, COLOUR.ink, 14);
    });

    label(ctx, "Readings:", COLS.policy, 132, COLOUR.dim, 14);
    readings.forEach((o, t) => {
      const x = 100 + 80 * t;
      ctx.fillStyle = OBS[o].colour;
      ctx.fillRect(x, 116, 70, 24);
      label(ctx, OBS[o].name, x + 35, 133, OBS[o].text, 13, "center");
    });
    if (!readings.length) label(ctx, "none yet: pick the reading after step 1", 100, 132, COLOUR.dim, 14);

    if (readings.length) {
      const disagreement = now.pi.reduce((acc, p, k) => acc + (p - now.pi0[k]) * P.G[k], 0);
      label(ctx, `β = β₀ + (q − q₀)·G = ${P.beta0} ${disagreement >= 0 ? "+" : "−"} ${Math.abs(disagreement).toFixed(3)}` +
        ` = ${now.beta.toFixed(3)},   γ = 1/β = ${now.gamma.toFixed(3)}`, COLS.policy, 172, COLOUR.ink, 15);
    }
    label(ctx, shallow
      ? "Re-created policies share one past F, so q = q₀ and γ cannot move."
      : "F accumulates along each policy; β restarts from β₀ at every update.",
      COLS.policy, 198, COLOUR.dim, 14);

    // γ after each reading, with the prior as a line.
    ctx.strokeStyle = COLOUR.rule;
    ctx.beginPath();
    ctx.moveTo(PLOT.x[0] - 40, yOf(1));
    ctx.lineTo(PLOT.x[2] + 40, yOf(1));
    ctx.stroke();
    label(ctx, "prior γ = 1", PLOT.x[0] - 120, yOf(1) + 5, COLOUR.dim, 12);
    const points = [0, 1, 2].filter((t) => t <= readings.length)
      .map((t) => gammaAfter(readings.slice(0, t), shallow).gamma);
    ctx.strokeStyle = COLOUR.ink;
    ctx.beginPath();
    points.forEach((g, t) => (t ? ctx.lineTo(PLOT.x[t], yOf(g)) : ctx.moveTo(PLOT.x[t], yOf(g))));
    ctx.stroke();
    points.forEach((g, t) => {
      ctx.fillStyle = COLOUR.ink;
      ctx.beginPath();
      ctx.arc(PLOT.x[t], yOf(g), 5, 0, 2 * Math.PI);
      ctx.fill();
      label(ctx, g.toFixed(3), PLOT.x[t], yOf(g) - 10, COLOUR.ink, 13, "center");
    });
    ["start", "after step 1", "after step 2"].forEach((text, t) =>
      label(ctx, text, PLOT.x[t], PLOT.bottom + 22, COLOUR.dim, 12, "center"));
  }
  draw();
});

// --- Widget 4 ------------------------------------------------------------------
// "A correct D helps most at the start ... A learned D can do worse than a flat
// one ... If the agent's habits change, the learned D still points to the old
// starting cells."
mount("starting-prior", (el) => {
  const W = 720, H = 310;
  const AREA = { left: 70, right: 690, top: 30, bottom: 250 };
  const SERIES = [["given", "given D (module 4's three cells)", "#39758d"],
                  ["flat", "flat D", "#888"],
                  ["learned", "learned D (60 nights of counts)", HABIT]];
  const S = DATA.startingPrior;
  let view = "holds";
  let hover = null;

  const controls = row(el);
  button(controls, "Habit holds", () => { view = "holds"; hover = null; draw(); });
  button(controls, "Habit moved", () => { view = "moved"; hover = null; draw(); });
  const ctx = ctxOf(el, W, H);
  const canvas = ctx.canvas;

  function scales() {
    return view === "holds"
      ? { n: S.steps, yMax: 0.45, xLabel: "step within the night", yLabel: "belief on the true cell" }
      : { n: S.nights, yMax: 0.25, xLabel: "nights since the sleeping cells moved", yLabel: "belief on the true cell at step 1" };
  }
  const xOf = (i, n) => AREA.left + ((i - 1) / (n - 1)) * (AREA.right - AREA.left);
  const yOf = (v, yMax) => AREA.bottom - (v / yMax) * (AREA.bottom - AREA.top);

  function points() {
    const { n, yMax } = scales();
    if (view === "holds") {
      return SERIES.flatMap(([key, name]) => S[key].map((v, i) =>
        ({ x: xOf(i + 1, n), y: yOf(v, yMax), text: `${name}, step ${i + 1}: ${v.toFixed(3)} ± ${S[key + "Se"][i].toFixed(3)}` })));
    }
    return S.moved.learned.map((v, i) =>
      ({ x: xOf(i + 1, n), y: yOf(v, yMax), text: `learned D, night ${i + 1}: ${v.toFixed(3)}` }));
  }

  function draw() {
    const { n, yMax, xLabel, yLabel } = scales();
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = COLOUR.rule;
    ctx.beginPath();
    ctx.moveTo(AREA.left, AREA.top);
    ctx.lineTo(AREA.left, AREA.bottom);
    ctx.lineTo(AREA.right, AREA.bottom);
    ctx.stroke();
    for (let v = 0; v <= yMax + 1e-9; v += 0.1) label(ctx, v.toFixed(1), AREA.left - 8, yOf(v, yMax) + 4, COLOUR.dim, 12, "right");
    label(ctx, yLabel, AREA.left, AREA.top - 10, COLOUR.dim, 13);
    label(ctx, xLabel, (AREA.left + AREA.right) / 2, AREA.bottom + 36, COLOUR.dim, 13, "center");

    if (view === "holds") {
      for (let i = 1; i <= n; i++) label(ctx, String(i), xOf(i, n), AREA.bottom + 18, COLOUR.dim, 12, "center");
      SERIES.forEach(([key, name, colour], s) => {
        ctx.strokeStyle = colour;
        ctx.fillStyle = colour;
        ctx.beginPath();
        S[key].forEach((v, i) => (i ? ctx.lineTo(xOf(i + 1, n), yOf(v, yMax)) : ctx.moveTo(xOf(i + 1, n), yOf(v, yMax))));
        ctx.stroke();
        S[key].forEach((v, i) => {
          const x = xOf(i + 1, n), se = S[key + "Se"][i];
          ctx.beginPath();
          ctx.moveTo(x, yOf(v - se, yMax));
          ctx.lineTo(x, yOf(v + se, yMax));
          ctx.stroke();
          ctx.beginPath();
          ctx.arc(x, yOf(v, yMax), 3.5, 0, 2 * Math.PI);
          ctx.fill();
        });
        label(ctx, name, AREA.right - 250, AREA.bottom - 58 + 18 * s, colour, 13);
      });
      label(ctx, `step 1: flat ${S.flat[0].toFixed(3)}, given ${S.given[0].toFixed(3)}, learned ${S.learned[0].toFixed(3)}`,
        AREA.left, H - 8, COLOUR.ink, 13);
    } else {
      for (let i = 10; i <= n; i += 10) label(ctx, String(i), xOf(i, n), AREA.bottom + 18, COLOUR.dim, 12, "center");
      ctx.fillStyle = "rgba(136,136,136,0.2)";
      ctx.fillRect(AREA.left, yOf(S.moved.flat + S.moved.flatSe, yMax), AREA.right - AREA.left,
        yOf(S.moved.flat - S.moved.flatSe, yMax) - yOf(S.moved.flat + S.moved.flatSe, yMax));
      ctx.strokeStyle = "#888";
      ctx.beginPath();
      ctx.moveTo(AREA.left, yOf(S.moved.flat, yMax));
      ctx.lineTo(AREA.right, yOf(S.moved.flat, yMax));
      ctx.stroke();
      label(ctx, `flat D ${S.moved.flat.toFixed(3)} (± s.e.)`, AREA.right - 180, yOf(S.moved.flat, yMax) - 8, "#666", 13);
      ctx.fillStyle = HABIT;
      S.moved.learned.forEach((v, i) => {
        ctx.beginPath();
        ctx.arc(xOf(i + 1, n), yOf(v, yMax), 3, 0, 2 * Math.PI);
        ctx.fill();
      });
      const first = S.moved.learned.slice(0, 10).reduce((a, b) => a + b, 0) / 10;
      const last = S.moved.learned.slice(-10).reduce((a, b) => a + b, 0) / 10;
      label(ctx, `learned D (keeps counting): first 10 nights ${first.toFixed(3)}, last 10 nights ${last.toFixed(3)}`,
        AREA.left, H - 8, HABIT, 13);
    }
    if (hover) label(ctx, hover, AREA.right, AREA.top - 10, COLOUR.ink, 13, "right");
  }

  canvas.addEventListener("mousemove", (event) => {
    const box = canvas.getBoundingClientRect();
    const x = ((event.clientX - box.left) * canvas.width) / box.width;
    const y = ((event.clientY - box.top) * canvas.height) / box.height;
    let nearest = null, distance = 12;
    for (const p of points()) {
      const d = Math.hypot(p.x - x, p.y - y);
      if (d < distance) { nearest = p; distance = d; }
    }
    const text = nearest ? nearest.text : null;
    if (text !== hover) { hover = text; draw(); }
  });
  draw();
});

mountAll();

/**
 * The belief math, in JavaScript.
 *
 * A reimplementation of aif/beliefs.py and aif/generative_model.py for the
 * browser widgets. The Python stays the tested reference; aif/beliefs.py
 * writes aif/belief_cases.json and demo() below asserts this file reproduces
 * every posterior in it.
 *
 * A belief is a plain array of length n_states that sums to one: never a
 * single position. A and B are arrays of arrays, not typed arrays: the grid
 * is 25 states and clarity wins.
 */

export function uniformBelief(n) {
  return new Array(n).fill(1 / n);
}

export function update(prior, A, obs) {
  const likelihood = A[obs];
  const unnormalised = likelihood.map((l, s) => l * prior[s]);
  const evidence = unnormalised.reduce((a, b) => a + b, 0);
  if (evidence === 0) {
    throw new Error(
      `observation ${obs} is impossible under this belief: ` +
        "every state that could produce it has probability zero",
    );
  }
  const posterior = unnormalised.map((x) => x / evidence);
  return { posterior, likelihood, unnormalised };
}

export function predict(belief, B, action) {
  const n = belief.length;
  const predicted = new Array(n).fill(0);
  for (let nextState = 0; nextState < n; nextState++) {
    for (let state = 0; state < n; state++) {
      predicted[nextState] += B[nextState][state][action] * belief[state];
    }
  }
  return predicted;
}

export function surprise(prior, A, obs) {
  const p = A[obs].reduce((acc, a, s) => acc + a * prior[s], 0);
  if (p === 0) return Infinity;
  return -Math.log(p);
}

// Action order matches worlds/gridworld.py ACTIONS.
const MOVES = [
  [-1, 0], // up
  [1, 0], // down
  [0, -1], // left
  [0, 1], // right
  [0, 0], // stay
];

// The grid wraps: stepping off one side arrives on the other. No walls, so no
// cell is special. A wall would stop the agent, and being stopped is
// information: walk west, fail to move, and you have learned you are against
// the west edge without observing anything. The lessons are about what looking
// buys, so movement must not leak position. Wrapping also makes B doubly
// stochastic, which is what licenses lesson 2's claim that a prediction step
// can only ever widen a belief. Mirrors GridWorld.move in worlds/gridworld.py.
function moveFrom(row, col, action, rows, cols) {
  const [dr, dc] = MOVES[action];
  return [(((row + dr) % rows) + rows) % rows, (((col + dc) % cols) + cols) % cols];
}

export function observationModel({ rows, cols, sensorNoise }) {
  const n = rows * cols;
  const A = Array.from({ length: n + 1 }, () => new Array(n).fill(0));
  for (let s = 0; s < n; s++) {
    const row = Math.floor(s / cols);
    const col = s % cols;
    const neighbours = [];
    for (let a = 0; a < 5; a++) {
      const [r, c] = moveFrom(row, col, a, rows, cols);
      if (r !== row || c !== col) neighbours.push(r * cols + c);
    }
    if (sensorNoise > 0 && neighbours.length > 0) {
      A[s][s] = 1 - sensorNoise;
      for (const nb of neighbours) A[nb][s] += sensorNoise / neighbours.length;
    } else {
      A[s][s] = 1;
    }
  }
  return A;
}

// Lamp world: every tile is lit or dark and the agent senses only which.
// Two observations, DARK=0 and LIT=1, so A[o][s] is 2 by n_states. `lit` is a
// boolean per tile. Each column is [0.2, 0.8] on a lit tile and [0.9, 0.1] on a
// dark one: the false readings are what make a wrong answer possible at all.
export function lampObservationModel(lit, pLit = 0.8, pDark = 0.9) {
  return [
    lit.map((l) => (l ? 1 - pLit : pDark)),
    lit.map((l) => (l ? pLit : 1 - pDark)),
  ];
}

// `reliability` is how much of the agent's belief lands on the cell it aimed
// at. At 1 every column holds a single one and B is a permutation: the belief
// moves across the grid without changing shape. Below 1, the remainder is
// shared equally between the cell the agent started in and the four neighbours
// of the intended cell, which is what makes moving cost certainty.
//
// Destinations are accumulated rather than assigned, because they collide:
// under `stay` the intended cell IS the origin, so one cell takes both the
// reliability and a share of the spread, and it has to keep both. Columns sum
// to one either way.
export function transitionModel({ rows, cols, reliability = 1 }) {
  const n = rows * cols;
  const nActions = 5;
  const B = Array.from({ length: n }, () =>
    Array.from({ length: n }, () => new Array(nActions).fill(0)),
  );
  for (let s = 0; s < n; s++) {
    const row = Math.floor(s / cols);
    const col = s % cols;
    for (let a = 0; a < nActions; a++) {
      const [r, c] = moveFrom(row, col, a, rows, cols);
      B[r * cols + c][s][a] += reliability;
      if (reliability >= 1) continue;
      const spillTo = [s]; // the cell it started in, then the target's neighbours
      for (let na = 0; na < nActions; na++) {
        if (MOVES[na][0] === 0 && MOVES[na][1] === 0) continue;
        const [nr, nc] = moveFrom(r, c, na, rows, cols);
        spillTo.push(nr * cols + nc);
      }
      const share = (1 - reliability) / spillTo.length;
      for (const d of spillTo) B[d][s][a] += share;
    }
  }
  return B;
}

export function mulberry32(seed) {
  let t = seed >>> 0;
  return function () {
    t += 0x6d2b79f5;
    let x = Math.imul(t ^ (t >>> 15), 1 | t);
    x ^= x + Math.imul(x ^ (x >>> 7), 61 | x);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

function approx(a, b, tol = 1e-9) {
  return Math.abs(a - b) <= tol;
}

let failures = 0;
function check(condition, message) {
  if (!condition) {
    failures += 1;
    console.error(`FAIL: ${message}`);
  }
}

async function demo() {
  const { readFileSync } = await import("node:fs");
  const { fileURLToPath } = await import("node:url");

  const u = uniformBelief(4);
  check(u.length === 4 && u.every((x) => approx(x, 0.25)), "uniformBelief");

  // update: perfect 2-state sensor, prior 50/50, see observation 0 -> [0.9, 0.1]
  const A2 = [
    [0.9, 0.1],
    [0.1, 0.9],
  ];
  const r = update([0.5, 0.5], A2, 0);
  check(approx(r.likelihood[0], 0.9) && approx(r.likelihood[1], 0.1), "update likelihood");
  check(approx(r.unnormalised[0], 0.45) && approx(r.unnormalised[1], 0.05), "update unnormalised");
  check(approx(r.posterior[0], 0.9) && approx(r.posterior[1], 0.1), "update posterior");
  check(approx(r.posterior[0] + r.posterior[1], 1.0), "update sums to one");

  // update: uninformative observation leaves the belief alone
  const flat2 = [
    [0.5, 0.5],
    [0.5, 0.5],
  ];
  const unchanged = update([0.5, 0.5], flat2, 0).posterior;
  check(approx(unchanged[0], 0.5) && approx(unchanged[1], 0.5), "update uninformative");

  // update: impossible observation throws
  let threw = false;
  try {
    update([1, 0], [[0, 1], [1, 0]], 0);
  } catch (e) {
    threw = true;
  }
  check(threw, "update throws on zero evidence");

  // surprise: zero when certain and the expected thing happened
  const certain2 = [
    [1, 0],
    [0, 1],
  ];
  check(approx(surprise([1, 0], certain2, 0), 0.0), "surprise zero when certain");
  check(approx(surprise([0.5, 0.5], certain2, 0), Math.log(2)), "surprise ln 2 on a coin flip");

  // predict: mass moves, never created; state 0 -> state 1, state 1 -> state 1
  const B2 = [
    [[0], [0]],
    [[1], [1]],
  ];
  const p = predict([1, 0], B2, 0);
  check(approx(p[0], 0.0) && approx(p[1], 1.0), "predict moves mass");
  check(approx(p[0] + p[1], 1.0), "predict conserves mass");

  // observationModel: perfect sensor on 5x5 is the identity on the first 25 rows
  const Aid = observationModel({ rows: 5, cols: 5, sensorNoise: 0 });
  check(Aid.length === 26 && Aid[0].length === 25, "observationModel shape");
  for (let s = 0; s < 25; s++) {
    check(approx(Aid[s][s], 1.0), `observationModel identity col ${s}`);
  }
  const An = observationModel({ rows: 5, cols: 5, sensorNoise: 0.3 });
  let col7 = 0;
  for (let o = 0; o < 26; o++) col7 += An[o][7];
  check(approx(col7, 1.0), "observationModel noisy column sums to one");
  check(approx(An[7][7], 0.7), "observationModel keeps 1 - noise on the true tile");

  // transitionModel: right from tile 0 lands on tile 1; up from tile 0 stays
  const B = transitionModel({ rows: 5, cols: 5 });
  check(B.length === 25 && B[0].length === 25 && B[0][0].length === 5, "transitionModel shape");
  const RIGHT = 3;
  const UP = 0;
  check(approx(B[1][0][RIGHT], 1.0), "transitionModel right from 0");
  check(approx(B[20][0][UP], 1.0), "transitionModel up from 0 wraps to the bottom row");
  // Default reliability is still the permutation: exactly one 1 per column.
  for (let a = 0; a < 5; a++) {
    for (let s = 0; s < 25; s++) {
      const ones = B.filter((rowS) => approx(rowS[s][a], 1.0)).length;
      check(ones === 1, `transitionModel permutation column ${s} action ${a}`);
    }
  }

  // Spread B: mass moves off the intended cell, but every column still sums to
  // one, including at the walls where the intended cell and the origin coincide.
  const Bs = transitionModel({ rows: 5, cols: 5, reliability: 0.7 });
  for (let a = 0; a < 5; a++) {
    for (let s = 0; s < 25; s++) {
      let colSum = 0;
      for (let d = 0; d < 25; d++) colSum += Bs[d][s][a];
      check(approx(colSum, 1.0), `spread B column ${s} action ${a} sums to one`);
    }
  }
  // Tile 7 is interior: right lands on 8, and 8 keeps only the reliability.
  check(approx(Bs[8][7][RIGHT], 0.7), "spread B keeps reliability on the intended cell");
  check(Bs[7][7][RIGHT] > 0, "spread B leaves some mass on the origin");
  // The grid wraps, so no move is ever blocked and the origin keeps only its
  // one share of the spill, everywhere, including what used to be corners.
  check(approx(Bs[20][0][UP], 0.7), "spread B wraps at the top row");
  check(Bs[0][0][UP] > 0 && Bs[0][0][UP] < 0.7, "spread B leaves the origin one share, even at an edge");

  // B is doubly stochastic: rows sum to one as well as columns, because the
  // wrap means every cell receives exactly as much as it sends. That is the
  // property behind lesson 2's "predicting can only ever widen a belief".
  for (let a = 0; a < 5; a++) {
    for (let d = 0; d < 25; d++) {
      let rowSum = 0;
      for (let st = 0; st < 25; st++) rowSum += Bs[d][st][a];
      check(approx(rowSum, 1.0), `spread B row ${d} action ${a} sums to one`);
    }
  }

  // And the consequence, checked directly: prediction never lowers entropy.
  const nEnt2 = (b) => {
    let h = 0;
    for (const q of b) if (q > 0) h -= q * Math.log(q);
    return h;
  };
  let ent = new Array(25).fill(0);
  ent[12] = 1;
  let prevH = nEnt2(ent);
  for (const a of [3, 1, 2, 0, 3, 3, 4, 1, 3, 3]) {
    ent = predict(ent, Bs, a);
    const h = nEnt2(ent);
    check(h >= prevH - 1e-12, `prediction never narrows a belief (action ${a})`);
    prevH = h;
  }

  // lampObservationModel: columns are distributions, and a lit reading lifts
  // every lit tile by the same amount and drops every dark one.
  const litMask = [true, false, false, true, false];
  const Alamp = lampObservationModel(litMask);
  check(Alamp.length === 2 && Alamp[0].length === 5, "lampObservationModel shape");
  for (let s = 0; s < 5; s++) {
    check(approx(Alamp[0][s] + Alamp[1][s], 1.0), `lampObservationModel column ${s} sums to one`);
  }
  check(approx(Alamp[1][0], 0.8) && approx(Alamp[1][1], 0.1), "lampObservationModel lit row");
  check(approx(Alamp[0][0], 0.2) && approx(Alamp[0][1], 0.9), "lampObservationModel dark row");
  // Rows are not distributions: this is the point the lesson turns on.
  check(!approx(Alamp[1].reduce((a, b) => a + b, 0), 1.0), "lampObservationModel rows are not distributions");

  // The lesson's world: 25 tiles, 3 lit, uniform prior, one lit reading.
  const lit25 = Array.from({ length: 25 }, (_, s) => s === 0 || s === 9 || s === 18);
  const A25 = lampObservationModel(lit25);
  const post1 = update(uniformBelief(25), A25, 1).posterior;
  check(approx(post1[0], 0.032 / 0.184, 1e-9), "lamp posterior on a lit tile");
  check(approx(post1[1], 0.004 / 0.184, 1e-9), "lamp posterior on a dark tile");
  check(approx(post1.reduce((a, b) => a + b, 0), 1.0), "lamp posterior sums to one");
  // The whole sequence the lesson prints, folded from the uniform prior.
  const expected = [0.1739, 0.2991, 0.2199, 0.3131, 0.2584, 0.3217, 0.3318];
  let lb = uniformBelief(25);
  [1, 1, 0, 1, 0, 1, 1].forEach((o, i) => {
    lb = update(lb, A25, o).posterior;
    check(approx(lb[0], expected[i], 5e-5), `lamp sequence step ${i + 1}: expected ${expected[i]} got ${lb[0].toFixed(4)}`);
  });

  // cross-check against the Python-written fixture
  const path = fileURLToPath(new URL("../aif/belief_cases.json", import.meta.url));
  const fixture = JSON.parse(readFileSync(path, "utf8"));
  for (const [i, c] of fixture.updates.entries()) {
    const got = update(c.prior, c.A, c.obs).posterior;
    check(
      got.length === c.posterior.length && got.every((x, j) => approx(x, c.posterior[j], 1e-7)),
      `fixture case ${i}: expected ${JSON.stringify(c.posterior)} got ${JSON.stringify(got)}`,
    );
  }

  // The transition half of the fixture. The JS `reliability` is the Python's
  // 1 - slip: both name the mass staying on the intended cell. This compares
  // every entry, so a spread that drifts to different-but-plausible cells
  // fails here rather than silently disagreeing with the reference.
  const t = fixture.transitions;
  const Bfix = transitionModel({ rows: t.rows, cols: t.cols, reliability: 1 - t.slip });
  let worstB = 0;
  for (let d = 0; d < t.B.length; d++) {
    for (let st = 0; st < t.B[d].length; st++) {
      for (let a = 0; a < t.B[d][st].length; a++) {
        worstB = Math.max(worstB, Math.abs(t.B[d][st][a] - Bfix[d][st][a]));
      }
    }
  }
  check(worstB < 1e-9, `fixture B differs from the Python by ${worstB}`);

  const startFix = new Array(t.rows * t.cols).fill(0);
  startFix[t.predict_from] = 1;
  const gotPredicted = predict(startFix, Bfix, t.predict_action);
  check(
    gotPredicted.every((x, j) => approx(x, t.predicted[j], 1e-9)),
    "fixture predict differs from the Python",
  );

  // Prediction conserves probability: this is the lesson-2 claim that B @
  // belief needs no normalisation, and it has to hold after many steps, not
  // just one, since that is how the widget uses it.
  let acc = uniformBelief(25);
  for (let i = 0; i < 8; i++) {
    acc = predict(acc, Bfix, i % 5);
    check(approx(acc.reduce((x, y) => x + y, 0), 1.0, 1e-12), `predict conserves mass at step ${i + 1}`);
  }

  // Normalised entropy, the quantity the lesson-2 spread bar draws.
  const nEnt = (b) => {
    let h = 0;
    for (const q of b) if (q > 0) h -= q * Math.log(q);
    return h / Math.log(b.length);
  };
  const spike = new Array(25).fill(0);
  spike[10] = 1;
  check(approx(nEnt(spike), 0, 1e-12), "entropy is 0 on a one-hot belief");
  check(approx(nEnt(uniformBelief(25)), 1, 1e-12), "entropy is 1 on a uniform belief");

  if (failures > 0) {
    console.error(`aif.js: ${failures} check(s) failed`);
    process.exit(1);
  }
  console.log("aif.js ok");
}

// Run demo() only when executed directly with node, never in the browser.
if (
  typeof process !== "undefined" &&
  process.argv[1] &&
  import.meta.url === `file://${process.argv[1]}`
) {
  demo();
}

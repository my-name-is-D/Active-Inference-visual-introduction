/** Verified A-learned, B-fixed comparison; all runs are precomputed. */
import { mount, mountAll } from './widget.js';
import { A_FIXED_B_DATA as DATA } from './module-5-data.js';
import { B_FIXED_A_DATA as B_DATA } from './module-5-b-data.js';
import { JOINT_DATA } from './module-5-joint-data.js';

const SVG = 'http://www.w3.org/2000/svg';
const COLOURS = ['#111', '#ffdb58', '#39758d', '#a7a9ac', '#6e9c55', '#e67e50'];
const TEXT = ['', '', 'H', '', '', 'W'];
const POLICY = {
  uniform: { label: 'Random walk', colour: '#b45309', shade: '#f3dcc5' },
  state: { label: 'AIF without A novelty', colour: '#39758d', shade: '#d1e3e9' },
  aif: { label: 'AIF with A novelty', colour: '#19775e', shade: '#cde4dc' },
};

function node(tag, attrs = {}, text) {
  const element = document.createElementNS(SVG, tag);
  for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, value);
  if (text !== undefined) element.textContent = text;
  return element;
}

mount('outer-product', element => {
  element.classList.add('outer-product');
  const svg = node('svg', { viewBox: '0 0 560 140', role: 'img',
    'aria-label': 'The observation column times the state-belief row gives the matrix added to the counts' });
  const cell = (x, y, width, height, fill) => svg.append(node('rect', {
    x, y, width, height, fill, stroke: '#fff', 'stroke-width': 2, rx: 2
  }));
  const label = (x, y, content, size = 15) => svg.append(node('text', {
    x, y, 'text-anchor': 'middle', 'font-size': size,
    'font-family': 'ui-monospace, monospace', fill: '#30343b'
  }, content));

  label(58, 20, 'o', 18);
  for (let row = 0; row < 4; row++) cell(43, 32 + row * 25, 30, 23, row === 2 ? '#ffdb58' : '#111');

  label(105, 84, '⊗', 24);
  label(230, 45, 'q(s)', 18);
  for (let col = 0; col < 6; col++) cell(139 + col * 31, 58, 29, 30, '#19775e');

  label(345, 84, '=', 24);
  label(452, 20, 'o ⊗ q(s)', 18);
  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 6; col++) {
      cell(365 + col * 30, 32 + row * 25, 28, 23, row === 2 ? '#ffdb58' : '#d7d9d8');
    }
  }
  element.append(svg);
});

// Real joint posterior from verify_lesson5.py: origin belief [.6,.3,.1] on
// three works cells, one action and one dark observation, restricted to
// those same three cells as destinations. Reuses outer-product's grammar:
// belief as cell opacity, not printed numbers, so the two widgets read as
// the same kind of picture. q(s_{t-1}) here is the same strip of shaded
// cells that outer-product calls q(s); the grid on the right is what q(s)
// becomes once it is spread over both an origin and a destination.
mount('joint-transition', element => {
  element.classList.add('joint-transition');
  const origin = [0.60, 0.30, 0.10];   // q(s_{t-1}) on cells (1,1), (1,2), (2,1)
  const joint = [
    [0.036, 0.018, 0.378],
    [0.018, 0.009, 0.189],
    [0.006, 0.003, 0.063],
  ];
  const svg = node('svg', { viewBox: '0 0 560 140', role: 'img',
    'aria-label': 'The belief about where the agent started spreads into a joint belief about where it started and ended' });
  const cell = (x, y, width, height, fill, opacity = 1) => svg.append(node('rect', {
    x, y, width, height, fill, opacity, stroke: '#fff', 'stroke-width': 2, rx: 2,
  }));
  const label = (x, y, content, size = 15) => svg.append(node('text', {
    x, y, 'text-anchor': 'middle', 'font-size': size,
    'font-family': 'ui-monospace, monospace', fill: '#30343b',
  }, content));

  // Renders a run of tokens as one <text>, each token either plain text or
  // {sub: '...'} for a subscript, so labels read as real notation (s_{t-1},
  // the conditioning bar) instead of LaTeX source dropped in as literal text.
  const mathLabel = (x, y, tokens, size = 16) => {
    const text = node('text', { x, y, 'text-anchor': 'middle', 'font-size': size,
      'font-family': 'ui-monospace, monospace', fill: '#30343b' });
    for (const token of tokens) {
      if (typeof token === 'string') {
        text.append(node('tspan', {}, token));
      } else {
        text.append(node('tspan', { 'font-size': size * 0.72, 'baseline-shift': '-25%' }, token.sub));
      }
    }
    svg.append(text);
  };

  const originX = 60, gridX = 260, gridTop = 40, gridSize = 31;
  const gridWidth = gridSize * 3 - 2, gridCenter = gridX + gridWidth / 2;

  mathLabel(originX + 25, 20, ['q(s', { sub: 't-1' }, ')'], 16);
  origin.forEach((value, row) => cell(originX, gridTop + row * 25, 30, 23, '#19775e', 0.15 + 0.85 * value));
  label(originX + 15, gridTop + 25 * 3 + 18, 'origin belief', 11);

  label(160, gridTop + 40, 'step u,', 13);
  label(160, gridTop + 56, 'observe o', 13);
  label(160, gridTop + 20, '→', 24);

  mathLabel(gridCenter, 20, ['q(s', { sub: 't' }, ', s', { sub: 't-1' }, ' | o, u)'], 16);
  const maximum = Math.max(...joint.flat());
  joint.forEach((row, r) => row.forEach((value, c) => {
    cell(gridX + c * gridSize, gridTop + r * 25, gridSize - 2, 23, '#19775e', 0.1 + 0.9 * (value / maximum));
  }));
  label(gridCenter, gridTop + 25 * 3 + 18, 'destination →', 11);
  // Same layout as "destination →": plain rotated text with the arrow glyph
  // built in. A rotated "↓" turns sideways with the text and stops pointing
  // down, so "←" is used instead: rotated -90°, it reads as pointing down.
  const originLabelX = gridX - 14, originLabelY = gridTop + 25 * 1.5;
  svg.append(node('text', { x: originLabelX, y: originLabelY, 'text-anchor': 'middle',
    'font-size': 11, 'font-family': 'ui-monospace, monospace', fill: '#30343b',
    transform: `rotate(-90 ${originLabelX} ${originLabelY})` }, '← origin'));

  element.append(svg);
});

function curve(series, baselines, step, smoothing) {
  const lastStep = DATA.steps.at(-1);
  const svg = node('svg', { viewBox: '0 0 620 305', role: 'img',
    'aria-label': `Observation-model error for a random walk, AIF without A novelty, and AIF with A novelty over ${lastStep} steps` });
  svg.append(node('title', {}, 'Fraction of initial error remaining over the three learned works columns; lower is better.'));
  const x = value => 55 + 535 * value / lastStep;
  const y = value => 245 - 190 * value;
  [0, .25, .5, .75, 1].forEach(value => {
    svg.append(node('line', { x1: 55, x2: 590, y1: y(value), y2: y(value), stroke: '#deded8' }));
    svg.append(node('text', { x: 47, y: y(value) + 4, 'text-anchor': 'end', 'font-size': 12 }, value.toFixed(2)));
  });
  [0, 500, 1000, 1500, 2000].forEach(value =>
    svg.append(node('text', { x: x(value), y: 267, 'text-anchor': 'middle', 'font-size': 12 }, value)));
  svg.append(node('text', { x: 323, y: 294, 'text-anchor': 'middle', 'font-size': 12 }, 'Steps'));
  svg.append(node('text', { x: 14, y: 150, transform: 'rotate(-90 14 150)', 'text-anchor': 'middle', 'font-size': 12 }, 'Initial A error remaining'));
  const normalise = data => data.mean.map(value => value / data.mean[0]);
  const points = values => values.map((value, index) => `${x(index)},${y(value)}`).join(' ');

  for (const key of ['uniform', 'state', 'aif']) {
    const data = series[key], values = normalise(data);
    const errors = data.se.map(value => value / data.mean[0]);
    const upper = values.map((value, index) => `${x(index)},${y(value + errors[index])}`);
    const lower = values.map((value, index) => `${x(index)},${y(value - errors[index])}`).reverse();
    svg.append(node('polygon', { points: [...upper, ...lower].join(' '), fill: POLICY[key].shade, opacity: .55 }));
    if (smoothing) svg.append(node('polyline', { points: points(normalise(baselines[key])), fill: 'none',
      stroke: POLICY[key].colour, 'stroke-width': 1.5, 'stroke-dasharray': '5 4', opacity: .65 }));
    svg.append(node('polyline', { points: points(values), fill: 'none', stroke: POLICY[key].colour, 'stroke-width': 3 }));
    svg.append(node('circle', { cx: x(step), cy: y(values[step]), r: 4, fill: POLICY[key].colour }));
    const offset = { uniform: -9, state: -9, aif: 16 }[key];
    svg.append(node('text', { x: x(step) + 7, y: y(values[step]) + offset,
      fill: POLICY[key].colour, 'font-size': 12, 'font-weight': 700 }, data.visits[step].toFixed(0)));
    const labelOffset = { uniform: -8, state: -8, aif: 15 }[key];
    svg.append(node('text', { x: 584, y: y(values[lastStep]) + labelOffset,
      fill: POLICY[key].colour, 'text-anchor': 'end', 'font-size': 12 }, POLICY[key].label));
  }
  svg.append(node('line', { x1: x(step), x2: x(step), y1: 32, y2: 245, stroke: '#888', 'stroke-dasharray': '2 3' }));
  svg.append(node('text', { x: x(step), y: 20, 'text-anchor': 'middle', 'font-size': 11, fill: '#555' }, 'Works-cell visits'));
  return svg;
}

function map(title, world, learned = null, snapshot = 0, density = null, densityColour = null) {
  const svg = node('svg', { viewBox: '0 0 230 255', role: 'img', 'aria-label': title });
  svg.append(node('title', {}, title));
  svg.append(node('text', { x: 115, y: 20, 'text-anchor': 'middle', 'font-size': 15, 'font-weight': 650 }, title));
  for (let state = 0; state < 25; state++) {
    const row = Math.floor(state / 5), col = state % 5, truth = world.truth[state];
    let kind = truth, confidence = .8;
    const worksIndex = DATA.works.indexOf(state);
    if (learned && worksIndex >= 0) {
      const column = learned.maps[snapshot].map(observation => observation[worksIndex]);
      kind = column.indexOf(Math.max(...column)); confidence = column[truth];
    }
    const x = 15 + col * 41, y = 34 + row * 41;
    svg.append(node('rect', { x, y, width: 38, height: 38, rx: 2,
      fill: COLOURS[kind], opacity: .35 + .65 * confidence, stroke: '#fff', 'stroke-width': 1 }));
    if (density) {
      const maximum = Math.max(...density, 1);
      const radius = 15 * Math.sqrt(density[state] / maximum);
      if (radius > 0) svg.append(node('circle', { cx: x + 19, cy: y + 19, r: radius,
        fill: densityColour, opacity: .7, stroke: '#fff', 'stroke-width': .7 }));
    }
    if (worksIndex >= 0) svg.append(node('rect', { x: x + 2, y: y + 2, width: 34, height: 34,
      fill: 'none', stroke: '#b54835', 'stroke-width': 2, 'stroke-dasharray': '4 3' }));
    if (TEXT[kind]) svg.append(node('text', { x: x + 19, y: y + 24, 'text-anchor': 'middle',
      fill: '#fff', 'font-size': 15, 'font-weight': 700 }, TEXT[kind]));
    if (learned && worksIndex >= 0) svg.append(node('text', { x: x + 34, y: y + 36,
      'text-anchor': 'end', fill: '#fff', 'font-size': 9 }, `${Math.round(confidence * 100)}%`));
  }
  const wall = (x1, y1, x2, y2) => {
    svg.append(node('line', { x1, y1, x2, y2, stroke: '#fff', 'stroke-width': 9,
      'stroke-linecap': 'square' }));
    svg.append(node('line', { x1, y1, x2, y2, stroke: '#8f251c', 'stroke-width': 5,
      'stroke-linecap': 'square' }));
  };
  wall(16, 74.5, 52, 74.5);   // between (1,1) and (2,1)
  wall(57, 115.5, 93, 115.5); // between (2,2) and (3,2)
  svg.append(node('rect', { x: 12, y: 31, width: 84, height: 84, fill: 'none', stroke: '#b54835', 'stroke-width': 2.5 }));
  return svg;
}

const CONFIDENCE_STREAM = ['lit', 'lit', 'dark', 'lit', 'lit', 'lit', 'dark', 'lit'];
const CONFIDENCE_COLOURS = { dark: '#111', lit: '#ffdb58' };
const CONFIDENCE_START = {
  small: { label: 'Low confidence', counts: [.1, .4],
    nextInformation: [1.149192998, .125269033],
    novelty: [.330053826, .122897482, .074622432, .119568101, .094033702,
      .077324855, .065605978, .060913149, .053875076] },
  large: { label: 'High confidence', counts: [20, 80],
    nextInformation: [.019800052, .001245313],
    novelty: [.004956260, .004907234, .004859168, .004813839, .004767595,
      .004722231, .004677721, .004635620, .004592736] },
};

function confidenceCounts(key, step) {
  const counts = [...CONFIDENCE_START[key].counts];
  for (const observation of CONFIDENCE_STREAM.slice(0, step)) counts[observation === 'dark' ? 0 : 1] += 1;
  return counts;
}

function confidenceBar(label, value, maximum, colour) {
  const row = document.createElement('div'); row.className = 'confidence-bar-row';
  const name = document.createElement('span'); name.textContent = label;
  const track = document.createElement('span'); track.className = 'confidence-bar-track';
  const fill = document.createElement('span'); fill.className = 'confidence-bar-fill';
  fill.style.width = `${100 * value / maximum}%`; fill.style.background = colour; track.append(fill);
  const number = document.createElement('strong'); number.textContent = value.toFixed(value < 10 ? 1 : 0);
  row.append(name, track, number); return row;
}

mount('counts-confidence', element => {
  element.classList.add('confidence-widget');
  const controls = document.createElement('div'); controls.className = 'smoothing-controls';
  const conditionLabel = document.createElement('label'); conditionLabel.textContent = 'Starting counts';
  const condition = document.createElement('select');
  [['small', 'Low confidence in A column: total 0.5'], ['large', 'High confidence in A column: total 100']].forEach(([value, label]) => {
    const option = document.createElement('option'); option.value = value; option.textContent = label; condition.append(option);
  });
  conditionLabel.append(condition); controls.append(conditionLabel); element.append(controls);

  const stream = document.createElement('div'); stream.className = 'confidence-stream'; element.append(stream);
  const panels = document.createElement('div'); panels.className = 'confidence-panels'; element.append(panels);
  const stepLabel = document.createElement('label'); stepLabel.className = 'smoothing-step';
  const stepOutput = document.createElement('output');
  stepLabel.append(document.createTextNode('Observations incorporated: '), stepOutput);
  const slider = document.createElement('input'); slider.type = 'range'; slider.min = 0;
  slider.max = CONFIDENCE_STREAM.length; slider.step = 1; slider.value = 0;
  stepLabel.append(slider); element.append(stepLabel);
  const buttons = document.createElement('div'); buttons.className = 'confidence-buttons';
  const previous = document.createElement('button'); previous.type = 'button'; previous.textContent = 'Previous';
  const next = document.createElement('button'); next.type = 'button'; next.textContent = 'Add next observation';
  buttons.append(previous, next); element.append(buttons);
  const readout = document.createElement('p'); readout.className = 'smoothing-readout'; element.append(readout);

  function draw() {
    const key = condition.value, step = Number(slider.value);
    const counts = confidenceCounts(key, step), total = counts[0] + counts[1];
    const probability = counts.map(value => value / total);
    const novelty = CONFIDENCE_START[key].novelty[step];
    stepOutput.value = step; stepOutput.textContent = step;
    stream.replaceChildren(...CONFIDENCE_STREAM.map((observation, index) => {
      const chip = document.createElement('span'); chip.textContent = observation;
      chip.className = `${observation} ${index < step ? 'used' : ''}`; return chip;
    }));

    const raw = document.createElement('section'); raw.className = 'confidence-panel';
    const rawTitle = document.createElement('h3'); rawTitle.textContent = 'Counts a'; raw.append(rawTitle,
      confidenceBar('dark', counts[0], 108, CONFIDENCE_COLOURS.dark),
      confidenceBar('lit', counts[1], 108, CONFIDENCE_COLOURS.lit));
    const rawValue = document.createElement('p'); rawValue.textContent = `a = [${counts[0].toFixed(1)}, ${counts[1].toFixed(1)}], total ${total.toFixed(1)}`; raw.append(rawValue);

    const estimate = document.createElement('section'); estimate.className = 'confidence-panel';
    const estimateTitle = document.createElement('h3'); estimateTitle.textContent = 'Normalised estimate A';
    const stack = document.createElement('div'); stack.className = 'confidence-stack';
    const dark = document.createElement('span'); dark.style.width = `${100 * probability[0]}%`; dark.className = 'dark';
    const lit = document.createElement('span'); lit.style.width = `${100 * probability[1]}%`; lit.className = 'lit';
    stack.append(dark, lit);
    const estimateValue = document.createElement('p'); estimateValue.textContent = `p(dark) ${probability[0].toFixed(3)} · p(lit) ${probability[1].toFixed(3)}`;
    estimate.append(estimateTitle, stack, estimateValue);

    const noveltyPanel = document.createElement('section'); noveltyPanel.className = 'confidence-panel';
    const noveltyTitle = document.createElement('h3'); noveltyTitle.textContent = 'Expected novelty of the next observation';
    const noveltyTrack = document.createElement('div'); noveltyTrack.className = 'confidence-novelty-track';
    const noveltyFill = document.createElement('span'); noveltyFill.style.width = `${100 * novelty / Math.log(2)}%`; noveltyTrack.append(noveltyFill);
    const noveltyValue = document.createElement('p');
    noveltyValue.textContent = `${novelty.toFixed(6)} nats · averaged over the next dark or lit observation`;
    noveltyPanel.append(noveltyTitle, noveltyTrack, noveltyValue);
    panels.replaceChildren(raw, estimate, noveltyPanel);

    const change = step ? probability[CONFIDENCE_STREAM[step - 1] === 'lit' ? 1 : 0] : probability[1];
    readout.textContent = step === 0
      ? `Both conditions begin at A = [0.2, 0.8]. ${CONFIDENCE_START[key].label} in this column's observation probabilities is carried by the count total, not by a different estimate.`
      : `After ${CONFIDENCE_STREAM[step - 1]}: the observed outcome now has probability ${change.toFixed(3)}; expected information gain about this A column from the next observation is ${novelty.toFixed(6)} nats.`;
    previous.disabled = step === 0; next.disabled = step === CONFIDENCE_STREAM.length;
  }
  condition.addEventListener('change', draw); slider.addEventListener('input', draw);
  previous.addEventListener('click', () => { slider.value = Math.max(0, Number(slider.value) - 1); draw(); });
  next.addEventListener('click', () => { slider.value = Math.min(CONFIDENCE_STREAM.length, Number(slider.value) + 1); draw(); });
  draw();
});

mount('novelty-branches', element => {
  element.classList.add('novelty-branches-widget');
  const select = document.createElement('select');
  [['small', 'Low counts'], ['large', 'High counts']].forEach(([value, label]) => {
    const option = document.createElement('option'); option.value = value; option.textContent = label; select.append(option);
  });
  const flow = document.createElement('div'); flow.className = 'novelty-flow';
  element.append(select, flow);

  function draw() {
    const example = CONFIDENCE_START[select.value];
    const counts = example.counts;
    const total = counts[0] + counts[1];
    const probabilities = counts.map(count => count / total);
    const novelty = example.novelty[0];
    const current = document.createElement('div'); current.className = 'novelty-current';
    const currentTitle = document.createElement('strong'); currentTitle.textContent = 'Current belief about one cell';
    const currentCounts = document.createElement('span'); currentCounts.textContent = `Counts: [dark ${counts[0]}, lit ${counts[1]}]`;
    const currentModel = document.createElement('span'); currentModel.textContent = 'Model: dark 20% · lit 80%';
    current.append(currentTitle, currentCounts, currentModel);
    const question = document.createElement('div'); question.className = 'novelty-question';
    question.textContent = 'What might the agent observe next?';
    const branches = document.createElement('div'); branches.className = 'novelty-options';
    ['dark', 'lit'].forEach((observation, index) => {
      const updated = [...counts]; updated[index] += 1;
      const branch = document.createElement('div'); branch.className = observation;
      const title = document.createElement('strong'); title.textContent = observation.toUpperCase();
      const probability = document.createElement('span'); probability.textContent = `Probability: ${probabilities[index].toFixed(2)}`;
      const update = document.createElement('span'); update.textContent = `Counts become [${updated[0]}, ${updated[1]}]`;
      const change = document.createElement('span'); change.textContent = `Model change: ${example.nextInformation[index].toFixed(3)} nats`;
      const contribution = document.createElement('span');
      contribution.textContent = `${probabilities[index].toFixed(2)} × ${example.nextInformation[index].toFixed(3)} = ${(probabilities[index] * example.nextInformation[index]).toFixed(3)}`;
      branch.append(title, probability, update, change, contribution);
      branches.append(branch);
    });
    const result = document.createElement('div'); result.className = 'novelty-result';
    const resultTitle = document.createElement('strong'); resultTitle.textContent = 'NOVELTY';
    const resultCalculation = document.createElement('span');
    resultCalculation.textContent = `${probabilities[0].toFixed(2)} × ${example.nextInformation[0].toFixed(3)} + ${probabilities[1].toFixed(2)} × ${example.nextInformation[1].toFixed(3)}`;
    const resultValue = document.createElement('span'); resultValue.textContent = `= ${novelty.toFixed(3)} nats`;
    result.append(resultTitle, resultCalculation, resultValue);
    flow.replaceChildren(current, question, branches, result);
  }

  select.addEventListener('change', draw);
  draw();
});

mount('smoothing-counts', element => {
  element.classList.add('smoothing-counts-widget');
  const label = document.createElement('label'); label.textContent = 'Evidence available';
  const select = document.createElement('select');
  [['then', 'When dark arrived (before smoothing)'],
    ['now', 'After the new lit observation clarified the path (after smoothing)']].forEach(([value, text]) => {
    const option = document.createElement('option'); option.value = value; option.textContent = text; select.append(option);
  });
  label.append(select); element.append(label);
  const calculation = document.createElement('div'); calculation.className = 'smoothing-count-calculation';
  const explanation = document.createElement('p'); explanation.className = 'smoothing-readout';
  element.append(calculation, explanation);

  function table(caption, headings, rows, rowLabels = []) {
    const box = document.createElement('div'); box.className = 'smoothing-count-factor';
    const title = document.createElement('strong'); title.textContent = caption;
    const grid = document.createElement('table');
    if (headings) {
      const head = document.createElement('tr');
      if (rowLabels.length) { const corner = document.createElement('th'); corner.className = 'row-label'; head.append(corner); }
      headings.forEach(value => { const cell = document.createElement('th'); cell.textContent = value; head.append(cell); });
      grid.append(head);
    }
    rows.forEach((values, index) => {
      const row = document.createElement('tr');
      if (rowLabels.length) {
        const rowLabel = document.createElement('th'); rowLabel.className = 'row-label'; rowLabel.scope = 'row';
        rowLabel.textContent = rowLabels[index]; row.append(rowLabel);
      }
      values.forEach(value => { const cell = document.createElement('td'); cell.textContent = value; row.append(cell); });
      grid.append(row);
    });
    box.append(title, grid); return box;
  }

  function draw() {
    const revised = select.value === 'now';
    const belief = revised ? [.9, .1] : [.5, .5];
    const observation = table('Observation o', null, [['1'], ['0']], ['dark', 'lit']);
    const location = table('State belief q', ['Cell 1', 'Cell 2'], [[belief[0].toFixed(2), belief[1].toFixed(2)]]);
    const contribution = table('o ⊗ q', ['Cell 1', 'Cell 2'], [
      [belief[0].toFixed(2), belief[1].toFixed(2)], ['0', '0'],
    ], ['dark', 'lit']);
    const multiply = document.createElement('b'); multiply.textContent = '×';
    const equals = document.createElement('b'); equals.textContent = '=';
    calculation.replaceChildren(observation, multiply, location, equals, contribution);
    explanation.textContent = revised
      ? 'Revised assignment: replace the dark row [0.50, 0.50] with [0.90, 0.10]. The lit row stays [0, 0], and the total dark evidence is still one.'
      : 'Initial assignment: the dark row is [0.50, 0.50] and the lit row is [0, 0]. The one dark observation is shared equally between the two possible cells.';
  }

  select.addEventListener('change', draw);
  draw();
});

mount('smoothing-comparison', element => {
  element.classList.add('smoothing-widget');
  const controls = document.createElement('div'); controls.className = 'smoothing-controls';
  const modeLabel = document.createElement('label'); modeLabel.textContent = 'Sensor';
  const mode = document.createElement('select');
  [['night', 'Night: lamps only'], ['day', 'Day: landmarks visible']].forEach(([value, label]) => {
    const option = document.createElement('option'); option.value = value; option.textContent = label; mode.append(option);
  });
  mode.value = 'day'; modeLabel.append(mode);
  const pathLabel = document.createElement('label'); pathLabel.textContent = 'Path density';
  const pathPolicy = document.createElement('select');
  [['aif', 'AIF with A novelty'], ['state', 'AIF without A novelty'], ['uniform', 'Random walk']].forEach(([value, label]) => {
    const option = document.createElement('option'); option.value = value; option.textContent = label; pathPolicy.append(option);
  });
  pathLabel.append(pathPolicy);
  const toggleLabel = document.createElement('label'); toggleLabel.className = 'smoothing-toggle';
  const toggle = document.createElement('input'); toggle.type = 'checkbox';
  toggleLabel.append(toggle, document.createTextNode('Use a five-step smoothing window'));
  controls.append(modeLabel, pathLabel, toggleLabel); element.append(controls);
  const explanation = document.createElement('p'); explanation.className = 'smoothing-explanation'; element.append(explanation);
  const key = document.createElement('div'); key.className = 'smoothing-key';
  ['100% = starting error', '↓ lower is better', 'coloured number = works visits'].forEach(text => {
    const item = document.createElement('span'); item.textContent = text; key.append(item);
  });
  element.append(key);
  const chartBox = document.createElement('div'); chartBox.className = 'smoothing-a-chart'; element.append(chartBox);
  const maps = document.createElement('div'); maps.className = 'smoothing-maps'; element.append(maps);
  const stepLabel = document.createElement('label'); stepLabel.className = 'smoothing-step';
  const output = document.createElement('output');
  stepLabel.append(document.createTextNode('Inspect after '), output, document.createTextNode(' steps'));
  const slider = document.createElement('input'); slider.type = 'range'; slider.min = 0;
  slider.max = DATA.snapshots.length - 1; slider.step = 1; slider.value = DATA.snapshots.length - 1;
  slider.setAttribute('aria-label', 'Step checkpoint'); stepLabel.append(slider); element.append(stepLabel);
  const readout = document.createElement('p'); readout.className = 'smoothing-readout'; readout.setAttribute('aria-live', 'polite'); element.append(readout);
  const note = document.createElement('p'); note.className = 'smoothing-note'; element.append(note);

  function draw() {
    const snapshot = Number(slider.value), step = DATA.snapshots[snapshot], window = toggle.checked ? 5 : 1;
    const uniform = DATA.series[`${mode.value}_uniform_w${window}`];
    const state = DATA.series[`${mode.value}_state_w${window}`];
    const aif = DATA.series[`${mode.value}_aif_w${window}`];
    const baselines = { uniform: DATA.series[`${mode.value}_uniform_w1`],
      state: DATA.series[`${mode.value}_state_w1`], aif: DATA.series[`${mode.value}_aif_w1`] };
    output.value = step; output.textContent = step;
    chartBox.replaceChildren(curve({ uniform, state, aif }, baselines, step, toggle.checked));
    const world = DATA.worlds[mode.value];
    const pathSeries = { uniform, state, aif }[pathPolicy.value];
    maps.replaceChildren(map(`${POLICY[pathPolicy.value].label} path`, world, null, snapshot,
        pathSeries.density[snapshot], POLICY[pathPolicy.value].colour),
      map('Before learning', world, aif, 0), map(`Active inference after ${step} steps`, world, aif, snapshot));
    explanation.textContent = toggle.checked
      ? 'Solid curves use the five-step window; faint dashed curves show the corresponding result without it.'
      : 'All three agents update A after every observation. Their action-selection rules differ.';
    readout.textContent = `After ${step} steps: random walk ${uniform.visits[step].toFixed(0)} works visits, A error ${uniform.mean[step].toFixed(3)}; ` +
      `AIF without A novelty ${state.visits[step].toFixed(0)} visits, error ${state.mean[step].toFixed(3)}; ` +
      `AIF with A novelty ${aif.visits[step].toFixed(0)} visits, error ${aif.mean[step].toFixed(3)}.`;
    note.textContent = 'Curves show the fraction of initial error remaining; the readout gives raw total-variation error. Mean over 32 runs; shading is ±1 standard error. The path-density map shows one reproducible run, because averaging paths would hide their variation. B is known and fixed at reliability 0.8. Random-walk runs retain matched traces across window settings; the active agent may change later actions when smoothing changes its model.';
  }
  [mode, pathPolicy, toggle].forEach(input => input.addEventListener('change', draw));
  slider.addEventListener('input', draw); draw();
});

const B_STRATEGIES = {
  uniform: { label: 'Random walk', colour: '#777', dash: '2 4' },
  state: { label: 'AIF without B novelty', colour: '#2f6f9f', dash: '7 4' },
  active: { label: 'AIF with B novelty', colour: '#19775e', dash: null },
};

function bChart(mode, index) {
  const svg = node('svg', { viewBox: '0 0 620 310', role: 'img',
    'aria-label': 'Mean transition-model error in the works area over steps on a logarithmic horizontal scale' });
  const x = step => 56 + 525 * Math.log2(step + 1) / Math.log2(8193);
  const y = value => 250 - 195 * value / .12;
  [0, .04, .08, .12].forEach(value => {
    svg.append(node('line', { x1: 56, x2: 581, y1: y(value), y2: y(value),
      stroke: value === 0 ? '#333' : '#deded8', 'stroke-dasharray': value === 0 ? '5 4' : '' }));
    svg.append(node('text', { x: 48, y: y(value) + 4, 'text-anchor': 'end', 'font-size': 12 }, value.toFixed(2)));
  });
  [0, 32, 128, 512, 2048, 8192].forEach(step =>
    svg.append(node('text', { x: x(step), y: 272, 'text-anchor': 'middle', 'font-size': 11 },
      step >= 1000 ? `${step / 1024}k` : step)));
  svg.append(node('text', { x: 320, y: 300, 'text-anchor': 'middle', 'font-size': 12 }, 'Steps (log₂(step + 1) scale)'));
  svg.append(node('text', { x: 15, y: 155, transform: 'rotate(-90 15 155)', 'text-anchor': 'middle', 'font-size': 12 }, 'Works-area B error'));
  const selectedStep = B_DATA.snapshots[index];
  Object.entries(B_STRATEGIES).forEach(([strategy, style], order) => {
    const rows = B_DATA.series[`${strategy}_${mode}_move_stay`].filter(row => row.step <= 8192);
    const upper = rows.map(row => `${x(row.step)},${y(row.mean[1] + row.se[1])}`);
    const lower = rows.map(row => `${x(row.step)},${y(row.mean[1] - row.se[1])}`).reverse();
    svg.append(node('polygon', { points: [...upper, ...lower].join(' '), fill: style.colour, opacity: .12 }));
    svg.append(node('polyline', { points: rows.map(row => `${x(row.step)},${y(row.mean[1])}`).join(' '),
      fill: 'none', stroke: style.colour, 'stroke-width': 3,
      ...(style.dash ? { 'stroke-dasharray': style.dash } : {}) }));
    const row = rows[index];
    svg.append(node('circle', { cx: x(row.step), cy: y(row.mean[1]), r: 4, fill: style.colour }));
    svg.append(node('text', { x: 575, y: y(rows.at(-1).mean[1]) + (order % 2 ? 15 : -8),
      fill: style.colour, 'text-anchor': 'end', 'font-size': 12 }, style.label));
  });
  svg.append(node('line', { x1: x(selectedStep), x2: x(selectedStep), y1: 40, y2: 250,
    stroke: '#888', 'stroke-dasharray': '2 3' }));
  svg.append(node('text', { x: 575, y: 245, 'text-anchor': 'end', 'font-size': 11 }, 'perfect model = 0'));
  return svg;
}

function collapseChart() {
  const svg = node('svg', { viewBox: '0 0 390 210', role: 'img',
    'aria-label': 'Position belief during five repeated blocked commands at night and by day' });
  const x = step => 42 + step * 62;
  const y = value => 170 - 130 * value;
  [0, .5, 1].forEach(value => {
    svg.append(node('line', { x1: 42, x2: 352, y1: y(value), y2: y(value), stroke: '#e1e1dc' }));
    svg.append(node('text', { x: 34, y: y(value) + 4, 'text-anchor': 'end', 'font-size': 11 }, value.toFixed(1)));
  });
  [0, 1, 2, 3, 4, 5].forEach(step => svg.append(node('text', {
    x: x(step), y: 190, 'text-anchor': 'middle', 'font-size': 11 }, step)));
  svg.append(node('text', { x: 197, y: 207, 'text-anchor': 'middle', 'font-size': 11 }, 'Repeated blocked commands and observations'));
  for (const [key, colour, label] of [['night', '#555', 'Night'], ['day', '#19775e', 'Day']]) {
    const values = B_DATA.collapse[key];
    svg.append(node('polyline', { points: values.map((value, step) => `${x(step)},${y(value)}`).join(' '),
      fill: 'none', stroke: colour, 'stroke-width': 3 }));
    values.forEach((value, step) => svg.append(node('circle', { cx: x(step), cy: y(value), r: 3.5, fill: colour })));
    svg.append(node('text', { x: 358, y: y(values.at(-1)) + 4, fill: colour, 'font-size': 12 }, label));
  }
  svg.append(node('text', { x: 14, y: 105, transform: 'rotate(-90 14 105)', 'text-anchor': 'middle', 'font-size': 11 }, 'q(true cell)'));
  return svg;
}

mount('b-learning-comparison', element => {
  element.classList.add('smoothing-widget');
  const controls = document.createElement('div'); controls.className = 'smoothing-controls';
  const modeLabel = document.createElement('label'); modeLabel.textContent = 'Sensor';
  const mode = document.createElement('select');
  [['night', 'Night: dark or lit'], ['day', 'Day: landmarks visible']].forEach(([value, text]) => {
    const option = document.createElement('option'); option.value = value; option.textContent = text; mode.append(option);
  });
  mode.value = 'day'; modeLabel.append(mode); controls.append(modeLabel); element.append(controls);
  const chartBox = document.createElement('div'); chartBox.className = 'smoothing-a-chart'; element.append(chartBox);
  const label = document.createElement('label'); label.className = 'smoothing-step';
  const output = document.createElement('output');
  label.append(document.createTextNode('Inspect after '), output, document.createTextNode(' steps'));
  const slider = document.createElement('input'); slider.type = 'range'; slider.min = 0;
  slider.max = B_DATA.snapshots.indexOf(8192); slider.step = 1; slider.value = slider.max;
  slider.setAttribute('aria-label', 'B-learning checkpoint'); label.append(slider); element.append(label);
  const readout = document.createElement('p'); readout.className = 'smoothing-readout'; element.append(readout);
  const note = document.createElement('p'); note.className = 'smoothing-note'; element.append(note);
  const diagnostic = document.createElement('section'); diagnostic.className = 'b-diagnostic';
  const diagnosticTitle = document.createElement('h3'); diagnosticTitle.textContent = 'Why the wall cannot be learned at night';
  const diagnosticNote = document.createElement('p');
  diagnosticNote.textContent = 'Fixed diagnostic: five blocked commands are prescribed, so the action-selection control does not change this trace.';
  const inset = document.createElement('div'); inset.className = 'b-collapse'; inset.append(collapseChart());
  diagnostic.append(diagnosticTitle, diagnosticNote, inset); element.append(diagnostic);
  function draw() {
    const index = Number(slider.value), step = B_DATA.snapshots[index];
    output.value = step; output.textContent = step;
    chartBox.replaceChildren(bChart(mode.value, index));
    readout.textContent = Object.entries(B_STRATEGIES).map(([strategy, style]) => {
      const rows = B_DATA.series[`${strategy}_${mode.value}_move_stay`];
      const row = rows[index];
      return `${style.label}: B error ${row.mean[1].toFixed(3)}, wall stay ${row.mean[2].toFixed(3)}, ${row.blocked_mean.toFixed(0)} blocked attempts`;
    }).join(' · ');
    note.textContent = 'All three agents use the same move-or-stay transition prior and count update. Only action selection differs. Mean over 32 runs; bands are ±1 standard error.';
  }
  mode.addEventListener('change', draw); slider.addEventListener('input', draw); draw();
});

const JOINT_SCOPES = {
  works: { label: 'Works only', colour: '#19775e' },
  region: { label: 'Larger region', colour: '#39758d' },
  world: { label: 'Whole map except Home', colour: '#b45309' },
};

function jointChart(metric, title) {
  const svg = node('svg', { viewBox: '0 0 620 310', role: 'img',
    'aria-label': `${title} while A and B are learned together` });
  const x = step => 56 + 520 * Math.log2(step + 1) / Math.log2(4097);
  const y = value => 248 - 190 * value;
  [0, .25, .5, .75, 1].forEach(value => {
    svg.append(node('line', { x1: 56, x2: 576, y1: y(value), y2: y(value), stroke: '#deded8' }));
    svg.append(node('text', { x: 48, y: y(value) + 4, 'text-anchor': 'end', 'font-size': 12 }, value.toFixed(2)));
  });
  JOINT_DATA.snapshots.forEach(step => svg.append(node('text', {
    x: x(step), y: 270, 'text-anchor': 'middle', 'font-size': 11,
  }, step >= 1024 ? `${step / 1024}k` : step)));
  svg.append(node('text', { x: 316, y: 299, 'text-anchor': 'middle', 'font-size': 12 }, 'Steps (log₂(step + 1) scale)'));
  svg.append(node('text', { x: 15, y: 153, transform: 'rotate(-90 15 153)',
    'text-anchor': 'middle', 'font-size': 12 }, `Initial ${metric ? 'B' : 'A'} error remaining`));
  svg.append(node('text', { x: 316, y: 22, 'text-anchor': 'middle',
    'font-size': 15, 'font-weight': 650 }, title));
  Object.entries(JOINT_SCOPES).forEach(([key, style], order) => {
    const data = JOINT_DATA.series[key];
    const values = data.mean.map(row => row[metric] / data.mean[0][metric]);
    const errors = data.se.map(row => row[metric] / data.mean[0][metric]);
    const upper = values.map((value, index) => `${x(JOINT_DATA.snapshots[index])},${y(value + errors[index])}`);
    const lower = values.map((value, index) => `${x(JOINT_DATA.snapshots[index])},${y(value - errors[index])}`).reverse();
    svg.append(node('polygon', { points: [...upper, ...lower].join(' '), fill: style.colour, opacity: .1 }));
    svg.append(node('polyline', { points: values.map((value, index) =>
      `${x(JOINT_DATA.snapshots[index])},${y(value)}`).join(' '), fill: 'none',
      stroke: style.colour, 'stroke-width': 4 }));
    svg.append(node('text', { x: 570, y: y(values.at(-1)) + [-8, 4, 15][order],
      fill: style.colour, 'text-anchor': 'end', 'font-size': 11 }, style.label));
  });
  return svg;
}

mount('joint-learning', element => {
  element.classList.add('joint-learning-widget');
  const charts = document.createElement('div'); charts.className = 'joint-learning-charts';
  const a = document.createElement('div'); a.append(jointChart(0, 'Learning observations (A)'));
  const b = document.createElement('div'); b.append(jointChart(1, 'Learning transitions (B)'));
  charts.append(a, b); element.append(charts);
  const key = document.createElement('p'); key.className = 'smoothing-note';
  key.textContent = 'Each curve shows the fraction of its own starting error that remains. Lower is better; bands are ±1 standard error over 16 runs.';
  element.append(key);
});

mountAll();

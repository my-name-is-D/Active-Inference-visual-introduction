/** Replays Python-generated five-agent walks; no planner is reimplemented here. */
import { mount, ctxOf, drawIsoPlane, label, COLOUR } from './widget.js';
import { AGENT_WALKS } from './agent-comparison-data.js';

const HOME = 4;
const LAMPS = new Set([9, 13]);
const KINDS = ['aif', 'sophisticated', 'reward', 'qlearning', 'tabular'];
const ACTIONS = ['north', 'south', 'west', 'east', 'stay'];
const LIT_MASK = Array.from({ length: 25 }, (_, s) => LAMPS.has(s));
const DESCRIPTIONS = {
  aif: 'Plans a fixed three-action sequence using preferred outcomes and expected information gain; replans after each actual observation.',
  sophisticated: 'Plans over possible future observations and chooses a new continuation for each; uses the same preference and information terms as AIF.',
  reward: 'Plans a fixed three-action sequence for expected preference utility. It updates its position belief, but its sequence score has no explicit information term.',
  qlearning: 'Learns action values from sampled experience using its full Bayesian position belief. Its values are approximate and depend on training.',
  tabular: 'Learns a small table keyed only by the latest observation. The belief map shown here is a reader-only diagnostic; this agent never uses it to act.',
};

function node(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
}

// Format dynamic numeric readouts without interpreting their text as HTML.
function boldNumbers(el) {
  const parts = el.textContent.split(/(-?\d+(?:\.\d+)?%?)/g);
  el.replaceChildren(...parts.map((part, index) =>
    index % 2 ? node('strong', '', part) : document.createTextNode(part)));
}

function drawLessonTwoView(ctx, frame, mode, predicted = false) {
  const belief = predicted ? frame.predicted : frame.belief;
  const stacked = !predicted;
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  if (!predicted) {
    label(ctx, 'where the agent is', 28, 28, COLOUR.ink, 16);
    label(ctx, 'where it thinks it is', 28, 303, COLOUR.ink, 16);
    drawIsoPlane(ctx, 175, 67, 27, {
      // The red dot is reader-only truth.
      agent: frame.state,
      lit: LIT_MASK,
      home: HOME,
      homeHidden: mode === 'states',
    });
    label(ctx, 'The agent does not see the red dot.', 45, 254, COLOUR.agent, 13);
  }
  const peak = Math.max(...belief, 1e-9);
  drawIsoPlane(ctx, 175, stacked ? 341 : 66, 25, { heights: belief.map((p) => p / peak) });
  const probabilityY = stacked ? 529 : 254;
  label(ctx, 'highest cell probability: ', 28, probabilityY, COLOUR.dim, 13);
  const valueX = 28 + ctx.measureText('highest cell probability: ').width;
  ctx.font = 'bold 13px system-ui, sans-serif';
  ctx.fillText(`${(100 * peak).toFixed(1)}%`, valueX, probabilityY);
  ctx.canvas.setAttribute('role', 'img');
  ctx.canvas.setAttribute('aria-label',
    predicted
      ? `Step ${frame.step}. Predicted position belief after movement, before the observation has a highest cell probability of ${(100 * peak).toFixed(1)} percent.`
      : `Step ${frame.step}. Red dot at row ${Math.floor(frame.state / 5)+1}, ` +
        `column ${frame.state % 5+1}; highest position-belief probability ` +
        `${(100 * peak).toFixed(1)} percent.`);
}

function agentPanel(title, kinds, initial) {
  const root = node('section', 'walk-agent-panel');
  const head = node('label', 'walk-panel-head');
  head.appendChild(node('span', '', title));
  const select = node('select');
  for (const kind of kinds) {
    const option = node('option', '', AGENT_WALKS.modes.observations.agents[kind].label);
    option.value = kind;
    select.appendChild(option);
  }
  select.value = initial;
  head.appendChild(select);
  const status = node('p', 'walk-panel-status');
  const motion = node('p', 'walk-motion');
  const maps = node('div', 'walk-canvas-panel');
  const mapCtx = ctxOf(maps, 350, 550);
  const readout = node('div', 'walk-readout');
  const explanation = node('p', 'walk-explanation');
  const details = node('details', 'walk-details');
  const summary = node('summary');
  details.appendChild(summary);
  const underneath = node('div');
  const predictionPanel = node('div', 'walk-prediction');
  predictionPanel.appendChild(node('strong', '', 'Position belief after movement, before the observation'));
  const predictionCtx = ctxOf(predictionPanel, 350, 275);
  details.append(underneath, predictionPanel);
  root.append(head, status, motion, maps, readout, explanation);
  return { root, details, summary, select, status, motion, mapCtx, readout, explanation,
           underneath, predictionPanel, predictionCtx };
}

function renderAgent(panel, mode, walk, step) {
  panel.summary.textContent = `What is happening in ${AGENT_WALKS.modes[mode].agents[panel.select.value].label} process?`;
  const at = step;
  const frame = walk[at];
  drawLessonTwoView(panel.mapCtx, frame, mode);
  panel.status.textContent = frame.at_home
    ? `At home · step ${at}`
    : frame.reached_home ? `Home visited earlier · now elsewhere · step ${at}`
    : at === 10 ? 'Home not reached in 10 steps'
    : `Still moving · step ${at}`;
  boldNumbers(panel.status);
  panel.explanation.textContent = DESCRIPTIONS[panel.select.value];
  panel.motion.textContent = at === 0
    ? 'At the starting position · no move yet'
    : `Command ${frame.action_name.toUpperCase()} · ` +
      (frame.action_name === 'stay' ? 'agent stayed in place' :
        `agent moved ${frame.action_name}`);
  if (at === 0) {
    const observation = frame.reading === null ? '' :
      ` Tabular Q alone has an initial ${AGENT_WALKS.modes[mode].readings[frame.reading]} observation to index its table.`;
    panel.readout.textContent = `Same starting position across agents in this run. Starting belief entropy: ${frame.entropy.toFixed(2)} nats.${observation}`;
    boldNumbers(panel.readout);
    panel.underneath.textContent = 'Choose Next step to see the first action, noisy landing, sensor observation, and Bayesian update.';
    panel.predictionPanel.hidden = true;
    return;
  }
  const observation = AGENT_WALKS.modes[mode].readings[frame.reading];
  panel.readout.textContent = `${frame.action_name} → ${observation} observation. ` +
    `Belief entropy ${frame.before_entropy.toFixed(2)} → ${frame.predicted_entropy.toFixed(2)} ` +
    `after the agent's movement prediction → ${frame.entropy.toFixed(2)} after the observation (nats).`;
  boldNumbers(panel.readout);
  const trueP = frame.belief[frame.state];
  panel.underneath.replaceChildren();
  const observe = node('p', '',
    `Observe: the ${observation} observation had predicted probability ${(100 * frame.reading_probability).toFixed(1)}%.`);
  boldNumbers(observe);
  observe.append(node('br'), document.createTextNode("Bayes' rule reweights the predicted cells and normalises."));
  const check = node('p', '',
    `Reader-only check: the updated belief gives the true red-dot position ${(100 * trueP).toFixed(1)}%.`);
  boldNumbers(check);
  check.append(node('br'), document.createTextNode('The agent never sees the dot.'));
  panel.underneath.append(observe, check);
  const table = node('table', 'walk-action-values');
  table.appendChild(node('caption', '', 'Action values before the move (higher is preferred within this agent). Values across different agents have different scales.'));
  const header = table.createTHead().insertRow();
  const values = table.createTBody().insertRow();
  ACTIONS.forEach((action, index) => {
    const heading = node('th', '', action);
    heading.scope = 'col';
    header.appendChild(heading);
    const cell = node('td', '', frame.action_values[index].toFixed(2));
    if (frame.action_values[index] === Math.max(...frame.action_values)) boldNumbers(cell);
    values.appendChild(cell);
  });
  panel.underneath.appendChild(table);
  panel.underneath.appendChild(node('p', '',
    `For this first action only: expected preference cost ${frame.pragmatic.toFixed(2)}, expected epistemic cost ${frame.epistemic.toFixed(2)} nats. AIF uses both; the reward planner uses preference only. These are not whole three-step scores.`));
  boldNumbers(panel.underneath.lastElementChild);
  panel.predictionPanel.hidden = false;
  drawLessonTwoView(panel.predictionCtx, frame, mode, true);
}

mount('agent-walk-comparison', (el) => {
  const root = node('div', 'walk-widget');
  const controls = node('div', 'walk-controls');
  controls.appendChild(node('span', 'walk-control-heading', 'Two agents · same start and random draws'));
  const modeLabel = node('label', 'walk-control');
  modeLabel.appendChild(node('span', '', 'Preference'));
  const mode = node('select');
  for (const [value, title] of [['observations', 'See home marker'],
                                 ['states', 'Be at hidden home marker']]) {
    const option = node('option', '', title);
    option.value = value;
    mode.appendChild(option);
  }
  mode.value = 'states';
  modeLabel.appendChild(mode);
  controls.appendChild(modeLabel);

  const first = agentPanel('First agent', ['aif', 'sophisticated'], 'aif');
  const second = agentPanel('Compare with', ['sophisticated', 'reward', 'qlearning', 'tabular'], 'reward');
  const status = node('p', 'walk-status');
  status.setAttribute('aria-live', 'polite');
  const nav = node('div', 'walk-nav');
  const previous = node('button', '', 'Previous step');
  previous.type = 'button';
  const next = node('button', '', 'Next step');
  next.type = 'button';
  const another = node('button', '', 'Another run');
  another.type = 'button';
  const count = node('span', 'walk-count');
  nav.append(previous, next, another, count);
  const agents = node('div', 'walk-agents');
  agents.append(first.root, second.root);
  root.append(controls, status, nav, agents, first.details, second.details);
  el.appendChild(root);

  let step = 0;
  let run = 0;
  function walks() {
    const agents = AGENT_WALKS.modes[mode.value].agents;
    return [agents[first.select.value].walks[run],
            agents[second.select.value].walks[run]];
  }
  function render() {
    const [a, b] = walks();
    const last = AGENT_WALKS.protocol.max_steps;
    step = Math.min(step, last);
    const sameHistory = a.slice(1, step + 1).every((frame, index) =>
      frame.action === b[index + 1].action &&
      frame.reading === b[index + 1].reading);
    status.textContent = `C(home)=0.50 · shared step ${step} of ${last} · ${mode.value === 'states'
      ? 'home is dark to both sensors' : 'home has its own noisy observation'} · ` +
      (sameHistory
        ? 'same moves and observations so far: shared position beliefs match'
        : 'different moves or observations: position beliefs can diverge');
    count.textContent = `Run ${run + 1} of ${AGENT_WALKS.protocol.seeds.length}`;
    boldNumbers(count);
    boldNumbers(status);
    previous.disabled = step === 0;
    next.disabled = step === last;
    renderAgent(first, mode.value, a, step);
    renderAgent(second, mode.value, b, step);
  }
  previous.addEventListener('click', () => { step -= 1; render(); });
  next.addEventListener('click', () => { step += 1; render(); });
  another.addEventListener('click', () => {
    run = (run + 1) % AGENT_WALKS.protocol.seeds.length;
    step = 0;
    render();
  });
  first.select.addEventListener('change', () => { step = 0; render(); });
  second.select.addEventListener('change', () => { step = 0; render(); });
  mode.addEventListener('change', () => { step = 0; render(); });
  render();
});

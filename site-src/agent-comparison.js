/** Replays Python-generated five-agent walks; no planner is reimplemented here. */
import { mount, ctxOf, drawIsoPlane, label, COLOUR } from './widget.js';
import { AGENT_WALKS } from './agent-comparison-data.js';

const HOME = 4;
const LAMPS = new Set([9, 13]);
const KINDS = ['aif', 'sophisticated', 'reward', 'qlearning', 'tabular'];
const ACTIONS = ['north', 'south', 'west', 'east', 'stay'];
const LIT_MASK = Array.from({ length: 25 }, (_, s) => LAMPS.has(s));
const DESCRIPTIONS = {
  aif: 'Plans a fixed three-action sequence using preferred outcomes and expected information gain; replans after each actual reading.',
  sophisticated: 'Plans over possible future readings and chooses a new continuation for each; uses the same preference and information terms as AIF.',
  reward: 'Plans a fixed three-action sequence for expected preference utility. It updates its position belief, but its sequence score has no explicit information term.',
  qlearning: 'Learns action values from sampled experience using its full Bayesian position belief. Its values are approximate and depend on training.',
  tabular: 'Learns a small table keyed only by the latest reading. The belief map shown here is a reader-only diagnostic; this agent never uses it to act.',
};

function node(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
}

function drawLessonTwoView(ctx, frame, mode, predicted = false) {
  const belief = predicted ? frame.predicted : frame.belief;
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  label(ctx, 'where the agent is', 28, 28, COLOUR.ink, 16);
  label(ctx, predicted ? 'predicted position before seeing' : 'where it thinks it is',
    380, 28, COLOUR.ink, 16);
  drawIsoPlane(ctx, 190, 67, 27, {
    // The red dot is reader-only truth, including before the sensor reading.
    agent: frame.state,
    lit: LIT_MASK,
    home: HOME,
    homeHidden: mode === 'states',
  });
  const peak = Math.max(...belief, 1e-9);
  drawIsoPlane(ctx, 530, 66, 25, { heights: belief.map((p) => p / peak) });
  label(ctx, 'red dot: actual cell', 45, 254, COLOUR.agent, 13);
  label(ctx, `highest cell probability: ${(100 * peak).toFixed(1)}%`,
    382, 254, COLOUR.dim, 13);
  ctx.canvas.setAttribute('role', 'img');
  ctx.canvas.setAttribute('aria-label',
    predicted
      ? `Step ${frame.step}. Reader-only red dot marks the actual cell; predicted position belief before the reading has a highest cell probability of ${(100 * peak).toFixed(1)} percent.`
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
  const mapCtx = ctxOf(maps, 700, 275);
  const readout = node('div', 'walk-readout');
  const explanation = node('p', 'walk-explanation');
  const details = node('details', 'walk-details');
  details.appendChild(node('summary', '', 'What is happening underneath?'));
  const underneath = node('div');
  const predictionPanel = node('div', 'walk-prediction');
  predictionPanel.appendChild(node('strong', '', 'Predicted position before the reading'));
  const predictionCtx = ctxOf(predictionPanel, 700, 275);
  details.append(underneath, predictionPanel);
  root.append(head, status, motion, maps, readout, explanation, details);
  return { root, select, status, motion, mapCtx, readout, explanation,
           underneath, predictionPanel, predictionCtx };
}

function renderAgent(panel, mode, walk, step) {
  const at = step;
  const frame = walk[at];
  drawLessonTwoView(panel.mapCtx, frame, mode);
  panel.status.textContent = frame.at_home
    ? `At home · step ${at}`
    : frame.reached_home ? `Home visited earlier · now elsewhere · step ${at}`
    : at === 10 ? 'Home not reached in 10 steps'
    : `Still moving · step ${at}`;
  panel.explanation.textContent = DESCRIPTIONS[panel.select.value];
  panel.motion.textContent = at === 0
    ? 'At the starting position · no move yet'
    : `Command ${frame.action_name.toUpperCase()} · ` +
      (frame.action_name === 'stay' ? 'agent stayed in place' :
        `agent moved ${frame.action_name}`);
  if (at === 0) {
    const reading = frame.reading === null ? '' :
      ` Tabular Q alone has an initial ${AGENT_WALKS.modes[mode].readings[frame.reading]} reading to index its table.`;
    panel.readout.textContent = `Same starting position across agents in this run. The agent does not see the red dot. Starting belief entropy: ${frame.entropy.toFixed(2)} nats.${reading}`;
    panel.underneath.textContent = 'Choose Next step to see the first action, noisy landing, sensor reading, and Bayesian update.';
    panel.predictionPanel.hidden = true;
    return;
  }
  const reading = AGENT_WALKS.modes[mode].readings[frame.reading];
  panel.readout.textContent = `${frame.action_name} → ${reading} reading. ` +
    `Belief entropy ${frame.before_entropy.toFixed(2)} → ${frame.predicted_entropy.toFixed(2)} ` +
    `after the agent's movement prediction → ${frame.entropy.toFixed(2)} after the reading (nats).`;
  const trueP = frame.belief[frame.state];
  const ordering = frame.action_values.map((value, a) =>
    `${ACTIONS[a]} ${value.toFixed(2)}`).join(' · ');
  panel.underneath.replaceChildren();
  panel.underneath.appendChild(node('p', '', 'Predict: B moves probability across cells. The middle entropy above measures uncertainty after movement, before the reading.'));
  panel.underneath.appendChild(node('p', '',
    `Observe: the reading had predicted probability ${(100 * frame.reading_probability).toFixed(1)}%. Bayes' rule reweights the predicted cells and normalises.`));
  panel.underneath.appendChild(node('p', '',
    `Reader-only check: the updated belief gives the true red-dot cell ${(100 * trueP).toFixed(1)}%. The agent never sees the dot.`));
  panel.underneath.appendChild(node('p', '',
    `Action values before the move (higher is preferred within this agent): ${ordering}. Values across different agents have different scales.`));
  panel.underneath.appendChild(node('p', '',
    `For this first action only: expected preference cost ${frame.pragmatic.toFixed(2)}, expected epistemic cost ${frame.epistemic.toFixed(2)} nats. AIF uses both; the reward planner uses preference only. These are not whole three-step scores.`));
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
  root.append(controls, status, nav, first.root, second.root);
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
      ? 'home is dark to both sensors' : 'home has its own noisy reading'} · ` +
      (sameHistory
        ? 'same moves and readings so far: shared position beliefs match'
        : 'different moves or readings: position beliefs can diverge');
    count.textContent = `Run ${run + 1} of ${AGENT_WALKS.protocol.seeds.length}`;
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

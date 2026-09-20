// Figure 7: what the Figure 6 runs look like in aggregate.
//
// Figure 6 shows one run at a time, so a reader cannot tell whether what they
// watched was typical. These panels answer that from 300 seeds per task.
//
// Two panels, both visible at once, because they answer the two halves of the
// lesson's question and the reader should see the trade-off without clicking:
//
//   left    localization over the nine moves: belief entropy above, and the
//           accuracy check below. Entropy alone measures confidence, and a
//           belief can concentrate on the wrong cell, so the pair is what
//           makes "knows where it is" honest.
//   right   how often each agent reached the actual home cell.
//
// The task selector matches Figure 6's preference selector. Changing it is a
// change of sensory task, not an isolated change to C: the home-marker task
// gives home its own reading, the hidden-home task does not.
//
// Every number is computed by verification/figure6_check.py and exported by
// verification/export_figure6_curves.py. Nothing is recomputed here.
import { FIGURE6_CURVES } from './figure6-curves-data.js';
import { COLOUR, ctxOf, label, mount } from './widget.js';

const W = 760;
const H = 300;
const PAD = { left: 52, right: 14, top: 42, bottom: 40 };

function node(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
}

// Shared axis frame: maps data coordinates into the padded plot box once, so
// the three panels cannot drift apart.
function axes(ctx, { xMax, yMin, yMax, yLabel, yTicks, xLabel }) {
  const x0 = PAD.left;
  const y0 = H - PAD.bottom;
  const x1 = W - PAD.right;
  const y1 = PAD.top;
  const sx = (v) => x0 + (v / xMax) * (x1 - x0);
  const sy = (v) => y0 - ((v - yMin) / (yMax - yMin)) * (y0 - y1);

  ctx.strokeStyle = COLOUR.dim;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x0, y1);
  ctx.lineTo(x0, y0);
  ctx.lineTo(x1, y0);
  ctx.stroke();

  for (const t of yTicks) {
    const y = sy(t);
    ctx.strokeStyle = 'rgba(0,0,0,0.08)';
    ctx.beginPath();
    ctx.moveTo(x0, y);
    ctx.lineTo(x1, y);
    ctx.stroke();
    label(ctx, String(t), x0 - 8, y + 4, COLOUR.dim, 12, 'right');
  }
  for (let s = 0; s <= xMax; s += 1) {
    label(ctx, String(s), sx(s), y0 + 16, COLOUR.dim, 12, 'center');
  }
  ctx.save();
  ctx.translate(14, (y0 + y1) / 2);
  ctx.rotate(-Math.PI / 2);
  label(ctx, yLabel, 0, 0, COLOUR.dim, 12, 'center');
  ctx.restore();
  if (xLabel) label(ctx, xLabel, (x0 + x1) / 2, H - 8, COLOUR.dim, 12, 'center');
  return { sx, sy };
}

function drawSeries(ctx, scale, values, colour, hovered) {
  ctx.strokeStyle = colour;
  ctx.lineWidth = hovered ? 3 : 1.8;
  ctx.globalAlpha = hovered === null ? 1 : hovered ? 1 : 0.25;
  ctx.beginPath();
  values.forEach((v, s) => {
    const x = scale.sx(s);
    const y = scale.sy(v);
    if (s === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = colour;
  values.forEach((v, s) => {
    ctx.beginPath();
    ctx.arc(scale.sx(s), scale.sy(v), hovered ? 3.5 : 2.5, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.globalAlpha = 1;
}

mount('figure6-curves', (el) => {
  const data = FIGURE6_CURVES;
  const root = node('div', 'curves-widget');

  const controls = node('div', 'walk-controls');
  const taskLabel = node('label', 'walk-control');
  taskLabel.appendChild(node('span', '', 'Task'));
  const task = node('select');
  for (const [value, block] of Object.entries(data.tasks)) {
    const option = node('option', '', block.title);
    option.value = value;
    task.appendChild(option);
  }
  task.value = 'hidden_home';
  taskLabel.appendChild(task);
  controls.appendChild(taskLabel);
  controls.appendChild(node('span', 'walk-control-heading',
    `${data.protocol.evaluation_runs} runs per agent`));
  root.appendChild(controls);

  const entropyCanvas = node('div', 'curves-panel');
  const accuracyCanvas = node('div', 'curves-panel');
  const homeCanvas = node('div', 'curves-panel');
  root.appendChild(entropyCanvas);
  root.appendChild(accuracyCanvas);
  root.appendChild(homeCanvas);
  const readout = node('p', 'curves-readout');
  readout.style.minHeight = '1.4em';
  root.appendChild(readout);
  el.appendChild(root);

  const ctxEntropy = ctxOf(entropyCanvas, W, H);
  const ctxAccuracy = ctxOf(accuracyCanvas, W, H);
  const ctxHome = ctxOf(homeCanvas, W, 240);
  if (!ctxEntropy || !ctxAccuracy || !ctxHome) return;

  let hover = null;

  function agents() {
    return data.tasks[task.value].agents;
  }

  function drawEntropy() {
    const ctx = ctxEntropy;
    ctx.clearRect(0, 0, W, H);
    label(ctx, 'How concentrated is the position belief?', PAD.left, 16, COLOUR.ink, 14);
    const steps = data.protocol.steps;
    const scale = axes(ctx, { xMax: steps, yMin: 0, yMax: 3.3,
      yTicks: [0, 1, 2, 3], yLabel: 'entropy (nats)' });
    for (const [kind, agent] of Object.entries(agents())) {
      drawSeries(ctx, scale, agent.entropy_by_step, agent.colour,
        hover === null ? null : hover === kind);
    }
    label(ctx, 'lower = more concentrated, not necessarily more correct',
      PAD.left, 32, COLOUR.dim, 12);
  }

  function drawAccuracy() {
    const ctx = ctxAccuracy;
    ctx.clearRect(0, 0, W, H);
    label(ctx, 'Is the most likely cell the true cell?', PAD.left, 16, COLOUR.ink, 14);
    const steps = data.protocol.steps;
    const scale = axes(ctx, { xMax: steps, yMin: 0, yMax: 100,
      yTicks: [0, 25, 50, 75, 100], yLabel: 'correct (%)',
      xLabel: 'moves taken' });
    for (const [kind, agent] of Object.entries(agents())) {
      drawSeries(ctx, scale, agent.accuracy_by_step, agent.colour,
        hover === null ? null : hover === kind);
    }
  }

  function drawHome() {
    const ctx = ctxHome;
    const height = 240;
    ctx.clearRect(0, 0, W, height);
    label(ctx, 'How often did it reach the actual home cell?', PAD.left, 16, COLOUR.ink, 14);
    const entries = Object.entries(agents());
    const x0 = PAD.left;
    const y0 = height - 62;
    const top = PAD.top + 6;
    const bandWidth = (W - PAD.right - x0) / entries.length;
    const sy = (v) => y0 - (v / 100) * (y0 - top);
    ctx.strokeStyle = COLOUR.dim;
    ctx.beginPath();
    ctx.moveTo(x0, top);
    ctx.lineTo(x0, y0);
    ctx.lineTo(W - PAD.right, y0);
    ctx.stroke();
    for (const t of [0, 25, 50, 75, 100]) {
      const y = sy(t);
      ctx.strokeStyle = 'rgba(0,0,0,0.08)';
      ctx.beginPath();
      ctx.moveTo(x0, y);
      ctx.lineTo(W - PAD.right, y);
      ctx.stroke();
      label(ctx, String(t), x0 - 8, y + 4, COLOUR.dim, 12, 'right');
    }
    entries.forEach(([kind, agent], i) => {
      const cx = x0 + bandWidth * (i + 0.5);
      const bw = bandWidth * 0.5;
      ctx.globalAlpha = hover === null || hover === kind ? 1 : 0.3;
      ctx.fillStyle = agent.colour;
      ctx.fillRect(cx - bw / 2, sy(agent.home_pct), bw, y0 - sy(agent.home_pct));
      // 95% bootstrap interval over evaluation seeds.
      ctx.strokeStyle = COLOUR.ink;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(cx, sy(agent.home_interval[0]));
      ctx.lineTo(cx, sy(agent.home_interval[1]));
      ctx.moveTo(cx - 5, sy(agent.home_interval[0]));
      ctx.lineTo(cx + 5, sy(agent.home_interval[0]));
      ctx.moveTo(cx - 5, sy(agent.home_interval[1]));
      ctx.lineTo(cx + 5, sy(agent.home_interval[1]));
      ctx.stroke();
      label(ctx, `${agent.home_pct.toFixed(1)}%`, cx, sy(agent.home_interval[1]) - 8,
        COLOUR.ink, 13, 'center');
      const words = (agent.label + (agent.uses_belief ? '' : ' *')).split(' ');
      const lines = words.length > 2
        ? [words.slice(0, words.length - 2).join(' '), words.slice(-2).join(' ')]
        : [words.join(' ')];
      lines.forEach((line, k) => label(ctx, line, cx, y0 + 18 + k * 14, COLOUR.dim, 12, 'center'));
      ctx.globalAlpha = 1;
    });
    label(ctx, '* acts from its latest reading alone; its belief curves above are a reader-only diagnostic',
      x0, height - 8, COLOUR.dim, 12);
  }

  function draw() {
    drawEntropy();
    drawAccuracy();
    drawHome();
    if (hover === null) {
      readout.textContent = 'Hover an agent name in the bars to trace it through both panels above.';
    } else {
      const agent = agents()[hover];
      readout.textContent = `${agent.label}: reached home ${agent.home_pct.toFixed(1)}% ` +
        `(95% interval ${agent.home_interval[0].toFixed(1)} to ${agent.home_interval[1].toFixed(1)}), ` +
        `ending at entropy ${agent.entropy_by_step[agent.entropy_by_step.length - 1].toFixed(2)} nats ` +
        `with the most likely cell correct ${agent.accuracy_by_step[agent.accuracy_by_step.length - 1].toFixed(1)}% ` +
        `of the time and true-cell log loss ${agent.final_log_loss.toFixed(2)}.`;
    }
  }

  // Hover on the bar panel highlights that agent in all three panels, so a
  // reader can follow one agent between "reached home" and "knew where it was".
  ctxHome.canvas.addEventListener('mousemove', (e) => {
    const rect = ctxHome.canvas.getBoundingClientRect();
    const scale = ctxHome.canvas.width / rect.width;
    const mx = (e.clientX - rect.left) * scale;
    const entries = Object.keys(agents());
    const bandWidth = (W - PAD.right - PAD.left) / entries.length;
    const index = Math.floor((mx - PAD.left) / bandWidth);
    const next = index >= 0 && index < entries.length ? entries[index] : null;
    if (next !== hover) { hover = next; draw(); }
  });
  ctxHome.canvas.addEventListener('mouseleave', () => { hover = null; draw(); });
  task.addEventListener('change', () => { hover = null; draw(); });

  draw();
});

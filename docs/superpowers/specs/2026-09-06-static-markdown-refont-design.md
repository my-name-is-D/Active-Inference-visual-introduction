# Refont: static Markdown pages with JS widgets

Date: 2026-09-06

## Why

The tutorial currently ships via `marimo export html-wasm`. Measured at work
package 0: about 20 seconds of Pyodide boot per page load, about 28 MB of
duplicated runtime assets per page, and the `mo.ui.matplotlib` widget needed a
`mo.state` cycle workaround because it has no `on_change`. pymdp was already
ruled out for being JAX-first. For a read-to-understand tutorial the loss of
"reader edits Python cells" is small.

The replacement: each lesson is one static HTML page built from a Markdown
source, with prose and math rendered at build time, non-interactive plots
pre-rendered to SVG by the existing Python, and the few interactions that
matter written as small JS widgets. Hosting stays static on GitHub Pages: no
server, no recurring cost.

A Pyodide-for-math-only option was benchmarked (see Alternatives) and rejected:
about 8 MB over the wire and 8 to 12 seconds before the first widget is usable,
versus instant for a hand-written JS twin of about 80 lines of arithmetic.

## Naming

No `NB` / `NB0` abbreviations anywhere. Every file and page name is explicit,
topic-style, and stands alone describing its content. The first lesson,
currently `nb/NB1_perception.py` titled "Where am I?", becomes
`content/perception-and-belief.md` building to `perception-and-belief.html`.

## File layout

```
content/
  perception-and-belief.md      prose + math + <figure>/<widget> tags
site-src/
  aif.js                        JS twin of aif/; demo() at bottom
  widgets.js                    interactive widgets, one mount per id
  page.css                      shared page styles (from index.html)
  katex/                        vendored KaTeX, no runtime CDN
build.py                        markdown -> html, renders static plots
build.sh                        thin wrapper: exec uv run python build.py "$@"
aif/ worlds/ rendering/         kept: reference impl + static plot rendering
aif/belief_cases.json           written by aif/beliefs.py demo(), read by aif.js demo()
index.html                      landing page, link updated
.github/workflows/deploy.yml    marimo wheel-check step removed; still runs ./build.sh
```

## Components

### build.py

Replaces `build.sh`'s body. Steps:

1. Read each `content/*.md`.
2. Render Markdown to HTML with the `markdown` package and extensions
   `tables`, `fenced_code`, `attr_list`. Build-time dependency only; never
   shipped to the browser. Added to `pyproject.toml` under an optional `build`
   extra.
3. Math left in place: `$...$` and `$$...$$` pass through untouched; KaTeX
   auto-render runs client-side from the vendored `site-src/katex/` files.
4. `<figure src="NAME" ...>` in the Markdown: build.py imports
   `rendering/plots.py`, produces `NAME.svg` into the output directory, and
   rewrites the tag to reference it. The mapping from NAME to a specific
   plot call lives in a small dict in build.py (name -> callable returning a
   matplotlib figure).
5. `<widget id="NAME">` in the Markdown becomes
   `<div class="widget" data-widget="NAME"></div>`. `widgets.js` finds these
   on load and mounts the matching widget.
6. Wrap the rendered body in a page shell carrying the `<head>`, the styling
   currently inline in `index.html` (moved to `site-src/page.css`), and
   `<script>` tags for KaTeX, `aif.js`, `widgets.js`.
7. Write `site/perception-and-belief.html`. Copy `index.html` (with its link
   updated), `site-src/*.css`, `site-src/*.js`, `site-src/katex/`, and write
   `site/.nojekyll`.

The set of pages to build is a list in build.py, as `NOTEBOOKS` is in the
current `build.sh`.

### aif.js

Plain functions mirroring `aif/beliefs.py` and `aif/generative_model.py`.
Arrays are arrays of arrays, not typed arrays (the grid is 25 states; clarity
wins).

```
uniformBelief(n)
update(prior, A, obs)      -> { posterior, likelihood, unnormalised }
predict(belief, B, action)
surprise(prior, A, obs)
observationModel({ rows, cols, sensorNoise })  -> A     shape (n+1) x n
transitionModel({ rows, cols })                -> B     shape n x n x 5
```

`observationModel` and `transitionModel` reproduce the noise/neighbour logic
of `aif/generative_model.py` for the lit-grid, no-slip case that the first
lesson needs. Dark tiles and slip are out of scope until a later lesson needs
them.

A `demo()` at the bottom, run with `node site-src/aif.js`:

- reads `aif/belief_cases.json` and asserts `update` reproduces every
  `posterior` to a tolerance;
- self-checks: `uniformBelief(4)` sums to 1, `surprise` is 0 when the agent
  was certain and it saw what it expected, `predict` conserves mass.

Exits non-zero on any failure, prints `aif.js ok` otherwise.

### aif/belief_cases.json

The cross-check fixture. `aif/beliefs.py`'s `demo()` gains a few lines that
write a JSON list of `{ prior, A, obs, posterior }` objects covering the cases
it already exercises (perfect sensor, uninformative observation, the plan's
worked exercise) plus a noisy 5x5 case. Committed to the repo so `node
aif.js` works without running Python first, and regenerated whenever
`python3 aif/beliefs.py` runs.

### widgets.js

One file. A `mount(id, fn)` registry; on `DOMContentLoaded` it walks
`[data-widget]` divs and calls the registered function with the element.

Three widgets for this lesson:

| id | replaces | interaction |
|---|---|---|
| `sensor-noise-walk` | `mo.ui.slider` + fixed route + "an observation is not a position" plot | range input 0 to 0.6 step 0.05; canvas draws the two "steps from goal" lines for truth vs sensor; redraws on `input` |
| `a-matrix` | the `A` heatmap that tracked the slider | reads the same noise value; canvas heatmap of `A`; text readout of what tile 7's column implies |
| `watch-it-move` | the `mo.state` step-through | previous / next / restart buttons; canvas draws belief-on-grid, one bar per tile, and a ring at the true position for the current step |

`sensor-noise-walk` and `a-matrix` share one noise value: the simplest version
is a single `<input type="range">` owned by `sensor-noise-walk` that
dispatches a custom event the `a-matrix` widget listens for. If that proves
awkward, each widget carries its own slider. Decide during implementation.

Rendering is plain Canvas 2D. The fixed route is
`["right","right","down","down","right","down"]`, matching the current
notebook. Seed-dependent sensor lies: the JS uses a small seeded PRNG
(mulberry32) so the "wrong on N readings" count is stable, as the Python
`seed=4` makes it stable now.

### Static figures (matplotlib SVG at build time)

- "knowing nothing: every tile equally likely" (`draw_belief_pair` on a
  uniform belief)
- "after seeing tile 7 once" (`draw_belief_pair` on the posterior, highlight 7)
- "expected things are cheap, unexpected things are not" (the surprise curve)

The two `mo.accordion` blocks become `<details><summary>` in the Markdown.

## Data flow

Build: `content/*.md` + `rendering/plots.py` --build.py--> `site/*.html` +
assets.

Runtime: page loads; KaTeX renders math from the static HTML; `widgets.js`
mounts widgets; a widget calls `aif.js` functions and draws to a canvas on
every interaction. No network after load. No Python in the browser.

## Error handling

- build.py fails loudly (non-zero exit) if a `content/*.md` references a
  `<figure src>` with no entry in the plot dict, or a `<widget id>` with no
  registration in `widgets.js` (checked by a simple grep of widgets.js).
- `aif.js` `update` throws on a zero-evidence observation, matching the Python
  `ValueError`.
- Widgets guard against a missing canvas context and log to console rather
  than throwing, so one broken widget does not blank the page.

## Testing

- `python3 aif/beliefs.py`, `python3 aif/generative_model.py`,
  `python3 worlds/gridworld.py`, `python3 rendering/plots.py`: unchanged
  module self-checks. `beliefs.py` also regenerates `belief_cases.json`.
- `node site-src/aif.js`: JS twin against the fixture and self-checks.
- `./build.sh && python3 -m http.server -d site`: load
  `perception-and-belief.html`, confirm math renders, each widget responds,
  browser console is clean.

## Files deleted

- `nb/` entire directory (`NB1_perception.py`).
- `pyproject.toml`: the `[tool.marimo.runtime]` block, the
  `[tool.hatch.build.targets.wheel]` block (existed only to feed marimo), and
  `marimo` from `dependencies`.
- `build.sh` body replaced by the one-line wrapper.
- README sections specific to marimo, WASM, Pyodide, `mo.ui.matplotlib`, and
  asset sharing between pages: rewritten for the static build.
- `.github/workflows/deploy.yml`: the "Check local wheels were bundled" step
  (marimo-specific) removed; the stale `setup-uv` comment corrected. `uv` is
  still used to run `build.py`. A `node site-src/aif.js` check step is added.

Kept: `aif/`, `worlds/`, `rendering/`, `index.html`, `LICENSE`,
`.github/workflows/deploy.yml`, `.gitignore`, `uv.lock` (regenerated).

## Out of scope

- Later lessons (imagination, expected free energy, novelty, preferences,
  when-not-to-use). Each reuses `aif.js`, `widgets.js`, and the build; each is
  its own `content/*.md` added later.
- Dark-grid and slip logic in `aif.js` (added when a lesson needs them).
- Syntax highlighting of code blocks (the `markdown` fenced_code extension
  emits the right classes; a highlighter can be vendored later).

## Alternatives considered

- **Keep marimo.** Rejected: the 20-second load is the problem this refont
  exists to solve.
- **Pyodide for math only.** Benchmarked (Node/V8, median of 3): init about
  4.0 s, `loadPackage("numpy")` about 1.0 s, first import about 1.1 s, so
  about 6 s cold to ready and about 8 to 12 s in a browser; about 8.4 MB
  gzipped over the wire. One interaction is about 10 ms. Rejected: it spends
  half the load budget the refont is trying to reclaim, to avoid rewriting
  about 80 lines of arithmetic.
- **Precompute all widget states at build time to JSON.** Works only for
  discrete pre-enumerable parameters. Fits this lesson but not a lesson with
  a continuous control, so it does not generalise.
- **Stdlib-only Markdown renderer.** Rejected: owning a Markdown parser is the
  over-engineering here. `markdown` is pure Python, build-time only, never
  reaches the reader.

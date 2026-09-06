# Active inference, by running it

An interactive tutorial on active inference, aimed at readers who find the
mathematics off-putting, are unconvinced the theory buys them anything, or are
unsure how to use it concretely. The organising question throughout is **what
does this give me that a reward function does not?**

The notebooks run Python in the browser through WebAssembly, so there is
nothing to install to read them.

## Status

Work package 0 is complete: the deployment pipeline works and its assumptions
are verified. No teaching content exists yet.

## Running locally

```sh
uv run marimo edit nb/NB0_smoke.py   # edit a notebook
./build.sh site                      # build the static site
python3 -m http.server --directory site
```

The exported site must be served over HTTP. Opening `index.html` from the
filesystem will not work.

## How the build works

Each notebook is exported to its own directory under `site/`, so opening one
page does not execute the others. `build.sh` is the single source of truth and
CI runs the same script.

**The one thing that is easy to get wrong:** marimo bundles the local packages
a notebook imports into wheels for the browser, and it finds them through
`[tool.marimo.runtime] pythonpath` in `pyproject.toml`. Notebooks live in
`nb/` while the packages live at the repository root, so without that setting
the export still succeeds and the import then fails in the reader's browser
with no build error. The deploy workflow fails the build if any notebook
export is missing its wheel, because the failure is otherwise invisible until
a reader hits it.

`build.sh` also deletes `CLAUDE.md` and similar files from the export, since
marimo copies files it finds beside the notebook into the output.

## Measurements from the deployment spike

Measured with headless Chromium against the built site on a local server, so
these are lower bounds: a real reader over a real connection will be slower.

| Measurement | Result |
|---|---|
| Cold load to first output | 22.5 s |
| Cold load to full execution | 29.5 s |
| Second page, warm cache | 20.5 s |
| Site size, two notebooks | 57 MB |
| Runtime in browser | Python 3.14.2, numpy 2.4.3, emscripten |

**Interaction idioms**, all confirmed working in the export:

| Idiom | Result |
|---|---|
| `mo.ui.slider` driving a recomputed figure | works |
| `mo.ui.dropdown` selection | works |
| `mo.accordion` collapsible depth section | works |
| `mo.ui.matplotlib` selection on a figure | works |

A plain `plt.figure` renders as a static image and cannot be clicked. Wrapping
an `Axes` in `mo.ui.matplotlib` renders it to a canvas instead and returns the
dragged region in data coordinates, so the reader can pick a tile on the grid
directly. A drag over the tile at row 3, column 1 returned `x_min=0.62,
x_max=1.18, y_min=2.94, y_max=3.50`, whose midpoint rounds to that tile.

The selection is a box rather than a point, so a single tile is picked by
rounding the midpoint. Desktop toolkits such as tkinter are not an option:
they need an OS window, which does not exist in the browser sandbox, and they
are not shipped in Pyodide.

**On sharing assets between pages.** Every notebook export contains its own
identical 28 MB copy of the marimo and Pyodide assets. Loading a second page
with a warm cache took 20.5 s against 23.5 s cold, so the browser cache is
doing less than hoped and most of the cost is Pyodide starting up rather than
downloading. Deduplicating the asset directories would cut the site size
roughly in half but would not make the second page meaningfully faster.

The practical consequence is that **each page load costs the reader about
twenty seconds**, which argues for fewer, longer notebooks rather than many
short ones, and for the landing page saying so plainly.

## Relationship to pymdp

[pymdp](https://github.com/infer-actively/pymdp) is the reference
implementation for active inference in discrete state spaces, and the right
choice for real work. Nothing here depends on it. Everything is re-derived in
plain numpy because the purpose is comprehension rather than deployment, and
because pymdp 1.0.0 is JAX-first, which cannot run in the browser.

## Conventions

`A`, `B`, `C` and `D` keep their conventional names, because every paper and
pymdp use them, and a glossary closes the gap. Everything else gets full
words: `expected_free_energy` rather than `efe`. The asymmetry is deliberate.

`C` is defined in log space, for example `[0, -2]`, rather than as a
normalised distribution, which keeps the hand-computation exercises feasible.
Papers vary, and a distribution is recovered by exponentiating and
normalising.

## Licence

Apache License 2.0. See [LICENSE](LICENSE).

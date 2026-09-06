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

Local figures come from headless Chromium against the built site on a local
server. The deployed figure was measured by hand on the published page.

| Measurement | Result |
|---|---|
| Cold load, deployed page | about 30 s |
| Cold load to first output, local | 22.5 s |
| Cold load to full execution, local | 29.5 s |
| Second page, warm cache, local | 20.5 s |
| Site size, two notebooks | 57 MB |
| Runtime in browser | Python 3.14.2, numpy 2.4.3, emscripten |

**Interaction idioms**, all confirmed working on the deployed page:

| Idiom | Result |
|---|---|
| `mo.ui.slider` driving a recomputed figure | works |
| `mo.accordion` collapsible depth section | works |
| `mo.ui.matplotlib` picking a tile on a figure | works |

A plain `plt.figure` renders as a static image and cannot be clicked. Wrapping
an `Axes` in `mo.ui.matplotlib` returns the selected region in data
coordinates, so the reader can pick a tile on the grid directly.

Three things about that widget are worth knowing before building on it.

- **It reads as a click, not a drag.** The widget is a box selector, so a
  click yields a box of zero area, and press-and-drag does not track the
  pointer the way a drag normally would. Tell the reader to click.
- **Pass `debounce=True`.** The default streams the value during the
  interaction and passes through the empty selection, which makes any output
  downstream flicker away mid-click.
- **The empty value is an `EmptySelection` object, not `None`.** Test it with
  a plain truth check. A shift-drag gives a `LassoSelection` carrying
  `vertices` rather than a box, so code that assumes a box will raise.

Tile geometry needs care. With `extent=(0, 5, 5, 0)` a tile `(r, c)` covers
`x` in `[c, c+1]` and `y` in `[r, r+1]`, so a coordinate maps to a tile by
taking its floor, and ticks belong at the tile centres. Putting tile centres
on the integers instead makes the ticks label the boundaries between tiles,
and the grid then looks misaligned with what the reader is clicking.

Desktop toolkits such as tkinter are not an option: they need an OS window,
which does not exist in the browser sandbox, and they are not shipped in
Pyodide.

**On sharing assets between pages.** Every notebook export contains its own
identical 28 MB copy of the marimo and Pyodide assets. Loading a second page
with a warm cache took 20.5 s against 23.5 s cold, so the browser cache is
doing less than hoped and most of the cost is Pyodide starting up rather than
downloading. Deduplicating the asset directories would cut the site size
roughly in half but would not make the second page meaningfully faster.

The practical consequence is that **each page load costs the reader about
twenty seconds**, which argues for fewer, longer notebooks rather than many
short ones, and for the landing page saying so plainly.

**One figure, updated in place.** When the reader interacts with a figure,
that figure must change. The smoke notebook currently lights the picked tile
by drawing a second grid below the pickable one, which leaves two grids on
screen showing different states. That is a limitation of the spike, not a
pattern to copy: the teaching notebooks must update the figure the reader
clicked.

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

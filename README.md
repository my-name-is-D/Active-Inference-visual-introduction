# Active inference, by running it

An interactive tutorial on active inference, aimed at readers who find the
mathematics off-putting, are unconvinced the theory buys them anything, or are
unsure how to use it concretely. The organising question throughout is **what
does this give me that a reward function does not?**

Each lesson is one static HTML page built from a Markdown source. The prose and
maths are rendered at build time; the interactive parts are small JavaScript
widgets that draw to a canvas. Nothing runs in the browser except those
widgets, so the pages load immediately.

## Status

Work package 0 (the deployment pipeline) and the first three lessons,
`content/lesson-1.md` through `content/lesson-3.md`, are done. The remaining
lessons are listed on the landing page.

## Running locally

```sh
$EDITOR content/lesson-3.md                # edit a lesson
$EDITOR site-src/lesson-3.js               # edit that lesson's widgets
$EDITOR site-src/widget.js                 # edit the shared widget layer
./build.sh site                            # build the static site
python3 -m http.server --directory site
```

The exported site must be served over HTTP. Opening `index.html` from the
filesystem will not work (the pages load `aif.js` as an ES module).

## How the build works

`build.py` is the whole build, and `build.sh` is a one-line wrapper so CI runs
the same thing. For each entry in `PAGES` it:

- renders the Markdown to HTML with the `markdown` package;
- replaces `<figure src="NAME">` with an `<img>` to a `NAME.svg` it generates
  by calling into `rendering/plots.py`, so the non-interactive figures are
  real matplotlib output;
- turns `<widget id="NAME">` into a mount point that `site-src/widgets.js`
  finds and fills;
- wraps the result in a page shell that loads `page.css`, vendored KaTeX,
  `aif.js`, and `widgets.js`.

`markdown` is a build-time dependency only, under the `build` extra in
`pyproject.toml`. It never reaches the reader.

## The JavaScript twin

`site-src/aif.js` reimplements the belief maths from `aif/` (the update,
prediction, surprise, and the `A` and `B` models) so the widgets can run it in
the browser. The Python in `aif/` stays the tested reference.

`aif/beliefs.py`'s `demo()` writes `aif/belief_cases.json`, a set of
`(prior, A, observation)` cases with the posterior the Python produces.
`node site-src/aif.js` asserts the JavaScript reproduces every one. CI runs
both, so the two implementations cannot drift apart silently.

## Relationship to pymdp

[pymdp](https://github.com/infer-actively/pymdp) is the reference
implementation for active inference in discrete state spaces, and the right
choice for real work. Nothing here depends on it. Everything is re-derived in
plain numpy because the purpose is comprehension rather than deployment, and
nothing here runs in the browser anyway.

## Conventions

`A`, `B`, `C` and `D` keep their conventional names, because every paper and
pymdp use them, and a glossary closes the gap. Everything else gets full
words: `expected_free_energy` rather than `efe`. The asymmetry is deliberate.

`C` is defined in log space, for example `[0, -2]`, rather than as a
normalised distribution, which keeps the hand-computation exercises feasible.
Papers vary, and a distribution is recovered by exponentiating and
normalising.

The figures print numbers on anything that carries a claim, so no argument
depends on colour alone, and use perceptually uniform colourmaps.

When the reader interacts with a figure, that figure changes in place; a second
figure never appears below it to show the result.

## Licence

Apache License 2.0. See [LICENSE](LICENSE).

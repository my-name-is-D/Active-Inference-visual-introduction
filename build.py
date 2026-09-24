#!/usr/bin/env python3
"""Build the static site into site/.

Each module is one Markdown file under content/. This renders it to a
standalone HTML page: prose and math from Markdown, non-interactive figures
pre-rendered to SVG by rendering/plots.py, and <widget> tags left as mount
points for site-src/widgets.js. Nothing runs Python in the browser.
"""

import re
import shutil
import sys
import hashlib
from pathlib import Path

import markdown

ROOT = Path(__file__).parent
CONTENT = ROOT / "content"
SITE_SRC = ROOT / "site-src"
INDEX_PAGE = CONTENT / "index_page.md"

# (slug, markdown path). The slug is the output file name and the URL.
PAGES = [
    ("module-1", CONTENT / "module-1.md"),
    ("module-2", CONTENT / "module-2.md"),
    ("module-3", CONTENT / "module-3.md"),
    ("module-4", CONTENT / "module-4.md"),
    ("module-5", CONTENT / "module-5.md"),
]


def _fig_knowing_nothing():
    sys.path.insert(0, str(ROOT))
    from aif.beliefs import uniform_belief
    from rendering.plots import draw_belief_pair

    flat = uniform_belief(25)
    fig, _ = draw_belief_pair(
        flat, 5, 5, title="knowing nothing: every cell equally likely"
    )
    return fig


def _fig_after_seeing_cell_7():
    sys.path.insert(0, str(ROOT))
    from aif.beliefs import uniform_belief, update
    from aif.generative_model import observation_model
    from rendering.plots import draw_belief_pair
    from worlds.gridworld import GridWorld

    world = GridWorld(size=(5, 5), sensor_noise=0.3)
    A = observation_model(world)
    posterior, _, _ = update(uniform_belief(25), A, 7)
    fig, _ = draw_belief_pair(
        posterior, 5, 5, title="after seeing cell 7 once", highlight=7
    )
    return fig


def _fig_surprise_curve():
    import matplotlib.pyplot as plt
    import numpy as np

    probabilities = np.linspace(0.01, 1.0, 200)
    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    ax.plot(probabilities, -np.log(probabilities), linewidth=2)
    ax.set_xlabel("probability the agent gave to what it saw")
    ax.set_ylabel("surprise (nats)")
    ax.set_title("expected things are cheap, unexpected things are not")
    return fig


# <figure src="NAME"> maps to a function returning a matplotlib Figure.
FIGURES = {
    "knowing-nothing": _fig_knowing_nothing,
    "after-seeing-cell-7": _fig_after_seeing_cell_7,
    "surprise-curve": _fig_surprise_curve,
}


MATH_BLOCK = re.compile(r"\$\$.+?\$\$|\$[^$\n]+?\$", re.DOTALL)


def render_markdown(text):
    """Markdown to an HTML fragment, leaving $...$ and HTML tags alone.

    Math is pulled out first and put back afterwards. Markdown treats a
    backslash as an escape, which silently turns the row separator in
    \begin{bmatrix} a \\ b \end{bmatrix} into a single backslash and
    collapses the matrix to one row.
    """
    spans = []

    def stash(match):
        spans.append(match.group(0))
        return f"\x00MATH{len(spans) - 1}\x00"

    text = MATH_BLOCK.sub(stash, text)
    md = markdown.Markdown(extensions=["tables", "fenced_code", "attr_list"])
    html = md.convert(text)
    for i, span in enumerate(spans):
        html = html.replace(f"\x00MATH{i}\x00", span)
    return html


def first_heading(text):
    match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else "Active inference"


def render_figures(html, out_dir):
    """Replace <figure src="NAME"> with an <img> to a generated SVG."""

    def replace(match):
        name = match.group(1)
        if name not in FIGURES:
            raise SystemExit(f"unknown figure: {name!r} (not in FIGURES)")
        import matplotlib

        matplotlib.use("Agg")
        fig = FIGURES[name]()
        fig.savefig(out_dir / f"{name}.svg", format="svg", bbox_inches="tight")
        return f'<img src="{name}.svg" alt="{name.replace("-", " ")}">'

    return re.sub(r'<figure\s+src="([^"]+)"\s*>\s*</figure>', replace, html)


def render_widgets(html):
    """Turn <widget id="NAME"> into a div widgets.js can find."""
    html = re.sub(
        r'<widget\s+id="([^"]+)"\s*>\s*</widget>',
        r'<div class="widget" data-widget="\1"></div>',
        html,
    )
    # Markdown wraps a lone block tag in <p>...</p>, which is invalid around a
    # <div>. Unwrap the widget and figure divs.
    return re.sub(
        r"<p>(\s*<div class=\"widget\"[^>]*></div>\s*|\s*<img [^>]*>\s*)</p>",
        r"\1",
        html,
    )


PAGE_SHELL = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="page.css?v={version}">
<link rel="stylesheet" href="katex/katex.min.css">
</head>
<body>
<main>
{body}
</main>
<script defer src="katex/katex.min.js"></script>
<script defer src="katex/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body, {{
    delimiters: [
      {{left: '$$', right: '$$', display: true}},
      {{left: '$', right: '$', display: false}}
    ]
  }})"></script>
<script type="module" src="aif.js?v={version}"></script>
<script type="module" src="widget.js?v={version}"></script>
<script type="module" src="{slug}.js?v={version}"></script>
</body>
</html>
"""


def asset_version(slug):
    assets = [SITE_SRC / "page.css", SITE_SRC / "aif.js",
              SITE_SRC / "widget.js", SITE_SRC / f"{slug}.js"]
    if slug == "module-4":
        assets.append(SITE_SRC / "agent-comparison.js")
    if slug == "module-5":
        assets.extend([SITE_SRC / "module-5-data.js",
                       SITE_SRC / "module-5-b-data.js",
                       SITE_SRC / "module-5-joint-data.js"])
    digest = hashlib.sha256()
    for asset in assets:
        digest.update(asset.read_bytes())
    return digest.hexdigest()[:12]


def module_nav(i):
    """Previous / menu / next links, in PAGES order."""

    def link(j, arrow_left):
        slug, md_path = PAGES[j]
        name = re.sub(r"^Module \d+\.\s*", "", first_heading(md_path.read_text()))
        label = f"Module {j + 1}. {name}"
        text = f"&larr; {label}" if arrow_left else f"{label} &rarr;"
        return f'<a href="./{slug}.html">{text}</a>'

    prev = link(i - 1, True) if i > 0 else "<span></span>"
    nxt = link(i + 1, False) if i + 1 < len(PAGES) else "<span></span>"
    return (f'<nav class="module-nav">{prev}'
            f'<a href="./index.html">Back to menu</a>{nxt}</nav>')


def build_page(slug, md_path, out_dir):
    text = md_path.read_text()
    html = render_markdown(text)
    html = render_figures(html, out_dir)
    html = render_widgets(html)
    html += "\n" + module_nav([s for s, _ in PAGES].index(slug))
    version = asset_version(slug)
    page = PAGE_SHELL.format(title=first_heading(text), body=html, slug=slug,
                             version=version)
    (out_dir / f"{slug}.html").write_text(page)


def build_index(out_dir):
    """Render the Markdown introduction into the existing homepage shell."""
    # "## Going further" goes after the template's module list, the rest before it.
    intro, sep, further = INDEX_PAGE.read_text().partition("\n## Going further")
    body, tail = (render_widgets(render_figures(render_markdown(t), out_dir))
                  for t in (intro, sep + further))
    template = (ROOT / "index.html").read_text()
    body_start = template.index("<main>") + len("<main>")
    rest_start = template.index("<h2>What this is not</h2>")
    footer = template.index("<footer>")
    page = (template[:body_start] + "\n" + body + "\n\n  " + template[rest_start:footer]
            + tail + "\n\n  " + template[footer:])
    (out_dir / "index.html").write_text(page)


def copy_assets(out_dir):
    names = ["page.css", "aif.js", "widget.js", "agent-comparison.js",
             "agent-comparison-data.js", "figure6-curves.js",
             "figure6-curves-data.js", "module-5-data.js",
             "module-5-b-data.js", "module-5-joint-data.js"] + [f"{slug}.js" for slug, _ in PAGES]
    for name in names:
        shutil.copy(SITE_SRC / name, out_dir / name)
    module_4_js = out_dir / "module-4.js"
    comparison_version = hashlib.sha256(
        (SITE_SRC / "agent-comparison.js").read_bytes()
    ).hexdigest()[:12]
    module_4_js.write_text(module_4_js.read_text().replace(
        "./agent-comparison.js", f"./agent-comparison.js?v={comparison_version}"
    ))
    module_5_js = out_dir / "module-5.js"
    data_version = hashlib.sha256(
        (SITE_SRC / "module-5-data.js").read_bytes()
    ).hexdigest()[:12]
    module_5_js.write_text(module_5_js.read_text().replace(
        "./module-5-data.js", f"./module-5-data.js?v={data_version}"
    ))
    b_data_version = hashlib.sha256(
        (SITE_SRC / "module-5-b-data.js").read_bytes()
    ).hexdigest()[:12]
    module_5_js.write_text(module_5_js.read_text().replace(
        "./module-5-b-data.js", f"./module-5-b-data.js?v={b_data_version}"
    ))
    joint_data_version = hashlib.sha256(
        (SITE_SRC / "module-5-joint-data.js").read_bytes()
    ).hexdigest()[:12]
    module_5_js.write_text(module_5_js.read_text().replace(
        "./module-5-joint-data.js", f"./module-5-joint-data.js?v={joint_data_version}"
    ))
    katex_src = SITE_SRC / "katex"
    if katex_src.exists():
        shutil.copytree(katex_src, out_dir / "katex")
    (out_dir / ".nojekyll").touch()


def main(out_dir="site"):
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for slug, md_path in PAGES:
        print(f"--- building {md_path.name} -> {out}/{slug}.html")
        build_page(slug, md_path, out)
    print(f"--- building {INDEX_PAGE.name} -> {out}/index.html")
    build_index(out)
    copy_assets(out)
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"--- built {out} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(*sys.argv[1:])

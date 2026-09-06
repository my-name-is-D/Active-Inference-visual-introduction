#!/usr/bin/env bash
# Build the static site into site/.
#
# Every notebook becomes its own page, so opening one does not execute the
# others. marimo bundles the local packages a notebook imports into wheels,
# which is why [tool.marimo.runtime] pythonpath must list the repository root:
# without it the export still succeeds and the browser import then fails.
set -euo pipefail

OUT=${1:-site}
rm -rf "$OUT"
mkdir -p "$OUT"

cp index.html "$OUT/"
touch "$OUT/.nojekyll"

# name:path pairs. The name is the URL directory under the site root.
NOTEBOOKS=(
  "smoke:nb/NB0_smoke.py"
  "smoke-second:nb/NB0_smoke_second.py"
)

for entry in "${NOTEBOOKS[@]}"; do
  name="${entry%%:*}"
  path="${entry#*:}"
  echo "--- exporting $path -> $OUT/$name"
  uv run marimo export html-wasm "$path" -o "$OUT/$name" --mode run
  # marimo copies agent instruction files it finds beside the notebook into the
  # export. Those are internal and must not be published.
  rm -f "$OUT/$name"/CLAUDE.md "$OUT/$name"/AGENTS.md "$OUT/$name"/HANDOFF.md
done

echo "--- built $OUT ($(du -sh "$OUT" | cut -f1))"

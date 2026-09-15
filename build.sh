#!/usr/bin/env bash
# Build the static site into site/ (or the directory given as $1).
set -euo pipefail
exec uv run --extra build python build.py "$@"

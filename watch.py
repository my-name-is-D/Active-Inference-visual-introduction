#!/usr/bin/env python3
"""Rebuild the site automatically whenever site-src/ or content/ changes.

No new dependency: polls mtimes every 0.5s rather than using an OS file-watch
API. Good enough for hand-editing a handful of files; not meant for CI.

    python3 watch.py

Then edit site-src/*.js or content/*.md and refresh the browser; no need to
run build.py by hand after each change.
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
WATCH_DIRS = [ROOT / "site-src", ROOT / "content", ROOT / "index.html"]


def snapshot():
    state = {}
    for target in WATCH_DIRS:
        paths = [target] if target.is_file() else target.rglob("*")
        for p in paths:
            if p.is_file():
                state[p] = p.stat().st_mtime
    return state


def build():
    result = subprocess.run([sys.executable, "build.py"], cwd=ROOT)
    return result.returncode == 0


def main():
    print("--- watching site-src/ and content/ for changes (Ctrl+C to stop)")
    build()
    last = snapshot()
    try:
        while True:
            time.sleep(0.5)
            current = snapshot()
            if current != last:
                changed = sorted(set(current) ^ set(last)) or sorted(
                    p for p in current if current[p] != last.get(p)
                )
                names = ", ".join(p.name for p in changed[:5])
                print(f"--- change detected ({names}), rebuilding")
                build()
                last = current
    except KeyboardInterrupt:
        print("\n--- stopped")


if __name__ == "__main__":
    main()

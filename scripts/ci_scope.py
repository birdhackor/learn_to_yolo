"""Run Linux for prose changes; include Windows/macOS for runtime changes or an unknown diff."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

try:
    from .course_execution import code_signature
except ImportError:
    from course_execution import code_signature


def runtime_changed(names, notebook_changed):
    for name in names:
        if name.endswith(".ipynb"):
            if notebook_changed(name):
                return True
        elif (name.endswith((".py", ".yml", ".yaml", ".lock", ".in", ".ini"))
              or name.startswith("requirements-") or name in {".python-version", ".gitattributes", "section-map.json"}
              or name.startswith(("miniyolo/", "lesson_cases/", "data/", "docs/assets/", "artifacts/checks/"))):
            return True
    return False


def main():
    full = os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch"
    try:
        base = os.environ.get("BASE_SHA") or "HEAD^"
        names = subprocess.check_output(["git", "diff", "--name-only", "-z", base, "HEAD"]).decode().split("\0")

        def changed(name):
            raw = subprocess.check_output(["git", "show", f"{base}:{name}"], stderr=subprocess.DEVNULL)
            return code_signature(json.loads(raw)) != code_signature(json.loads(Path(name).read_bytes()))

        full = full or runtime_changed(names, changed)
    except (subprocess.CalledProcessError, OSError, ValueError, KeyError, TypeError):
        full = True
    matrix = [{"os": "ubuntu-latest"}]
    if full:
        matrix += [{"os": "windows-latest"}, {"os": "macos-latest"}]
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as stream:
        stream.write("matrix=" + json.dumps(matrix) + "\n")
        stream.write("full=" + str(full).lower() + "\n")
    print(json.dumps(matrix))


if __name__ == "__main__":
    main()

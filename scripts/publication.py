"""Write public build metadata only after verifying the complete CPU execution bundle."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import tomllib

try:
    from .course_execution import ROOT, lessons, validate_bundle
except ImportError:
    from course_execution import ROOT, lessons, validate_bundle


def build_info(root: Path, output: Path, revision: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Publication requires the full Git commit SHA")
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if revision != actual:
        raise ValueError("Publication revision differs from the checked-out commit")
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"],
                                    cwd=root, text=True).strip()
    if dirty:
        raise ValueError("Commit the publication sources first; the working tree has uncommitted changes")
    receipt = validate_bundle(root, output)
    return {"revision": revision, "lesson_count": len(lessons(root)), "publication_ready": True,
            "execution_key": receipt["key"], "execution_revision": receipt["executed_revision"],
            "execution_date": receipt["executed_at"], "execution_scope": "CPU notebook experiment cells",
            "builder": "zensical", "builder_version": importlib.metadata.version("zensical")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--execution", type=Path, default=ROOT / "outputs/lesson-execution")
    args = parser.parse_args()
    info = build_info(ROOT, args.execution, args.revision)
    config = tomllib.loads((ROOT / "zensical.toml").read_text())["project"]
    site = (ROOT / config["site_dir"]).resolve()
    if not site.is_relative_to(ROOT) or site == ROOT or not (site / "index.html").is_file():
        raise ValueError("Build the site into a repository subdirectory first")
    (site / "build-info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"Publication metadata: {info['revision']}; execution: {info['execution_revision']}")


if __name__ == "__main__":
    main()

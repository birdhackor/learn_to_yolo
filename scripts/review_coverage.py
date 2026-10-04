"""Bind each review in reviews/ to the exact page text, figures and code it covered.

    python scripts/review_coverage.py                         # list pages whose review no longer matches
    python scripts/review_coverage.py --write PAGE [PAGE ...]  # after reviewing PAGE: record what was covered

A review covers a page's text, the SVG figures the page shows, and, for a lesson page, the lesson
program with every repository module it imports. The digest leaves out the generated execution-record
block and the release tag in Colab links, so re-rendering records or pinning a new tag keeps a review
valid; any other change to what the reader sees or to the code the page describes needs a new review.
Runs without PyTorch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import tomllib

from evidence_records import provenance

ROOT = Path(__file__).resolve().parents[1]
COVERAGE = ROOT / "reviews/coverage.json"
MARKER = "\n<!-- curriculum-evidence:start -->"
COLAB_TAG = re.compile(r"(colab\.research\.google\.com/github/birdhackor/learn_to_yolo/blob/)[^/]+/")


def site_pages() -> list[Path]:
    """Every page in the site navigation, in reading order."""
    def walk(node):
        if isinstance(node, str):
            yield node
        elif isinstance(node, dict):
            for value in node.values():
                yield from walk(value)
        elif isinstance(node, list):
            for item in node:
                yield from walk(item)
    config = tomllib.loads((ROOT / "zensical.toml").read_text())["project"]
    return [ROOT / config["docs_dir"] / page for page in walk(config["nav"])]


def review_file(page: Path) -> Path:
    """reviews/<lesson id>.md for a lesson page, reviews/<path with - for />.md for the others."""
    relative = page.relative_to(ROOT / "docs").with_suffix("")
    name = relative.name if relative.parts[0] == "lessons" else "-".join(relative.parts)
    return ROOT / "reviews" / f"{name}.md"


def digest(page: Path) -> dict[str, str]:
    """SHA-256 of what a review of this page covers."""
    text = COLAB_TAG.sub(r"\1<tag>/", page.read_text().split(MARKER)[0])
    covered = {page.relative_to(ROOT).as_posix(): hashlib.sha256(text.encode()).hexdigest()}
    for figure in sorted(set(re.findall(r"\]\(([^)\s]+\.svg)\)", text))):
        path = (page.parent / figure).resolve()
        covered[path.relative_to(ROOT).as_posix()] = provenance.file_sha256(path)
    case = ROOT / "lesson_cases" / f"{page.stem}.py"
    if page.parent.name == "lessons" and case.is_file():
        covered.update(provenance.repo_dependencies(case))
    return covered


def stale() -> list[str]:
    """Pages without a review, or changed (text, figure or code) since their review."""
    recorded = json.loads(COVERAGE.read_text()) if COVERAGE.is_file() else {}
    problems = []
    for page in site_pages():
        name = page.relative_to(ROOT).as_posix()
        entry = recorded.get(name)
        review = review_file(page)
        if entry is None or not review.is_file():
            problems.append(f"{name}: no review")
        elif entry["review"] != review.relative_to(ROOT).as_posix() \
                or entry["review_sha256"] != provenance.file_sha256(review):
            problems.append(f"{name}: review file changed or moved after it was recorded")
        else:
            now = digest(page)
            changed = sorted(k for k in set(entry["covered"]) | set(now) if entry["covered"].get(k) != now.get(k))
            if changed:
                problems.append(f"{name}: changed since its review: {', '.join(changed)}")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", nargs="+", type=Path, metavar="PAGE")
    args = parser.parse_args()
    if args.write:
        recorded = json.loads(COVERAGE.read_text()) if COVERAGE.is_file() else {}
        for page in args.write:
            page = page.resolve()
            review = review_file(page)
            assert review.is_file(), f"write the review first: {review.relative_to(ROOT)}"
            recorded[page.relative_to(ROOT).as_posix()] = {
                "review": review.relative_to(ROOT).as_posix(),
                "review_sha256": provenance.file_sha256(review), "covered": digest(page)}
        COVERAGE.write_text(json.dumps(dict(sorted(recorded.items())), ensure_ascii=False, indent=1) + "\n")
    problems = stale()
    print("\n".join(problems) or f"All {len(site_pages())} pages match their reviews.")
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()

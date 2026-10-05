"""Validate lesson pages, their code excerpts, fixed-version notebooks, review coverage and SVG assets."""

import ast
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from review_coverage import stale as stale_reviews

ROOT = Path(__file__).resolve().parents[1]
# A code block marked ``` { .python data-excerpt="lesson_cases/<id>.py" } quotes that file verbatim.
EXCERPT = re.compile(r'^( *)``` *\{ *\.python +data-excerpt="([^"]+)" *\}\n(.*?)^\1```', re.S | re.M)
PYTHON_BLOCK = re.compile(r'^( *)```(?:python|py)\b[^\n]*\n(.*?)^\1```', re.S | re.M)
SOURCES = sorted([*ROOT.glob("lesson_cases/*.py"), *ROOT.glob("miniyolo/*.py"), *ROOT.glob("scripts/*.py")])


def code_lines(source: str) -> list[str]:
    """Code without comments, blank lines or '...' elisions, whitespace collapsed."""
    lines = []
    for line in source.splitlines():
        quote, index = None, 0
        while index < len(line):
            if quote:
                if line[index] == "\\":
                    index += 1
                elif line.startswith(quote, index):
                    index += len(quote) - 1
                    quote = None
            elif line.startswith(('"""', "'''"), index):
                quote = line[index:index + 3]
                index += 2
            elif line[index] in "\"'":
                quote = line[index]
            elif line[index] == "#":
                line = line[:index]
                break
            index += 1
        code = " ".join(line.split())
        if code and code != "...":
            lines.append(code)
    return lines


SOURCE_LINES = {path: set(code_lines(path.read_text())) for path in SOURCES}


def excerpt_problems(page: Path) -> list[str]:
    text = page.read_text()
    problems = []
    for match in EXCERPT.finditer(text):
        line = text[:match.start()].count("\n") + 1
        source = ROOT / match.group(2)
        if source not in SOURCE_LINES:
            problems.append(f"{page.name}:{line}: excerpt names {match.group(2)}, not a lesson_cases/miniyolo/scripts file")
            continue
        missing = [code for code in code_lines(match.group(3)) if code not in SOURCE_LINES[source]]
        problems += [f"{page.name}:{line}: not in {match.group(2)}: {code}" for code in missing]
    for match in PYTHON_BLOCK.finditer(text):
        code = code_lines(match.group(2))
        verbatim = [path for path, lines in SOURCE_LINES.items() if len(code) >= 3 and set(code) <= lines]
        if verbatim:
            line = text[:match.start()].count("\n") + 1
            problems.append(f"{page.name}:{line}: copied verbatim from {verbatim[0].relative_to(ROOT)}; "
                            "mark it with data-excerpt so a later change to that file is caught")
    return problems


sections = json.loads((ROOT / "section-map.json").read_text())["sections"]
lessons = [s for s in sections if s["kind"] == "lesson"]
assert lessons and len({s["id"] for s in lessons}) == len(lessons)
assert {s["id"] for s in lessons} == {p.stem for p in (ROOT / "lesson_cases").glob("*.py")}
excerpts = []
for section in lessons:
    lesson_id = section["id"]
    assert section["status"] == "ready", lesson_id
    page = ROOT / section["page"]
    text = page.read_text()
    assert "{{COLAB:" not in text, f"Unresolved link: {lesson_id}"
    assert f"/blob/{section['source_ref']}/notebooks/{lesson_id}.ipynb" in text
    assert len(text) >= 800, f"Lesson appears incomplete: {lesson_id}"
    notebook = json.loads((ROOT / section["notebook"]).read_text())
    assert notebook["metadata"]["source_ref"] == section["source_ref"]
    code = "".join(notebook["cells"][-1]["source"])
    assert code == (ROOT / "lesson_cases" / f"{lesson_id}.py").read_text()
    ast.parse(code)
    for diagram in re.findall(r"!?\[[^\]]*\]\(([^)]+\.svg)\)", text):
        assert (page.parent / diagram).resolve().is_file(), f"Missing diagram: {diagram}"
    excerpts += excerpt_problems(page)
assert not excerpts, "Page code excerpts differ from the code they quote:\n" + "\n".join(excerpts)
reviews = stale_reviews()
assert not reviews, "Pages whose review does not cover the current text, figures or code:\n" + "\n".join(reviews)
for svg in (ROOT / "docs/assets/diagrams").glob("*.svg"):
    root = ET.parse(svg).getroot()
    assert root.attrib.get("viewBox"), f"SVG needs responsive viewBox: {svg}"
    namespace = {"s": "http://www.w3.org/2000/svg"}
    assert root.find("s:title", namespace) is not None, f"SVG needs accessible title: {svg}"
print(f"{len(lessons)} lesson pages, marked code excerpts, fixed-version notebooks, reviews of every page and SVG assets: OK")

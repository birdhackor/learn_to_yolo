"""Validate published lesson pairs, fixed source refs and review records."""

import ast
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sections = json.loads((ROOT / "section-map.json").read_text())["sections"]
lessons = [s for s in sections if s["kind"] == "lesson"]
assert len(lessons) == 42
for section in lessons:
    lesson_id = section["id"]
    assert section["status"] == "ready", lesson_id
    page = ROOT / section["page"]
    text = page.read_text()
    assert "{{COLAB:" not in text, f"Unresolved link: {lesson_id}"
    assert f"/blob/{section['source_ref']}/notebooks/{lesson_id}.ipynb" in text
    assert len(text) >= 800, f"Lesson appears incomplete: {lesson_id}"
    review = ROOT / "reviews" / f"{lesson_id}.md"
    assert review.is_file() and len(review.read_text()) > 40, f"Missing fresh-reader review: {lesson_id}"
    notebook = json.loads((ROOT / section["notebook"]).read_text())
    assert notebook["metadata"]["source_ref"] == section["source_ref"]
    code = "".join(notebook["cells"][-1]["source"])
    assert code == (ROOT / "lesson_cases" / f"{lesson_id}.py").read_text()
    ast.parse(code)
    for diagram in re.findall(r"!?\[[^\]]*\]\(([^)]+\.svg)\)", text):
        assert (page.parent / diagram).resolve().is_file(), f"Missing diagram: {diagram}"
for svg in (ROOT / "docs/assets/diagrams").glob("*.svg"):
    root = ET.parse(svg).getroot()
    assert root.attrib.get("viewBox"), f"SVG needs responsive viewBox: {svg}"
    namespace = {"s": "http://www.w3.org/2000/svg"}
    assert root.find("s:title", namespace) is not None, f"SVG needs accessible title: {svg}"
print("42 lesson pages, fixed-version notebooks, fresh-reader reviews and SVG assets: OK")

"""Validate the actual Zensical output before uploading a Pages artifact."""

from __future__ import annotations

import argparse
import json
import re
import tomllib
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

ROOT = Path(__file__).resolve().parents[1]
# Markup Python-Markdown and our pages produce inside <article>; anything else is text the
# Markdown parser mistook for a tag (e.g. "a<b and c>d").
ARTICLE_TAGS = {
    "a", "abbr", "article", "b", "blockquote", "br", "caption", "code", "col", "colgroup", "dd",
    "del", "details", "div", "dl", "dt", "em", "figcaption", "figure", "h1", "h2", "h3", "h4", "h5",
    "h6", "hr", "i", "img", "ins", "kbd", "li", "mark", "ol", "p", "pre", "small", "span", "strong",
    "sub", "summary", "sup", "table", "tbody", "td", "tfoot", "th", "thead", "tr", "ul",
}
ARTICLE_ATTRIBUTES = {
    "align", "alt", "class", "colspan", "decoding", "height", "href", "id", "lang", "loading",
    "name", "open", "rel", "role", "rowspan", "src", "style", "target", "title", "width",
}
MARKDOWN_TABLE_SEPARATOR = re.compile(r"\s*:?-+:?\s*")
ORDERED_ITEM = re.compile(r"( *)(\d+)\. ")


class Article(HTMLParser):
    """Visible article text, with the places where Markdown silently failed to render."""

    def __init__(self, source: str):
        super().__init__()
        self.problems: list[str] = []
        self.depth = 0
        self.stack: list[tuple[str, bool]] = []
        self.paragraph: list[str] | None = None
        self.feed(source)

    def inside(self, *tags) -> bool:
        return any(tag in tags for tag, _ in self.stack)

    def handle_starttag(self, tag, attributes):
        if tag == "article":
            self.depth += 1
        if not self.depth:
            return
        math = dict(attributes).get("class", "") == "arithmatex"
        if not self.inside("svg", "math"):
            if tag not in ARTICLE_TAGS | {"svg", "math"}:
                self.problems.append(f"text parsed as a <{tag}> tag")
            for name, _ in attributes:
                if name not in ARTICLE_ATTRIBUTES and not name.startswith(("data-", "aria-")):
                    self.problems.append(f"text parsed as attribute {name!r} of <{tag}>")
        if tag not in {"br", "hr", "img", "col"}:
            self.stack.append((tag, math))
        if tag == "p":
            self.paragraph = []

    def handle_endtag(self, tag):
        if not self.depth:
            return
        while self.stack:
            if self.stack.pop()[0] == tag:
                break
        if tag == "p" and self.paragraph is not None:
            text = "".join(self.paragraph).strip()
            if text.startswith("|") and text.count("|") > 2:
                self.problems.append(f"table rendered as a paragraph: {text[:60]!r}")
            self.paragraph = None
        if tag == "article":
            self.depth -= 1

    def handle_data(self, data):
        if not self.depth or self.inside("script", "style", "svg", "math"):
            return
        if self.paragraph is not None:
            self.paragraph.append(data)
        if self.inside("pre", "code") or any(math for _, math in self.stack):
            return
        for marker in ("???", "\\(", "\\)", "\\[", "\\]"):
            if marker in data:
                self.problems.append(f"unrendered {marker!r}: {data.strip()[:60]!r}")
        if not self.inside("a") and re.search(r"https?://", data):
            self.problems.append(f"URL shown as plain text: {data.strip()[:60]!r}")


def table_cells(row: str) -> int:
    """Columns Python-Markdown sees in a table row: unescaped pipes outside code spans."""
    row = row.strip()
    row = row[1:] if row.startswith("|") else row
    row = row[:-1] if row.endswith("|") and not row.endswith("\\|") else row
    count, index = 1, 0
    while index < len(row):
        if row[index] == "\\":
            index += 2
            continue
        if row[index] == "`":
            end = index
            while end < len(row) and row[end] == "`":
                end += 1
            close = row.find(row[index:end], end)
            index = close + end - index if close != -1 else end
            continue
        count += row[index] == "|"
        index += 1
    return count


def markdown_list_problems(path: Path) -> list[str]:
    """Numbered items that Python-Markdown renders as the start of a new list, so they show as "1.".

    A block inside a list item must be indented 4 spaces more than the item's marker; a less indented
    block after a blank line ends the list, and Python-Markdown ignores the number of the next item.
    """
    lines = path.read_text().splitlines()
    problems, fence, last_item = [], False, {}
    for number, line in enumerate(lines):
        if re.match(r" *(```|~~~)", line):
            fence = not fence
        item = None if fence else ORDERED_ITEM.match(line)
        if not item:
            continue
        indent, value = len(item.group(1)), int(item.group(2))
        previous = last_item.get(indent)
        if value > 1 and previous is not None:
            between = lines[previous + 1:number]
            breaks = [text for before, text in zip(between, between[1:]) if not before.strip() and text.strip()
                      and len(text) - len(text.lstrip()) < indent + 4 and not ORDERED_ITEM.match(text)]
            if breaks:
                problems.append(f"{path.relative_to(ROOT)}:{number + 1}: item {value} starts a new list and shows as 1 "
                                f"(indent the blocks of the previous item by {indent + 4} spaces)")
        last_item = {k: v for k, v in last_item.items() if k < indent}
        last_item[indent] = number
    return problems


def markdown_table_problems(path: Path) -> list[str]:
    """Rows whose column count differs from the header: Markdown drops or pads cells silently."""
    lines = path.read_text().splitlines()
    problems = []
    for number, line in enumerate(lines[:-1]):
        separator = lines[number + 1].strip().strip("|").split("|")
        if not line.lstrip().startswith("|") or not all(MARKDOWN_TABLE_SEPARATOR.fullmatch(c) for c in separator):
            continue
        width = table_cells(line)
        row = number + 1
        while row < len(lines) and lines[row].lstrip().startswith("|"):
            if table_cells(lines[row]) != width:
                problems.append(f"{path.relative_to(ROOT)}:{row + 1}: {table_cells(lines[row])} columns, header has {width}")
            row += 1
    return problems


class Page(HTMLParser):
    def __init__(self, source: str):
        super().__init__()
        self.ids: set[str] = set()
        self.links: list[str] = []
        self.article_links: list[str] = []
        self.generator = ""
        self.in_article = False
        self.feed(source)

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if attrs.get("id"):
            self.ids.add(attrs["id"])
        if tag == "article":
            self.in_article = True
        if tag == "meta" and attrs.get("name") == "generator":
            self.generator = attrs.get("content", "")
        if tag in {"a", "link"} and attrs.get("href"):
            self.links.append(attrs["href"])
            if tag == "a" and self.in_article:
                self.article_links.append(attrs["href"])
        if tag in {"img", "script"} and attrs.get("src"):
            self.links.append(attrs["src"])

    def handle_endtag(self, tag):
        if tag == "article":
            self.in_article = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    config = tomllib.loads((ROOT / "zensical.toml").read_text())["project"]
    site = (ROOT / config["site_dir"]).resolve()
    base = config["site_url"]
    base_parts = urlsplit(base)
    version = (ROOT / "requirements-docs.txt").read_text().strip().split("==")[1]
    pages = {
        path: Page(path.read_text()) for path in site.rglob("*.html")
    }
    assert pages, "Build the site first"
    assert not (ROOT / "mkdocs.yml").exists(), "Use the native Zensical config"
    for path, page in pages.items():
        assert page.generator == "zensical-" + version, path
        relative = path.relative_to(site).as_posix()
        page_url = urljoin(base, relative.removesuffix("index.html"))
        for reference in page.links:
            target = urlsplit(urljoin(page_url, reference))
            if (target.scheme, target.netloc) != (base_parts.scheme, base_parts.netloc):
                continue
            assert target.path.startswith(base_parts.path), (path, reference)
            local = site / unquote(target.path[len(base_parts.path):])
            if local.is_dir():
                local /= "index.html"
            assert local.resolve().is_relative_to(site), (path, reference)
            assert local.is_file(), (path, reference, "missing output")
            if target.fragment and local in pages:
                fragment = unquote(target.fragment).split(":~:text=", 1)[0]
                if fragment:
                    assert fragment in pages[local].ids, (path, reference, "missing anchor")

    search = json.loads((site / "search.json").read_text())
    indexed_pages = {item["location"].split("#", 1)[0] for item in search["items"]}
    assert search["config"]["lang"] == ["zh-TW"], "Wrong search language"
    registry = json.loads((ROOT / "section-map.json").read_text())
    lessons = [s for s in registry["sections"] if s["kind"] == "lesson"]
    for lesson in lessons:
        location = "lessons/" + lesson["id"] + "/"
        output = site / location / "index.html"
        assert output in pages, lesson["id"]
        assert location in indexed_pages, (lesson["id"], "not searchable")
        expected = (
            "https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/"
            + lesson["source_ref"] + "/" + lesson["notebook"]
        )
        assert expected in pages[output].article_links, (lesson["id"], "Colab mismatch")

    rendering = [f"{path.relative_to(site)}: {problem}"
                 for path in pages for problem in Article(path.read_text()).problems]
    rendering += [problem for path in sorted((ROOT / config["docs_dir"]).rglob("*.md"))
                  for problem in markdown_table_problems(path) + markdown_list_problems(path)]
    assert not rendering, "Markdown did not render as written:\n" + "\n".join(rendering)

    files = [p for p in site.rglob("*") if p.is_file()]
    for path in files:
        assert not path.is_symlink(), path
        assert path.suffix not in {".pt", ".pth", ".onnx", ".ipynb", ".zip", ".tar", ".gz"}, path
        with path.open("rb") as source:
            assert not source.read(80).startswith(b"version https://git-lfs.github.com/spec/v1"), path
    report = {
        "builder": "zensical", "version": version,
        "html_pages": len(pages), "lesson_pages": len(lessons),
        "all_local_links_and_anchors": "passed", "colab_pairs": "passed",
        "markdown_rendering": "passed: tables, numbered lists, admonitions, math, links and no text parsed as tags",
        "search_language": search["config"]["lang"],
        "all_lessons_indexed": True, "artifact_files": len(files),
        "artifact_bytes": sum(p.stat().st_size for p in files),
        "artifact_boundaries": "passed",
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()

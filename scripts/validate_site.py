"""Validate the actual Zensical output before uploading a Pages artifact."""

from __future__ import annotations

import argparse
import json
import tomllib
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

ROOT = Path(__file__).resolve().parents[1]


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

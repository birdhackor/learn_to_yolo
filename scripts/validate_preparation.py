"""Check dataset manifests, Colab notebook format and publication boundaries."""

import ast
import json
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "data/manifest.json").read_text())
ids = set()
for dataset in manifest["datasets"]:
    assert dataset["id"] not in ids, "Duplicate dataset ID"
    ids.add(dataset["id"])
    if dataset["status"] != "download-ready":
        continue
    assert dataset.get("license") and dataset.get("license_url")
    assert dataset["resources"], "Ready dataset requires resources"
    names = set()
    for resource in dataset["resources"]:
        assert resource["filename"] == Path(resource["filename"]).name
        assert resource["filename"] not in names
        names.add(resource["filename"])
        assert resource["bytes"] > 0
        assert len(resource["sha256"]) == 64
        int(resource["sha256"], 16)
        assert urlsplit(resource["url"]).scheme == "https"
section_map = json.loads((ROOT / "section-map.json").read_text())
section_ids = set()
for section in section_map["sections"]:
    assert section["id"] not in section_ids, "Duplicate section ID"
    section_ids.add(section["id"])
    assert section["status"] in {"planned", "ready"}
    if section["status"] == "ready":
        assert (ROOT / section["page"]).is_file(), "Missing reading page"
        assert (ROOT / section["notebook"]).is_file(), "Missing Colab notebook"
        assert section["source_ref"], "Ready section requires a source version"
    else:
        assert section["page"] is None and section["notebook"] is None
    assert set(section["datasets"]).issubset(ids), "Unknown dataset reference"
for path in (ROOT / "notebooks").glob("*.ipynb"):
    notebook = json.loads(path.read_text())
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["kernelspec"]["language"] == "python"
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
for path in (ROOT / "docs").rglob("*"):
    if path.is_file():
        assert path.stat().st_size < 5 * 1024 * 1024, f"Large website asset: {path}"
        assert path.suffix not in {".pt", ".pth", ".onnx", ".zip", ".tar", ".gz"}
        with path.open("rb") as source:
            assert not source.read(80).startswith(b"version https://git-lfs.github.com/spec/v1")
print("Dataset manifest, section pairing, notebook code and website publication boundaries: OK")

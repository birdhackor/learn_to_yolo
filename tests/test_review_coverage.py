"""A displayed model-output image must invalidate the page's old review when changed."""

import importlib
import json
from pathlib import Path


def test_changed_raster_output_requires_a_new_review(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    coverage = importlib.import_module("review_coverage")
    monkeypatch.setattr(coverage, "ROOT", tmp_path)
    monkeypatch.setattr(coverage, "COVERAGE", tmp_path / "reviews/coverage.json")
    (tmp_path / "docs/assets").mkdir(parents=True)
    (tmp_path / "reviews").mkdir()
    page = tmp_path / "docs/index.md"
    page.write_text("# Result\n\n![actual prediction](assets/prediction.png)\n")
    image = tmp_path / "docs/assets/prediction.png"
    image.write_bytes(b"first recorded prediction")
    review = tmp_path / "reviews/index.md"
    review.write_text("Checked the displayed prediction against the recorded output.\n")
    (tmp_path / "zensical.toml").write_text(
        '[project]\ndocs_dir = "docs"\nnav = [{Home = "index.md"}]\n')
    recorded = {"docs/index.md": {
        "review": "reviews/index.md",
        "review_sha256": coverage.provenance.file_sha256(review),
        "covered": coverage.digest(page),
    }}
    coverage.COVERAGE.write_text(json.dumps(recorded))
    assert coverage.stale() == []

    image.write_bytes(b"different model output, unchanged page text")
    assert coverage.stale() == [
        "docs/index.md: changed since its review: docs/assets/prediction.png"]

    recorded["docs/index.md"]["covered"] = coverage.digest(page)
    coverage.COVERAGE.write_text(json.dumps(recorded))
    assert coverage.stale() == []

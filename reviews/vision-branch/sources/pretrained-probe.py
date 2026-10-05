"""Independently check saved official-pretrained evidence using NumPy and a bad-cache probe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from miniyolo.provenance import file_sha256
from scripts.run_dino_pretrained import verified_download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repeat", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    repeat = json.loads(args.repeat.read_text())
    assert report["source_record"]["entrypoint_sha256"] == file_sha256(ROOT / "scripts/run_dino_pretrained.py")
    for key in ("official_code", "checkpoint"):
        assert file_sha256(report[key]["path"]) == report[key]["expected_sha256"]
        assert repeat[key]["cache_hit"]
    assert report["features"]["cls_cosines"] == repeat["features"]["cls_cosines"]
    assert report["patch_neighbors"] == repeat["patch_neighbors"]
    assert [x["rgb_pixels_sha256"] for x in report["inputs"]["images"]] == [x["rgb_pixels_sha256"] for x in repeat["inputs"]["images"]]
    features_path = Path(report["features"]["saved_npz"])
    assert file_sha256(features_path) == report["features"]["saved_npz_sha256"]
    for image in report["inputs"]["images"]:
        assert file_sha256(image["path"]) == image["file_sha256"]
    with np.load(features_path) as saved:
        patches = saved["patch"].astype(np.float64)
        unit = patches / np.linalg.norm(patches, axis=-1, keepdims=True)
        query = report["patch_neighbors"]["query_index"]
        cosine = unit[1] @ unit[0, query]
        recorded = np.asarray(report["patch_neighbors"]["cosine_map_16x16"]).ravel()
        cosine_error = float(np.max(np.abs(cosine - recorded)))
        assert cosine_error < 2e-6
        assert int(cosine.argmax()) == report["patch_neighbors"]["top8"][0]["patch_index"]
        gram = unit @ unit.transpose(0, 2, 1)
        gram_error = float(np.max(np.abs(gram - saved["gram"])))
        assert gram_error < 2e-6
        assert np.max(np.abs(np.diagonal(gram, axis1=1, axis2=2) - 1)) < 1e-10
    with tempfile.TemporaryDirectory() as temp_dir:
        corrupt = Path(temp_dir) / "corrupt.pth"
        corrupt.write_bytes(b"incorrect cached artifact")
        try:
            verified_download("https://example.invalid/never-requested", corrupt, "0" * 64)
        except ValueError as error:
            assert "Cached SHA-256 mismatch" in str(error)
        else:
            raise AssertionError("Corrupt cache was accepted")
    results = {"passed": True, "probe_sha256": file_sha256(__file__),
               "report_sha256": file_sha256(args.report), "repeat_report_sha256": file_sha256(args.repeat),
               "source_code_matches": True, "official_artifact_hashes_match": True,
               "saved_png_npz_hashes_match": True, "repeat_values_equal": True,
               "numpy_float64_cosine_max_abs_error": cosine_error,
               "numpy_float64_gram_max_abs_error": gram_error,
               "corrupted_cache_rejected_without_network": True,
               "scope": "Saved DINOv2 frozen CPU feature evidence; no training, benchmark or DINOv3 execution."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

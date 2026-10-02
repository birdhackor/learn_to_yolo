"""Exercise the lesson's real OpenCV file adapter and detector-to-tracker bridge.

Uses lossless FFV1 AVI so RGB and predictions can be compared exactly. This is
file/codec verification, not a physical camera or video-quality benchmark.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_lesson(sid):
    name = "video_verification_" + sid.replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, ROOT / f"lesson_cases/{sid}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(output):
    started = time.perf_counter()
    torch.set_num_threads(2)
    video = load_lesson("18-video")
    tracking = load_lesson("19-tracking")
    output.mkdir(parents=True, exist_ok=True)
    source = output / "lossless-fixture.avi"
    frames = list(video.synthetic_frames(count=12, fps=20))
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"FFV1"), 20, (96, 64))
    try:
        if not writer.isOpened():
            raise RuntimeError("OpenCV cannot encode FFV1; this verification needs a lossless codec")
        for frame in frames:
            writer.write(cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    assert source.is_file() and source.stat().st_size > 0

    # Wrap the real constructor to observe real handles, without replacing I/O.
    constructor, opened = cv2.VideoCapture, []

    def capture(*args, **kwargs):
        handle = constructor(*args, **kwargs)
        opened.append(handle)
        return handle

    with patch.object(cv2, "VideoCapture", capture):
        recovered = list(video.opencv_frames(str(source)))
        assert len(recovered) == len(frames)
        assert all(np.array_equal(a.rgb, b.rgb) for a, b in zip(frames, recovered))
        assert [f.index for f in recovered] == list(range(12))
        assert all(abs(f.timestamp_s - f.index / 20) < 1e-10 for f in recovered)
        assert not opened[-1].isOpened(), "EOF must release the real capture"
        adapter = video.opencv_frames(str(source))
        assert next(adapter).index == 0 and opened[-1].isOpened()
        adapter.close()
        assert not opened[-1].isOpened(), "Explicit generator close must release capture"
        error_adapter = video.opencv_frames(str(output / "missing.avi"))
        try:
            next(error_adapter)
        except RuntimeError as error:
            assert "Cannot open video source" in str(error)
        else:
            raise AssertionError("Missing video file accepted")
        assert not opened[-1].isOpened()

    model = video.fit_detector()
    in_memory = list(video.run_stream(iter(frames), model))
    file_started = time.perf_counter()
    from_file = list(video.run_stream(video.opencv_frames(str(source)), model))
    file_pipeline_wall = time.perf_counter() - file_started
    assert len(from_file) == 12
    for original, loaded in zip(in_memory, from_file):
        assert original["index"] == loaded["index"]
        assert original["source_timestamp_s"] == loaded["source_timestamp_s"]
        assert original["image"].size == loaded["image"].size == (96, 64)
        assert np.array_equal(np.asarray(original["image"]), np.asarray(loaded["image"]))
        assert all(torch.equal(original["prediction"][key], loaded["prediction"][key])
                   for key in ("boxes", "scores", "labels"))

    tracker = tracking.Tracker(motion=True, max_age=2)
    ids, boxes = [], []
    for result in from_file:
        pred = result["prediction"]
        # The tiny tracker has no class gate: associate only class 0 here.
        selected = pred["boxes"][pred["labels"] == 0]
        assigned = tracker.update(selected, result["index"])
        assert len(assigned) == len(selected) and len(set(assigned)) == len(assigned)
        ids.append(assigned)
        boxes.append(selected.tolist())
    assert any(ids), "The bridge must process actual nonempty model detections"
    assert any(not row for row in ids), "This fixture also exercises missed/empty frames"
    from_file[0]["image"].save(output / "first-overlay.png")
    report = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "code_commit_scope": "Base checkout commit before the extension is committed; script and lesson SHA-256 identify the executed working-tree bytes",
        "script_sha256": sha256(__file__),
        "lesson_case_sha256": {sid: sha256(ROOT / f"lesson_cases/{sid}.py")
                               for sid in ("18-video", "19-tracking")},
        "device": "cpu", "torch": str(torch.__version__), "numpy": str(np.__version__),
        "opencv": str(cv2.__version__), "model_parameters": sum(p.numel() for p in model.parameters()),
        "training": {"steps": 160, "samples": 32, "seed": 7, "data_seed": 4100,
                     "batch_size": 8, "optimizer": "Adam", "learning_rate": .01},
        "video": {"codec": "FFV1", "container": "AVI", "frames": 12, "source_fps": 20,
                  "source_size_hw": [64, 96], "last_timestamp_s": recovered[-1].timestamp_s,
                  "sha256": sha256(source), "bytes": source.stat().st_size},
        "rgb_pixels_exact": True, "detector_predictions_exact": True,
        "overlays_exact": True,
        "capture_released": {"eof": True, "early_generator_close": True, "open_failure": True},
        "boxes_per_frame": [len(r["prediction"]["boxes"]) for r in from_file],
        "tracking": {"class_id": 0, "motion": True, "max_age": 2, "ids": ids,
                     "original_pixel_boxes": boxes,
                     "limits": "Actual detector input; no GT identity matching, IDF1/HOTA or tracking quality claim"},
        "file_pipeline_wall_seconds": file_pipeline_wall,
        "file_pipeline_timing_scope": "Entire file consumption including capture/open/read/decode, CPU preprocess/model/postprocess/drawing and Python collection; excludes prior training, encoding and disk output; no display or network queue",
        "verification_wall_seconds": time.perf_counter() - started,
        "verification_timing_scope": "Inside verify(), including fixture encoding, decoding and resource checks, model initialization/training, two inference streams, tracking, first-overlay save and hashes; excludes imports, process startup and final JSON write",
        "limits": "Controlled lossless generated video, CPU only; no physical camera, lossy MP4/color-space parity, variable-frame-rate PTS, live/network latency or formal throughput benchmark",
    }
    destination = ROOT / "artifacts/checks/curriculum/video-file.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/runs/video-file")
    args = parser.parse_args()
    print(json.dumps(verify(args.output), indent=2))


if __name__ == "__main__":
    main()

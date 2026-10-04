"""Which code each saved execution record is bound to, and whether that code is unchanged.

A record stores dependencies_sha256: the SHA-256 of every repository file its entry files import,
directly or indirectly (miniyolo.provenance.repo_dependencies). It stays current until one of those
files changes. Lesson records are bound to lesson_cases/<id>.py (scripts/verify_curriculum.py); the
records below are bound to the entry files listed here. This module loads provenance.py by path, so
it needs no PyTorch (the Pages build has none), and it is not imported by any code that makes a
record, so editing this list never changes a record's binding.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("provenance", ROOT / "miniyolo/provenance.py")
provenance = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(provenance)

C = "artifacts/checks/curriculum/"
# Made on the CPU recording machine by scripts/record_evidence.py. Lesson cases that a script loads
# by path at run time are listed as extra entries.
CPU_RECORDS = {
    "artifacts/checks/grid-learning.json": ("miniyolo/train.py",),
    C + "custom-data-160-step.json": ("scripts/run_custom_data_learning.py",),
    C + "custom-data-learning.json": ("scripts/run_custom_data_learning.py",),
    C + "01-small-cnn-learning.json": ("scripts/run_learning_extensions.py", "lesson_cases/01-small-cnn.py"),
    C + "03-comparison-learning.json": ("scripts/run_learning_extensions.py", "lesson_cases/03-comparison.py"),
    C + "04-localization-learning.json": ("scripts/run_learning_extensions.py", "lesson_cases/04-localization.py"),
    C + "10-multiscale-learning.json": ("scripts/run_multiscale_learning.py", "lesson_cases/10-multiscale.py"),
    C + "video-file.json": ("scripts/verify_video_file.py", "lesson_cases/18-video.py", "lesson_cases/19-tracking.py"),
    C + "fashion-mnist-learning.json": ("scripts/run_fashion_cnn.py",),
}
# Made on one NVIDIA L4 by a manually started GitHub Actions workflow that runs the code on Modal;
# the record is that run's result.json artifact.
GPU_RECORDS = {
    C + "deployment-gpu.json": (("miniyolo/deployment_gpu.py",), "deployment-gpu.yml"),
    "artifacts/checks/gpu-smoke.json": (("miniyolo/gpu_smoke.py",), "gpu-smoke.yml"),
}


def entries(record: str) -> tuple[str, ...]:
    return CPU_RECORDS[record] if record in CPU_RECORDS else GPU_RECORDS[record][0]


def bound_code(record: str) -> dict[str, str]:
    return provenance.repo_dependencies(*(ROOT / entry for entry in entries(record)))


def is_current(record: str) -> bool:
    """True if the record exists and every file of the code it is bound to is unchanged."""
    path = ROOT / record
    return path.is_file() and json.loads(path.read_text()).get("dependencies_sha256") == bound_code(record)

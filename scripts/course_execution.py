"""Reuse a verified complete CPU lesson run; keep transient results outside Git.

The notebook bootstrap cells are identified but not executed: they install packages and clone
published tags. check_lesson_runtime executes every experiment cell in a fresh CPU process.
Historical scientific records are checked separately and are never overwritten by this cache.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 1
THREADS = "2"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def lessons(root: Path) -> list[dict]:
    rows = json.loads((root / "section-map.json").read_bytes())["sections"]
    rows = [row for row in rows if row["kind"] == "lesson"]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("The lesson registry is empty or has duplicate identifiers")
    for row in rows:
        for name in (row["notebook"], f"lesson_cases/{row['id']}.py"):
            path = root / name
            if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
                raise ValueError(f"Missing or invalid lesson input: {name}")
    return rows


def code_signature(notebook: dict) -> dict:
    return {"kernel": notebook.get("metadata", {}).get("kernelspec", {}),
            "cells": [{"source": "".join(cell["source"]), "metadata": cell.get("metadata", {})}
                      for cell in notebook["cells"] if cell["cell_type"] == "code"]}


def input_key(root: Path = ROOT) -> str:
    """Identify actual execution inputs; authored prose and ignored downloads are independent.

    All code and assets are included conservatively, without a per-lesson dependency graph.
    A Markdown file read by an experiment must be added here, just like any other runtime input.
    """
    digest = hashlib.sha256()

    def include(name, raw):
        digest.update(name.encode() + b"\0" + hashlib.sha256(raw).digest())

    rows = lessons(root)
    include("lessons", json.dumps([(row["id"], row["notebook"]) for row in rows]).encode())
    for row in rows:
        path = root / row["notebook"]
        notebook = json.loads(path.read_bytes())
        code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        if not code or "".join(code[-1]["source"]) != (root / f"lesson_cases/{row['id']}.py").read_text():
            raise ValueError(f"Notebook experiment differs from its lesson program: {row['id']}")
        include(row["notebook"], json.dumps(code_signature(notebook), sort_keys=True).encode())
    paths = set(root.glob("requirements-*.txt")) | set(root.glob("requirements-*.in")) | set(root.glob("*.lock"))
    paths.update(root / name for name in (".gitattributes", ".python-version", "data/manifest.json"))
    for directory in ("lesson_cases", "miniyolo", "scripts"):
        paths.update((root / directory).rglob("*.py"))
    for directory in ("docs/assets", "artifacts/checks", "reviews/vision-branch/sources"):
        paths.update(path for path in (root / directory).rglob("*")
                     if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc")
    for path in sorted(paths):
        if path.is_file():
            include(path.relative_to(root).as_posix(), path.read_bytes())
    include("execution-contract", json.dumps({"schema": SCHEMA, "threads": THREADS}).encode())
    return digest.hexdigest()


def runtime() -> dict:
    # Include every installed distribution: optional backends can affect execution, too.
    packages = sorted((dist.metadata["Name"], dist.version) for dist in importlib.metadata.distributions())
    processor = platform.processor()
    if Path("/proc/cpuinfo").is_file():
        processor = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                          if line.startswith("model name")), processor)
    environment = {name: os.environ.get(name, "") for name in
                   ("ONEDNN_MAX_CPU_ISA", "ATEN_CPU_CAPABILITY", "PYTHONHASHSEED", "OPENBLAS_NUM_THREADS")}
    return {"python": platform.python_version(), "system": platform.system(), "release": platform.release(),
            "machine": platform.machine(), "processor": processor, "packages": packages,
            "environment": environment, "threads": THREADS, "device": "cpu"}


def execution_key(root: Path = ROOT) -> str:
    return sha(json.dumps({"inputs": input_key(root), "runtime": runtime()}, sort_keys=True).encode())


def validate_report(root: Path, report: dict) -> None:
    expected = [row["id"] for row in lessons(root)]
    results = report.get("results", [])
    if (report.get("device") != "cpu" or report.get("passed") != len(expected)
            or report.get("total") != len(expected) or [row.get("id") for row in results] != expected):
        raise ValueError("Missing or incomplete CPU lesson results")
    for row in results:
        if row.get("passed") is not True or row.get("exit_code") != 0 or not row.get("stdout", "").strip():
            raise ValueError(f"Unsuccessful lesson result: {row.get('id')}")


def validate_bundle(root: Path, output: Path, key: str | None = None) -> dict:
    receipt = json.loads((output / "execution.json").read_bytes())
    raw = (output / "report.json").read_bytes()
    recorded_key = sha(json.dumps({"inputs": receipt.get("input_key"), "runtime": receipt.get("runtime")},
                                 sort_keys=True).encode())
    if (receipt.get("schema_version") != SCHEMA or receipt.get("input_key") != input_key(root)
            or receipt.get("key") != recorded_key
            or receipt.get("report_sha256") != sha(raw) or (key is not None and receipt.get("key") != key)):
        raise ValueError("Execution inputs, runtime or cached results changed")
    validate_report(root, json.loads(raw))
    return receipt


def ensure(root: Path = ROOT, output: Path | None = None, revision: str = "unknown") -> dict:
    output = output or root / "outputs/lesson-execution"
    if (not (root / "outputs").resolve().is_relative_to(root.resolve())
            or not output.resolve().is_relative_to((root / "outputs").resolve())
            or output.resolve() == (root / "outputs").resolve() or output.is_symlink()):
        raise ValueError("The execution cache must be a child directory of outputs/")
    key = execution_key(root)
    try:
        receipt = validate_bundle(root, output, key)
    except (OSError, ValueError, KeyError, TypeError):
        if output.exists():
            shutil.rmtree(output)
        output.mkdir(parents=True)
        subprocess.run([sys.executable, str(root / "scripts/check_lesson_runtime.py"),
                        "--report", str(output / "report.json")], cwd=root, check=True,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": THREADS,
                            "MKL_NUM_THREADS": THREADS, "MPLBACKEND": "Agg"})
        raw = (output / "report.json").read_bytes()
        validate_report(root, json.loads(raw))
        if execution_key(root) != key:
            raise ValueError("Execution inputs changed during the run; no success receipt was saved")
        receipt = {"schema_version": SCHEMA, "key": key, "input_key": input_key(root),
                   "runtime": runtime(), "report_sha256": sha(raw), "executed_revision": revision,
                   "executed_at": datetime.now(timezone.utc).isoformat()}
        (output / "execution.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        receipt = validate_bundle(root, output, key)
        print(f"Executed all {len(lessons(root))} CPU lesson experiments.")
    else:
        print(f"Reused verified CPU results from {receipt['executed_revision']} ({receipt['executed_at']}).")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("key", "ensure", "verify"))
    parser.add_argument("--revision", default="unknown")
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    if args.command == "key":
        key = execution_key()
        print(key)
        if args.github_output:
            with args.github_output.open("a", encoding="utf-8") as stream:
                stream.write(f"key={key}\n")
    elif args.command == "ensure":
        ensure(revision=args.revision)
    else:
        validate_bundle(ROOT, ROOT / "outputs/lesson-execution", execution_key())
        print("CPU execution bundle: OK")


if __name__ == "__main__":
    main()

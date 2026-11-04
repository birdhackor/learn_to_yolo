"""Verify explicitly downloaded fixed data offline, without changing its historical downloader."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

try:
    from .download_data import ROOT, verify
except ImportError:
    from download_data import ROOT, verify


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--asset", help="Verify only this exact manifest filename")
    parser.add_argument("--output", type=Path, default=ROOT / "data/downloads")
    args = parser.parse_args()
    datasets = json.loads((ROOT / "data/manifest.json").read_text())["datasets"]
    dataset = next((row for row in datasets if row["id"] == args.dataset), None)
    if dataset is None or dataset["status"] != "download-ready":
        parser.error("Select a download-ready dataset from scripts/download_data.py list")
    resources = [row for row in dataset["resources"] if not args.asset or row["filename"] == args.asset]
    if not resources:
        parser.error("Unknown asset filename")
    try:
        for resource in resources:
            path = args.output / dataset["id"] / resource["filename"]
            if not verify(path, resource):
                raise ValueError(f"Missing or changed fixed input: {path}; fetch explicitly if missing")
            print(f"Verified offline: {path.name}")
    except (OSError, ValueError) as error:
        print(f"Offline verification failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Download preparation datasets with byte-count and checksum validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request


ROOT = Path(__file__).resolve().parents[1]


def verify(path: Path, resource: dict) -> bool:
    if not path.is_file() or path.stat().st_size != resource["bytes"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == resource["sha256"]


def fetch_resource(resource: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if verify(destination, resource):
        print(f"Verified cache: {destination.name}")
        return
    if destination.exists():
        raise ValueError(
            f"Existing file failed verification: {destination}. "
            "Move or remove it before retrying."
        )
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists():
        raise ValueError(f"Partial download exists: {partial}; remove it to retry.")
    request = urllib.request.Request(
        resource["url"], headers={"User-Agent": "learn-to-yolo-data-preparation"}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            with partial.open("xb") as target:
                downloaded = 0
                while chunk := response.read(1024 * 1024):
                    downloaded += len(chunk)
                    if downloaded > resource["bytes"]:
                        raise ValueError("Response exceeded expected resource size")
                    target.write(chunk)
        if not verify(partial, resource):
            raise ValueError(f"Size or SHA256 mismatch: {resource['filename']}")
        partial.rename(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    print(f"Downloaded and verified: {destination.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["list", "fetch"])
    parser.add_argument("dataset", nargs="?")
    parser.add_argument("--asset", help="Only download this exact filename")
    parser.add_argument("--output", type=Path, default=ROOT / "data/downloads")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    datasets = manifest["datasets"]
    if args.command == "list":
        for dataset in datasets:
            print(f"{dataset['id']}: {dataset['status']} — {dataset['purpose']}")
        return 0
    if not args.dataset:
        parser.error("fetch requires a dataset ID; run list first")
    dataset = next((d for d in datasets if d["id"] == args.dataset), None)
    if dataset is None:
        parser.error(f"Unknown dataset: {args.dataset}")
    if dataset["status"] != "download-ready":
        parser.error("Candidate only; see docs/preparation/data.md for next steps")
    resources = dataset["resources"]
    if args.asset:
        resources = [r for r in resources if r["filename"] == args.asset]
        if not resources:
            parser.error("Unknown asset filename")
    try:
        for resource in resources:
            fetch_resource(resource, args.output / dataset["id"] / resource["filename"])
    except (OSError, ValueError) as error:
        print(f"Download failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

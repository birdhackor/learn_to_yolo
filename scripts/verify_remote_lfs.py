"""Fetch committed LFS objects into an empty cache and check bundle contents."""

import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile

from github_auth import git_environment, safe_output


ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "data/manifest.json").read_text())
with tempfile.TemporaryDirectory(prefix="yolo-remote-lfs-") as temporary:
    storage = Path(temporary)
    result = subprocess.run(
        ["git", "-c", f"lfs.storage={storage}", "lfs", "fetch", "origin", "HEAD"],
        cwd=ROOT,
        env=git_environment(),
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise SystemExit(safe_output(result.stdout + result.stderr))
    for asset in manifest["lfs_assets"]:
        oid = asset["sha256"]
        downloaded = storage / "objects" / oid[:2] / oid[2:4] / oid
        assert downloaded.stat().st_size == asset["bytes"]
        assert hashlib.sha256(downloaded.read_bytes()).hexdigest() == oid
        if asset["id"] == "fashion-mnist-v1":
            dataset = next(d for d in manifest["datasets"] if d["id"] == "fashion-mnist")
            with tarfile.open(downloaded) as archive:
                assert set(archive.getnames()) == set(asset["members"])
                for resource in dataset["resources"]:
                    member = archive.extractfile("fashion-mnist/" + resource["filename"])
                    assert member is not None
                    content = member.read()
                    assert len(content) == resource["bytes"]
                    assert hashlib.sha256(content).hexdigest() == resource["sha256"]
                license_file = archive.extractfile(asset["included_license"])
                assert license_file is not None
                assert license_file.read() == (ROOT / "data/licenses/FASHION_MNIST_LICENSE.txt").read_bytes()
        print(f"Remote LFS download and archive checks passed: {asset['path']}")

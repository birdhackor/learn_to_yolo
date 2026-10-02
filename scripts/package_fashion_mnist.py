"""Build the fixed Fashion-MNIST LFS bundle, including the upstream license."""

import hashlib
import json
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "data/manifest.json").read_text())
dataset = next(d for d in manifest["datasets"] if d["id"] == "fashion-mnist")
asset = next(a for a in manifest["lfs_assets"] if a["id"] == "fashion-mnist-v1")
members = []
for resource in dataset["resources"]:
    source = ROOT / "data/downloads/fashion-mnist" / resource["filename"]
    assert source.stat().st_size == resource["bytes"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == resource["sha256"]
    members.append((source, "fashion-mnist/" + source.name))
members.append((ROOT / "data/licenses/FASHION_MNIST_LICENSE.txt", "fashion-mnist/LICENSE"))
target = ROOT / asset["path"]
target.parent.mkdir(parents=True, exist_ok=True)
with tarfile.open(target, "w", format=tarfile.PAX_FORMAT) as archive:
    for path, name in members:
        info = archive.gettarinfo(str(path), arcname=name)
        info.mtime = info.uid = info.gid = 0
        info.uname = info.gname = ""
        info.mode = 0o644
        with path.open("rb") as source:
            archive.addfile(info, source)
assert target.stat().st_size == asset["bytes"]
assert hashlib.sha256(target.read_bytes()).hexdigest() == asset["sha256"]
print(f"Reproducible bundle verified: {asset['path']} ({asset['bytes']} bytes)")

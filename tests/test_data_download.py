"""Offline verification never downloads or overwrites a missing/modified fixed input."""
import hashlib
import json
import sys

import pytest

from scripts import download_data, verify_data


@pytest.mark.parametrize("state", ["valid", "missing", "modified"])
def test_verify_is_offline_and_preserves_local_files(tmp_path, monkeypatch, state):
    raw = b"fixed source"
    resource = {"filename": "source.bin", "url": "https://example.com/source.bin", "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest()}
    path = tmp_path / "data/downloads/example/source.bin"
    path.parent.mkdir(parents=True)
    if state != "missing":
        path.write_bytes(raw if state == "valid" else b"local modification")
    (tmp_path / "data/manifest.json").write_text(json.dumps({"datasets": [
        {"id": "example", "status": "download-ready", "resources": [resource]}]}))
    monkeypatch.setattr(verify_data, "ROOT", tmp_path)
    monkeypatch.setattr(download_data.urllib.request, "urlopen", lambda *a, **k: pytest.fail("Unexpected network"))
    monkeypatch.setattr(sys, "argv", ["verify_data.py", "example"])
    assert verify_data.main() == (0 if state == "valid" else 1)
    if state == "missing":
        assert not path.exists()
    else:
        assert path.read_bytes() == (raw if state == "valid" else b"local modification")

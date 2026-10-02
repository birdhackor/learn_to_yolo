#!/usr/bin/env bash
set -euo pipefail
cd /workspace/learn_to_yolo
python3 -m venv .venv-docs
.venv-docs/bin/python -m pip install --no-cache-dir -r requirements-docs.txt
python3 -m venv .venv-model
.venv-model/bin/python -m pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
.venv-model/bin/python -m pip install --no-cache-dir -r requirements-model.txt
git lfs version
git lfs install --local
python3 scripts/validate_preparation.py
python3 scripts/validate_lessons.py
.venv-model/bin/python -m pytest tests/test_core.py
.venv-docs/bin/zensical build --clean --strict
python3 scripts/validate_site.py

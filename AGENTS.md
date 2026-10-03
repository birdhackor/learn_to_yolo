# Repository Guidelines

## Project Structure & Module Organization

`miniyolo/` contains reusable PyTorch models, data loaders, geometry, targets, losses, inference, metrics, and training. `lesson_cases/` holds standalone CPU experiments; matching pages and notebooks live in `docs/lessons/` and `notebooks/`. Keep their identifiers aligned through `section-map.json`. Diagrams belong in `docs/assets/diagrams/`, tests in `tests/`, and maintenance utilities in `scripts/`. `reviews/` and `artifacts/checks/` preserve review and execution evidence.

## Build, Test, and Development Commands

Run commands from the repository root. Use Python 3.12 and follow `README.md` to create `.venv-model` and `.venv-docs` environments with pinned requirements; install the CPU PyTorch build for experiments.

- `PYTHONPATH=. .venv-model/bin/python lesson_cases/00-warmup.py`: run one lesson.
- `.venv-model/bin/python -m pytest tests/`: run the test suite.
- `.venv-model/bin/python -m miniyolo.train --steps 160 --samples 32 --device cpu`: train the synthetic detector.
- `python3 scripts/validate_preparation.py` and `python3 scripts/validate_lessons.py`: validate manifests, lesson pairs, and reviews.
- `python3 scripts/validate_curriculum_evidence.py`: check source hashes and saved outputs.
- `.venv-docs/bin/zensical build --clean --strict`, then `python3 scripts/validate_site.py`: build and validate the website.
- `.venv-docs/bin/zensical serve`: preview documentation locally.

## Coding Style & Naming Conventions

Use four-space Python indentation, `snake_case` functions and modules, and `PascalCase` classes. Preserve tensor shapes and coordinate conventions in docstrings. Lesson filenames use numbered, hyphenated identifiers such as `07-training`. Keep educational prose in Traditional Chinese. No formatter or linter is configured; match surrounding code.

## Testing Guidelines

Use pytest files named `test_*.py` and functions named `test_*`. Add deterministic CPU checks for changed behavior, including empty inputs, invalid annotations, numerical stability, and checkpoint compatibility where applicable. No numeric coverage threshold is configured.

After lesson changes, synchronize notebook code and regenerate actual evidence with `.venv-model/bin/python scripts/verify_curriculum.py --section 07-training`. This updates the notebook, lesson page, and evidence JSON; rerun consistency validators afterward.

## Commit & Pull Request Guidelines

History uses concise English imperative subjects, for example `Fix homepage figure` or `Add bounded manual L4 TensorRT parity verification`, without Conventional Commit prefixes. Keep commits focused. PRs should describe the change, affected lessons, validation results, related issues where applicable, and screenshots for visual changes.

## Data, Releases & Agent Instructions

Keep temporary outputs in ignored `artifacts/runs/`. Use Git LFS for curated datasets, checkpoints, and exports as defined in `.gitattributes`; preserve licenses and checksums. Never overwrite published lesson tags; generate future notebook releases with `python3 scripts/build_lesson_notebooks.py --ref <new-release-tag>` and refresh execution evidence afterward. Keep credentials out of commits. Agents should think and plan in English and communicate with users in Traditional Chinese.

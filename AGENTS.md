# Repository Guidelines

## Project Structure & Module Organization

`miniyolo/` contains reusable PyTorch models, data loaders, geometry, targets, losses, inference, metrics, and training. `lesson_cases/` holds standalone CPU experiments; matching pages and notebooks live in `docs/lessons/` and `notebooks/`. Keep their identifiers aligned through `section-map.json`. Diagrams belong in `docs/assets/diagrams/`, tests in `tests/`, and maintenance utilities in `scripts/`. `reviews/` and `artifacts/checks/` preserve review and execution evidence. Raw clear-tutorial review records live in the private repo `birdhackor/learn_to_yolo-reviews`, clone it into the ignored `reviews/clear-tutorial/` when you need them; see `reviews/README.md`.

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

## Repository Size & History Rewrites

On 2026-10-11, about 97% of `.git` was raw review records (mostly screenshots) still in the current tree, and old versions took about 1%. Size problems here come from what gets committed, not from history.

- Commit review screenshots, captured pages and third-party papers to the private reviews repo, not here. Colab notebooks run `git clone --depth 1 --branch <tag>` and download the whole tree at that tag, so check `git ls-tree -r -l <ref> | awk '{s+=$4} END {print s}'` before tagging.
- Measure before deciding. For tracked files, compare `st_size` with `st_blocks*512`. For `.git`, split objects into those still in `HEAD` and those only in history with `git cat-file --batch-all-objects --batch-check='%(objectname) %(objecttype) %(objectsize) %(objectsize:disk)'`, `git rev-list --objects --all` and `git ls-tree -r HEAD`.
- Before removing a path, find every reader: hard-coded paths, `.github/workflows`, doc links (`blob/main`, `tree/main`, tag URLs) and the review gate. Run validators under a `sys.addaudithook` logger to see which files they open. `review_coverage.py` hashes each site page (text, figures, lesson code) and its review file, so any edit to either, even a link fix, needs a review entry and `review_coverage.py --write`. New files outside the site navigation are not checked.
- If a test fails after a change, run it on the unchanged commit before calling it a regression.
- Delete by naming each path. Remove `.git/objects/pack/tmp_pack_*` garbage only when `lsof` shows no user, `.git` has no `*.lock`, and no pack index refers to it.
- Before rewriting history, push every branch and tag to a private archive and compare with `git ls-remote`. Later archive pushes use new ref names. Keep published lesson tags unchanged by rewriting only commits after the latest tag, and use a full-tree rewrite such as `git filter-branch --index-filter '…' -- <branches> ^<tag>`. `git filter-repo --refs <tag>..<branch>` only filters changes made inside the range, so files inherited from the tag survive. Compare every rewritten commit's blob ids and modes with the original minus the removed path, then run the tests and validators.
- Read the `on:` triggers in `.github/workflows/` before a forced push; on 2026-10-11 all five were `workflow_dispatch`. Push with `--force-with-lease=<ref>:<expected old>` and `--atomic`. To switch an existing clone, run `git fetch --prune --force origin '+refs/heads/*:refs/remotes/origin/*' '+refs/tags/*:refs/tags/*'`, `git reset --hard origin/main`, `git update-ref -d ORIG_HEAD`, `git reflog expire --expire=now --all`, then `git gc --prune=now`.

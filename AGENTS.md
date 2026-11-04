# Repository Guidelines

## Project Structure

`miniyolo/` contains reusable PyTorch components. `lesson_cases/`, `docs/lessons/` and `notebooks/` share identifiers through `section-map.json`. Diagrams belong in `docs/assets/diagrams/`, tests in `tests/`, and maintenance tools in `scripts/`. Preserve tensor shapes, coordinate conventions, lesson identifiers and published lesson tags.

Use Python 3.12. Keep educational prose and user communication in Traditional Chinese; think and plan in English. Match surrounding Python style (four spaces, snake_case); no formatter is configured.

## Validation and Execution

- Runtime: install the CPU PyTorch wheel, then `requirements-model.txt`. Tests additionally use `requirements-test.txt`; website builds only use `requirements-docs.txt`. Video and GPU dependencies remain optional.
- `python3 scripts/validate_preparation.py` checks manifests and notebook format.
- `python3 scripts/validate_lessons.py` checks current lesson/code/notebook/figure consistency. Editorial history is not a build input.
- `python3 scripts/validate_curriculum_evidence.py` checks preserved scientific records against their source programs.
- `.venv-model/bin/python -m pytest tests/` runs deterministic CPU tests. Add meaningful checks for behavior changes, empty inputs, invalid data and failure paths where relevant.
- `.venv-model/bin/python scripts/course_execution.py ensure --revision "$(git rev-parse HEAD)"` validates or reruns the complete CPU experiment bundle in ignored `outputs/lesson-execution/`. Its conservative key includes ordered notebook code, shared code, data manifests, assets, scientific records, requirements and the actual runtime. Prose changes can reuse successful results; unknown, changed or incomplete inputs require a rerun. Add any new runtime input, including Markdown read by code, to the key.
- Build with `.venv-docs/bin/zensical build --clean --strict`; use `python3 scripts/validate_site.py` for links, search, figures and notebook pairing. Formal publication also writes `build-info.json` using `scripts/publication.py`, then validates with `--require-publication`.
- When lesson programs change, rebuild notebooks for a new release tag and regenerate affected scientific evidence with `scripts/verify_curriculum.py` / `scripts/record_evidence.py`. The transient CPU cache never replaces historical scientific measurements or relabels their dates. Review changed scientific claims and figures against the new records.
- Compare a failing test with the unchanged base before calling it a regression. Do not skip product tests to conceal a failure.

## Editorial Review and Publication

Use `.agents/skills/clear-tutorial/SKILL.md` for substantive teaching changes. Review readability, correctness and coherence over changed material and necessary context. Keep independent first-reading observations distinct from later technical answers; state actual scope and unverified work. A small maintenance change does not restart the entire historical review process.

Record findings, fixes, validation and limitations briefly in the PR. Detailed sessions, frozen drafts and screenshots are temporary work, normally under ignored `work/` or `outputs/`, with CI artifacts retained 30 days (temporary detailed review artifacts at most 90 days). Never add historical review hashes, original-reviewer callbacks or replay chains as deployment requirements. AI review does not establish student learning outcomes.

Pages remains manually triggered; its CPU execution job is separate from its minimal website build. CI runs on PRs/main, without duplicate feature-branch push runs. Runtime, dependency and workflow changes include Windows/macOS; prose changes need Linux. Unavailable diff information runs the full matrix. Linux's release-pin checks require full non-partial history; students and other jobs use shallow clones without LFS payloads.

Check published `build-info.json` revision and real notebook/download links after deployment. Keep publication revision and actual execution revision separate when reusing results. GPU/Modal work and model/data publication remain manual and require the author's authorization; this maintenance work does not authorize paid compute.

## Storage and Archives

- Ordinary Git keeps teaching sources, small figures, executable public notebooks, implementation, tests, generators, seeds, fixed manifests, licenses, provenance, important scientific results and `lesson-releases.json`.
- Keep temporary outputs, downloads and full training logs in ignored storage. Use existing Git LFS or versioned model/data releases for fixed large datasets and weights; download explicitly and verify sizes and SHA-256 before use. Tests and lesson experiments must not silently fetch assets. Do not overwrite modified local data or published releases.
- Identify real readers before moving files: workflows, code, notebooks and documentary links. Classify by use, not directory names such as `reviews` or `evidence`. Long-term evidence supporting a claim must remain available.
- Retired review machinery and deployment reports are archived at `ccb7c89b4b5ae27e3d318e5e514cec5c1b569e3f`; use immutable links for removed paths. `reviews/vision-branch/sources/` remains tracked because it contains scientific provenance and results. See `reviews/README.md`.
- New raw review sessions do not need a separate private-repository commit or push. The existing private `birdhackor/learn_to_yolo-reviews` remains an optional historical archive, not a deployment dependency.
- Publish and anonymously cold-download/verify replacement assets before removing tracked originals. Preserve attribution, licenses, reconstruction recipes and original bytes. Prefer an existing suitable transport over introducing a new one.
- No history rewrite, tag movement or force push is part of this work. Size measurements may inform a decision but are not prerequisites for adopting a sound mechanism.

## Commits and PRs

Use focused English imperative commit subjects. Explain concrete behavior, affected material, meaningful validation and limitations. Include screenshots when visual behavior changes. Keep credentials out of commits; keep generated website and execution outputs out of Git.

# Application closure — independent fresh-reader review

Reviewer role: basic Python/PyTorch and university mathematics; first reading of this project; unfamiliar with YOLO. Scope: `docs/lessons/18-video.md`, `19-tracking.md`, their notebooks, `lesson_cases/18-video.py`, `19-tracking.py`, `scripts/verify_video_file.py`, diagrams, and the corresponding execution JSON. This initial review was formed independently, without reading earlier reviewers' findings. No教材/case/script changes were made by this reviewer.

## Initial reading record — 2026-10-02

- Lesson 18 is understandable after its stated image inference and letterbox prerequisites. The Frame contract, RGB/BGR conversion, source timestamp versus processing time, restored source coordinates, lazy consumption, and distinction between missed detections and dropped frames are explicit. The last timestamp `.55s` versus playback duration `.6s` is explained rather than left implicit (`docs/lessons/18-video.md:11`, `:13`, `:29`, `:43`, `:51`).
- Lesson 19 explains why perfect boxes can still change identity, supplies the actual crossing geometry, and distinguishes exact matching from greedy matching. `max_age`, the missed-frame gap in velocity prediction, ID order, and the local definition of ID switch are sufficiently concrete (`docs/lessons/19-tracking.md:20`, `:24`, `:36`, `:38`, `:40`, `:46`).
- The new application bridge filters class 0 before matching, calls the tracker even on empty frames, and states that the measured application IDs are not a new tracking-quality score. I can follow why ID 1 disappears and ID 2 is later created (`docs/lessons/19-tracking.md:60`, `:75`, `:77`).
- The FFV1 experiment gives a reproducible generated file, rather than assuming a camera or a downloaded clip. Its pixel-exact comparison is explicitly limited to this lossless fixture, and the timing boundary excludes training, encoding, output writes, and live/display queues (`docs/lessons/18-video.md:65`, `:74`). No physical-camera test or long training is required for this lesson's stated outcome.
- The figures and execution text agree: video panels use frames 0/5/11 with the source timestamps and predicted-box counts; tracking SVG lists the reported 3 versus 0 switches and the same per-frame IDs. The separate A/B display lanes are explained (`docs/lessons/19-tracking.md:44`; `docs/assets/diagrams/19-tracking.svg:2`, `:21`, `:23`, `:39`, `:41`, `:75`, `:78`). Embedded video images were additionally inspected as recorded below; I did not run a browser layout check.

## Required finding

### APP-R1 — Notebook optional verification command does not import the repository package in a clean environment

Initial status: required, open at initial reading. Location: `notebooks/18-video.ipynb:92` and `notebooks/19-tracking.ipynb:92`; the bootstrap changes only the notebook process's `sys.path` at each notebook's `:70`. The optional code cell instructs `!python scripts/verify_video_file.py`, while the webpage command correctly uses `PYTHONPATH=.` (`docs/lessons/18-video.md:71`).

Evidence: from the repository root, `env -u PYTHONPATH .venv-model/bin/python scripts/verify_video_file.py` returned exit 1. The stack was `verify_video_file.py:42` → `load_lesson` at `:31` → `lesson_cases/18-video.py:11`, ending in `ModuleNotFoundError: No module named 'miniyolo'`. The command failed before output creation or its Git metadata call. The initial script SHA-256 was `2cb0b3047e91fb555d36bf0f7f2bedcd7b1bc87542c1266819a7b85fe3eea823`, matching the then-current `artifacts/checks/curriculum/video-file.json:5`.

Reader impact: a reader following either notebook's optional experiment in a fresh Colab process cannot reach the promised file/resource/tracker checks. The kernel's `sys.path.insert(...)` is not inherited by `!python`.

Minimal correction: add `!PYTHONPATH=. python scripts/verify_video_file.py` to both notebook instructions, or have the verification script add its resolved repository `ROOT` to `sys.path` before `load_lesson`. The latter also makes the standalone script usable directly. Author has since informed me that the latter fix was applied; this paragraph preserves the initial finding and does not count as reviewer closure. Independent recheck is recorded separately below.

## Optional improvement

### APP-O1 — Make the lesson 18 own-file snippet's function-loading step explicit for local readers

Location: `docs/lessons/18-video.md:76`–`:84`. “執行本節完整case定義各函數” is clear in Colab after the complete experiment cell, but a local reader who previously ran `python lesson_cases/18-video.py` may not realize those definitions are absent from a later Python process. The subsequent snippet starts with `model = fit_detector()`.

Minimal improvement: provide the same short `runpy.run_path('lesson_cases/18-video.py')` loading pattern used in lesson 19, or explicitly say to execute the snippet in the same notebook/kernel as the full lesson cell. This is optional because the text already states that the functions must first be defined; the actual adapter and closing pattern work.

## Independent execution and evidence

CPU only, repository cwd, existing `.venv-model/bin/python`, PyTorch `2.9.1+cpu`, two threads. No GPU, network, Action, or Git command was run. The script's full `verify()` was not used for the initial functional check because it writes the author's fixed report and calls Git for provenance.

- Executed the exact `runpy` code block extracted from `docs/lessons/19-tracking.md:63`–`:72`. It imported both cases without calling their `main()`, trained once, and printed `0 [1]`, `1 [1]`, frames 2–5 `[]`, `6 [2]`, `7 [2]`, frames 8–11 `[]`. This agrees with `docs/lessons/19-tracking.md:77` and `artifacts/checks/curriculum/video-file.json:63`.
- Independently wrote a temporary 12-frame FFV1 AVI through real OpenCV I/O, then used the lesson's actual adapter and model. File SHA-256 was `1c124d4aae1e30c4eed7f6c2e5cb98613ad1e059bce7f657c78be00c83fd64dc`, identical to the author's fixture SHA (`artifacts/checks/curriculum/video-file.json:34`). RGB pixels, boxes/scores/labels, and overlay pixels all matched the memory stream exactly. Box counts were `[1,1,0,0,0,0,1,1,0,0,0,0]`; last source timestamp was `.55s`. File-to-tracker IDs matched the snippet.
- Wrapped real `VideoCapture` handles to count actual `release()` calls. EOF, `closing` with an early `break`, `closing` with a controlled consumer exception, and open failure each called release exactly once, and the resulting handle was closed. This independently verifies the lifetime behavior described at `docs/lessons/18-video.md:65`, `:76` and implemented at `lesson_cases/18-video.py:41`–`:54`.
- At initial reading, both notebook experiment cells were byte-identical to the corresponding cases, and the case/script SHA-256 values matched the existing `video-file.json`. The lesson 18 median table agrees with its displayed JSON and `artifacts/checks/curriculum/18-video.json:11`; those recorded timings are not claimed to be reproduced by my run.
- Extracted and viewed the exact embedded PNGs from `docs/assets/diagrams/18-video.svg`: frame 0 has a yellow prediction and `.93` score, frames 5 and 11 show the moving red object with no prediction. This agrees with the stated per-frame box counts; no regenerated model panel was substituted for the supplied figure.

Initial execution details are saved in `/tmp/closure-reader-application-initial.json`; the reproducible outputs and result summary above are retained in this review even if that temporary file expires.

## Independent recheck

APP-R1: independently verified fixed after the author's change, not closed by an author assertion. The correction is `sys.path.insert(0, str(ROOT))` at `scripts/verify_video_file.py:24`. Updated script SHA-256: `f7820880f0c89c715c58b2f7ad4682ee9283db4d1a4d76a75787831f155f5b4b`.

Recheck used `env -u PYTHONPATH .venv-model/bin/python -I -` from repository cwd. Before reading the script, neither the repository root nor `''` was present in Python's import path. Loading the corrected script inserted its own resolved root; the complete functional `verify()` then passed real FFV1 encode/decode, pixel/prediction/overlay comparisons, capture lifetime assertions, and the actual tracker bridge. This directly checks the missing import-path prerequisite without relying on a notebook kernel's `sys.path` or inherited `PYTHONPATH`.

To respect the requested no-Git/no-author-report-mutation scope, the Git metadata subprocess was intercepted and the fixed report write was redirected into a temporary directory; no Git process ran. This recheck validates the functional flow, not the commit metadata or the shell's argument parsing. The author's actual `artifacts/checks/curriculum/video-file.json` SHA-256 stayed `60e21b2e17b8ce676a1f9699bec7f615d06beb5a1421fce091cbfdf5d3e80605` before and after; its recorded script hash matches the corrected script. Recheck details: `/tmp/closure-reader-application-recheck.json`.

Status after the first recheck: **0 required unresolved**; APP-O1 remained optional and open at that point. The unpublished `lessons-v0.3.0` ref is a publication step owned by the author, not a reader defect at this review stage.

## Second independent recheck — own-file and tracker snippets

APP-O1: independently verified fixed after the author's documentation change. Lesson 18 now explicitly loads its functions using `runpy.run_path`, sets two PyTorch threads, and explains that a previous standalone shell command cannot define functions in the next Python process (`docs/lessons/18-video.md:76`, `:79`–`:87`, `:90`). This resolves the fresh-reader ambiguity while keeping the initial observation above.

I extracted the latest code blocks directly from the two Markdown files and executed each in a separate, fresh `.venv-model/bin/python -c` subprocess, from repository cwd, with `PYTHONPATH` absent. I did not predefine `video`, `model`, or the lesson functions in either subprocess. Lesson 18's only replacement was the literal `'clip.mp4'` with the path to a temporary, independently generated 12-frame FFV1 AVI. Lesson 19's code block at `docs/lessons/19-tracking.md:63`–`:74` was executed verbatim, including its new `torch.set_num_threads(2)`.

Both subprocesses exited 0 with empty stderr. Lesson 18 printed indices 0–11 and box counts `[1,1,0,0,0,0,1,1,0,0,0,0]`. Lesson 19 printed IDs `[[1],[1],[],[],[],[],[2],[2],[],[],[],[]]`, unchanged from the documented application bridge. The temporary AVI SHA remained `1c124d4aae1e30c4eed7f6c2e5cb98613ad1e059bce7f657c78be00c83fd64dc`.

Executed snippet SHA-256 values: lesson 18 `49b36477f115377be6b841eaaaa7c8ef332221fa0a40a3daefa1f886770999a5`; lesson 19 `27eda04026e0c63a2c29281a7046fe9f5048d93be12c81a54febb54e06497d66`. Details and exact stdout: `/tmp/closure-reader-application-snippet-recheck.json`. Author reports `18-video.json`, `19-tracking.json`, and `video-file.json` were hash-identical before and after. No教材/case/script or root artifact was changed by the reviewer; no network, GPU, Action, or Git command was run.

Final scope status: **0 required unresolved; 0 optional unresolved**. Both APP-R1 and APP-O1 were independently checked after author fixes.

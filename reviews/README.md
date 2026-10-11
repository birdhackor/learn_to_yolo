# 審查紀錄

這個目錄主要放每一頁的審查紀錄（課程頁是 `<節>.md`，其他頁用路徑命名）與 `coverage.json`，另有格式維護核對（`formatting/`）等附件。`scripts/validate_lessons.py`（經由 `scripts/review_coverage.py`）在這個目錄裡只讀逐頁紀錄與 `coverage.json`。

## 逐段原始紀錄（`clear-tutorial/`）不在公開 repo

依 clear-tutorial skill 做的逐段首次閱讀、技術與銜接審閱的原始紀錄，從 2026-10-11 起存在不公開的 repo `birdhackor/learn_to_yolo-reviews`。這批紀錄以網頁截圖為主，留在公開 repo 會讓每次 clone、下載 ZIP 和 Colab 環境格多下載數百 MiB。

- `lessons-v0.6.1` 以前的輪次（`16f6910`、`full-review-2026-10-06`、`vision-v0.6.0`、`release-v0.6.1-2026-10-06` 等）仍可在 [該 tag 的目錄](https://github.com/birdhackor/learn_to_yolo/tree/lessons-v0.6.1/reviews/clear-tutorial) 查看。
- 之後的輪次只在不公開的 repo。
- 要查看或新增紀錄時，把它 clone 到 `reviews/clear-tutorial/`（已列入 `.gitignore`），新紀錄在那個 repo 內 commit。本目錄各頁審查紀錄裡指向 `clear-tutorial/` 的相對連結保留原文，有這個 clone 時才能開啟。

## commit hash 對照

移出時改寫了 `lessons-v0.6.1` 之後的公開歷史：下表的 commit 都換了 hash，內容只少了 `reviews/clear-tutorial/`；`lessons-v0.6.1` 及更早的 commit 與所有 lesson tag 不變。審查紀錄、目錄名稱和舊文字裡的 hash 是改寫前的值；改寫前的完整歷史（所有分支與 tag）封存在不公開的 `birdhackor/learn_to_yolo-archive`，舊 hash 可以在那裡直接查到。

| 改寫前 | 改寫後 | 日期 | commit |
|---|---|---|---|
| `19520ba` | `641a916` | 2026-10-06 | Record lessons v0.6.1 publication verification |
| `851cc8d` | `0af9bc0` | 2026-10-07 | Improve tutorial writing and review guidance |
| `f825cd0` | `bc18ec8` | 2026-10-07 | Rewrite course opening and foundational lessons |
| `a3ac9b7` | `fd6f015` | 2026-10-07 | Record opening and chapter A publication verification |
| `eda087a` | `e9ef78d` | 2026-10-07 | Clarify tutorial rationale and qualification review standards |
| `8fb6ff8` | `3a3f17e` | 2026-10-07 | Generalize tutorial teaching and review criteria |
| `631176c` | `ebfdbb4` | 2026-10-07 | Evaluate three tutorial skill candidates |
| `5dc66a8` | `02e4383` | 2026-10-07 | Evaluate cross-domain tutorial calibrations |
| `d5b4693` | `aa4b637` | 2026-10-07 | Finalize tutorial skill and clarify chapter A |
| `9e3a1e4` | `8774303` | 2026-10-07 | Record public verification of chapter A revision |
| `2f72a20` | `2e9fbc8` | 2026-10-08 | Record three independent tutorial skill stability trials |
| `b387bb5` | `6b2812e` | 2026-10-08 | Adopt tutorial skill with coordinator review of optional findings |
| `694a924` | `ab02109` | 2026-10-08 | Review current opening and chapter A with finalized tutorial skill |
| `9d5b9db` | `3a0c78e` | 2026-10-08 | Record pending opening and chapter A improvements |
| `0eced55` | `fa4c426` | 2026-10-08 | Record distinct neuron labels as a pending diagram improvement |
| `b6eb401` | `9744d0d` | 2026-10-08 | Record diagram review retrospective and Astra advice |
| `93dc8d8` | `4e9c737` | 2026-10-08 | Refine diagram traceability checks in tutorial skill |
| `80308d5` | `2951a98` | 2026-10-08 | Clarify later lessons and repair diagram references |
| `3d84cad` | `230ba70` | 2026-10-08 | Record public verification of the revised lessons |
| `7a8b9d7` | `cf3c672` | 2026-10-08 | Preserve independent public verification samples |
| `e3075da` | `1bdf7a2` | 2026-10-08 | Rewrite B–E lessons around the learning sequence |
| `7810893` | `00de477` | 2026-10-08 | Record public verification of the B–E rewrite |
| `66ef554` | `55cfb4b` | 2026-10-08 | Use section letters consistently in lesson headings |
| `147c0e9` | `e0e955d` | 2026-10-08 | Record public checks for section letter prefixes |

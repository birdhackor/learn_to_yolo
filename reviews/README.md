# 審閱與發布歷史

2026-11-04 將逐頁審閱、coverage 指紋、舊截圖與發布 log 移出目前工作樹。完整公開版本固定在 [整理前的 commit](https://github.com/birdhackor/learn_to_yolo/tree/ccb7c89b4b5ae27e3d318e5e514cec5c1b569e3f/reviews)。這次沒有改寫歷史；原檔仍在該版本，教材引用改用完整 SHA。

新的修改按實際範圍核對易讀性、正確性與連貫性，PR 留發現、修正、驗證與限制。詳細 session、凍結稿、截圖與發布 log 放忽略的 `work/`、`outputs/` 或短期 artifact；一般 CI 保留 30 天，詳細臨時審閱最多 90 天。不需要每輪另外推送私人 repo，也不將歷史審閱作部署依賴。

必要科學來源與實測結果仍長期可查；本目錄的 `vision-branch/sources/` 保留來源版本、授權與預訓練實測資料。`artifacts/checks/curriculum/` 的重要結果照常保留，與臨時執行快取分開。固定資料使用既有 LFS／資料下載流程；先公開並冷下載驗證替代包，再搬移原始內容。

較早的逐段原始紀錄另在不公開的 `birdhackor/learn_to_yolo-reviews`，可按需 clone 到忽略的 `reviews/clear-tutorial/`；它是歷史查詢來源。`lessons-v0.6.1` 以前的輪次仍可在[該 tag](https://github.com/birdhackor/learn_to_yolo/tree/lessons-v0.6.1/reviews/clear-tutorial)查看。以下保留 2026-10-11 的歷史對照，不作現行操作要求。

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

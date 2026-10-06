# 新版 clear-tutorial：方法介紹原句普查

本輪覆蓋導覽中的 **69 頁：52 節課程、17 頁支援文件**。依新版 skill，保存四題的頁面自查與方法介紹的原句對照。覆核後保留 **1 項增加理解負擔、2 項選讀改善**；在這次檢查範圍內未辨認出阻礙主要概念的問題。現行 DINO 22.1 已交代基本介紹，原先的缺漏不再出現在此版。

**本輪只增加審閱紀錄，沒有修改教材、實驗、notebook 或 skill。**

## 版本與實際範圍

- 教材與 skill 的來源版本：[`838ecb7c008183e6048eb417fb46d1472ab35f9c`](https://github.com/birdhackor/learn_to_yolo/commit/838ecb7c008183e6048eb417fb46d1472ab35f9c)。
- 標準：[SKILL.md](../../../.agents/skills/clear-tutorial/SKILL.md) 與其 review-protocol、writing-prompt、README；四題保持不變，三欄是原句證據欄。
- 執行方式：同一位協調者依來源順序自查，非盲讀。逐頁定位方法介紹、目的、步驟關係與已寫出的證據界線，疑點再追讀上下文。不是逐句全文驗收。
- 紀錄：69 組頁面四題、137 筆方法對照。方法依教學重要性選取，包含重複、輔助與歷史背景項目；**137 不是互不重複的新方法數，也不是所有縮寫的完整清單**。
- 時序：有問題的介紹依行號重建早期可見內容，另列後來釐清的位置。沒有當場收取首次使用與節末的獨立讀者答案，不能當成新版 skill 三輪流程的完整驗收。

## 發現與處理建議

| 編號 | 影響 | 位置 | 原文已教什麼／還缺什麼 | 建議 |
| --- | --- | --- | --- | --- |
| MS001 | 增加理解負擔 | 1：softmax，L32、L166；L257 才完整說明基本功能 | logits、交叉熵用途與直接傳 logits 的規則已教；L166 也稱 softmax 的結果為機率。首次提醒前，還缺「把 logits 轉成各類機率」的直接介紹。 | 在 L32 附近補基本作用與交叉熵內部處理的關係；完整公式保留選讀。 |
| MS002 | 選讀改善 | 21.1：ViT 開頭 L5 | 全名、原圖轉 token 的問題、attention 與 CLS 分類目的都在本頁交代。開頭沒有整體路線摘要，但現有逐步教學可支持當前操作。 | 可用一句話串起圖片、patch 向量、attention、整圖表示與分類。 |
| MS003 | 選讀改善 | 08.2 L264 的 RNG；12.1 L9 的 FCOS；12.3 L11 的 TOOD | RNG 的中文含義與存檔用途已教，英文全名在 21.4 L31 才出現。FCOS／TOOD 是歷史引文，當前 anchor-free／TAL 的用途已教；不能判成方法完全未知。 | 早期 RNG 介紹補英文全名；歷史引文可使用完整論文名稱。 |

這些問題維持未修改狀態，具體原句、已教部分、影響、後來釐清位置與建議存於 [issues.json](issues.json)。其中 MS003 是一組名稱補充，包含三處，不能把問題編號數當成出現位置數。

MS001 的來源：[L30–32 的 logits 與交叉熵介紹](https://github.com/birdhackor/learn_to_yolo/blob/838ecb7c008183e6048eb417fb46d1472ab35f9c/docs/lessons/01-small-cnn.md#L30)、[L166 的重複轉換說明](https://github.com/birdhackor/learn_to_yolo/blob/838ecb7c008183e6048eb417fb46d1472ab35f9c/docs/lessons/01-small-cnn.md#L166)、[L257 的 softmax 定義](https://github.com/birdhackor/learn_to_yolo/blob/838ecb7c008183e6048eb417fb46d1472ab35f9c/docs/lessons/01-small-cnn.md#L257)。公式延後本身合理；要改善的是使用前的基本功能銜接。

## DINO 與疑點覆核

現在的 [22.1 L7](https://github.com/birdhackor/learn_to_yolo/blob/838ecb7c008183e6048eb417fb46d1472ab35f9c/docs/lessons/22-views.md#L7) 已明寫「DINO（self-distillation with no labels，無標籤的自蒸餾）」，並區分 ViT 產生表示、DINO 指定如何訓練，以及沒有人工類別答案時的學習目標。[L9](https://github.com/birdhackor/learn_to_yolo/blob/838ecb7c008183e6048eb417fb46d1472ab35f9c/docs/lessons/22-views.md#L9) 交代訓練後保留 backbone，特徵供分類、定位使用；是否有用仍需評估。[L45](https://github.com/birdhackor/learn_to_yolo/blob/838ecb7c008183e6048eb417fb46d1472ab35f9c/docs/lessons/22-views.md#L45) 明確說本頁先生成 view，不更新 ViT。因此，基本介紹沒有延後到 23.1。

ViT 的暫定疑點經覆核後降為選讀改善：[21.1 L5、31、48](https://github.com/birdhackor/learn_to_yolo/blob/838ecb7c008183e6048eb417fb46d1472ab35f9c/docs/lessons/21-patches.md#L5) 的正文已連接原圖 token、attention 與整圖分類。MLP、LayerNorm 的英文全名與分工則已在 [21.2 L52](https://github.com/birdhackor/learn_to_yolo/blob/838ecb7c008183e6048eb417fb46d1472ab35f9c/docs/lessons/21-attention.md#L52) 教過，不能只看 21.3 就報漏寫。暫定判斷、覆核理由及自查草稿清理記於 [adjudications.json](adjudications.json)，沒有改寫既有實驗或盲讀紀錄。

## 紀錄與驗證界線

| 檔案 | 內容 |
| --- | --- |
| [manifest.json](manifest.json) | 69 頁版本、內容 SHA-256、引用圖與四份 skill 檔案的指紋，以及檢查範圍。 |
| [page-notes.jsonl](page-notes.jsonl) | 每頁四題簡式答案、方法三欄原句與判斷、問題連結。定位行號為 `introduction_anchor_line`，不是當場首次閱讀的證明。 |
| [issues.json](issues.json) | 三項發現的具體位置與依據。 |
| [adjudications.json](adjudications.json) | 疑點覆核及自查草稿修訂說明。 |
| [verification.json](verification.json) | 實際執行的覆蓋、欄位、逐字引用與來源不變檢查。 |

檢查確認導覽與全部 docs Markdown 相符，69 頁各有一筆紀錄；每頁四題與每個方法三欄完整，需補說明者連到問題清單。773 次引用均與指定來源行相符；69 頁、74 個 Markdown 圖連結指向的獨立圖檔、四份 skill 與導覽均與凍結版本相同。這些檢查證明紀錄可對照來源，**不能證明每個語意判斷都正確或初學者都能理解**。

本輪沒有新做獨立首次閱讀、技術審查者及銜接讀者的三輪檢查，沒有重跑瀏覽器／圖像版面、CPU／GPU、全部實作或外部歷史來源查證。圖檔指紋只確認版本，沒有視覺驗收作用。先前 [全書 review](../full-review-2026-10-06/README.md) 的程式與版面證據保留原範圍，不算本輪重新通過；[DINO 小實驗](../dino-core-experiment-2026-10-06/README.md) 的舊版來源也不代表現在的教材。

這輪結果可用來安排修補與下一次獨立複查，不宣稱全書符合 skill 的全部完成條件，也不能據此保證零漏報。

# 22.3 從零實作 DINO 核心：v0.6.0 審閱紀錄

日期：2026-10-05。此頁是協調者依原始獨立報告整理的索引，不能取代當場閱讀紀錄。

兩個 global views、舊 center 產生 teacher 目標、stop-grad、student optimizer、teacher／center EMA。CPU160步；80+80恢復狀態及下一批view與連續160一致。

圈號缺字改為 ASCII 1–4，raw圖與實際網站均複查。技術者另用非預設溫度／EMA／crop／LR及 dropout 的3+2續訓重算；官方論文及程式直接查證。

- 第一輪：[逐段首讀](clear-tutorial/vision-v0.6.0/dino-first-reader.md)，[原始 notes](clear-tutorial/vision-v0.6.0/first-read/dino.jsonl)，[局部修後複查](clear-tutorial/vision-v0.6.0/dino-recheck.md)。實際已讀前置及限制以報告為準。
- 第二輪：[非作者技術／原始來源／數值核查](clear-tutorial/vision-v0.6.0/technical-review.md)。
- 第三輪：[另一位讀者前後銜接](clear-tutorial/vision-v0.6.0/transitions-review.md)，十頁 scope 無必改。
- 視覺：[獨立實際 Zensical 桌機／手機](clear-tutorial/vision-v0.6.0/guide-visual-review.md)，包括必要圖與 MathJax。
- 執行：[本節正式紀錄](../artifacts/checks/curriculum/22-distillation.json)，[全部52本 notebook 本機實跑](clear-tutorial/vision-v0.6.0/local-notebook-runtime.json)。程式、圖、正文以 [coverage.json](coverage.json) 的內容指紋約束。

各輪都完成自己的指定範圍；AI 首讀不是真人學生試讀。未執行 Google Colab 託管工作階段、未跑新的GPU、未評測自然照片泛化。官方預訓練實跑只屬23.1選讀，不是全套官方模型重現。

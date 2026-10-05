# 23.1 DINO 版本與官方特徵：v0.6.0 審閱紀錄

日期：2026-10-05。此頁是協調者依原始獨立報告整理的索引，不能取代當場閱讀紀錄。

原始DINO、DINOv2／iBOT／registers、DINOv3 Gram anchoring 分開。預設手工 Gram 例無下載；選讀實際跑固定官方 DINOv2 ViT-S/14，22,056,576 參數，CPU224圖的CLS與patch特徵。

官方預訓練 top1 0.833138 紅塊、top2 0.768546 綠色圓盤都展示；不把相似度當概率。DINOv3只講原論文機制，未訓練。技術者核paper、固定code及實體SHA，再重算保存的features；不冒稱他重跑official forward。

- 第一輪：[逐段首讀](clear-tutorial/vision-v0.6.0/dino-first-reader.md)，[原始 notes](clear-tutorial/vision-v0.6.0/first-read/dino.jsonl)，[局部修後複查](clear-tutorial/vision-v0.6.0/dino-recheck.md)。實際已讀前置及限制以報告為準。
- 第二輪：[非作者技術／原始來源／數值核查](clear-tutorial/vision-v0.6.0/technical-review.md)。
- 第三輪：[另一位讀者前後銜接](clear-tutorial/vision-v0.6.0/transitions-review.md)，十頁 scope 無必改。
- 視覺：[獨立實際 Zensical 桌機／手機](clear-tutorial/vision-v0.6.0/guide-visual-review.md)，包括必要圖與 MathJax。
- 執行：[本節正式紀錄](../artifacts/checks/curriculum/23-dino-versions.json)，[全部52本 notebook 本機實跑](clear-tutorial/vision-v0.6.0/local-notebook-runtime.json)。程式、圖、正文以 [coverage.json](coverage.json) 的內容指紋約束。

各輪都完成自己的指定範圍；AI 首讀不是真人學生試讀。未執行 Google Colab 託管工作階段、未跑新的GPU、未評測自然照片泛化。官方預訓練實跑只屬23.1選讀，不是全套官方模型重現。

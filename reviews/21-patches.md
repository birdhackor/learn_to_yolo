# 21.1 圖片切成 patch：v0.6.0 審閱紀錄

日期：2026-10-05。此頁是協調者依原始獨立報告整理的索引，不能取代當場閱讀紀錄。

從可見 RGB 材料開始，patch 為 row-major，raw 192 維與 learned 32 維分開；CLS／位置向量的意義在使用前交代。

首讀指出「前面」指向尚未出現的實驗，已改成下一小節並局部複查。

- 第一輪：[逐段首讀](clear-tutorial/vision-v0.6.0/vit-first-reader.md)，[原始 notes](clear-tutorial/vision-v0.6.0/first-read/vit.jsonl)，[局部修後複查](clear-tutorial/vision-v0.6.0/vit-recheck.md)。實際已讀前置及限制以報告為準。
- 第二輪：[非作者技術／原始來源／數值核查](clear-tutorial/vision-v0.6.0/technical-review.md)。
- 第三輪：[另一位讀者前後銜接](clear-tutorial/vision-v0.6.0/transitions-review.md)，十頁 scope 無必改。
- 視覺：[獨立實際 Zensical 桌機／手機](clear-tutorial/vision-v0.6.0/guide-visual-review.md)，包括必要圖與 MathJax。
- 執行：[本節正式紀錄](../artifacts/checks/curriculum/21-patches.json)，[全部52本 notebook 本機實跑](clear-tutorial/vision-v0.6.0/local-notebook-runtime.json)。程式、圖、正文以 [coverage.json](coverage.json) 的內容指紋約束。

各輪都完成自己的指定範圍；AI 首讀不是真人學生試讀。未執行 Google Colab 託管工作階段、未跑新的GPU、未評測自然照片泛化。官方預訓練實跑只屬23.1選讀，不是全套官方模型重現。

## 2026-10-06：最新版 clear-tutorial 全套重審

本次以 `64a25d4fbcff5577965c29efbbcb5d9898ba95d9` 凍結來源從頭閱讀，不把以前的審閱當作此次首次閱讀。方法、完整範圍與限制見[本輪報告](clear-tutorial/full-review-2026-10-06/README.md)。

- 首次閱讀：主要讀者 `vision` 實讀本頁 7 個凍結單元；首次使用／前文方法範圍四題位置為 21-patches/00:first_use, 21-patches/03:first_use, 21-patches/04:first_use，頁末為 21-patches/06。[當時理解與問題](clear-tutorial/full-review-2026-10-06/first-read/vision.jsonl)與[分段披露](clear-tutorial/full-review-2026-10-06/first-read/vision-disclosures.jsonl)按原樣保留；實際前置閱讀見[該組報告](clear-tutorial/full-review-2026-10-06/reports/vision.json)。
- 處置：[決策表](clear-tutorial/full-review-2026-10-06/decisions.json)。本頁處置：R010；各項原位置、分級、實際改寫／保留理由見決策表。
- 非作者技術／證據：[本頁所屬報告](clear-tutorial/full-review-2026-10-06/rechecks/technical-vision.json)，只以報告列出的正文、實作、數值、圖與實際執行範圍作結論。
- 另一位讀者的前文→本節→後文與網站：[第三輪紀錄](clear-tutorial/full-review-2026-10-06/rechecks/transitions-visual.json)。52節正文有閱讀紀錄；實看圖／公式的頁面與截圖另列，不將捕捉或DOM載入當成每張圖可讀。

本輪未留下已裁定的必要問題。所有讀者均為 AI，沒有真人學生學習效果驗收。原首讀中仍有漏報、引用未支持全部主張及明說／推論混分，見[獨立裁定](clear-tutorial/full-review-2026-10-06/rechecks/record-adjudication.md)；不能宣稱四題保證抓到所有缺漏或原始紀錄嚴格規則全合格。程式與依賴、正式CPU紀錄、Notebook、建置和全站掃描的實際檢查見[驗證結果](clear-tutorial/full-review-2026-10-06/verification.json)。本頁最新文字、所用SVG／raster圖片與實驗依賴綁定在[coverage.json](coverage.json)。

# v0.6.1：方法介紹的局部修正與複查

本輪承接[方法介紹普查](../method-survey-2026-10-06/README.md)，基準 commit 為 `561860b0a07d88eaff4c79bdabe6bafb011e69cc`，目標 tag 為 `lessons-v0.6.1`。這是修改處的有界複查，不是全書新的首次盲讀驗收；所有審閱者均為 AI，沒有真人學生學習成效測試。

## 修改與處置

實際改寫五頁正文，詳見[處置表](decisions.json)：

- 第 1 章：在要求直接交入 logits 前，先說 softmax 把分數轉成類別機率，以及交叉熵的作用。首讀另外發現開場把前節的一次更新說成「已學會把 2 變成 4」；改成預測從 2 拉近答案 4，保留原卡點及修後確認。
- 第 21.1 節：開場補上 patch 向量、attention、整圖表示與分類的整體路線，交代本節準備向量的用途。
- 第 8.2、12.1、12.3 節：補 RNG、FCOS、TOOD 的完整名稱與中文意義；保留它們原本的用途及歷史引用範圍。

另將目前版本的入口更新為新 tag：52 本 notebook 的環境格與 metadata、52 節 Colab 連結，以及 README 和八個支援頁面的目前版本參照。歷史驗證範圍沿用原紀錄。[變更範圍核對](scope-audit.json)確認模型、課程實驗程式及 skill 未改，52 本 notebook 的實驗格與保存輸出也未改。

## 審閱方法與原始紀錄

1. **首次閱讀者**只收到凍結的指定文字批次，不提供作者答案、差異或技術審查。第 1 章先給實際前節與開場，再記錄判斷；修正開場後局部核回，才提供後半頁。第 21.1 節先給實際相關前文與開場，再提供餘文。[原始輸入](reader-input/)與[內容指紋](source-fingerprints.json)保留；[第 1 章初讀](first-reader-01-start.json)、[修後核回](first-reader-01-correction.json)、[節末](first-reader-01-end.json)、[21.1 初讀](first-reader-21-start.json)、[節末](first-reader-21-end.json)均未覆寫。RNG／FCOS／TOOD 只做[名稱附近的局部閱讀](first-reader-names-local.json)，不算三頁完整首讀。
2. **非作者技術審閱者**核對五頁的相關正文與實作、官方文件／固定版本來源、52 本 notebook，以及六份刷新紀錄；另在獨立 CPU 小例核對交叉熵和亂數／checkpoint 行為。[技術報告](technical-review.json)、[靜態核對](technical-static-audit.json)、[來源清單](technical-sources/manifest.json)及[局部執行結果](technical-cpu/result.json)列出實際範圍，沒有必要修正項目。
3. **另一位銜接審閱者**讀實際相關前文、本次修改與下一節，核對版本／歷史驗證說法，實際觀看五個修改段落的桌面 1280px、手機 390px 共十張 Zensical 截圖。[第三輪報告](transitions-visual-review.json)未發現必要銜接或視覺問題；[截圖與捕捉資訊](browser/)保留。

首讀的主要修改概念留有開場與節末紀錄，但餘文以整批提供，沒有替每個中間新概念重新設逐段關卡；首讀者也未看圖。名稱補充是局部文字複查。第三輪僅實看修改段落，不涵蓋全書圖片、換頁公式或所有裝置。這些紀錄不能擴稱全頁或全書嚴格首讀通過；其餘既有審閱仍保留原範圍。

## 執行與發布前檢查

依 AGENTS.md 實際重跑六個受影響的 CPU 課程：00-warmup、01-small-cnn、08-own-data、12-anchor-free、12-assignment、21-patches；[執行 log](curriculum-refresh.log)及[新舊輸出核對](output-comparison.json)確認六份 stdout、實驗程式與依賴指紋相同。其他 CPU／GPU 紀錄依來源指紋仍有效，未啟動 GPU 訓練。

既有測試共 101 項通過，見 [pytest log](pytest.log)。嚴格網站建置見 [build log](build.log)；manifest、課程配對／審閱、執行證據與生成網站的最終檢查見 [verification.json](verification.json)。本輪每頁審閱以追加方式記錄，更新 coverage，不改寫歷史問題。

公開發布驗證要在新 tag 推送及 Pages 部署後才執行，結果保存於 repository 的 `artifacts/checks/curriculum-publication.json` 與 `artifacts/checks/curriculum-release-bootstrap.json`，以各自 `source_ref` 為準。這份發布前紀錄不聲稱新版公開驗證已完成，也不聲稱在 Google Colab 的託管 runtime 實測。

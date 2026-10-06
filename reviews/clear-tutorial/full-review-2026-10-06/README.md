# 2026-10-06 全套教材重審

本輪依最新版 clear-tutorial 重讀全部 52 節，改寫其中 26 節的介紹、圖文對應、讀數與證據說明，並同步首頁、詞表及維護頁。凍結來源為 `64a25d4fbcff5577965c29efbbcb5d9898ba95d9`；本輪沒有修改 SKILL、`lesson_cases/`、`miniyolo/` 或發布 tag。Notebook 的完整實驗程式仍與 `lessons-v0.6.0` 相同，這次更新的是保存輸出。

主要修正：

- 22.1 就介紹 DINO 的完整名稱、無標籤自蒸餾的問題、teacher/student 分工、ViT 架構與訓練方法的差別，以及保留下游可用特徵的目的；不把這些必要介紹延至22.3。
- 21.2 在使用完整 block 的排列性質之前，給出 MLP、LayerNorm 與 residual 的當下角色和理由；名稱補充以真正首次使用處為準。
- 第4章嵌入同次CPU的實際定位與前處理圖；第8章讓框號直接連到類別、score、TP/FP；第17–19章用真實結果的直式衍生圖，保留像素、框、ID及時間。
- 第17章另觀察原程式的10+160+160步梯度和權重更新，明列補充檢查不屬原notebook斷言、不用其開銷計速度；非作者另外比較probe與無probe的最終權重，逐位元一致。
- 對齊本次CPU的正文數字、曲線、框與正式紀錄；21/22曲線與23框圖內綁定來源JSON/SHA，第22.4明說上下圖取不同次執行。22.3解釋紀錄是裁剪前梯度長度。

## 首次閱讀與紀錄限制

六位新讀者以無作者對話的上下文開始，各自真正補讀指定前文。完整路線分配52個目標頁，每頁只指定一位主要讀者，前提閱讀有重複；共享檔案系統靠遵守範圍隔離，沒有技術存取隔離。

共保存995個閱讀單元、212個四題檢查點：149次首次使用、11次前文已教方法的範圍檢查，以及52次頁末。每段記錄完成才開下一段。原始理解、問題與後來釐清不回寫，見 [first-read](first-read/)、[披露與原始來源](manifest.json)、[結構統計](first-read/structural-summary.json) 及 [各組最終報告](reports/)。

[最後結構與原檔核對](rechecks/final-structural-check.json)確認143份凍結檔與基線逐位元相同、記錄先於下一段披露，以及53本notebook的程式格未改；這些是結構／來源檢查，不是語義理解通過。

**四題不能保證不漏抓。** 本輪vision讀者仍把22.1的DINO必要介紹當成後教，沒有列問題；獨立查核者另確認這個漏項，不能將作者補寫算作該讀者成功發現。另有BatchNorm/iBOT名稱漏項，以及答案包含多個核心主張卻只引用其中部分、把合理推論標為正文明說的紀錄缺陷。正文已有支持的情況不改判成教材完全沒教。

[獨立紀錄裁定](rechecks/record-adjudication.md) 人工檢查52頁的主首次／頁末共104個檢查點，並補查命名、unknown、問題與前文；212個檢查點的848答案／1680 claims另作字面引文與順序檢查。後者不等同1680條語義支持逐項證明，本輪不能統稱嚴格盲讀規則全部合格。原始筆記照原樣保留。

## 修正後的查證

[decisions.json](decisions.json) 區分首次讀者發現、作者自查、獨立紀錄裁定與第二／三輪發現；每項都有來源、分級、處理與保留理由，不把後文澄清倒填為早期已知。

非作者技術報告分工如下：

- [基礎](rechecks/technical-foundations.json)：00、01、03 identity/comparison、04兩頁與初期詞表／維護頁；保存該時點快照。之後的00及維護delta由下份再核。
- [偵測與演進](rechecks/technical-detector-evolution.json)：其他30頁及00／詞表／維護delta，實際讀正文、實作和保存證據，按疑點短CPU與一手來源核對。
- [應用與06/08](rechecks/technical-applications.json)：17–20、06評估及08自訂資料；含獨立梯度/權重、衍生圖保留與官方來源核對。
- [ViT／DINO](rechecks/technical-vision.json)：21–23十頁；核圖／正文／JSON／notebook、一手來源、排列性質與裁剪、最新曲線和框。

四份技術報告合計覆蓋52節。報告的每個指紋只證明它記錄的時點；總結文件、決策與索引稍後追加不當成模型程式改動。最新綁定另保存於 final-bindings.json。

[第三輪](rechecks/transitions-visual.json) 由另一位讀者從實際前文讀到本節與下一節，保存52節及首頁／路線／詞表的閱讀證據，另看真實桌機和手機頁面、圖、MathJax與instant navigation。已捕捉但未實看的圖不算視覺通過。第4章長公式拆行後另核最新站點；部分圖（含第4、20章）仍有手機細字偏小，相鄰正文已有必要對應，逐頁保留為可讀性改善，沒有概括宣稱所有手機圖字通過。

全套73個正文圖位置的桌機／手機共146個呈現，以及3張折疊圖的6個呈現，已逐張實看。20頁的手機輔助細字限制保存於第三輪的 TVIS01／02／04／05，決策 R040 明列保留理由。長圖截圖裡的固定頁首遮罩另用相同CSS尺寸重捕補核，不把截圖瑕疵當成圖source缺字。這仍不是每段版面、每個公式或所有折疊程式的逐畫面驗收；無圖頁的正文閱讀另如實記錄。

所有讀者均為AI，沒有真人學生學習效果實驗；沒有新增GPU、自然照片全訓練或官方DINO重訓。舊L4及其他補充紀錄仍以原保存範圍為準。

## 重跑與驗證

每個改過的課程頁已用 `scripts/verify_curriculum.py --section <id>` 更新實際CPU證據、notebook保存輸出與頁尾。52節與12項補充紀錄的程式／依賴指紋仍有效。101項pytest、課程／準備／證據／網站驗證與strict build的實際結果另保存於 [verification.json](verification.json)。

全站瀏覽器掃描保存56頁×桌機／手機共112個DOM檢查及圖／開場捕捉，見 [browser-final/results.json](browser-final/results.json)。載入、外層溢出、MathJax及runtime檢查不等同逐圖可讀性判定；真正實看範圍見第三輪報告。基線掃描與代表圖在 [browser-baseline](browser-baseline/)。

重跑命令（repo根目錄）：

```bash
.venv-model/bin/python -m pytest tests/ -q
python3 scripts/validate_preparation.py
python3 scripts/validate_lessons.py
python3 scripts/validate_curriculum_evidence.py
.venv-docs/bin/zensical build --clean --strict
python3 scripts/validate_site.py
python3 reviews/clear-tutorial/full-review-2026-10-06/reproduction/check_structure.py
```

補充第17章更新檢查：

```bash
.venv-model/bin/python reviews/clear-tutorial/full-review-2026-10-06/reproduction/check_capstone_updates.py
```

它真的執行10+160+160步，會重寫自己的檢查紀錄；另一次執行的耗時不能替換正常CPU紀錄。首讀gate/setup與瀏覽器掃描程式在 reproduction/，保留本輪環境路徑，需要按自己的workspace調整；不要重跑setup覆蓋既有紀錄。原凍結頁可直接從 frozen/ 或凍結commit取出，原SVG來源以manifest和該commit為準，預覽在 previews/。

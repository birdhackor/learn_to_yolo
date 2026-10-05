# clear-tutorial 重審與改寫

這次從 `16f69103d1f216d1024a40610623083557324701` 的教材重新讀起，保留「當時讀到這裡理解了什麼」及卡點，不用讀完全文後的專家摘要代替。審閱依 [SKILL.md](../../../.agents/skills/clear-tutorial/SKILL.md) 與 [review-protocol.md](../../../.agents/skills/clear-tutorial/references/review-protocol.md) 執行；AI 能幫忙找問題，這不是實際學生學習成效測試。

## 修改方向

問題集中在：先給符號再給材料、主例還沒走完就插入歷史與 API 細節、同一頁混合手算／訓練／評測的證據，以及圖太小或和文字的索引不一致。

首頁、暖身與 CNN 改成先看任務、圖片及答案，再逐步接到計算、更新與結果。後續章節把必要比較條件留在正文，重跑指令、長推導和歷史細節改為選讀；圖中的框、責任格、target 與輸出保持一致。新增的窄版圖涵蓋素材、資料流、座標、尺度、裁切、雙 head 和實測結果。[作者處理說明](author-changes/root.md)及其他作者報告記錄實際改動，不作為獨立通過證據。

技術輪也抓到真正的錯誤，例如 YOLOv10 的 top-1 初選不能保證衝突後每 GT 最多一正樣本；用固定官方方法跑出反例後，修正13章及16章的回顧，並和 YOLO26 衝突後的 topk2 分開。原方法、輸入、plain JSON、來源 SHA 與上游授權保存在 [technical/probes](technical/probes/)。

## 第一輪：凍結版本、逐段開放

六位讀者以不繼承作者歷史的上下文開始。每次取得一個單位，先保存理解、材料／圖的含義、下一步預測、原文卡點與小變化推理，才開放下一單位。共享檔案系統只有行為規範，沒有技術隔離。

| 組別 | 實際記錄 | 目標單位 | 目標頁數 | 原始紀錄 |
| --- | ---: | ---: | ---: | --- |
| foundations | 110 | 110 | 14 | [JSONL](first-read/foundations.jsonl) |
| grid | 131 | 58 | 7 | [JSONL](first-read/grid.jsonl) |
| evolution | 148 | 84 | 11 | [JSONL](first-read/evolution.jsonl) |
| modern | 109 | 48 | 8 | [JSONL](first-read/modern.jsonl) |
| applications | 100 | 35 | 4 | [JSONL](first-read/applications.jsonl) |
| reference | 143 | 133 | 15 | [JSONL](first-read/reference.jsonl) |
| 合計 | 741 | 468 | 59 | 其餘273單位是實際補讀前提 |

每筆紀錄保存來源 commit、頁面／閱讀單位／圖指紋、行號、當時筆記。原圖靜態預覽39張保存在 [first-read-figures](first-read-figures/)，SHA 映射見 [source-manifest.json](source-manifest.json)；這些是原 SVG 預覽，不是 Zensical 手機頁面。全文可由固定基線和行號重建；[gate-source.txt](gate-source.txt)保存實際協調程式。

三個協調者限制另有保留，沒有當成讀者或教材的錯：

- grid 早期漏發07-data；在原過程中插入7個補讀單位，先前疑惑沒有覆寫。
- applications 的前提包漏發06-evaluation；它不懂 AP 算法的紀錄不能證明教材未教 AP。第三輪實際先讀06再接07。
- grid 在讀完08前收到作者改寫brief，所以08後段不能宣稱完全未受提示的首次盲讀。第三輪改由此前未讀08全文的 modern 讀者，先實讀06、07前文，再順讀08，記理解後才看其他讀者原問題；這是獨立順序複查，不冒稱全新上下文或逐段盲讀。

原始卡點73項（含前提與重複觀察）保存在 [decisions.json](decisions.json)。保留原疑問，另記具體處理與真正核回；選讀圖解未新增的地方明列保留，不假稱全部建議已實作。

## 第二輪、第三輪

第二輪由未參與該組改寫的人讀實作、數字與來源，必要時在隔離副本執行。第三輪再換一位讀者，依實際前文→本節→下一節檢查主線與概念轉換。第三輪可以讀全文，明確是非盲讀的銜接查核；不是把第二輪答案交給讀者照著說。

| 範圍 | 非作者技術核對 | 另一讀者銜接複查 |
| --- | --- | --- |
| 首頁／路線與00–06、07-data | [foundations](technical/foundations.md)；首頁與路線由reference、07-data由grid核技術 | [foundations](transitions/foundations.md) |
| 07–08 | [grid](technical/grid.md) | [grid](transitions/grid.md) |
| 09–12 | [evolution](technical/evolution.md) | [evolution](transitions/evolution.md) |
| 13–20 | [modern-applications](technical/modern-applications.md) | [modern-applications](transitions/modern-applications.md) |
| 其他15頁與README | [reference](technical/reference.md) | [reference](transitions/reference.md) |

各報告列實際輸入、圖與 source 指紋、理解／變化推理、發現與修正後核回。圖作者的瀏覽器自查不等於獨立技術或讀者通過。以前的完整頁面審查仍在 `reviews/`，沒有追溯改成首次盲讀。

## 執行與視覺證據

模型、訓練程式與資料生成沒有變更；對改動章節重產真實 CPU 輸出。重跑程序的模型數字、框及指標重現，03與17的計時變動依新紀錄處理；這不是效能評測。保留已驗證且源碼指紋未過期的160/1600步、GPU/checkpoint、TensorRT等既有紀錄，不為文字修訂加開GPU。

網站已實際開啟59頁、1280與390兩種視窗，核圖片、數學與整頁溢出；重要新圖另有可視檢查。MathJax也檢查換頁、返回、錨點、快速換頁與延遲工作佇列。最終實際結果見 [verification.json](verification.json)；公開網站與全新checkout的發布後結果沿用專案的 [curriculum-publication.json](../../../artifacts/checks/curriculum-publication.json)及 [curriculum-release-bootstrap.json](../../../artifacts/checks/curriculum-release-bootstrap.json)。

沒有實際真人讀者、Colab服務上的逐節執行或真實相機測試；不把AI與本機／Actions CPU檢查當成那些驗證。

## 新版發布後核回

`lessons-v0.5.0`固定在`a6ec0d60c65ba97eeef9dc55c2e7d110b34f5edb`。Pages [部署](https://github.com/birdhackor/learn_to_yolo/actions/runs/37299823086)與[全新tag驗證](https://github.com/birdhackor/learn_to_yolo/actions/runs/37299900616)已完成。公開59篇文章／59張SVG逐位元符合strict build，42個教材notebook的pin及最後一格符合tag，4個先前tag保持原commit。

乾淨Linux環境實際跑5種notebook環境情況、README16條指令、42個lesson cases及72個tests；42cases的assert皆通過，36份stdout逐字相同，6份stdout不同。後者在bootstrap摘要只保留case ID，沒有逐案完整差異，不能把它們全稱為已核實的純計時差。這與本機重產30cases和基線比較只變03/17計時是不同檢查；正文以指定機器的保存紀錄解讀，不把不同runner數字當新效能評測。

公開頁面另實際開6頁的桌面／手機呈現，公開MathJax也完成換頁與捲動，詳browser目錄。發布後紀錄寫入main後不移tag；新tag中的前期verification狀態仍如實保留在git。

# grid 作者修改紀錄

作者範圍：`docs/lessons/13-dual-assignment.md`、`docs/lessons/17-capstone.md`、`docs/lessons/08-own-data.md`，以及專用、非實驗程式生成的新圖 `docs/assets/diagrams/13-dual-head-flow.svg`。沒有修改 lesson_cases、miniyolo、scripts、notebooks、reviews、coverage；沒有 commit、push、訓練或 GPU 執行。下列是作者處理及自查，並非獨立驗收。

已讀 `.agents/skills/clear-tutorial/SKILL.md` 與 `references/review-protocol.md`，遵守主線先於選讀、例子切換及證據範圍的要求。使用者指定不改 notebook／執行來源，故沒有按一般 repo 流程再生成證據；三頁既有 `curriculum-evidence` 區塊與凍結版本逐字相同。

原始問題來源：`reviews/clear-tutorial/16f6910/first-read/modern.jsonl`、`applications.jsonl`、`grid.jsonl`。只按三頁篩選實讀完整原始 notes，包括 applications 中 08-own-data 的 prerequisite 紀錄。

## 按 issue／unit ID 處理

| 原始 ID | 處理 | 保留的限制／待複查 |
| --- | --- | --- |
| modern-0062（13 主線負擔） | 品質表先走每 GT top-1、p0 衝突及具體 owner，再切到可執行程式的全域 toy。全域枚舉、greedy、Hungarian 放選讀；toy 練習另標選讀。 | 程式仍是 exact 全域最優，沒有改成 top-1。人工表衝突仍用品質，數值 owner 明寫不是官方 CIoU 的程式輸出。 |
| modern-0061／0063／0064 的圖解需求 | 新 SVG 同時標訓練特徵、兩路 loss 梯度、detach 停止位置與官方推論只留 one head。圖內各自標官方 top-10/top-1 與 toy top-2/全域最優，並說明官方有框+分類，toy 只有分類。 | toy 沒量完整偵測 AP／速度；推論 top-k 與訓練 per-GT 選法不同。 |
| modern-0064（兩物件必有兩正） | toy 在候選足夠時要求每 GT 恰好一個；官方 top-1 的正文說明為每 GT 最多一個，衝突後可沒有。不再把兩物件直接推成兩個正樣本。 | 相同品質公式只支持每 GT 排名的條件推論，不保證 head 訓練後預測相同、候選主人相同或無重複偵測。 |
| applications-0067（17 編號） | 原位置明寫「validation 圖片 #4」與「不是預測編號」。 | 圖片編號與同圖內排序索引仍有不同用途，不能跨模型用排序索引辨認同一框。 |
| applications-0068（17 圖的 IoU 判讀，optional） | 緊鄰原圖增加 #4、#14 的 IoU／定位 FP 三欄表；class 0 與高分範圍放表前。手機自查發現四欄版太寬，改為三欄。 | 不手改會被 `save_panel`／verify_curriculum 覆蓋的 SVG。表中 0.4917 仍低於 0.5，不由「看起來更近」判 TP。 |
| applications-0061（08「接續」） | 正文與操作折疊都明寫：第二次從相同 seed／初始化獨立重新訓練總1600步；prior-diagnostic只讀報告核對，不接160步checkpoint／Adam。 | 160／1600是已有數據；協調者 brief 的3200誤寫未帶入教材，未新增實驗。 |
| applications-0062（08 非零梯度≠參數更新） | 梯度表說明改為有限且非零的反傳訊號；另對照 optimizer.step 與訓練前後整體參數 L2 差值。 | 沒有逐步參數差值紀錄，不能宣稱每一步參數實際改變；CPU 重載一致也未證明接續訓練路徑一致。 |
| grid-0120（08 診斷與指紋旁支） | 160 失敗診斷直接接獨立1600結果、比較前提與曲線。SHA-256、重建舊fixture、輸出目錄及兩組命令集中在「自己跑」選讀。 | 保留相同 train/validation、固定設定、新 test、最後步模型及 test 不選參的必要條件。 |
| grid-0121（08 圖解與 CLI／保存旁支） | 曲線→四圖逐例TP/FP/FN→整體mAP限制連續呈現；CLI相容性、梯度/ckpt、計時分成操作選讀。 | 保留 letterbox 座標、四圖取樣方式、score/NMS/配對設定與小合成資料不能代表真實泛化的限制。 |

08 部分後段的 grid 原始紀錄收到提前作者 brief，協調者已記錄其盲讀限制；此處沒有將它冒稱完全獨立新發現。原始紀錄沒有覆寫，也沒有自行將問題標 closed。

## 回查的程式與證據

- 13：實讀 `lesson_cases/13-dual-assignment.py` 的 exact/greedy/many owner、target、detach、兩次 backward 及 step 的檢查，確認正文仍描述原 toy。官方機制沿用本頁既有固定 commit 的論文、head.py、tal.py 來源；此作者修改沒有重新外部核證官方全套實作。
- 17：實讀 `lesson_cases/17-capstone.py` 的 draw、tile、tag_svg、save_panel，及 `scripts/verify_curriculum.py` 的 SITE_FIGURES 複製入口。新增表逐筆對上 `artifacts/checks/curriculum/17-capstone.json` 的 stdout report：baseline 圖4 prediction0/1 IoU 0.4917936623／0.3034204543；baseline 圖14 0.2275093794；changed 圖14 0.4916878343。未重訓。
- 08：回查 `scripts/record_evidence.py` 的160與1600命令；實讀 `scripts/run_custom_data_learning.py` 的 seed、prior report核對、重新建model/optimizer、訓練與重載流程，以及 `miniyolo/train.py` 的 optimizer_step。沒有將 report 核對誤當續訓，也沒有執行這些訓練指令。

## 作者檢查結果

- `python3 scripts/validate_preparation.py`：通過（manifest、section pairing、notebook code／發布邊界）。
- `python3 scripts/validate_lessons.py`：未通過，停在正文／圖修改後 review 指紋不涵蓋 current text 的清單，包括本三頁及其他協作者的既有修改。沒有改 reviews 或 coverage 來消除檢查。
- SVG XML 解析通過；三頁原有執行證據區塊與 snapshot 逐字比較一致。
- `.venv-docs/bin/zensical build --clean --strict`（短暫 root 配置、輸出到 ignored grid-author-site）：通過。暫時配置已清除，沒有變更原 zensical.toml。
- Playwright + 現有 `/usr/bin/chromium`：實際看三頁 1280×800、390×844 的 Zensical 呈現；圖片均載入，整頁無超過 viewport 的橫向溢出，13 MathJax 公式正常。對新13圖及17新增表另保存相同寬度的 component 截圖。13手機圖字／梯度路徑可辨；08品質、獨立重跑说明與選讀可辨。17最後三欄表已實看桌面及手機截圖：手機表寬311px，小於343px的正文容器，三列IoU與判定完整可讀，不需橫向捲動。

作者預覽：`artifacts/runs/clear-tutorial-16f6910/grid-author-browser/`（results.json、results-17-final.json、desktop/mobile inline、figure、math、IoU table、08 rerun）。這些是作者排版證據；修改後獨立首次閱讀、技術與前後銜接仍待協調者安排，不能由此標通過。

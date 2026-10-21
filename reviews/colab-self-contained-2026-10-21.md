# 教材內嵌程式與 Colab 選跑：2026-10-21

部分程式摘錄省略了初始化、更新檢查或練習需要的執行順序，讀者要開完整 notebook 才能核對。這次把理解所需的片段與題目前提補回原位置，並明說 Colab 供想實跑的人選用。

基線為 `eec3edc59003ec2775578025a436613844a8d72a`。檢查範圍包含兩個閱讀入口與全部 52 節；局部修改 11 節及兩個入口，章節順序、圖、實驗程式、notebook、固定 tag 與既有實測數據不變。

| 頁面 | 局部修改 |
| --- | --- |
| [首頁](../docs/index.md)、[閱讀路線](../docs/learning-path.md) | 說明必要程式在教材內、實跑可選；修正兩個過時的排錯定位。 |
| [A.0](../docs/lessons/00-warmup.md) | 補模型建立、w=1 初始化及更新後重新預測，對照原手算。 |
| [A.3.1](../docs/lessons/03-identity.md) | 補原樣捷徑探針的 block 建立與主分支歸零。 |
| [B.7.4](../docs/lessons/07-training.md) | 保留迴圈的 shape、有限梯度、權重變化與正格數檢查，供刪 step 題判斷。 |
| [B.8.2](../docs/lessons/08-own-data.md) | 補 classes、洩漏副本與輸出 shape 斷言，對應 anns 的角色與座標。 |
| [C.12.3](../docs/lessons/12-assignment.md) | 補練習用的分數與四個檢查，保留真實執行順序。 |
| [C.12.4](../docs/lessons/12-dfl.md) | 補初始梯度、訓練後機率／期望檢查及相鄰 bin 的範圍斷言。 |
| [C.15.1](../docs/lessons/15-attention-bridge.md) | 補七個 print 與斷言的先後，以及改 target 後三塊梯度仍含非零值的已知條件。 |
| [C.16.1](../docs/lessons/16-dfl-free.md) | 補改右邊距離時要對照的 loss、範圍與解碼檢查。 |
| [C.16.2](../docs/lessons/16-inference-head.md) | 把 top-k 練習的原 shape 斷言放在題前。 |
| [D.20](../docs/lessons/20-deployment.md) | 題前給固定例的 score 上界，讓學生能預測首個失敗點；實跑措辭改為選用。 |
| [E.21.4](../docs/lessons/21-training.md) | 補齊 train_step 的梯度分組、optimizer.step 與回傳，保留原函式。 |

## 三輪獨立審核

三類審閱由不同、未參與撰稿的 AI 負責。初判先封存，後續修稿核回另存，不回寫原答。

- **易讀性審查**：全部 52 節逐頁篩查 Colab 依賴；原定 11 個改動頁按必要閱讀單位保存理解，入口兩頁明記為有界重讀。另發現 C.15.1、D.20 的兩個題目前提缺口，修後追加核回。[原始報告](https://github.com/birdhackor/learn_to_yolo-reviews/blob/main/colab-self-contained-2026-10-21/readability/report.md)
- **正確性審查**：核對最終 13 頁的改動小節、源碼、notebook 與保存證據，於獨立 CPU 副本執行案例及練習變體。E.21.4 以單步與短續訓核對新增函式，原 60 步曲線和 held-out 數據沿用。後加範圍與修稿查核標明提示來源。[實際執行範圍](https://github.com/birdhackor/learn_to_yolo-reviews/blob/main/colab-self-contained-2026-10-21/correctness/README.md)
- **連貫性審查**：核對必要前文、改動小節、練習及相鄰後文，並檢查桌面與手機的片段位置、折疊與橫捲。原判、修後核回與實際視判分開保存。[最終報告](https://github.com/birdhackor/learn_to_yolo-reviews/blob/main/colab-self-contained-2026-10-21/coherence/final.md)

這是局部修補的三輪核對，加全課外部依賴篩查；不是全書重新盲讀，也不代表真人學生學習成效驗收。沒有為未改的科學結果重產數據或重驗 GPU／公開 Colab 託管環境。

## 發現與處理

| 編號 | 分級 | 處理 |
| --- | --- | --- |
| R01 | 必要：題目前提 | C.15.1 補七個 print 與梯度斷言的先後。 |
| C03 | 必要：修後查漏 | 同題補三塊梯度仍各含非零值，才能排除前一條斷言先失敗。 |
| R02 | 必要：題目前提 | D.20 給出固定三來源的所有候選 score<0.07；獨立實跑確認兩端最高約 0.06095，變體確於 B=1 的非空斷言停止。 |
| C01 | 可選：既有導航 | 排錯導引改指 README 本機步驟與 A.0 實有的操作補充。 |
| C02 | 可選：變數對應 | 說明 anns 與 annotations 的角色相同、框均在 64×64 座標。 |
| C04 | 可選：執行措辭 | D.20 練習可先依頁內材料作答，再選擇實跑。 |

六項均採用局部修正；原始分級、理解收益、修正鏈與前後指紋保存在 [repairs/manifest.json](https://github.com/birdhackor/learn_to_yolo-reviews/blob/main/colab-self-contained-2026-10-21/repairs/manifest.json)。精確初稿與最終來源見同輪的 manifest.json、final-manifest.json；各頁現行正文、圖與程式另由本 repo 的 coverage.json 綁定。

三輪修後核回均完成，沒有尚待處理的必要問題。

## 驗證紀錄

最終檢查及實頁核對的命令、結果與實際範圍，保存於[本輪原始紀錄](https://github.com/birdhackor/learn_to_yolo-reviews/tree/main/colab-self-contained-2026-10-21)。新增程式沿用 data-excerpt 來源標記，由既有驗證器檢查摘錄與原程式的一致性。

以下檢查全部通過：

- `validate_preparation.py`、`validate_lessons.py`、`validate_curriculum_evidence.py`：52 節配對、摘錄及 notebook，以及 52 份課節／12 份補充紀錄均與現行程式一致。
- `review_coverage.py`：69 個網站頁面符合各自審核指紋；本輪更新其中 13 頁。
- Zensical strict build、`validate_site.py`：全站連結、章節錨點、52 節 Colab 配對、搜尋與 Markdown 呈現通過。
- `test_review_coverage.py`、`test_notebook_bootstrap.py`：7 tests passed；`git diff --check` 通過。

桌面／手機擷取與 DOM 檢查、主 Agent 抽樣實看，以及第三輪實際視判各自列出範圍；不把擷取圖片或建置成功當成所有畫面都已人工檢視。

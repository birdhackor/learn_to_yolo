# 新讀者審查 B：05 assignment 至 07 held-out

審查日期：2026-10-02。讀者假設：只懂基本 Python、PyTorch、NN、CNN；大學數學不熟，也不知道本專案背景。依指定順序全文閱讀，沒有用既有 reviews 作為結論。此報告的行號對應本輪審查時的工作目錄版本。

只新增本報告。沒有修改教材、程式、notebook、圖或既有檢查產物；沒有使用 GPU、發送外部訊息或發布 Colab tag。SVG 以 CPU Chromium 在 `/tmp/curriculum-reader-b` 渲染並目視檢查。9 個案例在既有 PyTorch 2.9.1+cpu 環境重跑，全部 exit 0；stdout 與本輪 JSON 完全相同，case SHA256、notebook cell 3 原始碼與保存輸出也全部相同。Colab 環境格只有逐行閱讀，尚未用全新 Colab runtime 執行；`lessons-v0.2.0` 尚待本輪完成後發布是已知流程，不列為教材錯誤。

優先處理：05 的變體練習會撞到基線斷言；06 與 07 對「整份資料無 GT」的 mAP 規則不同；07 notebook 仍欠三步／160 步實驗的操作銜接，且 07-inference 的 .72 題目不是現有 fixture 的分數。人工結果和真正學得結果的主要標示已清楚，沒有發現把人工 AP 當成模型效果的數值造假或敘述。

## 05-assignment

**讀過檔案：** `docs/lessons/05-assignment.md` 全文、`docs/assets/diagrams/05-assignment.svg` 全文及渲染圖、`lesson_cases/05-assignment.py` 全文、`notebooks/05-assignment.ipynb` 全部 4 個 cells 與保存輸出、`artifacts/checks/curriculum/05-assignment.json` 全文。

**核對數字：** 圖 64×64、4×4 格、格寬 16；紅／藍中心 (12,12)/(44,44)，row/col 為 (0,0)/(2,2)，target 均 [.75,.75,.25,.25]。SVG 框與白點的縮放位置吻合。單图正 2／負 14，含空圖 batch 正 2／負 30，ignore 0；shape (2,4,4,4)/(2,4,4)。重跑分項 loss .0871→.0861、.7031→.6839、.5235→.4744 均吻合。梯度斷言正確檢查負格框／類別為零、objectness 非零。文中有說明這是人工 feature map 上 head 的兩步更新。

**問題與建議：**

- **P1，練習可操作性：** 頁面第 65 行叫讀者把藍框改成 `[20,36,36,52]`，但案例第 34 行仍斷言正格 `[2,2]`。我在記憶體中只替換該框，實際得到 AssertionError；正確的新位置是 `[2,1]`。同段 S=8 也不能只改 `build` 的 grid，因 head 的人工 features、prediction shape 與基線斷言仍固定 4×4。建議明確保留主實驗，在其後新增獨立 `build([moved])`／`build([two], grid=8)` 呼叫及各自斷言，而不是讓初學者猜哪些基線 assertion 要改。獨立核對 S=8 的兩個正格為 (1,1)/(5,5)，target [.5,.5,.25,.25]、負格 62；若含原來的空圖則整批負格是 126。
- **P2，首遇術語：** 頁面第 5、11–15 行首次大量使用 slot、annotation、objectness、logit、neck、proposal、single/two-stage。框容量的解釋很好，但「objectness 是本模型這格是否分配物件的分數」到第 32 行才從 target 反推；logit 的白話定義則在下一節。建議在七通道清單旁加兩句「slot＝一個可輸出框的位置」「logit＝尚未 sigmoid/softmax 的原始實數」，並先給 objectness 本課含義。proposal 等歷史對比可留作附註，減少在主責任流程前的術語負擔。

## 06-decode-nms

**讀過檔案：** `docs/lessons/06-decode-nms.md` 全文、`docs/assets/diagrams/06-decode-nms.svg` 全文及渲染圖、`lesson_cases/06-decode-nms.py` 全文、`notebooks/06-decode-nms.ipynb` 全部 cells 與保存輸出、`artifacts/checks/curriculum/06-decode-nms.json` 全文。

**核對數字：** row1/col2 得中心 (36,28)、寬高 (16,8)、xyxy [28,24,44,32]；另一框 [23.2,24,39.2,32] 的交集 89.6、聯集 166.4，IoU=7/13=.5384615。三候選原順序 score [.64,.72,.855]，NMS .5 保留索引 [2,1]；score .75 只保留背景框。NMS .6 的獨立呼叫得到 [2,1,0]，score .70 重新 decode 得 [.72,.855]。不同類相同框與空輸入檢查通過。SVG 同比例畫框，人工真值、預測、去重前後及「高分錯框仍在」都清楚。

**問題與建議：** 沒有發現阻斷理解或數字不一致。第 58–68 行以獨立變數及呼叫做練習，操作方式比 05 清楚，可當其他節的範本。小幅改善：第 54 行 letterbox、metadata、padding、scale 是順序讀到此處時的新詞，建議直接連到 04-coordinates 的對應例子，附「補邊後輸入畫布」一句即可，無需在本節再展开座標轉換。

## 06-evaluation

**讀過檔案：** `docs/lessons/06-evaluation.md` 全文、`docs/assets/diagrams/06-evaluation.svg` 全文及渲染圖、`lesson_cases/06-evaluation.py` 全文、`notebooks/06-evaluation.ipynb` 全部 cells 與保存輸出、`artifacts/checks/curriculum/06-evaluation.json` 全文。跨節契約另外閱讀 `miniyolo/metrics.py`。

**核對數字：** 按分數排序 FP/TP/FP/TP；累積 P/R 為 (0,0)、(.5,1/3)、(1/3,1/3)、(.5,2/3)，最終 TP2/FP2/FN1。IoU 9/11，重複框失配原因是 GT 已使用。AP50=1/3；score≥.85 後 AP=1/6、R=1/3；移除已知最高分 FP 後 AP=5/9。SVG 的原始點、右側最大值包絡、矩形面積均吻合。無預測有 GT 的 AP=0、無 GT 類別 AP=None，人工背景圖片 FP 被計入。

**問題與建議：**

- **P1，跨節評估契約：** 頁面第 57 行、案例第 56 行說「整份資料無 GT 時 mAP=None」。07 的實際 `miniyolo.metrics.evaluate_ap` 第 78 行却回 `map=0.0`。已實際呼叫全空 GT／空預測得到 `{ap_per_class: {0: None, 1: None}, map: 0.0, precision: 0.0, recall: 0.0}`。這不是正常主例的小數誤差，而是讀者從手算評估器過渡到共用評估器時的定義變動。建議共用評估器也返回 None，或明確說明 07 的返回慣例；為了不把「無可評估真值」混同「有真值但全漏掉」，統一成 None 更易理解。修改若影響呼叫端應一併確認。
- **P2，數學可讀性：** 第 49–51 行的 `sum((recall_next - recall_previous) * precision_envelope_next)` 是示意而非可直接執行的 Python，對基本 Python 讀者可補註「以下為公式的偽碼」，或用本例 `[1/3,1/3]` 寬度與 `[.5,.5]` 高度的 `sum(w*h for ...)`。圖上第三個點沒有 1/3 的 y 刻度，但正文已有完整數字，屬可選改善，非錯圖。

## 07-data

**讀過檔案：** `docs/lessons/07-data.md` 全文、`docs/assets/diagrams/07-data.svg` 全文及渲染圖、`lesson_cases/07-data.py` 全文、`notebooks/07-data.ipynb` 全部 cells 與保存輸出、`artifacts/checks/curriculum/07-data.json` 全文；另讀 `miniyolo/data.py` 全文。

**核對數字：** 固定紅 xyxy [8,12,24,28]、藍 [40,36,56,52]，半開切片各 16×16；紅／藍 channel sum 各 256，綠為零。批次 (2,3,64,64)、物件數 [2,0]，由框／類別重建畫素的 assertion 是逐值相等。SVG 以 4 倍座標作示意，框位置／尺寸吻合。生成 8 張資料的 range、dtype、shape 與正面積檢查通過。練習將紅框右界改 28，對應切片 12:28,8:28、面積／通道總和 320 正確。

**問題與建議：** 沒有發現主例或自主練習的錯誤。**P2，固定圖與生成圖的銜接：** 第 36、40 行从人工純色零背景切到 `ShapeDataset`，但生成器實際使用微量隨機背景 `[0,.04)`、紅 [.95,.10,.10]／藍 [.10,.10,.95]，框尺寸和位置也會變。建議加一句區分「固定 scene 用 0/1 精確核對；生成資料有輕微背景雜訊與不同尺寸，之後才用來訓練」，避免讀者以為後續仍是同一張 256 畫素的固定圖。這不影響本節既有數字。

## 07-targets

**讀過檔案：** `docs/lessons/07-targets.md` 全文、`lesson_cases/07-targets.py` 全文、`notebooks/07-targets.ipynb` 全部 cells 與保存輸出、`artifacts/checks/curriculum/07-targets.json` 全文；另讀 `miniyolo/targets.py` 全文及它使用的座標契約。此頁未引用 SVG，也沒有獨立 `07-targets.svg`；已核對沿用的 `07-data.svg` 固定框。

**核對數字：** 紅中心 (16,20)→(gx1,gy1)→target [0,.25,.25,.25]；藍中心 (48,44)→(gx3,gy2)→[0,.75,.25,.25]。正格清單 [0,1,1]/[0,2,3]，batch 正 2／負 30；碰撞框 [10,14,26,30] 中心 (18,22)，也由 (1,1) 負責，確實拋 ValueError。練習 [4,4,12,12] target [.5,.5,.125,.125] 正確。

**問題與建議：** 沒有數值錯誤，邊界 x=16 歸第二欄的解釋尤其有用。**P2，小型手算練習到實作：** 第 45 行的單物件練習若直接替換 scene，原案例兩物件數量／位置斷言會失敗。手算題本身正確；若希望配合 notebook「試做自主練習」實際執行，建議給獨立 `small_scene`、`small_target=build_targets([small_scene],...)`，不要要求重寫已驗證主例。可加一張含 row/col 與中心邊界的格圖，但現有數字足以理解，不是必須新增 SVG。

## 07-loss

**讀過檔案：** `docs/lessons/07-loss.md` 全文、`lesson_cases/07-loss.py` 全文、`notebooks/07-loss.ipynb` 全部 cells 與保存輸出、`artifacts/checks/curriculum/07-loss.json` 全文；另讀 `miniyolo/losses.py` 全文。此頁没有引用 SVG，也沒有獨立 `07-loss.svg`。

**核對數字：** 零 logits 得 sigmoid=.5、兩類各 .5；box=(.5²+3×.25²)/4=.109375，obj/class=ln2=.693147，total=5×.109375+2ln2=1.933169。正／負 objectness logit 梯度 −1/32/+1/32=−.03125/+.03125，負格 box/class 梯度零。全空圖 box/class=0、obj=total=ln2，有限 backward 正確。獨立重做「複製兩張」與「兩類 logits 同加10」，三項 mean loss／CE 均保持不變。

**問題與建議：** 沒有發現數學或 loss 實作錯誤。**P2，練習與梯度尺度：** 第 62 行只問 loss 不變，容易讓新讀者以為複製 batch 後每個輸出位置的 `.grad` 也不變；其實分母由16成32，單位置 obj 梯度變 ±.015625，共享參數累積後才保持相同。建議用一句補上這個差別，並提供独立兩張 prediction／target，而不是改原案例後仍使用固定 `(1,)`／`(15,)` 梯度斷言。對不熟 ln 的讀者，也可補 `math.log(2)` 是這裡 .693147 的取得方式。

## 07-training

**讀過檔案：** `docs/lessons/07-training.md` 全文、`lesson_cases/07-training.py` 全文、`notebooks/07-training.ipynb` 全部 cells 與保存輸出、`artifacts/checks/curriculum/07-training.json` 全文。160 步補充另讀 `artifacts/checks/grid-learning.json`、`docs/assets/diagrams/grid-learning-curve.svg`、`docs/assets/diagrams/grid-learning-predictions.svg`，均全文或逐元素並目視渲染；也閱讀 `miniyolo/train.py`、`miniyolo/models.py`，核對 JSON 指向的 history、checkpoint、4 張原始 PNG。

**核對數字：** 快速案例 4 圖、每圖1物件，正4/負60；三步 total .9817/.9248/.8323，正 objectness .1202/.119/.120，負 .1202/.1181/.1169。各自與 case、notebook、JSON 一致，均值是 step 更新前的 prediction；沒有把三步說成泛化成功。

160 步設定 train32/val16/test16、seed7/700/7000、batch8、width8、lr.01、4×4 grid、CPU2 threads；JSON 訓練 loop 時間 .736058 秒。用保存的 160 步 checkpoint 在 CPU 重評所有16張：initial validation mAP=.0017507003，final validation=.8035714286（每類 .8571428571/.75，P=.9375、R=.8333333333），test=.7748917749（每類 .8571428571/.6926406926，P=.8823529412、R=.7894736842），完全吻合 JSON 和頁面四捨五入。

loss history 共160點，首 total=.9946234822、尾=.0038105887。SVG 四條曲線各160點，與 history 換算座標的誤差均 ≤.05 SVG pixel，符合一位小數的座標四捨五入；尾 box=.00072015、obj=.00017974、cls=.00003010。圖板4個內嵌 PNG 與紀錄指向的原始 PNG bytes 完全相同，四圖的實際預測 boxes/scores 與 checkpoint 重新推論最大差皆0；圖板框座標和 labels 相符。第一圖左上藍框確實偏上。這些是學得的結果，不是 07-inference fixture。

**問題與建議：**

- **P2，CNN 教學缺口：** 第 14 行直接呼叫 `GridDetector(width=8)`，notebook 也只 import，不顯示模型或 source link。新讀者知道 CNN 卻無從看見 64×64 如何變成 4×4×7。建議補 shape trace：`[B,3,64,64]→[B,8,32,32]→[B,16,16,16]→[B,32,8,8]→adaptive pool [B,32,4,4]→3×3 conv→1×1 head [B,7,4,4]→permute [B,4,4,7]`，並解釋 width 是第一層通道數。連到 `miniyolo/models.py` 對應 class 即可，不必複製整份模組。
- **P2，初始數值銜接：** 上節零 logits 的 obj=.5，而這節開始約 .12；`models.py` 第 74–75 行實際初始化 wh bias −1.8、obj bias −2，sigmoid(−2)≈.1192。建議說明這是人工選的稀疏物件初始化偏置，與預訓練／已學得無關，否則 .12 看起來像未交代的能力。
- **P2，notebook 標題與後續操作：** notebook cell0 仍叫「少量 overfit 與偵測診斷」，但它只跑3步，沒有做少量 overfit 或160步。頁面第 48–52 行 CLI 可執行，CLI 保存 PNG／history／checkpoint／report，却沒有說輸出具體路徑，也沒有在 notebook 提供160步的獨立可選 cell。建議改 notebook 標題對齊「三步訓練與診斷」，補一個可選 CPU subprocess cell，清楚列出 `artifacts/runs/grid-learning/{checkpoint.pt,history.json,loss.png}` 和 `artifacts/checks/grid-learning.json`。這樣 Colab 讀者才能照做已實測學習階段，並知道網站 SVG 是另行產生的圖板。

## 07-inference

**讀過檔案：** `docs/lessons/07-inference.md` 全文、引用的 `docs/assets/diagrams/07-data.svg`、`lesson_cases/07-inference.py` 全文、`notebooks/07-inference.ipynb` 全部 cells 與保存輸出、`artifacts/checks/curriculum/07-inference.json` 全文；另讀 `miniyolo/inference.py` 全文。沒有獨立 `07-inference.svg`。

**核對數字：** 未訓練模型候選數 [0,0,0]；人工 fixture 才有 [2,1,0]。第二張確實清除藍畫素，紅通道256、藍0。target x offset0 被 clamp成.0001，所以紅 x1/x2=8.0016003/24.0016003，誤差 .0016 pixel 小於 .02；單張 batch 仍回長度1 list。鄰格重複紅框 NMS IoU1→.5 的候選數2→1。每一處「人工」標示和「未訓練 no quality claim」都清楚，不容易誤當前節保存的模型。

**問題與建議：**

- **P2，練習與 fixture 不一致：** 第 21、47 行以假設 obj=.8、class=.9 給 score=.72，再問 threshold改.75會刪框。手算正確，但實際 fixture 第 34–36 行 obj logit=10、class logits=10/−10，score≈.9999546。已獨立實測同份 fixture threshold=.25／.75 都保留紅框。建議明示「這是另一个手算候選」，或在可修改實驗新增單格 obj=`torch.logit(.8)`、class logits=`log([.9,.1])` 的獨立 fixture，分别 decode .25/.75、得到1/0。不要讓讀者直接改主fixture後看見與答案相反的結果。
- **P2，decoder 合約補充：** `decode_grid` 第 26–27 行會把框邊界 clip 到输入范围、並第32行排除非正面積框；06 的 standalone `decode` 沒有此步。07 的頁面只講反算／filter／NMS，建議補一行說「本節共用 decoder 也裁切超出輸入邊界的框」，並說明這與 target 不能以 clamp 掩飾非法標註是兩種情況。主 fixture 都在範圍內，所以目前數字不受影響。

## 07-heldout

**讀過檔案：** `docs/lessons/07-heldout.md` 全文、`lesson_cases/07-heldout.py` 全文、`notebooks/07-heldout.ipynb` 全部 cells 與保存輸出、`artifacts/checks/curriculum/07-heldout.json` 全文；引用的兩張 `grid-learning-*.svg`、`artifacts/checks/grid-learning.json` 及其實際圖和 checkpoint 已如上一節逐項核對。另讀 `miniyolo/metrics.py` 全文。

**核對數字：** 人工兩GT、三預測，排序TP/FP/FP，最終 P=1/3、R=.5，包絡高度1×寬.5，AP=.5；第二類None，所以mAP=.5。三步真正小模型用 train seed7／held-out901 各4張、候選score .01、NMS .5、matching .5，實測每類AP0、mAP/P/R0，与保存輸出完全一致。文字明确稱 pipeline smoke，沒有把人工.5或三步0當成功效果。另一个160步實驗候選門檻.05、分割7/700/7000，頁面已清楚逐项写出，與該節3步的.01/901無混用。

**問題與建議：**

- **P1，沿用評估器的空 GT 規則：** 與 06-evaluation 所列同一問題；本頁第 11、21 行很好地解釋無GT類別為None，但沒有交代整批都無GT時共用器会回mAP0。建議隨共用評估器與06一起統一，避免以0誤判成全漏檢。
- **P2，章節完成標準與已實測補充：** 第 36–38 行「完整里程碑仍需要足夠訓練」與後面160步成功例可同時成立，但讀者不清楚自己執行 notebook 的3步後還欠什麼、頁面160步是否已滿足本章哪項目標。建議接一句「本節 notebook 只完成評估器／三步管線檢查；下方160步提供同分佈受控任務的學習證據，想重現請執行該獨立命令」，并链接07-training的可選160步操作。这样无需把主fixture换成预填成功结果。
- **P2，練習可操作性：** 第 42 行刪掉兩FP，AP仍.5、P变1/R仍.5的手算正確；若讀者直接刪原 `predictions` 兩框，案例第22行原P=1/3斷言會失败。建议像06-evaluation已有的 `cleaned` 独立变量，再调用一次 `evaluate_ap`；保留主fixture作为回归核对。

## 發布前建議驗收

1. 05 的藍框移動／S=8 以獨立呼叫實際通過，不需要拆改固定基線斷言。
2. 06／07 的全無GT契約一致，並在教材說明「None」與「有GT且AP=0」的差異。
3. 07-inference 的 .72/.75 練習有對應獨立輸入，或清楚標為手算假設。
4. 07-training notebook 标题、CNN shape trace、初始obj偏置與160步可選操作補齐；若新增執行格，保存對應輸出，但不把160步學得結果塞進人工fixture。
5. 最後重新生成 notebook，再於發布 `lessons-v0.2.0` 後驗證新 Colab runtime bootstrap。本審查的 case通過與保存輸出一致，不等同新tag clone／安装格已驗證。

## 獨立複查：2026-10-02 修訂後

保留上面初讀，不以修訂說明預設通過。重新閱讀 9 節網頁，核對現行 notebook 的說明 cells／實驗碼／保存輸出、case SHA256 與 curriculum JSON，另讀修訂的 `miniyolo/metrics.py` 和 notebook 產生器的 07-training 補充。檢查與試跑均使用 `/tmp/curriculum-reader-b` 工作目錄、既有 `.venv-model/bin/python`、`PYTHONPATH=/workspace/learn_to_yolo`；160 步的新增圖、checkpoint 和 JSON 都留在 `/tmp`，沒有覆寫正式圖或 report。

**實跑證據：** 9 個原案例再跑全部 exit 0，stdout 與正式 JSON 逐字相同；9 個 case SHA、notebook 主實驗碼與保存輸出仍全部一致。沒有新增數值錯誤，也沒有把人工 fixture 換成學得結果。

| 初讀項目 | 複查結論與證據 |
| --- | --- |
| 05 藍框移動／S=8 練習撞基線 assertion（P1） | **已解決。** 現頁 69–81 行清楚保留 main、另開 cell。把新增 code block 原樣執行，moved 正格為 `[[0,0],[2,1]]`；同一 moved 輸入 S=8 為 `[[1,1],[5,3]]`，負格62，target均 `[.5,.5,.25,.25]`。原兩框 S=8 的不同結果及含空圖負格126也有另行說明。 |
| 05 slot／annotation／objectness／logit 首遇解釋（P2） | **已解決主要障礙。** 第9行在輸出規格前補了白話定義；讀者可以先理解本模型監督含義，再看責任流程。proposal 等歴史詞仍較簡略，但不阻斷主實驗。 |
| 06-decode-nms letterbox／metadata 首遇缺少連結（P2） | **已解決。** 第7行有「等比例縮放再補邊」「還原所需紀錄」及04座標頁連結；主例與兩個獨立門檻練習仍通過。 |
| 06／07 整批無 GT 的 mAP=None／0.0 差異（P1） | **已解決。** `metrics.py` 第42、78行與06一致。獨立測試06和07兩個評估器：無GT＋無預測皆None；無GT＋背景誤報皆None；有GT＋無預測皆0.0。這保留「無可評估真值」與「全漏檢」差異。 |
| 06 AP 示意式看似可直接跑 Python（P2） | **已解決。** 第50行已標公式偽碼。原有AP=1/3、1/6、5/9輸出未變。原圖少一個1/3 y刻度的可選建議未改，仍非錯圖或必修項目。 |
| 07-data 人工scene和ShapeDataset差異（P2） | **已解決。** 第46行準確列出尺寸／位置變化、微量背景雜訊和兩色值。另做紅框擴寬的獨立練習，畫素／標註相等、紅通道總和320，吻合答案。 |
| 07-targets 單物件練習的執行指引（P2） | **仍待改進，不影響主例。** 第45行仍是手算答案，未說保留main、另建small_scene；若把原scene改成單物件，固定2正格斷言仍會失敗。自行建立獨立small_scene後，正格`[[0,0,0]]`、target`[.5,.5,.125,.125]`實跑通過。建議加一句「另開cell用單物件輸入呼叫build_targets，不改主例」即可。 |
| 07-loss 複製batch的loss／單位置梯度差異與ln（P2） | **部分解決。** 第64行已說BCE分母、提醒不要沿用固定梯度shape，並給`math.log(2)`。独立重做兩張相同圖，三項mean loss不變，但正格單位置obj梯度確為−.03125→−.015625；共同加10後兩類logits確為[10,10]且CE不變。新增文字用的是「另加空圖」；仍建議直接補「複製相同圖時每個輸出位置梯度也減半，共享參數累加後保持同一尺度」，讓它直接回答前一段的練習。 |
| 07-training CNN shape trace／width（P2） | **已解決。** 第9–15行補完整trace及model source。逐層真實forward核對得到 `[2,8,32,32]→[2,16,16,16]→[2,32,8,8]→[2,32,4,4]→[2,32,4,4]→[2,7,4,4]→[2,4,4,7]`，與文中吻合。 |
| 07-training 初始objectness≈.12來源（P2） | **已解決。** 第15行交代人工wh/obj bias、非預訓練。實際obj bias=−2，sigmoid=.1192029193；不是前節零logits的.5。 |
| 07-training notebook標題／160步操作與保存路徑（P2） | **已解決。** notebook cell0改成「三步訓練與診斷」；既有cell2 Markdown明列「主例只跑三步」及另開code cell的160步CPU命令、顯示loss PNG與產物位置，網頁79–91行亦有可複製的subprocess版本。從/tmp執行相同160步命令成功，初始validation、最終validation、test三份指標字典與正式report完全相同；產生checkpoint、160點history、980×560 loss PNG和4張validation PNG。訓練loop本次約.610秒，只是此次機器／負載的觀察，不改正式.736秒紀錄。另把三步例的step移除，forward/loss/gradient仍成功、最終參數變更assertion如預期失敗，答案正確。 |
| 07-inference .72/.75題目與主fixture分數不符（P2） | **已解決。** 第49–63行明說主fixture接近1，另給完整獨立輸入。新增code block原樣執行得score=.7199999690，門檻.70/.75候選數1/0，NMS仍固定.5，全部斷言通過。新snippet選格內x=.5，只測分數／過濾，未再宣稱它是原來x offset0的固定紅框。 |
| 07-inference decoder裁切邊界與06不同（P2） | **仍待補充，無主例数值影響。** 網頁仍未說共用decoder會clip輸入外的預測框並排除非正面積框。建議在解碼／顯示段補一行，區分「裁切預測」與「不要clamp掩飾非法GT」。 |
| 07-heldout 空GT規則（P1） | **已解決。** 共用評估器與06已統一None，獨立邊界案例通過；3步和160步有GT資料的結果不變。 |
| 07-heldout 三步notebook與160步補充的完成標準（P2） | **仍可改進。** 第38行仍寫完整里程碑需要足夠訓練，後面的160步另有學得結果；人工／三步／160步都已明確分開，不會混成同一模型。可再連到07-training的可選命令，說明3步只完成評估管線、160步提供同分佈受控任務的學習證據。 |
| 07-heldout 刪兩FP練習保留主例斷言（P2） | **仍待執行指引。** 手算答案正確，原文未補獨立cleaned變數；直接改主predictions仍碰到P=1/3固定斷言。独立cleaned輸入實跑AP=.5、P=1、R=.5，吻合答案；建議補「保留main、另建cleaned呼叫evaluate_ap」，無需改主fixture。 |

**本輪結論：** 初讀 P1 都已解決，新補的獨立練習和160步命令可操作，未發現新的阻斷問題。剩餘P2是上述幾項簡短教學說明／手算題執行指引，可用一兩句完成，不需要重新訓練或改人工答案。160步真實學得結果、人工fixture和三步管線檢查仍清楚區分。新Colab tag clone／安裝格的全新runtime驗證仍留待發布階段，本複查沒有把本地CPU成功等同該項已完成。

## 最終定點複查：剩餘 P2 修訂後

再次讀取最新檔案，沿用 `/tmp/curriculum-reader-b` 與 `.venv-model` CPU 環境試做，沒有改正式 case、圖或 report。

- **07-targets 已解決：** 第47–54行明說保留main、在新cell建立new_box。新增snippet原樣執行通過，唯一正格`[[0,0,0]]`、target`[.5,.5,.125,.125]`。
- **07-heldout 兩項已解決：** 第71行把三步評估流程與160步合成學習結果分開，連到training可選Colab操作；第73–82行指定在main原fixture斷言後、ShapeDataset之前插入cleaned實驗。我在記憶體中將新增snippet原樣插入指定位置並執行完整main：原fixture的AP=.5/P=1/3/R=.5斷言保留且通過，cleaned得到AP=.5/P=1/R=.5，後續三步held-out結果仍為0。沒有局部變數作用域或基線斷言衝突。
- **07-inference 已解決：** 第9行準確說明共用decoder的預測框裁切及非正面積排除，也區分非法GT不能以clamp掩飾。另用兩個人工候選定點驗證：越界框裁切成`[0,0,18,18]`，另一個有限極端logit產生零寬框而被排除；與`inference.py`第26–32行一致。
- **07-loss 已解決：** 最新第64行直接區分「兩張相同圖各自prediction槽的梯度減半」與「同一CNN參數收到兩圖梯度累加」。除上轮人工零logits的−.03125→−.015625檢查外，本次以同一GridDetector對單圖／複製兩圖分別backward：每個輸出位置梯度減半，共享參數梯度相同，最大絕對差`3.35e-8`，符合浮點誤差。

**最終狀態：** 本範圍初讀與前次複查提出、需要修正的P1/P2問題均已解決，未發現新的理解或操作障礙。原先可選的圖刻度美化不構成未完成項目。人工fixture、三步流程檢查和160步實際學習證據仍分明；Colab新tag／全新runtime驗證屬後續發布階段，未在本定點複查宣稱完成。

# 第 2 輪非作者技術審查：07／08 八頁

審查者：`clear_first_foundations`。基準 commit：`16f69103d1f216d1024a40610623083557324701`；實際審查的是本次 working tree 的終稿，指紋列於末尾。依 `.agents/skills/clear-tutorial/SKILL.md` 與 `references/review-protocol.md` 執行。我沒有撰寫以下八頁；先前作者工作只涉及第 11 章四頁。

結論：八頁未發現未解的 blocking 技術問題。07-training 的 PNG 檔案內容說明有一項 burden，作者已修正，實際核回後關閉。這是程式、資料、數字與圖面座標審查，沒有進行本組 desktop／mobile browser 驗收，也沒有執行真人 Colab、相機或照片泛化測試。

## 實際方法與範圍

- 完整閱讀八頁現文及八個 lesson case；核對 `data`、`custom_data`、`targets`、`models`、`losses`、`geometry`、`inference`、`metrics`、`train`、`checkpoint`、`provenance` 實作，另讀 `detect_image.py`、`run_custom_data_learning.py`、`render_learning_evidence.py`。
- 八個 lesson case 在 `/tmp/clear-grid-tech` 隔離副本、既有 `.venv-model/bin/python`、PyTorch `2.9.1+cpu`、CPU 2 threads 執行。每個 case 正常結束，stdout 與對應原始 JSON 逐字相同。訓練 case 各最多 3 步；沒有修改既有紀錄、教材、notebook 或 source。
- 額外 CPU fixture 核對 mean 分母、重複 batch 與共享參數梯度、混合空圖、二次 sigmoid／softmax、分類平移、score 門檻、微小框、AP 排序與重複框、兩種原圖方向的 letterbox、只在 padding 的框。
- `run_custom_data_learning.py --fixture --steps 3` 只在 `/tmp` 產生新的短紀錄，核对 format v2 的模型、optimizer、RNG 載入一致性，以及獨立 `detect_image.py` subprocess 的原圖尺寸／座標。另做 1 步 grid CLI，確認 validation PNG 的實際內容是原始 RGB 輸入。
- 重新生成 seed 7000／7001 的合成 PNG 與 manifest，不訓練模型；兩份 manifest、PNG inventory、decoded RGB inventory 均重現既有紀錄的 hash，train／validation 不變，兩組 test 的 RGB hash 集合無交集。
- 載入既有 checkpoint 做純 CPU 推論。custom 160／1600 checkpoint 的 SHA-256 與各自紀錄一致；三個 split 的 AP/P/R/TP/FP/FN 全部逐值重現。grid-learning 紀錄未刊 checkpoint hash，本次另列實際檔案 hash，該檔的 validation／test 評估也逐值重現紀錄。這些是複算既有固定模型，不是新訓練，也不是新增用 test 調參的實驗。
- 用重新生成的輸入圖與原始 JSON 重畫三張實測 SVG，與現行 `grid-learning-curve.svg`、`grid-learning-predictions.svg`、`08-custom-learning.svg` 逐位元組一致。三張其他示意 SVG 讀取 XML，按標明比例手算框、中心、負責格、padding；此方法不等於 browser 的字形／遮擋／手機可讀性驗收。

可重現的本輪腳本及結果：`/tmp/clear-grid-tech/audit_grid.py`、`audit-results.json`、`reevaluate_checkpoints.py`、`checkpoint-reevaluation-results.json`、`checkpoint-loss-results.json`、`final-page-inventory.json`。這些是本轮輔助紀錄，沒有取代 repo 既有 JSON。沒有 GPU 或 160／1600 步重新訓練。

## 逐頁結果

### 07-data

實際閱讀 `lesson_cases/07-data.py`、`ShapeDataset`、`collate`，執行完整 case，並核對 `07-data.svg` 座標與正文。圖片 `[C,H,W]`、batch `[B,C,H,W]`、RGB float32 0–1，boxes float `[N,4]` xyxy pixel、labels long `[N]`，空圖保留 `[0,4]`／`[0]`，均與實作一致。切片採 `[channel,y,x]`，半開框沒有額外 `+1`。

固定紅框 `[8,12,24,28]`、藍框 `[40,36,56,52]` 都是 16×16、面積 256；CPU stdout 為 `(2,3,64,64)`、counts `[2,0]`、紅藍 channel sum `256.0`，8 張 generated Dataset 的契約斷言通過。原始數字來源 `artifacts/checks/curriculum/07-data.json`。

SVG 輸入原點 `(64,60)`、4 倍放大，紅色 rect `(96,108,64,64)`、藍色 `(224,204,64,64)` 正確。圖上顯示色與實際 `(1,0,0)`／`(0,0,1)` 有區分。自主練習紅框改寬到 20，面積 320；舊 `8:24` 切片仍只測 256，答案指出這個限制。x/y 誤交換雖通過範圍檢查，仍會被畫素逐值比對抓到，與 case 的檢查位置相符。

疑點：無未解技術 issue。generated 圖的背景噪聲與固定純色圖不同，正文已區分；pixel 對標註的逐值檢查只针对固定例子，沒有宣稱通用標註品質驗證。

### 07-targets

閱讀 `build_targets` 的中心正規化、floor、格內 offset、碰撞拒絕及輸入檢查；CPU 執行兩張圖 fixture，核對正文、練習及 `object-journey.svg`。頭部 `[B,4,4,5+C]` 的 xy 是格內比例，wh 是整張圖比例；`[b,gy,gx]` 與 `(x,y)` 軸序有明確分開。

紅框中心 `(16,20)` → `(gx,gy)=(1,1)` → `[0,.25,.25,.25]`；藍框中心 `(48,44)` → `(3,2)` → `[0,.75,.25,.25]`。正格 2、負格 30，stdout 與 `07-targets.json` 相同。空圖增加 16 個 objectness 背景格，但未替背景建立有意義的 box/class 監督。兩中心同格的 fixture 被拒絕，沒有默默丟 GT。

SVG 原點 `(156,136)`、3 倍，紅框 `(180,172,48,48)`、中心 `(204,196)`、負責格 `(204,184,48,48)` 正確。紅框可跨 4 格，分配只由中心決定；格線上的 x=16 歸右格，与 floor 相符。練習 `[4,4,12,12]` 的格內 .5/.5 與 wh .125/.125，以及 `[40,2,48,10]` 的 .75/.375 可直接手算。正文也指出 sigmoid 不能以有限 logit 精確輸出 0 的限制。

疑點：無未解技術 issue。每格一框與拒絕碰撞是本專案簡化，沒有寫成完整 YOLO 的 assignment 規則。

### 07-loss

完整閱讀 `grid_loss`、case 與現文，CPU 重做零 logit、全空圖及自主練習。三個 mean 的分母正確：box `4×Npos`，objectness `B×S×S`，classification `Npos`；總 loss `5×box+obj+class`。box 經 sigmoid 做 MSE，BCE／CE 直接吃原始 logit。

一張紅框零 logit：box `.109375`、obj/class `ln(2)=.693147`、total `1.933169`，obj 正格梯度 `−1/32`、負格 `+1/32`、分類 `[-.5,.5]`，與 `07-loss.json` 一致。背景 box/class 梯度為 0；全空圖用連接計算圖的 0 避免對空 tensor mean 得 NaN，objectness 仍有限。

獨立補測相同 batch 重複兩次，loss 不變，每個輸出 logit 梯度減半；同一共享參數展開成兩份時，加總回原梯度。`[red,empty]` 時 obj 正格梯度變 `−1/64`、分類不變、空圖 box/class 梯度 0。二次 sigmoid 令 obj loss `.942827...`；二次 softmax 在零 logit 的 CE 值仍相同，梯度卻為 `[-.25,.25]`；class logits 同加 10 的 CE 不變，支持正文區分「loss 看似一樣」與梯度正確。

疑點：無未解技術 issue。本節只 backward，未 optimizer.step；現文沒有把輸出 logit 的梯度冒充參數更新或偵測能力。

### 07-training

閱讀 `GridDetector`、lesson case、`optimizer_step`、`train`，實際執行 3 步 case 與另一次隔離 CLI 1 步。width 8 的 stride conv 空間依序 32²、16²、8²，pool 成 4²、head 7 channels，再 permute `[B,4,4,7]` 正確。wh bias −1.8、obj bias −2 是初始設定，不是訓練權重。

3 步固定 4 張圖：step 0/1/2 total `.9817/.9248/.8323` 與分項、pos/neg objectness 均逐字重現 `07-training.json`。梯度有限且非零、首個參數張量更新前後不等，支持「有更新」；打印值来自该步更新前的 prediction。這個 case 不要求三點單調，也不宣稱泛化。

160 步補充實驗是另一個 32/16/16、seed 7/700/7000、batch 8、width 8、Adam .01、CPU 2 threads 的既有實測。逐條核對 `grid-learning.json` 的 160 個更新前 minibatch loss、total 權重及曲線；第 16 步 class loss `.0068348` 支持「約第 16 步低於 .01」。初始 validation mAP50 `.0017507`、最後 `.8035714`、test `.7748918`，表格近似值正確；candidate .05、NMS .5、matching .5 与實作一致。時間 `1.184570304...` 只指 optimizer loop，未當全程耗時。

載入既有 checkpoint 純推論重算：validation `15/16=.9375` precision、`15/18=.8333333` recall；test `15/17=.8823529`、`15/19=.7894737`。first-4 validation 圖為固定索引 0–3、非效果挑圖；其中紅色 score `.980`、IoU 約 .47 的 FP 與藍色 TP 能由紀錄重畫一致。圖中 box 是未乘 5 的 raw term，說明正確。

**TGRID-01，burden，已關閉。** 初讀輸出檔表說 validation PNG 是「真值與預測」，但 `train.py` 直接存 RGB。作者改為「前 4 張 validation 的 RGB 輸入圖；真值與預測框見 report.json 及本頁圖板」。我讀了實際新文字，並在隔離 CLI 1 步比對 PNG 每個 pixel 與 regenerated validation image 完全相同、JSON 有 target/prediction；修正準確。除此以外未解 issue 為 0。

### 07-inference

完整閱讀 `decode_grid`、NMS、case，執行未訓練及人工 logits fixture，另做 score／微小框練習。xy 解碼 `(grid_xy+sigmoid(offset))/S×(W,H)`，wh `sigmoid×(W,H)`，轉 xyxy 後 clamp；positive-area 過濾与 score≥門檻、同類 NMS、總分排序一致。每格只取最大 class，跨類不互相 NMS；NMS 刪 `IoU>門檻`，matching 的 `IoU≥.5` 是另一件事。

`07-inference.json` 的未訓練 `[0,0,0]`、人工 `[2,1,0]`、紅框 `[8.0016002655,12,24.0016002655,28]`、NMS `2→1` 均重現。人工 logit 的 clamp 令理想 0 offset 移動 .0016 pixel，與 .02 容差相容。batch=1 未 squeeze。鄰格重複框 IoU 約 .998、門檻 1 保留兩框／.5 保留一框，case 的 assert 真正驗證其結果。

補測 obj .8、class .9 得 score 約 .72，cutoff .70 留、.75 刪；x offset .5 的框 `[16,12,32,28]` 正確。tw=−18 的極小 width 在 float32 的角點相減成零，實際被 positive-area 過濾。這是浮點／解碼限制，正文沒有把它寫成標註合法性的限制。

疑點：無未解技術 issue。兩張共用靜態示意是標註，不是假裝人工 fixture 或未訓練模型已偵測成功；非正方形還原導到 08-own-images。

### 07-heldout

完整閱讀 `evaluate_ap`、half-open IoU、fixture 與獨立 held-out 執行。按同類跨圖 score 排序、只在相同圖片找未使用同類 GT、最大 IoU≥.5 配一次；micro P/R 包含全部保留框。無 GT 類別的 AP 是 None，排除 mAP 平均但該類 FP 仍算入 micro precision。AP 採 all-points precision envelope，沒有等同 COCO 多 IoU AP。

人工兩個 GT、預測一 TP／背景 FP／重複 FP：class-0 AP .5、class-1 None、mAP .5、precision 1/3、recall .5，與 `07-heldout.json` 及 CPU case 相同。獨立把背景 FP 分數改 .95，AP 變 .25、P/R 不變，支持排序與面積解釋；fixture 沒經 NMS 是為了測重複 FP 的配對規則。3 步訓練後另一個 seed 901 的 held-out case AP/P/R 皆 0，正文正確當流程證據。

160 步補充和 training 頁共用同一紀錄，沒有把它混成 3 步成果。CPU saved-model 重算 test 的 19 GT／17 預測／15 TP，分類 AP `.8571429/.6926407` 的平均 `.7748918`；validation 18 GT／16 預測／15 TP。訓練圖／驗證圖／test 以獨立 seed 生成、test 不 optimizer.step 的路徑確認。模型選擇／候選截斷／AP 種類／小樣本差異均有本專案限制。

疑點：無未解技術 issue。正文對本書與 VOC/COCO 配對差別已有明示，本輪支持本 repo 實作與 fixture，不宣稱完整官方 evaluator 的重現。

### 08-own-images

完整閱讀 case、`letterbox`、`undo_letterbox`、`load_grid_checkpoint`、`detect_image`；CPU 重做 3 步舊格式 checkpoint、安全載入、合成 PNG 及人工框還原，另手算原圖與輸入圖的座標。`weights_only=True` 表示限制 pickle 類型，不是只保存模型權重；config/class names 檢查及 strict state_dict 的 keys/shapes 均有實際依據。

原圖 W=120/H=80，理想 scale 64/120，實際整數 resize W=64/H=43，因此 `sx=64/120`、`sy=43/80`，上 padding 10／下 11。原框 `[20,10,60,30]` → `[10.6667,15.375,32,26.125]` → 原框，與 `08-own-images.json` 重現一致。metadata 原圖 size 是 `(H,W)`，報告 `original_size` 是命名的 width/height，沒有顛倒。全圖映射 `[0,10,64,53]`；只在上 padding 的框還原 clamp 後高為 0，应刪除。錯用單一理想 scale 的 y 會變 `10.078125/30.234375`，支持正文提醒。

SVG 左原圖 2.5 倍、右輸入 4 倍：原紅框 `(66,81,100,50)`、右紅框 `(410.667,117.5,85.333,43)`，有效圖高 172、上下 padding 40/44，尺寸正确且兩側倍率不同有標明。圖示灰色僅顯示 padding，實際補 0。

3 步模型與 reload 的 logits 逐值相同，預設顯示門檻 .25 下 0 框，原尺寸 PNG／JSON 正常保存；沒有拿訓練 config 的 .05 AP cutoff 當顯示門檻。錯 class count 的 checkpoint 被拒絕。原圖像素／框在前處理及輸出畫圖的路徑確認；RGB、alpha 及 EXIF 的限制現文明示。

疑點：無未解技術 issue。尚未真人 Colab 上傳／相機、EXIF 裝置照片或真实照片品質測試；不能從合成 PNG 的通過推論它們已驗收。metadata fallback 舊 `scale` 是相容路徑，新的 mapping 實際使用 `scale_xy`。

### 08-own-data

完整閱讀終稿、`validate_annotations`／Dataset、一步 case 與完整自訂學習脚本。manifest 原圖 pixel xyxy、實際 PNG 尺寸、source_id/split 契約、class 順序、空圖、box 有限／合法、bool label 拒絕，以及與圖同步 letterbox，均對回實作。Dataset 先驗全部 records/實際圖，再過濾 split，不因讀 train 而漏驗其他 split。

case 的 6 張 PNG、各 split 2 張、3 類 head `(2,4,4,8)`、同步框 `[10.6666669846,15.375,32,26.125]`、一步 loss `1.4407` 逐字重現 `08-own-data.json`。檢查 source leakage、壞 row、bool class、空圖尺寸、NaN、實際尺寸錯誤及 cross-split 完全重複 RGB。source/path 檢查只驗 metadata，原文保留人工來源盤點／近重複限制，沒有把異名複製圖的檢查推成完全獨立資料。

完整實驗兩份原始 JSON 全部對回來源：24/12/12、3/3/3 空圖、21/9/9 GT、每類 train 7／val/test 3、12 個 source、3 類 width8 模型 15,544 參數、Adam .01、batch8、score .1、NMS/matching .5。160 步完整 train loss `1.5501668453→.1697731167`，train mAP `.0068027211`、validation 0；第 158 步尖峰及最低預測高 `.0545401 pixel` 與紀錄一致。

1600 是重置 seed 後新建 model/Adam 的獨立重訓。腳本 prior report 只拿來檢查設定／資料／history，没有載權重或 optimizer。兩份 JSON 前 160 步四個 loss 分項逐值相同；本輪重新生成 manifest／每張 PNG／RGB inventory 的 hash 全部匹配，train/validation 一致、test 7000→7001 為新圖。這支持現文明確「總共 1600 步，非接續 160」；沒有新跑這兩段訓練。

原始 1600 checkpoint hash 匹配後 CPU 複算：完整 train loss `.0001170479736`，train AP/P/R 全 1（21 TP、0 FP/FN）；validation mAP `.2962962963`、P 3/11、R 3/9（3 TP、8 FP、6 FN）；新 test mAP `.7777777778`、P=R=7/9（7 TP、2 FP/FN）。初始化與兩份最後模型的完整 24 張 loss 四分項也逐值重現，沒有拿末尾 minibatch 的 `.0000994738` 代替 full-train loss。

原始梯度 L2 min/max `.0241971575/32.5447463989`，表示每步 backward 得到有限、非零整體梯度；原文現在明說未逐步量 weight difference，不能推出每步每個參數都變。全程初末 weight L2 `17.2168382176` 本輪可從相同 seed 的初始參數與 hash 匹配的 checkpoint 獨立重算。這兩類證據与「每步 optimizer.step 有執行」的程式路徑分開。

format v2 的原始 raw/decoded、optimizer、RNG exact reload flags 與來源檢查一致；本輪另跑隔離 3 步重現这些检查及独立 CLI 原圖輸出，僅證明同 CPU reload。沒有宣稱此處測過跨裝置 bitwise 或 resumed optimization parity。時間 `10.877128771` 是 loop、`14.002426073` 是 run() 直到 report 寫出前，有明確排除啟動/import/安裝/最終 report write。

SVG 從 1600 JSON 加 seed700 的 letterbox 圖重畫逐 byte 一致：紅 TP IoU `.5845539`、額外黃 FP IoU `.2411931`、藍 FP `.4058015`／FN、黃 FP `.4206646`／FN、空圖無預測。標線 160、尖峰 158、起點 minibatch `1.5648255` 與完整 train `1.5501668` 的差異現文正確。圖/評估用 64×64；CLI 使用 original-frame。原始 record 的 validation 原圖 W80/H120，metadata sx=43/80、sy=64/120、left padding10；景橫／景直兩種实际 ratio 都獨立核對，未以名義 scale 冒充實際整數 resize。

疑點：無未解技術 issue。低 validation AP 是誠實限制而非 acceptance failure；真實資料、camera、真人 Colab、未來 user-supplied 1600 指令不在本輪執行範圍。train 拒絕同格碰撞、val/test 保留全部 GT 的限制有明示，未靠丟真值提高 AP。

## 版本與變更核回

第一次版本核回時，八頁 Colab pin 由 `lessons-v0.4.1` 統一為待發布 `lessons-v0.5.0`，07-training 再改 TGRID-01 那一個 PNG 表格列。我將該時版本只反轉這兩種已知變更，八頁的 SHA-256 全部精確回到起讀指紋，確認當時其餘正文未漂移；結果保存在 `final-page-inventory.json`。其後又核回下面三項小幅文字改善，末尾表格採這次最新指紋；页尾 stdout 仍逐字對回 JSON。

### 最後的 readability 技術 delta 核回

- **07-data**：實際讀到新句「每個矩形整個落在某一格、寬高 8～15，因此中心嚴格格內、xy 不到端點 0」，再核 `data.py` 的 randint 上界與 x1/y1 範圍。64/4=16，w/h 為整數 8–15，格內 x1∈[0,16−w]，所以中心相對位置介於 w/2 與 16−w/2，嚴格在 (0,16)；xy 介於 .25 與 .75，敘述正確。限定「預設設定／64×64」沒有推成任意尺寸一律 8–15。
- **07-targets**：實際讀到「原始標註清單 → build_targets → 格子訓練目標 target」的新段落。輸入每圖 boxes/labels 可變長，輸出固定格子四欄，與 `collate`／`build_targets`／lesson case 相符；不再要求靠變數名稱是否多一個 s 來分辨語義。本節無模型、示範 pred 只在註解的限定仍正確。
- **07-loss**：實際讀到新限定「15 個輸出 logits 的梯度量合計，CNN 權重需乘各格對權重變化率，再帶方向相加」。15×.03125=.46875 正確；連鎖律 ∂L/∂θ=Σᵢ(∂L/∂zᵢ)(∂zᵢ/∂θ) 支持新句。不能用 .46875 直接代替任一卷積權重梯度。既有混合／重複 batch CPU fixture 支持分母与共享參數方向相加；此次只改說明，沒有 source／數值漂移。

以上三項都已實際閱讀核回，沒有新增 blocking 或未解 burden。

`lessons-v0.5.0` 尚未發布，本輪沒有將新 tag URL 宣稱已可用，也未將目前 404 解釋為 credentials 缺失。release tag／Colab 真人入口的驗收仍需發布後進行。此發布狀態不是上述本地 CPU 技術結論的替代證據。

## 最終指紋

下表是本輪實際審查及核回的檔案 SHA-256。圖來源／XML／從數據重畫的核對不等於 browser 檢查；頁面若再改需按 delta 重新核回。

### 正文

| 檔案 | SHA-256 |
|---|---|
| `docs/lessons/07-data.md` | `f086a8694e46320912d340add2c1339c2a04fedc061c014092962d839ef97730` |
| `docs/lessons/07-targets.md` | `a3816d6f82e0bb9859557d6771ec2978d66304ee4f2a2f6670d8e4514421c205` |
| `docs/lessons/07-loss.md` | `2b3a4f33c0e12feab2c8f52f97fee44bfdbfcaa8e8329ff44811c4720288ae52` |
| `docs/lessons/07-training.md` | `8fa15d5a6bfd8917f52e947a95218e1d07b8e476b8234b5a294ffbaf694db72d` |
| `docs/lessons/07-inference.md` | `79ae6881d2ff9144a905d7f559679e416c2bdf7226d8e4212a816df23dbd1beb` |
| `docs/lessons/07-heldout.md` | `e107fe133e6d000025fe0dfb22ebb15a3cfcde4da7bf6c6715d912a605a93445` |
| `docs/lessons/08-own-images.md` | `332da2c487b9e3591b61dbb02fc182bbaa1f4afc20461e2c05da332d7f822006` |
| `docs/lessons/08-own-data.md` | `74fb732a9acb9ff83431ce2878ec217299ec851057717a69c638888d90a57dd3` |

### 圖片

| 檔案 | SHA-256 |
|---|---|
| `docs/assets/diagrams/07-data.svg` | `7c45416667c1e130c9ef4433a600435edd4f2f8de0c20869487e3f7e0429c1ab` |
| `docs/assets/diagrams/object-journey.svg` | `a80d4e43b9e2dc76d13f76137b0689d6700a2a2ee54faa502607e3401e629d78` |
| `docs/assets/diagrams/08-own-images.svg` | `237df7f604a77f6818a697ae2cff0ad2b39cf28d6e3722acad6d22dbcf32e1e5` |
| `docs/assets/diagrams/grid-learning-curve.svg` | `818baea73d025d34b16eb67e16947d46b7615f49b6d4618f3c3f0567c120682d` |
| `docs/assets/diagrams/grid-learning-predictions.svg` | `8eb7ba62adfa0df235a3530982ed6c73cdec7ced2ae47fcceabb1eac8b2b0bd4` |
| `docs/assets/diagrams/08-custom-learning.svg` | `83036ad187af44e64bf54200e47b47b414dfab96ef03b4e5db79321d187f09a7` |

### 實作與 lesson case

| 檔案 | SHA-256 |
|---|---|
| `lesson_cases/07-data.py` | `3e82d40e0dafb7c07813e8521472bc6dc220eefae4ebd576353fa3daa3a24215` |
| `lesson_cases/07-heldout.py` | `ada566e4f1c294156116f7771c2bf31dfc7d23770a7e76a56331a595cf888568` |
| `lesson_cases/07-inference.py` | `75011612e6f4c8aee853cdcaec7e7d64d44499e70464cdbb108d98e06c937c10` |
| `lesson_cases/07-loss.py` | `1fa1c9ace4eb1399130d340267059a00d204cf682bb56f33ca2f0a11a3ad84d0` |
| `lesson_cases/07-targets.py` | `95804f80642c0f83830c7f88de9e3a4aa540582e67698af1cc8edab3bf2cd9c0` |
| `lesson_cases/07-training.py` | `fe1037df4ece00bef8810d5d3a31184c1a862b926306688759bcb5cb748afc99` |
| `lesson_cases/08-own-data.py` | `d3be8a09b62a77c542ba48a93e10c36f5a8a2f34e587b2e207c4d7e41c8f300a` |
| `lesson_cases/08-own-images.py` | `d812cdb29d585e4c6ea6be4a3e1e502259be7bd48547cda749378a89a8aff65a` |
| `miniyolo/__init__.py` | `785b058b2b011124243b06ee59df8e59dfa75b77f050b897479e5be636763fc9` |
| `miniyolo/checkpoint.py` | `2144e2afa4d89382a7b3cd253755bb851e35fdc0847ee7179dee02dbbf4ca49b` |
| `miniyolo/custom_data.py` | `7dd9999f03dbc32ce1ec951c58121c8f746d381b0244e4c2a82cca59e7ca7e55` |
| `miniyolo/data.py` | `cccad00e2c4f96eb6567eafc9e12248379c6b715fc1790d75518a253baa6181d` |
| `miniyolo/figures.py` | `155222c5bb0d6942bcc231d5c32c6f1a4db662293c8cf909560f1f9b75a3ea15` |
| `miniyolo/geometry.py` | `6a6b57d3493888e99dae4a012dab78127b8b543a0e01d8107963a3d6e63d8483` |
| `miniyolo/inference.py` | `995ac8f942d0c1e43d94f2efb3b7adcb91a9feb6b9bab923615209f0c190c313` |
| `miniyolo/losses.py` | `81fa9331c9a2aebe2e9c6c453a5313e64566bb20c3fe77ca45ec9df53424d4e4` |
| `miniyolo/metrics.py` | `53ce982e37cbd96c784f75c7d30faf99d52f79ab83ca7b8114eb21b4327330e0` |
| `miniyolo/models.py` | `49d029ea4ba2650ce8933cf97e3d25dc7aff2ca4e972eec19cdda17b0f4900e6` |
| `miniyolo/provenance.py` | `21769813c36b45f513c9f0d6125194f3baa5ae265278ffd44a796d298d124ed3` |
| `miniyolo/targets.py` | `2c8e32f2845b3bf970c77304c5cca0f999083d36a4d5af2f75f14aa89b823f16` |
| `miniyolo/train.py` | `aa5567f11be6ca7aee97bd082b9d5964414b12b73453763c915c7ffb409c4276` |
| `scripts/detect_image.py` | `69738c487cfad24bea2009384712ca0ba958ac1fd394e2226ae33d7de9682c6a` |
| `scripts/render_learning_evidence.py` | `89d764abed5b7e5e6fd998abbe1f324ed1eee195ffc0fe6a7694f1b799cee952` |
| `scripts/run_custom_data_learning.py` | `b8b3b442e4e2349173a2eb8198b33f966d1e75d80c8b1edb07d1af4d3b5e99e5` |

### 原始數據紀錄

| 檔案 | SHA-256 |
|---|---|
| `artifacts/checks/curriculum/07-data.json` | `f9acacbb73cb8f6edafc383c2f5be0ba215a0c56db8f14cc2aca5711d31b946a` |
| `artifacts/checks/curriculum/07-targets.json` | `26cf6375a7747f2190d273e222610b534d1b5f094f52e404112334c02dace320` |
| `artifacts/checks/curriculum/07-loss.json` | `27c8020d6b08fa20d74921d8373d6d5d3d01ef8105d34f891acd2a01f65100ad` |
| `artifacts/checks/curriculum/07-training.json` | `13f6792e07b1d98882c67b8978f85a328bcbfd4ba454cacccf2c0b0cfa1dbd4d` |
| `artifacts/checks/curriculum/07-inference.json` | `d82f3587c30c4d84b5987a0dc51d992f3a446abbd90f9b7ff1457336fa1d1157` |
| `artifacts/checks/curriculum/07-heldout.json` | `38186fc33a15af761c5e4bc5baf00a047cf4009afbb4b46967eb66add6c5c332` |
| `artifacts/checks/curriculum/08-own-images.json` | `74941dd06348105327dcb18b1828531137222cafc31570a7edd3d140ac32373c` |
| `artifacts/checks/curriculum/08-own-data.json` | `096eb292085f114c186ed269e8391ff056496af404468b93acf2890be4bf2d26` |
| `artifacts/checks/grid-learning.json` | `c66887b113691433f4f8787cb55fbbb6f4dacf26d635565dd259b5988417a298` |
| `artifacts/checks/curriculum/custom-data-160-step.json` | `d1b5582921555b93338f26088ceb2d4e3a84b8a4b04266965df86a4ea37b0f96` |
| `artifacts/checks/curriculum/custom-data-learning.json` | `fb22ce8356fbd747f967c3b82df53bd55de840d826d44fc73abfdba315f8e584` |

### 本輪 checkpoint 與輔助檢查

| 檔案 | SHA-256 |
|---|---|
| `artifacts/runs/grid-learning/checkpoint.pt` | `d8fd0bc8d308a00f07a4072687fd4ab0e1697bc3b2a40374ab1cd42d0b72c17e` |
| `artifacts/runs/custom-data-learning/checkpoint.pt` | `5ad263945f2a9c2b10cf2ef2a10d637b50aa18acc29683ae7c29dca64e1c9ad6` |
| `artifacts/runs/custom-data-learning-1600/checkpoint.pt` | `e1ba1ee09413729599abc39e90e53911ce71e5982d617686e846d0426d99b3d9` |
| `/tmp/clear-grid-tech/audit_grid.py` | `1bada590911912ce65de4008fec51bcd7aece3bc4b0df5320a4bd0392884c9fe` |
| `/tmp/clear-grid-tech/audit-results.json` | `21bd2646049abb39f64cb64ed906bee68cd1050ae567df30b20ebdd573a83c5b` |
| `/tmp/clear-grid-tech/reevaluate_checkpoints.py` | `05428181ec0734b13518980777cbefbe06fc8799e03c9cf73bce277889851115` |
| `/tmp/clear-grid-tech/checkpoint-reevaluation-results.json` | `87be639b83764deca89f242759c327b1cd8a053581393fc48f22d413632c8e78` |
| `/tmp/clear-grid-tech/checkpoint-loss-results.json` | `3f2b0e0c6bd4ab9f6448f5efbae86b83c065e6b28e3944d4d2f3d54deac4f89a` |
| `/tmp/clear-grid-tech/final-page-inventory.json` | `22869bd806c917bcdf1f74e42046fc7160173b0e2550bfc272db3ef43620821c` |

## 08 自訂資料兩張可讀圖的追加核回

協調者在實際 Zensical desktop 上發現合併版文字過小後，改用同一份既存 JSON 重排為兩張 SVG。我實際讀回 08-own-data 的曲線／四張圖說與補充連結，閱讀 `/tmp/clear-tutorial-readable-custom.py`，另寫獨立 XML／數值檢查 `/tmp/clear-grid-tech/check_readable_svgs.py` 並在 CPU 執行。本次沒有重訓 1600 步，未修改既存 JSON、checkpoint 或原始合併圖。這是數據與圖來源核回；我未另做 browser 檢查，不能以 XML 正確宣稱 desktop／mobile 已驗。

loss 圖有完整 1600 個 polyline 點，逐點核對 `x=60+(step−1)×430/1599`、`y=285−total×105`；每個座標只有 SVG 三位小數的捨入差，最大差 0.0004997811 SVG 單位。紅虛線實際位於第 160 步，橫軸明說各次更新前；正文將 minibatch 起点約 1.56 與完整 24 張 train 起点 1.55017 分開，沒有把更新前 loss 說成更新後。兩張圖都含 `title`、`desc`、520 寬 viewBox 及同一 source JSON SHA-256。

預測圖的四張圖順序是 validation index 1、2、3、0。逐一 base64 解碼比對：PNG bytes 和原合併圖完全一致，也符合 JSON 的四個 `letterbox_png_sha256` 與現存 PNG。全部 3 個 GT／4 個 prediction rect 的角點與寬高，都符合 `(panel_origin)+(64×64 letterbox coordinate)×220/64`；不是原圖 80×120／120×80 座標。用 JSON 框重新算 IoU，再依同類、score 排序、每 GT 最多一次、IoU≥0.5 重算 TP／FP／FN，全部一致：

| 固定圖 | 預測 score／IoU（SVG 依兩位／三位小數顯示） | 配對 |
|---|---|---|
| 紅 index 1 | 紅 0.9999966621／0.5845538703；黃 0.1581750810／與紅 GT 0.2411930556，沒有同類 GT | 紅 TP、黃 FP，沒有 FN |
| 藍 index 2 | 藍 0.4621165991／0.4058014689 | FP，藍 GT 同時 FN |
| 黃 index 3 | 黃 0.9946288466／0.4206645747 | FP，黃 GT 同時 FN |
| 空 index 0 | 沒有 GT 或預測 | 沒有 TP／FP／FN |

所有顏色、score、IoU、FP／TP、兩個 FN 標示與正文四個項目相符。原合併圖留在補充連結，原 JSON 與原合併圖指紋未變。此 delta 沒有新增 blocking 或 burden。

| 本次核回檔案 | SHA-256 |
|---|---|
| `docs/lessons/08-own-data.md`（取代前表的此頁最終指紋） | `51fd720683e2e36df9ec5b85618caedd6a2f75c4c29e70aa23fbd5c5c5d2f671` |
| `docs/assets/diagrams/08-custom-loss-readable.svg` | `a4d3ce1757586212317e71d15b8212a0780ee30e502e5e28c39959be628b11ed` |
| `docs/assets/diagrams/08-custom-predictions-readable.svg` | `0f935cc449e1f52af61c02a9897e41c68bac866848f008680d325c2a5a1ec157` |
| `/tmp/clear-grid-tech/check_readable_svgs.py` | `f162466744bbf5ad0d7d1b76b5515ebcd42ec21df3c0a470b69b3920dd7f351d` |
| `/tmp/clear-tutorial-readable-custom.py`（閱讀來源，沒有重執行） | `e80265b41a550e63a01a4c969c733c1cab778ebc36a5cf461bf6f3d634dcad03` |

逐值核回結果保存在 `/tmp/clear-grid-tech/readable-svg-check-results.json`。

## 07 推論模式措辭的追加核回

實際重新讀 `docs/lessons/07-inference.md` 第 75–99 行。第 93 行已將 inference-mode 的絕對限制收窄為「對新產生的 tensor 有更多限制，接回某些需要梯度的運算時會報錯」，並將本例限定為純推論；之後要接回訓練計算時建議 `no_grad()`。`eval()` 對 Dropout／BatchNorm 的模式作用與停止記錄計算圖仍分開，本例的 GridDetector 沒有這些層。lesson case／模型沒有改動。

用現有 PyTorch 2.9.1+cpu、2 threads 跑最小 CPU fixture（結果 `/tmp/clear-grid-tech/inference-mode-delta-results.json`）：inference-mode 產生的常數 2 乘上可訓練權重時，因 backward 需保存這個輸入而報 `Inference tensors cannot be saved for backward`；同樣常數由 no-grad 產生，能接訓練乘法並得到權重梯度 2。另一個只加該 inference 常數的例子仍能 backward，權重梯度 1，說明「某些運算」的限定有意義，不可寫成所有需要梯度的運算都失敗。沒有重新訓練，也沒有把 no-grad 說成訓練分支本身會記錄梯度。

這個 delta 沒有新增 blocking／burden。此頁最新正文 SHA-256 是 `37e293a9a25016581b92fc216eb3d868b1c80c828c074ee6593a61006b289bfc`，取代上面正文表的較早版本。這次只核措辭與 CPU 行為，没有另做 browser。

| 本次核回正文 | SHA-256 |
|---|---|
| `docs/lessons/07-inference.md` | `37e293a9a25016581b92fc216eb3d868b1c80c828c074ee6593a61006b289bfc` |

## 07 訓練兩張可讀圖的追加核回

協調者改完連結後，實際重新讀 `07-training.md` 第 138–180 行，核對新曲線／圖板、相鄰數值表、標籤指向、FP／FN 例與實驗限制。閱讀作者暫存生成腳本 `/tmp/07-grid-readable-preview/make.py` 作來源依據；沒有執行會修改 repo 的作者生成腳本或以其自查當獨立核對。我另寫並執行 CPU/XML 檢查 `/tmp/clear-grid-tech/check_grid_readable_svgs.py`，結果存 `/tmp/clear-grid-tech/grid-readable-svg-check-results.json`。

新 loss 圖有 **四條**曲線：total、未乘 5 的 box、objectness、classification，每條全部 160 個點。逐點對回既存 `artifacts/checks/grid-learning.json` 的 loss_history，映射 `x=64+(step−1)/159×426`、`y=438−280×loss/1.25`；只有 .3f 座標捨入差，最大 0.0004995144 SVG 單位。圖中 total=5×box+objectness+classification、32 張／batch 8／seed 7／lr 0.01、每點是更新前該批 loss，均與 JSON 及正文一致；沒有把 validation／test 評分畫成訓練 loss。

新 prediction 圖為固定 index 0–3，同一個 2×2 順序。4 個 PNG data URI 與原生成 SVG 一字不差；逐一 base64 解碼，其 bytes 與現存 4 張 `artifacts/runs/grid-learning/validation-0*.png` 相同。全部 6 GT／6 prediction rect 的資料屬性保留原完整 xyxy，實際圖座標符合 panel 原點＋原 64×64 pixel 座標×220/64；六位小數的座標捨入差≤0.0000005 SVG 單位。自行從框算 IoU，依每圖 score 高到低、同類／尚未配對 GT 重算配對，得到 5 TP、1 FP、1 FN，與圖中文字及正文一致：

| 固定圖／框號 | 原 score／重算 IoU | 圖上判定 |
|---|---|---|
| 0：#0 藍 | 0.9998024106／0.6238343189 | TP |
| 0：#1 紅 | 0.9804154634／0.4734199232 | FP；紅 GT 同時 FN |
| 1：#0 藍 | 0.9999747276／0.7290351608 | TP |
| 1：#1 紅 | 0.9982468486／0.7538482653 | TP |
| 2：#0 紅 | 0.9999918938／0.7713662106 | TP |
| 3：#0 紅 | 0.9999998808／0.8992239606 | TP |

score／IoU 的三位／兩位顯示精度沿原圖，#k 在框旁對到各圖下方資料；正文已改成同一指向。綠虛線 GT／橙實線 prediction、紅 class0／藍 class1、score≥0.05、同類 NMS IoU>0.5、配對 IoU≥0.5 皆一致。四張只是圖板，AP50 使用全部 16 張 validation；表格 mAP50 初始0.0018／最終0.8036、test0.7749、precision15/17／recall15/19 仍對原 JSON，未隨重排改數字。

两圖均含 title／desc、520 寬 viewBox、實際 source JSON SHA-256，所有文字原始字級至少 22。這是來源／數字核回，沒有自己執行 Zensical desktop／mobile；作者 standalone SVG browser 自查不算我完成獨立視覺驗收。原生成兩圖與 JSON hash 未變，本次未重訓 160 步、未啟動 GPU。沒有新增 blocking／burden。

| 本次最終核回檔案 | SHA-256 |
|---|---|
| `docs/lessons/07-training.md`（取代前正文表較早版本） | `444cfaaad5a648dcd95147beedbc4d167f95812956f042983171b82f491d429a` |
| `docs/assets/diagrams/07-grid-loss-readable.svg` | `8975637e5d4912b68f9bbd9a02dd092c31e9d0f029cc5f6268f730d3fe14adec` |
| `docs/assets/diagrams/07-grid-predictions-readable.svg` | `ac0aef6ee5429918f840f78711b29e6abeebc48235456d0e6fb8b151f0c0ecc3` |
| `/tmp/clear-grid-tech/check_grid_readable_svgs.py` | `1fb611c18386508c1ee2a26f8d1e8f2eeb08524803ee3c3429ba3c38108deeba` |
| `/tmp/07-grid-readable-preview/make.py`（閱讀，未重執行） | `5f7b1f99e06f6928415d3adb6e47ff416efaa1911fc08bb2c3218fd98c6a1ca6` |

## 07 獨立評估頁換圖後的追加核回

最終指紋檢查發現 `07-heldout.md` 也剛改用上述新圖，因此實際重讀本頁全文，包含人工 AP 例、三步 smoke、160 步補充與新圖相鄰說明。source、程式、數字沒有變；本頁人工 FP 排到 TP 前使 AP 從 .5 到 .25、刪掉低分 FP 只改 precision 不改 AP，仍對既有 CPU fixture 與手算。160 步同一次實驗、16 圖／18 GT／16 pred／15 TP 的 validation 與 test 的 19 GT／17 pred／15 TP、各類 AP／mAP／precision／recall 及門檻，均仍對前面已核的同一 JSON，不用再重訓。

發現 TGRID-02（burden）：換圖後原圖說還叫讀者找 `#k class c score s` 與 `class 0 score 0.980`，新 SVG 實際使用紅／藍類名。已回報協調者修正，然後實際重新讀第 165–171 行：現在第一行写 `#k 紅／藍 score s`，同句定紅=class0／藍=class1；圖0指到 `#1 紅 score 0.980`，圖旁框號對到圖下資料，0.47 IoU／FP／紅 GT FN 沒變。**TGRID-02 已關閉**，不以換連結本身當成圖說已核。

本頁新圖來源与全部 6 GT／6 prediction、PNG bytes／IoU 配對，沿用上一追加節实际核回的同一張 `07-grid-predictions-readable.svg`；本頁 loss 補充鏈接也指同一新 loss 圖。没有 browser 或新 GPU／160 步訓練。最终仍为 0 未解 blocking／burden。

| 本次最終核回正文 | SHA-256 |
|---|---|
| `docs/lessons/07-heldout.md`（取代前正文表較早版本） | `24d6e1216e4845d996f54b568b7b84c03e0c2868f3da0d538bd30122584c741e` |

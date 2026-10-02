# 獨立技術準確性審查 B：assignment 到 held-out

審查日期：2026-10-02。範圍為 05-assignment、06-decode-nms、06-evaluation、07-data、07-targets、07-loss、07-training、07-inference、07-heldout 共 9 節。這是直接讀教材、程式與保存證據的技術審查，未讀取或引用 reader reports 作為判斷依据。僅本檔由 reviewer 寫入；未改教材、程式、notebook 或 evidence，未做 GPU／長訓練／推送。

## 結論與確實發現

未發現 P1 級核心數學／訓練／指標錯誤。中心責任、格內 xy 與整圖 wh、loss 的 mean 分母、空圖片的背景梯度、人工 AP 面積，以及 160 步補充實驗的保存 checkpoint 指標皆有相互一致的證據。教材明確稱 MiniYOLO 為簡化模型，沒有冒充完整 YOLOv1。初讀版本有兩項 P2 準確性問題；它們不推翻已保存的這批數字：

1. **B-01，P2：三步訓練的總梯度有限性宣稱強於原斷言。** 初读 `docs/lessons/07-training.md:34` 稱檢查可「避免無窮大總量也透過」，但原 `lesson_cases/07-training.py:25–27` 與 notebook cell 3 只有逐元素 `isfinite`，接著 `grad > 0`。有限 float32 元素相加仍可能 overflow：独立驗算 `torch.full((4,),1e38)` 的每項都有限，`abs().sum().item()` 卻是 `inf`，而 `inf > 0` 通過。應明確檢查 `math.isfinite(grad)`，或縮窄教材宣稱。原 case SHA256 為 `e10e2718e387f58d203bf716548905298fd53d1ff2c4ffc44e5d42f5dbca3fca`。主 reviewer 在本次審查期間已加入 `math.isfinite(grad) and grad > 0`；本項保留初讀發現，等待正式同步 notebook／JSON 後的獨立複查。
2. **B-02，P2：重複框必為 FP 的一般化敘述不符合本課 matching 邊界。** 初讀 `docs/lessons/06-evaluation.md:15` 泛稱「後來重複框算 FP，即使 IoU 很高」。兩個 evaluator 實際都先排除已用 GT，再從剩餘 GT 找最高 IoU。若同圖同類 GT 是 `[0,0,10,10]`、`[2,0,12,10]`，兩個預測都等於第一個 GT、score=.9/.8，第二框能改配第二個 GT（半開 IoU=2/3），兩個 evaluator 的 flags 都是 `[1,1]`，AP=1。官方 VOC devkit 先對**全部同類 GT**找最大 IoU，再檢查該 GT 的 difficult／det 狀態；同例第二框是 duplicate FP，AP=.5。應把重複框句子限定本節固定 fixture，并明示本課「最佳尚未配 GT」是 toy 契約，AP 積分同 VOC 不代表整個 evaluator 等同官方 VOC。无需為此強行改變 toy 的程式契約。

P2 表示會讓讀者對檢查保障或一般評估規則形成錯誤認知；非已觀察到這個小模型本輪發散，也非認為自訂 toy matching 本身不合法。待發布的 `lessons-v0.2.0` 不列為技術錯誤。

## 直接讀過與獨立核查方式

- 九節的 `docs/lessons/<id>.md`、`lesson_cases/<id>.py`、`artifacts/checks/curriculum/<id>.json` 全文，以及各 notebook 的所有 4 個 cells（0–3，含 markdown、完整環境格、完整實驗格及保存 output）均已讀過。
- 共享核心全文：`miniyolo/targets.py`、`losses.py`、`inference.py`、`metrics.py`、`data.py`、`models.py`、`train.py`，另讀 `geometry.py`、`checkpoint.py` 與 `scripts/render_learning_evidence.py`。未用退出碼替代語義驗證。
- 初讀時 9 個 notebook cell 3 與 case 逐字相同，保存 output 與 JSON stdout 逐字相同，case 的 SHA256 皆吻合 JSON。9 个環境格的 hash 相同（前 12 碼 `5baba95bf951`），逐字讀過其固定版本、LFS skip、torch／其他依賴安裝及重啟條件。
- 用現有 `.venv-model/bin/python` 在 CPU 小量重跑九節 case：stdout 都精確吻合保存 JSON；另做 batch duplication 梯度、空 GT、重疊 GT fallback 與非正方形座標的獨立驗算。未重做 160 步訓練；對已保存 checkpoint 作純推論重算，另重建 seed=7 的未訓練模型核對 initial validation。
- 完整讀 `artifacts/checks/grid-learning.json`、其指定的 `artifacts/runs/grid-curriculum/history.json`，唯讀載入指定 `checkpoint.pt`，確認 steps=160、config 相同。检查 160 条 history 的 step 顺序及最後各項 loss 對應報告。
- 對 05-assignment、06-decode-nms、06-evaluation、07-data 及兩張 grid-learning SVG 讀 XML／座標，並以 Chromium 將 SVG 渲染到 `/tmp` 後逐張視覺檢查。曲線 4×160 個點都按 history 正確映射；圖板 4 個嵌入 PNG 與原保存檔逐位元相同，12 個 GT／prediction 框全部符合 JSON 的放大座標與取整。圖板沒有把人工 fixture 畫成訓練成果。

## 原始來源核查

### YOLOv1

直接讀提供的原論文全文 `/tmp/yolo_sources/1506.02640.txt` 第 2 節及 2.2 節，尤其文字行 114–134、193–223、224–284（原文双栏抽取顺序需结合章节阅读）。來源：[arXiv 1506.02640](https://arxiv.org/abs/1506.02640)。

- 原文中心責任：“If the center of an object falls into a grid cell, that grid cell is responsible for detecting that object.”
- 原文座標：“The width and height are predicted relative to the whole image.” x/y 則是格內偏移，與本課兩種正規化一致；使用 sigmoid 是本課額外設計，原文 final layer 使用 linear activation。
- 原文 confidence 為 `Pr(Object) × IoU(pred, truth)`，有物件時期望 confidence 對應 IoU；非本課固定正格 target=1 的獨立 occupancy objectness。推論式 (1) 再乘 `Pr(Class_i | Object)`。
- 原文每格 B 框，其責任 predictor 按 “highest current IOU with the ground truth” 決定；VOC 設定是 S=7、B=2、C=20，輸出 `S×S×(5B+C)`。本課每格一個框，省略格內多框 IoU 競爭，不能靠 NMS 恢復同 slot 容量不足的第二個 GT。
- 原文式 (3) 是平方誤差的加總：`λcoord × Σ 1obj_ij[(x−x̂)²+(y−ŷ)²+(√w−√ŵ)²+(√h−√ĥ)²] + Σ 1obj_ij(C−Ĉ)² + λnoobj × Σ 1noobj_ij(C−Ĉ)² + Σ 1obj_i Σc(p(c)−p̂(c))²`，`λcoord=5`、`λnoobj=.5`。本課 `5×positive sigmoid-coordinate mean MSE + all-cell mean BCEWithLogits + positive mean CE` 明顯不同。05-assignment、07-data／targets／loss／training／inference 均明示簡化，尤其 07-loss 主動列出省略平方根 wh、IoU confidence target、多框責任，足以避免誤稱原版。

### VOC、COCO 與 PyTorch

官方原始頁及程式均實際取得，而非只依記憶：

- [VOC2012 官方頁](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) 所連 [官方 devkit tar](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar)。只讀取其中 `VOCcode/VOCevaldet.m`、`VOCap.m`，保存於 `/tmp/accuracy-b-sources/`。`VOCap.m:3–9` 的補端點、右至左 precision max、recall 變化積分與本課 all-points 公式相同；`VOCevaldet.m:73–101` 則明確是全 GT 先取 max 再判已配對。官方使用 inclusive pixel `+1` 面積及 difficult ignore；本課明示半開座標且无 difficult metadata，不能直接當作官方 VOC benchmark。
- [COCO 官方 cocoeval.py](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py) 全文存 `/tmp/accuracy-b-sources/cocoeval.py`。`evaluateImg:265–299` 跳過已配普通 GT，並另處理 crowd／ignore；`accumulate` 用 101 個 recall 門檻插值，預設 IoU=.50:.05:.95、maxDets=[1,10,100]，還有面積範圍與忽略規則。故本課 AP50 與 COCO AP50 也不應宣稱數值完全相同，更不能當 AP@[.50:.95]。教材已经正确解释單門檻／101 recall／其他規則差異。
- [PyTorch v2.9.1 原始 loss.py](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/nn/modules/loss.py)：MSELoss 文件明示 mean 除**所有元素數**；BCEWithLogits 結合 sigmoid 與 BCE，mean 除 loss 元素數；未加 class weight／ignore 的 CE 接 unnormalized logits，按有效樣本數 mean。正文列出的 `4Npositive`、`BS²`、`Npositive` 三個分母均與實際呼叫吻合。另取得同版本 functional.py，確認 functional 默認 reduction='mean'。

## 逐節結果

### 05-assignment

**讀過來源：** [正文](../docs/lessons/05-assignment.md)、[case 全文](../lesson_cases/05-assignment.py)、[notebook 全部 cells](../notebooks/05-assignment.ipynb)、[JSON](../artifacts/checks/curriculum/05-assignment.json)、`05-assignment.svg`，並對照 targets／losses／models 與 YOLOv1 原文。

**核實結果：** 64×64／S=4 的紅藍中心 (12,12)/(44,44) → row/col (0,0)/(2,2)，格內 xy=.75/.75，全圖 wh=.25/.25；兩圖 batch 正2負30，无 ignore。負格 class=-1 是這個独立 `build` 的容器值，共享 `build_targets` 填0但也只讀 positive，兩者沒有相互矛盾的語義。box/class 只用正格；retain_grad 验证的是 head 輸出位置，正文也正確區分共享 head 權重仍會更新。case total 未乘5，但本節僅介紹 mask／人工 head 更新，未稱為後續 `grid_loss`。兩步 box .0871→.0861、obj .7031→.6839、class .5235→.4744 重跑吻合，未被包裝成從影像訓練效果。移動藍框與 S=8 的答案皆正確。

**確實錯誤／嚴重度：** 無新增錯誤。單槽碰撞明確拋錯，属于有意的容量限制。

### 06-decode-nms

**讀過來源：** [正文](../docs/lessons/06-decode-nms.md)、[case](../lesson_cases/06-decode-nms.py)、[notebook 全部 cells](../notebooks/06-decode-nms.ipynb)、[JSON](../artifacts/checks/curriculum/06-decode-nms.json)、`06-decode-nms.svg`，共享 inference／geometry 與 YOLO 原始 confidence 式。

**核實結果：** row1/col2、xy=.25/.75 → center (36,28)，wh=.25/.125 → size (16,8)，xyxy `[28,24,44,32]`；scores .64/.72/.855 由 obj×最大 softmax 得到。重复框 IoU=89.6/166.4=7/13≈.53846，NMS >.5 抑制低分者、等於門檻則保留，class-wise 與共享 NMS 一致。keep=[2,1]；提高 score 到 .75 只留背景錯框；NMS=.6 保留 [2,1,0]；score=.70 新陣列只留 .72/.855。不同類相同框及空輸入契約正確，重跑輸出精確相同。第6章 decoder 不 clamp，第7章共享 decoder clamp 且过滤退化框，07-inference 已明說此差異。

**確實錯誤／嚴重度：** 無。人工反例與 score 非 precision 的聲明成立。

### 06-evaluation

**讀過來源：** [正文](../docs/lessons/06-evaluation.md)、[case](../lesson_cases/06-evaluation.py)、[notebook 全部 cells](../notebooks/06-evaluation.ipynb)、[JSON](../artifacts/checks/curriculum/06-evaluation.json)、`06-evaluation.svg`，共享 metrics、官方 VOCevaldet/VOCap、COCO evaluator。

**核實結果：** 主例 flags=[FP,TP,FP,TP]，TP2／FP2／FN1，末 P=.5、R=2/3；包絡積分 AP=1/3，不是鋸齒曲線梯形積分。score≥.85 的 AP=1/6／R=1/3；刪已知最高分FP 后 AP=1/3+2/9=5/9。重跑全部相同。每類跨圖排序且只能同圖同類配對，無 GT 類別 AP=None、全無GT map=None、有GT無pred AP=0；背景圖可增加 FP。正文將分母0的 micro P/R約定為0，沒有聲稱這是所有工具通用定義。AP50／單門檻 mAP／COCO 多門檻與101recall區別正確。

**確實錯誤／嚴重度：** B-02（P2），重複框句子的泛化超過其「尚未配 GT」契約，並應補官方 VOC matching 的差異。当前人工 fixture 的所有數字无誤。

### 07-data

**讀過來源：** [正文](../docs/lessons/07-data.md)、[case](../lesson_cases/07-data.py)、[notebook 全部 cells](../notebooks/07-data.ipynb)、[JSON](../artifacts/checks/curriculum/07-data.json)、`07-data.svg`，data／targets／geometry。

**核實結果：** RGB CHW float32，紅切片 `[0,12:28,8:24]`、藍 `[2,36:52,40:56]` 精確符合 pixel 半開 xyxy，各256畫素；修改 x2=28 對應20×16=320。由 annotations 重建 mask 與原影像相同是有內容的檢查，不只驗shape。變長 target 保留 list、空框 shape [0,4]、背景圖保留监督均正確。ShapeDataset 用每索引私有 generator seed=base+1009×index，不依取樣順序；4×4不同cell的生成限制在docstring及教材有聲明。固定0/1場景與後續帶[0,.04)背景／.95色塊生成資料亦有區分。8張契約檢查與保存 stdout吻合。

**確實錯誤／嚴重度：** 無。合成資料的簡單／碰撞规避為已說明限制。

### 07-targets

**讀過來源：** [正文](../docs/lessons/07-targets.md)、[case](../lesson_cases/07-targets.py)、[notebook 全部 cells](../notebooks/07-targets.ipynb)、[JSON](../artifacts/checks/curriculum/07-targets.json)，共享 targets／geometry／models、YOLOv1 原文。

**核實結果：** 紅 `[8,12,24,28]` center (16,20) → cell(x1,y1)、target `[0,.25,.25,.25]`；藍center(48,44) → cell(x3,y2)、target `[0,.75,.25,.25]`；整批positive indices `[[0,1,1],[0,2,3]]`、正2負30。邊界floor、row/col與xy順序、空圖、同格同類碰撞拋錯皆正确。寬高相對完整W/H而非cell；独立補驗 `(H,W)=(32,64)` 的 `[16,8,32,16]` target `[.5,.5,.25,.25]` 解码可精確還原。sigmoid有限logit無法恰到0的限制在07-inference 已解釋。

**確實錯誤／嚴重度：** 無。省略原版多框IoU責任明確標為教學簡化。

### 07-loss

**讀過來源：** [正文](../docs/lessons/07-loss.md)、[case](../lesson_cases/07-loss.py)、[notebook 全部 cells](../notebooks/07-loss.ipynb)、[JSON](../artifacts/checks/curriculum/07-loss.json)，losses／targets、YOLOv1式(3)、PyTorch 2.9.1 MSE/BCE/CE原始文件。

**核實結果：** 零logits sigmoid=.5、class softmax=.5，box=(.25+.0625+.0625+.0625)/4=.109375；obj與class各ln2；total=5×.109375+2ln2=1.933169。mean分母依序4Npos／BS²／Npos，无 class weights 或额外 noobj weight。obj logit梯度 `(σ(z)−t)/16` → 正−.03125、背景+.03125；對負格 box/class 所有通道梯度都0。全空 batch 的兩個連圖scalar0可backward，obj仍有1/32背景梯度，避免空mean NaN。補驗兩份独立prediction槽的mean loss均不變，每槽obj梯度−.015625，與正文「共享CNN兩張貢獻再累加」相符。CE共同平移不變的練習正確。正格零x target需要sigmoid极限是既有可表示性限制，不改变這裡的人工loss算式。

**確實錯誤／嚴重度：** 無。未誤稱BCE/CE為原YOLOv1 loss。

### 07-training

**讀過來源：** [正文](../docs/lessons/07-training.md)、[case](../lesson_cases/07-training.py)、[notebook 全部 cells](../notebooks/07-training.ipynb)、[JSON](../artifacts/checks/curriculum/07-training.json)，models／train／losses／data／targets，補充grid-learning JSON／history／checkpoint與兩張SVG。

**核實結果：** CNN三次stride2輸出32/16/8，再adaptive pool4與conv/head，permute得到[4,4,4,7]。wh bias−1.8／obj−2 是人工初始化，无預訓練。4張train每圖1物件，正4負60，Adam=.01，三個真實optimizer steps；zero_grad／backward／step與原參數副本檢查方向正确。保存 total .9817/.9248/.8323 與所有分項、正負均值重跑精確吻合；均值確在更新前prediction上計算。教材無三步已學會／泛化的宣稱。可選160步命令與三步的檔案／展示範圍有清楚區分，SVG不隨三步自動更新。

**確實錯誤／嚴重度：** B-01（P2），原有限梯度總量保障缺少reduction後的finite判斷；主 reviewer 已开始修正，需同步證據後再複查。除此之外，三步執行與160步受控效果敘述有證據。

### 07-inference

**讀過來源：** [正文](../docs/lessons/07-inference.md)、[case](../lesson_cases/07-inference.py)、[notebook 全部 cells](../notebooks/07-inference.ipynb)、[JSON](../artifacts/checks/curriculum/07-inference.json)，inference／geometry／models／targets／data與07-data图。

**核實結果：** eval與inference_mode用途有正確區分。每格最大softmax類別、sigmoid(obj)×probability、score≥threshold、同類 NMS 與返回list均吻合实现；最多16候選、batch=1保留軸、空結果[N=0,4]契約正确。one_image 清掉藍通道，不是只刪標註。未訓練模型counts [0,0,0]與人工fixture [2,1,0]分别標示，後者與人工已知答案的誤差限.02pixel相符。零offset的clamp1e−4造成x約.0016pixel偏差，保存`[8.0016002655,12,24.0016002655,28]`正确。相鄰同類人工重疊框 NMS=1得2／=.5得1。獨立 .72 score fixture 確在.70保留、.75刪除，未拿接近1的主fixture误驗證。原圖非正方形時undo_letterbox與GT不准clamp掩錯的說明合理。

**確實錯誤／嚴重度：** 無。未把人工框還原當作模型能力。

### 07-heldout

**讀過來源：** [正文](../docs/lessons/07-heldout.md)、[case](../lesson_cases/07-heldout.py)、[notebook 全部 cells](../notebooks/07-heldout.ipynb)、[JSON](../artifacts/checks/curriculum/07-heldout.json)，train／data／models／metrics／inference，補充grid-learning JSON／history／checkpoint與兩張SVG、官方VOC／COCO原始碼。

**核實結果：** 兩圖2個GT，人工 TP/FP/duplicateFP、只到R=.5，因此AP=.5、末P=1/3、无GT类None，刪後兩FP后AP仍.5、P=1。三步訓練仅seed7的4張圖，heldout seed901只推論／評估，实测0 AP 被正确稱pipeline smoke。正文正確說反覆用heldout選參數會成validation，最終test需另留。

160步補充設定的train32/seed7、val16/seed700、test16/seed7000、64px、batch8、width8、Adam .01、score≥.05、NMS=.5、matching=.5 全与保存config及train程式一致。32／16／16張影像hash的三對交集全部0；各split的每索引生成seed集合也不相交。代码optimizer loop只索引train；validation僅更新前後評估，test僅更新後評估，未參與backward或模型選擇。這能支持程式上的切分隔離，不能證明作者在程式外从未看test後調參，正文有「固定設定」與单seed小樣本的限定。

獨立重建初始化後 `initial_validation map=.0017507002801120447`、P=.01171875、R=1/6 全精確吻合。保存checkpoint純推論重算val mAP=.8035714285714286（class AP .8571428571/.75、GT14/4、15TP/16pred）；test mAP=.7748917748917749（class AP .8571428571/.6926406926、GT7/12、15TP/17pred、R15/19）。四張val範例 boxes/scores/labels都精確吻合，第一張左上预测上移約3.15px的圖與文字相同。曲線及图板确来自保存结果，mAP用16張而非圖板4張。只有「相同受控生成分佈能學動」的有限结论，没有外推照片、benchmark優勢或現代機制比較。

**確實錯誤／嚴重度：** B-02 的官方 VOC matching 邊界說明也應覆蓋本節對VOC的參考；其自身人工fixture及160步指標正確。無额外數值錯誤。

## 時間與證據範圍

JSON九節都明示`process_wall_seconds`涵盖**整个subprocess**（import、計算、輸出、圖／檔案），非模型benchmark。保存值依序為：

| 節 | 保存 subprocess 秒 |
| --- | ---: |
| 05-assignment | 2.073 |
| 06-decode-nms | 1.3255 |
| 06-evaluation | 1.3358 |
| 07-data | 1.3551 |
| 07-targets | 1.1974 |
| 07-loss | 1.1632 |
| 07-training | 1.9452 |
| 07-inference | 1.1136 |
| 07-heldout | 1.9139 |

补充 `grid-learning.json training_seconds=.7360584409999547` 对應train.py在初始化／资料／initial evaluation之后、训练loop之前起表，loop及其中日志后止表，final evaluation／checkpoint／画图均在止表后。CPU硬体AMD EPYC 9V74、torch2.9.1+cpu、2threads与正文约.74秒一致。没有将此解释成纯模型kernel耗时、整程耗时、GPU速度或真实资料训练预估。此次九节重跑墙钟约1.17–2.09秒，与保存运行可以自然不同；不作速度比较，也不替换作者原计时。没有重新训练160步来宣称复现原秒数。

## 複查待辦

主 reviewer 修改後，独立检查 B-01 的实际finite条件、notebook cell3／case一致、JSON SHA與新运行输出；以及 B-02 的文字是否限定fixture、明示本課matching与官方VOC不同，并保持既有toy计算契约。不得因文件已改或程序退出0就自动宣告修复。

## 修訂後獨立定點複查：B-01、B-02 均關閉

2026-10-02 依主 reviewer 的修訂通知，重新讀取 `lesson_cases/07-training.py`、對應 notebook cell 3、`artifacts/checks/curriculum/07-training.json`、index 的該節項目，以及 `docs/lessons/07-training.md`、`06-evaluation.md`、`07-heldout.md` 的相關段落；也重讀共享 metrics 的配對實作及先前取得的官方 VOC 原始匹配程式。原始發現與初讀數字保留在上方，以下為修訂後狀態。

**B-01：已關閉。** case 已 `import math`，在逐元素有限檢查與梯度 reduction 之後、`optimizer.step()` 之前，明確執行 `assert math.isfinite(grad) and grad > 0`。從實際修訂 case 的 AST 抽出三個梯度檢查／reduction statements，直接執行於受控的真實 `Parameter.grad`，而非另外手寫一份理想檢查：

| 邊界輸入 | 逐元素有限 | reduction 結果 | 實際條件 |
| --- | --- | ---: | --- |
| `[1,2,3,4]` | True | 10 | 通過 |
| 四項 `1e38` float32 | True | inf | 拒絕 |
| 四項0 | True | 0 | 拒絕 |
| 含inf | False | 未進入reduction | 拒絕 |

這證明原先會透過的「有限元素、總量 overflow」邊界現在確實被拒絕。修訂 case SHA256 為 `fe1037df4ece00bef8810d5d3a31184c1a862b926306688759bcb5cb748afc99`，與新 JSON 及 index 完全一致；notebook cell3 與 case 逐字相同，保存 output 與 JSON stdout 逐字相同。新 evidence 的執行時間為 `2026-10-02T17:27:01.568433+00:00`，subprocess wall time=1.8978 秒，計時範圍仍有正確限定。独立再次 CPU 跑 case，三步全部 stdout 精確吻合保存值、stderr 為空，並保留參數變化與正格數檢查。因此正文關於總量有限的宣稱現在成立；上方舊1.9452秒是初讀紀錄，不是修訂後證據。

**B-02：已關閉。** `06-evaluation.md:15` 已將 duplicate FP 限定為「本節主例中……找不到另一個達標GT」；第17行新增本課先排除已用GT、官方VOC先對全部GT找最高IoU再檢查已用狀態的差異，以及重疊GT時可能判定不同、COCO crowd／ignore／候選數上限與正式benchmark需用官方工具的限定。`07-heldout.md:9` 也直接補上相同matching差異，連結人工AP章，並明說以下數字不是官方VOC／COCO benchmark成績。兩處都與前次實讀原始 evaluator 相符，沒有把同一AP積分公式當成同一完整評估協議。

重新執行原重疊GT反例：GT=`[0,0,10,10]`、`[2,0,12,10]`，兩個預測均為第一個框、分數=.9/.8。共享 evaluator 仍回 AP=1、P=1、R=1，独立 lesson evaluator flags仍為 `[1,1]`；按已讀官方VOC的 max-first 邏輯則 `[1,0]`、AP=.5。保留這個差異正是已說明的toy契約，修訂没有以改变程式來掩蓋它。独立再跑 `06-evaluation.py`，所有主例、截斷及移除FP的 stdout 精確吻合原 JSON，case hash、notebook與原證據仍一致。文字修改沒有擾動本課既有人工數值或補充模型指標。

本輪定點複查未發現新增問題；B-01、B-02 無剩餘待辦。此關閉結論限於以上兩項修訂與其直接依賴，未重做GPU、160步訓練或未要求的外部發布。

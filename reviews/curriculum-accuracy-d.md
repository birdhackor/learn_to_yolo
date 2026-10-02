# 獨立技術核實 D：增強、IoU loss、YOLOv8／v10 機制

核實日期：2026-10-02。範圍為 `11-augmentation`、`11-iou-loss`、`12-anchor-free`、`12-decoupled-head`、`12-assignment`、`12-dfl`、`13-dual-assignment`、`13-nms-free`。本報告從教材與原始來源重新核對，未使用舊 reader reports 作為判定依據。

**結論：八節沒有發現需要修正的實際技術錯誤。** 下述官方差異均已有教學簡化聲明；不要求小案例重現完整 detector。也不把待發布的 `lessons-v0.2.0` tag 算作技術錯誤。

嚴重度採：P1＝核心機制或梯度路徑錯誤；P2＝數值、單位、mask、索引等會使本節結果或結論錯誤；P3＝可選的精確化說明。每節均列出判定；「無」表示未發現實際錯誤，而非宣稱已验证完整模型效果。

## 核實與執行範圍

- 完整閱讀八份 `docs/lessons/*.md`、八份 `lesson_cases/*.py`、八份 notebook 的全部 32 個 cells、八份 `artifacts/checks/curriculum/*.json`，以及本組唯一引用的 `docs/assets/diagrams/11-iou-loss.svg` 的全部內容與幾何配置。
- 每份 notebook 均有 4 cells。兩個 code cells 均通過語法編譯；實驗 cell 3 與對應 case 原始碼逐字一致，並用 `.venv-model/bin/python` 在 CPU 各自獨立重跑，全部 exit 0。
- 八份 case SHA-256 均與 JSON 相符；重跑 stdout、notebook 保存的 stream stdout 均與各自 JSON stdout 完全相符。沒有把既有 `passed: true` 當作獨立核實的替代。
- notebook 環境 cell 1 已完整靜態核對，沒有執行 clone／安裝或聲稱已驗證全新 Colab。tag 的發布狀態依本輪範圍排除。
- 另外執行短小探針：矩形影像 flip／混合 keep mask；DIoU 梯度有限差分；非零長寬比 CIoU 對照作者 implementation；作者 v10 TAL 的真實幾何衝突；DFL 批次梯度與邊界；作者 v10 top-k postprocess 的重複框反例。探針只在 `/tmp`／記憶體執行，未改正式產物，未用 GPU 或長訓練。

## 11-augmentation

**來源與實際讀取內容**

- [YOLOv4 原論文][v4] §2.2 明列 geometric distortion 的「random scaling, cropping, flipping, and rotating」；§3.2 說 Mosaic「mixes 4 training images」；§3.4 將 Mosaic 列入 BoF。
- [YOLOv5 v6.0 augmentations.py][v5aug] `copy_paste` 用 `w - l[3]`、`w - l[1]` 變換框邊界；`random_perspective` 在 `[0,width]`／`[0,height]` 裁切，再用同一 candidate mask 過濾整列 targets；`box_candidates` 計算變換前後面積比。實際下載閱讀該作者版本，沒有把論文的 Mosaic 當成這份 flip／crop 實驗。

**核實結果**

- 半開像素區間 `[8,24)` 在 W=64 下變 `[40,56)`；像素索引 `63-x` 與框邊界 `64-x` 的區別正確。clone 與讀舊框也避免覆寫 x1 後再拿新 x1 算 x2。
- crop offset `(16,8)` 將 `[8,12,24,28]` 移成 `[-8,4,8,20]`，裁切後 `[0,4,8,20]`，面積 128／256=.5。`.5` 保留、`.6` 剔除，以及 labels 同 mask、空 boxes `[0,4]`／空 Long labels `[0]` 均正確。
- 補充探針以 `[3,48,64]` 的矩形影像確認寬軸 flip 與雙 flip 復原；三框混合保留 mask `[True,True,False]` 後，labels 同步成 `[0,1]`，不是只在單框特例上吻合。
- flip 後中心 `(48,20)` 在 64 圖、4×4 grid 的位置為 `(x3,y1)`。crop 後需重新 resize／letterbox 與建立 targets 的敘述正確。

**簡化與嚴重度：實際錯誤無。** 可見比例 `.5` 是本例明寫的標註策略，並非 YOLOv4／v5 的必須閾值。CPU、合法框、合法 crop 範圍內的小 helper 沒有一般 GPU device／非法輸入處理，符合本節限定；不把它視為可直接替代正式 augmentation library。教材已指出保留像素卻刪掉標註的風險，也未聲稱增強提升 AP。

## 11-iou-loss

**來源與實際讀取內容**

- [GIoU 原論文][giou] Algorithm 2 steps 5–9、§3.1：`GIoU = IoU − (Ac − U)/Ac`、`LGIoU = 1 − GIoU`；穩定性段落明寫 GT 面積 `Ag > 0`。本次另外下載原始 PDF 並轉文字閱讀。
- [DIoU／CIoU 原論文][diou]「The Proposed Method」Eq. (6)–(12)：`ρ²(b,bgt)/c²`，其中 c 是最小包圍框對角；`v=(4/π²)(arctan(wgt/hgt)−arctan(w/h))²`，`α=v/((1−IoU)+v)`。Eq. (12) 後亦討論原作者對梯度分母 `w²+h²` 的工程處理。
- [作者 v10 metrics.py][v10metrics] `bbox_iou` 的 CIoU 路徑以 `torch.no_grad()` 計算 alpha；[YOLOv4][v4] §3.4 明列 detector BoF 的 CIoU-loss。

**核實結果**

- G、P 各面積 256，union512，C 面積640，GIoU=−.2，loss1.2；左移一像素後 C 面積624，loss `2−512/624=1.179487`。中心 x 梯度 .02 與 lr50 的真實一步更新吻合。
- DIoU 中 `ρ²=576`、`c²=1856`，loss1.310345。補充 double precision 有限差分得到中心 x 梯度約 `.0124851367`，與 autograd 一致；lr100 更新到約38.75149及 loss1.294497正確。
- 純 IoU 的零中心梯度前提是本例的嚴格非重疊區；教材已排除接觸邊界，沒有把分段邊界的梯度也說成零。中心尺寸固定16×16，因此結果不是任意框參數化的全域梯度定理。
- 本例比例相同，v=0，CIoU=DIoU；P=G 時 clamp 避免 alpha 的0/0，四個 loss 都為0。GT 的有限座標與正寬高前提在正文、程式 assertion 相符。
- 補充非等比例 P=`[32,12,56,28]` 時 CIoU forward 與作者 `bbox_iou(...,CIoU=True)` 的 epsilon 差異內相符，梯度有限。detach alpha 仍保留 v 與中心項反傳。
- SVG 的 G/P/C 寬高、兩中心 24 pixel 距離、C 對角平方1856、10倍畫圖比例、640→624與1.2→1.179487標示均與正文相符。

**簡化與嚴重度：實際錯誤無。** 這是自動微分及 detached alpha 的常見 CIoU 寫法，不逐項複製原作者 Eq. (12) 後的梯度修改；教材明確稱為 implementation choice，並承認其主實驗未驗證非零 v 的梯度。該 helper 的任意非法 prediction 也沒有完整修復保證，但本例 `box_from_center` 始終產生正16×16框；不能把 pwh clamp 當成修復錯誤框的承諾。未把單框更新推廣成 detector AP 結論。

## 12-anchor-free

**來源與實際讀取內容**

- [固定 Ultralytics tal.py][utal] `make_anchors`：`grid_cell_offset=0.5`；`dist2bbox`：`x1y1 = anchor_points - lt`、`x2y2 = anchor_points + rb`；`bbox2dist` 是相反方向。
- [固定 Ultralytics head.py][uhead] `Detect._get_decode_boxes` 先用 DFL／距離解碼，再乘 strides。源碼的點與距離處於特徵格單位，解碼後再乘 stride；教材以像素 point 和格單位 distance 寫同一幾何，單位契約已明說。

**核實結果**

- stride8 的 `(column,row)=(2,2)` cell center 是 `(20,20)`；人工 `(24,24)` 的确不在此中心格網上，正文已指出，沒有假冒官方中心點。
- `[12,16,40,36]` 對 point `(24,24)` 的像素距離 `[12,8,16,12]`，格單位 `[1.5,1,2,1.5]`；往返完全一致。`[1,4]` 實際省略 B 軸，正文對 P=1 的說明正確。
- point `(48,24)` 要求右距離−8，與這個全正距離表示衝突。assignment 仍必要；anchor_points 作為參考點也不等於預设尺寸 anchor box。
- softplus + Smooth L1 的80步更新確實將 loss `.376236→.000012`，解碼近真值；練習 `(20,20)`、stride4 得 `[2,1,5,4]` 正確。

**簡化與嚴重度：實際錯誤無。** softplus 小模型、單點及 Smooth L1 是刻意教學選擇，不是官方 DFL head。教材已區分早期無尺寸 anchor 的 grid detector、現代四邊表示與 NMS-free；亦未聲稱 softplus 表示必然提高 AP。

## 12-decoupled-head

**來源與實際讀取內容**

- [固定 Ultralytics head.py][uhead] `Detect.__init__` 分別建立 `cv2` 框與 `cv3` 類別 `ModuleList`，框 output 為 `4*reg_max`，class output 為 nc；`forward_head` 分別執行 box_head／cls_head。legacy 路徑與現行 DWConv 路徑均可在同一固定檔案核對，未拿它當成完整的歷史 YOLOv8 訓練復現。

**核實結果**

- `[2,4,4,4]` 與 `[2,2,4,4]` 的 channel／spatial／batch 軸正確；`boxes` 是 raw logits，softplus 後才是 distances。
- 兩條分支各自單獨反傳時另一分支 `.grad is None`；共享 backbone 兩次都有梯度，總梯度逐元素等於兩份之和。清梯度與 retain_graph 的使用正確，一次 step 真實改變 backbone。
- 參數 `224+584+584+36+18=1446`；直接 `8→6` 1×1 head 是54個 head 參數，且教材指出容量不同而不能公平歸因。
- cosine −.0073僅為這批人工 target 診斷；沒有被描述為模型一般性能或 decoupling 的必然收益。分類權重乘2會同時改共享 backbone 梯度的練習正確。

**簡化與嚴重度：實際錯誤無。** 全正樣本、連續距離、單尺度是清楚隔離梯度路徑的例子；完整模型只對選定正樣本計框 loss 的界線已注明。此處不 detach 與後續 v10 one-to-one 分支 detach 並不矛盾，兩節驗證的架構目的不同。

## 12-assignment

**來源與實際讀取內容**

- [TOOD 原論文][tood] §3.2.1 Eq. (9)：`t = s^α × u^β`，s 為分類分數、u 為 predicted box 的 IoU；training assignment 段落「select m anchors having the largest t values」。§3.2.2 的 instance normalization 將最大 normalized target 設為該 instance 最大 IoU。
- 同篇 supplementary「Optimization」實際衝突規則是「the object with the minimal area」。這與後續 Ultralytics 版本不是同一規則，不能以 TOOD 論文代替現行 source 查證。
- [固定 Ultralytics tal.py][utal] `get_box_metrics`、`get_pos_mask`、`select_topk_candidates`、`select_highest_overlaps` 及 normalization；[loss.py][uloss] `v8DetectionLoss` 建立 alpha=.5、beta=6，供 assigner 的 score／box detach，之後用未 detach logits 計 BCE。

**核實結果**

- 用同一組真實 predicted boxes 算每個 GT 的 IoU，p1 對 A=2/3、對 B=3/7；p2 對 A=.1875、對 B=.9。資格是參考點在框內，並非預測框與 GT 有交集；p2 對 A 有 IoU 但被 inside mask 排除，敘述正確。
- 品質 A=`[.225,.311111,0,0]`、B=`[0,.146939,.567,0]`；top2後依參與衝突候選的純IoU分配 owner `[0,0,1,-1]`。p1移為GT B後 owner `[0,1,1,-1]` 正確。
- 背景−1、空GT、二元foreground target `[1,1,1,0]` 與 mean BCE gradient `[-.125,-.125,-.125,.125]` 正確；assignment不反傳，而另建的可學logits可以反傳，程式和正文一致。

**官方差異與嚴重度：實際錯誤無。** 本例是明寫的部分 TAL：alpha1／beta2、純 IoU、嚴格 metric>0、top2、二元foreground loss。固定官方 source 用 nonnegative **CIoU** 作 overlap、alpha=.5／beta6（loss實際呼叫）、quality class target 正規化，而且給定 valid GT topk_mask 後可選到零 metric，不等同本例的 metric>0規則；其衝突 `overlaps.max(1)` 也不是只比較本例 selected GT 子集合。現行檔案還含小物件候選擴張與 topk2；教材已明确指出不是完整 TAL，也提到小物件擴張，這些差異不構成計算錯誤。若將來提升為 source 導讀，逐項補出此差異可屬 P3 精確化，当前不要求修改。

## 12-dfl

**來源與實際讀取內容**

- [GFL 原論文][gfl] §3 的 DFL 段落、Eq. (4)–(6)：`yhat = Σ P(yi) yi`，相鄰兩點加權 CE；明寫 `Δ=1 for simplicity`，並指出不同分佈能有相同積分 target。Eq. (6) 後给出 minimum solution 的相鄰 bin 權重。
- [固定 Ultralytics loss.py][uloss] `DFLoss.__call__`：`target.clamp_(0, reg_max - 1 - 0.01)`、tl／tr／wl／wr；此固定版本用一次 log_softmax 加两次 gather 实现 CE 等價計算。
- [固定 Ultralytics block.py][ublock] `DFL`：不可訓練的積分 convolution 權重为 `arange(c1)`，reshape 後沿 bin 軸 softmax。

**核實結果**

- y=1.25、target `[0,.75,.25,0]`，uniform p=.25、期望1.5、DFL log4、梯度 `[.25,-.5,0,.25]` 全正確。400步結果與紀錄一致。
- 兩人工分佈 `[0,.75,.25,0]`、`[.375,0,.625,0]` 都有期望1.25；DFL能區分，而只用期望誤差不能，敘述正確。
- K4的距離支撐范围 `[0,3]`；有限softmax logits 的期望实际嚴格位於 `(0,3)`，端點可視為極限。教材「最多」是在描述支撐范围；K16、stride8 的上限120pixel与600／6600个raw輸出數正確。
- 本 helper 要求 `0≤target<K−1`，保证右鄰索引合法；target3在toy被assert拒絕，而固定官方 DFLoss clamp成2.99。本例已明寫與官方的邊界處理差異，不是在宣稱原論文禁止最末端的單bin target。
- 補充兩筆批次探針確認：mean DFL 的每個 logit 梯度是 `(p_j−q_j)/N`，因此絕對值≤1/N；單筆單邊為≤1。**DFL loss值沒有有限上界**，網路參數／座標梯度也不能直接套此logit界線。期望解碼的導數為 `p_j(j−μ)`，由 `p_j(1−p_j)(j−μ_others)` 得每個 logit 導數絕對值≤`(K−1)/4`；乘stride後也要乘相應stride。正文未提出相反的上界主張。

**簡化與嚴重度：實際錯誤無。** 一條邊、K4、無IoU整框loss是已声明的演示；未把分佈寬度稱為經校準不確定性，也未由小步更新推出 AP 收益。最後期望容差與 bin 概率 assertion 對現有範例成立。

## 13-dual-assignment

**來源與實際讀取內容**

- [YOLOv10 原論文][v10] §3.1「Dual label assignments」明确說「we adopt the top one selection」，並说它在作者實驗達到與 Hungarian matching 相同效果；不是宣稱算法相同。
- 「Consistent matching metric」Eq. (1) 是 `m(α,β)=s·p^α·IoU(bhat,b)^β`，s 是點在GT内的 spatial prior。αo2o=rαo2m、βo2o=rβo2m（r>0）使同輸入下 `mo2o=mo2m^r`；預設r=1。推導首先假设初始兩head产生相同p／IoU，不能理解為不同head訓練後始終輸出相同排序；Appendix A亦讀取核對此假设與比例條件。
- [作者固定 v10 loss.py][v10loss] `v10DetectLoss` 實際建立 `v8DetectionLoss(...tal_topk=10)`／`(...tal_topk=1)`；二者 alpha=.5、beta6。[作者 tal.py][v10tal] 實際 topk／衝突處理；[head.py][v10head] `forward_feat([xi.detach() for xi in x], self.one2one_cv2, self.one2one_cv3)`。

**核實結果**

- toy品質表列舉6个 injective 配對，A→p1、B→p0总1.73，owner `[1,0,-1]`；既定最大值先選貪心總1.10。top2加最高quality衝突給 many owner `[0,0,1]`；這裡改用quality而非前節純IoU，正文已明确區分。
- features.detach放在one head前，該head仍有梯度，backbone無one-loss梯度；總loss由many分支提供backbone梯度，兩head都一步更新，與作者位置一致。
- 兩路共用quality表只能示意一致的評分來源，未完整實现式(1)／指數比例／quality targets；正文最后直接承认這一点，所以没有錯稱已验证consistent理論。
- 補充用作者 TaskAlignedAssigner 原始 class 與真實GT／prediction geometry，令兩GT的獨立top1都選p0，衝突後得到 `[0,-1,-1]`；沒有把另一GT補配到p1或p2。GT為 A=`[0,0,20,16]`、B=`[12,0,32,16]`，點=`[(16,8),(8,8),(24,8)]`，P=`[A,[2,0,22,16],B]`，兩類scores=`[[.9,.88],[.85,1e-8],[.1,1e-16]]`。此探針不借不可能的IoU表，確認top1與toy全域枚舉的覆蓋語義确實不同。

**簡化與嚴重度：實際錯誤無。** 全域枚舉明确標成另一個極小教學實驗；正文两次指出不是YOLOv10实际assigner，也未宣稱全域owner是官方輸出。官方top1是局部候選選擇加衝突處理，不能推导成每GT必获一个positive；教材已有覆盖不同的界線。可選补充上面完整consistent公式及官方loss固定鏈接，属于P3加深來源導讀，不是必須修正。

## 13-nms-free

**來源與實際讀取內容**

- [YOLOv10 原論文][v10] §3.1 推論使用one-to-one head；[作者固定 head.py][v10head] 在export路径跳過one-to-many並调用 `ops.v10postprocess`。
- 为核对函数实际內容，另下載閱讀[作者固定 ops.py][v10ops] 的 `v10postprocess`：先按候選最大class score取topk，再按展平的class scores取topk，沒有兩框IoU抑制。
- [作者固定 predictor][v10predict] 明確 `preds = preds["one2one"]`，`mask = preds[...,4] > self.args.conf`。非export的head可回傳兩路字典；實際predictor選one2one，因此不能只見字典含many便推論它把many當部署答案。

**核實結果**

- p0/p1的IoU为90/110=.8181818；mean BCE真實教出约.959／.041。many无NMS3框、many加NMS2框、one无NMS `[0,2]` 正確。
- NMS tie不固定winner，case除框數外真的检查p2与p0/p1其中之一的覆蓋。沒有把「剩兩框」单独当作成功。
- 作者原始postprocess对人工scores `[.92,.90,.80,.05]` 取max_det2確實返回p0/p1两个A框，证明topk沒有每物件唯一性的幾何保證；不等同NMS。
- 人工已知框、按.92/.90/.80排序時TP/FP/TP，precision2/3、recall1，all-points interpolated AP=`.5×1+.5×(2/3)=5/6` 正確；去重后只留两个正确框AP1，top2漏B的recall=.5，與本節限定的人工評分口徑一致。
- threshold.99超過两路本次所有約.959高分，確实全空；不能把零輸出數當模型成功。正文練習已要求更改覆盖與数量assertion，不把score threshold和NMS IoU threshold混為一項。

**簡化與嚴重度：實際錯誤無。** 框位置完全固定、每候選獨立參數、直接指定正target，没有CNN／dual assignment／框学习；正文反复标明界線。它支持监督改变高分重复倾向，不支持真实任务NMS-free AP或延迟提升；最终完成条件中的held-out AP、重复FP、漏检和端到端时间均有注明。

## 本輪需修正項目

沒有 P1／P2 或其他必須修正的技術項目。上述 P3 都是可選的 source 導讀精確化，不宜把合理教學簡化列為錯誤，也不需為本輪核實新增長訓練。

[v4]: https://arxiv.org/pdf/2004.10934v1
[giou]: https://arxiv.org/pdf/1902.09630v2
[diou]: https://arxiv.org/pdf/1911.08287v1
[gfl]: https://arxiv.org/pdf/2006.04388v1
[tood]: https://arxiv.org/pdf/2108.07755v3
[v10]: https://arxiv.org/pdf/2405.14458v2
[v5aug]: https://github.com/ultralytics/yolov5/blob/v6.0/utils/augmentations.py
[uhead]: https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/nn/modules/head.py
[ublock]: https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/nn/modules/block.py
[utal]: https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/utils/tal.py
[uloss]: https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/utils/loss.py
[v10head]: https://github.com/THU-MIG/yolov10/blob/453c6e38a51e9d1d5a2aa5fb7f1014a711913397/ultralytics/nn/modules/head.py
[v10loss]: https://github.com/THU-MIG/yolov10/blob/453c6e38a51e9d1d5a2aa5fb7f1014a711913397/ultralytics/utils/loss.py
[v10tal]: https://github.com/THU-MIG/yolov10/blob/453c6e38a51e9d1d5a2aa5fb7f1014a711913397/ultralytics/utils/tal.py
[v10metrics]: https://github.com/THU-MIG/yolov10/blob/453c6e38a51e9d1d5a2aa5fb7f1014a711913397/ultralytics/utils/metrics.py
[v10ops]: https://github.com/THU-MIG/yolov10/blob/453c6e38a51e9d1d5a2aa5fb7f1014a711913397/ultralytics/utils/ops.py
[v10predict]: https://github.com/THU-MIG/yolov10/blob/453c6e38a51e9d1d5a2aa5fb7f1014a711913397/ultralytics/models/yolov10/predict.py

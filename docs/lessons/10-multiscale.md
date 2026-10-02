# YOLOv3 機制：同一個 pixel 框看兩種尺度

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.2.0/notebooks/10-multiscale.ipynb){ .md-button }

前置：[grid targets](07-targets.md)、[anchor](09-anchors.md)。小物件只有8×8pixel，若整張64×64只保留4×4特徵，每個位置涵蓋16pixel，區域性訊號容易被壓縮。本節研究增加一個較細的8×8預測head，先看shape、encode/decode與成本，再問是否值得留下。

歷史機制：[YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767) 使用三個尺度、anchor與由深層向高解析度融合的特徵。本章簡化為stride8／16兩個尺度，沿用每格單框、互斥兩類及grid loss；沒有Darknet-53、三組anchor或完整原版配置。本次起始是第7章grid分支，僅研究預測尺度；第11章再獨立加入融合，避免把效果同時歸因給兩件事。

## 同一個小框的兩種 target

原框 `[5,5,13,13]`，中心 `(9,9)`、寬高 `(8,8)`，單位pixel。64×64輸入下，4×4 grid的stride=16；8×8 grid的stride=8。stride表示相鄰特徵位置相隔多少輸入pixel，不是說感受野剛好等於stride。

| 專案 | 4×4／stride16 | 8×8／stride8 |
| --- | --- | --- |
| center ÷ stride | `(.5625,.5625)` | `(1.125,1.125)` |
| cell `(gx,gy)` | `(0,0)` | `(1,1)` |
| 格內 xy | `(.5625,.5625)` | `(.125,.125)` |
| normalized wh | `(.125,.125)` | `(.125,.125)` |
| decode cx | `(0+.5625)×16=9` | `(1+.125)×8=9` |

注意wh仍除以全圖64，不跟grid變；第9章若改成anchor表示，則該尺度anchor也必須有清楚單位。本節沒有混用兩種參數化。兩種target都應還原到同一個框，不能讓小框因尺度不同被放大兩倍。

## 兩個 head 的實際連線

影像 `[1,3,64,64]` 經三次stride2卷積，得到fine feature `[1,16,8,8]`；再一次stride2得到coarse feature `[1,32,4,4]`。各自的1×1卷積輸出7通道，轉為 `[1,8,8,7]` 與 `[1,4,4,7]`。七通道依序是`[tx,ty,tw,th,obj_logit,class0_logit,class1_logit]`，即4框值＋1物件分數＋2類別；xy／wh在decode時經sigmoid，wh成全圖寬高比例。每格一框，可把fine／coarse視為`[1,64,7]`與`[1,16,7]`，合計80個位置；80是score篩選與NMS之前的候選數，不是最後顯示框數。

案例將8×8小物件只分配fine，24×24大物件只分配coarse；這是人工固定的尺寸責任，不是原版最佳anchor assignment。

```python
fine_feature = early(image)
coarse_feature = deep(fine_feature)
fine_logits = fine_head(fine_feature).permute(0,2,3,1)
coarse_logits = coarse_head(coarse_feature).permute(0,2,3,1)
loss = fine_grid_loss + coarse_grid_loss
```

fine只在`(gx=1,gy=1)`標obj=1／class0，coarse只在`(2,2)`標obj=1／class1。未分配給某head的物件不形成正格：大物件中心在fine的`(5,5)`仍是objectness負例，小物件中心在coarse的`(0,0)`也仍是負例；它們沒有框／class監督。這是本章簡化的尺寸責任規則，沒有ignore設計，不能當成完整YOLOv3的監督。

責任只依尺寸，與class id無關，兩head都可預測0／1兩類；case把小物件class改成1，確認正格mask不變。本章類別為softmax互斥單標籤，原版YOLOv3使用各類獨立logistic分數，兩者不是同一head定義。

負格提供各尺度背景監督。推論時先將所有尺度decode到同一個輸入pixel坐標，再串接boxes/scores/labels，最後做一次分類別NMS；不能直接串接不同尺度的cell索引，也不能只做各尺度內部NMS而漏掉跨尺度重複框。推論沒有GT尺寸可查，不能先按真實物件大小刪掉另一head；應讓各head的分數與共同NMS處理候選。

案例刻意讓兩尺度都人工預測同一小框，先保留完整decode結果，再沿候選軸拼接：

```python
boxes = torch.cat([p['boxes'] for p in decoded_scales],dim=0)  # [2,4]
scores = torch.cat([p['scores'] for p in decoded_scales],dim=0) # [2]
labels = torch.cat([p['labels'] for p in decoded_scales],dim=0) # [2]
for cls in labels.unique():
    ids = torch.where(labels == cls)[0]
    keep = ids[nms(boxes[ids],scores[ids],iou_threshold=.5)]
```

同類兩框重疊為1，單次分類別NMS留下其中一個，數量2→1。本toy每尺度本來只有一框，所以各自decode helper內的NMS不會代替跨尺度去重；完整case會收集各類keep索引。

執行 https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.2.0/notebooks/10-multiscale.ipynb 或在 repo 根目錄執行 `PYTHONPATH=. python lesson_cases/10-multiscale.py`。應核對表內兩個target、head shapes、候選數`64+16=80`；兩head均有非零梯度並做一次更新。最後人工logits分別經兩種尺度decode，同時還原小框，再真的cat boxes／scores／labels及分類別NMS，核對跨尺度重複框2→1。這是機制驗證，沒有小物件AP提升證據。

## 收益、代價與適用條件

較細特徵提供更多空間位置，小框在stride8下涵蓋約一格而非半格，可能更容易保留細節，也減少某些同格碰撞。代價是候選數從16到80，logits元素從112到560，額外head與高解析度feature增加計算及記憶體。細尺度沒有憑空增加原影象素：若原圖小物件已被resize到2pixel，增head仍不能恢復消失的資訊。

正式對照固定來源切分、訓練步數與評估門檻，分別報小／大物件、候選數與端到端時間。若新增head卻只延長訓練或改變輸入尺寸，便無法把改善單獨歸因多尺度。本節先保留兩head作後續融合實驗，保留的是可研究的介面，不是宣告效果較好。

自主練習：中心 `(28,20)` 在stride8下落哪格？答案：`(gx=3,gy=2)`，格內xy`(.5,.5)`。wh為16×16時兩尺度都使用`.25,.25`。常見錯誤是xy索引對調、把fine wh除以8、某個head沒有loss卻期待它學會。

## 從單步接線到40步實際預測

沿用同一個TwoScale模型與上方1張、2個矩形資料，seed7、Adam lr=.01，延長到40步。兩head每次都有有限非零梯度，權重L2差異為6.504；更新前loss由3.6594降至最後一次更新前的0.0703。

![40步兩尺度模型的真實曲線與預測框](../assets/diagrams/10-multiscale-learning.svg)

右圖綠框為GT、橙框為模型logits經兩尺度decode、合併與分類別NMS得到的框，沒有人工塞入高分答案。本次同一張訓練圖mAP50為0.5000；這次還沒完整學會本例：小物件AP50為0、大物件為1，平均才是.5；另有一個低分背景框。loss下降不保證小框IoU達標。這只是固定訓練圖的檢查，**不是獨立資料的小物件AP提升證據**，也沒有與單尺度做品質對照。

可在本節Colab環境格後另開code cell：`!python scripts/run_multiscale_learning.py`。[完整紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/10-multiscale-learning.json)。

此處沿用第7章的target／loss／decode契約，backbone另用TwoScale，不是對GridDetector原封不動加head。若要只比較多head的品質，應固定TwoScale的backbone、初始化和預算，另做只留coarse head的對照。Colab補充腳本後可用`from IPython.display import SVG, display; display(SVG(filename="docs/assets/diagrams/10-multiscale-learning.svg"))`查看當次結果。

曲線橫軸是訓練步序，每點loss在該次更新前量測；框與AP則是在40次更新全部完成後評估。

<!-- curriculum-evidence:start -->

## 本輪實際執行紀錄

本節範例已於 2026-10-02 使用 PyTorch 2.9.1+cpu 在 CPU 執行，程式中的斷言全部通過。以下是該次輸出；人工輸入、短步更新與模型效果的意義仍依本頁說明區分。[完整紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/10-multiscale.json)

??? example "展開本次實際輸出"

    ```text
    cross-scale duplicate boxes before/after class-wise NMS 2 -> 1
    stride16 small target [0.5625, 0.5625, 0.125, 0.125]
    stride8 small target [0.125, 0.125, 0.125, 0.125]
    head shapes (1, 8, 8, 7) (1, 4, 4, 7) candidates before score/NMS 80
    both heads received gradients; one update; scale encode/decode checked
    ```

<!-- curriculum-evidence:end -->

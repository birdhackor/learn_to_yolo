# IoU 類 loss：沒有重疊時還能往哪裡移

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.2.0/notebooks/11-iou-loss.ipynb){ .md-button }

前置：[IoU與框](04-localization.md)、[grid loss](07-loss.md)。座標MSE衡量各數字的差，評估卻依框重疊判定。這次只研究定位loss，固定兩個pixel框與框表示，不同時修改head、assignment或augmentation。

歷史來源：[GIoU](https://arxiv.org/abs/1902.09630) 引入最小包圍框項；[DIoU／CIoU](https://arxiv.org/abs/1911.08287) 再考慮中心距離與長寬比；[YOLOv4](https://arxiv.org/abs/2004.10934) 使用CIoU。本章以一個可微中心參數實驗四個定位loss，不重現完整v4/v5訓練配置。起始是相同的人工框；實驗後確認梯度行為，是否在detector保留仍需固定資料與預算評估。

![真值G、預測P、包圍框C與中心距離](../assets/diagrams/11-iou-loss.svg)

## 兩個16×16框，卻完全不碰

GT紅框 `G=[8,12,24,28]`，中心 `(16,20)`；prediction `P=[32,12,48,28]`，中心 `(40,20)`。寬高都16，面積各256。交集0、union512，因此IoU=0，`L_IoU=1−IoU=1`。

框P尚未碰到G，小幅左右移動仍沒有交集。此處交集寬的clamp落在負值區，導數為0，所以單純IoU loss對中心的梯度是`[0,0]`。這是本例的非重疊區，不是說IoU在所有位置都沒有梯度；邊界處還有分段與不可微的細節。

pixel xyxy的MSE則是 `(24²+0²+24²+0²)/4=288`，與normalized loss的尺度不同；不能拿這個288和本節約1的數字直接比較效果。本節都用pixel幾何計算，IoU比例自身無單位。

符號補充：ρ是兩個中心的距離，c是包圍框對角線長度，平方後都用pixel²；π是圓周率。atan是反正切，把寬高比轉成角度再比較。alpha／α是控制長寬比懲罰份量的係數，不是optimizer learning rate。clamp(min=...)是把過小值設到下限，避免分母0；detach只在反傳時把該係數當常數。

## 包圍框與中心距離補上訊號

最小包圍框C為 `[8,12,48,28]`，寬40、高16、面積640。GIoU=`IoU−(area(C)−union)/area(C)=0−128/640=−.2`，所以`L_GIoU=1.2`。負GIoU合法，說明兩框相隔，不能clamp成0而丟掉訊號。

若P往左移1pixel，union仍512，C寬從40變39、面積640變624，`L_GIoU=2−512/624=1.179487`，比1.2低，且兩框仍未碰到。本例GIoU中心梯度是`[.02,0]`；SGD減去正的x梯度，所以x變小。case真的以lr50做一步，把中心40移到39，驗證數值與方向。

DIoU加入中心平方距離除以包圍框對角平方：`ρ²=24²=576`，`c²=40²+16²=1856`，`L_DIoU=1−IoU+ρ²/c²=1.310345`。ρ²與c²都是pixel²，比例本身無單位。向左移可減小中心項，所以梯度下降有方向，儘管第一步後仍不一定重疊。

CIoU再加長寬比項`αv`，其中 `v=(4/π²)(atan(wG/hG)−atan(wP/hP))²`。本例兩者比例都1，v=0，所以CIoU等於DIoU。若比例不同纔有額外項。案例以detach的alpha權重計算梯度，明確展示一個常見實現選擇，不把它當成所有庫的唯一寫法。

```python
center_penalty = squared_center_distance / squared_enclosing_diagonal
v = 4 / math.pi**2 * (atan(gt_ratio)-atan(pred_ratio))**2
alpha = (v / (1-iou+v).clamp(min=1e-6)).detach()
ciou_loss = 1-iou + center_penalty + alpha*v
```

分母下限處理相同框時iou=1、v=0造成的0/0；此時alpha與v的乘積為0，CIoU loss也為0。正文與case都使用相同clamp。

## 實際一次更新的核對

執行 https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.2.0/notebooks/11-iou-loss.ipynb 或在 repo 根目錄執行 `PYTHONPATH=. python lesson_cases/11-iou-loss.py`。初始loss應為IoU1、GIoU1.2、DIoU／CIoU1.310345；純IoU中心gradient為零。也核對GIoU梯度`.02,0`與左移1pixel後1.179487。再以DIoU做一次真實SGD，檢查x中心小於40、loss下降且梯度有限。學習率100在這個pixel中心的區域性實驗中用來讓變化可見，不能直接複製到detector所有網路參數。完全相同的GT與prediction時四項都為0。

收益是定位目標與框幾何更直接相連，GIoU／DIoU在本例沒有交集時提供訊號；代價是分段幾何、數值保護、權重與梯度行為需要重新驗證。CIoU的額外項不代表每一種資料一定更好；非正格仍不應計算定位loss，objectness／class也不會被它自動取代。

正式整合時保持decoder、target與mask不變，只把正格定位loss換掉，重新檢查總loss權重與空正格分支。fixed train/validation/test、輸入尺寸及預算，再報告held-out AP與收斂；本節一框移動不能證明AP提升。

本例前提是 GT 有有限座標、`x2>x1`、`y2>y1`，即寬高都為正；程式會拒絕不合法 GT。epsilon 只保護分母，不能修復錯誤標註。這次兩框長寬比相同，因此沒有驗證非零 CIoU 長寬比項的梯度；alpha 的 detach 把它當固定權重，仍保留 v 與中心項的梯度。

常見錯誤：xyxy次序反了產生負面積；沒有epsilon導致零框除0；把GIoU限制到[0,1]；以IoU loss代替全部objectness／class監督。自主練習：若P=G，C面積與union都是256、中心距離0、比例項0，四個loss是多少？答案：都0。若只把P往左移4pixel，變`[28,12,44,28]`，兩框仍隔4pixel，plain IoU仍是1且梯度為0，DIoU則`1+400/(36²+16²)=1.257732`。左移8pixel會剛碰到分段邊界，不能沿用這個零梯度結論。

<!-- curriculum-evidence:start -->

## 本輪實際執行紀錄

本節範例已於 2026-10-02 使用 PyTorch 2.9.1+cpu 在 CPU 執行，程式中的斷言全部通過。以下是該次輸出；人工輸入、短步更新與模型效果的意義仍依本頁說明區分。[完整紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/11-iou-loss.json)

??? example "展開本次實際輸出"

    ```text
    GIoU gradient [0.019999999552965164, 0.0] after 1-pixel real SGD move 1.179487
    initial losses {'iou': 1.0, 'giou': 1.2, 'diou': 1.310345, 'ciou': 1.310345}
    non-overlap plain IoU center gradient [0.0, 0.0]
    DIoU center after one real step [38.751487731933594, 20.0] loss 1.294497
    exact boxes: all four losses zero
    ```

<!-- curriculum-evidence:end -->

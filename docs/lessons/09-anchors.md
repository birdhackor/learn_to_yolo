# YOLOv2 機制：anchor 是尺寸起點

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.2.0/notebooks/09-anchors.ipynb){ .md-button }

前置：[grid targets](07-targets.md)、[loss](07-loss.md)。上一版每格直接回歸 normalized 寬高；這次問「若多數物件的尺寸集中在幾種形狀，是否可以從合適的起點回歸修正量」。anchor 是預先指定的寬高先驗，沒有繫結某一類。

歷史機制：[YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242) 引入尺寸聚類、anchor 框與限制中心在格內的參數化。本章簡化：64×64、4×4 grid、每格兩個 pixel 尺寸 anchor `[16,16]`、`[8,8]`；只用一個人工物件學責任與 offset，不重現 Darknet-19 或完整 YOLOv2 訓練。本次僅替換框表示和槽位責任，先不與多尺度疊加；是否保留須等固定資料對照。

這裡`log`就是自然對數`ln`，`exp`是它的反運算：`tw=ln(2)≈.6931 → exp(tw)=2 → anchor寬16×2=32`。倍率為正，但log修正值可正可負。此case直接把raw槽位表當作可學參數，沒有圖片CNN；更新成功不代表已接好anchor detector。

## 同一個紅框改成怎樣的參數

紅框 `[8,12,24,28]` 中心 `(16,20)`、尺寸 `(16,16)`，仍在cell `(gx=1,gy=1)`，gx／gy是整數格座標。tx／ty是模型原始logits，不是格內比例；另以ox／oy表示比例。中心解碼為 `cx=(gx+sigmoid(tx))×16`、`cy=(gy+sigmoid(ty))×16`。寬高改成 `w=anchor_w×exp(tw)`、`h=anchor_h×exp(th)`。

選 `[16,16]` 時，target 的 tw/th=`log(16/16)=0`；選 `[8,8]` 時則為 `log(2)=.693147`。都是同一個物件，只是起點不同。本例的格內比例是`ox=cx/16−gx=0`、`oy=cy/16−gy=.25`。encode則為`tx=logit(clamp(ox,1e-4,1-1e-4))`、y同理；logit(p)=`log(p/(1−p))`，因此tx約−9.21024、ty約−1.09861。若誤把比例0當tx=0，sigmoid(0)=.5，中心就解到x24而不是16。

case中的logit encode只用於驗證encode/decode；訓練中心loss直接比較`sigmoid(raw tx/ty)`與比例target`(0,.25)`。有限logit無法得到精確比例0，本例用1e-4近似，還原誤差約.0016pixel。

這裡 wh 的 anchor 和解碼結果同為 pixel，所以比值無單位。若換成 feature-map 單位，anchor 必須一併除以 stride=16；不能用 pixel anchor 再多乘一次 stride。

## 一格兩槽，不等於兩種類別

輸出 `[B,4,4,A,5+C]`，本例 A=2、C=2，最後兩個類別 logits 在每個 anchor slot 都存在。兩個 anchor 都可以預測紅類或藍類，框槽與 class 軸獨立。候選數從16增加到32；類別數改成3時，候選數仍32，只是最後軸由7變8。

以「同中心，只比較尺寸」的 IoU 選 best anchor：16×16 對16×16是1，對8×8是.25。教學責任規則是最佳槽為 positive；同一 cell 其他槽若尺寸 IoU>.2 設 ignore；其餘 negative。因此本例 1 positive、1 ignore、30 negative。ignore 不參與 objectness loss，也沒有框／類別 loss。這個簡化 ignore 規則用來呈現三種狀態，不能代稱原版所有訓練細節。

```python
best = size_iou(gt_wh[None], anchors).argmax()
target_log_wh = torch.log(gt_wh / anchors[best])
valid = ~ignore
obj_loss = F.binary_cross_entropy_with_logits(raw[...,4][valid], obj_target[valid])
```

執行 https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.2.0/notebooks/09-anchors.ipynb 或在 repo 根目錄執行 `PYTHONPATH=. python lesson_cases/09-anchors.py`。應看到尺寸 IoU `[1,.25]`、best=0、log wh `[0,0]`、計數 `1/1/30`，再透過 encode/decode、ignore 梯度0、非正槽框梯度0及一次真實 SGD 更新。這是槽位學習實驗，沒有已訓練偵測AP。

## 收益與代價

好的先驗讓尺寸修正量更接近0，適合形狀集中的資料；多槽也增加每格容量。代價是先驗數、尺寸、assignment和ignore都需要設定，候選數增加也提高後處理成本。相同 cell 的兩個物件若都選同一 best anchor，仍會衝突；anchor 不是保證任何兩物件都能分開。exp 還可能產生很大框，訓練需監測log尺寸與有限值。

常見錯誤：把 anchor0 固定為紅類；以位置 IoU 代替尺寸聚類 IoU；exp 前加 sigmoid 使大尺寸比值受限；先驗單位不一致。自主練習：32×16物件配16×16anchor，tw/th是多少？答案：`[ln2,0]`。若換8×8anchor則`[ln4,ln2]`。下一節只改先驗選法，研究尺寸聚類。



<!-- curriculum-evidence:start -->

## 本輪實際執行紀錄

本節範例已於 2026-10-02 使用 PyTorch 2.9.1+cpu 在 CPU 執行，程式中的斷言全部通過。以下是該次輸出；人工輸入、短步更新與模型效果的意義仍依本頁說明區分。[完整紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/09-anchors.json)

??? example "展開本次實際輸出"

    ```text
    size IoU [1.0, 0.25] best anchor 0 log wh [0.0, 0.0]
    ratio offsets [0.0, 0.25] encoded logits [-9.210240364074707, -1.0986123085021973]
    positive/ignore/negative 1 1 30
    decoded [8.00160026550293, 12.0, 24.00160026550293, 28.0] one slot update completed
    ```

<!-- curriculum-evidence:end -->

# Grid MiniYOLO：獨立資料與評估證據

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.2.0/notebooks/07-heldout.ipynb){ .md-button }

前置：[完整推論](07-inference.md)、[AP50](06-evaluation.md)。本節要區分「評估程式跑通」和「模型真的在未見圖片上偵測成功」。一個有限的 AP 數字，可以來自隨機模型或過短訓練；要報效果，還需儲存資料切分、儲存的模型權重、協議與圖板。

本節沿用本書 grid 教學模型，評估採 IoU=.5 的按類別 AP。AP 定義可參照 [Pascal VOC evaluation](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit)；此處使用 all-points interpolated AP，不是 VOC2007 的 11 點近似，也不是 COCO 多 IoU 平均。版本歷史不改變本節協議。

本書配對先排除已用GT，再找最高IoU；官方VOC則先對全部GT找最高IoU，再判它是否已用，重疊GT時可能得到不同結果。[人工AP章](06-evaluation.md)列出差異。以下數字使用本書的簡化評估器，不是官方VOC／COCO benchmark成績。

## 人工失敗例先校準評估器

GT（ground truth）是標註真值；TP是與真值正確配對的預測，FP是未正確配對的誤報，FN是未被配對的真值，即漏檢。precision=`TP/(TP+FP)`，表示保留預測中有多少正確；recall=`TP/(TP+FN)`，表示真實物件中找到多少。PR是precision–recall曲線，AP（average precision）是插值後PR曲線面積；mAP是各類AP的平均，本頁只平均有GT的類別。

框以pixel `[x1,y1,x2,y2]`表示。兩張圖片各有一個紅類 GT：A 為 `[8,12,24,28]`，B 為 `[40,36,56,52]`。人工 predictions 在 A 放三個框，B 完全漏掉。這份 fixture 刻意直接傳給 evaluator，保留重複框以檢查一次 matching；它不是 NMS 後的模型結果。

| score 排序 | 位置與判定 | 累計 precision | 累計 recall |
| --- | --- | --- | --- |
| .9 | A 正確框，TP | 1 | .5 |
| .8 | A 背景框，FP | .5 | .5 |
| .7 | A 同一 GT 的重複框，FP | 1/3 | .5 |

B 的 GT 沒配到任何框，是 FN。GT 只能在同張圖、同類別被成功配對一次；A 的框不能拿去配 B。曲線只達 recall=.5。all-points插值在每個recall位置取往更高recall看去的最高precision：本例recall 0到.5的高度為1，.5到1為0，故面積`1×.5+0×.5=.5`，即AP50=.5。後兩個FP降低最後precision，但沒有降低這份人工例子的插值面積；不是最後 precision 1/3，也不是 `precision×recall`。第二類沒有 GT，AP 回傳 None，本節 mAP 只平均有 GT 的類別，因而仍是 .5。不要把沒有樣本的類說成 AP=0 或 AP=1。

## 接上真正的獨立圖片管線

案例建立 seed=7 的 4 張 train 與 seed=901 的 4 張 held-out，每張一個物件，固定 64×64。只用 train 做 3 次 Adam 更新，再以 `eval()`、`inference_mode()` 在 held-out 推論。評估候選截斷 score=.01、NMS IoU=.5、matching IoU=.5；score門檻先刪低分候選；NMS（非極大值抑制）在同圖、同類的預測框之間比較IoU，刪掉低分重複框；matching則在同圖、同類的預測框與GT之間比較IoU，達門檻就判TP並標記GT已配對。NMS是prediction對prediction，評估matching是prediction對GT，即使門檻同為.5也不是同一程式。低候選門檻的用途是保留 PR 排序範圍，不代表所有框都應顯示給使用者。

```python
pred = decode_grid(model(heldout_images), image_size=64,
                   score_threshold=.01, nms_iou=.5)
metrics = evaluate_ap(pred, heldout_targets,
                      num_classes=2, iou_threshold=.5)
```

執行 https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.2.0/notebooks/07-heldout.ipynb 或在 repo 根目錄執行 `PYTHONPATH=. python lesson_cases/07-heldout.py`。第一行人工 fixture 應核對 AP=.5、precision=1/3、recall=.5；第二部分列出本次三步模型實測的 pipeline smoke 數字。這些數字是程式執行結果，不能當作從零訓練已成功的效能證據。沒有預先填入模型 AP，也不把人工框 .5 稱為訓練成果。

## 完成第 7 章還需要什麼

三步只能驗證可跑，完整里程碑仍需要足夠訓練，並在沒參與訓練的圖片上觀察正確框、分類、漏檢和背景誤報。正式紀錄至少固定資料來源／seed／split、model 與 checkpoint、輸入尺寸、候選 threshold、NMS IoU、matching IoU、每類 AP、時間與成功／失敗疊圖。合成圖的獨立 seed 只能支援同一生成分佈的結論，不能外推到照片。

獨立切分的收益是排除記憶訓練圖片造成的錯覺；代價是可用訓練資料減少，小測試集的結果波動也大。若用 held-out 結果反覆選參數，它已成為 validation；最終 test 必須另留，不能每次都拿來改設定。真實影片更需按來源分組，而不是隻隨機切圖片。

常見錯誤：每張圖各算 AP 再平均；只保留顯示 threshold 以上的框卻與低門檻版本比較；對重複框重複配 GT；把無 GT 類別塞 0 進平均。自主練習：若刪掉 .8 與 .7 的 FP，AP50 會是多少？答案：仍是 .5，因 B 仍漏檢；precision 會升為 1，但 recall 仍 .5。畫面更乾淨，未必代表涵蓋能力提升。

## 補充：一次已實測的合成資料短訓練

前面的三步案例只檢查程式。在相同核心上，另外做了 **160 次 CPU 參數更新**：train 32 張（seed 7）、validation 16 張（seed 700）、test 16 張（seed 7000），每張 64×64、batch 8、Adam learning rate .01、模型 width 8。模型從零初始化；這些圖片都是紅／藍矩形，物件中心刻意分在不同格，尚未包含同格衝突或真實照片的複雜背景。

```bash
python -m miniyolo.train --steps 160 --samples 32 --device cpu
```

此命令在 repository 根目錄執行，會存 checkpoint、逐步 loss、圖與完整報告。用網頁的實測圖先核對，不需要先重跑：

![實測160次CPU更新的loss曲線](../assets/diagrams/grid-learning-curve.svg)

| 固定協議下的結果 | 數值 |
| --- | --- |
| 更新前 validation mAP50 | 0.0018 |
| 更新後 validation mAP50 | 0.8036 |
| 最後一次 test mAP50 | 0.7749 |
| test precision／recall | 0.8824／0.7895 |

這裡 mAP50 只平均有真值類別的 all-points interpolated AP，matching IoU=.5；decode候選 score≥.05、同類 NMS IoU=.5。它不是 COCO AP@[.50:.95]。這個固定設定未用 test 修改模型；只有一個 seed 與很小的測試集，不應把小數差異解讀成穩定的效能優勢。

![四張獨立validation圖的真值與實測預測框](../assets/diagrams/grid-learning-predictions.svg)

綠色虛線是真值，橙色實線是模型框；顏色本身是圖中的兩類物件。第一張左上框明顯向上偏，表示分數高也不等於位置完全正確。這四張圖只是圖板，mAP使用全部16張。已有框接近真值、mAP從接近零上升，支援「這條管線能在受控任務學動」；它仍不能回答模型是否認得照片裡的行人，也沒有證明某個現代機制比較好。

原始設定與數值保留在 [實測 JSON](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/grid-learning.json)。本輪重跑只計訓練 loop，在 AMD EPYC 9V74 80-Core Processor、PyTorch 2.9.1+cpu、2 threads 上約 0.74 秒；不包含程式啟動、資料建立、圖或評估。不同機器需自行量測，不能用此時間預估真實資料訓練。

兩條操作路徑的目標不同：本節notebook的三步只驗證held-out評估接通；160步補充才提供合成模型學得結果。想重跑後者可依[三步訓練頁的可選Colab操作](07-training.md)另開cell，不把三步輸出當成160步checkpoint。

FP練習可在main人工fixture的原斷言之後、建立ShapeDataset之前新增：

```python
cleaned = [{k: v[:1].clone() for k, v in predictions[0].items()}, predictions[1]]
clean_metrics = evaluate_ap(cleaned, targets, num_classes=2, iou_threshold=.5)
assert abs(clean_metrics['map']-.5)<1e-6
assert clean_metrics['precision']==1 and clean_metrics['recall']==.5
```

原predictions與主例斷言保留，避免改掉回歸核對。

<!-- curriculum-evidence:start -->

## 本輪實際執行紀錄

本節範例已於 2026-10-02 使用 PyTorch 2.9.1+cpu 在 CPU 執行，程式中的斷言全部通過。以下是該次輸出；人工輸入、短步更新與模型效果的意義仍依本頁說明區分。[完整紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/07-heldout.json)

??? example "展開本次實際輸出"

    ```text
    artificial evaluation fixture {'ap_per_class': {0: 0.5, 1: None}, 'map': 0.5, 'precision': 0.3333333333333333, 'recall': 0.5}
    3-step held-out PIPELINE SMOKE, not trained detector evidence {'ap_per_class': {0: 0.0, 1: 0.0}, 'map': 0.0, 'precision': 0.0, 'recall': 0.0}
    ```

<!-- curriculum-evidence:end -->

# CSP：分一部分通道走較短的路

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.3.0/notebooks/11-csp.ipynb){ .md-button }

前置：[卷積shape](01-small-cnn.md)、[shortcut](03-identity.md)。問題是每個stage（在同一特徵解析度處理特徵的一組網路區段）都讓全部通道經過重複卷積，是否能把部分特徵留在較短路徑，最後再融合？本節不改預測head、資料或loss，單獨實作channel split。

梯度是loss對某個數值的變化率；backward計算這些變化率，讓模型有更新方向。以下文中「梯度資訊」不表示圖片已分成某種指定語義。

歷史機制：[CSPNet](https://arxiv.org/abs/1911.11929) 提出Cross Stage Partial Network，討論跨stage分流與梯度資訊；[YOLOv4](https://arxiv.org/abs/2004.10934) 採CSPDarknet53，並有其他訓練／neck設計。YOLOv5來自另一個工程分支，可對照固定版本[官方v6.0 common.py](https://github.com/ultralytics/yolov5/blob/v6.0/models/common.py)的C3等模組；不能把所有v4/v5差異歸於CSP。本章簡化為8通道的一次切分、兩個小卷積與concat／1×1融合，不重現完整CSPDarknet或C3。起始為相同輸入輸出的區域性block，實驗後保留作學習模組，效能選擇留待固定預算評估。

## 先把兩條路的shape對起來

輸入 `x[B,8,8,8]` 是NCHW：batch、channel、高、寬。`x.chunk(2,dim=1)` 得左／右兩份 `[B,4,8,8]`。左份直接旁路，右份經兩個3×3卷積，stride1、padding1，所以仍 `[B,4,8,8]`。以channel軸concat得到 `[B,8,8,8]`，再經1×1卷積混合兩路。

```python
bypass, transformed = x.chunk(2,dim=1)
processed = branch(transformed)
joined = torch.cat([bypass,processed],dim=1)
y = fuse(joined)
```

![兩路channel切分與參數成本](../assets/diagrams/11-csp.svg)

concat是通道接在一起，不是shortcut的逐值相加。左路雖沒經過兩個3×3，最後仍經學得的fuse，所以整個模組不保證 `y=x`；不能用ResNet的F=0身份對映結論套用這裡。

先預測一個真的數值例子：a有兩通道`[1,2]`，b有兩通道`[10,20]`，兩者shape都是`[1,2,1,1]`。沿channel concat得到`[1,4,1,1]`、值`[1,2,10,20]`；逐值add得到`[1,2,1,1]`、值`[11,22]`。case先執行並核對這個例子。channel只是位置索引，切分不保證某支只含邊緣或只含顏色。

`x.grad`表示loss對輸入各元素的變化率。反傳先經fuse，concat再把對應channel的梯度分回兩路：前四個回旁路，後四個經branch回輸入。因此本例分別列印前／後四channel梯度絕對值總和。兩者有限且非零隻支持路徑連通，不能證明原論文討論的梯度重複程度下降或偵測訓練收益。

## 參數成本可以逐項手算

比較的Full block讓8channel全部經兩個3×3，再用相同8→8的1×1。帶bias卷積參數為 `out×in×k²+out`。

| 模組 | 兩個3×3 | 最後1×1 | 合計 |
| --- | --- | --- | --- |
| Full 8→8 | `2×(8×8×9+8)=1168` | `8×8+8=72` | 1240 |
| CSP 4→4 | `2×(4×4×9+4)=296` | 72 | 368 |

輸入輸出shape相同，但容量與中間計算不同。參數少不能直接推出精度相同，也不代表實際硬體延遲按比例下降；split／concat的記憶體操作亦有成本。這是容量成本比較，不是同參數量的公平精度實驗。

執行 https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.3.0/notebooks/11-csp.ipynb 或在 repo 根目錄執行 `PYTHONPATH=. python lesson_cases/11-csp.py`。兩模型各做forward、平方均值loss、backward及一次SGD；檢查輸出`[2,8,8,8]`，輸入前後4通道梯度均有限且非零，印出兩者絕對值總和；參數梯度也全為有限值，參數為`[1240,368]`。隨機feature上的平方loss沒有偵測含義，不報告AP或「重複梯度下降多少」。

收益是部分通道走較少卷積，在本例顯著減少卷積參數；代價是分流與融合配置、可能降低表達容量。若通道已很少、資料很難，過度切分未必合適。正式實驗應在同一detector位置替換block，固定資料／步數／評估，並同時報參數、時間及品質。

常見錯誤：沿width而非channel切分；兩支空間大小不同仍concat；把concat說成相加；把區域性參數減少當成整版YOLO更快。自主練習：輸入C=10等分後每路5，fuse10→10，CSP參數是多少？答案：兩個5→5 3×3各230、fuse110，共570；空間shape仍不變。

<!-- curriculum-evidence:start -->

## 本輪實際執行紀錄

本節範例已於 2026-10-02 使用 PyTorch 2.9.1+cpu 在 CPU 執行，程式中的斷言全部通過。以下是該次輸出；人工輸入、短步更新與模型效果的意義仍依本頁說明區分。[完整紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/11-csp.json)

??? example "展開本次實際輸出"

    ```text
    concat (1, 4, 1, 1) [1.0, 2.0, 10.0, 20.0] add (1, 2, 1, 1) [11.0, 22.0]
    full input gradient abs sums first4/last4 0.033093 0.029087
    full output (2, 8, 8, 8) finite backward and step
    CSP input gradient abs sums first4/last4 0.255476 0.019409
    CSP output (2, 8, 8, 8) finite backward and step
    parameters full/CSP [1240, 368] capacity differs; no accuracy comparison
    ```

<!-- curriculum-evidence:end -->

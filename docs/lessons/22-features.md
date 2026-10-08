# E.22.4 特徵能不能使用：凍結後，試著讀出新圖的顏色

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.6.1/notebooks/22-features.ipynb){ .md-button }

[上一節](22-distillation.md)確認 student 真的更新、teacher 與 center 各走自己的路。現在要回到取得特徵的用途：**如果不再修改 backbone，其他讀取器能從它的表示判斷沒看過的紅藍矩形嗎？自監督更新有沒有比訓練起點多帶來幫助？**

## 固定兩份 backbone，再開始用顏色答案

材料仍是 32×32 RGB 單矩形，紅為 0、藍為 1；尺寸、位置、亮度的抽樣不按類別分配。train／validation／test 分別為 128／64／64 張，seed 為 101／202／303，每批紅藍各半。

先只用 train images，從隨機權重做 160 步自監督更新，再取 **teacher backbone**。本頁程式自行完成這段短訓練，不依賴是否跑過 22.3。下面並列兩份特徵：

- **SSL**：Self-Supervised Learning（自監督學習），指這次更新後 teacher 的特徵。
- **random**：相同初始化、尚未更新的 backbone，作為比較起點。

若只看到 SSL 答對，仍不知道答案是不是原本就很容易讀出。保留 random，可以判斷這 160 步是否在同一任務、同一讀法下帶來差別。

評分時兩份都讀完整原圖，不抽新的 crop，只取 `forward_features(images)["cls"]`，每張得到 32 維表示。K=16 的投影分佈曾用來訓練，現在不拿它的槽位當顏色答案。

把 backbone **凍結**：不計算它的梯度，不交給 optimizer，權重副本也要在前後保持相同。現在才拿出 labels，讓後面的讀取器借用或學習顏色答案。validation 只報分數、不挑設定，test 不參與更新；本次事先固定溫度、步數、checkpoint 與讀取器設定。

## 最直接的讀法：找一張表示最像的 train 圖

把 128 張 train 圖轉成特徵，附上各圖的紅藍 label，建立 **reference bank（參考庫）**。新圖片作 query（查詢圖），找庫內特徵最接近的一張，再借用它的答案。

這叫 **1-NN（1-nearest-neighbor，一個最近鄰）**，沒有需要訓練的分類頭。相近用 **cosine similarity（餘弦相似度）**：每個向量除以自己的長度，再取內積，方向越相近越接近 1。這裡消除了長度影響，與 attention 中未正規化的 Q／K 內積不同。

![實際test0藍色query與SSL、random兩套特徵各自找到的train參考圖片](../assets/diagrams/22-neighbors.svg)

query 是 test 0。SSL 找到 train 59，random 找到 train 64，兩張參考圖都是藍色。位置和大小不用相同，讀取器借的是顏色 label；這一例顯示兩種特徵都能找到同色圖，還不能判定哪種更好。

```python
train_features = extract_features(backbone, train.images)  # [128,32]
test_features = extract_features(backbone, test.images)    # [64,32]
accuracy, neighbors = nearest_neighbor_accuracy(
    train_features, train.labels, test_features, test.labels
)
```

`neighbors` 記錄各 query 找到的 train 索引；`accuracy` 是 64 張 query 的顏色答對比例。train labels 此刻才附入參考庫，test labels 只計分，沒有回到自監督 loss。test 圖不能放進庫，否則可能直接找到自己。

## 另一種讀法：只讓線性頭學顏色

接一個 `Linear(32,2)`，用 train 特徵與紅藍 labels 做 120 步交叉熵更新，叫 **linear probe（線性探測）**。probe 是簡單的讀取器，探查表示中是否已有可用線索，不替 ViT 新增 attention 或隱藏層，也不重訓 backbone。

不同特徵維度尺度可能不同，所以先用 train 特徵估每維平均與標準差。train、validation、test 都沿用這份統計做標準化：第 d 維由 `f_d` 變成

\[
\frac{f_d-\mu_d}{\max(s_d,0.01)}.
\]

`μ_d`、`s_d` 都從 128 張 train 圖的第 d 維算出，標準差用除以 128 的版本（`unbiased=False`），最小取 0.01，避免除以極小值。這是在**跨圖片**整理各維尺度；與 21.3 的 LN 在**每個 token 內**整理特徵不同。test 不能重新估統計，否則它也參與設定讀取器。

兩個線性頭都由 seed 900 初始化，用 Adam、學習率 0.02、全 train batch、固定 120 步。兩套 backbone 各用自己的 train 統計，但其餘評分與訓練條件相同。只有線性頭更新，這段是有標籤的下游訓練；無標籤的是前面取得 backbone 的階段。

## 兩種讀法都成功，但沒有分出 SSL 的收益

兩份 backbone 使用相同 TinyViT 結構、同一資料切分與 1-NN 規則；線性頭也使用同樣起點與更新設定。主要差別是 backbone 是否做過這 160 步自監督更新。

|特徵與讀法|validation 顏色答對|test 顏色答對|
|---|---|---|
|random + 1-NN|64／64|64／64|
|SSL + 1-NN|64／64|64／64|
|random + linear probe|64／64|64／64|
|SSL + linear probe|64／64|64／64|

保存的 CPU 結果顯示，兩種讀取器都能使用兩套特徵，解答這兩批色彩題；**沒有顯示 SSL 比 random 更好**。紅藍線索在像素中已經很直接，隨機小網路也保留了足夠資訊，這組評測沒有分出額外收益。

如果給 SSL 更多 probe 步數，或讓 random 用另一種較弱的讀取器，就會改變比較問題。本表保持這些條件相同，但仍只是一個 seed 的合成顏色任務，沒有自然照片或其他任務的優劣結論。

## Loss、特徵變化與答題結果，不能互相代替

![來自實際JSON的160步自監督loss曲線，以及凍結teacher與random特徵的統計和下游比較](../assets/diagrams/22-features.svg)

上半部是 [22.3 那次執行](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/22-distillation.json) 的 160 步跨 view loss；下半部是本頁另一次同設定重訓後的 test 特徵與評分。它們不是同一次執行的完整前後配對。

曲線不一路下降：teacher 與 center 同時改變，本次學習目標不是固定的紅藍答案。因此曲線顯示優化過程，不能只以頭尾或是否單調下降判定用途。

下半部的 **feature std** 跨 64 張 test 圖，分別算每個 CLS 維度的標準差，再平均 32 維；**mean pair cosine** 是不同 test 圖兩兩正規化後的平均相似度，不包含與自己比較。random 的 std／cosine 約 0.084／0.992，SSL 約 0.330／0.856。

SSL 特徵在這些統計上更分散，但「分散」不是顏色讀取的唯一條件；random 的向量比較相近，也已足夠解題。反過來，不為零的 std 只能排除「全部特徵完全相同」這個極端，不能代替上面的獨立任務評分。

執行 `PYTHONPATH=. python lesson_cases/22-features.py` 或頁首 notebook，可查兩份 backbone 的 frozen 檢查、1-NN 找到的索引及線性頭設定。完整輸出在[實際紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/22-features.json)。本頁結果只涵蓋這套矩形規則，不能算作重現原版 DINO 的自然影像表現。

## 把新題放進參考庫，會高估什麼

如果把 test 特徵與 test labels 一起加進 1-NN 庫，其他不變，高正確率還能說明辨認了沒看過的圖片嗎？

??? note "參考答案"

    不能。query 可能找到相同的自己，直接借用自己的真值；即使特徵對新題不好用，也能得高分。庫應只放 train，test 只做 query 與最後計分。

我們已把「取得特徵」與「檢查用途」分開。接下來用 [23.1](23-dino-versions.md)看看官方後續版本如何兼顧整圖與局部表示，再用 [23.2](23-detection-bridge.md)把自己的 patch 特徵交給單物件定位頭。

??? note "原論文的評測範圍"

    [DINO §3.1〈Network architecture〉](https://arxiv.org/abs/2104.14294) 取 backbone 輸出作下游特徵，投影頭服務自監督目標；§3.2 用凍結特徵的線性分類與 k-NN 評測。本書用 32 維 CLS 與簡化 1-NN，資料量、k 值與協議沒有重現原論文，不能直接比數字。

<!-- curriculum-evidence:start -->

## 實際執行紀錄

本節的完整程式於 2026-10-08 在 AMD EPYC 9V74 80-Core Processor（2 個執行緒）上用 PyTorch 2.9.1+cpu 執行，程式裡的 assert 全部通過。下面是那次印出的原始輸出；輸出裡若有計時或訓練得到的數字，換一台電腦會略有不同。每個數字的意思，以本頁正文的說明為準。[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/22-features.json)

??? example "展開本次實際輸出"

    ```text
    {
      "ssl_steps": 160,
      "probe_steps": 120,
      "ssl_seed": 7,
      "split": {
        "train": {
          "samples": 128,
          "seed": 101
        },
        "val": {
          "samples": 64,
          "seed": 202
        },
        "test": {
          "samples": 64,
          "seed": 303
        }
      },
      "task": "red vs blue rectangle color, both classes share position/size generation",
      "feature_source": "frozen teacher CLS [N,32]; no projection head in downstream",
      "ssl_first_loss": 1.9427120685577393,
      "ssl_last_loss": 1.8742694854736328,
      "controls": "same data, TinyViT initial backbone, probe head seed900, optimizer/lr/120steps; train-only feature standardization",
      "validation_use": "report only; no hyperparameter selection",
      "test_use": "evaluation only",
      "models": {
        "ssl_teacher": {
          "nearest_neighbor": {
            "val": {
              "accuracy": 1.0,
              "correct": 64,
              "count": 64,
              "first_query_neighbor_train_index": 64,
              "first_query_class": "blue",
              "first_neighbor_class": "blue"
            },
            "test": {
              "accuracy": 1.0,
              "correct": 64,
              "count": 64,
              "first_query_neighbor_train_index": 59,
              "first_query_class": "blue",
              "first_neighbor_class": "blue"
            }
          },
          "linear_probe": {
            "train_accuracy": 1.0,
            "val_accuracy": 1.0,
            "test_accuracy": 1.0,
            "steps": 120,
            "head_seed": 900,
            "final_loss": 0.0001775920100044459
          },
          "diagnostics_test": {
            "feature_std": 0.32950639724731445,
            "normalized_feature_std": 0.057701025158166885,
            "mean_pair_cosine": 0.8559923768043518,
            "mean_output_entropy": 1.7631640434265137,
            "marginal_output_entropy": 2.2008631229400635
          },
          "backbone_unchanged": true,
          "backbone_has_no_grad": true
        },
        "random_baseline": {
          "nearest_neighbor": {
            "val": {
              "accuracy": 1.0,
              "correct": 64,
              "count": 64,
              "first_query_neighbor_train_index": 100,
              "first_query_class": "blue",
              "first_neighbor_class": "blue"
            },
            "test": {
              "accuracy": 1.0,
              "correct": 64,
              "count": 64,
              "first_query_neighbor_train_index": 64,
              "first_query_class": "blue",
              "first_neighbor_class": "blue"
            }
          },
          "linear_probe": {
            "train_accuracy": 1.0,
            "val_accuracy": 1.0,
            "test_accuracy": 1.0,
            "steps": 120,
            "head_seed": 900,
            "final_loss": 0.0025564345996826887
          },
          "diagnostics_test": {
            "feature_std": 0.08420964330434799,
            "normalized_feature_std": 0.014960691332817078,
            "mean_pair_cosine": 0.9917242527008057
          },
          "backbone_unchanged": true,
          "backbone_has_no_grad": true
        }
      },
      "elapsed_seconds": 2.8715969000040786,
      "limitation": "a simple color task may already be solved by random features; equal accuracy is not evidence of SSL improvement or natural-image transfer"
    }
    ```

<!-- curriculum-evidence:end -->

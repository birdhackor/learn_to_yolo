# 14 YOLO11 特徵模組：拆路徑、保留中間成果、再融合

[開啟 Colab](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.2.0/notebooks/14-feature-module.ipynb) · 原始碼：`lesson_cases/14-feature-module.py`

前置是[residual捷徑](03-identity.md)、[CSP](11-csp.md)與channel concatenation（沿通道串接）。residual保留原值再加轉換結果；本例bottleneck是兩層窄卷積的殘差小塊，hidden是每條內部分支的channel數。兩個3×3卷積可以轉換特徵，但所有訊號都走同樣深度。能否讓一部分走短路徑，另一部分經較深轉換，並把中間成果一起交給最後投影？本節由YOLO11官方配置中的C3k2切入，只研究其中可看懂的split–transform–concatenate機制。

歷史機制是YOLO11配置採C3k2等模組，整體還含其他backbone、neck與attention設定，版本收益不能全部歸因於C3k2。本次起點為8-channel特徵，本章用C3k2／C2f啟發的小模組：1×1投影後分兩路、兩個殘差bottleneck依次處理一條路、串接所有中間輸出，再1×1融合。省略官方Conv的BN／activation配置及可選C3k內部結構，名稱為`SplitAggregate`，不冒充完整C3k2。

![a、b、b1、b2保留後串接的路徑](../assets/diagrams/14-split-paths.svg)

## 用channel帳本理解路徑

輸入shape`[B,C,H,W]=[2,8,8,8]`，先以1×1卷積`8→8`投影；將channel切成a、b，每份`[2,4,8,8]`。a直接保留，b經第一個bottleneck得b1，再由b1經第二個得b2。每個bottleneck使用兩個3×3卷積與`x+F(x)`，shape保持不變。

| 保留的特徵 | 深度 | channel數 |
| --- | --- | --- |
| a | 投影後直接留下 | 4 |
| b | 投影後直接留下 | 4 |
| b1 | 一個bottleneck | 4 |
| b2 | 兩個bottleneck | 4 |

沿channel串接為`[2,16,8,8]`，不是把H或W擴大。最後1×1卷積`16→8`，輸出重新成`[2,8,8,8]`，因此可以放回原有介面。1×1融合是學得的跨channel組合，並非直接平均四條路。

```python
a, b = self.project(x).chunk(2, dim=1)
paths = [a, b]
for block in self.blocks:
    paths.append(block(paths[-1]))
return self.fuse(torch.cat(paths, dim=1))
```

短路徑讓後端能直接讀取較少轉換的特徵，長路徑則提供較大的局部感受範圍。只有某段輸出既未串接、也未供後續計算使用時，該段才完全收不到loss梯度。b1即使不直接交給concat，仍會經b2收到梯度，所以「第一個bottleneck有梯度」不能單獨證明保留中間成果的路徑。

案例在訓練後另做一次檢查forward：對a、b、b1、b2與concat tensor呼叫`retain_grad()`，核對每份特徵與對應concat槽的值及順序。backward後，逐槽檢查concat的直接梯度，也檢查每份特徵的總梯度均非零。b1的總梯度同時包含直接融合路徑和經b2的間接路徑；這兩項不能混稱成同一個檢查。

Conv2d本例含bias，參數為`out×in×kernel高×kernel寬+out`；4→4、3×3就是144+4=148。進階梯度診斷可稍後讀：retain_grad讓中間tensor也保留反傳梯度；輸出中的gradient L1是梯度絕對值加總，用來確認非零，不是新增L1訓練loss。

## 本次到底測什麼

固定seed7生成一組特徵，target是向右迴圈移動一格的`x.roll(1,-1)`。這是可控的區域性轉換任務，用來驗證模組可以訓練；迴圈邊界與零padding並不完全相容，所以沒有要求loss為零。本例對`SplitAggregate`做30次MSE backward和SGD step，確認最後loss比初值小。沒有真實圖片、GT框或AP，因此它不能證明YOLO11更準。

同時建立兩層`8→8`的plain3×3模組，但只統計其參數，不拿未訓練plain與訓練split比較loss。plain為1,168參數；split為800：投影72，四個`4→4`3×3卷積共592，融合136。計算減少來自此例hidden=4的窄bottleneck，以及1×1的使用，不能推廣成所有split模組都更省。

執行`PYTHONPATH=. python lesson_cases/14-feature-module.py`，應看到輸入輸出同為`(2,8,8,8)`、四份4channel串接後16→8、參數1168／800、MSE下降，以及每個concat槽直接梯度與各特徵總梯度的檢查通過。這是CPU小張量實驗，所需記憶體很小；實際detect latency需另外測。

## 收益和代價要一起記錄

保留不同深度特徵讓融合層選擇適合任務的訊號，窄bottleneck也可降低部分卷積成本。代價是中間activation要保留至concat，讀寫與concat暫存可能增加記憶體；深支與shortcut、hidden寬度、block數也變成設計選項。參數少不一定延遲短，因為不同裝置對小卷積與資料搬移的效率不同。

若想在MiniYOLO正式替換，先保持輸入輸出shape、相同資料切分、訓練步數和decode；只改這個模組。再記錄AP50、參數、端到端時間與失敗圖。完整YOLO11配置還有多項變動，這個區域性替換的結果只能解釋本模組，不能代表整版收益。

常見錯誤是把chunk誤當每兩個channel交錯分組、concat接在空間軸、融合卷積inputchannel仍寫8，或給跨stage殘差兩支不同shape。官方`C3k2`可切換`c3k`路徑，本例沒有那個開關；不要把所有bottleneck實現都稱為完全相同的模組。

自主練習：只將`main()`的模型建立改成`module = SplitAggregate(blocks=3)`，hidden維持4。答案為五份4channel、concat`(2+3)×4=20`，fuse20→8。建構式已用`(2+blocks)×hidden`自動設定fuse，印出尺寸與梯度檢查也隨實際block數改變，不需手改輸出字串。參數為投影72＋6個3×3卷積888＋融合168＝1,128。它已接近plain參數，而且保留了更多中間activation；「拆路徑就更省」不是可用的普遍結論。

來源查覈：2026-10-02。[YOLO11官方配置](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/cfg/models/11/yolo11.yaml)、[C3k2／C2f／Bottleneck定義](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/nn/modules/block.py)。



<!-- curriculum-evidence:start -->

## 本輪實際執行紀錄

本節範例已於 2026-10-02 使用 PyTorch 2.9.1+cpu 在 CPU 執行，程式中的斷言全部通過。以下是該次輸出；人工輸入、短步更新與模型效果的意義仍依本頁說明區分。[完整紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/14-feature-module.json)

??? example "展開本次實際輸出"

    ```text
    input/output: (2, 8, 8, 8) (2, 8, 8, 8)
    concatenated channels: 4 + 4 + 4 + 4 = 16; fuse 16 -> 8
    parameters plain / split: 1168 800
    local transformation MSE 1.0722 -> 1.0049
    concat order, each direct concat-slot gradient, and each path total gradient: verified
    direct concat-slot gradient L1: [0.2967, 0.3349, 0.3481, 0.2779]
    ```

<!-- curriculum-evidence:end -->

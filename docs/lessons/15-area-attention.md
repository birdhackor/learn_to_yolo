# 15.2 YOLOv12 Area Attention：互動範圍是一筆預算

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.6.1/notebooks/15-area-attention.ipynb){ .md-button }

上一節讓每個位置用 QK 選來源，再加權讀 V。一層就能直接讀遠處，但 full attention 要為 N 個位置算 N² 次配對。希望保留較廣的內容互動，又不在每層付出全圖配對成本時，可以先限制讀取範圍：位置分區，同區仍互相選讀，跨區的配對不算。

YOLOv12 稱這個做法 Area Attention（區域注意力）。省掉配對也意味著少了一條資訊路徑。我們沿用上一節的 QKV 流程，只把 4 個位置擴成 16 個，讓區域邊界看得見，再只改遠處的一個 token，看它還能不能影響左上。

## 先確定 token0 的來源名單

固定一張 4×4 特徵圖，照 row-major 逐列由左到右排成 token0～15，shape `[B,N,C]=[1,16,2]`。第 i 個 token 的兩個 channel 都是 i/16，例如 token0=`[0,0]`、token5=`[5/16,5/16]`。QKV 投影仍以三份單位矩陣起步，Q=K=V=token，方便手算。

full attention 讓 token0 讀全部 0～15。area 則把攤平後的序列切成 A 個等長連續段，每段只在內部 attention；A 是區域數，即程式的 `areas`。A=4 時每段四個，token0 只讀 0～3：

![4×4 逐列編號：四條帶與兩條帶的對照](../assets/diagrams/15-area-layout.svg)

底色相同表示同區；紅框是接收者 token0，紫框 15 是主例改動的來源，紫框 3 留給練習。左圖 A=4 是四條橫帶；右圖 A=2 時每帶兩列、八個 token。這個圖讓你先辨認「讀得到誰」，再解讀後面的平均值與干預。

四區不是四個 2×2 方塊。依序切段沒有先改排列，所以每四個就是一整列；要切 window（方塊內 attention），得先讓 `[0,1,4,5]`、`[2,3,6,7]` 等同方塊 token 相鄰，算完還要排回原位。兩種區域切法限制的是不同來源。

## 一張 16×16 表，換成四張 4×4 表

分區沒有換 QKV 參數。full 和 area 都呼叫同一份 `projection`，只把 `areas` 從 1 改成 4。這樣輸出差異才由讀取範圍造成，不混入兩次隨機初始化的差異。

Q、K、V 各從 `[1,16,2]` reshape 成 `[B×A,N/A,C]=[4,4,2]`。第 0 軸現在是四份區域：0～3、4～7、8～11、12～15。矩陣乘法只在同一份內配對，得到 `[4,4,4]`：第幾區、區內接收位置、區內來源位置。token0 的表沒有 token15 的欄，softmax 也只在 0～3 分配。

下面依完整 `attend` 函式簡化：B、N、C 對應原程式 b、n、c，省略開頭 assert，末行先存 output；原函式回傳 output 和 weights。

```python
# B, N, C 是 tokens 三個軸的長度，本例 [1,16,2]（完整程式寫成 b, n, c = tokens.shape）
# projection 是 nn.Linear(2, 6, bias=False)：把每個 token 從 2 維變成 6 維，再沿最後一軸切成三等份
q, k, v = projection(tokens).chunk(3, dim=-1)  # Q、K、V 各 [1,16,2]
# 方括號裡的 for 對 q、k、v 各做一次同樣的 reshape；N // areas = 4 是每區的 token 數
q, k, v = [t.reshape(B * areas, N // areas, C) for t in (q, k, v)]  # 各 [1,16,2] -> [4,4,2]
# weights 是 attention 權重表：[4,4,2] @ [4,2,4] -> [4,4,4]，4 份各自相乘
# 分數除以 √C=√2，再沿最後一軸（來源）做 softmax
weights = (q @ k.transpose(-2, -1) / math.sqrt(C)).softmax(-1)
output = (weights @ v).reshape(B, N, C)  # [4,4,4] @ [4,4,2] -> [4,4,2]，再依序接回 [1,16,2]
```

每區各完成 QKᵀ/√2→來源軸 softmax→加權 V，再依原順序接回 `[1,16,2]`。這裡 d=C=2，所以縮放寫成 √C；不是任意多 head 都應使用 √C。`N/A` 是每區位置數，A 必須大於 0，N 必須能被 A 整除；原函式會檢查這兩項。

現在 full 的 affinity（讀取表）是 `[1,16,16]`、256 個數；area 的是 `[4,4,4]`、64 個數。少的是當次 forward 的配對結果，QKV 的 6×2=12 個可學參數一個也沒少。這四張小表的代價，接著用一次輸入改動顯示。

## 只改 token15，左上還會跟著變嗎？

token0 的 q0=`[0,0]`，和每個 key 的內積都是 0。softmax 因此均勻分配：full 每來源 1/16，area 每來源 1/4。token0 輸出就是可讀 V 的平均，兩個 channel 同值：

| 路徑 | 可讀來源 | token0 原輸出 |
| --- | --- | --- |
| full | 0～15 | `7.5/16=0.46875` |
| area，A=4 | 0～3 | `1.5/16=0.09375` |

接著只把 token15 的兩個 channel 各加 5，其他都不動。這是干預：故意改一項輸入，觀察哪個輸出受它影響。

full 中 token15 的 K、V 都改了，但 q0 是 0，QK 分數仍全為 0，所以比例仍各 1/16。token0 多讀到 `5/16=0.3125`，變成 0.78125。area 的來源名單沒有 token15，它的 K、V 都不參與 token0 的計算，輸出仍是 0.09375。

q0=0 讓數值能用平均手算，並不是跨區隔離的原因。換成其他 query，area 中仍沒有 token15 的配對與 V 路徑。這個關係可直接預測：改同區來源才有影響左上的路，改別區來源沒有。

完整程式以 `torch.allclose` 比較干預前後的 token0，得到 `full_affected`、`area_affected`。`query_index=0`、`changed_index=15`，下面保留了同區判定與斷言；省略兩行就是計算這兩個 affected 的比較：

``` { .python data-excerpt="lesson_cases/15-area-attention.py" }
area_size = tokens.shape[1] // areas  # 每區的 token 數：16 // 4 = 4
same_area = query_index // area_size == changed_index // area_size  # 0 // 4 = 0，15 // 4 = 3：不同區，False
...
assert full_affected and area_affected == same_area  # full 要受影響；area 受不受影響，要和「是否同區」一致
```

`index//area_size` 是區號。本例 token0 在區 0，token15 在區 3，所以 `same_area=False`，結果為 full=True、area=False。練習改區數或改動位置時，斷言會依相同規則自動更新，不需刪掉檢查。

## 少算 1/A 配對，整層卻未必快 1/A

一般每區 N/A 個接收者，各配 N/A 個來源，A 區共

\[
A(N/A)^2=N^2/A
\]

個 pair（接收、來源的配對）。A=4 使本例 256→64，留下四分之一；有多 head 和 batch 時，讀取表元素數還要乘 head 數 M 和圖片數 B。QKV 投影仍處理全部 N 個 token，輸出也要重排，所以這個比例不代表整層或整模型延遲變成四分之一。

同樣維持 A，圖的邊長兩倍使 N 四倍，N²/A 仍十六倍。分區調整互動範圍這筆預算，沒有把任意解析度的成本變線性。GPU 的實際時間還受 FlashAttention、dtype、資料排列和程式實作影響；本節只數 pair 和存下的表，不作 GPU benchmark。

讀取範圍也不等於卷積的局部鄰域。以左上 token0 為例：

| 一層操作 | 真正能直接讀的 token |
| --- | --- |
| 3×3 卷積，padding 1 | 0、1、4、5 |
| area，A=4 | 0、1、2、3 |
| full | 0～15 |

area 可直接選讀同列遠處 token3，卻少掉正下方 token4；3×3 則相反。卷積在結構裡先指定局部鄰域、跨位置共用濾鏡；attention 用內容算比例。疊卷積能擴大感受野，不同分區也會改可讀範圍。這些是資訊路徑的比較，本例沒有訓練 CNN，也沒有支持誰的偵測較準。

## 單層分區的限制，怎麼連回完整 YOLOv12？

我們故意拿掉其他跨位置路徑，讓干預只反映分區。官方 `AAttn` 還有對整圖 V 做的位置卷積，會把相鄰列混到輸出；投影後的 BN 在訓練模式用整批、所有位置的統計，別區改動也可能改這區。若把它們留下，token0 改不改就不能只歸因於同區與否。

完整網路的後續卷積、下採樣也會跨相鄰帶；官方 stride 32 那層還使用 area=1，也就是 full attention。所以本例的「token15 讀不到 token0」只指這一層被隔離的 area 計算，不能說整個 YOLOv12 永遠跨不了區。反過來，若每層都保持同樣分帶、又沒有任何跨帶路徑，單靠 area attention 就傳不過去。

官方論文描述 H/A×W 的橫帶或 H×W/A 的直帶，預設 A=4；固定程式版本把攤平序列依序切段。本例 H=4、W=4 恰好是橫帶。若 H=6、W=4、A=4，每區六個 token，第一區會是首列四格加次列前兩格，不再是一整條長方形。要畫區域時先查順序和 H、W，不能只看到 A=4 就畫四象限。

執行 `PYTHONPATH=. python lesson_cases/15-area-attention.py`，對照表大小 256／64、原 token0 輸出 0.46875／0.09375、干預後 0.78125／0.09375。程式最後另用 area 輸出對原 tokens 算 MSE，backward 和 SGD 一次，確認 QKV 權重連到 loss；它沒有訓練偵測器，沒有遠距或小物件 AP 結論。

操作時把區域與 head 分清：area 切位置，attention head 切 channel；只加 head、不分區，每組仍有 N² 配對。接回時用原 B，不是 B×A，否則元素數對不上。若要評估偵測收益，還需對齊資料、訓練和計算預算，再看失敗類型與品質，不能以這次表變小代替。

## 選讀：官方部件與其他分區成本

??? note "YOLOv12 為什麼切成長條，不切方塊？"

    論文的理由是速度：不必另外切 window，只要一次 reshape，所以比較快。切成 4 段時，每個位置在這一層能讀的範圍縮成原來的 1/4，論文認為仍然夠大。

??? note "選讀：官方 AAttn 的投影、head 與位置卷積"

    - **多 head（multi-head）**：這裡的 head 是 attention head，不是 [12.2](12-decoupled-head.md) 的偵測 head（把特徵轉成預測的輸出模組）。多 head 把 Q、K、V 的 C 個 channel 分成 M 組，每組各算一張權重表、各自加權，最後把各組的結果接回 C 個 channel。官方 `AAttn` 用 `num_heads` 表示組數，`head_dim` 表示每組的 channel 數。本例只有 1 組、2 個 channel（相當於 `num_heads=1`、`head_dim=C=2`），所以程式沒寫這兩個變數。area 分的是位置，head 分的是 channel，兩者切的是不同的軸。
    - **位置卷積**：官方對整張 V 特徵圖（還沒攤平、也還沒分區的 V）做的逐 channel 卷積，每個 channel 各用自己的卷積核（論文寫 7×7，頁尾連結的程式版本是 5×5）。它不分區，結果加到 attention 輸出上，用來補位置資訊。
    - **輸出投影**：attention 之後再接的一層線性轉換（官方用 1×1 卷積，後面同樣接 BN）。
    - **Q／K、V 投影後的 BN**：官方用兩個 1×1 卷積（`self.qk`、`self.v`）分別算出 Q／K 與 V，每個卷積後接 BN（BatchNorm，批次正規化），不接激勵函數；本節改用一個不帶 BN 的 `nn.Linear` 一次算出 Q、K、V。BN 在訓練模式下會用整批資料（所有位置）的平均與變異數，改動別區的 token 也會影響這一區的輸出，所以本節一併拿掉。

??? note "多 head 延伸（可跳讀）"

    若有 M 個 head，每個 head 的 channel 數 `D=C/M`，Q／K／V 可以寫成 `[B×A,M,N/A,D]`，attention 權重表是 `[B×A,M,N/A,N/A]`，縮放因子是 `√D`。本例 M=1，所以 D=C/1=C，縮放因子 √D 就是程式裡的 √C；大小為 1 的 M 軸省略不寫，所以權重表只有三維 `[B×A,N/A,N/A]`。

??? note "如果改成固定每區的 token 數呢？"

    假設每區固定 S 個 token，讓區域數跟著 N 變：A=N/S。pair 數變成 `A×S²=(N/S)×S²=N×S`，只和 N 成正比。代價是每區只占全圖的 S/N：圖越大，每個位置在這一層能讀到的比例越小。本節和官方程式都是固定 A（area 數在建立模型時就決定，不隨輸入大小改變），所以邊長變兩倍時，pair 數仍變 16 倍。

## 自主練習

自主練習：在完整程式（Colab 裡「本節可修改的完整實驗」下方那格，或本機的 `lesson_cases/15-area-attention.py`）的 `main()` 裡，每題都從原始程式（`areas = 4`、`changed_index = 15`）出發，只改一行。先預測兩個 token 是否同區、token0 輸出是多少，再執行核對。程式裡干預前後的兩次 area 計算、pair 數的 assert 和印出的文字，都讀同一個 `areas` 變數；`changed_index` 也只在 `main()` 開頭設定一次。所以不用改任何 assert，也不要刪掉原本的檢查來讓程式通過。

1. 把 `areas = 4` 改成 2，保持 N=16。每區幾個 token？pair 數多少？token0 讀得到哪些 token？token0 輸出是多少？
2. 把 `areas` 改成 1。pair 數和 token0 輸出是多少？改 token15 會不會影響 area 的 token0？
3. 先把 `areas` 改回原本的 4，再把 `changed_index = 15` 改成 3。干預後，full 和 area 的 token0 輸出各是多少？

??? note "參考答案"

    **第 1 題**：兩區各 8 個 token，pair 數 `2×8²=128`（權重表 `(2, 8, 8)`）。token0 可讀 0 到 7，仍讀不到 15，所以干預 token15 不影響 area 的 token0。token0 輸出是 0 到 7 的平均除以 16：3.5/16=0.21875。

    **第 2 題**：areas=1 就是 full：pair 數 256，token0 輸出 0.46875。這時 token15 和 token0 同區，改 token15 會影響 area 的 token0（干預後也是 0.78125，和 full 相同）。`same_area` 算出 True，`full_affected`、`area_affected` 也都是 True，所以 `assert full_affected and area_affected == same_area` 不用改，照樣通過。

    **第 3 題**：token3 和 token0 在同一條帶，所以改 token3 時，full 和 area 的 token0 輸出都會變。full 從 0.46875 變成 0.78125（多 5/16）；area 從 0.09375 變成 1.34375（多 5/4：每區 4 個 token，attention 權重各 1/4）。

參考來源：[YOLOv12 論文](https://arxiv.org/abs/2502.12524)、[作者 AAttn 的 area reshape、CPU attention 與位置卷積](https://github.com/sunsmarterjie/yolov12/blob/2abab7153a065fb2925e8088e9ca2b19016ab7d6/ultralytics/nn/modules/block.py)（CPU attention 指 `AAttn` 不用 FlashAttention 時，例如在 CPU 上，改用一般矩陣乘法算 attention 的那段程式）。

<!-- curriculum-evidence:start -->

## 實際執行紀錄

本節的完整程式於 2026-10-08 在 AMD EPYC 9V74 80-Core Processor（2 個執行緒）上用 PyTorch 2.9.1+cpu 執行，程式裡的 assert 全部通過。下面是那次印出的原始輸出；輸出裡若有計時或訓練得到的數字，換一台電腦會略有不同。每個數字的意思，以本頁正文的說明為準。[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/15-area-attention.json)

??? example "展開本次實際輸出"

    ```text
    full affinity shape/count: (1, 16, 16) 256
    area affinity shape/count: (4, 4, 4) 64
    first-token output full=0.46875, area=0.09375
    changing token 15 affects token 0: full=True, area=False; same area=False
    after intervention: full=0.78125, area=0.09375
    area-attention backward/step verified; loss=0.0048
    ```

<!-- curriculum-evidence:end -->

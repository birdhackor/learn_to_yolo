# 15.2 YOLOv12 Area Attention：互動範圍是一筆預算

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.4.0/notebooks/15-area-attention.ipynb){ .md-button }

上一節的 full attention 讓每個位置讀全圖，代價是要算的權重數隨 token 數的平方成長：feature map 越大，越貴。YOLOv12 的省法是把 token 分成幾個區域（area），各區只在自己內部做 attention，再接回原圖。這樣會少哪些計算，又失去哪一些互動？

本節除了數權重表有多大，也做一個干預實驗：故意只改一個遠處 token 的值、其他都不動，看第一個 token（token0）的輸出會不會跟著變。讀完你能算出分區省下多少計算，也能用實驗說明：分區之後，這一層有哪些位置彼此讀不到。

前置：[上一節](15-attention-bridge.md)的 attention 分三步：Q 乘 K 的轉置（除以 √d）得到分數、沿來源軸做 softmax、再用這些權重加權 V。其中 d 是每個 query／key 向量的元素個數；本例 d 等於每個 token 的 channel 數 C=2，所以程式寫成 √C。一個接收位置配一個來源位置，叫一個 pair；每個 pair 要算一個權重，N 個 token 共有 N² 個 pair。標題把互動範圍比作一筆預算：本節說的「互動預算」，就是一層要算的 pair 數。

YOLOv12 的 Area Attention 把攤平後的 token 依序切成 A 段，每段是一個 area（A 就是程式裡的 `areas`，也就是區域數），每段只在內部做上一節的 attention。論文作者的官方程式（類別名稱 `AAttn`）還有下面四樣東西，本節先全部拿掉：

- **多 head（multi-head）**：這裡的 head 是 attention head，不是 [12.2](12-decoupled-head.md) 的偵測 head（把特徵轉成預測的輸出模組）。多 head 把 Q、K、V 的 C 個 channel 分成 M 組，每組各算一張權重表、各自加權，最後把各組的結果接回 C 個 channel。官方 `AAttn` 用 `num_heads` 表示組數，`head_dim` 表示每組的 channel 數。本例只有 1 組、2 個 channel（相當於 `num_heads=1`、`head_dim=C=2`），所以程式沒寫這兩個變數。area 分的是位置，head 分的是 channel，兩者切的是不同的軸。
- **位置卷積**：官方對整張 V 特徵圖（還沒攤平、也還沒分區的 V）做的逐 channel 卷積，每個 channel 各用自己的卷積核（論文寫 7×7，頁尾連結的程式版本是 5×5）。它不分區，結果加到 attention 輸出上，用來補位置資訊。
- **輸出投影**：attention 之後再接的一層線性轉換（官方用 1×1 卷積，後面同樣接 BN）。
- **Q／K、V 投影後的 BN**：官方用兩個 1×1 卷積（`self.qk`、`self.v`）分別算出 Q／K 與 V，每個卷積後接 BN（BatchNorm，批次正規化），不接激勵函數；本節改用一個不帶 BN 的 `nn.Linear` 一次算出 Q、K、V。BN 在訓練模式下會用整批資料（所有位置）的平均與變異數，改動別區的 token 也會影響這一區的輸出，所以本節一併拿掉。

本節從 full attention 出發，full 和 area 兩條路共用同一份 QKV 權重，只把 `areas` 從 1 改成 4，所以兩者的差別只來自「誰能讀誰」。拿掉位置卷積和其他路徑，是為了單獨檢驗「分區」這一個改動造成的互動限制。本節也不比較完整的 YOLOv12 和 CNN 誰比較準。

## 區域的形狀由 token 順序決定

![4×4 逐列編號：四條帶與兩條帶的對照](../assets/diagrams/15-area-layout.svg)

看圖時注意：A 是區域數 `areas`；底色相同的格子屬於同一個 area。紅框標出 query，也就是 token0；紫框標出本節會改動的 token：15 用在主實驗，3 用在練習。

本例的 feature map 是 4×4，N=16。照 row-major 順序（逐列由左到右）編號，每四個 token 是一列。tokens 的 shape 是 `[B,N,C]=[1,16,2]`：B 是圖片數，N 是 token 數，C 是每個 token 的 channel 數。`areas=4` 時，Q、K、V 各自 reshape 成 `[B×A,N/A,C]=[4,4,2]`。N/A 是 N 除以 A，也就是每區的 token 數（不是「不適用」的縮寫）。

下面是依完整程式的 `attend` 函式改寫的簡化版：軸長照正文寫成大寫的 B、N、C，省略了開頭的 assert（檢查 A 大於 0、N 能被 A 整除），最後一行也只把結果存進 `output`（`attend` 是把輸出和權重表 `weights` 一起回傳）。

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

為什麼這樣就只在區內算？reshape 照 row-major 順序，把每 4 個連續 token 放成第 0 軸的一份：第 0 份是 token0–3，第 1 份是 token4–7，依此類推，所以四個 area 就是四條水平帶。

`@` 遇到三維張量時，第 0 軸的每一份各自配對，只對最後兩軸做矩陣乘法：`[4,4,2] @ [4,2,4]` 就是做 4 次「4×2 乘 2×4」，得到 4 張 4×4 的分數表。分數除以 √2、再做 softmax，就是 4 張 attention 權重表（上一節叫 affinity；它是每次 forward 算出的中間結果，不是可學參數）。token0 那張表只有 token0–3 這四個來源，沒有 token15；softmax 也只在這四個來源之間分配。權重表 `[4,4,4]` 的三個軸依序是：第幾區、區內第幾個接收位置、區內第幾個來源位置。`weights @ v` 同樣是 4 份各自相乘；最後的 reshape 只是把四段照原來的順序接回 `[1,16,2]`，位置不會亂。

full 和 area 都用同一個 `projection` 呼叫 `attend`：`areas=1` 就是 full，`areas=4` 就是 area。QKV 權重沒有重新初始化，兩條路的差別只有誰能讀誰。共用同一份 QKV 權重來比較，比各跑一份隨機初始化的模型，更能把差異歸因於互動範圍。

這四條帶不是四個 2×2 方塊。把圖切成小方塊、只在方塊內做 attention 的做法叫 window（和第 5 章的滑動視窗 sliding window 不同）。程式沒有先重排 token，所以切出來的是橫帶。若要 2×2 方塊，得先把 token 重新排好，讓同一方塊的 4 個 token 相鄰：[0,1,4,5]、[2,3,6,7]、[8,9,12,13]、[10,11,14,15]，再每 4 個切一段；算完還要照相反的順序排回原位。

??? note "YOLOv12 為什麼切成長條，不切方塊？"

    論文的理由是速度：不必另外切 window，只要一次 reshape，所以比較快。切成 4 段時，每個位置在這一層能讀的範圍縮成原來的 1/4，論文認為仍然夠大。

論文把 feature map 切成 l 段（l 就是本節的 A），每段是 (H/l)×W 的橫帶或 H×(W/l) 的直帶，預設 l=4；本例的 `areas=4` 正是這個預設。官方程式（`AAttn`）的做法則是攤平後依序切段：H 能被 area 數整除時，剛好切出橫帶；不能整除時，切出的段就不再是整列，甚至可能不是長方形。例如 H=6、W=4、A=4 時每區 6 個 token，第 0 區是第 0 列的 4 格，再加上第 1 列的前兩格。所以切出的形狀取決於 H、W 與 area 數，不能只看到 area=4 就畫成四象限。本例特意選 4×4，讓區域邊界很好檢查。

## 計算和記憶體怎麼減少

full 的 attention 權重表 shape 是 `[1,16,16]`，有 256 個數；area 的是 `[4,4,4]`，共 64 個數（程式輸出把兩者都印成 affinity）。少掉的是這些每次 forward 算出的中間結果，不是參數：可學的 QKV 權重兩條路共用同一份，也就是 `nn.Linear(2, 6, bias=False)` 的 6×2=12 個，分區前後一個也沒少。

一般有 A 個同樣大小的區域時，每區 N/A 個接收位置各配 N/A 個來源位置，每區有 (N/A)² 個 pair；共 A 區，所以每個 head 的 pair 數是 `A×(N/A)²=A×N²/A²=N²/A`。有 M 個 head 時，權重表的元素數還要乘上 M 與 batch 數 B。本例單 head、B=1，權重表從 256 個變成 64 個，是原來的 1/4；一般 A 個等大區域就是原來的 1/A。這只是 attention 的 pair 項。QKV 投影仍要處理全部 16 個 token，所以不能說整個模型跑一次的時間（延遲）變成四分之一。

??? note "多 head 延伸（可跳讀）"

    若有 M 個 head，每個 head 的 channel 數 `D=C/M`，Q／K／V 可以寫成 `[B×A,M,N/A,D]`，attention 權重表是 `[B×A,M,N/A,N/A]`，縮放因子是 `√D`。本例 M=1，所以 D=C/1=C，縮放因子 √D 就是程式裡的 √C；大小為 1 的 M 軸省略不寫，所以權重表只有三維 `[B×A,N/A,N/A]`。

若輸入 feature map 的邊長變兩倍，N 變四倍；即使 `areas` 不變，pair 數 N²/A 仍變 16 倍。分區控制的是互動預算（一層要算的 pair 數：full 是 N²，area 是 N²/A），並沒有讓任意解析度都變便宜。

??? note "如果改成固定每區的 token 數呢？"

    假設每區固定 S 個 token，讓區域數跟著 N 變：A=N/S。pair 數變成 `A×S²=(N/S)×S²=N×S`，只和 N 成正比。代價是每區只占全圖的 S/N：圖越大，每個位置在這一層能讀到的比例越小。本節和官方程式都是固定 A（area 數在建立模型時就決定，不隨輸入大小改變），所以邊長變兩倍時，pair 數仍變 16 倍。

實際跑多快，還取決於本節沒有量的幾件事：

- GPU 上 attention 的寫法（例如 FlashAttention 把計算切成小塊，不必存下整張權重表）。
- 資料型別（dtype，例如 float32 或 float16：兩者都是浮點數，差在每個數用幾位元存）。
- 記憶體佈局（資料在記憶體裡怎麼排）。
- GPU 程式的實作（GPU 領域把這種程式叫 kernel，和卷積核是不同的意思）。
- 投影本身的計算量。

所以 CPU 小實驗的 pair 數，不能當成 GPU 上的速度測試（benchmark）。

## 遠處資訊能不能傳過來

本例第 i 個 token 的兩個 channel 都是 i/16（i=0…15），例如 token5=[5/16,5/16]。QKV 權重的初值設成 identity（輸入原樣輸出），所以 Q、K、V 都等於 tokens。token0 的 query 是 q0=[0,0]，和每個 key 的內積都是 0，除以 √2 仍是 0；exp(0)=1，所以 softmax 後，每個可讀來源的 attention 權重都是 1/(可讀來源數)：full 是 1/16；A=4 時每區 4 個，是 1/4；A=2 時每區 8 個，是 1/8。

attention 權重全部相等，token0 輸出就是可讀 token 的平均。token0 輸出的兩個 channel 數值相同，程式印的是第一個。full 平均全部 16 個：0 到 15 的平均是 7.5，所以 token0 輸出是 7.5/16=0.46875。area 只平均第一帶的 token0–3：0 到 3 的平均是 1.5，得 1.5/16=0.09375。

接著做干預：只把 token15 的兩個 channel 各加 5，其他 token 都不動，看 token0 輸出會不會變。

- **full**：token15 的 K 也跟著變了，但 q0=[0,0]，和它的內積仍是 0，所以 attention 權重仍是均勻的 1/16。只有 token15 的 V 多了 5，token0 輸出多了 5/16=0.3125，從 0.46875 變成 0.78125。
- **area**：token0 那張權重表裡根本沒有 token15，所以 token0 輸出仍是 0.09375，完全不變。

選 token0（q0=[0,0]）只是為了方便手算。area 的結論不靠 q0 是零向量：換成別的 query 向量，token0 那張表裡仍然沒有 token15。

這個結論是程式實際比對出來的。完整程式用斷言（assert）逐值比對干預前後的 token0 輸出：`full_affected`、`area_affected` 記錄 full 和 area 的 token0 輸出有沒有改變（用 `torch.allclose` 比較），`query_index=0` 是 token0，`changed_index=15` 是被改的 token。下面摘自完整程式的 `main()`，中文註解是本頁加的；`...` 處略去的兩行，就是算出 `full_affected`、`area_affected` 的那兩行：

``` { .python data-excerpt="lesson_cases/15-area-attention.py" }
area_size = tokens.shape[1] // areas  # 每區的 token 數：16 // 4 = 4
same_area = query_index // area_size == changed_index // area_size  # 0 // 4 = 0，15 // 4 = 3：不同區，False
...
assert full_affected and area_affected == same_area  # full 要受影響；area 受不受影響，要和「是否同區」一致
```

`same_area` 依 `areas` 自動算出兩個 token 是否同區，所以練習改 `areas` 或 `changed_index` 時，這個 assert 不用跟著改。

這也展示了收益與限制：減少跨區的 pair，同時也限制了這一層的遠距資訊交換。不過在完整的網路裡，資訊可以繞路跨區：

- 原版同一層的位置卷積會混入上下相鄰列（別區）的 V。
- 後面的卷積與下採樣（例如 stride 2 的卷積，把 feature map 的邊長縮成一半）也會讓相鄰位置的資訊流動。
- 官方設定檔 `yolov12.yaml` 在 stride 32（解析度最小）的那層，直接用 area=1，也就是 full attention。

反過來說，如果每層都用同樣的分帶、又沒有這些路徑，只靠 area attention 本身永遠跨不了帶。本實驗只有一層，也沒有那些路徑，所以不能斷言完整的 YOLOv12 跨區永遠不互動。

完整程式最後對 area 輸出與原 tokens 計算 MSE，做 backward 和一次 SGD，確認 QKV 權重有梯度。執行 `PYTHONPATH=. python lesson_cases/15-area-attention.py`，對照印出的這幾行：

- `full affinity shape/count`、`area affinity shape/count`：權重表 256 個與 64 個。
- `first-token output`：干預前的 token0 輸出，full 是 0.46875，area 是 0.09375，和前面手算的值相同。
- `changing token 15 affects token 0`：依序印出 `full_affected`、`area_affected`、`same_area`。`full=True, area=False` 表示干預影響 full、不影響 area；`same area=False` 表示 token15 和 token0 不同區。
- `after intervention`：干預後 full 變成 0.78125，area 仍是 0.09375，也和手算的值相同。
- 最後一行 `area-attention backward/step verified`：backward 與 step 成功。

本節只用 CPU，不需要 FlashAttention 或 GPU。

## 和 CNN 比較：每層讀得到哪些位置

以 token0 為例，一層能讀到的位置：

- 一層 3×3 卷積（padding=1）：0、1、4、5。會跨到下一條帶，卻讀不到同一列的 token3。
- A=4 的 area attention：0–3。讀得到整列，卻讀不到正下方的 token4。
- full attention：全部 16 個。

CNN 疊多層後，讀得到的範圍（感受野）會逐層擴大。兩者的權重來源也不同：CNN 事先假設只有局部（鄰近）的位置有關，而且各位置共用同一組卷積核，這是訓練前就寫進設計的假設（先驗）；attention 的權重則由當下輸入的 Q、K 算出，輸入不同，權重就不同。兩者在投影、channel 數和深度上的成本也不一樣。

若要比較偵測上的收益，應該對齊資料、訓練預算和計算預算（計算預算要盡可能定義清楚），再看遠距關係或小物件等失敗類型。本節只單獨檢驗 full 和 area 的差別，沒有訓練 CNN，不比較誰較準。

常見錯誤：

- **N 不能被 A 整除（例如 N=16、A=3）卻仍然 reshape**：元素數對不上，reshape 會直接報錯。完整程式 `attend` 函式開頭的 `assert areas > 0 and n % areas == 0` 就是在擋這件事。
- **把 area 當成多 head**：只分 head、不分區時，每個 head 仍讓每個位置讀全圖（每個 head 都有 N² 個 pair）；限制讀得到哪些位置的是分區。
- **接回時搞混 B 和 B×A**：reshape 後第 0 軸的長度是 B×A，依序是第 0 張圖的 A 區、第 1 張圖的 A 區……。接回時要用原本的 B，寫成 `reshape(B, N, C)`；誤寫成 `reshape(B * areas, N, C)`，元素數對不上，會報錯。
- **讓 full 和 area 各用一份不同的隨機 QKV 權重**：差異可能來自權重不同，而不是分區。這叫混雜：比較時另有一個因素也不同，差異就無法只歸因於想比較的那一項。所以本節讓兩條路共用同一份 QKV 權重。
- **以為這一層各區互不相通，就代表整個模型都互不相通**：這樣會只憑單層實驗，就下結論說完整的 YOLOv12 跨區永遠不互動。原版的位置卷積不受分區限制，會把上下相鄰列（別區）的 V 混進來；本節拿掉它，也拿掉 BN，token0 變不變才只取決於分區。後面的卷積、下採樣，以及 stride 32 那層的 full attention，也會讓資訊跨區（見前面〈遠處資訊能不能傳過來〉）。

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

本節的完整程式已於 2026-10-02 用 PyTorch 2.9.1+cpu 在 CPU 上執行過，程式裡的 assert 檢查全部通過。下面是那次印出的原始輸出；每個數字的意思，以本頁正文的說明為準。[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/15-area-attention.json)

??? example "展開本次實際輸出"

    ```text
    full affinity shape/count: (1, 16, 16) 256
    area affinity shape/count: (4, 4, 4) 64
    first-token output full=0.4688, area=0.0938
    changing token 15 affects token 0: full=True, area=False; same area=False
    after intervention: full=0.7812, area=0.0938
    area-attention backward/step verified; loss=0.0048
    ```

<!-- curriculum-evidence:end -->

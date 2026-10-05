# 18 影片串流：處理每一幀，並分清 FPS 與延遲

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.5.0/notebooks/18-video.ipynb){ .md-button }

本節把圖片偵測搬到影片上。影片是一連串的幀（frame：影片中的一張畫面）。偵測器（detector）不必換，只要把「讀一張圖 → 前處理 → 模型 → 後處理 → 畫框」放進迴圈，每一幀跑一次。讀完本節，你能把影片一幀一幀送進同一個模型，並分清「每秒處理完幾幀」和「一幀要等多久才有結果」這兩件事。

前置：會用[完整圖片推論](07-inference.md)把模型輸出變成框；懂 [letterbox 座標轉換](04-coordinates.md)，也在[用自己的圖片](08-own-images.md)看過非正方形的圖怎麼補邊縮成 64×64、框怎麼還原回原圖。

和單張圖片相比，影片多了四個新問題：

- 幀的時間：每一幀都要帶著它在來源中的時間戳（見〈先固定每一幀的格式〉）。
- 來源大小：寬 96、高 64 的畫面要先 letterbox 成 64×64，算出的框再還原回原圖（也在〈先固定每一幀的格式〉）。
- 佇列（queue）：處理比來源慢時，來不及處理的幀會排隊等待，延遲越來越大（見〈處理速率與延遲是兩回事〉）。
- 資源釋放：讀完或中途停止時，要關閉影片檔或相機，交還給系統（見〈接真影片的同一個入口〉）。

本節先用程式自己畫的假影片：深色底上的紅方塊，每幀往右移 4 畫素。用它測一條真的推論管線（從讀入畫面到畫好框的整串步驟）。之後再提供可替換的影片／相機 adapter（轉接函式：把影片檔或相機的畫面轉成本節的 `Frame` 格式，一幀一幀交出來）。本節預設不開相機，也不使用相機權限。

完整程式從零訓練 GridDetector：32 張合成矩形圖（資料 seed 4100；模型初始化 seed 7）、訓練 160 步、每批 8 張、Adam、學習率 0.01；不讀取任何既有的 checkpoint。影片來源有 12 幀，每秒 20 幀，也就是每 50 毫秒一幀。「來源每秒產生幾幀」叫來源幀率，單位寫成 FPS（frames per second，每秒幀數），所以本例的來源幀率是 20 FPS。每幀是寬 96、高 64 的 RGB 畫面（array shape `[64,96,3]`，依序是高、寬、通道）。紅方塊是 14×14 畫素，第 i 幀的左上角在 x=4+4i、y=20。這不是攝影機錄的影片：時間戳是用「第 i 幀 ÷ 20」算出來的，不是時鐘量到的。模型的預測則是真的 forward 結果，沒有用真值框冒充偵測。

## 先固定每一幀的格式

每一幀都包成一個 `Frame`，裡面只有三個欄位。下面是完整程式裡的定義，程式相同，註解換成中文：

``` { .python data-excerpt="lesson_cases/18-video.py" }
@dataclass
class Frame:
    index: int          # 第幾幀：從 0 開始遞增的整數
    timestamp_s: float  # 來源時間（秒）：這一幀在影片中的時刻，不是處理花了多久
    rgb: np.ndarray     # 畫面：uint8、HWC（高、寬、通道）、RGB 順序
```

`@dataclass` 讓 Python 自動寫好建立物件的程式。所以完整程式寫 `Frame(i, i / fps, rgb)` 就能建立第 i 幀，之後用 `frame.index`、`frame.rgb` 取出欄位。

第 i 幀的時間戳是 i/fps 秒，相鄰兩幀相差 1/fps 秒。本例 fps=20，間隔是 1/20 秒=50 毫秒。第 0 幀是 0 秒；最後的第 11 幀是 11/20=0.55 秒，也就是 550 毫秒，不是 11 毫秒。12 幀、每幀播放 50 毫秒，總播放時間是 12/20=0.6 秒。時間戳的最後值與總播放時長不同，因為最後一幀仍占一個播放間隔。

每一幀從來源到畫出框，要經過下面幾步。算法和第 8 章寬 120、高 80 的圖片相同，兩者都是 3:2：

1. 來源 array shape `[64,96,3]`（uint8，HWC）轉成 float32、換成 CHW、除以 255，得到 `[3,64,96]`。
2. letterbox 到 64×64。縮放比例取 min(64/64, 64/96)=2/3，寬 96 剛好縮成 64，左右不補 padding。內容高度 64×2/3≈42.67，取整為 43，所以縱向的實際比例是 43/64。剩下 64−43=21 列，上補 10 列、下補 11 列。
3. 加上 batch 軸，變成 `[1,3,64,64]`；模型的原始輸出（raw）是 `[1,4,4,7]`。
4. decode 得到的框是 64×64 輸入座標。再用 letterbox 回傳的 metadata 還原成寬 96、高 64 的來源座標，畫在原始幀上；不能直接把輸入座標畫回寬圖。

`run_stream` 把這幾步包成一個 generator。先看一個小例子：

```python
def count_up():
    for i in range(3):
        print('準備', i)
        yield i                        # 交出 i 後暫停，等外面要下一筆
for x in count_up(): print('拿到', x)  # 每要一筆，函式才往下跑到下一個 yield
```

含 `yield` 的函式叫 generator（產生器）。呼叫它時，函式本體還不會執行，所以單寫 `count_up()` 時，裡面的 `print` 一個都不會執行（在 Colab 只會顯示 `<generator object count_up at …>`，也就是這個還沒開始跑的 generator 物件本身）。使用結果的 for 迴圈每要一筆，它才往下跑到下一個 `yield`，交出一個值後暫停。所以上面的程式依序印出：準備 0、拿到 0、準備 1、拿到 1、準備 2、拿到 2。

`run_stream` 也是這樣：只呼叫 `run_stream(frames, model)` 不會開始逐幀工作；for 迴圈來要結果時，它才讀下一幀，處理完再用 `yield` 交出。下面是簡化過的核心片段，和完整程式不同的地方是：省略了函式開頭的說明文字、斷言（assert）、計時與畫框；letterbox 前的 tensor 另取名 `tensor_rgb`（完整程式裡 letterbox 前後的 tensor 都叫 `image`）；交出的 dict 只留三個欄位，完整程式還會交出疊好框的畫面 `image` 與各段計時 `ms`。`model` 傳進來前已設成 eval。`@torch.no_grad()` 寫在 `def` 的上一行，讓函式裡的計算都不記錄計算圖，用途和第 0 章的 `with torch.no_grad():` 一樣。`run_stream` 是 generator，PyTorch 只在它往下執行時關閉梯度，`yield` 交出結果時就恢復原狀，所以外面的 for 迴圈不受影響；如果改成在函式裡用 `with torch.no_grad():` 包住迴圈，generator 暫停時，外面的程式也會停在不記錄計算圖的狀態。

```python
@torch.no_grad()
def run_stream(frames, model):
    for frame in frames:
        # .copy()：保險起見，複製成獨立、可寫的陣列
        # permute 把 HWC 換成 CHW，除以 255 把值縮到 [0,1]：[64,96,3] uint8 → [3,64,96] float32
        tensor_rgb = torch.from_numpy(frame.rgb.copy()).permute(2, 0, 1).float() / 255
        # 推論時沒有 GT 框，所以傳 shape [0,4] 的空框；回傳的 image 是 [3,64,64]
        image, _, metadata = letterbox(tensor_rgb, torch.empty(0, 4), size=64)
        raw = model(image[None])  # image[None] 在最前面加上 batch 軸：[1,3,64,64]
        # score 門檻 0.1、NMS 的 IoU 門檻 0.5；[0] 取這唯一一張圖的結果
        prediction = decode_grid(raw, score_threshold=.1, nms_iou=.5)[0]
        # 用這一幀自己的 metadata，把框還原到寬 96、高 64 的來源座標
        prediction['boxes'] = undo_letterbox(prediction['boxes'], metadata)
        yield {'index': frame.index,  # 交出這一幀的結果，然後暫停
               'source_timestamp_s': frame.timestamp_s,
               'prediction': prediction}
```

`frames` 可以是任何能用 for 逐筆取出 `Frame` 的東西（iterable，可迭代物件），例如 list，或完整程式裡產生假影片的 `synthetic_frames`（它也是 generator）。所以來源不必一次全部載入記憶體。

完整程式的主程式用 `list(run_stream(...))` 一次收齊 12 幀的結果，是為了存成 GIF；這只適合小示範。每筆結果都含一整張疊了框的畫面，`list()` 會把每一幀的結果同時留在記憶體裡。以 1920×1080 的 RGB 畫面為例，光是畫素值一幀就有 1920×1080×3 位元組，約 6.2MB；10 分鐘、30 FPS 的影片有 18000 幀，全部留住至少要 112GB。所以正式的長影片要逐幀寫出，不能 `list()` 整段影片。

每一幀都用同一個模型：不重建 model 或 optimizer，也不做 backward。需要追蹤身份（跨幀判斷哪些框屬於同一個物件）時，另接第 19 章〈[簡易 tracking](19-tracking.md)〉。相鄰兩幀的第 0 個框未必是同一個物件，所以不能把每幀的框序號當成固定的物件 ID。

## 把管線時間拆開量

完整程式把每一幀的處理分成四段計時：前處理、模型、後處理（decode／NMS／還原座標），以及畫框。做法是在每一段前後各呼叫一次 `time.perf_counter()`（高精度計時器，單位秒），兩次相減再乘 1000，就是這一段花了幾毫秒。在 CPU 上，PyTorch 的運算回傳時就已經算完（同步執行），所以這樣量得準；GPU 則會先把工作排隊、稍後才算（非同步）：每次讀 `time.perf_counter()` 之前（每一段的開頭和結尾），都要先呼叫 `torch.cuda.synchronize()` 等它算完，或改用 `torch.cuda.Event` 計時；否則 GPU 的計算時間會被算進後面第一個要等結果的步驟。

正式的 12 幀開始之前，完整程式先暖機（warmup）。暖機在 3.3 節〈[Plain／residual 對照](03-comparison.md)〉說明過：同一個程式裡，第一次執行某段計算，常會多花一些只需要做一次的準備時間，所以正式計時前先跑幾次，把這些一次性的準備做掉，這幾次的時間丟掉不算。不暖機的話，這些準備時間會算在最先處理的第 0 幀頭上。本節在 `main()` 裡用 1 幀暖機：

``` { .python data-excerpt="lesson_cases/18-video.py" }
# main() 內，model 是剛訓練好的偵測器
warmup = list(run_stream([next(synthetic_frames(count=count, fps=fps))], model))  # 暖機：只處理 1 幀
assert len(warmup) == 1 and len(warmup[0]['prediction']['boxes']) > 0, 'warm-up must reach NMS and drawing'
results = list(run_stream(synthetic_frames(count=count, fps=fps), model))  # 正式的 12 幀
...
# ms 的每一項（preprocess、model、postprocess、drawing、total）各取 12 幀的中位數
timing = {key: float(np.median([r['ms'][key] for r in results])) for key in results[0]['ms']}
```

暖機那一行由內往外讀：`next(...)` 是手動向 generator 要下一筆（for 迴圈每一圈做的也是這件事），這裡只向新建的 `synthetic_frames` 要第一筆，得到另外產生的一份第 0 幀；`[ ]` 把它裝進只有 1 幀的 list，當成 `run_stream` 的來源；最外層的 `list()` 把結果要完，`run_stream` 才真的處理這一幀。這一幀走完整條管線：前處理、模型、後處理、畫框。它不在 12 幀裡：正式的 `results` 另外新建一個 `synthetic_frames`，從第 0 幀重新開始。暖機的時間不進任何統計；它的結果只用在下一行的斷言，以及 report 的 `warmup_frames`（暖機幀數，值是 1）。report 是 `main()` 印出、也存成 `artifacts/lesson-18/report.json` 的摘要，頁尾〈實際執行紀錄〉印出的 JSON 就是它。

暖機用的畫面要有物件，而且模型要真的在上面留下框：這樣後處理才會呼叫 NMS，畫框這一段也才會畫出框、寫上分數，這些步驟第一次執行時的準備才會一起在暖機時做掉。斷言就檢查兩件事：暖機剛好 1 幀，而且這一幀有框。

本次用 PyTorch 2.9.1、CPU、2 個執行緒（threads：程式裡可以同時進行的工作流程）。下表就是上面的 `timing`，也就是 report 裡的 `median_ms`：12 幀各段時間的中位數（median）。一次性的準備已經在暖機時做掉，所以第 0 幀也算在內。12 是偶數，`np.median` 取排序後第 6、7 小兩個值的平均。數字是這次小模型的實測，不是速度保證。

| 階段 | 中位數（毫秒） | 包含 |
| --- | --- | --- |
| 前處理 | 0.276 | uint8→tensor、縮放、padding |
| 模型 | 0.310 | 一次 forward |
| 後處理 | 0.439 | score 篩選、NMS、框還原 |
| 畫框 | 0.070 | PIL 影像與文字框 |
| 全流程 | 1.100 | 上述各項的逐幀總時間 |

四項中位數相加是 0.276+0.310+0.439+0.070=1.095，比全流程的 1.100 少。這不是漏算：每一幀內，四段相加正好等於這一幀的總時間。差異來自「先各取中位數再相加」，因為各項的中位數可能來自不同的幀。例如三幀的前處理是 1、2、5 毫秒，模型是 5、1、2 毫秒：兩項的中位數都是 2，相加得 4；但三幀的總時間是 6、3、7，中位數是 6。

這些數字也沒有包含影片解碼（把壓縮的影片檔還原成一張張畫素陣列；和第 6～7 章把模型輸出轉成框的 decode 不同）、網路、磁碟、螢幕更新，或相機那端的佇列。

### 處理速率與延遲是兩回事

先固定兩個說法：

- 處理速率：程式每秒處理完幾幀，也就是[術語快速查](../glossary.md)裡的 throughput。
- 延遲（latency）：一幀從被擷取到處理完成，經過多久。這裡包含排隊等待的時間，比第 12 章「跑一次推論花的時間」範圍更廣。

這兩個量也都和來源幀率（來源每秒產生幾幀）不同。

表中全流程 1.100 毫秒，換算成 1000 毫秒 ÷ 1.100 毫秒≈每秒 909 幀。這是用典型單幀計算時間粗略換算的數字。實際持續處理速率要用處理完的幀數除以總耗時量，未必等於中位數的倒數；接上相機後還有其他時間要算。接上相機後，每秒實際處理完幾幀，還受影片解碼、相機幀率與排隊限制，而且不會超過相機的幀率。本例也量不到排隊：合成來源只是把時間戳標成每 50 毫秒一幀，程式沒有用 `time.sleep`（讓程式暫停指定的秒數）真的去等；下一幀一要，就立刻產生。

再看處理比來源慢的情況。30 FPS 的相機每 1000/30≈33.3 毫秒送來一幀。假設處理一幀要 50 毫秒，而且來不及處理的幀全部排隊（佇列沒有上限），延遲就會越來越大，看到的結果越來越舊。這時處理速率卻穩定在每秒 20 幀，光看它看不出問題。所以要決定：保留全部幀、丟掉舊幀只處理最新的一幀，或降低解析度。降低解析度的目標，是讓每幀處理時間低於來源間隔 33.3 毫秒。要量實際的畫面年齡（結果完成時，這一幀已經是多久以前拍的），必須同時記下擷取時間、排隊時間與完成時間。

??? note "算算看：延遲怎麼累積"

    相機每 1000/30=100/3≈33.3 毫秒送來一幀，所以第 k 幀（k 從 0 起）在 100k/3 毫秒到達。處理一幀固定 50 毫秒，程式一刻不停。

    全部排隊時，第 k 幀要等前面 k 幀都做完，在 50(k+1) 毫秒才做完。延遲=完成時刻−到達時刻=50(k+1)−100k/3=50+50k/3 毫秒，約 50+16.7k：

    - 第 0、1、2 幀：50、66.7、83.3 毫秒。
    - 第 30 幀（1 秒時拍到）：550 毫秒。
    - 第 300 幀（10 秒時拍到）：5050 毫秒，約 5 秒。

    每多一幀，延遲就多約 16.7 毫秒，不會停下來；處理速率卻始終是每秒 1000÷50=20 幀。

    只處理最新一幀時，每做完一幀，就拿當下最新到達的那一幀，沒輪到的舊幀丟掉：

    - 0～50 毫秒：處理第 0 幀，延遲 50 毫秒。
    - 50 毫秒時，最新的是第 1 幀（33.3 毫秒到達）：做到 100 毫秒，延遲 66.7 毫秒。
    - 100 毫秒時，第 3 幀剛好到達：做到 150 毫秒，延遲 50 毫秒；第 2 幀被跳過。
    - 之後照這個規律重複。

    延遲不再累積；本例在 50～67 毫秒之間，代價是約每 3 幀有 1 幀沒處理。一般來說，這種做法的延遲最多是處理時間加一個來源間隔。

## 真實預測，也保留失敗

[![實際連續幀輸出，黃色是模型預測](../assets/diagrams/18-video.svg)](../assets/diagrams/18-video.svg)

圖上的小字可點開原尺寸圖放大查看。

圖中三格依序是第 0、5、11 幀，也就是第一幀、中間幀與最後一幀。紅方塊是畫面裡的物件本身，不是畫上去的真值（GT）框；黃框是模型預測，框上的白字是它的 score。每格下方第一行是幀號與來源時間戳；第二行的「處理 … ms」是這一幀全流程的處理時間（單次量測，不是中位數），「框 … 個」是這一幀的框數。

單幀的處理時間也看這一幀要做多少事。第 0 幀有框：後處理要跑 NMS，畫框這一段也要真的畫出框、寫上分數；第 5、11 幀沒有框，這兩件事都不用做。所以比較單幀時間之前，先看各幀有沒有框。這是每幀工作量的差別，和暖機要先做掉的一次性準備是兩回事。

本次每幀框數為 `[1,1,0,0,0,0,1,1,0,0,0,0]`。12 幀全都完成處理，卻有 8 幀沒有 score 高於 0.1 的框。這是模型漏檢，不是串流掉幀（掉幀：來源的某些幀沒被處理就被跳過）。

有框也不等於框得準。用[檔案實測紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/video-file.json)保存的框計算，第 0、1、6、7 幀的預測框和紅方塊的 IoU（交集面積÷聯集面積）約 0.50、0.47、0.30、0.46；這份紀錄怎麼來，見下方摺疊區〈進階：用無損影片檔驗證 adapter〉。

本節的 score 門檻是 0.1：每個候選框拿自己的 score 去比，低於 0.1 就丟掉。它比第 7 章畫圖用的顯示門檻 0.25（也是 `decode_grid` 的預設值）低。完整程式 `run_stream` 裡的英文註解寫明了理由：在這些 letterbox 過的畫面上，這個只訓練 160 步的偵測器，多數幀的最高 score 都不到 0.25；門檻用 0.1，較弱的偵測也會留下來。本節沒有為了好看而填入 GT 框；同一段註解也寫明，門檻 0.1 下的框數不是準確率。

為什麼漏檢？以下只是可能原因；本節沒有做對照實驗，不能當成已證實的因果。

- 訓練圖是沒有 padding 的 64×64 正方形；每個框都完整落在一個 16×16 的格子內，寬高 8～15 畫素。影片幀經 letterbox 後，上下有 padding。
- 物件在模型輸入中約寬 9.3、高 9.4 畫素（14×2/3、14×43/64），仍在訓練框的大小範圍內。所以物件大小和訓練時相近，不是差異所在。
- 物件在輸入中的 y 約 23.4～32.8（20×43/64+10、34×43/64+10），每一幀都略微跨過 y=32 的格線。x 方向（輸入中的 x 是來源 x 乘 2/3），第 2～4、8～10 幀還橫跨 x=16 或 x=32 的格線，這 6 幀都沒有框；沒跨線的第 0、1、6、7 幀都有框；但第 5、11 幀在 x 方向沒有跨線，也沒有框。所以「跨格」最多只是部分原因。

在 repo 根目錄執行 `PYTHONPATH=. python lesson_cases/18-video.py`，`main()` 會用斷言檢查：暖機剛好 1 幀而且得到了框；正式結果有 12 筆，每張輸出圖都是寬 96、高 64；12 個 index 依序是 0～11；最後時間戳是 0.55 秒。它會產生 `artifacts/lesson-18/stream.gif`、`report.json` 與靜態圖 `panel.svg`；本頁的圖，就是頁尾執行紀錄那次執行產生的 `panel.svg`。這條路不需要相機或 GPU，快速驗證只要訓練 160 步、暖機推論 1 次，再推論 12 次。

## 接真影片的同一個入口

完整程式提供 `opencv_frames(source)` 這個 adapter，把影片檔或相機的畫面轉成一幀幀的 `Frame`。它用 OpenCV（Python 裡叫 `cv2`）讀影片。本課固定用 `opencv-python-headless`，這是不含開視窗功能的版本，適合 Colab 這類雲端環境。OpenCV 的四個套件 `opencv-python`、`opencv-contrib-python`、`opencv-python-headless`、`opencv-contrib-python-headless` 共用 `cv2` 這個名字，同一環境只能裝一種。已經裝了別的 OpenCV 套件時（Colab 通常已經預裝），先用 `python -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless` 全部移除，再只裝本課的這一個；在 Colab 的 code cell 裡，指令前面要加 `!`。

adapter 裡有三個重點：

- 用 `cv2.VideoCapture('clip.mp4')` 開啟檔案。開啟後，程式會占用這個檔案或相機，用完要呼叫 `capture.release()` 交還系統。VideoCapture 物件被回收時，OpenCV 也會自動釋放，但何時回收不由你控制：物件還被變數或錯誤訊息留著時，在它被回收或程式（Colab 的工作階段）結束之前，相機會一直被占住。
- OpenCV 讀出的顏色順序是 B、G、R（藍、綠、紅），所以每幀先轉成 RGB。不轉的話紅色會變成藍色，類別就錯了（本書紅是類別 0、藍是類別 1）。
- `capture.release()` 寫在 `try…finally` 的 `finally` 區塊裡。讀到檔尾、generator 被中途關閉，或 adapter 自己出錯（例如檔案打不開）時，`finally` 都會執行，檔案或相機因此會被釋放。這就是開頭說的「資源釋放」。外面的迴圈提前 `break` 或推論出錯時，generator 會停在 `yield`，要等 Python 回收它（確認已沒有任何東西用到它，把它清掉）時才會自動關閉。它若還存在變數裡，或 Colab 還留著那次錯誤的資訊（traceback），就不會被回收，檔案或相機會繼續被占著。所以後面〈換成自己的影片〉用 `closing`，離開 `with` 時一定把它關閉，不必等回收。

換來源時，其他程式不用改：`run_stream(opencv_frames('clip.mp4'), model)` 就能處理影片檔。相機來源可以傳 0（代表第一支相機），但本節沒有開啟或驗證相機。

??? note "完整程式裡的 `opencv_frames`（加了中文註解）"

    ``` { .python data-excerpt="lesson_cases/18-video.py" }
    def opencv_frames(source):
        """Optional real file/camera adapter. Never called by this lesson's main()."""
        import cv2  # optional: pip install opencv-python-headless
        capture = cv2.VideoCapture(source)  # source：檔名（字串）或相機編號（例如 0）
        try:
            if not capture.isOpened():  # 開不起來就報錯；finally 仍會執行
                raise RuntimeError(f'Cannot open video source: {source}')
            fps = capture.get(cv2.CAP_PROP_FPS)  # 影片記錄的幀率；下面只在大於 0 時採用
            started, index = time.perf_counter(), 0  # 從這裡開始計時，幀號從 0 起
            while True:
                ok, bgr = capture.read()  # 讀下一幀；讀不到（例如到了檔尾）時 ok 是 False
                if not ok:
                    break
                # 來源是字串而且 fps 大於 0：用 index/fps；否則用開啟後經過的秒數
                timestamp = index / fps if isinstance(source, str) and fps > 0 else time.perf_counter() - started
                yield Frame(index, timestamp, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))  # 轉成 RGB 再交出
                index += 1
        finally:
            capture.release()  # 讀到檔尾、被中途關閉，或 try 裡面出錯，最後都會走到這裡
    ```

每一幀的時間戳，adapter 依情況用兩種算法：

| 情況 | 時間戳怎麼來 | 說明 |
| --- | --- | --- |
| 來源是字串（例如檔名），而且讀得到 FPS | 第 i 幀=i/fps 秒 | 例如 20 FPS 時，第 11 幀是 11/20=0.55 秒 |
| 其他情況，例如相機傳 0，或讀不到 FPS | 用 `time.perf_counter()` 量開啟來源後經過的秒數 | 這是程式拿到這一幀的時刻，不是相機感光元件拍下它的時刻 |

`perf_counter()` 是單調（monotonic）計時器：只會往前走，不受電腦校時影響，適合量經過的時間。

可變幀率（相鄰幀的時間間隔不固定）的影片或網路串流，不能用 i/fps 推算時間，應該改讀每一幀的 PTS（presentation timestamp，呈現時間戳：播放時這一幀該出現的時刻）。本案例沒有實作可變幀率與 PTS；adapter 也不會自己分辨這些情況，只要來源是字串、又讀得到 FPS，就一律用 i/fps。用固定 FPS 推算出的時間，不要當成真實的擷取時間。

### 換成自己的影片

換自己的 `clip.mp4` 時，先在 repo 根目錄執行 `python -m pip install -r requirements-video.txt`，安裝固定版本的依賴（環境裡已經有別的 OpenCV 套件時，先照上面的說明移除），並把影片檔放在 repo 根目錄。下面的程式用 `runpy` 載入完整程式裡的函式，在新開的 Python 工作階段也能執行。先說明三個寫法：

1. 檔名 `18-video.py` 以數字開頭，又有連字號，不能寫成一般的 `import`。`runpy.run_path` 會執行這個檔，並回傳一個 dict，裡面裝著檔案執行後的所有名稱（函式、類別、匯入的模組等）。所以用中括號 `video['fit_detector']` 取出函式，再加 `()` 呼叫。
2. 檔尾寫著 `if __name__ == '__main__': main()`，這個條件只在直接執行這個檔時成立。用 `run_path` 執行時，`__name__` 不是 `'__main__'`，所以 `main()` 不會被呼叫，不會重跑 12 幀示範。
3. `closing(x)` 讓 `with` 區塊結束時（包括 `break` 跳出或出錯）自動呼叫 `x.close()`。這裡的 x 是 `opencv_frames` 這個 generator；它若還停在 `yield`，被 `close()` 時就會在那裡結束並執行 `finally`，於是呼叫 `capture.release()`。

注意：`fit_detector()` 每呼叫一次，就重新訓練一次模型（160 步）。

```python
import runpy
import torch
from contextlib import closing
torch.set_num_threads(2)  # PyTorch 的 CPU 運算只用 2 個執行緒（和完整程式一樣）；OpenCV 解碼另有自己的執行緒
video = runpy.run_path('lesson_cases/18-video.py')  # 執行完整程式檔，回傳裝著檔內名稱的 dict
model = video['fit_detector']()  # 取出函式再呼叫：重新訓練 160 步，得到合成矩形模型
with closing(video['opencv_frames']('clip.mp4')) as frames:  # 離開 with 時自動呼叫 frames.close()
    for result in video['run_stream'](frames, model):  # for 每要一筆，才處理一幀
        print(result['index'], len(result['prediction']['boxes']))  # 印出幀號與框數
```

這個用合成圖訓練的模型，仍不能直接辨識任意照片裡的物件；換真實來源前，要先有對應的模型與類別。

在 Colab 跑過本節完整程式格後，`run_stream`、`opencv_frames` 已在同一個工作階段定義好，可以直接呼叫，不需要 runpy。在自己電腦用 `PYTHONPATH=. python lesson_cases/18-video.py` 跑完，程式就結束了，函式不會留給之後另開的 Python，所以才需要上面的 runpy 寫法。

本機命令需先依 [README 環境步驟](https://github.com/birdhackor/learn_to_yolo#readme)安裝固定依賴，並在 repository 根目錄執行；Colab 則先跑本節環境格。

??? note "進階：用無損影片檔驗證 adapter"

    先說明幾個名詞。FFV1 是一種無損影片編碼：交給它的像素會原樣存回來。本實驗用 OpenCV 寫入 8 位元 BGR 畫面，OpenCV 讓 FFV1 直接存 RGB 類的格式，所以讀回後每個畫素都不變，能逐值比對；若先把畫面轉成 YUV 4:2:0（把顏色資訊縮成四分之一解析度的格式）再編碼，即使用 FFV1，顏色也已經變了。AVI 是裝影片的檔案格式（容器）。MP4 常用的 H.264 除了有損壓縮會丟掉一些細節，通常也會先轉成 YUV 4:2:0，所以讀回的畫素會略有不同。

    同一個 adapter 已用 12 幀的 FFV1 無損 AVI 檔實測：把前面 `synthetic_frames` 產生的畫面轉成 BGR 寫入檔案，再用 adapter 讀回成 RGB，畫素與原本完全相同；模型的框、分數、類別與疊圖也都相同。12 幀的時間戳仍是 0 至 0.55 秒。程式也核對了三種情況：讀到檔尾、中途關閉 generator、檔案打不開，OpenCV 的影片物件都已關閉。[檔案實測紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/video-file.json)保留了影片的 SHA-256（檔案指紋）與結果。這次沒有測實體相機，也不能保證有損 MP4 編解碼後的畫素完全相同。

    可自己重跑這段不用下載的檔案實驗（環境裡已經有別的 OpenCV 套件時，先照〈接真影片的同一個入口〉的說明移除）：

    ```bash
    python -m pip install -r requirements-video.txt
    PYTHONPATH=. python scripts/verify_video_file.py
    ```

    結果寫在 `artifacts/runs/video-file/result.json`。上面連結的檔案實測紀錄，是同一支程式加上 `--record artifacts/checks/curriculum/video-file.json` 產生的。

    `verify_video_file.py` 還會產生 `artifacts/runs/video-file/lossless-fixture.avi` 及疊圖，也把真實預測接給第 19 章的 tracker（追蹤器）。這段影片處理的計時包括開啟影片、讀取與解碼、前後處理、模型、畫框及收集結果；不含先前的訓練、影片編碼、寫入磁碟、顯示或網路佇列。不要把這 12 幀的小測試當成影片的正式效能。

## 收益、代價與常見錯誤

收益：同一個圖片模型可以接不同的來源，同時保留座標還原與時間戳的約定。

代價：要處理影片解碼（把影片檔還原成畫素陣列）、顏色順序與尺寸等格式、來不及處理時的排隊，以及寫出結果（例如 GIF）或顯示畫面的成本。

常見錯誤：

- OpenCV 讀出的 BGR 沒有轉成 RGB。
- 每一幀都重新載入權重。
- 沒有把框還原回原圖座標（扣掉 padding、再除以縮放比例）。
- 把幀的序號（index）當成毫秒。
- 只量了模型 forward，就當成端到端（從拿到畫面到結果出來）的延遲回報。

## 自主練習

練習 1、2 要改的是完整程式的最後一行 `main()`（在 `if __name__ == '__main__':` 底下，前面的縮排要保留）：在 Colab 是最後一個程式格的最後一行，在本機是 `lesson_cases/18-video.py` 的最後一行。完整程式的斷言不需要修改：和幀數、時間戳有關的斷言，以及 summary 與 GIF 每幀時長，都用 `count`、`fps` 計算，會自動跟著改。

練習 1：把最後一行改成 `main(count=24)`（fps 保持 20）。先自己算：最後一幀的時間戳與總播放時間各是幾秒？紅方塊從第幾幀開始被右邊界裁切？哪一幀完全看不到？最後一幀若沒有預測框，算不算模型漏檢？靜態圖 `artifacts/lesson-18/panel.svg` 選第 0 幀、中間幀與最後一幀，中間幀由 `save_panel` 裡的 `(len(results)-1)//2` 決定（`//` 是整數除法，捨去小數），這次是哪三幀？

??? note "參考答案"

    - 時間戳：最後一幀是第 23 幀，時間戳 23/20=1.15 秒；24 幀的總播放時間是 24/20=1.2 秒。
    - 裁切：第 i 幀紅方塊的左緣 x=4+4i，寬 14。右緣 x+14 超過 96 就會被裁：4+4i+14>96，得 i>19.5，所以從第 20 幀開始。第 20、21、22 幀的 x 是 84、88、92，只剩 12、8、4 畫素寬。第 23 幀的 x=96，已經在畫面外，看不到紅方塊；report 的 `object_visible_per_frame` 最後一個值是 false。
    - 漏檢：不算。第 23 幀畫面裡沒有物件，沒有東西可漏。
    - 靜態圖：中間幀=⌊(24−1)/2⌋=11，所以是第 0、11、23 幀，不沿用預設 12 幀時的 0、5、11（⌊(12−1)/2⌋=5）。

練習 2：把最後一行改成 `main(fps=10)`（count 保持 12，紅方塊每幀仍右移 4 畫素）。最後時間戳、總播放時間、GIF 每幀時長、紅方塊每秒移動幾畫素，各變成多少？每幀的框數會變嗎？

??? note "參考答案"

    - 最後時間戳是 11/10=1.1 秒，總播放時間是 12/10=1.2 秒。
    - GIF 每幀時長：`round(1000/fps)`=100 毫秒，原本是 50 毫秒。
    - 每一幀的畫面都和 fps=20 時一模一樣，模型的框也與 fps=20 時相同。改變的只是每一幀代表的時間：同樣 4 畫素的位移現在花 0.1 秒，所以每秒從 80 降為 40 畫素。這與模型的計算速度無關。
    - GIF 以 1/100 秒為單位記錄每一幀的顯示時間，50、100 毫秒剛好存得下；30 FPS 時 `round(1000/30)`=33 毫秒，會被存成 30 毫秒，播放略快。

練習 3：30 FPS 的相機、處理一幀要 50 毫秒，來不及處理的幀全部排隊。3 秒時拍到的第 90 幀，什麼時候處理完？延遲多少？改成只處理最新一幀，延遲大約多少？

??? note "參考答案"

    全部排隊時，第 k 幀在 50(k+1) 毫秒做完，所以第 90 幀在 50×91=4550 毫秒做完。它在 90÷30=3 秒，也就是 3000 毫秒時到達，延遲=4550−3000=1550 毫秒=1.55 秒；代入〈算算看：延遲怎麼累積〉的 50+50k/3 也一樣：50+1500=1550。這段時間處理速率一直是每秒 20 幀。

    只處理最新一幀時，延遲不再累積，本例在 50～67 毫秒之間；代價是約每 3 幀有 1 幀沒處理。

<!-- curriculum-evidence:start -->

## 實際執行紀錄

本節的完整程式於 2026-10-05 在 INTEL(R) XEON(R) PLATINUM 8573C（2 個執行緒）上用 PyTorch 2.9.1+cpu 執行，程式裡的 assert 全部通過。下面是那次印出的原始輸出；輸出裡若有計時或訓練得到的數字，換一台電腦會略有不同。每個數字的意思，以本頁正文的說明為準。[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/18-video.json)

??? example "展開本次實際輸出"

    ```text
    {
      "frames": 12,
      "source_fps": 20,
      "source_duration_s": 0.6,
      "source_last_timestamp_s": 0.55,
      "gif_frame_duration_ms": 50,
      "source_size_hw": [
        64,
        96
      ],
      "object_visible_per_frame": [
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true
      ],
      "model_input_hw": [
        64,
        64
      ],
      "boxes_per_frame": [
        1,
        1,
        0,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        0
      ],
      "warmup_frames": 1,
      "median_ms": {
        "preprocess": 0.2764804958133027,
        "model": 0.30956500268075615,
        "postprocess": 0.43933201231993735,
        "drawing": 0.06989399844314903,
        "total": 1.1000390077242628
      },
      "camera_adapter": "provided, not executed",
      "limits": "synthetic lazy producer; no capture/codec/display/network queue latency measured"
    }
    GIF: artifacts/lesson-18/stream.gif; actual static panel: artifacts/lesson-18/panel.svg
    ```

<!-- curriculum-evidence:end -->

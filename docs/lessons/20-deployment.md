# 20 ONNX／TensorRT：匯出後先證明同一個輸入得到同一個結果

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.6.1/notebooks/20-deployment.ipynb){ .md-button }

你已經有圖片推論管線，也知道偵測品質與處理時間要分開看。現在要把模型交給另一個執行程式，例如部署到伺服器或裝置，讓它能持續接收輸入、回傳結果。換了執行方式，先要回答：同一張圖還會得到相同的框嗎？確認結果後，才有意義比較速度。

本節先把 PyTorch 模型匯出成 ONNX（Open Neural Network Exchange，通用的模型交換格式）。檔案保存 forward 的運算圖與權重，再由 ONNX Runtime（ORT，讀取並執行 ONNX 的程式）在 CPU 上計算。這讓使用端不必用 PyTorch 執行模型，也提供另一個可測的後端（backend，實際算出模型結果的程式）。

## 交付的不只有權重，還有輸入輸出約定

模型換了後端，圖片仍要依原方式轉色、縮放、補邊，raw 仍要解碼、NMS、還原到原圖。如果接收者只拿到模型檔，卻把 BGR 當 RGB、漏除以 255 或讀錯輸出軸，成功執行也會得到錯的框。因此先把這些操作寫成部署契約：兩端同意的輸入、輸出與外部工作。

這次用第 7 章 GridDetector（width 8、15,511 個參數），輸入 `[B,3,64,64]`，raw 輸出 `[B,4,4,7]`。每格最後七項依序是四個框 logits、一個 objectness、兩個類別 logits。先用 softmax 取得兩類機率，選機率最高的類別；score 是 `sigmoid(obj)` 乘上這個類別的機率，一個候選得到一個分數。它不是[16.2 的部署 head](16-inference-head.md)，不能換用那一節只以 sigmoid 算類別分數的 decoder。

程式從零建立模型，用合成圖做一次 forward、grid loss、backward 和 SGD 更新，再固定這份權重匯出，不下載 pretrained 權重、不需要 torchvision。一次更新讓案例包含實際更新過的模型，但不足以證明偵測品質；一致性檢查需要同一份權重，品質則另用[第 17 章](17-capstone.md)的獨立圖片評估。匯出前仍切到 eval，雖然這個模型沒有會隨 train／eval 改變行為的層。

### 同一張來源，各自保留還原資訊

用三張 uint8 HWC RGB 雜訊圖做測試，每值 0～255，shape 順序是高、寬、通道。以寬×高說，它們是 96×64、80×48、48×80；例如第一張 array shape 是 `(64,96,3)`。選非正方形圖，是為了讓前後處理真的經過縮放、padding 與座標還原。這裡不是測是否找對物件，因此圖裡不用有真實物件，只需兩端收到同一份輸入。

每張各自轉 float32、CHW、除以 255，再 letterbox 到 64×64。第一張寬縮成 64，高約 42.67、取整成 43，上補 10、下補 11。每張都有自己的 metadata（實際縮放比例與 padding 等變換紀錄），後處理必須拿該張的紀錄還原，不能整個 batch 共用第一張的比例。

![部署流程：ONNX 只保存藍色模型區塊，橘色前後處理共用，metadata 繞過模型](../assets/diagrams/20-deployment.svg)

圖中橘色是外部 Python 的前後處理，藍色是模型 raw 計算；`grid.onnx` 只包藍色部分。右側橘線帶 metadata，直接送到後處理，不經模型。藍區左側的白框是各後端的執行方式：PyTorch 用原模型，ORT 讀 ONNX，TensorRT 先把 ONNX 建成 engine 再執行。紫圈 1、2、3 標出稍後的檔案、raw 與還原框檢查位置。

交付這份 ONNX 時，檔案與呼叫介面要寫清楚。手機上可左右滑動表格查看完整欄位：

| 項目 | 本次契約 |
| --- | --- |
| 輸入名稱 `images` | float32、RGB、CHW、值 0～1，shape `[B,3,64,64]` |
| 可變軸 | 只有 batch B；通道、高、寬固定 |
| 輸出名稱 `raw_grid` | shape `[B,4,4,7]`，還沒 decode |
| 檔案外的前處理 | RGB／CHW 轉換、除以 255、letterbox，保存每圖 metadata |
| 檔案外的後處理 | 用上述 score 公式 decode，截斷 score<0.05，按類別 NMS（IoU>0.5 刪除），再 undo_letterbox |
| 回傳框座標 | 每張原圖上的 xyxy 畫素座標 |

這份契約也是 API（Application Programming Interface，程式呼叫介面）的使用約定：呼叫者按名稱送 tensor，接收 raw 後完成外部處理。若另包成網路服務，還要沿用同樣的顏色、shape、分數與座標約定。

## 匯出與載入：兩端要支持同一份運算規格

ONNX 的運算圖記下 forward 做了哪些運算、彼此怎麼接，例如 Conv、Relu、AveragePool。它不保存[第 2 章](02-diagnostics.md)為 backward 追蹤梯度的 PyTorch 計算圖。本次 opset=17：opset 是 ONNX 運算規格的版本號，匯出器與 ORT 都要支援這些運算的定義。

以下是加中文註解的簡化寫法。ORT 載入模型後建立 session 物件，之後可反覆呼叫 `session.run`。完整程式另固定 ORT 執行緒數，並檢查 session 的輸入規格。

```python
# example 就是完整程式的 images[:1]：一張 [1,3,64,64] 的樣本圖
torch.onnx.export(model, example, 'grid.onnx',  # 完整程式存到 artifacts/lesson-20/grid.onnx
    input_names=['images'], output_names=['raw_grid'],  # 替輸入、輸出取名，之後用名字指定
    dynamic_axes={'images': {0: 'batch'}, 'raw_grid': {0: 'batch'}},  # 只有第 0 軸可變
    opset_version=17, dynamo=False)  # dynamo=False：用舊版匯出器（見下方說明）
onnx.checker.check_model(onnx.load('grid.onnx'))  # 第一層檢查：檔案結構（預設不查各運算的型別與 shape）
# providers 指定 ORT 用哪種硬體執行，這裡是 CPU
# （完整程式另外傳入 sess_options，固定 ORT 的執行緒數）
session = ort.InferenceSession('grid.onnx', providers=['CPUExecutionProvider'])

# 取自完整程式的 B 迴圈：輸入要是 NumPy 陣列；run 回傳 list，[0] 取出 raw_grid
ort_raw = session.run(['raw_grid'], {'images': batch[:b].numpy()})[0]
```

舊版匯出器拿 `example` 實際跑一次 forward，記下運算。`dynamic_axes` 的結構是「輸入／輸出名稱→{軸號: 名稱}」，宣告兩者第 0 軸可變、叫 `batch`；沒宣告的 3、64、64 沿用樣本大小固定。`session.run` 用字典按輸入名 `images` 送 NumPy 陣列，用 `['raw_grid']` 指定輸出，回傳 list 的第 0 筆就是 raw。

??? note "為什麼用 dynamo=False"

    PyTorch 有新、舊兩套把模型轉成 ONNX 的匯出器（exporter）。從 2.9 版起，預設是以 torch.export 為基礎的新匯出器（`dynamo=True`）。本例指定 `dynamo=False`，改用舊的 TorchScript 追蹤式匯出器（拿樣本實際跑一次、記下運算），因為本節的 opset 17 與 `dynamic_axes` 是用它測過的。

    執行時會印出一段 DeprecationWarning（棄用警告：提醒這個功能將來可能移除），開頭是 `You are using the legacy TorchScript-based ONNX export`。這是預期中的警告，不是錯誤，程式照常執行。舊匯出器將來可能被移除，所以升級 PyTorch 或改用新匯出器時，要重跑本節的比對，並改用該版本支援的 `dynamic_shapes`（新匯出器宣告可變軸的參數）與相依套件。



## 可變 batch，要真的送進不同張數

ORT 的 `session.get_inputs()[0].shape` 是 `['batch',3,64,64]`，所以完整程式實際跑 B=1、2、3。每圈先檢查輸入為 `[B,3,64,64]`，再檢查兩端 raw 都為 `[B,4,4,7]`。這個輸入斷言很有用：只有兩張時，`batch[:3]` 不報錯，只回傳兩張；不查 shape，原本聲稱的 B=3 測試就悄悄變成 B=2。

batch 可變不等於圖片高寬可變。程式刻意送 80×80，並確認 ORT 拒絕：輸入契約已把高寬固定為 64，ORT 在計算前就會擋下。PyTorch 的 `AdaptiveAvgPool2d` 雖可以將其他尺寸平均成 4×4，現有 decode 卻仍按 64×64、每格 16 畫素換算；80×80 每格應是 20 畫素，直接使用會放錯框。

即使放寬 ONNX 輸入規格也不夠：匯出時這個池化按 64×64 樣本記成固定 2×2 平均池化，80×80 會產生 5×5，不是 4×4。要支持動態 H、W，需要重新處理池化、候選生成和 decode，再逐尺寸測試；不能只在 `dynamic_axes` 為高寬取名字。

## 三層驗證，把差異定位到有用的位置

模型檔產生與 session 載入，還沒證明算出的結果一致。檢查從檔案開始，再進到 raw，最後到使用者看到的框；一層通過不能代替下一層。

**第一層：檔案結構。** `onnx.checker` 查格式、opset 中是否有該運算、輸入輸出個數、必填屬性等，例如 AveragePool 的 `kernel_shape`。本節用預設呼叫，不查每個運算的 dtype 和 shape 是否相容；那要 `full_check=True`。checker 也不比數值，所以檔案合法不代表算出的數和 PyTorch 一樣。

**第二層：raw 數值。** 同份輸入分別交給 PyTorch 與 ORT，逐值比較 B=1、2、3 的 raw。float32 約七位有效數字，運算每次都捨入；不同後端的加總順序與實作可能不同，不能要求逐位元完全相同。改用 allclose，讓每個值滿足

\[
|a-r|\le\text{atol}+\text{rtol}\times|r|.
\]

其中 a 是受檢查值、r 是參考值。atol（absolute tolerance，絕對容許差）提供固定額度，rtol（relative tolerance，相對容許差）按參考值大小增加額度。r=2.0、兩者都 `1e-5` 時，允許差為 0.00003。這裡 raw 就採 `atol=1e-5, rtol=1e-5`；既有紀錄的三種 batch 最大絕對差都約 `4.77e-7`，即 0.000000477，記在 `max_abs_raw_errors`。

??? note "加總順序為什麼會影響結果"

    數學上，三個數相加時先加哪兩個，結果都一樣，例如 \((a+b)+c=(a+c)+b\)；電腦的浮點數卻不一定。float32 在 \(10^8\) 附近，相鄰兩個能表示的數相差 8，所以 \(10^8+1\) 會被捨入回 \(10^8\)。取 \(a=10^8\)、\(b=1\)、\(c=-10^8\)：\((a+b)+c=(10^8+1)-10^8=0\)，但 \((a+c)+b=(10^8-10^8)+1=1\)。同樣三個數，先算哪兩個，答案就不同。卷積要把很多乘積加起來，加總順序一換，末幾位就可能改變。



**第三層：還原到原圖的框。** 兩份 raw 各走相同 decode、NMS、還原流程，再逐圖比框、score、類別。框以原圖畫素計，`atol=1e-4, rtol=1e-5`；分數 `atol=1e-6, rtol=1e-5`；類別完全相同。框座標數值較大，所以固定允許差比 raw 寬。

還要比這一層，是因為門檻會放大很小的差。假設同一候選兩端分數為 0.0500001、0.0499999，raw 可能很接近，score 門檻 0.05 卻讓一端保留、一端刪除；NMS 的 IoU 若接近 0.5，也可能改變去留。這是門檻機制的假設例，不是本次實測。

完整程式在每圈 raw 比完後接這段：

``` { .python data-excerpt="lesson_cases/20-deployment.py" }
# 兩份 raw 各自 decode、NMS、還原到原圖；每張原圖得到一組框、分數與類別
pytorch_predictions, ort_predictions = restore(torch_raw, metadata[:b]), restore(ort_raw, metadata[:b])
for a, o in zip(pytorch_predictions, ort_predictions):  # 一次比一張原圖
    torch.testing.assert_close(a['boxes'], o['boxes'], rtol=1e-5, atol=1e-4)
    torch.testing.assert_close(a['scores'], o['scores'], rtol=1e-5, atol=1e-6)
    assert torch.equal(a['labels'], o['labels'])  # 類別必須完全相同
# 兩個空結果也會通過上面的比較，所以每張原圖都至少要解出一個框
box_counts = [len(a['boxes']) for a in pytorch_predictions]
assert all(c > 0 for c in box_counts), f'B={b}: a source has no restored box {box_counts}'
```

最後兩行要求每張預期有候選的圖都真的解出框。否則兩端門檻太高、都得到空結果時，沒有任何框可比較，三項一致性檢查仍通過。這個斷言防的是空對空的驗證漏洞，不是在產品上要求所有圖片一定有物件。空圖本來可以都沒框，正式測試要另放空圖確認一致。

B=3 那圈 `restored_boxes_per_source=[16,16,16]`，16 格候選都過了 0.05，也沒被 NMS 刪除。這只保證測試有實際框可比，不能表示模型學會偵測：來源是雜訊，模型只更新一步，分數略高於門檻。80×48 與 48×80 兩張各有 8 個框完全在補邊，還原裁到原圖後寬或高為 0，因此有框可比也不表示框落在內容區。

這三張檢查了橫圖、直圖與三種 batch，仍沒有涵蓋所有輸入。實際交付還要放入空圖、小物件、極端長寬比、擁擠和門檻敏感案例，將差異記在它首先出現的階段。共同前後處理降低了兩端操作不一致的機會，但一致性本身不能證明共同的那份程式或模型品質正確。

## 確認結果後，量使用者要等的那段時間

我們想知道換 ORT 後整段圖片處理能省多少，而不只是 model 變快多少。完整程式在同一台 CPU、2 個執行緒上量 raw（模型本身），以及 RGB 前處理→raw→decode／NMS→還原框的端到端時間。兩者都不含讀寫磁碟、顯示、網路或相機佇列。

raw 每項先暖機 3 次，再連續量 20 次取中位數。端到端則兩條管線各暖機 3 次，量 40 輪：每輪兩端各跑一次，PyTorch 先與 ORT 先交替。兩次靠近、先後輪流，可減少電腦中途變忙或時脈改變只算到某一端的問題。各後端取自己 40 次的中位數，同輪 PyTorch−ORT 的 40 個差也另取中位數。

頁尾那次執行的數字如下，單位為毫秒：

| 項目 | PyTorch | ORT |
| --- | --- | --- |
| raw，batch 1 | 0.160 | 0.074 |
| 端到端，batch 1 | 2.529 | 2.331 |
| raw，batch 2 | 未量 | 0.068 |

raw 約快 2.2 倍（完整值 0.160267÷0.074453≈2.15），但本節沒有驗證 ORT 哪個實作造成加速。模型時間只約占 PyTorch 端到端 6.3%（0.160÷2.529）；相減約 2.37 毫秒，是前後處理成本的粗估，並非逐段量測。

成對的時間差記在 report 的 `median_ms` 字典中，欄位叫 `paired_torch_minus_ort_preprocess_to_restored_boxes`，這次為 0.162501 毫秒；正值表示 ORT 較快。這是 40 個同輪差的中位數，不必等於表中兩個端到端中位數相減。report 的 `median_ms` 字典與程式同名的計時函式不是同一個東西。

??? note "為什麼不能拿表中兩格相減"

    兩個中位數的差，一般不等於每一輪差值的中位數。舉一個只有 3 輪的假設例子（單位：毫秒）：

    | 輪 | PyTorch | ORT | 同一輪的差 |
    |---|---|---|---|
    | 1 | 2 | 4 | −2 |
    | 2 | 3 | 1 | 2 |
    | 3 | 6 | 5 | 1 |

    PyTorch 的中位數是 3（第 2 輪），ORT 的中位數是 4（第 1 輪），相減得 −1，看起來 ORT 比較慢；可是三輪裡有兩輪是 ORT 比較快，三個差 −2、2、1 的中位數是 1。兩個中位數取自不同的輪，那兩輪電腦的狀況不一定一樣；同一輪相減，比的是緊接著發生、條件差不多的兩次執行。



0.162501÷2.529305≈6.4%，可粗估本次整段觀察到的差距。raw 中位數的差是 0.160267−0.074453≈0.086 毫秒，和成對整段差約 0.163 不必相等：raw 測試連續只跑模型，管線中模型前後夾著其他工作，執行條件不同。本節沒有分別量各階段受了多少影響，不能把整段差全部歸到模型。

這次模型 raw 快約 2.2 倍，整段卻只見約 6.4% 的差，說明前後處理不可忽略。這是一台裝置的一次觀察，成對量只能減少共同變化；若干擾只拖慢其中一條，端到端甚至可能 ORT 較慢。重新執行看當次 `report.json` 的兩端中位數、成對差與 `end_to_end_paired_rounds=40`，不能把 raw speedup（原時間÷新時間）當成產品總加速。

### batch 能提高吞吐，也可能讓請求等更久

服務（serving）把模型放在伺服器上，接收圖片並回傳結果。現在要選每來一張立刻算，還是等幾張一起算。一次 B=2 的 batch latency 是算完兩張的時間；以典型 raw 時間粗估 throughput（張／秒）是 B÷時間秒數，持續吞吐仍要用總張數÷總耗時實測。

用 ORT 完整數字，B=1 約 1000÷0.0744525=13,431 張／秒；B=2 約 2000÷0.067778=29,508 張／秒，約 2.2 倍。這只含模型 raw，不含前後處理或等待湊 batch。

假設每 10 毫秒才來一張，第一張等第二張就多等約 10 毫秒；兩張分開算 2×0.074453，一起算 0.067778，省的只有約 0.081 毫秒。因此離線大量處理可能重視吞吐，低流量即時請求則可能被湊 batch 拖慢。這和影片排隊一樣，要按需求決定何時等待、何時立即處理。

## 在 CPU 上完成自己的匯出驗證

本機依 [README 環境步驟](https://github.com/birdhackor/learn_to_yolo#readme)安裝固定套件，在 repo 根目錄 `PYTHONPATH=. python lesson_cases/20-deployment.py`；Colab 先跑環境格，再完整實驗。需要 `onnx==1.19.1`、`onnxruntime==1.23.2`，既有頁尾紀錄是 PyTorch 2.9.1 CPU，當次版本也存入 report 的 `versions`。

核對 `artifacts/lesson-20/grid.onnx` 確實產生、checker 通過、ORT 真正執行 B=1、2、3、raw 與還原框比較通過，而且每圈每張測試來源有框。能 import 套件或印出 provider 名稱只證明安裝與設定，不代表這些檢查完成。交付時把模型檔、契約、比對範圍、版本與時間範圍放在一起，接收者才能重現。

## 自主練習

練習都在完整程式上做：在 Colab 裡改「本節可修改的完整實驗」下面那一格；本機則改 `lesson_cases/20-deployment.py`，兩者是同一份程式。會用到的是下面這幾行（摘自完整程式，`...` 表示省略的部分）：

``` { .python data-excerpt="lesson_cases/20-deployment.py" }
# 三張來源；array shape 依序是高、寬、通道
sources = [rng.integers(0, 256, shape, dtype=np.uint8)
           for shape in ((64, 96, 3), (48, 80, 3), (80, 48, 3))]
processed = [preprocess(rgb) for rgb in sources]      # 每個元素是 (letterbox 後的圖, metadata)
batch = torch.stack([item[0] for item in processed])  # 取出每張圖，疊成一個 tensor
metadata = [item[1] for item in processed]            # 取出每張圖的 letterbox 資訊，還原框時用
...
with torch.no_grad():  # 不記錄計算圖；少了這行，model(...).numpy() 會報 RuntimeError
    for b in batches_checked:  # batches_checked 是 [1, 2, 3]
        assert batch[:b].shape == (b, 3, 64, 64)
        torch_raw = model(batch[:b]).numpy()
        ort_raw = session.run(['raw_grid'], {'images': batch[:b].numpy()})[0]
        assert torch_raw.shape == ort_raw.shape == (b, 4, 4, 7)
        ...
```

**練習 1**：先預測再執行。`processed` 和 `metadata` 各有幾份？`batch` 疊了幾張圖？可以在 `metadata = ...` 那行下面加一行 `print(len(processed), len(metadata), batch.shape)` 核對（縮排和那行對齊）。B=3 那一圈，`batch[:b]` 和 `ort_raw` 的 shape 各是多少？輸出裡的 `batches_checked` 應該是什麼？

??? note "參考答案"

    `processed` 和 `metadata` 都是三份，每張來源一份；`batch` 疊了三張圖。加的那行印出 `3 3 torch.Size([3, 3, 64, 64])`。B=3 那一圈，`batch[:3]` 是 `[3,3,64,64]`，`ort_raw` 是 `[3,4,4,7]`，兩個斷言都通過。輸出的 `batches_checked` 是 `[1, 2, 3]`，`max_abs_raw_errors` 有三個數（每種 batch 一個），`restored_boxes_per_source` 也有三個數（B=3 那一圈每張原圖一個）。最後一行印出 `real ONNX export + checker + ORT CPU + restored-box parity completed`，表示 raw 與三張原圖的還原框都比對通過，每張原圖也都解出了框。

**練習 2**：把 `sources` 裡的 `(80, 48, 3)` 刪掉，只留兩張來源，其他程式都不改。執行前先預測：哪一行斷言會失敗？在 B 等於多少時失敗？

??? note "參考答案"

    B=1、2 都通過；到 B=3 那一圈，`assert batch[:b].shape == (b, 3, 64, 64)` 失敗。只有 2 張時，`batch[:3]` 不會報錯，仍回傳 2 張，shape 是 `(2,3,64,64)`，不等於 `(3,3,64,64)`。這個斷言就是用來擋下「只有兩張，卻當成 B=3 測試」。做完記得把第三張加回去。

**練習 3**：三張來源都在的情況下，把 `preprocess` 函式裡 `letterbox(image, torch.empty(0, 4), size=64)` 的 `size=64` 改成 `size=80`。哪個檢查會先擋下：ORT 拒絕輸入，還是 Python 的斷言？

??? note "參考答案"

    Python 的斷言先擋下。輸入變成 `(1,3,80,80)`，所以 B=1 那一圈的 `assert batch[:b].shape == (b, 3, 64, 64)` 就失敗了，程式還沒走到 ORT。就算拿掉這個斷言，ORT 也會拒絕 80×80 的輸入；完整程式在迴圈之後那段 try／except，檢查的就是這件事。要支援 80×80，得重新設計匯出與 decode（見上方「動態 batch 不是動態空間」）；不要放寬或刪掉斷言，掩蓋規格不符的錯誤。

**練習 4**：參考值是 3.2、`atol=1e-5`、`rtol=1e-5` 時，受檢查的值最多可以差多少，仍算通過？

??? note "參考答案"

    \(10^{-5}+10^{-5}\times3.2=4.2\times10^{-5}\)，也就是 0.000042。差值不超過 0.000042 就通過；參考值的絕對值越大，允許差也越大。

**練習 5**：三張來源都在的情況下，把 `restore` 函式裡 `decode_grid(...)` 的 `score_threshold=.05` 改成 `score_threshold=.07`。先預測：第三層的三個比較（框、分數、類別）會不會失敗？哪一行斷言會擋下，在 B 等於多少時？

??? note "參考答案"

    第三層的三個比較都不會失敗；擋下的是框數的斷言。B=1 那一圈，`assert all(c > 0 for c in box_counts)` 失敗，訊息是 `B=1: a source has no restored box [0]`。門檻 0.07 比每個候選的分數都高，兩邊的候選全部被丟掉，都是空結果；兩個空結果逐一比較時找不到任何差異，所以兩個 `assert_close` 和 `torch.equal` 都通過。這正是框數斷言要擋下的情況。做完記得改回 `.05`。



## 有 NVIDIA GPU 時，換成 TensorRT

CPU 的 ONNX 路徑已經能執行、比對。若使用端是 NVIDIA GPU，可再用 TensorRT 建置：它讀入 ONNX，替各層挑選適合該 GPU 的實作，得到 engine（由 TensorRT 載入執行的最佳化模型，不能直接當 .exe）。它改變的是執行安排，輸入輸出契約與一致性檢查仍要保留。

TensorRT 需要可用的 NVIDIA GPU、相容 driver、CUDA（Compute Unified Device Architecture，NVIDIA 的 GPU 計算平台）與 TensorRT 版本。Colab 本節跑的是 CPU 案例，沒有執行這條 GPU 路徑。engine 也通常受 GPU 架構與 runtime（載入執行它的程式庫）版本限制，不能把它當跨裝置通用模型檔。

### 精度設定會改變哪些數值

加速的一種方式是使用較低精度的數值格式，節省記憶體與運算成本。要先知道允許的捨入範圍，再比對 raw、框與門檻敏感案例：

| 格式 | 是什麼 | 精度與範圍 |
|---|---|---|
| FP32 | 32 位元浮點數，就是前面一直用的 float32 | 約 7 位有效數字 |
| TF32（TensorFloat-32，張量浮點 32 格式） | Ampere 架構（NVIDIA 顯示卡的一個世代）起的 NVIDIA GPU（L4 也包含在內）做卷積、矩陣乘法時可用的格式：相乘前先把 FP32 的輸入捨入成 TF32，乘積仍用 FP32 加總 | 範圍和 FP32 相同；尾數只有 10 位（FP32 有 23 位），精度和 FP16 相當 |
| FP16 | 16 位元浮點數，比較省記憶體，在支援的 GPU 上常比較快 | 約 3 位有效數字；能表示的最大值是 65504 |
| INT8 | 8 位元整數。把數值改用整數表示，叫量化 | 有號 8 位元只有 −128～127 這 256 個整數；要先經校準或量化感知訓練等流程決定縮放。校準會用有代表性的圖片量各層數值的分布；量化感知訓練則在訓練中模擬量化誤差 |



FP16 不只要求原輸出近似；分數或 IoU 靠近門檻時，微小改變就可能增減框，還要重新評估 AP。INT8 需要代表性的校準或量化感知訓練，不能只加一個開關就期待品質不變。

### 建 engine 時，要宣告允許的 shape

`trtexec` 是 TensorRT 附的命令列工具（CLI，command-line interface）。下面三條是對照操作的示意，**沒有在任何機器上執行過**。既有 L4 實測用的是 TensorRT 10.13.3.9 Python API，結果在下一小節。

```bash
trtexec --onnx=artifacts/lesson-20/grid.onnx --saveEngine=grid-fp32.engine --noTF32 \
  --minShapes=images:1x3x64x64 --optShapes=images:2x3x64x64 --maxShapes=images:4x3x64x64
trtexec --onnx=artifacts/lesson-20/grid.onnx --saveEngine=grid-fp16.engine --fp16 --noTF32 \
  --minShapes=images:1x3x64x64 --optShapes=images:2x3x64x64 --maxShapes=images:4x3x64x64
trtexec --loadEngine=grid-fp16.engine --shapes=images:1x3x64x64
```

依序是讀 ONNX 建 FP32 engine、讀同份 ONNX 建允許 FP16 的 engine、載入 FP16 engine 以 B=1 執行。兩次建置都 `--noTF32`，停用可能預設允許的 TF32 乘法；輸入是 float32、檔名寫 fp32，並不能保證全部用完整 FP32 運算。

行尾 `\` 將命令接到下一行。`images` 對應匯出時的輸入名，`1x3x64x64` 是 `[B,3,H,W]`。min／opt／maxShapes 合成 optimization profile（最佳化設定）：min、max 宣告可用範圍 B=1～4、高寬固定 64；opt=B=2，是建置時選最快實作主要調校的 shape。執行時仍須使用範圍內的 shape。

先記錄 GPU、driver、CUDA、TensorRT 與精度設定，使用已通過 ORT 比對的 ONNX。再依實際版本確認命令旗標與 Conv、Relu 等運算子受支援。TensorRT 10.12 起，`--fp16`／`BuilderFlag.FP16` 這種逐層自選精度做法已棄用；10.13 仍可用，官方推薦 strongly typed network（強型別網路，`--stronglyTyped`），各 tensor 按 ONNX 宣告的型別運算，例如先匯出 FP16 ONNX。本頁示意與 L4 實測沿用 10.13 的舊做法，升版要重新確認。

先以 FP32、相同 RGB 輸入比 raw 和 decode 後結果，再測 FP16。`trtexec` 時間不會自動包含本書 letterbox／NMS，端到端需另外量。這樣才能知道改成 GPU engine 後，真正使用的流程改善多少。

### GPU 計時要等運算完成

CPU 把工作 enqueue（放進 GPU 待辦佇列）後會立刻繼續，不等算完。這個佇列叫 CUDA stream，與[第 18 章](18-video.md)的影片串流不是同一概念。若排完就停錶，只量到提交工作時間。

用 CPU 計時器時，要在讀時間前以 `torch.cuda.synchronize()` 等 GPU 完成；用 CUDA events 則在 GPU 佇列記兩個時間戳，讀差值前仍要等結束 event 完成，例如 event 的 `synchronize()`。本節 CPU 的 PyTorch／ORT 算完才返回，沒有這個問題。

## L4 GPU 上的 TensorRT 實測 { #l4-results }

既有實驗透過 GitHub Actions 啟動 Modal，租用 NVIDIA L4 資料中心 GPU，以 TensorRT Python API 建兩個 engine：FP32、TF32 停用，以及允許 FP16、TF32 停用。每個都實際跑 B=1、2、3、4，和同張 L4 上 PyTorch 比 raw、decode 框、類別、候選順序、分數；CPU ORT 也先與該份 PyTorch 比過。所有比較通過、decode 確實有框，並非空對空。

輸入來自 8 張 64×64 合成圖，沒有 letterbox，所以框差以該畫面的畫素計。下表取 B=1～4 中最大的差；時間只量 B=1，20 次中位數，包含每次執行設定與同步：

手機上可左右滑動表格，查看完整欄位。

| 建置設定 | raw 最大絕對差 | 框最大差（畫素） | raw B=1 中位數 |
| --- | --- | --- | --- |
| FP32、TF32 停用 | 7.63e−6 | 3.81e−6 | 0.147 ms |
| 允許 FP16、TF32 停用 | 7.63e−6 | 3.81e−6 | 0.147 ms |

允許 FP16 不等於每層都是 FP16。TensorRT builder（建 engine 的元件）試不同 tactic（實作方法）選快者；flag 只允許用 FP16，若 FP32 更快或沒有 FP16 實作，仍可選 FP32、混用精度。兩列差與時間近似，可能多數層仍是 FP32，但沒有逐層精度稽核，不能確認，也不能由這張表說 FP16 提速。它是建置旗標的比較，不是強制全 FP16 的品質／速度比較；INT8 未測。

L4 與 CPU 案例同架構、15,511 參數，但前者 Adam 40 步、後者 SGD 1 步，權重不同，輸出不能對照；硬體、後端、計時範圍也不同，速度不能比。L4 的 0.147 ms 包含設定與同步，不是純 GPU 計算時間，兩邊都沒評 AP。

??? note "實驗紀錄與重跑方式（可略過）"

    - **實測紀錄**：[Actions 實測 37217013901](https://github.com/birdhackor/learn_to_yolo/actions/runs/37217013901)。版本是 PyTorch 2.9.1+cu128（+cu128 表示這版 PyTorch 是針對 CUDA 12.8 編譯的）／CUDA build 12.8、TensorRT 10.13.3.9，GPU 是 NVIDIA L4。模型在固定 8 張合成圖上用 Adam 更新 40 次（loss 0.9848→0.0710）後匯出。
    - **Python 介面的步驟**：建置出 engine 後，把它反序列化（deserialize：把存成位元組的 engine 還原成能執行的物件），再用 `set_input_shape`（設定這次輸入的 shape）、`set_tensor_address`（告訴 engine 輸入、輸出放在 GPU 記憶體的哪裡）、`execute_async_v3`（把計算排進 GPU 的佇列）執行。optimization profile 是 B=1～4、固定 64×64。
    - **計時範圍**：先暖機 3 次，再取 20 次的中位數。包含 shape／address 設定、輸出 buffer（存放輸出的記憶體）配置、enqueue 與 stream／device 同步（等 GPU 做完）；不含前處理、decode、CPU 拷貝、engine 建置與 container（容器：打包好程式與執行環境的獨立執行單位）啟動。Actions client（在 GitHub Actions 上呼叫 Modal 的程式）的總時間 39.20 秒，則包含建置、啟動與傳輸。這個小模型的單次測試，不能當成正式訓練或服務的效能。
    - **檔案保存**：ONNX 與兩份 engine 存在專案專用的 Modal Volume（Modal 的雲端儲存空間）`projects/learn-to-yolo/deployment/github-37217013901-1-463d3f59fb80/`。明確 commit（正式寫入，讓其他容器讀得到）後，由另一個 CPU container 重新讀取，三個檔案的 SHA-256（由檔案內容算出的指紋，內容改一點就會不同）都相符。模型與 engine 沒有放進普通的 Git。
    - **重跑方式**：[完整 JSON](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/deployment-gpu.json)。重跑入口是手動觸發的 workflow（GitHub Actions 的自動化流程）`deployment-gpu.yml`；一般的 push（上傳提交）或 PR（請求合併修改）不會啟動 GPU。

參考來源：[PyTorch 2.9 ONNX 官方文件](https://docs.pytorch.org/docs/2.9/onnx.html)、[ONNX Runtime Python 入門](https://onnxruntime.ai/docs/get-started/with-python.html)、[TensorRT 10.13 官方 trtexec 範例](https://github.com/NVIDIA/TensorRT/blob/b8db91e15be2cae4465ac17fab19e0f969e45407/samples/trtexec/README.md#example-3-running-an-onnx-model-with-full-dimensions-and-dynamic-shapes)、[flags 定義](https://github.com/NVIDIA/TensorRT/blob/b8db91e15be2cae4465ac17fab19e0f969e45407/samples/common/sampleOptions.cpp)與 [BuilderFlag 的棄用註記](https://github.com/NVIDIA/TensorRT/blob/b8db91e15be2cae4465ac17fab19e0f969e45407/include/NvInfer.h#L8848-L8850)。

<!-- curriculum-evidence:start -->

## 實際執行紀錄

本節的完整程式於 2026-10-08 在 AMD EPYC 9V74 80-Core Processor（2 個執行緒）上用 PyTorch 2.9.1+cpu 執行，程式裡的 assert 全部通過。下面是那次印出的原始輸出；輸出裡若有計時或訓練得到的數字，換一台電腦會略有不同。每個數字的意思，以本頁正文的說明為準。[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/20-deployment.json)

??? example "展開本次實際輸出"

    ```text
    {
      "onnx": "artifacts/lesson-20/grid.onnx",
      "opset": 17,
      "providers": [
        "CPUExecutionProvider"
      ],
      "input_shape": [
        "batch",
        3,
        64,
        64
      ],
      "batches_checked": [
        1,
        2,
        3
      ],
      "max_abs_raw_errors": [
        4.76837158203125e-07,
        4.76837158203125e-07,
        4.76837158203125e-07
      ],
      "decoded_boxes_scores_labels_match": true,
      "restored_boxes_per_source": [
        16,
        16,
        16
      ],
      "dynamic_spatial": false,
      "spatial80_rejected": true,
      "median_ms": {
        "torch_raw_batch1": 0.16026700905058533,
        "ort_raw_batch1": 0.07445250230375677,
        "torch_preprocess_to_restored_boxes": 2.529304998461157,
        "ort_preprocess_to_restored_boxes": 2.3314560021390207,
        "paired_torch_minus_ort_preprocess_to_restored_boxes": 0.1625005024834536,
        "ort_raw_batch2": 0.06777750240871683
      },
      "end_to_end_paired_rounds": 40,
      "ort_raw_batch2_images_per_second": 29508.31660835559,
      "versions": {
        "torch": "2.9.1+cpu",
        "onnx": "1.19.1",
        "onnxruntime": "1.23.2"
      },
      "limits": "CPU float32 parity; one training step is not detection-quality evidence; this CPU case does not run TensorRT/GPU; separate L4 evidence is documented"
    }
    real ONNX export + checker + ORT CPU + restored-box parity completed
    ```

<!-- curriculum-evidence:end -->

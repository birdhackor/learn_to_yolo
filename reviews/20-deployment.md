# 審查紀錄：ONNX／TensorRT

審查範圍：`docs/lessons/20-deployment.md`、頁面上的圖（`docs/assets/diagrams/20-deployment.svg`），以及 `lesson_cases/20-deployment.py` 與它 import 的 repo 模組；頁尾自動產生的執行紀錄區塊不在範圍內，由 `scripts/validate_curriculum_evidence.py` 對照紀錄檢查。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面依目前的程式改寫後，由另一位 AI 獨立查核：在獨立的副本執行該節程式、照頁面做練習，逐句對照程式、執行紀錄與手算，檢查程式摘錄與網頁轉換，並從初學讀者（高中程度、數學好、程式新手）的角度看用詞與說明順序。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：有必要問題

第 1 輪查核不通過：有 1 項 must、4 項建議。所有執行都在暫存副本裡進行，repo 裡沒有執行任何程式或建置；repo 的 artifacts/lesson-20 與 site 的修改時間都早於這次編修。

程式與練習：
- 完整程式 exit 0，`restored_boxes_per_source` 是 [16, 16, 16]。
- 另外用暫存副本核對：每張原圖 16 個候選，分數 0.0600–0.0610，NMS 一個都沒刪。80×48 和 48×80 各有 8 個框整個落在補邊上，還原後高或寬變成 0；96×64 沒有這種框。頁面這段說法都正確。
- 練習 5 照頁面把 .05 改成 .07：得到 `AssertionError: B=1: a source has no restored box [0]`，三個比較都先通過，和參考答案一致。
- 練習 1 印出 `3 3 torch.Size([3, 3, 64, 64])`。
- 練習 2 在 B=3 的 shape 斷言失敗。
- 練習 3 在 B=1 的 shape 斷言失敗；拿掉那行斷言後，ORT 丟出 InvalidArgument（Got 80, Expected 64）。
- 計時流程的描述和 `median_ms`、`paired_median_ms` 一致：暖機 3 次、量 20 次、40 輪、第 1 輪 PyTorch 先跑、取每輪差值的中位數。3 輪的假設例子算得對。

先前審查意見與受程式改動影響的段落：10 條先前審查意見、13 條受程式改動影響的段落都已處理。
- 「本輪」全部刪除。
- 原本「沒檢查框數」的補丁說明已換成有標記的摘錄。
- L4 標題已改，錨點不變。
- 兩個多餘條目已刪。
- 執行紀錄區塊和 HEAD 逐位元相同。
- 第 3 行的 tag 在 HEAD 就已是 lessons-v0.4.0。
- 頁面找不到殘留的舊數字或舊 run 編號。

數字：
- L4 的數值都和 deployment-gpu.json 一致：7.63e−6、3.81e−6、0.147 ms、39.20 秒、Volume 路徑、loss 0.9848→0.0710、15,511 個參數。
- 紀錄綁定的 12 個檔案，SHA-256 都和目前檔案相同。
- 用 gh 確認 run 37217013901 成功，headSha 是 463d3f5。
- 頁面沒有出現本機量到的數字。

摘錄：摘錄比對工具在 repo 與暫存副本都輸出 []。兩個標記區塊的每一行在程式裡連續出現、順序相同、縮排一致。

呈現：
- zensical build --clean --strict 與 validate_site.py 都通過。
- 新加的 note 裡的表格正常呈現。
- SVG 沒有改動，有 viewBox、title、desc，用 qlmanage 預覽正常，內容與程式相符。
- 這次只改了頁面這一個檔案。

must：第 165 行用 no_grad 與 .numpy() 解釋「同一輪的差」和 raw 差距之間的落差。實測這兩件事約 0.002 毫秒，但落差有好幾個百分之一毫秒；落差主要來自模型那一步在管線裡的執行條件不同。照現在的寫法，讀者會學錯時間花在哪裡。

should 共 4 項：
- 簡化區塊的引言沒提到省略了輸入規格的斷言。
- 「成對量能處理別的程式搶 CPU」和「忙碌時仍可能變號」看起來互相矛盾。
- 成對差的數值與關係要等紀錄重產後補上；0.117 要列入待重產清單，並改用完整位數計算。
- `median_ms` 同時是函式名和 report 欄位名，容易混淆。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 必要 | docs/lessons/20-deployment.md 第 165 行（「怎麼讀這張表」第三點：「兩者不必相等：管線裡 PyTorch 那一步還多了進入 `torch.no_grad()` 和 `.numpy()` 轉換，量 raw 時沒有這兩件事。」） | 這句把「同一輪的差不等於 raw 的差距」歸因於 no_grad 和 .numpy()。可是這兩件事的成本很小。暫存副本實測：進出 `torch.no_grad()` 加上對 [1,4,4,7] 呼叫 `.numpy()` 約 0.002 毫秒；`model(x)` 是 188.5 µs，`with torch.no_grad(): model(x).numpy()` 是 190.6 µs。實際的落差卻有好幾個百分之一毫秒：暫存副本跑完整程式時，成對差 0.182、raw 差距 0.106；程式查核那 8 次，成對差 0.186–0.243，raw 差距 0.145–0.177。我另外把模型那一步夾在前處理與還原之間量，兩個 backend 都比連續呼叫時慢：PyTorch 多約 0.027 毫秒，ORT 多約 0.007–0.013 毫秒。這組量法兩邊都含 no_grad 與 .numpy()，所以落差主要來自執行條件不同，不是來自這兩個呼叫。紀錄重產後，讀者會拿頁尾的成對差和 0.117 比，以為多出的幾十微秒都是 no_grad 和 .numpy() 的成本。「時間花在哪裡」正是本段要教的事，讀者會因此學錯。 |
| 2 | 建議 | docs/lessons/20-deployment.md 第 51 行（簡化區塊的引言「和完整程式不同的地方（檔名、樣本的名字、執行緒設定），註解裡都有寫」） | 區塊省略了完整程式建立 session 後的 `assert session.get_inputs()[0].shape == ['batch', 3, 64, 64]`，註解和引言都沒提到。引言卻說不同的地方「註解裡都有寫」，這句話不完整。 |
| 3 | 建議 | docs/lessons/20-deployment.md 第 137 行與第 166 行 | 第 137 行把「別的程式開始搶 CPU」當作成對量能處理的例子。第 166 行又說電腦忙著跑別的程式時，結果仍可能改變，甚至變號。兩句之間沒有說明，初學讀者會覺得前後矛盾：成對量到底能不能處理別的程式？ |
| 4 | 建議 | docs/lessons/20-deployment.md 第 147、165–166 行 | 受程式改動影響的段落第 121、127–129 條要求寫出紀錄裡成對差的實際數值，以及它和 raw 差距的關係。頁面目前只寫欄位名稱和比較方法，因為現有紀錄還沒有這一項；編者在報告裡已註明這一點。另外，第 165 行的 0.117（0.153−0.036）是由表中數值算出來的，在編者的處理清單裡沒有明確標成待重產。它也是用四捨五入後的數相減；照頁面其他地方的做法用完整位數算，是 0.15258−0.03642≈0.116。 |
| 5 | 建議 | docs/lessons/20-deployment.md 第 134 行與第 147 行 | 第 134 行的 `median_ms` 是函式名稱：暖機 3 次後量 20 次。第 147 行的「頁尾紀錄 `median_ms` 裡的 `paired_...`」指的卻是 report 裡同名的欄位。初學讀者可能以為成對差也是 `median_ms` 函式量出來的。 |

修正必要問題時處理的項目（修正後由第 2 次複查確認）：

- trace line 3 （版本標籤過期）：保留不改。HEAD 的第 3 行已經是 blob/lessons-v0.4.0/notebooks/20-deployment.ipynb。
- trace line 95「本輪三種 batch…」：已改寫成「頁尾紀錄中，三種 batch 的最大絕對差都約 `4.77e-7`…；report 的 `max_abs_raw_errors` 記下了這三個數」。數字沿用現有紀錄，已列入待重產清單。
- trace line 98「（不是本輪實測）」：已改寫成「舉一個假設的情境（不是實測結果）」。
- trace line 100「程式沒有檢查解出的框數大於 0…」：補丁說明已刪除，換成三部分：有 data-excerpt 標記的第三層比較與框數斷言摘錄；說明為什麼兩個空結果也會通過、以及每圈每張原圖都要有框的段落；`restored_boxes_per_source` 的段落。最後這段寫明：值是 `[16, 16, 16]`、只記錄 B=3 那一圈、這些框不是偵測結果、80×48 與 48×80 各有 8 個框落在補邊上，還原後寬或高變成 0。「實際部署時…」兩句保留。
- trace line 116「本輪實測如下」：已改寫成「頁尾紀錄那次執行的實測如下」。表中數值沿用現有紀錄，已列入待重產清單。
- trace line 129「（同一段程式其他次執行的紀錄裡就出現過）」：已刪除。舊的計時波動整段改寫成以成對差為主軸。
- trace line 140「本輪用的是…」：已改寫成「頁尾紀錄用的是 ONNX 1.19.1、ORT 1.23.2、PyTorch 2.9.1 CPU 版，report 的 `versions` 欄也記下了這些版本」。
- trace line 247 標題：已改成「## L4 GPU 上的 TensorRT 實測 { #l4-results }」，錨點不變，validate_site 的錨點檢查通過。
- trace line 268「雲端資源：Modal 已停止…」：整條已刪除。
- trace line 270「兩份紀錄的範圍…」：整條已刪除。
- SVG docs/assets/diagrams/20-deployment.svg：沒有先前審查意見，總評是不受影響，所以沒有改動。
- impact line 100：處理方式同上面第 100 行的先前審查意見。另外依框數斷言的查核補上「8 個框面積為 0」的限制，並在暫存副本重跑 probe 確認。
- impact line 114（計時量法）：已改寫成兩點。raw 用 `median_ms` 函式：暖機 3 次，再連續量 20 次。端到端用 `paired_median_ms`：兩條管線先各空跑 3 次，再量 40 輪；第 1 輪 PyTorch 先、第 2 輪 ORT 先，依此交替；各取 40 次的中位數，40 個差也取中位數。另加一段說明為什麼要成對量。不含磁碟、顯示與相機佇列的那句保留。
- impact line 121（表格）：三列都沿用現有紀錄的值，已列入待重產清單。表下說明成對差記在紀錄的哪個欄位，也釐清 `median_ms` 欄和同名函式的差別。數值要等紀錄重產後才能填，已列入清單。另附一個 3 輪的假設例子，說明為什麼不能拿表中兩格相減。
- impact line 126（4.2 倍）：算法不變，沿用現有紀錄的完整位數（0.15258÷0.03642≈4.19），已列入待重產清單。
- impact line 127（ORT 少了約 7.9%）：已刪除。改成用同一輪的差和 raw 差距對照，並說明把成對差除以 PyTorch 端到端的中位數，就能估計整條流程少掉幾成。
- impact line 128（6.6%、2.18 毫秒、4.2 倍）：沿用現有紀錄的值，已列入待重產清單。「整條流程只少約 7.9%」改成「整條流程只少掉模型那一小段的差距」。
- impact line 129：已改寫成以成對差為主軸。raw 差距改用完整位數，約 0.116 毫秒。兩者不必相等的理由改成執行條件不同，這是本輪的必要，no_grad 和 .numpy() 的歸因已刪除。計時波動的說明和括號都已刪除。保留一句「一台電腦上一次執行」的提醒，並補上成對量只能抵消兩條管線一起受到的變化。結論「ORT 的 raw 明顯比較快、模型只占端到端的一小部分」保留。
- impact line 131：report.json 那句已補上 `paired_torch_minus_ort_preprocess_to_restored_boxes` 與 `end_to_end_paired_rounds`（40）。
- impact line 135（0.036 毫秒、27,454 張／秒）：沿用現有紀錄的值，已列入待重產清單。
- impact line 136（0.043 毫秒、46,966 張／秒）：沿用現有紀錄的值，已列入待重產清單。
- impact line 138（1.7 倍、0.03 毫秒、2×0.036、0.043）：沿用現有紀錄的值，已列入待重產清單。每 10 毫秒來一張的假設情境保留。
- impact line 140（成功條件）：已補上「而且每一圈的每張原圖都至少解出一個框」。
- impact line 276（執行紀錄區塊）：沒有手動修改，cmp 確認和 HEAD 相同，由 verify_curriculum.py 重產。
- 查核者的必要問題（第 165 行的 no_grad／.numpy() 歸因）：已改寫，詳見上面受程式改動影響的段落 line 129。頁面沒有任何本機數字。
- 查核者的建議（第 51 行簡化區塊的引言）：已補上「另外省略了建立 session 後確認輸入規格的斷言（見下方「動態 batch 不是動態空間」）」。
- 查核者的建議（第 137 行與第 166 行看似矛盾）：第四點已補上「成對量只能抵消兩條管線一起受到的變化；電腦同時忙著跑別的程式時，如果干擾剛好拖慢其中一條，結果仍可能改變」。
- 查核者的建議（成對差的數值、0.117）：0.117 已改成完整位數算出的 0.15258−0.03642≈0.116。成對差的值，以及它和 raw 差距的關係，現有紀錄裡沒有，已列入待重錄數值清單，紀錄重產後填入。
- 查核者的建議（`median_ms` 同時是函式名和欄位名）：表下那句已寫明，紀錄裡的 `median_ms` 欄放所有計時的中位數（表中的數字也在這裡），和同名函式不是同一個東西；同一輪的差由 `paired_median_ms` 算出。

### 第 2 次複查：通過

第 2 輪查核通過，沒有必要，只有 1 項建議。所有程式都在我自己的暫存副本裡執行，repo 裡沒有執行任何東西。

1. 程式敘述都和 lesson_cases/20-deployment.py 相符。
- lesson 結束碼 0；`restored_boxes_per_source` 是 [16, 16, 16]，`end_to_end_paired_rounds` 是 40。
- 計時說明正確：`paired_median_ms` 先兩邊各暖機 3 次，第 1 輪 PyTorch 先、第 2 輪 ORT 先；`median_ms` 是暖機 3 次、計時 20 次。
- 四個練習都照頁面的指示實際改過再執行：
  - 練習 1 印出 `3 3 torch.Size([3, 3, 64, 64])`。
  - 練習 2 在 B=3 那圈的 shape 斷言失敗，此時 `batch[:3]` 是 (2,3,64,64)。
  - 練習 3 在 B=1 失敗；再拿掉那個斷言，ORT 報 InvalidArgument，訊息是 index 2、3 拿到 80、預期 64。
  - 練習 5 得到 `AssertionError: B=1: a source has no restored box [0]`，三個比較都先通過了。
- 另外把框數那兩行刪掉、門檻設成 .07：整支程式照樣跑完並印出 completed，`restored_boxes_per_source` 是 []。這證實頁面「少了這兩行…照樣通過」的說法。
- probe 結果：分數都在 0.0600–0.0609，類別全部是 0。80×48 有 8 個框高為 0，48×80 有 8 個框寬為 0，96×64 沒有。這些框離內容區至少 0.27 畫素，不是浮點誤差剛好碰到邊界，所以「8 個框」在任何機器上都一樣，不算 Mac 數字。
- 其他：參數 15,511 個；ONNX 裡是 2×2 的 AveragePool；2.9.1 的匯出器預設 dynamo=True；DeprecationWarning 的開頭和頁面寫的一樣。

2. inventory 10 條、受程式改動影響的段落 13 條全部處理。舊的「沒檢查框數」說明已移除。全頁沒有修訂或審查的敘述；「這次」只出現在指 L4 那次執行的地方。

3. 數字
- 用頁尾紀錄的完整位數重算 4.19、6.6%、2.18、0.116、27,454、46,966、1.7、0.03，全部正確。頁面沒有 Mac 數字；等紀錄重產的值都在編輯的清單裡。
- diff 裡 L4 段的數字也都對得上 artifacts/checks/curriculum/deployment-gpu.json：7.63e−6、3.81e−6、0.147 ms、run 37217013901、39.20 秒、Volume 路徑、loss 0.9848→0.0710、版本，以及三個檔案的 SHA-256 核對。這些改動不在編輯這輪的報告裡。這份 GPU 紀錄綁定的 12 個程式檔雜湊和工作樹完全相同，所以這些數字不會再變。

4. 摘錄
- 摘錄比對工具輸出 `docs/lessons/20-deployment.md []`。兩個有標記的區塊，行的順序和相鄰關係也和程式一致；簡化區塊在引言裡說明了；正文沒有程式行號。
- validate_lessons 只因全站每一頁都還沒有審查紀錄而失敗，這是發布前的預期狀態；它的摘錄檢查和程式碼解析檢查都通過。

5. 第 165 行把差距歸因於「執行條件不同」：條件確實不同，本機實測也看到模型步驟夾在前後處理之間時比連續呼叫慢。管線裡 PyTorch 那一步還多了每次進入 no_grad 和 .numpy() 的少量時間，頁面沒提。但頁面沒有說這是唯一原因，也寫明沒有逐項量測，所以不列為問題，以免又回到第 1 輪那條必要。

6. 呈現
- 在暫存副本裡，`zensical build --clean --strict` 和 validate_site 的結束碼都是 0。建置出的 HTML 裡，摺疊區塊中的表格和兩個 data-excerpt 程式區塊都正常呈現。
- SVG 和 HEAD 相同；qlmanage 渲染乾淨，有 viewBox、title、desc，內容和程式相符。
- 和第 1 輪查核者的副本比較，lesson 20 的程式、notebook、CPU 與 GPU 紀錄、SVG、審查檔都沒有變動，只改了頁面。執行紀錄區塊和 HEAD 逐位元相同。

should：第 118 行建議放入空圖，但第 114–116 行的框數斷言要求每張圖至少一個框，兩者需要一句話區分，見下表。

記錄檔都在暫存副本，檔名以 20r2-* 開頭：run、ex1/ex2/ex3/ex5、ex5ng、build、validate-site、validate-lessons，以及 probe.py 和 timing.py。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/lessons/20-deployment.md 第 118 行（「實際部署時，還要放入空圖…」），對照第 114–116 行的框數斷言說明 | 第 114 行說，兩邊都沒有框時「第三層等於什麼都沒驗證」，所以每張原圖都要至少解出一個框。第 118 行接著建議實際部署時要放入空圖。換成真正訓練過的模型，空圖的正確結果就是兩邊都沒有框；照抄本例的斷言，空圖一定會失敗。初學者可能因此以為空圖不該放進測試，或去調低門檻讓空圖也出框。斷言失敗會直接報錯，讀者不會因此得到錯誤的「通過」，所以列為建議。 |

最後一次查核的建議事項，在下方〈定稿修正〉逐項處理。

## 來源對照

頁面上關於原始論文、官方程式與函式庫行為的說法，由 AI 打開頁面引用的來源（論文章節、固定 commit 的官方程式、官方文件）逐句核對。查閱的來源：

- torch 2.9.1 installed source (.venv-model/lib/python3.12/site-packages/torch/onnx/__init__.py): export() signature dynamo: bool = True (line 72), docstring on dynamic_shapes/dynamic_axes (lines 101-215), legacy-branch DeprecationWarning text (lines 321-336)
- torch 2.9.1 installed source torch/onnx/_internal/torchscript_exporter/utils.py lines 243-250 (legacy export runs model once, equivalent of torch.jit.trace)
- torch 2.9.1 installed source torch/onnx/_internal/torchscript_exporter/symbolic_opset9.py _adaptive_pool (line 1662 onward: AveragePool with uniform kernel/stride when the output size divides the input size)
- https://docs.pytorch.org/docs/stable/onnx.html (redirects to https://docs.pytorch.org/docs/2.14/onnx.html) — torch.onnx.export signature and 'Changed in version 2.9: dynamo is now True by default.'
- https://docs.pytorch.org/docs/2.9/onnx.html — torch.onnx.export dynamo default, dynamic_shapes vs dynamic_axes
- https://docs.pytorch.org/docs/2.9/notes/cuda.html — section 'Asynchronous execution' (event timing example with torch.cuda.synchronize() '# Wait for the events to be recorded!')
- onnx 1.19.1 installed source onnx/checker.py check_model docstring (lines 121-145, full_check runs shape inference); onnx.defs.get_schema('Conv'/'AveragePool', 17) kernel_shape required flag; local check_model run on Conv/AveragePool nodes without kernel_shape
- https://onnxruntime.ai/docs/get-started/with-python.html — PyTorch export with dynamic_axes, onnx.checker.check_model, ort.InferenceSession and run(None, {...})
- https://github.com/NVIDIA/TensorRT/blob/b8db91e15be2cae4465ac17fab19e0f969e45407/samples/common/sampleOptions.cpp — help text for --minShapes/--optShapes/--maxShapes (lines 2526-2545), --noTF32/--fp16/--int8/--stronglyTyped (lines 2604-2611), --saveEngine/--loadEngine (2651-2652), --shapes (2750); strongly-typed handling disabling kFP16 (lines 1268-1282)
- https://github.com/NVIDIA/TensorRT/blob/b8db91e15be2cae4465ac17fab19e0f969e45407/samples/trtexec/README.md — Description and 'Example 3: Running an ONNX model with full dimensions and dynamic shapes' (lines 105-126)
- https://github.com/NVIDIA/TensorRT/blob/b8db91e15be2cae4465ac17fab19e0f969e45407/include/NvInfer.h — BuilderFlag kFP16/kINT8 deprecated in 10.12 (lines ~8847-8854), kTF32 comment (lines 8868-8871), kVERSION_COMPATIBLE (8908), HardwareCompatibilityLevel (9318-9352), IInt8Calibrator deprecated in 10.1 (8203-8215)
- https://github.com/NVIDIA/TensorRT/blob/b8db91e15be2cae4465ac17fab19e0f969e45407/include/NvInferRuntime.h — OptProfileSelector kMIN/kOPT/kMAX comments (lines 2634-2636)
- https://github.com/NVIDIA/TensorRT/blob/b8db91e15be2cae4465ac17fab19e0f969e45407/include/NvInferVersion.h — version 10.13.0 build 35
- https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/how-trt-works.html — build phase 'selects the fastest available kernel ... for each layer on your target GPU'
- miniyolo/deployment_gpu.py lines 59-63 (author's L4 build uses trt.BuilderFlag.TF32 clear and trt.BuilderFlag.FP16 set), used only to connect the deprecation finding to the page's L4 run

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | 「有 NVIDIA GPU 時，TensorRT 怎麼接」：第 246 行的 trtexec 命令與第 254 行說明第 2 點（--fp16）、第 263 行「使用時要注意」第 3 點，以及 L4 段第 301 行「允許 FP16 不等於每層都以 FP16 執行」 | 頁面把「加 --fp16（BuilderFlag::kFP16）允許 FP16，由 builder 逐層挑精度」當成 TensorRT 的一般做法。但在頁面所連的 TensorRT 10.13 原始碼 include/NvInfer.h 裡，kFP16 的註解是 "Enable FP16 layer selection, with FP32 fallback. \deprecated Deprecated in TensorRT 10.12. Superseded by strong typing."。sampleOptions.cpp 也顯示：加了 --stronglyTyped 時，trtexec 會印出警告並停用 kFP16。作者 L4 實測用的 miniyolo/deployment_gpu.py 同樣是 config.set_flag(trt.BuilderFlag.FP16)。讀者無從得知，這條路在本頁自己使用的 10.13 版就已經棄用，升級後可能無法使用。 |
| 2 | 建議 | 「三層驗證，各抓不同的錯」第一層查核者（第 94 行）：「運算的屬性（…例如卷積核大小 `kernel_shape`）是否齊全」 | 查核者只對「必填」屬性的缺漏報錯，但 ONNX 規格中 Conv 的 kernel_shape 不是必填。onnx 1.19.1 的 schema 寫的是 required=False，並註明 "If not present, should be inferred from input W."。實測：opset 17 的 Conv 節點不帶 kernel_shape 也能通過 onnx.checker.check_model；AveragePool 少了它則報 "Required attribute 'kernel_shape' is missing."。用卷積核大小舉例，會讓讀者以為查核者會擋下缺 kernel_shape 的 Conv。 |
| 3 | 建議 | 「使用時要注意」第 9 點（第 269 行），以及第 271 行「或改用 CUDA events：在佇列裡打上時間戳記，由 GPU 自己記錄時間」 | 頁面把 CUDA events 寫成「同步」之外的另一個選項，卻沒說明讀取經過時間前，仍要等結束事件真的完成。PyTorch 官方 CUDA semantics（Asynchronous execution）的事件計時範例，在 end_event.record() 之後、start_event.elapsed_time(end_event) 之前，仍呼叫 torch.cuda.synchronize()，註解是 "# Wait for the events to be recorded!"。照頁面寫法，讀者可能記完事件就直接讀時間。 |
| 4 | 建議 | 「收益、代價與常見錯誤」末的參考來源（第 288 行）：[PyTorch ONNX 官方文件](https://pytorch.org/docs/stable/onnx.html) | 頁面關於匯出器的說法（2.9 起預設 dynamo=True、dynamo=False 是舊的 TorchScript 匯出器、DeprecationWarning 文字）都綁定 PyTorch 2.9.1，但連結指向 stable。stable 現在會導向 docs.pytorch.org/docs/2.14/onnx.html。目前 2.14 文件仍記載 "Changed in version 2.9: dynamo is now True by default."，也仍說明 dynamic_axes 用於 dynamo=False，內容尚未矛盾。但頁面自己說舊匯出器將來可能被移除，一旦 stable 不再記載 dynamo=False，讀者照連結就找不到本頁依據的版本說明。TensorRT 的連結則已固定到 commit。 |

各項的處理見下方〈定稿修正〉（來源為「來源對照」的列）。

## 定稿修正

上面各項意見與先前查核留下的建議，由 AI 逐項核實後處理：必要問題全部修正，建議事項只在修正明確、範圍小時採用。

| # | 來源 | 意見 | 處理 |
|---|---|---|---|
| 1 | 先前查核 | 建議放空圖，但照抄「每張原圖至少一個框」的斷言時，空圖一定失敗 | 已修正：補一句：空圖預期兩邊都沒有框，這類圖要確認兩邊一致；「每張原圖至少一個框」的斷言只用在預期會有框的圖，防的是兩邊都變成空結果卻照樣通過。程式不改。 |
| 2 | 來源對照 | `--fp16`／BuilderFlag::kFP16 在 TensorRT 10.12 已標為棄用，頁面沒有交代 | 已修正：取固定 commit 的 NvInfer.h，確認 kFP16 標為「Deprecated in TensorRT 10.12. Superseded by strong typing」，NvInferVersion.h 顯示該 commit 為 10.13。sampleOptions.cpp 加了 --stronglyTyped 時會停用 kFP16；kTF32 沒有棄用。在「使用時要注意」第 3 點補一句，說明這種做法已棄用、官方改推 strongly typed network（`--stronglyTyped`）。也寫明本頁命令與 L4 實測用的是 10.13 仍可使用的舊做法。 |
| 3 | 來源對照 | 查核者舉 Conv 的 `kernel_shape` 當必填屬性的例子，但它不是必填 | 已修正：用 onnx 1.19.1 實測：opset 17 的 Conv 少了 kernel_shape 仍通過 check_model，AveragePool 則報「Required attribute 'kernel_shape' is missing.」。重新匯出的 grid.onnx 確實含 1 個 AveragePool。改成「運算必填的屬性（…例如本例 AveragePool 的池化視窗大小 `kernel_shape`）」。 |
| 4 | 來源對照 | CUDA events 被寫成同步之外的另一個選項，沒說讀取時間差之前仍要等事件完成 | 已修正：第 9 點改成「必須同步……；改用 CUDA events 時，讀取時間前也要等事件完成」。說明段補上：讀取時間差之前，仍要等結束的戳記真的記下，例如對結束的 event 呼叫 `synchronize()`，或呼叫 `torch.cuda.synchronize()`。 |
| 5 | 來源對照 | PyTorch ONNX 文件連結指向 stable，頁面說法卻綁定 2.9 | 已修正：確認 https://docs.pytorch.org/docs/2.9/onnx.html 回 200，內容含「Changed in version 2.9: dynamo is now True by default」與 dynamic_axes 的說明。連結改為該固定版本，文字改成「PyTorch 2.9 ONNX 官方文件」。 |

修正後由另一位 AI 檢查這一批頁面（`docs/lessons/17-capstone.md`、`docs/lessons/18-video.md`、`docs/lessons/19-tracking.md`、`docs/lessons/20-deployment.md`）的改動，第 1 次：通過。檢查內容：每項改動是否符合程式、紀錄與引用的來源（需要時重算或重跑），回報已修正的必要問題是否真的修好、沒改的理由是否成立，改動是否符合寫作規範，網站嚴格建置與程式摘錄比對是否通過。

| # | 嚴重度 | 位置 | 留下的意見 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | docs/lessons/20-deployment.md 第 263 行（第 3 點新加的棄用說明）與第 288 行〈參考來源〉 | 新加的說法本身正確：固定 commit b8db91e 的 include/NvInfer.h 第 8848–8850 行寫著 `kFP16`「Deprecated in TensorRT 10.12. Superseded by strong typing.」。但頁面引用的兩個 TensorRT 來源（trtexec README、sampleOptions.cpp）都沒寫到棄用；trtexec 自己的 `--help` 也沒把 `--fp16` 標成棄用。讀者，或之後要升級 TensorRT 的維護者，沒辦法從頁面上的連結查證這句。 | 已修正；這項修正由下方〈後續編輯的檢查〉核對 |

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 1 輪：獨立查核之後的編輯

來源行為「參考來源：」。BuilderFlag 棄用連結指向 TensorRT b8db91e 的 include/NvInfer.h：NvInferVersion.h 為 10.13.0，第 8848–8850 行正是 kFP16 的「Enable FP16 layer selection, with FP32 fallback. Deprecated in TensorRT 10.12. Superseded by strong typing.」。正文「10.12 起 `--fp16`／BuilderFlag.FP16 標為棄用、改推 `--stronglyTyped`、10.13 仍可用」與此相符；sampleOptions.cpp 有 `--fp16`、`--stronglyTyped`，trtexec README 的 Example 3 錨點存在，PyTorch 2.9 ONNX 文件回 200。第 9 項的 CUDA events 同步說明正確。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | 「參考來源：」那一行的位置（〈收益、代價與常見錯誤〉之後、〈L4 GPU 上的 TensorRT 實測〉之前） | 這一行後面還有整節〈L4 GPU 上的 TensorRT 實測〉，所以它不在頁尾，和 status／curriculum／publish 說的「頁尾〈參考來源〉」對不上。L4 那節對 TensorRT Python 介面的說法（set_input_shape、set_tensor_address、execute_async_v3、builder 依 tactic 選精度）也沒有被這一行的來源涵蓋；其中 ONNX Runtime 入門連結沒有版本。 | 已修正：「參考來源：」移到〈L4 GPU 上的 TensorRT 實測〉之後、執行紀錄區塊之前。未改：沒有另加 TensorRT Python API 文件與固定版本的 ONNX Runtime 連結。 |

### 第 2 輪：上一輪的處理與審查紀錄：有必要問題

「參考來源：」現在在〈L4 GPU 上的 TensorRT 實測〉的摺疊區之後、執行紀錄區塊之前；檢查建出的 HTML，它是 </details> 之後的獨立段落。五個連結都回 200；NvInfer.h 第 8848–8850 行正是 kFP16 的 Deprecated in TensorRT 10.12；L4 段的 set_input_shape／set_tensor_address／execute_async_v3 與 miniyolo/deployment_gpu.py 第 72–84 行相符，tactic 的說法在紀錄的來源對照（how-trt-works）裡，所以「未改」的理由成立。頁面部分通過；紀錄有必要問題。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | reviews/20-deployment.md〈獨立查核〉第 1 次查核的「必要問題的修正」 | 紀錄的問題：清單只列 29 項中的前 12 項，沒有註明省略；被省略的第 25–29 項正是查核者那 1 項 must（第 165 行 no_grad／.numpy() 的歸因）與 4 項 should 的處理，所以這幾項發現在紀錄裡沒有處理。列出的 12 項反而都是內部清單，例如「trace line 3 [stale-version]：保留不改」「impact line 100：…」。 | 已修正：不再截斷，列出修正時處理的全部項目（包括查核者那 1 項必要問題與 4 項建議的處理），標題改成「修正必要問題時處理的項目」。 |
| 2 | 建議 | reviews/20-deployment.md〈來源對照〉 | 來源清單是英文，並含本機路徑，例如「torch 2.9.1 installed source (.venv-model/lib/python3.12/site-packages/torch/onnx/__init__.py): export() signature …」；最後一項在「IInt8Calibrator deprecated in 10.…」截斷。 | 部分修正：本機路徑換成 repo 內的相對路徑，不再截斷；英文的來源說明是查核者的原文，保留不譯。 |

### 第 3 輪：上一輪的處理與審查紀錄：通過

以腳本核對紀錄：第 1 次查核（1 項必要、4 項建議）的處理清單列出全部 29 項，包含查核者的必要問題與 4 項建議（上一輪要求，已做到）；來源對照 4 項與先前查核 1 項都在〈定稿修正〉處理；英文來源說明依處理欄決定保留原文，這點不再重報。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/20-deployment.md 第 45、55、63、87–91 行 | 內部標籤與殘句：處理清單的「checker must（第 165 行…）」「checker should（…）」×4、「先前審查意見 line 3 [stale-version]」，以及「（證據：暫存副本、bench20b.py）」。（第 146–147、159、173 行的 checker 指 onnx.checker，屬正常用語。） | 未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 4 輪：上一輪的處理：通過

第 3 輪第 1 項：第 87–91 行改成「查核者的必要問題／建議」，第 63 行改成「版本標籤過期」，bench20b 的證據括號已刪。處理說明屬實。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/20-deployment.md 第 45 行與第 3 輪第 1 項處理欄 | 第 45 行仍以「must：」標籤開頭；處理欄「「查核者的必要問題／should」改成…」是替換造成的病句（原字樣是 checker must／should）。 | 已處理：第 3 輪的處理說明改成統一的說明。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

## 紀錄重產後的檢查

本次由另一位 AI 在自己的隔離副本完整檢查第 20 課。範圍包含全文、目前與先前的審查紀錄、CPU JSON、沿用的 L4 JSON、完整程式與直接／間接 import 的 repo 模組、notebook 最後一格，以及頁面顯示的 SVG。依 `AGENTS.md` 與 `docs/preparation/publish.md` 第 6 節第 7 步逐句對照，從高中程度、數學好、程式新手的角度檢查名詞、閱讀順序、假設例子和練習答案。以下記的是查核時的快照；主審已回覆採用文字意見，修正版交由另一位 AI 檢查，這份紀錄不代替該次複查。

### 隔離方式、實際命令與涵蓋版本

用 `/tmp/lessons-v0.4.0-tools/extracted/usr/bin/rsync` 複製 `/tmp/lessons-v0.4.0-reviews/base/` 到 `/tmp/lessons-v0.4.0-reviews/review-20/`，設定 `LD_LIBRARY_PATH=/tmp/lessons-v0.4.0-tools/extracted/usr/lib/x86_64-linux-gnu`，排除 `.git`、`site`、`.venv*`。後續所有程式、練習變體、ONNX 與網站建置都在此副本執行。沒有修改 `/workspace/learn_to_yolo`，沒有 git、認證、推送、遠端 workflow 或 GPU 操作。

模型使用既有 Python 3.12.14、PyTorch 2.9.1+cpu、ONNX 1.19.1、ORT 1.23.2；模型命令前均設 `PYTHONPATH=.`。實際執行：

```bash
PYTHONPATH=. /workspace/learn_to_yolo/.venv-model/bin/python lesson_cases/20-deployment.py
PYTHONPATH=. /workspace/learn_to_yolo/.venv-model/bin/python artifacts/review20-exercises.py
PYTHONPATH=. /workspace/learn_to_yolo/.venv-model/bin/python artifacts/review20-proof.py
/workspace/learn_to_yolo/.venv-docs/bin/zensical build --clean --strict
/workspace/learn_to_yolo/.venv-model/bin/python scripts/validate_site.py
```

最後兩項皆 exit 0，Zensical 0.0.67，網站驗證包含 60 頁、42 課、連結／錨點／Colab 配對與 Markdown 轉換。另從 `scripts/validate_lessons.py` 載入實際的摘錄檢查函式，這一頁結果 `[]`；逐塊確認兩段標記摘錄的行序與相鄰關係。沒有重寫 coverage，也沒有把全站尚未完成的審查涵蓋宣稱為通過。

快照 SHA-256：

| 檔案 | SHA-256 |
|---|---|
| `docs/lessons/20-deployment.md` | `f53296bd7caef2e4318ca1270d7a19e650925bd1a4c291e8bd070588f2883f51` |
| `lesson_cases/20-deployment.py` | `bf01b4ab68543d80e4f4c0ac27740d87852b9095b7b6b005b9d2dcbaf7157cd3` |
| `docs/assets/diagrams/20-deployment.svg` | `45f53f5c670301be885cbc89767ee1195e8fbe908741eeec467b7f49ea6efda4` |
| `artifacts/checks/curriculum/20-deployment.json` | `48b1cd9f50e8e6df94bf6acb1fc49beed817e55da1f94ea40dd297f99f981145` |
| `artifacts/checks/curriculum/deployment-gpu.json` | `460c7ac5fd8ff28f4b867584d362e7172f4ed3b45a3b3ae5387af18de42383df` |
| `notebooks/20-deployment.ipynb` | `2898170995e7f67b17d34e29a90bd97cef611554a9cdbb61bd4bf5d448f3fec1` |

結束前核對這六個檔案和起始 base 逐 byte 相同。CPU 紀錄綁定的 9 個檔案、L4 紀錄綁定的 12 個檔案全部與副本 SHA-256 相符。`miniyolo/__init__.py` 引入的 data、models、targets、losses、inference、metrics 與它們引用的 geometry 都已讀過；另讀 `miniyolo/deployment_gpu.py` 對照 L4 行為。

### 程式、練習與防護的實際結果

預設案例 exit 0，真的產生 ONNX，checker 通過，ORT 使用 CPUExecutionProvider 執行 B=1、2、3；raw shape 分別為 `[1,4,4,7]`、`[2,4,4,7]`、`[3,4,4,7]`，最大 raw 差分別是 `1.1920928955078125e-7`、`4.76837158203125e-7`、`4.76837158203125e-7`。raw allclose、還原框／分數／類別比較以及每圈每張圖有框的斷言均通過；B=3 框數 `[16,16,16]`，80×80 被拒絕。notebook 最後一格與完整程式逐字相同，已存 stdout 與 CPU JSON 相同。

獨立重建相同 seed、一次 SGD 更新的模型後，再讀實際匯出的 ONNX 比對。三張圖各解出 16 框，分數約 0.059981～0.060949，類別都是 0；96×64 那張縮成 64×43、上補 10／下補 11；80×48 有 8 個框高為零，48×80 有 8 個框寬為零，96×64 沒有退化框。頁面對雜訊圖、模型品質、NMS 沒刪候選、框落在補邊及退化框限制的說明與實跑一致。

| 練習／反例 | 實際結果 |
|---|---|
| 練習 1 | 印出 `3 3 torch.Size([3, 3, 64, 64])`；B=3 輸入 `[3,3,64,64]`、raw `[3,4,4,7]`。 |
| 練習 2：刪第三張來源 | B=1、2 通過；B=3 的輸入 shape 是 `[2,3,64,64]`，shape 斷言失敗，exit 1。 |
| 練習 3：前處理改 80 | B=1 在輸入 shape 斷言失敗，exit 1；再單獨移除該斷言，ORT 報 InvalidArgument，index 2、3 為 Got 80／Expected 64。 |
| 練習 4 | `1e-5 + 1e-5 × 3.2 = 0.000042`。 |
| 練習 5：門檻改 0.07 | B=1 兩個 assert_close 與 labels 比較先通過，之後報 `B=1: a source has no restored box [0]`，exit 1。 |
| 移除練習 5 的框數防護 | B=1、2、3 都是空對空，程式 exit 0 並印 completed，證實防護的用途。 |
| 移除 B 迴圈外的 no_grad | `.numpy()` 報 `RuntimeError: Can't call numpy() on Tensor that requires grad`。 |

另外直接檢查匯出圖：恰有一個 AveragePool，`kernel_shape=[2,2]`、`strides=[2,2]`。移除它的必填 kernel_shape，checker 報 `Required attribute 'kernel_shape' is missing`；把模型輸入型別故意改 int64，預設 checker 通過，`full_check=True` 則指出 Conv 不支援 tensor(int64)。這證實頁面分清結構檢查與型別／shape 檢查。原圖的普通與 full checker 都通過。

PyTorch 輸入 80×80 得 `[1,4,4,7]`；在暫存記憶體副本中只把 ONNX 輸入高寬放寬到 80，並把輸出宣告改成 5×5 避免干擾性 shape 警告，實跑得 `[1,5,5,7]`。沒有更動教材 ONNX，這個反例核實「不能只替空間軸取名」和固定池化的說明。參數數量實算為 15,511。

練習 helper 首次檢查錯誤地只搜尋 stderr 最後一行；ORT 在錯誤尺寸資訊後還印一句提示，所以 helper 的檢查失敗。改為搜尋完整 stderr 後重跑通過。這是暫存檢查程式的問題，原案例與練習的預期結果沒有因此變動。

### 新紀錄數值與沿用的 L4 證據

從目前 CPU JSON 的完整位數重算，沒有以表格的四捨五入值代入：

| 計算 | 結果 |
|---|---|
| raw PyTorch／ORT | 2.2670967023，約 2.27／2.3 倍 |
| raw／PyTorch 端到端 | 7.7685201461%，約 7.8%；這是不同計時範圍中位數的估算 |
| PyTorch 端到端減 raw | 3.0238809995 ms，約 3.02 ms；未逐階段量測 |
| raw PyTorch減 ORT | 0.1423519861 ms，約 0.142 ms |
| 同一輪差的中位數 | 0.2490429906 ms |
| 上項／PyTorch 端到端中位數 | 7.5960672974%，約 7.6% |
| ORT B=1：1000／中位數毫秒 | 8,901.152254 張／秒 |
| ORT B=2：2000／中位數毫秒 | 20,217.539301 張／秒 |
| 上述兩個速率的比例 | 2.2713395663，約 2.3 倍 |
| 分開跑兩次 B=1減一次 B=2 | 0.1257660042 ms，約 0.13 ms |

三輪中位數例子、allclose 的 3e-5／4.2e-5，以及 float32 加總順序例子都重算正確。正文更新的表格與以上衍生數字沒有舊值殘留。

獨立預設執行的時間較原紀錄波動：raw PyTorch 0.421177 ms、ORT 0.0996225 ms，端到端中位數 4.740484／4.528337 ms，成對差中位數 0.686093 ms；只保存於暫存 stdout，沒有替換教材紀錄，也沒有把差異當作已證明的效能提升。原紀錄與重跑都沒有逐階段計時或量到長時間持續吞吐；數值一致性通過與效能估計的限制分開記錄。

L4 沒有重跑。既有 `deployment-gpu.json` 的 12 個依賴仍相符，數字與頁面一致：PyTorch 2.9.1+cu128／CUDA build 12.8／TensorRT 10.13.3.9、Adam 40 步、loss 0.984767→0.071048、15,511 個參數；FP32 與允許 FP16 的 B=1～4 最大 raw 差都 7.6293945e-6，框差 3.8146973e-6；B=1 中位數分別 0.146921／0.146962 ms，兩列四捨五入成 0.147 正確。候選總數各為 `[2,4,6,7]`，沒有空對空。run key、39.200435761 秒與 storage SHA-256 核對欄都相符。GPU Python 程式的計時範圍、同步、profile、TF32 關閉、FP16 允許混合精度，與正文相符。這次只核對既有證據與原碼，沒有查遠端 run 的新狀態。

### 原始來源與版本

引用函式庫與 TensorRT 的說法均重新查來源，沒有沿用舊審查結論。官方下載檔與 HTTP 結果保存在副本 `artifacts/runs/review20/sources/`。

- [PyTorch v2.9.1 torch/onnx/__init__.py](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/onnx/__init__.py)：第 72 行 `dynamo=True`、第 137～140／176～183 行 dynamic_shapes／dynamic_axes、2.9 預設變更與第 326 行起的 legacy 警告；下載 SHA-256 `22128fd9193ccca5e6146280304a5f236349d52c0881dbf7dc7e1a9cc8a0f673`，與已安裝 2.9.1 原碼相同。
- [PyTorch v2.9.1 torchscript_exporter/utils.py](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/onnx/_internal/torchscript_exporter/utils.py)：第 243～250 行說明非 ScriptModule 情況跑一次、相當於 jit.trace。[symbolic_opset9.py](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/onnx/_internal/torchscript_exporter/symbolic_opset9.py) 第 1662～1712 行核對 adaptive pool 轉成固定 AveragePool 的 kernel／stride。
- [PyTorch v2.9.1 CUDA semantics 原文](https://github.com/pytorch/pytorch/blob/v2.9.1/docs/source/notes/cuda.rst)：第 277～310 行 asynchronous execution，event 計時前仍等事件完成。頁面固定到 2.9 的 ONNX 與 CUDA HTML 在此次環境回 403，改用這些同版本官方原碼與官方文件原文；沒有聲稱這次 HTML 回 200。
- ONNX 1.19.1 已安裝的 `onnx/checker.py`（SHA-256 `d02da350cacfd9378996bedb0e043a9306ebdf9f51f7c98c49d43f12d0eb86f7`）：check_model 的 full_check 預設 False；opset 17 schema 顯示 Conv 的 kernel_shape 非必填、AveragePool 必填，並以上述實跑驗證。
- [ONNX Runtime Python 官方入門](https://onnxruntime.ai/docs/get-started/with-python.html)：重新取得 HTTP 200，核對 ONNX checker、InferenceSession、NumPy 輸入的 run 與 CPUExecutionProvider。這是沒有固定版本的官方入門頁，實驗行為則以已安裝的 ORT 1.23.2 實跑核對。
- TensorRT 使用頁面指定的固定 commit [`b8db91e15be2cae4465ac17fab19e0f969e45407`](https://github.com/NVIDIA/TensorRT/tree/b8db91e15be2cae4465ac17fab19e0f969e45407)。`include/NvInferVersion.h` 明列 10.13.0 build 35。`include/NvInfer.h` 第 8848～8850 行 kFP16 的 FP32 fallback／10.12 棄用、第 8868～8871 行 TF32 的 10 位輸入捨入／23 位累加／預設允許、第 8905 行起版本相容旗標、第 9318 行起硬體相容與第 10478 行起 strong typing 的型別推導；與正文對照。
- 同 commit 的 `samples/common/sampleOptions.cpp` 第 1268～1282 行 stronglyTyped 禁止 kFP16、第 2526～2545 行 min／opt／max shape、第 2604～2611 行 noTF32／fp16／stronglyTyped、第 2651～2652 行 save／load engine、第 2750 行 shapes；`samples/trtexec/README.md` 第 105～126 行 Example 3。逐條核對三條未實跑的 trtexec 命令及它們的說明，沒有把查原碼記成執行成功。
- 同 commit 的 `include/NvInferRuntime.h` 第 2634～2636 行 MIN／OPT／MAX、enqueueV3 的 stream 與記憶體同步要求；`python/src/infer/pyCore.cpp` 第 113～120、1332～1356 行把 set_input_shape／set_tensor_address／execute_async_v3 綁定到相應介面，核對 L4 段的 Python API。後者 SHA-256 `f6ead811b83ba5867565b6de720eef9bf83edc5e2514c094a97dc39afda1739c`。
- 同 commit 的 `tools/pytorch-quantization/README.md` 第 5 行說明模擬量化訓練、可匯出 ONNX 並由 TensorRT 8.0 以後匯入；SHA-256 `e351ba65fc4e7f68d1c7de2ade99dc5d49dc17af5ba32c79cb0fd7d5bdb9e325`。另讀 [TensorRT 官方量化入門](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html)，明列 PTQ／QAT；來源用於下表 INT8 意見，沒有測 INT8。
- [TensorRT 官方 How TensorRT Works](https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/how-trt-works.html)：build 階段替各層選最快 available kernel、輸出 serialized engine；這是未固定版本的官方補充說明。嘗試的 `/10.13.3/` 文件路徑回 404，旗標與 API 的版本核對仍用前述固定 commit。

### 呈現與發現的處理

Chromium 透過 HTTP 讀取建置頁面，4 個表格、10 個 details 正常；計時表格與頁中 SVG 的 locator 截圖成功。SVG 原檔以 Playwright 直接嵌入 HTML 渲染（720×764），另用 `/usr/bin/inkscape` 匯成 PNG；目視中文字、箭頭、三種 backend、metadata 繞行與三層比較皆清楚，39 個 SVG text 的 getBBox 沒有互相重疊。圖有 viewBox／title／desc，與模型只含 raw 的邊界一致。file:// 受瀏覽器政策拒絕、第一次直接開 SVG 截圖逾時，改用 HTTP 與嵌入渲染後成功，沒有修改政策。

| # | 嚴重度 | 位置／發現與具體修正 | 處理 |
|---|---|---|---|
| 1 | 必要 | 第 165～166 行：「同一輪的差，主要就是模型那一步在管線裡的差距」「整條流程只少掉模型那一小段的差距」。兩條管線使用同一套前後處理，但沒有逐階段計時；成對設計仍包含條件與計時波動，不能把觀察到的全差歸因為模型。新紀錄成對差 0.249043 ms 與獨立 raw 差 0.142352 ms 也不同。改成報告成對觀察值，明示未逐階段量測，不分配全差的因果來源。 | 主審回覆已採用，修正版待另一 AI 檢查。 |
| 2 | 建議 | 第 164 行「模型只占…7.8%；其餘約3.02毫秒花在…」。算式正確，但分子是獨立 raw benchmark 的中位數，不是管線中的模型階段時間。改為「以獨立量到的 raw 時间粗估…；相減約3.02 ms」，並明示沒有逐階段量測。 | 主審回覆已採用，修正版待另一 AI 檢查。 |
| 3 | 建議 | 第 147／165 行仍只列成對欄位名稱與算法，沒有填入 0.249043 ms 與約 7.6%；舊審查已寫明紀錄重產後要補。補實值並與約0.142 ms raw 差對照，避免讀者自行以兩個端到端中位數相減。 | 主審回覆已採用，修正版待另一 AI 檢查。 |
| 4 | 建議 | 第 170～175 行把 `B／單批中位數` 稱為實測吞吐量；8,901／20,218 及2.3倍算術正確，但這是按單批中位數換算的速率估計，沒有連續服務計時。標為估計；持續吞吐量應以總張數／總耗時量測，保留未含等 batch 與前後處理的限制。 | 主審回覆已採用，修正版待另一 AI 檢查。 |
| 5 | 建議 | 第 234 行 INT8 表格「要先用一批有代表性的圖片校準」過於概括。固定 commit 的官方量化訓練工具及官方指南提供 QAT；後面的第7點已經比較準確寫「校準或量化流程」。表格改為「需先經校準或量化感知訓練等流程決定縮放」，保持代表性資料的要求，不暗示所有INT8都必須走獨立校準。 | 主審回覆將改成校準或量化感知訓練；修正版待另一 AI 檢查。 |
| 6 | 必要（網站共用呈現） | 第20頁所有摺疊區展開後，實際 DOM 有26個 mjx-container，其中13個套在另一個 mjx-container 中。這是共用 MathJax 處理的重複 typeset，屬程式與靜態 HTML 驗證沒有涵蓋的瀏覽器問題，需修共用初始化並重新在瀏覽器查公式。 | 已回報主審，主審另行處理共用 MathJax 整合，後續由另一 AI 複查。 |

390px 的這次頁面檢查 document scrollWidth 為375，沒有整頁水平溢出；程式碼有超過視窗的 span 位於可捲動程式區塊內，沒有把它誤報為整頁問題。這一項不代替主審對其餘手機情況的檢查。

以上之外，沒有發現必要的程式、練習答案、更新數值、L4 引用或 SVG 問題。暫存證據包含 `artifacts/review20-default.stdout`／stderr、`artifacts/runs/review20/exercises.json`、`proof.json`、`render-details.json`、來源快照及 PNG。未實跑 TensorRT CLI／GPU／INT8、Colab 託管環境、遠端 tag／workflow；沒有真人學生測試。文字修正與共用公式修正的最終通過狀態，以之後另一位 AI 的修正複查為準。

### 修正後的獨立複查

2026-10-05，由與原第 20 節及原 UI 審查者不同的 AI，依 `AGENTS.md`、發布流程第 6 節第 7 步及指定規範，在自己的 `/tmp/lessons-v0.4.0-reviews/fix-20-ui/` 副本查核。副本以指定 rsync 方法建立；root 只讀，沒有修改 coverage、Git、遠端或 GPU。

**原五項文字發現均已採納，沒有剩餘必要問題。** 逐句讀取現行全文與修正前後文，對照原副本 diff、完整案例、CPU JSON stdout 與 notebook 最後一格，結果如下：

| 原發現 | 獨立複查結果 |
|---|---|
| 1：把整段成對差歸因於模型 | 第 165～166 行明說執行條件與波動會影響各階段，不能把全差歸到模型，也沒有逐項量測。與只量整段函式的原碼一致，通過。 |
| 2：7.8%／3.02 ms 像分段測量 | 第 164 行改成以獨立 raw 中位數粗估，3.02 ms 只說兩者相減，不分配到具體階段，通過。 |
| 3：漏成對差實值與百分比 | 第 147、165 行補入 0.249043 ms、raw 差約 0.142 ms 與觀察差約 7.6%；沒有改用兩個端到端中位數相減，通過。 |
| 4：把 B／單批中位數當持續吞吐 | 第 170 行明確標為粗估，實際持續吞吐需總張數／總耗時；仍保留等待 batch 與前後處理未包含的限制，通過。 |
| 5：INT8 暗示必須獨立校準 | 第 239 行改為有號 INT8、校準或量化感知訓練等流程，並解釋 QAT，與後文一致，通過。 |

從 JSON 完整位數重新計算：raw 加速比 2.2670967023；raw／PyTorch 端到端為 7.7685201461%；兩者相減 3.0238809995 ms；raw 差 0.1423519861 ms；成對差 0.2490429906 ms，除以 PyTorch 端到端中位數為 7.5960672974%。估計速率為 8,901.152254／20,217.539301 張／秒，比例 2.2713395663，兩次 B=1 減一次 B=2 為 0.1257660042 ms。正文的四捨五入均正確。三輪示例也手算得到兩個中位數差 −1、成對差中位數 1。

`paired_median_ms` 確實先各暖機 3 次，40 輪交替先後順序，取每輪差的中位數；raw 是各自連續 20 次的中位數。沒有逐階段或長時間持續吞吐紀錄。重新開啟固定 TensorRT commit 的[量化工具 README](https://github.com/NVIDIA/TensorRT/blob/b8db91e15be2cae4465ac17fab19e0f969e45407/tools/pytorch-quantization/README.md)，HTTP 200，核對 simulated quantization training 與 ONNX／TensorRT 匯入，支持 QAT 修正；來源 SHA-256 `e351ba65fc4e7f68d1c7de2ade99dc5d49dc17af5ba32c79cb0fd7d5bdb9e325`。

CPU 紀錄綁定的 9 個檔案全部相符。模型案例、CPU JSON、notebook 與 SVG 均與原第 20 節副本逐 byte 相同；notebook 最後一格與案例逐字相同，已存輸出與 JSON stdout 相同。本次依修正範圍沒有重跑未變更的 ONNX 案例、練習或 GPU，不把原審查的執行算成這次新執行。

獨立副本的嚴格 Zensical 建置與 `scripts/validate_site.py` 均 exit 0（0.0.67，60 頁／42 課）。Chromium 從 03 以 sidebar 真正點擊 04→07 loss→11 IoU loss→16 inference head→20；第 20 頁展開 details 後為 13 公式／13 容器／13 MathItems，nested、缺字、舊頁 MathItems 與 MathJax 警告皆 0。390px 下文件寬度 390，三輪與 L4 表格的內部 scrollLeft 可移動。原發現 6 的共用公式問題已通過同次 `fix-ui.md` 的獨立複查。

| 快照檔案 | SHA-256 |
|---|---|
| `docs/lessons/20-deployment.md` | `57dbd7dfffe7159a0fd91fc906e300291c586312f3c38776368346441035ddeb` |
| `lesson_cases/20-deployment.py` | `bf01b4ab68543d80e4f4c0ac27740d87852b9095b7b6b005b9d2dcbaf7157cd3` |
| CPU JSON | `48b1cd9f50e8e6df94bf6acb1fc49beed817e55da1f94ea40dd297f99f981145` |
| notebook | `2898170995e7f67b17d34e29a90bd97cef611554a9cdbb61bd4bf5d448f3fec1` |
| SVG | `45f53f5c670301be885cbc89767ee1195e8fbe908741eeec467b7f49ea6efda4` |

新證據在副本 `artifacts/runs/fix20ui/` 的 `arithmetic.json`、`browser.json`、來源及截圖。結束前頁面、JS、CSS 再次與 root 相同，自有 server 已停止。限制：只複查本次文字與共用呈現修正；未重掃全站，未測 GPU／TensorRT CLI／INT8、Colab 託管或遠端發布，也沒有真人學生測試。


## 2026-10-05 clear-tutorial 三輪重審

以上是原審查歷史；不追溯改成首次盲讀。這次由固定基線 `16f6910` 分段開放並保存當時理解，再修改、核技術及檢查銜接，詳見 [本輪方法與限制](clear-tutorial/16f6910/README.md)。

- 第一輪：[applications當場閱讀原始紀錄](clear-tutorial/16f6910/first-read/applications.jsonl)，基線來源與圖指紋保留；共享檔案系統不是技術隔離。
- 第二輪：[非作者技術／證據核對](clear-tutorial/16f6910/technical/modern-applications.md)，實際來源、數字及必要執行範圍見該報告。
- 第三輪：[另一位讀者前文→本節→後文複查](clear-tutorial/16f6910/transitions/modern-applications.md)，此輪完整頁閱讀非盲讀；受影響段落及圖另有delta核回。
- [原始卡點與具體處理](clear-tutorial/16f6910/decisions.json)保留未新增的選讀建議。原先前提包漏發及08提前brief的限制另列，沒有算成教材錯或冒稱08全程盲讀。

本輪修正後沒有未解的必要問題；這是AI閱讀／技術查核的實際範圍，不是學生學習成效驗收。全站實際Zensical桌面／手機、公式換頁與執行檢查見 [verification.json](clear-tutorial/16f6910/verification.json)，不以SVG檔存在或strict build取代視覺查核。

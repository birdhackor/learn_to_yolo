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

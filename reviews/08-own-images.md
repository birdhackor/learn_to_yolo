# 審查紀錄：自己的圖片推論

審查範圍：`docs/lessons/08-own-images.md`、頁面上的圖（`docs/assets/diagrams/08-own-images.svg`），以及 `lesson_cases/08-own-images.py` 與它 import 的 repo 模組；頁尾自動產生的執行紀錄區塊不在範圍內，由 `scripts/validate_curriculum_evidence.py` 對照紀錄檢查。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面依目前的程式改寫後，由另一位 AI 獨立查核：在獨立的副本執行該節程式、照頁面做練習，逐句對照程式、執行紀錄與手算，檢查程式摘錄與網頁轉換，並從初學讀者（高中程度、數學好、程式新手）的角度看用詞與說明順序。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：通過

結論：通過。沒有必要問題，只有 1 條 should（第 62 行「疊框 PNG」的用詞，見下表）。所有程式都在我自己的暫存副本執行。突變測試另用一份副本。repo 內只做過唯讀動作（git diff、cat、grep、stat、摘錄比對工具）。

1. 對程式的敘述（must 項）：全部成立。
- lesson 執行結果 exit 0，stderr 是空的，stdout 共 7 行（存在暫存副本）。第 1、2、7 行的框座標都取到小數第 4 位，印成 [[20.0, 10.0, 60.0, 30.0]] 和 10.6667。未取整時的 roundtrip 是 59.999996185302734，和頁面說的「60 實際算出 59.999996…」相符。roundtrip 用 allclose 檢查，atol 是 1e-4。
- 兩道 assert 都存在：report 記的 score_threshold 是 0.25，而且每個框的 score 都至少 0.25。第 3 行印出框數與門檻；第 4 行印路徑；第 5、6 行與頁面所寫一致。
- 輸出資料夾裡正好有 5 個檔。prediction.json 的 score_threshold 是 0.25、nms_iou 是 0.5，三個陣列都是空的。prediction.png 和 my_image.png 逐像素相同。
- 對照程式碼核對過的項目：
  - DISPLAY_SCORE_THRESHOLD 是 .25，等於 decode_grid 的預設值；decode 保留 score 至少為門檻的框，再按類別分開做 NMS。
  - score 等於 sigmoid(obj) 乘上 softmax 最大值；obj 的 bias 是 −2；sigmoid(−2)×0.5 是 0.0596。
  - nms_iou 先取 config 的值，沒有時用 .5；兩個門檻都要在 [0,1] 內，否則報錯。
  - 摘要只有 5 個鍵：boxes、score_threshold、nms_iou、output、class_names。
  - 預設輸出是 ROOT/artifacts/runs/predictions/my-image.png；腳本用 sys.path.insert 把 repo 根目錄加進路徑。
  - 畫框用橘色，文字標在框的左上方。
  - head.weight 的 shape 是 [7,32,1,1] 與 [8,32,1,1]。
  - 環境格會 os.chdir 進 repo。
  - .gitignore 涵蓋 artifacts/runs/ 與 artifacts/lesson-*/。
- 數值核對：練習 1、2 的答案，padding 例子的 −14.9 與 −3.7，以及只用理想 scale 時的 10.08、30.23、80.6，都和程式算出的一致。
- 照頁面指令實跑過，不設 PYTHONPATH：
  - 160 步訓練把 checkpoint、report.json、validation-00..03 寫到 artifacts/runs/grid-learning/。追蹤中的 grid-learning.json 雜湊沒變。
  - 對 my.png、my.jpg、validation-00.png 執行 CLI，都輸出 artifacts/runs/predictions/my-image.png 與同名 JSON，並印出含上述 5 個鍵的摘要。
  - 加 --score-threshold .05 時，摘要印出 0.05；給 1.5，或 --nms-iou −0.1，都被 ValueError 拒絕。
  - 從無關的資料夾執行也能跑。
- 突變測試：讓 decode 改用 config 的 0.05、報告仍寫 0.25，lesson 停在第 66 行的 AssertionError。所以頁面說這道檢查「是為了確定框不是用 0.05 篩的」屬實。

2. 先前審查意見與受程式改動影響的段落（must 項）：逐條都處理了。
- 第 3 行的 Colab 連結已經是 v0.4.0。
- 「訓練 CLI 會覆寫官方紀錄」那段警告整段刪除，前提已確認：train.py 的報告預設寫到 --output 目錄的 report.json。
- 第 61、62、94、97、116、119、144、146、148、185、196 行都已改寫，兩條 guard 條目也照辦。
- 全頁只剩第 59 行 lesson 那條指令還有 PYTHONPATH=.，那是應該保留的。artifacts/predictions/ 與 --report 都已不在頁面上。
- 沒有修訂或審查的敘述，也沒有對已修好缺陷留下的補丁說明。

3. 數字（must 項）：正文的數字都是確定值，沒有引用這台 Mac 跑出的數字。這台 Mac 上 3 步模型的框數是 0，正文沒有寫進去。需要等重錄紀錄的值，editor 已經列出。頁尾的執行紀錄區塊仍是舊輸出，要等 verify_curriculum 依重錄的紀錄重新產生，這是預期內的。

4. 摘錄（must 項）：
- 摘錄比對工具對 repo 與暫存副本都印出 []。
- 把摘錄改成 strict=False 時，check 會報錯，證明它真的會擋。
- 推論核心那段已標明是簡化版；正文沒有引用程式行號，提到的行號都是輸出的第幾行。

5. 可讀性：新名詞都有解釋。「顯示門檻」「候選截斷門檻」的用法和術語表一致；教學順序沒有被打亂。

6. 建置與 SVG（must 項）：
- zensical build --clean --strict exit 0，回報 No issues found；validate_site exit 0（log 在暫存副本與 -validate-site.log）。
- validate_lessons 只在審查涵蓋那一步失敗（08-own-images.md: no review），這是預期內的；摘錄與 notebook 的檢查都通過。
- SVG 沒有改動。它有 viewBox、title、desc，用 qlmanage 繪出來很乾淨，框 [10.67,15.375,32,26.125]、上 padding 10、下 padding 11 都和程式一致。
- 這次範圍內只有頁面本身被改動。

補充：editor 擔心的 08-own-data.md 矛盾已經不存在。該頁現在第 286 行的指令沒有 PYTHONPATH，第 289 行寫明腳本會自己把 repo 根目錄加進路徑，和本頁說法一致。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/lessons/08-own-images.md 第 62 行（快速實驗第 2 項）「存下疊框 PNG 與記錄原圖座標的 JSON」 | 這一項講的是 3 步模型這次的輸出。在暫存副本跑出 0 個框，prediction.png 和輸入圖逐像素相同。程式的 owner 已經把輸出第 4 行的 'annotated PNG' 改成 'prediction PNG'，理由正是「沒有框時還說有標註會誤導」。頁面這裡仍然寫「疊框 PNG」。讀者打開一張沒有框的 PNG，可能以為畫框失敗了；要往下讀約 130 行，到〈查看本節輸出檔〉才會看到「沒有框時就和原圖一樣」。 |

最後一次查核的建議事項，在下方〈定稿修正〉逐項處理。

## 定稿修正

上面各項意見與先前查核留下的建議，由 AI 逐項核實後處理：必要問題全部修正，建議事項只在修正明確、範圍小時採用。

| # | 來源 | 意見 | 處理 |
|---|---|---|---|
| 1 | 先前查核 | 快速實驗第 2 項寫「疊框 PNG」，可是 3 步模型可能 0 個框 | 已修正：改成「存下 PNG（原圖疊上留下的框，沒有框時就是原圖）與記錄原圖座標的 JSON」，不寫框數。 |

修正後由另一位 AI 檢查這一批頁面（`docs/lessons/06-decode-nms.md`、`docs/lessons/06-evaluation.md`、`docs/lessons/07-heldout.md`、`docs/lessons/08-own-images.md`、`docs/lessons/08-own-data.md`）的改動，第 1 次：通過。檢查內容：每項改動是否符合程式、紀錄與引用的來源（需要時重算或重跑），回報已修正的必要問題是否真的修好、沒改的理由是否成立，改動是否符合寫作規範，網站嚴格建置與程式摘錄比對是否通過。

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 3 輪：上一輪的處理與審查紀錄：通過

以腳本核對紀錄：第 1 次查核的建議在〈定稿修正〉處理；批次檢查掛在本頁；結構檢查通過。頁面沒有外部連結，紀錄沒有〈來源對照〉也沒有技術查核。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | docs/lessons/08-own-images.md 的 torch.load／weights_only／pickle 與 Pillow 說法；reviews/08-own-images.md | 頁面寫了函式庫行為：「weights_only=True 不是『只讀權重』，而是只允許還原 tensor、數字、字串、list、dict 這類單純的資料」「torch.load 底層用 pickle…來路不明的檔案可能夾帶會被執行的程式碼」。紀錄沒有記下這些說法核對過哪份官方文件。〈驗證範圍〉說函式庫文件的說法另由 AI 打開官方文件逐句核對，並把來源記在該頁紀錄。上一輪對 19-tracking 同類情形（點名卻沒連結）判為建議，所以這裡也列為建議；若把該說法讀成涵蓋所有函式庫說法，這頁就沒有紀錄支持。我讀過的內容和 PyTorch 文件相符。另有殘句：第 11 行「所有程式都在我自己的暫存副本執行：暫存副本。突變測試另用一份副本：暫存副本。」 | 已處理：另做一次函式庫說法的來源對照，見〈補做的來源對照〉。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 補做的來源對照

頁面上關於原始論文、官方程式與函式庫行為的說法，由 AI 打開原始來源逐句核對。結論：沒有誤述函式庫的句子（must 為 0），有 4 項建議，都是缺少必要的限定。 檢查範圍：逐句核對 docs/lessons/08-own-images.md 中關於函式庫與格式行為的敘述，包括 ??? note 摺疊區（兩個「參考答案」只有算術，「量測速度時」沒有函式庫敘述）。對照的是課程固定的版本：torch 2.9.1（PyTorch 2.9 文件與 v2.9.1 原始碼）、Pillow 12.0.0（12.0.0 tag 的原始碼與 docs）、NumPy 2.3.5、Python 3.12，並在 .venv-model 中實測。課程自己的程式行為與 curriculum-evidence 區塊依指示略過。 確認正確的敘述： - `from PIL import Image` 的寫法。 - RGB 圖經 `np.asarray` 得到 `[H,W,3]` uint8，L 模式得到 `[H,W]`。 - 2 維 tensor 做 `permute(2,0,1)`，以及把 4 或 2 個 channel 的輸入送進第一層卷積，都會報 RuntimeError。 - Pillow 的 `__array_interface__` 用 `tobytes()` 提供資料，所以陣列是唯讀的，`torch.from_numpy` 會發出 not writable 警告；`.copy()` 可以避免。 - `.float()/255` 得到 float32；`unsqueeze(0)` 加 batch 軸，以及 NCHW 的說明。 - `torch.allclose` 只依容差比對，不要求完全相等。 - `map_location='cpu'` 會把所有 tensor 載到 CPU。 - `weights_only=True` 會限制 unpickler，只允許 tensor、基本型別、dict 等；檔案若要呼叫不在允許清單的 global，會以 UnpicklingError 拒絕（用無害的 print 實測過）。 - pickle 可能在載入時執行程式碼。 - `strict=True` 會檢查缺少與多出的 key；shape 不符時，不論 strict 怎麼設都會報錯；把整個 checkpoint dict 傳給 load_state_dict 在 strict=True 下會報錯。 - `python -m` 會把目前目錄加到 sys.path 最前面。 - IPython 的 `!` 會把整行交給 shell。 4 項 should： 1. 手機 JPEG 的 EXIF 方向，Pillow 不會自動套用。 2. `convert('RGB')` 直接丟掉 alpha，不會鋪背景色。 3. 「weights_only 會拒絕惡意檔案」說得比 PyTorch 官方更絕對。 4. state_dict 的定義漏了 persistent buffer。 查過但不列入的邊角情況： - 16-bit 灰階 PNG 轉 RGB 時，大於 255 的值會被截成 255（實測 40000 變 255）。 - HEIC 不是 Pillow 12.0.0 支援的格式，但頁面沒有宣稱支援。 - 對 `.gitignore` 的白話解釋。 本次範圍外的觀察：artifacts/checks/curriculum/08-own-images.json 記錄的 case_sha256（5440e8a5…）和目前 lesson_cases/08-own-images.py 的 sha256（d812cdb2…）不同。頁尾證據區的輸出格式也和現行程式、以及第 61–62 行的說明對不上：框座標沒有四捨五入、第 3 行沒有 'at score threshold'、第 4 行寫的是 'annotated PNG'。證據可能需要重新產生，可另用 validate_curriculum_evidence.py 確認。

查閱的來源：

- PyTorch 2.9 官方文件〈Serialization semantics〉的「torch.load with weights_only=True」與「Saving and loading torch.nn.Modules」兩節：https://docs.pytorch.org/docs/2.9/notes/serialization.html
- PyTorch 2.9.1 的 torch.load docstring（weights_only、map_location 的說明與 pickle 警告），在固定版本環境 torch 2.9.1 中讀取；對應網頁 https://docs.pytorch.org/docs/2.9/generated/torch.load.html
- https://github.com/pytorch/pytorch，tag v2.9.1，torch/nn/modules/module.py：state_dict 與 load_state_dict 的 docstring；第 2433–2439 行 size mismatch 錯誤（不受 strict 影響）；第 2611–2633 行 strict 的缺少／多出 key 檢查與 RuntimeError
- https://github.com/pytorch/pytorch，tag v2.9.1，SECURITY.md 第 30、32 行（建議在沙箱執行不可信的模型；weights_only=True 是 "secure to our knowledge"）
- https://github.com/pytorch/pytorch，tag v2.9.1，torch/csrc/utils/tensor_numpy.cpp 第 205–213 行（warn_numpy_not_writeable）；以及 torch 2.9.1 的 torch.from_numpy docstring
- PyTorch 安全公告 GHSA-53q9-r3pm-6pq6（CVE-2025-32434：weights_only=True 仍可 RCE，影響 2.5.1 以前，2.6.0 修正）：https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6
- https://github.com/python-pillow/Pillow，tag 12.0.0，src/PIL/Image.py：第 233–239 行 _conv_type_shape（陣列 shape 為 (H,W) 或 (H,W,bands)）、第 716–726 行 __array_interface__（以 tobytes() 提供資料）、第 920–970 行 convert docstring、第 1035–1043 行調色盤透明度 UserWarning
- https://github.com/python-pillow/Pillow，tag 12.0.0，src/libImaging/Convert.c：第 423–432 行 rgba2rgb（只複製 RGB、丟掉 alpha），第 1532 行 RGBA→RGB 轉換表項目
- https://github.com/python-pillow/Pillow，tag 12.0.0，docs/handbook/concepts.rst（Modes：L、P、RGBA、LA、I;16）與 docs/handbook/image-file-formats.rst 的 PNG 節（第 850–896 行：可讀的模式與 transparency 資訊）
- https://github.com/python-pillow/Pillow，tag 12.0.0，src/PIL/ImageOps.py 第 687–698 行 exif_transpose docstring
- Python 3.12 官方文件〈Command line and environment〉的 -m 與 -P（-m 會把目前目錄加到 sys.path 最前面；執行腳本時加的是腳本所在目錄）：https://docs.python.org/3.12/using/cmdline.html
- IPython 9.17.1 文件〈IPython reference〉的 System shell access 節（! 開頭的整行交給作業系統 shell；課程沒有固定 IPython 版本）：https://ipython.readthedocs.io/en/stable/interactive/reference.html
- 在固定版本環境實測：.venv-model（Python 3.12.15、torch 2.9.1、Pillow 12.0.0、NumPy 2.3.5），腳本是暫存副本裡的 pil_check.py 與 torch_check.py；項目包括陣列唯讀與 from_numpy 警告、各模式的陣列 shape、RGBA／LA 轉 RGB、調色盤透明度警告、EXIF Orientation、strict 與 size mismatch、用無害 print 測試 weights_only 拒絕載入、permute 與卷積的 channel 錯誤
- 只為確認上下文而對照的程式（本次不審查）：scripts/detect_image.py、lesson_cases/08-own-images.py、miniyolo/models.py、notebooks/08-own-images.ipynb

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | 〈實際載入模型，推論指定圖片〉第 102 行「把 `my.png` 換成你的圖片路徑，jpg 也可以」；同一件事也出現在〈在 Colab 用自己的照片〉步驟 2（第 111 行），以及〈CLI 的步驟與預設門檻〉步驟 2「用 Pillow 讀圖，轉成 RGB」（第 134 行） | 頁面請讀者直接用手機照片，卻沒說 Pillow 讀 JPEG 時不會套用 EXIF Orientation 標籤。手機的直式照片常是橫向像素加一個方向標籤，看圖軟體會照標籤轉正，Pillow 的 `Image.open()` 和 `convert('RGB')` 不會。要轉正得另外呼叫 `ImageOps.exif_transpose()`，Pillow 12.0.0 的 docstring 寫："If an image has an EXIF Orientation tag, other than 1, transpose the image accordingly, and remove the orientation data."。在固定版本環境實測：Orientation=6 的 120×80 JPEG，open 和 convert 之後 size 還是 (120,80)，要經過 exif_transpose 才變成 (80,120)。`scripts/detect_image.py` 也沒做這一步，所以 JSON 的 `original_size`、框座標和疊框 PNG 都以檔案裡沒轉正的像素為準，方向可能和讀者在相簿看到的不同（例如差 90°）。本節主題就是座標，讀者很可能把這個差異誤認成座標換算出錯。 | 已修正：補上手機照片的 EXIF 方向：Pillow 不會照標籤轉正，輸出以檔案實際的像素方向為準，必要時先用 `PIL.ImageOps.exif_transpose()` 轉正另存。 |
| 2 | 建議 | 〈120×80 圖片怎麼進 64×64 網路〉第一段（第 17 行）：「透明圖或灰階圖同樣要先轉 RGB：帶透明度的 PNG 通常是 RGBA，有 4 個 channel……`convert('RGB')` 會把它們統一成 3 個 channel。」 | 這句只講 channel 數，沒說透明度怎麼處理。Pillow 12.0.0 的 RGBA→RGB 轉換（src/libImaging/Convert.c 的 `rgba2rgb`）只複製 R、G、B，alpha 直接丟掉，不會把圖鋪到白底或任何背景色上。固定版本實測：RGBA `(10,200,30, alpha=0)` 轉完是 `(10,200,30)`；LA→RGB 也一樣丟掉 alpha。所以透明的地方會露出檔案裡原本存的 RGB 值，可能和看圖軟體顯示的白底或棋盤格完全不同。另外，PNG 的透明度不只 RGBA 一種：Pillow 文件列出 PNG 也可能讀成 `LA`、`P` 等模式。調色盤（P）PNG 的透明度記在 `info['transparency']`，`np.asarray` 會得到 `[H,W]`；如果透明度以 bytes 記錄，轉 RGB 時 Pillow 會發出 UserWarning "Palette images with Transparency expressed in bytes should be converted to RGBA images"（Image.py 第 1035–1043 行）。「通常是 RGBA」有保留語氣，不算錯；但讀者照「先轉 RGB」處理去背圖時，會不知道背景顏色是從哪來的。 | 已修正：補上 `convert('RGB')` 只丟掉透明度、不鋪背景色，要固定背景色就先貼到指定顏色的底圖上。 |
| 3 | 建議 | 〈實際載入模型，推論指定圖片〉第 85 行：「`torch.load` 底層用 pickle（……），來路不明的檔案可能夾帶會被執行的程式碼，這個設定會拒絕它們。」 | 機制本身講對了：PyTorch 2.9 的〈Serialization semantics〉說，weights_only unpickler 只執行還原 tensor 和部分基本型別需要的函式，unpickling 時也不能動態 import；固定版本實測，`__reduce__` 會呼叫 global 的檔案確實被 `UnpicklingError` 擋下。問題在「會拒絕它們」讀起來像無條件的保證，官方沒有說到這個程度。v2.9.1 的 SECURITY.md 第 32 行寫 `weights_only=True` "is also secure to our knowledge even though it offers significantly larger surface of attack"，第 30 行建議在沙箱等隔離環境執行不可信的模型。2.5.1 以前也真的出過 weights_only=True 仍能執行任意程式碼的漏洞（CVE-2025-32434，2.6.0 修正）。同頁第 93 行提到網路上下載的權重，讀者可能把這句理解成「加了 weights_only=True，來路不明的檔案就可以放心載入」。 | 已修正：改成這個設定只還原允許清單裡的型別，清單以外的會報錯、不執行；並引用官方「就目前所知是安全的」，提醒來路不明的檔案只在可信或隔離的環境載入。 |
| 4 | 建議 | 〈實際載入模型，推論指定圖片〉第 70 行「`model_state_dict` 本身也是 dict，稱為 state_dict：key 是參數名稱，value 是該參數的 tensor」，以及第 87 行「`strict=True` 要求名稱一一對上：檔案缺了模型需要的參數，或多出模型沒有的參數，都會報錯」 | PyTorch 2.9 的定義是：state_dict 包含所有參數和 persistent buffer（〈Serialization semantics〉："A module's state dict contains all of its parameters and persistent buffers"；v2.9.1 的 `Module.state_dict` docstring："Keys are corresponding parameter and buffer names"）。`load_state_dict` 的說明也是 "Copy parameters and buffers"，strict 比對的是 `state_dict()` 的全部 key。本書的 GridDetector 只有卷積、沒有 BatchNorm，所以對這個 checkpoint 來說，key 確實全是參數。但這裡是在定義 state_dict 這個通用名詞，同頁又提到其他 YOLO 的權重。讀者換到有 BatchNorm 的模型時，會看到 `running_mean`、`running_var`、`num_batches_tracked` 這種不是參數的 key，可能以為檔案多了不該有的東西而把它們濾掉，結果在 strict=True 下反而因為缺 key 而報錯。 | 已修正：state_dict 的定義補上 buffer（本書模型沒有 buffer），`strict=True` 的說明也改成參數或 buffer。 |

### 第 4 輪：上一輪的處理：通過

逐句對照固定版本來源並實測。RGBA 句：Pillow 12.0.0 Convert.c 的 rgba2rgb 與轉換表第 1532 行，RGBA→RGB 只複製 RGB、丟掉 alpha；實測 (10,200,30,α=0) 轉成 (10,200,30)，L 模式陣列是 (H,W)。state_dict 定義與 strict=True 句：PyTorch 2.9〈Serialization semantics〉寫 state dict「contains all of its parameters and persistent buffers」；實測 size mismatch 在 strict=True 與 False 都報 RuntimeError，缺 key 也報錯；miniyolo、lesson_cases、scripts 都沒有 register_buffer 或 BatchNorm，head.weight [7,32,1,1]／[8,32,1,1] 屬實。weights_only／pickle 句：serialization notes 的 weights_only 限制；v2.9.1 SECURITY.md 第 32 行「secure to our knowledge」與隔離環境的建議；實測 __reduce__ 呼叫 print 的檔被 UnpicklingError 擋下，dict／list／str 可以讀。EXIF 句：ImageOps.exif_transpose docstring；scripts/detect_image.py 只做 open().convert('RGB')；實測 Orientation=6 的 JPEG 開檔與轉換後仍是 (120,80)，exif_transpose 後變成 (80,120)。都屬實，回應〈補做的來源對照〉4 項，合寫作規則；摘錄比對工具印出 []。B：紀錄有〈補做的來源對照〉、來源清單與 4 項處理；第 3 輪第 1 項屬實（第 11 行點名的殘句已拿掉路徑）。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | docs/lessons/08-own-images.md 第 17 行 | 「要固定背景色，先把圖貼到指定顏色的 RGB 底圖上再轉」沒說要用透明度當遮罩。照字面寫 bg.paste(im) 時，Pillow 不會使用 alpha，實測結果仍是 (10,200,30)，和 convert('RGB') 一樣。 | 已修正：補上以透明度當遮罩貼上，並附 `background.paste(image, mask=image.getchannel('A'))` 的寫法。 |
| 2 | 建議 | docs/lessons/08-own-images.md 第 85 行 | 「所以來路不明的檔案，仍應只在來源可信或隔離的環境中載入」自相矛盾：來路不明就不會是來源可信，初學者會不知道該怎麼做。SECURITY.md 的建議是在沙箱等隔離環境執行。 | 已修正：改成「最好只在隔離的環境（例如容器或虛擬機）中載入」。 |
| 3 | 建議 | reviews/08-own-images.md 第 52、98 行 | 〈補做的來源對照〉來源清單中，CVE-2025-32434 公告的網址被替換成「…/security/advisories/暫存副本」，連結失效；第 52 行「（log 在暫存副本與 -validate-site.log）」是殘句。 | 已處理：第 3 輪的處理說明改成統一的說明。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 5 輪：上一輪的處理：通過

兩句都查證過。
- 遮罩那句（第 17 行）：在暫存目錄用 .venv-model（Pillow 12.0.0、Python 3.12.15）實測 RGBA (10,200,30)，alpha 分別取 0、128、255。convert('RGB') 和不帶 mask 的 paste 都得到 (10,200,30)；background.paste(image, mask=image.getchannel('A')) 得到 (255,255,255)、(132,227,142)、(10,200,30)，和 Image.alpha_composite 結果相同，確實用到了 alpha。這回應了第 4 輪第 1 項。
- 隔離環境那句（第 85 行）：PyTorch v2.9.1 SECURITY.md 第 30 行寫 Prefer to execute untrusted models within a secure, isolated environment such as a sandbox (e.g., containers, virtual machines)，第 32 行寫 weights_only=True is also secure to our knowledge，頁面屬實。原句「來路不明卻要來源可信」的矛盾已消除，回應了第 4 輪第 2 項。
兩句讀來通順，符合寫作規則。副本建置、validate_site 與摘錄比對（[]）都通過。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | 第 17 行〈120×80 圖片怎麼進 64×64 網路〉末句 | 範例裡的 background 沒說怎麼建立（要和原圖同尺寸、填滿指定顏色）。「以透明度當遮罩」的遮罩是 0–255 的權重，半透明的地方會混色；術語表把 mask（遮罩）定義為 True／False 表，兩者不完全相同，初學者照抄時得自己猜。 | 已修正：補上底圖的建法 `Image.new('RGB', image.size, (255, 255, 255))`，並說明遮罩的值是 0 到 255 的比例：0 保留底色，255 用原圖，中間按比例混合。 |

### 第 5 輪：上一輪的處理：有必要問題

第 3 輪第 1 項是人工寫的，沒有由產生器重寫；它說「殘句已清理」，以第 11 行來看屬實。第 4 輪第 1、2 項是頁面修正，見 docs/lessons/08-own-images.md。第 4 輪第 3 項不實。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | 第 4 輪第 3 項處理欄（第 123 行） | 處理欄寫 2 處中 1 處已改寫或刪除、只剩第 52 行。但〈補做的來源對照〉來源清單第 98 行仍是「PyTorch 安全公告暫存副本pm-6pq6…：https://github.com/pytorch/pytorch/security/advisories/暫存副本pm-6pq6」，連結仍然失效，所以兩處都還在。原因是產生器只比對〈後續編輯的檢查〉以上的文字，而這一行在〈補做的來源對照〉裡。 | 已處理：用詞類的處理說明改成統一的說明（紀錄保留查核者的原文，只統一替換路徑與內部名稱），不再逐句計數。 |
| 2 | 建議 | 第 4 輪第 3 項處理欄開頭 | 開頭寫「第 3 輪的處理說明改由逐句比對紀錄產生」，但本頁第 3 輪只有一列（第 85 行），仍是人工寫的「…；殘句已清理。」，沒有重寫。 | 已處理：用詞類的處理說明改成統一的說明（紀錄保留查核者的原文，只統一替換路徑與內部名稱），不再逐句計數。 |

### 第 6 輪：上一輪的處理：通過

用 .venv-model 的 Pillow 12.0.0（Python 3.12.15）在暫存目錄做一張 4×1 的 RGBA 圖，存成 PNG 再讀回，四個像素是 (10,20,30,α0)、(0,0,0,α0)、(255,0,0,α128)、(0,0,255,α255)。convert('RGB') 只丟掉 alpha，透明處露出存的 (10,20,30) 與 (0,0,0)，和「常是黑色，但不保證」相符。照頁面兩行執行 Image.new('RGB', image.size, (255, 255, 255)) 與 background.paste(image, mask=image.getchannel('A'))：α0 得到底色 (255,255,255)，α255 得到原色 (0,0,255)，α128 得到 (255,127,127)，正好是 128/255 的比例混合（黑底時是 (128,0,0)），和 alpha_composite 的結果相同；不帶 mask 的 paste 則和 convert 一樣。頁面這幾句全部成立。用詞是現在式，沒有修訂敘事，全形標點與中英文間距都符合審查用的事實與寫作規範清單。建置：依指示 rsync 到暫存副本後，zensical build --clean --strict 結束碼 0（No issues found），validate_site.py 結束碼 0（60 個 HTML 頁、42 課，各項都 passed）；摘錄比對工具對本頁輸出 []；建好的 HTML 有這段新文字。沒有必要問題，只有兩項讓初學者照做更順的建議。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | 〈120×80 圖片怎麼進 64×64 網路〉第 17 行的兩段行內程式 | 行內程式裡的 image 沒有定義，下方程式區塊（第 43 行）又用 image 當 [3,80,120] tensor 的名字。初學者照抄到那段程式時會傳進 tensor，實測得到 ValueError: Size must be a list or tuple。調色盤（P 模式）的透明 PNG 沒有 A channel，image.getchannel('A') 實測會丟出 ValueError: The image has no channel "A"。頁面寫的是「通常是 RGBA」，所以不算錯，但照抄會卡住。另外，頁面沒說貼好之後要用 background 接著做下去。 | 未改：這一句說明作法，`image` 指這段說的 RGBA 圖（用 Pillow 開啟的圖），和下方程式區塊裡的 tensor 是不同的變數；調色盤模式的 PNG 不在這句說的 RGBA 範圍內。 |
| 2 | 建議 | 第 17 行末句「這裡的遮罩是 0 到 255 的比例：…」 | 術語表把 mask（遮罩）定義為 True／False 陣列，〈同名不同義〉的 mask 條也只列 True／False 的用法；這裡的遮罩卻是 0–255 的連續權重。「這裡的」只是暗示兩者不同，沒有明說。「中間的值兩者按比例混合」沒有數字例子，(255, 255, 255) 是白色也沒有寫。 | 未改：句中已明說這裡的遮罩是 0 到 255 的比例，並說明 0、255 與中間值的意思，照做不需要另外的例子。 |

### 第 6 輪：上一輪的處理：有必要問題

以同一條規則（第 3–5 輪處理欄不得與紀錄矛盾）檢查這份紀錄，找到三處矛盾。處理欄錯位、網址被破壞這兩項的成因在產生器，修正方法已在暫存目錄重產全部 59 份驗證過，只有預期的那幾行改變。第 3 輪 #1 的「殘句已清理」就第 11 行而言屬實，但仍是任務要求不再出現的「已清理」說法。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | 〈後續編輯的檢查〉第二個〈第 5 輪：上一輪的處理：有必要問題〉表格 #1（第 142 行）的處理欄 | 這項必要發現講的是第 4 輪第 3 項處理欄的計數，以及第 98 行失效的連結，處理欄顯示的卻是同一輪頁面那一節的處理「已修正：補上底圖的建法 `Image.new(...)`…」，和這項無關。原因在產生器：delta-handling.json 的 delta:fifth 對這一頁有 3 筆處理（頁面 1 筆、紀錄 2 筆），但 make_review_records.py 的 later 迴圈對兩節都用 handling.get(key).get(page)，各自從第 0 筆開始取，紀錄那一節因此錯開一筆。讀者會以為這項必要問題被一則無關的頁面修改帶過，真正的處理也沒有記下來。 | 已修正：產生器依各節的發現順序對應處理，不再錯開。 |
| 2 | 必要 | 第 4 輪 #3 的處理欄（第 123 行）；第二個第 5 輪 #2 的處理欄（第 143 行） | 第 4 輪 #3 寫「第 3 輪的處理說明改成統一的說明」，但本頁第 3 輪只有一列（第 85 行），處理欄仍是人工寫的「…見〈補做的來源對照〉；殘句已清理。」，不是統一說明。第 5 輪 #2 指出的正是這一點，它的處理「用詞類的處理說明改成統一的說明…」同樣不成立，因為那一列到現在都沒有改。 | 已修正：第 3 輪那一列改成統一的說明，和第 4、5 輪的處理一致。 |
| 3 | 必要 | 第 98 行〈補做的來源對照〉的來源清單；第 4 輪 #3 處理欄（第 123 行）所說的理由 | journal 原文是「PyTorch 安全公告 GHSA-53q9-r3pm-6pq6…：https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6」。產生器 CLEAN 裡用來比對暫存資料夾名的 `[\w-]+-r\d+`，把「暫存副本」當成資料夾名換成「暫存副本」，紀錄於是變成「暫存副本pm-6pq6」和一個打不開的網址。被換掉的既不是路徑，也不是內部名稱，所以第 4 輪 #3 說「只由產生器統一替換路徑與內部名稱」在這一項不成立。〈驗證範圍〉告訴讀者查閱的來源都記在審查紀錄裡，這個來源卻打不開，而且第 4、5 輪已經兩度點名，至今沒修。 | 已修正：比對暫存資料夾名的規則不再誤換 GHSA 編號，來源清單的安全公告編號與網址恢復原文。 |

# 審查紀錄：三步訓練與診斷

審查範圍：`docs/lessons/07-training.md`、頁面上的圖（`docs/assets/diagrams/grid-learning-curve.svg`、`docs/assets/diagrams/grid-learning-predictions.svg`），以及 `lesson_cases/07-training.py` 與它 import 的 repo 模組；頁尾自動產生的執行紀錄區塊不在範圍內，由 `scripts/validate_curriculum_evidence.py` 對照紀錄檢查。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面依目前的程式改寫後，由另一位 AI 獨立查核：在獨立的副本執行該節程式、照頁面做練習，逐句對照程式、執行紀錄與手算，檢查程式摘錄與網頁轉換，並從初學讀者（高中程度、數學好、程式新手）的角度看用詞與說明順序。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：通過

結論：通過。沒有必要，也沒有需要列出的建議。所有驗證都在暫存副本裡做，repo 內只做了唯讀操作。

1. 程式敘述（對照 lesson_cases/07-training.py 全文）
- 照指令重跑本節：exit 0，stdout 和頁尾執行紀錄逐字相同。
- 斷言清單和程式一致：shape、total 有限、梯度有限、梯度總量有限且大於 0；三步後檢查參數已改變，再檢查正格數是 4。
- 其他對照都正確：印出順序是 total、box、objectness、classification；`torch.no_grad()` 底下的監看三行排在 optimizer.step() 之後；第一個參數 shape 是 [8,3,3,3]；表格中各層 shape 和 head bias（−1.8／−2）都對；本批 4 個物件都是 class 0；3 步更新後正格 obj 約 0.115、類別機率約 0.69–0.74、score 不到 0.09。
- 練習照頁面做（刪掉 optimizer.step()）：三行數字完全相同，在「參數已改變」那道斷言報 AssertionError，正格數斷言沒有執行，最後一行沒有印出。和參考答案一致。
- 照頁面跑 `python -m miniyolo.train --steps 160 --samples 32 --device cpu`，也跑了頁面的 Colab 程式（IPython 用 stub 代替）：輸出都寫到 artifacts/runs/grid-learning/，包括 checkpoint.pt、history.json、loss.png、validation-00..03.png、report.json；追蹤中的 grid-learning.json SHA 沒變。
- loss.png 看過：藍 total、紅 box、綠 objectness、紫 classification；圖例 box (raw, before ×5) 和橫軸文字都和頁面相符。
- report.json 和正式紀錄由同一個 train() 寫出（record_evidence.py 只是加上 `--report`），所以頁面說「欄位相同」成立。

2. 清單處理
- 先前審查意見與受程式改動影響的段落每一條都處理了：覆寫警告、`--report`、配色提醒、「另看獨立資料」、舊的圖板讀法（pred 0／class 1 / 1.000）都刪掉或改寫；models.py 連結已改到 lessons-v0.4.0；標題已改。
- 沒有任何連結指向舊標題的錨點，只有舊審查檔 reviews/training-evidence.md 提到舊標題文字，那不是連結。
- 全頁沒有修訂或寫作過程的敘述。

3. 數字
- 沒有引入任何 Mac 實測的數字。
- 表格、計數、0.62／0.980／0.47、AMD EPYC／0.74 秒都和追蹤中的 grid-learning.json 一致，編修者都已列進待重錄數值清單。
- Mac 實跑的 loss_history：classification 第 16 步首次低於 0.01，objectness 約第 50 步接近 0，和頁面的約略說法相符。

4. 摘錄
- 摘錄比對工具在 repo 和副本上都輸出 []。validate_lessons 能一路跑到 reviews 那道斷言才停，表示 42 頁的摘錄檢查全部通過。
- 摘錄的 12 行順序和原檔一致，省略處都有標出並說明；正文沒有引用程式行號。

5. 建置
- 副本裡 `zensical build --clean --strict` exit 0（No issues found），`validate_site.py` exit 0。
- data-excerpt 區塊正常轉成 language-python 的程式區塊。
- 這次只改了 docs/lessons/07-training.md。相關的程式、notebook、紀錄和 SVG 都和 HEAD 相同；執行紀錄區塊和 HEAD 逐位元相同。

6. SVG（發布前的依賴，不是頁面問題）
- 追蹤中的 grid-learning-curve.svg 和 grid-learning-predictions.svg 有 viewBox、title、desc，qlmanage 渲染正常。
- 但它們還是舊版面，跟現在的 render_learning_evidence.py 和新頁面都對不上。原因是 grid-learning.json 還沒有 loss_history，要等 record_evidence.py --run 先訓練、再渲染。
- 我用追蹤中的預測，以現行渲染器重畫圖板再用 qlmanage 看：圖片 0 的 #0 是 TP、IoU 0.62，#1 是 class 0、score 0.980、FP、IoU 0.47，紅色真值標 FN，和頁面敘述完全一致。
- 用 Mac 紀錄重畫的曲線圖版面，也和頁面的讀法說明一致。
- 重錄後要再對一次圖片 0 那段的數字和表格。

證據檔都在副本旁：暫存副本（ql、ql-tracked、ql-repo 三個資料夾）。

## 來源對照

頁面上關於原始論文、官方程式與函式庫行為的說法，由 AI 打開頁面引用的來源（論文章節、固定 commit 的官方程式、官方文件）逐句核對。查閱的來源：

- https://arxiv.org/abs/1506.02640（YOLOv1，以 ar5iv HTML https://ar5iv.labs.arxiv.org/html/1506.02640 讀全文，即 arXiv 最新版 v5）：§2 Unified Detection（中心所在格負責；每格 B 個框與 confidence=Pr(Object)*IOU；沒有物件時 confidence 為 0、有物件時等於 IOU；每格一組條件類別機率；Pascal VOC 的設定 S=7、B=2、7×7×30）、式 (1)（測試時把類別條件機率乘上 confidence）、§2.1（最後一層用線性輸出）、§2.2 Training（xy 以格為基準、wh 以整張圖正規化，都落在 0 到 1；sum-squared error；λcoord=5、λnoobj=.5 的理由；預測寬高的平方根；由當下 IOU 最高的預測器負責）
- https://arxiv.org/abs/2004.10934（YOLOv4，以 ar5iv HTML https://ar5iv.labs.arxiv.org/html/2004.10934 讀全文）：§3.4〈Eliminate grid sensitivity〉，sigmoid 乘上大於 1 的係數（07-targets 提到這件事，但沒有附連結）
- https://pytorch.org/docs/stable/generated/torch.optim.Adam.html（轉址到 https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html）：演算法框（m_t、v_t、偏差修正、θ_t 更新式）與 betas 參數說明「running averages of gradient and its square」，預設值 (0.9, 0.999)
- https://github.com/cocodataset/cocoapi 的 commit 8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9，PythonAPI/pycocotools/cocoeval.py：第 506–508 行（iouThrs 是 .50:.05:.95 共 10 個門檻，recThrs 是 101 點）、第 370–376 行（沒有真值的類別直接跳過，precision 留在 -1）、第 396–410 行（先取 precision 包絡線，再在 101 個 recall 位置取值）、第 452–455 行（取平均時排除 -1）
- 本機 PyTorch 2.9.1（.venv-model）實測函式庫行為：cross_entropy 的 target 為 -1 時報 IndexError；類別放在最後一軸時報 RuntimeError（shape 不合）；mse_loss 與 cross_entropy 對空 tensor 取平均得到 NaN；torch.stack 的錯誤訊息開頭；對常數 0 呼叫 backward 會報錯；float32 的 sigmoid(-18)≈1.52e-8，sigmoid(-89)=0；24 附近的 ulp 是 1.9e-6，所以 24±w/2 會捨入成同一個數；inference_mode 產生的 tensor 不能拿去算梯度；squeeze 會刪掉所有長度為 1 的軸

這一頁沒有發現與來源不符的說法。

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 1 輪：獨立查核之後的編輯

GridDetector 連結 https://github.com/birdhackor/learn_to_yolo/blob/main/miniyolo/models.py 回 200，與站內紀錄連結同樣用 blob/main。miniyolo/models.py 的 GridDetector（三層 stride 2 卷積、AdaptiveAvgPool2d(4)、3×3 卷積、1×1 head、permute，以及 bias −1.8／−2）與頁面表格、bias 段落一致。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | 表格下一段末句：「完整程式見 [GridDetector 原始碼](…/blob/main/miniyolo/models.py)。」 | 同一頁的「完整程式」都指本節程式（Colab 最後一格，例如第 45、78、84 行），這裡卻用來指 models.py 裡的模型定義。初學者容易以為本節的完整程式就是 models.py。 | 已修正：改成「GridDetector 的完整定義在 miniyolo/models.py」。 |

### 第 2 輪：上一輪的處理與審查紀錄：通過

「GridDetector 的完整定義在 miniyolo/models.py」屬實（class GridDetector 在 miniyolo/models.py 第 53 行），連結回 200，和頁面其他 blob/main 連結一致，修好「完整程式」用詞混淆的發現，句子通順。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/07-training.md〈獨立查核〉 | 內部用語：「inventory trace 與 page impact 每一條都處理了」「正好就是 fixer 列的 …」。 | 未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 3 輪：上一輪的處理與審查紀錄：通過

以腳本核對紀錄：獨立查核通過；〈來源對照〉列出來源；〈後續編輯的檢查〉兩輪都有發現與處理；結構檢查通過。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/07-training.md 第 11、23、48 行 | 〈上述處理〉第 1 項寫「已修正：內部用語換成白話」，但產生器把「inventory trace 與 page impact」逐字換成「清單先前審查意見與受程式改動影響的段落每一條都處理了」，句子不通；另有「所有驗證都在副本暫存副本裡做」「證據檔都在副本旁：暫存副本.lesson.stdout、暫存副本、暫存副本.build.log…」。 | 未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 4 輪：上一輪的處理：通過

第 3 輪第 1 項：第 11 行「暫存副本」已清理；第 23 行病句已通順；第 48 行的檔名殘句已重寫。處理說明屬實。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/07-training.md 第 48 行與第 3 輪第 1 項處理欄 | 第 48 行「證據檔都在副本旁：暫存副本（ql、ql-tracked、ql-repo 三個資料夾）」仍以佔位字當路徑。處理欄「「先前審查意見」整組換成「先前審查意見」」是替換工具造成的同語反覆，讀不出改了什麼。 | 已處理：第 3 輪的處理說明改成統一的說明。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 5 輪：上一輪的處理：有必要問題

第 4 輪第 1 項講第 48 行的部分屬實：「證據檔都在副本旁：暫存副本（ql、ql-tracked、ql-repo 三個資料夾）」仍在。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | 第 4 輪第 1 項處理欄（第 96 行） | 發現另外點名第 3 輪第 1 項處理欄的同語反覆「「先前審查意見」整組換成「先前審查意見」」。該格已重寫，同語反覆已經不在，處理欄卻把兩個「先前審查意見」也列為「未改、仍在紀錄裡」；這個字串只出現在正文第 23 行。 | 已處理：用詞類的處理說明改成統一的說明（紀錄保留查核者的原文，只統一替換路徑與內部名稱），不再逐句計數。 |
| 2 | 建議 | 第 3 輪第 1 項處理欄（第 88 行） | 「點名的文字已不在紀錄裡」也涵蓋了引用的第 2 輪處理說法「已修正：內部用語換成白話」，但它仍在第 80 行。 | 已處理：用詞類的處理說明改成統一的說明（紀錄保留查核者的原文，只統一替換路徑與內部名稱），不再逐句計數。 |

## 紀錄重產後的檢查

2026-10-05，由另一位 AI 在獨立 rsync 副本查核 CPU 重產後的完整頁面、程式與 import 模組、新逐節紀錄、`grid-learning.json`、兩張實測 SVG 與既有審查紀錄。結論：通過，沒有必要問題或需修改的建議。原審查所列的「重錄後再對數字與圖」已完成。

- 使用 Python 3.12.14、PyTorch 2.9.1+cpu，實跑 `lesson_cases/07-training.py`：全部 assertion 通過，stdout 與新紀錄逐字相同；正文 step 0–2 的四位小數、total 手算、正負格平均與 box 微升／total 下降皆仍正確。shape hooks、資料／target／mask 及 loss 逐項手算都相符；正格 4、負格 60、4 個物件都是 class 0。梯度方向、共用 objectness bias 約 +0.058 的解釋與 Adam 更新相符。
- 照頁面移除唯一的 `optimizer.step()` 後重新執行：三行數字完全相同，最後檢查參數改變的 assertion 失敗；後面的正格 assertion 與成功行沒有執行，符合參考答案。三步後重新 forward 的正格 score 為 0.0774～0.0875，0.25 顯示門檻下沒有框；正文的診斷順序與「三步不宣稱泛化」說明成立。
- 實跑 `python -m miniyolo.train --steps 160 --samples 32 --device cpu`，160 筆 loss、四張 validation 圖的真值／預測及全部評估結果與新作者紀錄完全相同。validation mAP50 0.803571428571、test 0.774891774892；validation 為 18 GT／16 預測／15 TP，test 為 19 GT／17 預測／15 TP。初始 256 預測、3 TP、mAP50 0.001750700280 也重新核對。正文表格、計數、固定規則、overfit 與獨立資料評估的區分，以及單一 seed／小資料／合成任務的限制均成立。
- 用現行 `scripts/render_learning_evidence.py` 只讀新作者 JSON 重畫兩圖，與網站 SVG 逐位元相同；曲線的 160 點、未乘 5 的 box、更新前 loss 與配色正確。classification 首次 <0.01 在第 16 點；objectness 第 50 點約 0.0147，符合讀圖說明。圖板的 PNG 與 seed 700 對應圖片一致；圖片 0 的 #0 是 TP、IoU 0.6238→0.62，#1 是 class 0、score 0.9804→0.980、FP、IoU 0.4734→0.47，紅色 GT 是 FN，和正文完全相符。
- 新作者紀錄是 Intel Xeon Platinum 8573C、2 threads、PyTorch 2.9.1+cpu，訓練迴圈 1.184570304 秒，正文顯示 1.18 秒正確。此次獨立重跑為 6.290244746 秒，另行記錄，未替換作者紀錄；正文對計時範圍與不同機器不可外推的說明正確。
- strict Zensical build 與 `validate_site.py` 通過；本頁摘錄與 notebook code／output 皆相符。以 Chromium `--no-sandbox --disable-gpu --disable-dev-shm-usage` 在副本 HTTP server 渲染並實看兩張 SVG 與頁面，中文、標籤、框、曲線及程式區塊正常。從指定初學者角度逐段看過，沒有發現理解上的必要問題。
- 原始來源重新核對：YOLOv1 arXiv v5 原文；PyTorch v2.9.1 `torch/optim/adam.py` 的官方 API docstring／更新式與 `torch/nn/modules/pooling.py` 的 `AdaptiveAvgPool2d` docstring；COCO `cocoeval.py` 固定 commit `8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9` 的十個 IoU 門檻、101 recall 位置與平均規則。PyTorch 2.9 網頁文件在此環境回 403，因此改讀同版本官方原始碼內的 API 文件。沒有發現正文與來源不符的說法。

查核與執行都在獨立副本，主要 checkout 只讀；本次沒有修改教材、程式、作者紀錄、圖或 review coverage。


## 2026-10-05 clear-tutorial 三輪重審

以上是原審查歷史；不追溯改成首次盲讀。這次由固定基線 `16f6910` 分段開放並保存當時理解，再修改、核技術及檢查銜接，詳見 [本輪方法與限制](clear-tutorial/16f6910/README.md)。

- 第一輪：[grid當場閱讀原始紀錄](clear-tutorial/16f6910/first-read/grid.jsonl)，基線來源與圖指紋保留；共享檔案系統不是技術隔離。
- 第二輪：[非作者技術／證據核對](clear-tutorial/16f6910/technical/grid.md)，實際來源、數字及必要執行範圍見該報告。
- 第三輪：[另一位讀者前文→本節→後文複查](clear-tutorial/16f6910/transitions/grid.md)，此輪完整頁閱讀非盲讀；受影響段落及圖另有delta核回。
- [原始卡點與具體處理](clear-tutorial/16f6910/decisions.json)保留未新增的選讀建議。原先前提包漏發及08提前brief的限制另列，沒有算成教材錯或冒稱08全程盲讀。

本輪修正後沒有未解的必要問題；這是AI閱讀／技術查核的實際範圍，不是學生學習成效驗收。全站實際Zensical桌面／手機、公式換頁與執行檢查見 [verification.json](clear-tutorial/16f6910/verification.json)，不以SVG檔存在或strict build取代視覺查核。

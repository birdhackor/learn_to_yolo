# 審查紀錄：一次學習的超短暖身

審查範圍：`docs/lessons/00-warmup.md`，以及 `lesson_cases/00-warmup.py` 與它 import 的 repo 模組；頁尾自動產生的執行紀錄區塊不在範圍內，由 `scripts/validate_curriculum_evidence.py` 對照紀錄檢查。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面依目前的程式改寫後，由另一位 AI 獨立查核：在獨立的副本執行該節程式、照頁面做練習，逐句對照程式、執行紀錄與手算，檢查程式摘錄與網頁轉換，並從初學讀者（高中程度、數學好、程式新手）的角度看用詞與說明順序。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：通過

結論：通過。沒有必要等級的問題，只有一條 should：輸出說明段的句子順序有點跳。所有指令都在我自己的暫存副本執行，repo 內沒有執行任何東西。

1. 頁面對程式的描述全部屬實
- lesson_cases/00-warmup.py 與 HEAD 逐字相同，SHA-256 是 7929dea8…。
- 在暫存副本執行：exit 0，stderr 是空的，stdout 與頁面「核對輸出」的 4 行逐字相同。
- 照頁面指示一步一步做練習 1：
  - stepA，只改主 optimizer 的 lr=0.25：前 3 行照樣印出，第 3 行是 `weight: 1.00 -> 3.00; new_prediction=6.00; new_loss=4.00`，接著在第二行斷言出現 AssertionError。
  - stepB，再改第二行斷言：換成第三行斷言報錯。
  - stepC，三行都改：exit 0，第 4 行也印出來。
  - 三種情況都和頁面的說法、參考答案一致。
- 另外逐條驗證了正文的其他說法，都成立：
  - 把 alias／snapshot 插在 step 之前：alias 變成 1.8，snapshot 仍是 1.0，只寫 detach() 也會變成 1.8；backward 後印出的梯度是 `tensor([[-8.]])`。
  - 不包 no_grad 就 fill_ 會出現 RuntimeError；連做兩次 backward，梯度累加成 −16。
  - 預測先 detach、轉成 NumPy、或先 .item() 再算 loss，在本例都會報錯。
  - zero_grad 的 set_to_none 預設是 True。
  - after 是 1.7999999523162842，new_prediction 是 3.5999999046325684，兩個 `==` 都不成立。
  - eval 模式下 requires_grad 仍為 True；只呼叫 backward 時 w 不變；lr=0.125 時 w=2.0。
  - 全課程程式都沒有用 BatchNorm 或 Dropout；第 7 章確實用 Adam。
  - notebook 裡確實有「本節可修改的完整實驗」這一格。

2. 清單與受程式改動影響的段落都處理了
- 受程式改動影響的段落的 10 條全部處理：131、146、150、156、158、162（含 data-excerpt 標記）、165（依查核的更正改寫）、167、173；192 在自動紀錄區塊，未動，已確認與 HEAD 逐字相同。
- 假句「程式沒有印出更新後的預測值」已刪除。
- 先前審查意見第 3 行（Colab tag）：頁面已指向 lessons-v0.4.0，與 section-map 的 source_ref 一致。
- 先前審查意見第 1 行（「超短暖身」）：沒有改。改名要五處同步，超出只能改本頁的範圍，編輯已列入疑慮，我判定為已處理（說明了不改的理由）。
- 全頁搜尋過，沒有修訂、審查或製作經過的敘述。

3. 數字
- 新寫進頁面的數字都是確定值：3.60、3.0／6.0／4.0、3.5999999046…，都是 float32 的 IEEE 確定值，不是這台 Mac 量出來的。
- 等紀錄重產的值，編輯都已列出：4 行輸出，以及頁尾自動區塊目前還是舊的第 3 行和舊日期。

4. 摘錄
- 摘錄比對工具對 repo 和暫存副本都印出 []。
- 四個 data-excerpt 區塊我逐行對照過：`...` 省略的內容和註解說的一致，分別是 def main() 加兩行設定、gradient／before 兩行、三個 print 加三行斷言。
- 自己做了突變：改動摘錄引用的程式行，每個標記區塊都會被抓到；改動被省略的行不會報錯，這符合 `...` 省略的預期。
- 未標記的只有讀者自己插入的兩行小程式，不在程式裡。正文沒有引用程式行號。

5. 可讀性
- 新出現的詞都有說明：`...`、斷言、`.item()`、1e-5、new_prediction。斷言第一次出現在第 125 行，當場有解釋。
- 第 117 行的措辭配合摘錄中 new_loss 的 `.item()` 做了調整，前後不再看似矛盾。
- 唯一的問題是建議那條：第 156 行的段落順序。

6. 建置與改動範圍
- `zensical build --clean --strict` 通過（No issues found），validate_site 通過（markdown_rendering、colab_pairs、連結與錨點都 passed）。
- 在建置出的 HTML 裡，4 個 data-excerpt 區塊都正確渲染成 Python 程式區塊，沒有 Markdown 原文漏出來。
- 本節沒有列 SVG。git diff --check 乾淨，沒有未追蹤的檔案。
- 和 00-warmup 相關的其他檔案都沒被改：lesson、notebook、紀錄 JSON、section-map、learning-path、reviews。zensical.toml 和 docs/index.md 的改動是 status 頁改名，與本頁無關。

不擋發布、但要留意（編輯都已列入疑慮）：
- 頁尾紀錄區塊要等 verify_curriculum.py 重產；其中「完整紀錄（JSON）」連到 blob/main，Colab 按鈕卻釘在 tag。
- new_loss 和 after 的絕對容差，在 lr≥0.65 等學習率下，即使預測正確也會誤報；但頁面只要求 lr=0.25，那時的值是精確值，不受影響。
- reviews/00-warmup.md 不涵蓋新文字，發布前要重審。

證據檔：
- 練習變體 stepA／B／C.py、alias.py 與各自的輸出
- 摘錄突變用的副本
- 紀錄檔

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/lessons/00-warmup.md 第 156 行（「核對輸出」4 行輸出下方的說明段） | 新加的「第 2 行的 `prediction=2.00` 是更新前的預測；第 3 行的 `new_prediction=3.60` 是更新後的預測…」插在「第 2、3 行就是本節手算的數字」和「第 1 行的 `(1, 1)`…是同一個 shape 的不同印法」之間。結果整段的順序變成：第 1 行 → 第 2、3 行 → 第 2 行 → 第 3 行 → 再回到第 1 行的印法 → 斷言 → 第 4 行。讀者讀完第 3 行還得跳回第 1 行。內容沒有錯，也不會誤導，只是讀起來跳來跳去。 |

最後一次查核的建議事項，在下方〈定稿修正〉逐項處理。

## 讀者審查與技術查核

### 讀者審查（AI 以初學讀者身分閱讀、執行程式與練習）

方法：工作環境：所有動作都在暫存副本裡進行。副本用 rsync 從 repo 複製，排除 .git、site、.venv*、artifacts/runs、data/curated、data/downloads。我沒有在 repo 根目錄裡執行或寫入任何東西，結束前該 repo 的 git status 是空的。

【閱讀】
- 審查用的事實與寫作規範清單，以及 docs/lessons/00-warmup.md 全文。
- 用 zensical build --clean --strict 建站（exit 0、No issues found），讀 site/lessons/00-warmup/index.html 的正文。
- 用 headless Chrome 把所有摺疊區打開後截整頁圖，逐段看過：MathJax 公式、程式區塊、摺疊區都正常顯示。本頁沒有任何圖或 SVG，所以沒有圖要核對。
- 本節程式 lesson_cases/00-warmup.py。
- notebooks/00-warmup.ipynb，以及 scripts/build_lesson_notebooks.py 的格子範本（確認確實有「本節可修改的完整實驗」這一格）。
- docs/glossary.md。
- 讀者在本頁之前會讀到的頁面：docs/index.md、docs/learning-path.md、docs/status.md 的〈執行方式〉。
- README.md 的〈CPU 本機執行〉。
- 為了查證頁面說法另外看的：
  - docs/lessons/02-diagnostics.md 和 lesson_cases/02-diagnostics.py：確認第 2 章真的靠 None 梯度找出沒接上的參數。
  - 在 miniyolo／lesson_cases 搜尋 BatchNorm、Dropout：沒有。搜尋 Adam：第一次出現在 07-training.py。
  - scripts/validate_lessons.py 的摘錄比對規則：去掉註解和 `...` 後逐行比對。

【執行與嘗試】
- 本節程式：輸出和頁面〈核對輸出〉的 4 行逐字相同。
- 練習 1，完全照頁面做三次，結果都和參考答案一致：
  - 只把主 optimizer 改成 lr=0.25：前 3 行照樣印出，第 3 行是 `weight: 1.00 -> 3.00; new_prediction=6.00; new_loss=4.00`，接著第二行斷言報 AssertionError。
  - 再只改第二行斷言：換成第三行斷言報錯。
  - 三行斷言都改：全部通過，第 4 行印出。
- 延伸題 lr=0.125：印出 2.00／4.00／0.00。
- 練習 2：
  - backward 之後印出 model.weight，仍是 1。
  - 刪掉 `optimizer.step()`：第 3 行是 `weight: 1.00 -> 1.00`，接著第二行斷言報錯。
- 頁面要讀者插進程式的小實驗：
  - `print(model.weight.grad)` 印出 `tensor([[-8.]])`。
  - alias／snapshot 兩行插在 step 之前、step 之後印出：得到 `Parameter containing:`／`tensor([[1.8000]], requires_grad=True)` 和 `tensor([[1.]])`。
  - 把副本取名 before：出現 TypeError，印證頁面說不能取這個名字的理由。
  - 只用 `detach()` 存舊值：step 之後也變成 1.8。
- 另寫小程式核對頁面其他說法，全部成立：
  - 沒包 `torch.no_grad()` 就 `fill_`：RuntimeError。
  - 先 `.detach()`、轉成 NumPy、或先 `.item()` 再算 loss：三種都報錯。
  - 沒清梯度就再 backward 一次：梯度變成 −16。
  - after=1.7999999523162842、new_prediction=3.5999999046325684，剛好就是 float32 能存的 1.8 和 3.6。
  - PyTorch 2.9.1 的 `zero_grad` 預設 set_to_none=True。
  - eval 模式下輸出的 requires_grad 仍是 True，`no_grad` 裡面是 False。
  - 只重建 model、不重建 optimizer：w 一直是 1.0，loss 一直是 4.0。
  - 不加 `.mean()`：shape [1,1] 的 loss 照樣能 backward；shape [2,1] 會報 `grad can be implicitly created only for scalar outputs`。
- 照頁面原樣打本機指令 `PYTHONPATH=. python lesson_cases/00-warmup.py`：得到 `pyenv: python: command not found`（exit 127）；改用 python3 則是 `No module named 'torch'`。
- 驗證腳本：validate_site.py 通過。validate_lessons.py 的摘錄比對通過，整體會失敗只是因為審查紀錄還沒寫入。
- 延伸閱讀的兩個連結都回 200（會轉到 docs.pytorch.org）。

【沒有列為發現的事】
頁尾執行紀錄區塊的第 3 行少了 `new_prediction=3.60`。這份紀錄 JSON 綁定的 case_sha256 是 188068e6…，和目前程式的 7929dea8… 不同，表示它是用舊版程式跑的、正在重產，所以依指示不列為發現。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 必要 | 摺疊區「在自己的電腦執行」（docs/lessons/00-warmup.md 第 15、17 行）：`PYTHONPATH=. python lesson_cases/00-warmup.py`，以及 Windows 那句的 `python lesson_cases/00-warmup.py` | 頁面要讀者「依 README 安裝 PyTorch」，接著就打 `python …`。但 README 的安裝步驟是把 PyTorch 裝進 `.venv-model` 這個虛擬環境，而且沒有啟用它。所以照頁面原樣執行時，用到的不是那個環境的 Python。我在這台 Mac 的暫存副本照打，得到 `pyenv: python: command not found`（exit 127）；改用 `python3` 則是 `ModuleNotFoundError: No module named 'torch'`。初學者會以為 PyTorch 沒裝好而卡住。README 第 17 行和〈驗證範圍〉的〈執行方式〉都寫 `.venv-model/bin/python`；README 第 24 行也特別提醒，網頁上的 `python …` 要先 `source .venv-model/bin/activate`。本頁沒有轉述這一步。 |
| 2 | 建議 | 開頭「不想設定環境，就用頁首的『在 Colab 執行本節』按鈕。」（第 11 行） | 這是讀者第一次打開 Colab 的地方，但本頁和先讀的首頁、閱讀路線都沒說進了 Colab 之後要做什麼。缺的步驟有：怎麼執行一格；第一次執行從 GitHub 開啟的 notebook 時，Colab 通常會跳出「這份 notebook 不是 Google 編寫的」警告；要等環境格印出「固定教材版本： lessons-v0.4.0」才算準備好；PyTorch 不是 2.9.1 時，環境格要下載改裝，得等一會兒。另外，最後一格一打開就已經顯示存好的紀錄輸出，不是讀者自己跑出來的。notebook 第一格只寫「先執行下一格的環境格」。程式新手可能卡在第一步，或把存好的輸出拿去和〈核對輸出〉的 4 行比，以為自己已經完成。 |
| 3 | 建議 | 〈對應到五行程式〉第一段摘錄裡的 `... # 省略：def main(): 和兩行與手算無關的設定`（第 79 行），以及兩段摘錄之間（完整程式第 13 行） | 讀者在 Colab 對照完整程式時，會遇到三件頁面沒交代的事。(1) 被省略的兩行是 `torch.manual_seed(7)` 和 `torch.set_num_threads(2)`，頁面只說「與手算無關」，沒說它們是什麼。(2) 完整程式在 `optimizer.zero_grad(...)` 前一行還有 `model.train()`。這行夾在兩段摘錄之間，沒有 `...` 標記；頁面又說「下面五行就是訓練一步」，讀者會不確定它算不算訓練一步、是不是摘錄漏了。(3) 摘錄裡的中文註解是本頁加的，完整程式裡沒有，只有兩行英文註解。而且摘錄的程式頂格寫，完整程式卻縮排在 `def main():` 裡，讀者會找不到「同一行」。 |
| 4 | 建議 | 摺疊區「一般情況：多筆資料、多個輸入」第二、三段（第 31、33 行） | 同一段裡，中括號一下表示數值，一下表示 shape，而且都用一般文字顯示，看不出差別。「輸入是 [1,2,3]」「輸出是 [4,2]」是數值；「這裡是 [2,3]」「[2,3] 轉置後變成 [3,2]」「乘上 [3,2]，才得到 [1,2] 的輸出」是 shape。讀者容易把「乘上 [3,2]」讀成乘上數值 3 和 2，或以為輸出是 [1,2]，和前一段算出的 [4,2] 矛盾。權重矩陣本身和它的轉置也沒寫出來，讀者只能在腦中想像這次矩陣乘法。 |
| 5 | 建議 | 〈從輸入走到誤差〉第二段「shape 記錄每個軸有多長，每多包一層中括號就多一個軸。…第一軸長度 1…第二軸長度 1」（第 25 行） | 唯一的例子 `[[2.0]]` 每個軸長度都是 1。讀者看不出「軸的長度」怎麼數，也分不出哪一層括號對應第一軸。第 1 章起就要讀 `[B,C,H,W]`、`[8,3,64,64]` 這類 shape，這裡少了一個具體例子。 |
| 6 | 建議 | 〈從輸入走到誤差〉最後一段「下面程式用 .mean() 把所有數平均成一個數，軸就不見了」（第 41 行），以及〈對應到五行程式〉「mean 表示在多筆資料時取平均；本例一筆，所以與手算相同」（第 106 行） | 頁面強調 loss 是 scalar，卻沒說為什麼 loss 要整理成一個數。本例只有一筆資料，拿掉 `.mean()` 結果完全一樣：我實測 shape `[1,1]` 的 loss 照樣能 backward，梯度也是 −8。讀者會以為 `.mean()` 可有可無。但兩筆資料時不先平均，`loss.backward()` 會報 `RuntimeError: grad can be implicitly created only for scalar outputs`（我用 shape `[2,1]` 實測過）。 |
| 7 | 建議 | 〈對應到五行程式〉「backward 之後，可以用 print(model.weight.grad) 看梯度…」那一大段，以及其後 alias／snapshot 兩行程式的註解（第 108～113 行） | 照頁面把兩行插在 step 之前，step 之後自己加 `print(alias)`、`print(snapshot)`，實際印出的是 `Parameter containing:`、`tensor([[1.8000]], requires_grad=True)` 和 `tensor([[1.]])`，而且出現在原本 4 行輸出的前面。頁面沒說 alias 會多印「Parameter containing」和 `requires_grad=True`（requires_grad 要到下一節才解釋），也沒說 1.0 會印成 `1.`、1.8 會印成 `1.8000`。初學者可能以為自己做錯了。另外，這一段同時講了印梯度、別名與副本、Python list 類比、插入位置、為什麼不取名 before、縮排，句子很長，要做的動作散在不同句子裡。 |
| 8 | 建議 | 〈自主練習與答案〉練習 2「只呼叫 backward，會改變 w 嗎？」（第 176 行）與參考答案（第 184 行） | 這題的答案本頁第一段就直接講了（「還沒有。這一步只算出梯度」）。它也不像練習 1 有可以動手驗證的步驟，讀者只能憑記憶回答，練不到「用程式確認」。 |
| 9 | 建議 | 〈重建模型時，optimizer 也要重建〉第一段「在 Colab 這類可以分格執行的 notebook 裡，若只重跑 model = ... 那一格」（第 141 行） | 本節 notebook 的完整程式整段都在最後一格的 `main()` 裡，`model = ...` 和 `optimizer = ...` 在同一格。重跑那一格會兩者一起重建。所以讀者在本節的 Colab 找不到「model = ... 那一格」，也重現不了這個錯誤，可能以為自己找錯了格子。 |
| 10 | 建議 | 〈自主練習與答案〉「用來容許小數計算的微小誤差。例如 after 實際上是 1.7999999523…，new_prediction 是 3.5999999046…」（第 172 行） | 數學好的讀者會問：1−0.1×(−8) 明明剛好是 1.8，為什麼程式算出 1.7999999523…？頁面只說「小數計算的微小誤差」，沒說原因。我實測，這個值正是 float32 能存的、最接近 1.8 的數（3.5999999046… 也是 float32 的 3.6）。差異來自小數在電腦裡的存法，不是算錯。 |
| 11 | 建議 | 〈梯度在回答「旋鈕多轉一點，loss 怎麼變」〉「第二段令誤差 e=ŷ−y，於是 L=e²：(e+Δe)²−e²=2eΔe+(Δe)²。兩邊除以 Δe 得到 2e+Δe」（第 53 行），以及下一段「ΔL≈2eΔŷ」（第 55 行） | 頁面沒有說 (e+Δe)²−e² 就是 L 的變化量 ΔL。「兩邊除以 Δe」之後，左邊變成 ΔL/Δe，這也沒寫出來，讀者要自己補上「這是在算 L 的變化量除以 e 的變化量」。下一段用了「≈」，卻沒交代為什麼是約等於（因為略去了 (Δŷ)² 這一項），而前一段剛用的是「=」。 |
| 12 | 建議 | 〈對應到五行程式〉zero_grad 那一段（第 104 行）：「set_to_none=True 把 .grad 清成 None…目前的 PyTorch 預設就是 True。對下一次 backward 有算到梯度的參數來說…曾有梯度、這次沒算到梯度的參數則不同…」 | (1)「目前的 PyTorch」這種說法會隨時間失準，而本課程固定用 PyTorch 2.9.1。(2) 既然預設就是 True，讀者會問程式為什麼還特地寫出來，頁面沒說。(3)「曾有梯度、這次沒算到梯度的參數」沒有例子。本例只有一個參數，讀者想像不出參數什麼時候會「沒算到梯度」。 |
| 13 | 建議 | 〈從輸入走到誤差〉第二段「PyTorch 程式把資料和參數（本例的 x、目標、權重）裝在 tensor（張量）裡」（第 25 行）；摺疊區與〈對應到五行程式〉的「線性層」「層」（第 29、75 行）；〈梯度…〉說明 loss.backward() 的段落（第 65 行） | 「權重」第一次出現時，頁面沒說它就是前一段的參數 w；要到〈對應到五行程式〉的對照表「w→model.weight」才明說，讀者在那之前不確定兩者是不是同一個東西。「層」整頁沒有定義，而「線性層」的定義又用「…的層」來解釋自己。另外，術語表的 gradient 一列把 backward 叫「反向傳播」，第 2、3 章用「反傳」，本頁只寫 backward，讀者連不起來。ŷ、η 第一次出現時也沒有讀法。 |
| 14 | 建議 | 開頭第一段（第 5 行）：「程式跑完一次 loss.backward()，模型已經學到了嗎？還沒有。這一步只算出梯度；optimizer.step() 才修改參數。optimizer（優化器）是專門拿梯度去修改參數的物件。」 | 第一句就用到 `loss.backward()`、梯度、`optimizer.step()`、參數、物件，這些都要到下一段或更後面才解釋。本頁說「沒寫過 PyTorch 也能先讀完手算」，但這樣的讀者讀第一段只會看到一串陌生名詞，抓不到這節要回答的問題。 |
| 15 | 建議 | 〈梯度…〉最後一段「這次下降有手算證據；一般模型每步或每批的 loss 不保證都下降」（第 69 行）；〈核對輸出：看到這些就完成本節〉最後一段「收益是每一步都能追到具體數字；代價是這個線性問題不含影像、非線性與泛化…」（第 158 行） | 「這次下降有手算證據」語意不清。「收益／代價」前面沒有主詞，又緊接在「之後章節…改用 Adam」後面，讀者會以為在講之後章節的收益和代價。「問題不含泛化」也不通順：泛化是模型的能力，不是問題裡包含的東西。 |

### 技術查核（AI 對照原始論文、固定 commit 的官方程式、該節程式與手算）

方法：讀過的檔案（暫存副本，由 rsync 建立）：docs/lessons/00-warmup.md 全文、lesson_cases/00-warmup.py（只 import torch，沒有 miniyolo 模組）、審查用的事實與寫作規範清單、README.md 第 1–30 行、notebooks/00-warmup.ipynb 各格（第 3 格標題是「## 本節可修改的完整實驗」）、scripts/build_lesson_notebooks.py 第 142 行、scripts/validate_lessons.py（摘錄比對規則）、section-map.json、docs/glossary.md（術語一致）、docs/lessons/02-diagnostics.md 與 lesson_cases/02-diagnostics.py（set_to_none／None 梯度、detach 不報錯）、lesson_cases/07-training.py 與 miniyolo/train.py（使用 Adam）、docs/lessons/03-comparison.md、07-inference.md、14-feature-module.md、15-area-attention.md、16-inference-head.md、16-training.md（BN 與動量的說法）、reviews/00-warmup.md。另外 grep 了 miniyolo、lesson_cases、scripts 裡的 BatchNorm／Dropout（沒有使用）與各節的 optimizer。

一手來源：PyTorch 官方 repo 的 tag v2.9.1，commit d38164a545b4a4e4e0cf73ce67173f70574890b6（用 gh api 取得，與 .venv-model 裡安裝的 torch 2.9.1 原始碼逐檔 diff 相同）：
- https://github.com/pytorch/pytorch/blob/d38164a545b4a4e4e0cf73ce67173f70574890b6/torch/optim/optimizer.py 第 998–1014 行：`def zero_grad(self, set_to_none: bool = True)`；文件第 3 點 \"torch.optim optimizers have a different behavior if the gradient is 0 or None (... in the other it skips the step altogether)\"。
- torch/optim/sgd.py：第 84–103 行 `_init_group` 的 `if p.grad is not None:`；第 367–374 行 `param.add_(grad, alpha=-lr)`；SGD.__doc__ 的演算法（weight decay、momentum 分支與 \\(\\theta_t\\leftarrow\\theta_{t-1}-\\gamma g_t\\)）；預設值 momentum=0、dampening=0、weight_decay=0、nesterov=False。
- torch/nn/modules/linear.py：第 54 行 \"y = xA^T + b\"；第 73–76 行 weight shape (out_features, in_features)、初始值取自 U(−√k, √k)；第 117–124 行 kaiming_uniform_。
- torch/nn/modules/module.py 第 2872–2910 行：train()／eval() 的文件 \"This has an effect only on certain modules\"。
- torch/csrc/autograd/VariableTypeUtils.h 第 80–84 行：\"a leaf Variable that requires grad is being used in an in-place operation.\"
- torch/_tensor.py 第 573–581 行：\"This function accumulates gradients in the leaves\"。
- torch/autograd/__init__.py 第 198–201 行：只有 numel 為 1 時才能隱式建立 grad。
- torch/nn/modules/batchnorm.py 第 180–181 行與第 290–291 行。
- torch/nn/modules/dropout.py 第 36–49 行。
- torch/autograd/grad_mode.py 第 21–29 行。
- set_to_none 預設值的變更：tag v1.13.1（commit 49444c3e546bf240bed24a101e747422d1f8a0ee）的 torch/optim/optimizer.py 是 `set_to_none: bool = False`，tag v2.0.0（commit c263bd43e8e8502d4726643bc6fd046f0130ac0e）改成 `True`。

官方文件：
- https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html（頁面上的 pytorch.org 連結會 301 轉到這裡）：\"In a forward pass, autograd does two things simultaneously...\"，以及 backward 那段 \"computes the gradients from each .grad_fn, accumulates them in the respective tensor's .grad attribute, and using the chain rule, propagates all the way to the leaf tensors\"。
- https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html（stable 目前指向 2.14）與 https://docs.pytorch.org/docs/2.9/generated/torch.optim.SGD.html：簽名與 θt←θt−1−γgt，兩版相關內容相同。
- https://docs.pytorch.org/docs/2.9/notes/autograd.html〈Locally disabling gradient computation〉的 Evaluation Mode 與 No-grad Mode 兩小節。

執行的程式與指令（全部在暫存副本中執行，沒有寫入 repo 根目錄）：
1. `PYTHONPATH=. OMP_NUM_THREADS=2 MPLBACKEND=Agg .venv-model/bin/python lesson_cases/00-warmup.py`：exit 0，印出的 4 行與頁面〈核對輸出〉逐字相同（torch 2.9.1，Python 3.12.15）。
2. 摘錄比對：結果是 []。另外人工核對了 4 段摘錄的行序與 `...` 省略說明。
3. 自寫 probe.py，逐項實測頁面說法：
   - 不包 no_grad 時 fill_ 報 RuntimeError；backward 前 grad 是 None；loss.shape 是 torch.Size([])；`print(model.weight.grad)` 印出 `tensor([[-8.]])`；`print(x.shape)` 印出 `torch.Size([1, 1])`。
   - step 後 alias 是 1.8000、snapshot 是 1、只 detach 的副本也變成 1.8。
   - after=1.79999995231628417969、new_prediction=3.59999990463256835938、new_loss=0.1600000709；用 == 比較時兩者都是 False。
   - 不清梯度、連做兩次 backward 時 grad 是 −16，step 後 w=2.6。
   - 斷圖：detach 報 RuntimeError；`.numpy()` 報 RuntimeError；`.detach().numpy()` 與 `.item()` 在呼叫 backward 時報 AttributeError。
   - eval 下輸出的 requires_grad 是 True，no_grad 下是 False。
   - 重建模型但沿用舊 optimizer：3 步都不報錯，w 不變，loss 固定在 6.657573。
   - set_to_none 時，沒算到梯度的參數 grad 維持 None，step 也跳過它。
4. 練習 1 的各種改法：只改成 lr=0.25 時，先印出 3 行（weight: 1.00 -> 3.00; new_prediction=6.00; new_loss=4.00），再在第二行斷言 AssertionError；只改第二行斷言時，換成第三行斷言報錯；三處都改時印出 4 行、exit 0；lr=0.125 時 w=2.00、new_loss=0.00。
5. 照頁面說明插入 alias／snapshot 兩行：輸出符合頁面敘述。
6. `.venv-docs/bin/zensical build --clean --strict`：exit 0，No issues found。`python3 scripts/validate_site.py`：exit 0。另外抽出渲染後的 7 個程式區塊，並確認 59 個數學式都正確渲染、沒有殘留的 \\(。
7. `python3 scripts/validate_lessons.py`：所有摘錄檢查通過，只在審查涵蓋那一步失敗，屬於暫時狀態。`python3 scripts/validate_curriculum_evidence.py`：00-warmup 的紀錄過期，屬於重產中的暫時狀態，未列為發現。
8. 用 `gh api` 取得 tag 的 SHA 與各原始碼檔，並與已安裝的 torch 原始碼 diff。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | 「在自己的電腦執行」摺疊區（docs/lessons/00-warmup.md 第 15、17 行） | 頁面只說「依 README 安裝 PyTorch」，接著要讀者執行 `PYTHONPATH=. python lesson_cases/00-warmup.py`，Windows 則執行 `python lesson_cases/00-warmup.py`。但 README 是把 PyTorch 裝進 `.venv-model`，而且用完整路徑執行；README 第 24 行另外寫「要照網頁上的 `python ...` 命令執行，先用 `source .venv-model/bin/activate` 啟用環境，或一律改用完整路徑 `.venv-model/bin/python`。Windows 改用 `.venv-model\Scripts\python.exe`」。讀者照 README 的安裝區塊做完、直接打本頁的 `python` 時，用到的是系統的 Python，會出現 `ModuleNotFoundError: No module named 'torch'`；macOS 上可能直接是 `command not found: python`。Windows 那句寫「照 README…再執行 `python …`」，也和 README 的 Windows 指示（改用 `.venv-model\Scripts\python.exe`）不一致。 |
| 2 | 建議 | 〈兩個常混在一起的開關〉第二段（第 123 行）「BatchNorm（用平均與變異數把數值標準化的層）在 train 模式會更新它記錄的平均與變異數」 | 這句只寫了 train 模式會更新紀錄，漏了真正讓輸出不同的差別：train 模式用這一批資料自己的平均與變異數來標準化，eval 模式改用記錄下來的值。PyTorch v2.9.1 torch/nn/modules/batchnorm.py 第 180–181 行："Mini-batch stats are used in training mode, and in eval mode when buffers are None."；第 290–291 行："during training this layer keeps running estimates of its computed mean and variance, which are then used for normalization during evaluation." 只讀這句，讀者會以為 BatchNorm 在兩種模式下輸出相同、差別只在要不要更新紀錄。這和後面各節的說法不一致：07-inference 寫「有 Dropout／BatchNorm 的模型，輸出會和推論模式不同」，15-area-attention 寫「BN 在訓練模式下會用整批資料的平均與變異數」。 |
| 3 | 建議 | 〈對應到五行程式〉最後一段（第 117 行）「在本例，這三種做法都會讓程式報錯」 | 頁面只說本例會報錯，沒說為什麼：本例只有 w 一個參數，計算圖一斷，loss 就完全不連著任何參數。一般模型裡，loss 若還有一部分連著計算圖，剪斷其中一段不會報錯，只是那一段的參數 `.grad` 停在 None、學不到東西。第 2 章程式就是這種情況：lesson_cases/02-diagnostics.py 第 12–16 行把 `body(x).detach()` 接進 head 後 backward 沒有報錯，只有 `body.weight.grad is None`。讀者若把「斷圖就會報錯」當成一般規則，在真正的模型裡會漏掉這種不報錯的失敗。 |
| 4 | 建議 | 〈梯度在回答「旋鈕多轉一點，loss 怎麼變」〉最後一段（第 69 行）「SGD 用學習率 η=0.1 控制步伐。學習率是倍率：每步從 w 減掉的量是 η 乘上梯度」 | \(w-\eta g\) 只在 torch.optim.SGD 用預設值時成立（momentum=0、dampening=0、weight_decay=0、nesterov=False）。PyTorch v2.9.1 torch/optim/sgd.py 的 SGD 文件寫明：λ≠0 時 \(g_t\leftarrow g_t+\lambda\theta_{t-1}\)，μ≠0 時改用動量緩衝 \(b_t\)，最後才做 \(\theta_t\leftarrow\theta_{t-1}-\gamma g_t\)。頁面把這條式子寫成 SGD 本身的做法，沒說它依賴預設值；本頁連結的官方 SGD 說明頁就列著這些選項。第 16 章〈16-training〉才寫「本例的 SGD 是最基本的版本…沒有加其他機制（例如…動量）」。讀者之後用 `momentum=0.9` 這類設定時，會誤以為每步仍只減 η×梯度。 |
| 5 | 建議 | 〈對應到五行程式〉第 73 行的摘錄說明，與第 77–100 行兩段摘錄 | (a) 完整程式第 13 行的 `model.train()` 夾在兩段摘錄之間，摘錄和正文都沒提到它；〈兩個常混在一起的開關〉只說完整程式「在更新參數之後」用到 eval 與 no_grad。讀者打開 Colab 看到 `model.train()`，在頁面上找不到對應說明。PyTorch 2.9 autograd notes〈Locally disabling gradient computation → Evaluation Mode〉也建議 "always use model.train() when training and model.eval() when evaluating"。(b) 摘錄裡的中文 `#` 註解（例如 `# Linear(1, 1)：每筆輸入 1 個數…`、`# forward`）是本頁加的，完整程式裡沒有；摘錄也拿掉了 `def main():` 底下的 4 格縮排。第 73 行只說「完整程式還多了幾行」，沒說摘錄另外加了註解、改了縮排，讀者拿來對照 Colab 時，可能以為兩邊不是同一份程式。 |

各項的處理見下方〈定稿修正〉。

## 定稿修正

上面各項意見與先前查核留下的建議，由 AI 逐項核實後處理：必要問題全部修正，建議事項只在修正明確、範圍小時採用。

| # | 來源 | 意見 | 處理 |
|---|---|---|---|
| 1 | 審查意見 | 讀者 1（必要，本機指令） | 已改。指令換成 `PYTHONPATH=. .venv-model/bin/python lesson_cases/00-warmup.py`，並說明 `.venv-model/bin/python` 是什麼、只打 `python` 會出現哪兩種錯誤，以及之後各節的 `python …` 要先 `source .venv-model/bin/activate`（只對當下的終端機視窗有效）或換成完整路徑。Windows 改成先設 `$env:PYTHONPATH='.'`，再執行 `.venv-model\Scripts\python.exe …`。已在暫存副本重現原本的兩種錯誤，也確認 venv 的 bin 裡有 `python`。 |
| 2 | 審查意見 | 讀者 2（第一次用 Colab） | 已改。新增摺疊區〈第一次用 Colab〉四步：登入；用執行鈕或 Shift+Enter 執行環境格，遇到警告選擇仍要執行（按鈕名稱以 Colab 當時的畫面為準）；等「固定教材版本： lessons-v0.4.0」那一行印出；再執行最後一格，並說明打開時已顯示的是執行紀錄存下的輸出。『只要印出這一行就是成功』已對照環境格程式確認：每個步驟都用 check=True 或 raise，任何一步失敗就印不到這一行。 |
| 3 | 審查意見 | 讀者 3／技術 5（摘錄和完整程式的差異） | 已改。摘錄前補上：摘錄裡的註解都是本頁加的；除了 `import torch`，其餘各行在完整程式裡都寫在 `def main():` 底下，多縮排 4 格。第一段摘錄後補一段，說明 `manual_seed(7)` 與 `set_num_threads(2)` 的作用，以及兩者為何都不影響印出的數字。五行摘錄前說明完整程式先呼叫 `model.train()`，而新建的模型本來就在訓練模式。 |
| 4 | 審查意見 | 讀者 4（數值和 shape 都用中括號） | 已改。摺疊區裡的 shape 一律寫成「shape `[…]`」，數值改寫成「1、2、3 三個數」「4 和 2」，並用一個公式區塊寫出 W、Wᵀ，以及 [1 2 3] 乘 Wᵀ 等於 [4 2]。 |
| 5 | 審查意見 | 讀者 5（shape 只舉了各軸都是 1 的例子） | 已改。補上「以兩層括號為例」怎麼數兩個軸的長度，並舉 `[[1.0,2.0,3.0],[4.0,5.0,6.0]]`，shape 是 `[2,3]`。 |
| 6 | 審查意見 | 讀者 6（loss 為什麼要是一個數） | 已改。`mean` 段補上兩個原因（才比得出變小了沒、backward 要從這一個數往回算），以及多筆資料沒有併成一個數時的報錯訊息。已實測：shape `[2,1]` 會報錯，`[1,1]` 不會；頁面沒有宣稱 `[1,1]` 會報錯。 |
| 7 | 審查意見 | 讀者 7（alias／snapshot） | 已改。這段拆成三部分：別名與副本的差別；要插入的程式（含原有的 `optimizer.step()`，註明不要重複加）和實際多出的 3 行輸出；`Parameter containing:`、`requires_grad=True`、`1.`、`1.8000` 這幾種印法的說明。已把程式插進完整程式實跑，輸出與頁面所列相同。 |
| 8 | 審查意見 | 讀者 8（練習 2 沒有動手驗證） | 已改。題目加上刪掉 `optimizer.step()`（或在行首加 `#`）再執行。參考答案補上實測輸出 `weight: 1.00 -> 1.00; new_prediction=2.00; new_loss=4.00`、第二行斷言報錯的原因，以及第一行斷言照樣通過的原因。 |
| 9 | 審查意見 | 讀者 9（重建 optimizer 的情境在本節 notebook 重現不了） | 已改。改成「建模型與建 optimizer 寫在不同格時」才會發生，並補一句：本節的完整程式把兩者寫在同一格的 main() 裡，每次執行都一起重建。 |
| 10 | 審查意見 | 讀者 10（float32 誤差的原因） | 已改。補上原因：電腦用二進位、有限的位數存小數，float32 約只有 7 位有效數字。已實測 1.7999999523162842 與 3.5999999046325684 就是 float32 能存的、最接近 1.8 與 3.6 的數。 |
| 11 | 審查意見 | 讀者 11（推導沒寫出 ΔL、沒交代 ≈） | 已改。寫出 ΔL＝(e+Δe)²−e² 與 ΔL/Δe，再寫出 ΔL＝2eΔŷ+(Δŷ)²，說明 ≈ 略去的是 (Δŷ)²，它除以 Δw 後是 x²Δw，會趨近 0。另補一句：−7.96 和 −8 差的 0.04，正是 x²Δw＝4×0.01。 |
| 12 | 審查意見 | 讀者 12（zero_grad 段） | 已改。「目前的 PyTorch」改成「本課程用的 PyTorch 2.9.1」，說明程式仍明確寫出來的作用，並為「這次沒算到梯度的參數」加上例子。 |
| 13 | 審查意見 | 讀者 13（術語第一次出現） | 已改。權重第一次出現時說明它就是 w（weight）；在〈對應到五行程式〉定義「層」，摺疊區的線性層也不再用「…的層」解釋自己；backward 補上「反向傳播，也簡稱反傳」，和術語表、第 2 章一致；ŷ、η 附上讀法 y hat、eta。 |
| 14 | 審查意見 | 讀者 14（開頭第一段名詞太多） | 已改。先用白話問「算出參數往哪邊調之後，就算學到了嗎」，再對應到 `loss.backward()` 與 `optimizer.step()`。 |
| 15 | 審查意見 | 讀者 15（措辭） | 已改。「這次下降有手算證據」改成「這一步 loss 從 4 降到 0.16，每個數都能手算核對」；「收益／代價」那句改成「用一個參數的小例子，好處是…；限制是它沒有影像、沒有非線性的計算，也測不出泛化…」。 |
| 16 | 審查意見 | 技術 1 | 和讀者 1 是同一件事，一起處理。 |
| 17 | 審查意見 | 技術 2（BatchNorm） | 已改。改成 train 模式用這一批資料的平均與變異數來標準化，同時更新記錄的值；eval 模式改用記錄的值，所以同一筆輸入在兩種模式的輸出可能不同。已用 BatchNorm1d 實測，兩種模式的輸出確實不同。 |
| 18 | 審查意見 | 技術 3（斷圖不一定報錯） | 已改。說明本例三種寫法都報錯，是因為只有 w 一個參數；有好幾段參數的模型，用 `.detach()` 剪斷中間一段不一定報錯，只是前段參數的 .grad 停在 None，並指向第 2 章〈訓練診斷〉的失敗一。已重跑第 2 章的寫法確認。 |
| 19 | 審查意見 | 技術 4（SGD 預設值） | 已改。補一段說明 momentum、weight_decay 等選項預設都不啟用，本例因此每步正好是 w−ηg；設了這些選項，更新式會多出別的項。已實測：momentum=0.9 第二步起不同，weight_decay=0.1 第一步就不同。另查過全課程程式都沒有用到這兩個選項。 |
| 20 | 審查意見 | 技術 5 | 和讀者 3 是同一件事，一起處理。 |
| 21 | 審查意見 | 先前檢查意見 1（〈核對輸出〉說明段的句序） | 已改，只調整句序，改成依第 1、2、3、4 行的順序說明，數字與斷言的描述不變。 |
| 22 | 審查意見 | 需要動其他檔案（不在這次查核範圍） | 維護者需要執行 `python3 scripts/review_coverage.py --write docs/lessons/00-warmup.md`，把這份審查登錄進 reviews/coverage.json。暫存副本實測：登錄後本頁不再列為過期，covered 只有頁面本身與 lesson_cases/00-warmup.py。 |
| 23 | 審查意見 | 觀察，沒有更動 | 其他節頁面仍寫 `PYTHONPATH=. python …`。這和 README 的說明一致，本頁現在也交代了怎麼對應；只是直接跳讀後面章節的讀者只能靠 README 得知這一步，要改需動其他頁。 |
| 24 | 先前查核 | 〈核對輸出〉說明段句序跳來跳去 | 未改：頁面已經是照第 1、2、3、4 行的順序講（第 1 行的印法放在最前面），這個問題已經修好，不必再改。 |

上表「審查意見」各列的修正，由另一位 AI 逐項檢查（通過）。

23 項處理我都對照了目前的頁面，全部屬實，沒有 must；只有一項 should：Colab 第 3 步沒提醒讀者，pip 可能印出 ERROR 字樣但環境其實已經裝好。

核對方法：
- 在暫存副本執行本節程式：exit 0，4 行輸出和頁面〈核對輸出〉逐字相同（torch 2.9.1，Python 3.12.15）。
- 照頁面新加的步驟實際操作：本機指令的各種寫法；alias／snapshot 插入程式；練習 1 的三種改法；練習 2 的刪除與 `#` 兩種寫法；改 seed 與執行緒數；在 notebook 的命名空間把最後一格執行兩次。
- 用 probe 程式逐條實測新寫的 PyTorch 說法：shape、矩陣乘法、非 scalar 的 backward、float32、印出格式、BatchNorm、SGD 的 momentum 與 weight_decay、zero_grad、多段模型的 detach。
- 函式庫說法另外對照 PyTorch 2.9 官方文件（SGD、zero_grad、BatchNorm1d、set_num_threads、set_printoptions）。
- Colab 步驟對照 notebook 的環境格程式、tests/test_notebook_bootstrap.py（6 項通過）、build_lesson_notebooks.py 與 verify_curriculum.py 的填入輸出機制，以及網路上描述 Colab 警告的說明。

結果：
- `zensical build --clean --strict` exit 0，No issues found；`validate_site.py` exit 0。新增的摺疊區渲染成有序清單，矩陣公式正常，沒有殘留的 \(。
- validate_lessons 的摘錄檢查通過；它整體失敗，只是因為審查涵蓋還沒登錄。我在暫存副本執行 `--write` 後，本頁不再被列出。validate_curriculum_evidence 只因 00-warmup 紀錄正在重產而失敗，未列為發現。
- 頁面正文沒有敘述教材修改經過的字眼，CJK 與英數字之間都有空格，用的是臺灣用語。

需要維護者決定的事：另外 41 個課程頁都寫 `PYTHONPATH=. python …`，沒有指回 README，跳著讀的讀者會遇到本頁原本那個問題。

全程沒有寫入 repo：repo 的 git status 前後雜湊相同，find 也找不到被改動的檔案。

| 列 | 結果 | 檢查內容 |
|---|---|---|
| 1 | 成立 | 本機指令已改成 `PYTHONPATH=. .venv-model/bin/python …`，與 README 第 17、24 行一致。我在暫存副本實測：直接打 `python` 得到 `pyenv: python: command not found`（exit 127），`python3` 得到 `No module named 'torch'`，`source .venv-model/bin/activate` 後 `python` 指向 venv 並印出 4 行；Windows 句也和 README 的 `$env:PYTHONPATH='.'`、`.venv-model\Scripts\python.exe` 一致。 |
| 2 | 成立 | 四步都在，建站後是 4 項的有序清單。對照 notebook：第一個程式格就是環境格；版本行由 `print("固定教材版本：", REF)` 印出，排在所有 check=True／raise 之後；重新啟動的提示與 RuntimeError 訊息一致；tests/test_notebook_bootstrap.py 6 項通過；Colab 警告的內容和網路上的說明一致，按鈕名稱也有加保留語。最後一格存著紀錄輸出屬於發布時的狀態，由 verify_curriculum 填入、validate_curriculum_evidence 檢查。另有一項建議列在問題清單（pip 的 ERROR 字樣）。 |
| 3 | 成立 | 三項說明都已補上：摘錄裡的註解、多出的 4 格縮排、兩行設定與 `model.train()`。核對結果：四段摘錄的註解都不在完整程式裡；除了 import 以外，每行都在 main() 底下；seed 改成 12345 或拿掉、執行緒改成 1 或 8，4 行輸出都不變；新建 Linear 的 training 是 True；validate_lessons 的摘錄檢查通過（它只在後面的審查涵蓋步驟失敗）。 |
| 4 | 成立 | 數值改寫成「1、2、3 三個數」「4 和 2」，shape 都加上 shape 字樣並用程式字型。W、Wᵀ 與 [1 2 3]Wᵀ=[4 2] 我用 torch 算過，結果相同；建站後，摺疊區裡的公式也正確渲染成 arithmatex。 |
| 5 | 成立 | 「以兩層括號為例」的數法已補上；例子 `[[1.0,2.0,3.0],[4.0,5.0,6.0]]` 用 torch 實測，shape 是 torch.Size([2, 3])，和頁面相同。 |
| 6 | 成立 | 兩個原因與報錯訊息都已補上。實測：shape [2,1] 不先平均就 backward，會報 `RuntimeError: grad can be implicitly created only for scalar outputs`；[1,1] 不會報錯，頁面也沒有說它會。 |
| 7 | 成立 | 這段已拆開。我照頁面把 5 行（含原有的 step）對齊縮排插進完整程式實跑：4 行前面多出的 3 行和頁面逐字相同；`1.`、`1.8000` 的印法符合 PyTorch 2.9 set_printoptions 的預設 precision=4。把副本取名 before 時，第 3 行的 print 會出 TypeError，所以頁面「不取名 before」的理由成立。 |
| 8 | 成立 | 刪掉 `optimizer.step()` 和在行首加 `#` 兩種做法我都實跑過：第 3 行都印出 `weight: 1.00 -> 1.00; new_prediction=2.00; new_loss=4.00`，接著第二行斷言出現 AssertionError，第一行照樣通過，和參考答案一致。 |
| 9 | 成立 | 情境已改成「建模型與建 optimizer 寫在不同格」。notebook 最後一格確實包含整個 main()，逐字等於 lesson_cases/00-warmup.py；用同一個 namespace 執行兩次，輸出相同。 |
| 10 | 成立 | float32 的說明正確：np.float32(1.8)=1.79999995231628417969，np.float32(3.6)=3.59999990463256835938，`after == 1.8` 是 False。「約 7 位有效數字」和術語表 FP32 那一列一致。 |
| 11 | 成立 | 推導已寫出 ΔL、ΔL/Δe，以及 ΔL=2eΔŷ+(Δŷ)²，並說明略去的項除以 Δw 後是 x²Δw，推導正確。我重算 (3.9204−4)/0.01=−7.96，它和 −8 差 0.04，正是 4×0.01。 |
| 12 | 成立 | 已改成「本課程用的 PyTorch 2.9.1」，也說明程式為何仍寫出來，並補上沒算到梯度的例子。PyTorch 2.9 文件與本機安裝的版本都寫 zero_grad 預設 set_to_none=True，並寫明 .grad 是 None 時 step 會跳過該參數。 |
| 13 | 成立 | 權重＝w（weight）、層的定義、不再自我引用的線性層說明、backward（反向傳播，也簡稱反傳）、y hat／eta 都已補上；「權重」第一次出現的那句就有說明。用詞和術語表 gradient 一列的「反向傳播」、第 2 章的「反傳」一致。 |
| 14 | 成立 | 開頭先用白話問「算出往哪邊調之後就算學到了嗎」，再對應到 backward 與 step，並說明這些名詞後面會解釋；沒寫過 PyTorch 的讀者也抓得到本節要回答的問題。 |
| 15 | 成立 | 兩處措辭都已改：「這一步 loss 從 4 降到 0.16，每個數都能手算核對」，以及「好處是…；限制是…也測不出泛化…」，主詞清楚、語意通順。 |
| 16 | 成立 | 和第 1 項是同一件事，處理方式與實測結果見第 1 項。 |
| 17 | 成立 | BN 句和 PyTorch 2.9 的 BatchNorm1d 文件一致：train 模式用這一批的統計值並更新 running estimates，eval 模式改用 running estimates。BatchNorm1d 實測兩種模式輸出不同；grep 確認課程程式沒有用到 BatchNorm 或 Dropout。 |
| 18 | 成立 | 實測：單一參數時，detach、numpy、item 三種寫法都會報錯。body→detach→head 的兩段模型，backward 不報錯，body.weight.grad 是 None，step 也不改 body；第 2 章的〈失敗一〉正是這種寫法。 |
| 19 | 成立 | PyTorch 2.9 的 SGD 文件寫明 momentum=0、weight_decay=0。實測兩步：預設 [1.8, 1.96]；momentum=0.9 是 [1.8, 2.68]，第二步起才不同；weight_decay=0.1 是 [1.79, 1.9401]。課程程式都沒用到這兩個選項。 |
| 20 | 成立 | 和第 3 項是同一件事，處理方式與實測結果見第 3 項。 |
| 21 | 成立 | 〈核對輸出〉的說明段改成依第 1 行、第 2–3 行、斷言、第 4 行的順序，數字與斷言的描述沒變；`print(x.shape)` 實測印出 `torch.Size([1, 1])`，和頁面相同。 |
| 22 | 成立 | 不改的理由成立：寫入 reviews/coverage.json 是維護者的步驟。我在暫存副本執行 `review_coverage.py --write docs/lessons/00-warmup.md` 後，本頁不再被列出，covered 只有頁面本身與 lesson_cases/00-warmup.py。 |
| 23 | 成立 | 就本頁而言，不改的理由成立：README 第 24 行已寫明 `python …` 要先啟用環境或改用完整路徑，本頁也說明了。不過另外 41 個課程頁都寫 `PYTHONPATH=. python …`，也沒有指回 README 或本頁，跳著讀的讀者會遇到同一個問題；要不要改那些頁，需要維護者決定，不屬於本頁。 |

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | docs/lessons/00-warmup.md 第 17 行，〈第一次用 Colab〉第 3 步：「過程中可能印出一些安裝訊息；只要之後有印出這一行，就是成功了。若它顯示錯誤、要你重新啟動工作階段…」 | 環境格用 `pip install -q` 改裝 torch 2.9.1 時，runtime 裡若有預裝套件指定另一個 PyTorch 版本（Colab 通常預裝 torchvision、torchaudio，它們都指定同版 torch），pip 會印出 `ERROR: pip's dependency resolver does not currently take into account all the packages that are installed…`，並列出「… requires torch==…, but you have torch 2.9.1+cpu which is incompatible」，但結束代碼仍是 0，環境格會照常印出版本行。我在暫存副本用自製的 wheel 重現：加了 -q 仍會印出這段 ERROR，exit 0。頁面只稱它為「安裝訊息」，下一句又寫「若它顯示錯誤…」。初學者看到 ERROR 字樣，可能以為失敗而停下，或動手去修 torchvision。成功判準本身是對的，所以列為建議。 | 已修正：第 3 步補上可能出現 `ERROR:` 開頭、說其他預裝套件需要別版 PyTorch 的訊息，並說明只要印出版本那一行就是成功；環境格停在錯誤、沒有印出這一行時，才照訊息處理。 |

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 2 輪：上一輪的處理與審查紀錄：有必要問題

全文讀了 reviews/00-warmup.md：有獨立查核與方法、讀者審查、技術查核（方法列出 PyTorch v2.9.1 commit d38164a 的檔案），〈定稿修正〉24 列。問題在修正後的檢查與表格格式。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | 〈定稿修正〉之後的「修正後由另一位 AI 檢查改動（通過）」 | 紀錄沒有顯示 23 項「審查意見」修正經另一位 AI 檢查：掛上的文字是同批第 2 輪複查，原文「這一輪只改了 docs/lessons/01-small-cnn.md 與 docs/lessons/02-diagnostics.md。 - 00-warmup.md 和兩張 SVG 都與第 1 輪相同」（本頁沒有 SVG）與「並抽查了先前輪次的修改」；第 1 輪對 00 頁只核一條先前意見；〈後續編輯的檢查〉也沒有本頁。和〈審查〉「查到的問題修正後，都由另一位 AI 檢查修正」不符。 | 已處理：另請一位 AI 逐項檢查這些修正，結果列在〈定稿修正〉表格之後。 |
| 2 | 建議 | 〈定稿修正〉表格 | 前 23 列都是「｜ 1 ｜ 審查意見 ｜ 讀者 1（必要，本機指令）：已改。… ｜ ｜」：編號全是 1、處理寫在「意見」欄、「處理」欄空白；同一項先前意見出現兩次且結論相反：「先前檢查意見 1（〈核對輸出〉說明段的句序）：已改，只調整句序」與「｜ 24 ｜ 先前查核 ｜ 〈核對輸出〉說明段句序跳來跳去 ｜ 未改：頁面已經是照第 1、2、3、4 行的順序講…」。 | 已修正：編號依序遞增，處理放在「處理」欄。未改：兩列「〈核對輸出〉句序」是不同階段的紀錄：讀者審查後先調整了句序，後來的查核確認句序已經正確，所以寫「未改」。 |
| 3 | 建議 | 〈獨立查核〉摘要與〈定稿修正〉 | 內部用語：「2. inventory 與 page-impact 都處理了 - impact 的 10 條全部處理：131、146、150…」「編輯已列入 concerns」「需要動其他檔案（不在本任務範圍）：協調者需要執行 `python3 scripts/review_coverage.py --write docs/lessons/00-warmup.md`」；條列被壓成一行，摘要在句中以「…」截斷。 | 未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 3 輪：上一輪的處理與審查紀錄：通過

〈第一次用 Colab〉第 3 步對照 notebook 環境格：所有 check=True 的步驟成功之後，才會 print("固定教材版本：", REF)；已經 import 過 torch 時會丟出「PyTorch 已更新：請從選單重新啟動工作階段，再由第一格執行。」；pip 的相依衝突訊息走 logger.critical，用 -q 仍會印出「ERROR:」，但結束碼是 0。所以「印出版本行就是成功」「停在錯誤時照訊息做完，再從第一格執行」屬實，解決了逐項檢查留下的建議，讀來通順。紀錄：〈定稿修正〉24 列，表後附逐項檢查（23 列，對應審查意見各列）與它的建議及處理，滿足上一輪要求；其餘結構檢查都通過。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | docs/lessons/00-warmup.md 第 17 行（〈第一次用 Colab〉第 3 步） | ERROR 訊息被限定在「PyTorch 不是 2.9.1 時…需要別版 PyTorch」。但環境格之後一定會執行第二個 pip install（numpy==2.3.5、pillow==12.0.0、matplotlib==3.10.7），預裝套件若要求別版 numpy 或 Pillow（Colab 常見的 tensorflow、numba 就有這類上限），就算 PyTorch 已經是 2.9.1，也可能印出同樣以 ERROR: 開頭的段落。讀者會以為自己不在頁面說的情況裡。成功判準本身寫得對。 | 已修正：改成安裝套件時都可能印出這類訊息，不再只綁 PyTorch 改裝。 |
| 2 | 建議 | reviews/00-warmup.md 第 32、65–67、168、273 行 | 逐項檢查的發現表引用 pip 訊息時，被產生器改了字：「ERROR: pip's dependency 處理者 does not currently take into account…」（原文是 resolver），引文已經不是原句。〈上述處理〉第 3 項寫「摘要與清單不再截斷」「內部用語換成白話」，但仍有殘句：「證據檔：- 暫存副本- 練習變體 stepA／B／C.py、alias.py 與各自的輸出：…/暫存副本」「- 紀錄檔：…/暫存副本、暫存副本、暫存副本」「python3 摘錄比對工具 <暫存副本> docs/lessons/00-warmup.md」「165（依 check 的更正改寫）」。 | 引文已恢復成 pip 的原文（resolver）。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 4 輪：上一輪的處理：有必要問題

〈第一次用 Colab〉第 3 步對照 notebooks/00-warmup.ipynb 環境格。print("固定教材版本：", REF) 排在所有 check=True 與 raise 之後，第二個 pip install（numpy／pillow／matplotlib）一定會執行。pip 的相依衝突訊息走 logger.critical，以 ERROR: 開頭，用 -q 仍會顯示，結束碼是 0。grep lesson_cases／miniyolo：沒有 import torchvision 等預裝套件，第 18 章的 cv2 只在 main() 不呼叫的 adapter 裡。所以「安裝套件時可能印出 ERROR: … 只要之後有印出這一行，就是成功了」屬實，也不再只綁 PyTorch 改裝，回應第 3 輪第 1 項；讀來通順、合規。B：第 3 輪第 2 項處理不實。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | reviews/00-warmup.md 第 64–67 行〈證據檔〉清單，對應第 3 輪第 2 項處理 | 處理寫「殘句已清理」，但點名的殘句仍在。第 65 行「- 暫存副本練習變體 stepA／B／C.py、alias.py 與各自的輸出：…/暫存副本」中，「練習變體…與各自的輸出：…/暫存副本」逐字保留；第 67 行「- 紀錄檔：…/暫存副本」也是點名的殘句；第 66 行「- 摘錄突變用的副本：…/暫存副本」同類。引文恢復 resolver、第 32 行、第 168 行這幾項確實改好了。 | 已處理：第 3 輪的處理說明改成統一的說明。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 5 輪：上一輪的處理：通過

第 3 輪第 2 項屬實：pip 引文已恢復成 resolver（第 273 行），點名的〈證據檔〉殘句、指令殘句與「165（依查核的更正改寫）」都已經不在原處。第 4 輪第 1 項也屬實：第 64–67 行現在是「- 暫存副本練習變體 stepA／B／C.py、alias.py 與各自的輸出」「- 摘錄突變用的副本」「- 紀錄檔」，點名的四段（含「：暫存副本」）確實都不在了。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | 第 3 輪第 2 項處理欄（第 296 行） | 「點名的文字已不在紀錄裡」也涵蓋了發現引用的第 2 輪處理說法「摘要與清單不再截斷」，但它仍在第 2 輪第 3 項處理欄（第 287 行）。 | 已處理：用詞類的處理說明改成統一的說明（紀錄保留查核者的原文，只統一替換路徑與內部名稱），不再逐句計數。 |

### 第 6 輪：上一輪的處理：通過

拿 journal 原文（53 段、189 行有內容）逐行對照紀錄，沒有任何有內容的行被刪。唯一的差異在第 1 次查核〈證據檔〉：「- 暫存副本：暫存副本」這一行只寫了暫存路徑，它沒有被整行刪掉，而是黏到下一行開頭，變成第 65 行「- 暫存副本練習變體 stepA／B／C.py、alias.py 與各自的輸出」；下一行的內容完整保留。另外核對第 3–5 輪處理欄：第 273 行的 pip 引文已經恢復成 resolver；頁面第 17 行已改成「安裝套件時可能印出一些訊息…」；第 4 輪 #1 說的第 3 輪 #2 確實含統一說明；第 5 輪 #1 也屬實。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | 第 65 行（第 1 次查核〈證據檔〉第一項） | 產生器 CLEAN 裡 `(?:scratch｜暫存)…暫存副本\s*[）)]?` 這條規則，結尾的 \s* 會吃掉換行。結果只寫路徑的那一行沒有被刪，反而把「暫存副本」黏到下一個清單項目前面。同一個原因也造成 18-video 第 52 行、16-training 第 30 行、glossary、planning-course-research 與 4 份 research 紀錄的黏字，其中幾處在先前幾輪已被點名過。內容沒有遺失，所以列為建議。 | 已修正：產生器的路徑規則不再吃掉換行，只寫路徑的那一行直接刪除，不再黏到下一項。 |

## lessons-v0.4.1 版本引用檢查

2026-10-05，另一位獨立 AI 依 `AGENTS.md` 與發布流程第 7 步，從 root 用指定 rsync 建立自己的 `/tmp/lessons-v0.4.1-review-00/` 副本，排除 `.git`、`site`、`.venv*`。完整閱讀 `docs/lessons/00-warmup.md`、README、`notebooks/00-warmup.ipynb` 四格與其保存輸出，對照主例、notebook 產生器、section-map、新舊執行紀錄及 v0.4.0 的 `base/` 快照。root、coverage、Git、remote、GPU 與既有模型環境均未修改，沒有下載資料集。

**版本文字與 notebook。** 相對 v0.4.0 快照，第 00 課只有頁首 Colab URL 與〈第一次用 Colab〉第 3 步的 `固定教材版本： lessons-v0.4.1` 改動；README 只有開頭明示的固定 tag 改動。將這三處 v0.4.1 換回 v0.4.0，兩份全文分別與舊快照逐字相同。`00-warmup.py` 未變，notebook 最後格逐字等於它；環境格的 `REF`、clone 的 `--branch`、精確 tag 核對與最後印出的版本都使用同一個 REF，與正文新標記相符。

另外只讀掃描 42 節：manifest `source_ref`、notebook metadata、環境格 REF、頁首 Colab URL 全部是 `lessons-v0.4.1`，42 本 notebook 最後格分別與同名 case 逐字相同。這是靜態配對檢查，沒有宣稱本次執行全部 42 節。第 00 課的標記摘錄以 validator 原有 `excerpt_problems()` 查得 `[]`。初次匯入整支 validator 時，因尚待登記的第 00 課等頁面 coverage 過期而停下；後續只讀使用它的摘錄檢查函式，沒有改 coverage，也沒有把完整 `validate_lessons.py` 說成通過。

**主例、手算與練習。** cwd 與 `PYTHONPATH=.` 指向自己的副本，使用既有 Python 3.12.14、PyTorch 2.9.1+cpu 執行主例及 notebook 最後格。兩次均 exit 0、stderr 為空，四行輸出逐字符合當前保存紀錄和 notebook 的 stdout stream。notebook 的 `output.text` 是標準的 list of strings，串接後正好是紀錄的 stdout，沒有把 list 當成一個字串。當前 `00-warmup.json` 與 v0.4.0 base 的紀錄完全相同；另核對 2026-10-02 的較早紀錄，它的 case hash 不同、第 3 行尚未印 `new_prediction=3.60`，不拿它當作現行程式的完整輸出。現行紀錄、正文四行與 notebook 都使用 2026-10-05 那份輸出。

獨立用精確分數手算：更新前預測 2、loss 4、梯度 −8；學習率 0.1 得 w=1.8、預測 3.6、loss 0.16。有限差分 w=1.01 得 loss 3.9204，差商 −7.96，與梯度 −8 差 0.04。頁面多輸入線性層示例另用 CPU 計算，輸出確為 `[4,2]`；重複 backward 未清梯度得 −16，無 no_grad 的參數原地改寫、以及以 `.item()`／`.detach()`／NumPy 斷開本例 loss 都得到正文所說的錯誤。

所有練習變更都只放在自己副本的 `artifacts/runs/v041-review00/`，沒有刪除核對斷言來讓結果通過：

| 實跑項目 | 結果 |
| --- | --- |
| 練習 1：只改主 optimizer 的 lr=0.25 | 印 w=3、預測 6、loss 4，第二行舊答案斷言報 AssertionError，符合正文。 |
| 練習 1：只改第二行斷言、第三行仍保留 3.6 | 第三行斷言報 AssertionError，符合提醒。 |
| 練習 1：依答案更新兩行斷言 | exit 0，四行全部印出；替換用 new_optimizer 的 lr 維持 0.1。 |
| 延伸：lr=0.125，依手算更新答案斷言 | exit 0，w=2、預測 4、loss 0。 |
| 練習 2：註解 optimizer.step() | 印 w=1、預測 2、loss 4，梯度仍 −8，第二行斷言報 AssertionError。 |
| 正文 alias／snapshot 插入示例 | exit 0；前三行是 Parameter containing、1.8000、副本 1，後接原本四行。 |

主例、notebook 最後格與上述變體共 5 個成功執行、3 個預期的斷言失敗，分開記錄。查核輔助程式最後印 `V041_REVIEW00_AUDIT_PASS`，補充探針印 `V041_REVIEW00_PROBES_PASS`。

**來源、README 與頁面。** 重新開啟 PyTorch v2.9.1 官方 commit `d38164a545b4a4e4e0cf73ce67173f70574890b6` 的 [Linear](https://github.com/pytorch/pytorch/blob/d38164a545b4a4e4e0cf73ce67173f70574890b6/torch/nn/modules/linear.py)、[SGD](https://github.com/pytorch/pytorch/blob/d38164a545b4a4e4e0cf73ce67173f70574890b6/torch/optim/sgd.py) 與 [Optimizer](https://github.com/pytorch/pytorch/blob/d38164a545b4a4e4e0cf73ce67173f70574890b6/torch/optim/optimizer.py) 原始碼，核對 `xA^T+b`、權重 shape、bias=False、momentum／weight_decay 預設 0，以及 zero_grad 的 set_to_none=True 和 None 梯度跳過更新。這些相鄰說明沒有因升版而改變；本次沒有重做其他未受影響的函式庫主張之全面查核。

README 的 CPU 主例指令已以既有指定環境實跑；notebook 最後格與 case 一致的說明也直接核對。README 的「新 tag、不覆寫舊 tag」及發布後跑 Verify published lessons 的步驟保留，與此次新建版本的流程一致。本次沒有重新安裝依賴、執行 README 的完整 42 節 runner、完整 pytest、資料下載、其他訓練或 GPU 指令，不把這些列為本次驗證。

自己的 `zensical build --clean --strict` 與 `validate_site.py` 都 exit 0。以 Playwright／`/usr/bin/chromium` 對自有 HTTP server 做 390×844 瀏覽檢查，實際展開〈第一次用 Colab〉並查看截圖，確認 v0.4.1 完成標記已顯示、按鈕 href 指向 v0.4.1 notebook，article 沒有殘留 v0.4.0。瀏覽器結果印 `V041_REVIEW00_BROWSER_PASS`。自有 port 8818 server 已停止，root 的 8794 沒有操作。

**發布前限制。** 查核時 `lessons-v0.4.1` 尚未推送公開，因此未做公開 tag clone、託管 Colab、公開網站比對或全新的固定版 bootstrap／README 驗證，也不宣稱它們已成功。環境格另在 mock 探針中執行：只供給正確 tag 及既有相同 PyTorch 時，印出新完成標記；供給舊 tag 時，先報版本不符且不印完成標記。探針沒有真正執行 Git、pip 或 chdir，僅驗證本地控制流程，不能取代發布後 root 執行的 Actions。本報告是發布候選內容的獨立檢查，公開固定版實跑仍待新 tag 發布後完成。

必要問題：**0**；新增可選問題：**0**。上述未執行的公開驗證屬明示的後續發布步驟，不把尚未公開 tag 誤列為教材程式缺陷。

**本次快照 SHA-256。** 正文 coverage hash 用現行 `digest()` 只讀計算，排除頁尾紀錄並正規化 Colab tag；新正文期待輸出的 tag 仍有涵蓋。完整數值、42 節靜態清單、每個子程序輸出、mock 探針、原始來源與手機截圖保存在自己副本的 `artifacts/runs/v041-review00/`，主要結果是 `audit.json`、`probes.json`、`browser.json` 與 `colab-v041-mobile.png`。

```text
docs/lessons/00-warmup.md（新正文 coverage hash）
169c2dfb17ad031e7190999a8d5c7e65ee6aff0b63f3648335aa7bc04a8ce5e2
README.md（原檔 hash）
6732d75dab1314105722092ef0d02e7202c19a61087d7692bc8311f7d3442df3
notebooks/00-warmup.ipynb
c0cc2f7218339cf8a4bcf55ca2b92644ae326a98c98640f9468f7d9e9c701787
lesson_cases/00-warmup.py
7929dea8fc669b1cf5eb3fa252f479b54857e824508d52b64f0a12fb5c1ed93b
artifacts/checks/curriculum/00-warmup.json
853aa4af361f96f475de9e2fd639270640a0d1ff02be2584d5fd1e14a5aff0ef
```


## 2026-10-05 clear-tutorial 三輪重審

以上是原審查歷史；不追溯改成首次盲讀。這次由固定基線 `16f6910` 分段開放並保存當時理解，再修改、核技術及檢查銜接，詳見 [本輪方法與限制](clear-tutorial/16f6910/README.md)。

- 第一輪：[foundations當場閱讀原始紀錄](clear-tutorial/16f6910/first-read/foundations.jsonl)，基線來源與圖指紋保留；共享檔案系統不是技術隔離。
- 第二輪：[非作者技術／證據核對](clear-tutorial/16f6910/technical/foundations.md)，實際來源、數字及必要執行範圍見該報告。
- 第三輪：[另一位讀者前文→本節→後文複查](clear-tutorial/16f6910/transitions/foundations.md)，此輪完整頁閱讀非盲讀；受影響段落及圖另有delta核回。
- [原始卡點與具體處理](clear-tutorial/16f6910/decisions.json)保留未新增的選讀建議。原先前提包漏發及08提前brief的限制另列，沒有算成教材錯或冒稱08全程盲讀。

本輪修正後沒有未解的必要問題；這是AI閱讀／技術查核的實際範圍，不是學生學習成效驗收。全站實際Zensical桌面／手機、公式換頁與執行檢查見 [verification.json](clear-tutorial/16f6910/verification.json)，不以SVG檔存在或strict build取代視覺查核。

## 2026-10-05：v0.6.0 有界更新審閱

只核環境成功輸出的固定教材版本v0.6與notebook相同。

本次僅重查以上變更，既有正文的歷史審閱保留；沒有把全頁或全書重新標成首次盲讀。[導讀／實際網站審閱](clear-tutorial/vision-v0.6.0/guide-visual-review.md)、[新增支線第三輪及實際前置](clear-tutorial/vision-v0.6.0/transitions-review.md)、[首讀修後複查](clear-tutorial/vision-v0.6.0/vit-recheck.md)記錄各自範圍。全版52本notebook的本機cell執行另見[實跑](clear-tutorial/vision-v0.6.0/local-notebook-runtime.json)；不是Google Colab登入執行。來源與圖／程式指紋更新於[coverage.json](coverage.json)。

## 2026-10-06：最新版 clear-tutorial 全套重審

本次以 `64a25d4fbcff5577965c29efbbcb5d9898ba95d9` 凍結來源從頭閱讀，不把以前的審閱當作此次首次閱讀。方法、完整範圍與限制見[本輪報告](clear-tutorial/full-review-2026-10-06/README.md)。

- 首次閱讀：主要讀者 `foundations` 實讀本頁 10 個凍結單元；首次使用／前文方法範圍四題位置為 00-warmup/00:first_use, 00-warmup/01:first_use, 00-warmup/02:first_use, 00-warmup/06:first_use, 00-warmup/07:first_use，頁末為 00-warmup/09。[當時理解與問題](clear-tutorial/full-review-2026-10-06/first-read/foundations.jsonl)與[分段披露](clear-tutorial/full-review-2026-10-06/first-read/foundations-disclosures.jsonl)按原樣保留；實際前置閱讀見[該組報告](clear-tutorial/full-review-2026-10-06/reports/foundations.json)。
- 處置：[決策表](clear-tutorial/full-review-2026-10-06/decisions.json)。本頁處置：R001、R009；各項原位置、分級、實際改寫／保留理由見決策表。
- 非作者技術／證據：[本頁所屬報告](clear-tutorial/full-review-2026-10-06/rechecks/technical-foundations.json)，只以報告列出的正文、實作、數值、圖與實際執行範圍作結論。00 的 BN 首次命名及 04 的最後公式／紀錄變更另由偵測與演進技術報告補核。
- 另一位讀者的前文→本節→後文與網站：[第三輪紀錄](clear-tutorial/full-review-2026-10-06/rechecks/transitions-visual.json)。52節正文有閱讀紀錄；實看圖／公式的頁面與截圖另列，不將捕捉或DOM載入當成每張圖可讀。

本輪未留下已裁定的必要問題。所有讀者均為 AI，沒有真人學生學習效果驗收。原首讀中仍有漏報、引用未支持全部主張及明說／推論混分，見[獨立裁定](clear-tutorial/full-review-2026-10-06/rechecks/record-adjudication.md)；不能宣稱四題保證抓到所有缺漏或原始紀錄嚴格規則全合格。程式與依賴、正式CPU紀錄、Notebook、建置和全站掃描的實際檢查見[驗證結果](clear-tutorial/full-review-2026-10-06/verification.json)。本頁最新文字、所用SVG／raster圖片與實驗依賴綁定在[coverage.json](coverage.json)。

## 2026-10-06：v0.6.1 有界修正複查

僅將當前教材版本與環境提示從 lessons-v0.6.0 同步到 lessons-v0.6.1；舊驗證與歷史版本的範圍保留。

本輪方法、逐批閱讀原始紀錄、非作者技術核對、另一位讀者銜接與實際 Zensical 修改段落截圖見 [v0.6.1 局部複查](clear-tutorial/release-v0.6.1-2026-10-06/README.md)。本次只重新檢查修改處、必要上下文與版本一致性；其餘正文、程式及圖的既有審閱保留原範圍，不改標為整頁或全書新的首次盲讀驗收。

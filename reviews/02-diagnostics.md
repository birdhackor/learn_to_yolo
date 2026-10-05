# 審查紀錄：訓練診斷

審查範圍：`docs/lessons/02-diagnostics.md`、頁面上的圖（`docs/assets/diagrams/02-diagnostics.svg`），以及 `lesson_cases/02-diagnostics.py` 與它 import 的 repo 模組；頁尾自動產生的執行紀錄區塊不在範圍內，由 `scripts/validate_curriculum_evidence.py` 對照紀錄檢查。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面依目前的程式改寫後，由另一位 AI 獨立查核：在獨立的副本執行該節程式、照頁面做練習，逐句對照程式、執行紀錄與手算，檢查程式摘錄與網頁轉換，並從初學讀者（高中程度、數學好、程式新手）的角度看用詞與說明順序。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：通過

通過，沒有必要問題，只有一個建議。

1. 程式敘述都符合目前的程式。lesson_cases/02-diagnostics.py 從 lessons-v0.3.0 起沒有改過，sha256 37675b43… 和紀錄的 case_sha256 相同。在暫存副本執行 exit 0，stdout 和紀錄逐行相同，只有 learned weights 末位差 1 ulp。
   - 照頁面指示做練習 1：三行貼在 `train_x, train_y = …` 之後，縮排對齊。
   - 保留原斷言：在 `assert train_acc == 1.0 and val_acc == 0.0` 報 AssertionError。
   - 改成新斷言：exit 0，印出 `train_accuracy=1.00; validation_accuracy=1.00`。權重 ±0.4708／±0.0134，step=19 loss 0.5824／0.6156，正確類別機率 0.540～0.564，都和參考答案的約略值一致。
   - 其他核對過的：CPU 錯誤訊息 `IndexError: Target 2 is out of bounds.`、第一步梯度 [0.1, 0.5]、e^-0.1236≈0.88、e^-2.0153≈0.13。
   - 斷開的版本另外訓練 10 步：loss 從 0.688 降到 0.526，body 權重不變。所以「loss 可能下降」「body 一直不會更新」都成立。
   - 第 1 章的 40 步 loss 確實取自更新前的 forward。其他頁引用本頁的內容（None 梯度、計算圖、argmax、held-out、before 副本）都還在頁面上。

2. 四條先前審查意見都處理了：
   - 第 3 行：HEAD 已經是 lessons-v0.4.0。
   - 第 33 行：整句刪除，改寫後的推論成立。
   - 第 164、166 行：改成讀者執行後會看到的輸出。
   受程式改動影響的段落沒有本頁的項目。全頁沒有製作或修訂經過的敘述：「中文註解是本頁加的」是說明註解的來源，頁尾的「本次實際輸出」是自動產生的紀錄區塊。

3. 數字：這次編輯沒有加入新數字。練習 1 的數字上一個 commit 就已經在頁面上，只取到小數兩位，而且是確定性的計算，不是從這台 Mac 取的數字。02-diagnostics.json 缺 dependencies_sha256，is_current() 會判它過期並重產；依賴紀錄的數值，修正者已經列出。

4. 摘錄：三段都有標記，摘錄比對工具對 repo 和暫存副本都印出 []。我另外確認三段都是程式裡連續、照原順序的原文（程式第 12–16、17–24、26–29 行），正文沒有程式行號。
   要提醒一點：excerpt check 只核對每一行都出現在檔案裡，不核對順序和連續性。突變測試中，把失敗一摘錄裡的 `.detach()` 拿掉並不會被抓到，因為修好後的那一行也在程式裡。這是檢查器本身的涵蓋範圍，不是本頁的問題。

5. 建置與 SVG：在暫存副本跑 zensical build --clean --strict（No issues found）和 validate_site.py，都 exit 0。三段摘錄在 HTML 裡都是帶 data-excerpt 的 Python 區塊。SVG 沒有改，有 viewBox／title／desc；qlmanage 算圖乾淨，點座標、分界線端點 (60,249)／(420,321)、判對／判錯標籤都和程式一致。這一頁相關的檔案中，只有 docs/lessons/02-diagnostics.md 有改動。

另外，review_coverage 顯示本頁 no review。依審查用的事實與寫作規範清單，這是發布時才會完成的事。

紀錄位置
- run.stdout：課程程式的輸出
- ex1_keep_assert.*、ex1_new_assert.*：練習 1 兩種版本的程式與輸出
- probe.out：額外核對的數字與錯誤訊息
- build.log、validate-site.log：建置與網站檢查
- render/02-diagnostics.svg.png：SVG 算圖

暫存副本在暫存副本。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/lessons/02-diagnostics.md 練習 1 參考答案（第 166–170 行） | 參考答案現在帶讀者一行行看輸出：accuracy 那行、`learned weights` 那行、step=19 那行。但練習版本最後還會印出原題寫死的英文結語 `Synthetic distribution shift demonstrates: tiny-set overfit is not generalization.`，意思和練習結果相反（validation 也全對）。程式新手可能以為程式判定練習失敗。頁面也沒說要怎麼看出「改過的最後一個斷言也會通過」：實際上，看得到這行就代表所有斷言都過了。 |

最後一次查核的建議事項，在下方〈定稿修正〉逐項處理。

## 讀者審查與技術查核

### 讀者審查（AI 以初學讀者身分閱讀、執行程式與練習）

方法：讀了哪些：
- 定稿事實清單審查用的事實與寫作規範清單。
- 由上往下逐段讀 docs/lessons/02-diagnostics.md、lesson_cases/02-diagnostics.py、notebooks/02-diagnostics.ipynb（四格的內容與最後一格存的輸出）。
- 依讀者順序讀了前置的 docs/lessons/00-warmup.md、01-small-cnn.md，核對頁面引用的「第 0 章 2(ŷ−y)×x」「第 1 章 40 步曲線在更新前量」「第 1 章提過 cross entropy 接 logits」，三處都對得上。
- 對照 docs/glossary.md 的術語，並 grep 其他課頁引用第 2 章的地方，以及全課程 NaN／nan 的用法。
- 讀了 scripts/validate_lessons.py 的摘錄比對邏輯，它會去掉註解，所以頁面加的中文註解不影響。

執行與建置（都在暫存副本，以 rsync 建立）：
- 用 `PYTHONPATH=. OMP_NUM_THREADS=2 MPLBACKEND=Agg .venv-model/bin/python lesson_cases/02-diagnostics.py` 執行完整程式，exit 0。印出各行和頁面一致，learned weights 只有浮點數最後幾位不同。
- 用 `.venv-docs/bin/zensical build --clean --strict` 建置，exit 0。讀了 site/lessons/02-diagnostics/index.html 的 article：表格、摺疊區、數學、圖都正確轉換。
- 本機 http.server 搭配 headless Chrome（1280 寬）與 chrome-headless-shell（390 寬）截圖看實際版面。新版 headless Chrome 的視窗有最小寬度，390 寬時會顯示假的右側截斷；改用 headless-shell 確認頁面在 390 寬沒有橫向溢出。截圖在暫存副本。
- 用 `qlmanage -t -s 1200` 把 docs/assets/diagrams/02-diagnostics.svg 轉成 PNG 看圖，並逐一核對 SVG 的座標：訓練點、validation 點、分界線端點 (60,249)–(420,321)、上下判斷區、箭頭方向都和程式與文字相符。

練習：
- **練習 1 照頁面指示**：在 train_x 那行後加三行，最後斷言改成 val_acc == 1.0。
  - 結果：step=19 的 loss 是 0.5824／0.6156，accuracy 是 1.00／1.00，weights 約 ±0.4708／±0.0134，和參考答案（±0.47、±0.01、0.58／0.62、約 0.55）相符。
  - 我另外逐筆算正確類別機率，在 0.54～0.56 之間。
  - 不改斷言時出現 AssertionError，和頁面說的一樣。
- **練習 2**（02-ex2.py）：實測以下四種情況，backward 後 body.weight.grad 都是 None：
  - 有 detach
  - body 的計算包在 no_grad 裡
  - body 沒參與 forward
  - body 參數設成 requires_grad=False（答案沒列這一項）

另外核對的頁面說法：
- 曾有梯度的參數在 set_to_none=False 後變成全 0；第一步就用 set_to_none=False 時仍是 None。
- 把 [0,2] 交給 cross_entropy，CPU 上報 `IndexError: Target 2 is out of bounds.`。
- SGD 第一步梯度是 [[0.1,0.5],[-0.1,-0.5]]。
- 改用 Adam 後，a、b 兩個權重的比例變成 1:1，確實不再是 5 倍。

其他：
- 檢查期間沒有寫入 repo 根目錄。
- 我 rsync 之後，正式 repo 的 docs/lessons/00-warmup.md 被別人改過。受審頁、程式、SVG、術語表和 01 章都和正式 repo 逐位元相同；00 章的新改動和本頁不衝突。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 必要 | 〈收益、代價，以及何時可以停〉第一段（第 138 行），對照檢查步驟 3（第 13–17 行）與失敗二（第 74 行） | 這段說「步驟 3 抓最佳化問題，也就是訓練過程沒辦法把訓練資料的 loss 降下來，例如學習率不當」。可是步驟 3 自己列的優先檢查順序，第一項是「監督（label／target 有沒有對上每筆輸入）」，最後一項是「訓練路徑（有沒有 detach、optimizer 拿到的是不是目前模型的參數）」，兩項都是資料或程式問題。失敗二也說，兩份資料的類別對照表不一致時，「程式照跑，loss 卻常停在偏高的地方」，這正是步驟 3 會看到的現象。讀者若照這段總結記住「少量資料背不下來＝最佳化問題，先調學習率」，下次背不下來就會先去調學習率。這正是失敗二標題「標籤格式錯了，卻去調學習率」警告的錯誤。 |
| 2 | 建議 | 摺疊區〈為什麼 b 的權重固定是 a 的 5 倍〉第一段（第 101 行） | 整個論證建立在一句話上：「線性層權重的梯度，和第 0 章暖身的 2(ŷ−y)×x 是同一種形式：某個係數 × 輸入」。頁面沒說為什麼。第 0 章是平方誤差，這裡是 cross entropy，數學好的讀者會卡在「loss 不同，為什麼形式一樣」。接著舉的例子「第一步的梯度，類別 0 那一列是 [0.1, 0.5]」，讀者也沒辦法手算核對，因為頁面沒給那個係數是多少。 |
| 3 | 建議 | 練習 2 的參考答案（第 178 行），對照第 59 行的「刻意凍結」 | 第 59 行剛說「grad 是 None 算不算錯，要看原本打不打算訓練那一段」，練習 2 的答案卻直接跳到查斷圖，沒有先確認 body 本來就該學。答案也沒列最常見的凍結寫法：把參數設成 `requires_grad=False`。我實測過，這時 backward 後 `body.weight.grad` 同樣是 None。讀者以後碰到被凍結的 backbone，照答案查 detach、no_grad、有沒有參與 forward，三項都正常，就找不到原因。 |
| 4 | 建議 | 失敗三模型說明（第 95 行）與練習 1 程式第三行的註解（第 161 行） | 頁面只說「每類同一個點重複四次，共 8 筆」。前三章和術語表都沒解釋 `x.repeat(4, 1)`：它把整個 x（2 筆）沿第 0 軸接成 4 份，排列順序是「類別 0、類別 1、類別 0、類別 1……」交錯。練習 1 的 `labels.repeat(4)` 能和 extra_x 一筆筆對上，全靠這個交錯順序。程式新手看到註解「標籤 0、1 交錯，對上 extra_x 的順序」，不知道它從哪裡來，也判斷不了換一種寫法會不會對錯標籤。 |
| 5 | 建議 | 練習 1 的參考答案（第 168 行） | 照練習改完執行後，最後一行仍印出寫死的字串「Synthetic distribution shift demonstrates: tiny-set overfit is not generalization.」。這時 validation_accuracy 是 1.00，和這句話的意思相反。頁面沒有提醒，新手會以為自己哪裡沒改對。 |
| 6 | 建議 | 開頭第二段（第 7 行） | 原文是「三個刻意製造的失敗：先是兩個程式錯誤（梯度被切斷、標籤越界），再用 8 筆人工資料……訓練 20 步」。第三個只寫了做什麼（訓練 20 步），沒寫失敗在哪裡。讀者在開頭不知道第三個失敗是什麼，要讀到失敗三的標題才知道。 |
| 7 | 建議 | 失敗一、二、三的標題（第 22、61、80 行），對照開頭「依序該查哪四件事」（第 5 行） | 本節強調檢查要依序做，但失敗一對應步驟 2，失敗二才對應步驟 1，展示順序和檢查順序相反，頁面沒說原因。讀者容易誤以為應該先查梯度、再查標籤。 |
| 8 | 建議 | 圖下說明（第 93 行）與「代入 validation」段（第 113 行） | 第 93 行說分界線 b=−0.2a「下面會說明怎麼算出來」，但第 113 行只寫到「0.2a+b>0 時判成類別 1」，沒有明說分界線就是兩類分數相等、0.2a+b＝0 的地方。這個連結要讀者自己補上，第 93 行的承諾也就沒有明確兌現。 |
| 9 | 建議 | 檢查步驟 3（第 13 行） | overfit 在這裡定義成「只記住訓練資料，換新資料就不準」，比術語表的「換新資料卻可能變差」更絕對。同一段又把刻意背下幾筆資料也叫 overfit；練習 1 裡模型背下 16 筆訓練資料，validation 卻是 1.00。照頁面的定義，讀者會困惑練習 1 算不算 overfit，也可能誤以為背下訓練資料就一定在新資料上不準。 |
| 10 | 建議 | 檢查步驟 1、2、3、4（第 11、12、16、18 行） | 幾個用詞和術語表、第 1 章不一致：(1) 正文寫 nan（第 12、16 行），術語表、第 1 章和大多數後面章節都寫 NaN；(2) held-out 只寫英文，術語表與第 1 章對應的中文是「獨立資料」「獨立驗證資料」，讀者用中文查不到同一個詞；(3) overfit 沒附術語表的中文「過擬合」；(4) BCE 的 label「要用 0.0／1.0 的小數」，數學上 1.0 是整數，這裡真正要講的是 dtype 必須是浮點數。第 0 章用的說法是「浮點數（float）」。 |
| 11 | 建議 | 練習 1 的三行程式（第 158–162 行） | 三行的說明都寫在行尾註解裡，第二行的註解又長。網頁寬 1280 時，第二行註解「沿第 0 軸（筆數那一軸）接在原資料後面」已被截斷；手機寬度下三行註解都看不到，要在程式框裡橫向捲動才找得到。torch.cat 是什麼、標籤怎麼對上，都只寫在這些註解裡。 |

### 技術查核（AI 對照原始論文、固定 commit 的官方程式、該節程式與手算）

方法：閱讀的檔案（暫存副本，由 rsync 自 repo 建立）：
- 規範與本頁：審查用的事實與寫作規範清單、docs/lessons/02-diagnostics.md、lesson_cases/02-diagnostics.py（只 import torch，沒有用到 miniyolo）、docs/assets/diagrams/02-diagnostics.svg、notebooks/02-diagnostics.ipynb（第三格標題是「本節可修改的完整實驗」，最後一格與 lesson 檔逐字相同）。
- 交叉引用：docs/lessons/00-warmup.md（2(ŷ−y)×x）、docs/lessons/01-small-cnn.md（backbone、CE 接收 logits）、scripts/run_learning_extensions.py 第 74–92 行（第 1 章 40 步的 loss 是在更新前記錄）、miniyolo/losses.py（grid_loss = 5*box + objectness + classification）、對 miniyolo／lesson_cases／scripts 用 grep 找 optimizer（miniyolo/train.py:97 等處用 Adam）、docs/glossary.md、scripts/validate_lessons.py（摘錄比對的邏輯）、artifacts/checks/curriculum/02-diagnostics.json（只看 metadata）。

一手來源：
- PyTorch v2.9.1，commit d38164a545b4a4e4e0cf73ce67173f70574890b6，用 git ls-remote 解析 tag 後，以 curl 從 raw.githubusercontent.com/pytorch/pytorch/<commit>/<path> 下載：
  - aten/src/ATen/native/LossNLL.cpp 第 196–200 行：`TORCH_CHECK_INDEX(cur_target >= 0 && cur_target < n_classes, "Target ", cur_target, " is out of bounds.");`
  - aten/src/ATen/native/cuda/Loss.cu 第 158–163 行：`CHECK_INDEX_IN_CLASS` → `CUDA_KERNEL_ASSERT(INDEX >= 0 && INDEX < N_CLASSES);`
  - torch/optim/optimizer.py 第 998–1035 行：zero_grad，set_to_none 預設 True；「.grad s are guaranteed to be None for params that did not receive a gradient」；「in the other it skips the step altogether」。
  - torch/optim/sgd.py 第 87–95 行（`if p.grad is not None:`）與第 373、375 行（`param.add_(grad, alpha=-lr)`）。
  - torch/nn/modules/loss.py 第 1194、1286–1289 行（CrossEntropyLoss：「The target data type is required to be long when using class indices」）與第 820–821 行（BCEWithLogitsLoss：「Target: (*), same shape as the input」）。
  - c10/cuda/CUDAMiscFunctions.cpp 第 37–39 行：「CUDA kernel errors might be asynchronously reported … consider passing CUDA_LAUNCH_BLOCKING=1」。
- PyTorch 官方文件（頁面引用的連結）：https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html 與 https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html，兩者都轉址到 docs.pytorch.org/docs/2.14/…；另外讀了 https://docs.pytorch.org/docs/2.9/generated/torch.optim.Optimizer.zero_grad.html 與 https://docs.pytorch.org/docs/2.9/generated/torch.nn.CrossEntropyLoss.html。2.9 與 2.14 版的 set_to_none 說明，以及 Shape/Target、ignore_index 段落，文字相同。
- NVIDIA CUDA Runtime API，Data types 頁 https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html 的 enumerator cudaErrorAssert：「The device cannot be used again. All existing allocations are invalid. To continue using CUDA, the process must be terminated and relaunched.」

執行的指令（全部在暫存副本目錄，使用 .venv-model/bin/python，PyTorch 2.9.1，macOS arm64）：
1. 建立副本：`rsync -a --delete --exclude .git --exclude site --exclude '.venv*' --exclude artifacts/runs --exclude data/curated --exclude data/downloads <repo>/ <scratch>/`
2. 跑本節程式：`PYTHONPATH=. OMP_NUM_THREADS=2 MPLBACKEND=Agg python lesson_cases/02-diagnostics.py`，exit 0。9 行輸出都與頁面相同，只有 learned weights 的 float32 末位不同（隨機器而異）。
3. 摘錄檢查：摘錄比對輸出 []；另外人工核對三段摘錄的順序與連續性。
4. checks_torch.py，實測以下行為：CPU 越界時的 IndexError 原文、int32／float label 的錯誤、BCEWithLogitsLoss 收到 long label 與 [B] 對 [B,1] 時的錯誤、set_to_none=False 的語意、SGD 跳過 None 梯度、argmax 的 dtype 與 requires_grad 及之後 backward 報的錯、nan 梯度寫進參數。
5. exercise1_mine.py：在原程式加入頁面上的三行，並把斷言改成 val_acc == 1.0，結果 1.00/1.00、權重 ±0.4708/±0.0134、step 19 的 loss 0.5824/0.6156、正確類別機率 0.554／0.564／0.540。
6. 用 python3 寫遞迴式 c←c+0.2(1−σ(2.08c)) 重算 step 0/9/19 的 loss 與權重，並計算 e^{-0.1236}、e^{-2.0153}、ln2。
7. adam_ratio.py：比較 SGD、SGD 加 momentum、Adam 三種 optimizer 下的權重比例，分別是 5.0、5.0、1.0。
8. 檢查第一步梯度與 validation 代入值：[0.1,0.5] 與 ±0.96。
9. 渲染圖：`qlmanage -t -s 1200 -o <dir> 02-diagnostics.svg` 後檢視 PNG，並逐一核對 SVG 座標換算 (240+180a, 285−180b)。
10. 建置網站：`.venv-docs/bin/zensical build --clean --strict` 得 exit 0、No issues；`python3 scripts/validate_site.py` 得 exit 0。
11. `python3 scripts/validate_lessons.py` 只在審查覆蓋那一步失敗；這是審查進行中的暫時狀態，不列為問題。
12. 用 grep 在頁面中找敘述修訂經過的字眼，沒有找到。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 必要 | 「四個檢查步驟，各自回答不同問題」節末段落（原始檔第 20 行）：「在 GPU 上出錯時，錯誤訊息常看不出原因；可以從那批資料取幾筆搬到 CPU，重跑 forward 和 loss，CPU 通常會直接指出是 label 越界還是 shape 不合。」 | 這個做法在它最主要的情境（label 越界）照做會失敗。label 越界在 GPU 上觸發的是 kernel 內的斷言：PyTorch v2.9.1（commit d38164a545b4a4e4e0cf73ce67173f70574890b6）的 aten/src/ATen/native/cuda/Loss.cu 中，`CHECK_INDEX_IN_CLASS` 展開成 `CUDA_KERNEL_ASSERT(INDEX >= 0 && INDEX < N_CLASSES);`，錯誤碼是 cudaErrorAssert。NVIDIA CUDA Runtime API 對 cudaErrorAssert 的說明是：「The device cannot be used again. All existing allocations are invalid. To continue using CUDA, the process must be terminated and relaunched.」也就是說，在同一個 Colab 工作階段裡，不論是對那批 GPU tensor 呼叫 `.cpu()`，還是呼叫 `model.cpu()`，都會再丟出同一個 `CUDA error: device-side assert triggered`。讀者照頁面操作會卡住，還可能以為「搬到 CPU」這個方法沒用。另外，shape 不合是在 host 端檢查的（例如 `mat1 and mat2 shapes cannot be multiplied`、`Expected input batch_size ... to match target batch_size`），在 GPU 上本來就會直接報出清楚的錯，不必靠 CPU。 |
| 2 | 建議 | 「梯度很小時，怎麼判斷是不是梯度消失」摺疊區塊第 1 點：「和參數本身的大小比：看『學習率 × 梯度』相對於參數有多大」 | 「學習率 × 梯度」只有在不帶 momentum 的 SGD 下，才等於每一步參數實際改變的量。課程後面主要的訓練用的是 Adam，例如 miniyolo/train.py 第 97 行、lesson_cases/07-training.py、07-heldout、08-own-data、08-own-images、10-multiscale、17-capstone、18-video。Adam 每步的改變量約為 lr × m̂/(√v̂+ε)，即使梯度很小，改變量仍接近 lr 的量級。實測把本頁失敗三改用 Adam 訓練，兩欄權重等速成長，b/a 比為 1.0，正是這個性質。讀者若把這條規則用在 Adam 訓練上，會把真實更新量低估好幾個數量級，誤判成梯度消失。 |
| 3 | 建議 | 「收益、代價，以及何時可以停」第一段：「檢查步驟 1、2 抓資料與程式問題；步驟 3 抓最佳化問題，也就是訓練過程沒辦法把訓練資料的 loss 降下來，例如學習率不當」 | 這句和本頁檢查步驟 3 自己的清單不一致。步驟 3 失敗時優先查的四項是：監督（label／target 有沒有對上）、容量、學習率、訓練路徑（zero_grad→forward→loss→backward→step、detach、optimizer 拿到的參數）。其中監督與訓練路徑屬於資料與程式問題，容量也不是最佳化問題。把步驟 3 概括成「抓最佳化問題，例如學習率不當」，讀者在少量資料 overfit 失敗時，容易直接去調學習率，而這正是失敗二標題警告的做法。 |
| 4 | 建議 | 「失敗二」最後一段：「logits 稍微改一點，argmax 通常不變。這就像階梯函數：在平台上斜率是 0，梯度沒辦法告訴參數該往哪邊調。」 | 數學上的理由正確，但頁面沒說 PyTorch 裡實際會發生什麼。`argmax` 回傳 int64 tensor，requires_grad=False、沒有 grad_fn，根本沒有連著計算圖。實測（PyTorch 2.9.1）把 argmax 的結果轉成 float 拿去算 loss 再 backward，會直接報 `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`；若和其他 loss 項相加，梯度則傳不回模型。這和失敗一的斷圖是同一類問題。讀者照「斜率是 0」去想，會預期程式照跑、梯度全是 0，和實際看到的錯誤對不上。 |
| 5 | 建議 | 檢查步驟 1：「例如模型每筆只輸出 1 個分數、改用 `BCEWithLogitsLoss`……時，label 要用 0.0／1.0 的小數。」 | 這裡只講了 dtype，沒講 shape。BCEWithLogitsLoss 的 target 必須和 input 同 shape：v2.9.1 的 torch/nn/modules/loss.py 第 820–821 行寫著「Input: (*) … Target: (*), same shape as the input.」。模型每筆輸出 1 個分數時，logits 是 `[B,1]`；若沿用 cross entropy 的 `[B]` label、只把它改成 float，實測會報 `ValueError: Target size (torch.Size([2])) must be the same as input size (torch.Size([2, 1]))`。本節正是在教讀者核對 label 的 shape 與 dtype，這裡少講了一半的規定。 |
| 6 | 建議 | 練習 2 參考答案：「先查計算圖有沒有被切斷：有沒有 detach、body 的計算是不是包在 `no_grad` 裡、body 的參數有沒有真的參與 forward。」 | 清單漏了另一個同樣會讓 `body.weight.grad` 一直是 None 的常見原因：參數被設成 `requires_grad=False`。這是本頁「失敗一」末段談的「刻意凍結 backbone」最常見的寫法，載入預訓練模型的程式裡也很常見。照目前的清單查下去，找不到 detach、找不到 no_grad，參數也確實參與了 forward，讀者就會卡住。 |

各項的處理見下方〈定稿修正〉。

## 定稿修正

上面各項意見與先前查核留下的建議，由 AI 逐項核實後處理：必要問題全部修正，建議事項只在修正明確、範圍小時採用。

| # | 來源 | 意見 | 處理 |
|---|---|---|---|
| 1 | 先前查核 | 練習 1 最後仍會印出原題寫死的英文結語 | 已修正：參考答案補一句：這行是原題固定印出的文字，練習版本照樣會印；看到它，就代表所有斷言都通過。已照練習改寫程式實跑確認（1.00/1.00，最後印出結語）。 |
| 2 | 讀者審查 | （必要）把步驟 3 歸成「最佳化問題，例如學習率不當」，和步驟 3 自己的清單、失敗二的警告矛盾 | 已修正：改成：步驟 3 抓「連幾筆訓練資料都背不下來」的問題，照步驟 3 列的順序查（監督、容量、學習率、訓練路徑），不要一背不下來就先調學習率（和技術 3 一起處理）。 |
| 3 | 讀者審查 | 「梯度＝係數×輸入」缺連鎖律這一步，[0.1, 0.5] 也沒辦法核對 | 已修正：補上連鎖律 ∂L/∂w_k＝(∂L/∂z_k)x，並寫明核對位置：在失敗三訓練迴圈的 loss.backward() 之後加一行 print(model.weight.grad)，縮排和那行對齊，第一次印出的就是。實跑插入這行的版本，step 0 印出 [[0.1, 0.5], [-0.1, -0.5]]。 |
| 4 | 讀者審查 | 練習 2 的答案沒有先確認是不是刻意凍結，也沒列 requires_grad=False | 已修正：答案開頭補上「先確認 body 本來就該學」，清單補上 requires_grad=False 和檢查用的 print 寫法。凍結段也補上這種寫法。實測凍結後 body.weight.grad 是 None（和技術 6 一起處理）。 |
| 5 | 讀者審查 | x.repeat(4, 1) 的意思與交錯順序沒有解釋 | 已修正：模型說明段補上 repeat 沿第 0 軸接成 4 份、順序 0、1、0、1……，以及 labels.repeat(4) 的結果。實測確認順序。 |
| 6 | 讀者審查 | 練習 1 印出的固定結語和結果相反 | 已修正：和先前的檢查意見是同一件事，已在參考答案補上說明。 |
| 7 | 讀者審查 | 開頭沒說第三個失敗失敗在哪裡 | 已修正：開頭改成「再用 8 筆人工資料……訓練 20 步：訓練資料全對，刻意改過的 validation 卻全錯」。 |
| 8 | 讀者審查 | 三個失敗的展示順序和檢查順序相反，頁面沒說原因 | 已修正：在失敗一之前補一句：三個失敗照完整程式的執行順序排列，不是照檢查順序；實際除錯仍從步驟 1 查起。 |
| 9 | 讀者審查 | 分界線 b＝−0.2a 怎麼來的，「下面會說明」沒有兌現 | 已修正：「代入 validation」段補上：兩類分數相等的地方是 0.2a+b＝0，也就是圖中的黑線 b＝−0.2a；黑線上方判成類別 1，下方判成類別 0。已核對 SVG 的黑線與上下兩區和正文一致。 |
| 10 | 讀者審查 | overfit 的定義比術語表絕對，和練習 1 的結果衝突 | 已修正：步驟 3 改用術語表的定義「overfit（過擬合：把訓練資料學到幾乎背起來，換新資料卻可能變差）」，接著說明除錯時反過來用，只檢查背不背得下這幾筆資料。 |
| 11 | 讀者審查 | nan、held-out、overfit、BCE 的「小數」用詞和術語表、前幾章不一致 | 已修正：正文改寫成 NaN（並解釋 Not a Number）；held-out 附上「獨立資料」，overfit 附上「過擬合」；BCE 的 label 改成「浮點數（float）的 0.0／1.0」。 |
| 12 | 讀者審查 | 練習 1 的說明全寫在行尾長註解裡，窄螢幕看不到 | 已修正：三行程式的說明改成各自放在程式上方的一行註解，每行程式都保持短。 |
| 13 | 技術查核 | （必要）GPU 出現 device-side assert 後，「搬到 CPU 重跑」照做會失敗 | 已修正：改寫成：GPU 上 label 越界常只報 device-side assert，出錯後同一個行程就不能再用 GPU，連把資料搬回 CPU 都會報同一個錯；要先重新啟動（Colab 是重新啟動工作階段），讓模型和那批資料一開始就放在 CPU 上，再重跑 forward 和 loss。也補上更省事的做法：訓練前先在 CPU 上檢查整份 label。 |
| 14 | 技術查核 | 「學習率 × 梯度」只在 SGD 下等於實際更新量，Adam 不是 | 已修正：改成看每一步參數實際改變了多少（step 前後的差）相對於參數本身的大小；用本節這種預設設定的 SGD 時，這個差就是學習率 × 梯度，用 Adam 時不是。保留「不要只看梯度的絕對值」。 |
| 15 | 技術查核 | 步驟 3 被概括成「最佳化問題，例如學習率不當」 | 已修正：和讀者的必要項一起處理：步驟 3 抓「連幾筆訓練資料都背不下來」的問題，照步驟 3 的順序查（監督、容量、學習率、訓練路徑）。 |
| 16 | 技術查核 | argmax 在 PyTorch 裡實際上沒有接在計算圖上 | 已修正：補上：argmax 的結果是整數 tensor，沒有接在計算圖上；單獨拿它算 loss，backward 會直接報錯，和別的 loss 項相加時，這部分的梯度也傳不回模型。實測 argmax 是 int64，backward 報 element 0 of tensors does not require grad。 |
| 17 | 技術查核 | BCEWithLogitsLoss 的 label 只講了 dtype，沒講 shape | 已修正：補上 shape 也要和輸出相同：輸出是 [B,1] 時，label 也要是 [B,1]。實測沿用 [B] 的 label 會報 Target size must be the same as input size。 |
| 18 | 技術查核 | 練習 2 的檢查清單漏了 requires_grad=False | 已修正：練習 2 答案補上 requires_grad=False，並附檢查寫法 print([p.requires_grad for p in body.parameters()])；凍結段也補上這種寫法（和讀者 2 一起處理）。 |
| 19 | 修正後的檢查 | 「在第一次 backward() 之後」沒說是哪一次，插在失敗一會 NameError | 已修正：改成「在失敗三訓練迴圈的 loss.backward() 之後加一行 print(model.weight.grad)，縮排和那行對齊；第一次印出的就是」。在暫存副本把這行插進程式的那個位置實跑，exit 0，step 0 印出 [[0.1, 0.5], [-0.1, -0.5]]。 |

修正後由另一位 AI 檢查這一批頁面（`docs/lessons/00-warmup.md`、`docs/lessons/01-small-cnn.md`、`docs/lessons/02-diagnostics.md`）的改動，第 1 次：有必要問題。檢查內容：每項改動是否符合程式、紀錄與引用的來源（需要時重算或重跑），回報已修正的必要問題是否真的修好、沒改的理由是否成立，改動是否符合寫作規範，網站嚴格建置與程式摘錄比對是否通過。

| # | 嚴重度 | 位置 | 留下的意見 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | docs/lessons/02-diagnostics.md 第 103 行（〈為什麼 b 的權重固定是 a 的 5 倍〉摺疊區） | 新寫的「在第一次 backward() 之後印出 model.weight.grad 就看得到」沒說是哪一次 backward()。完整程式裡第一次 backward() 在失敗一（body／head），那時 model 還沒定義。照字面把 print 插在那裡，會得到 NameError: name 'model' is not defined。 | 由修正者再處理（上表來源為「修正後的檢查」的列） |

修正後由另一位 AI 檢查這一批頁面（`docs/lessons/00-warmup.md`、`docs/lessons/01-small-cnn.md`、`docs/lessons/02-diagnostics.md`）的改動，第 2 次：通過。檢查內容：每項改動是否符合程式、紀錄與引用的來源（需要時重算或重跑），回報已修正的必要問題是否真的修好、沒改的理由是否成立，改動是否符合寫作規範，網站嚴格建置與程式摘錄比對是否通過。

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 3 輪：上一輪的處理與審查紀錄：通過

以腳本核對紀錄：第 1 次查核的建議、讀者審查 11 項、技術查核 6 項與修正後檢查 1 項，都在〈定稿修正〉19 列處理；批次檢查兩輪的結果都記了；技術查核列出 PyTorch 文件來源；結構檢查通過。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/02-diagnostics.md 第 36、43、123 行 | 路徑殘句：「紀錄位置：暫存副本」「暫存副本在 …/暫存副本。」「3. 摘錄檢查：`python3 摘錄比對工具 <暫存副本> docs/lessons/02-diagnostics.md`」（指令已經不是能執行的形式）。 | 未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 4 輪：上一輪的處理：通過

第 3 輪第 1 項：點名的「紀錄位置：暫存副本」變成標題加檔名清單（第 36–41 行）；「暫存副本在暫存副本。」已刪；第 121 行指令改成「摘錄比對輸出 []」。處理說明屬實。

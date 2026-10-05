# 審查紀錄：VGG 風格小 CNN

審查範圍：`docs/lessons/01-small-cnn.md`、頁面上的圖（`docs/assets/diagrams/01-small-cnn-learning.svg`、`docs/assets/diagrams/01-small-cnn.svg`），以及 `lesson_cases/01-small-cnn.py` 與它 import 的 repo 模組；頁尾自動產生的執行紀錄區塊不在範圍內，由 `scripts/validate_curriculum_evidence.py` 對照紀錄檢查。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面依目前的程式改寫後，由另一位 AI 獨立查核：在獨立的副本執行該節程式、照頁面做練習，逐句對照程式、執行紀錄與手算，檢查程式摘錄與網頁轉換，並從初學讀者（高中程度、數學好、程式新手）的角度看用詞與說明順序。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：有必要問題

結論：不通過。有 1 個 must：第 11 行說三步實驗確認「梯度與參數更新都正確」，但程式只檢查梯度是有限值、權重有改變。另有 2 個 should：第 87 行「高度」容易被讀成矩形尺寸；第 196 行「JSON」沒有解釋。其餘六項檢查都通過。

所有程式都在暫存副本執行，repo 內沒有執行任何程式。

**1. 程式敘述**
- 課程程式 exit 0。stdout 和紀錄比，只有兩行不同：step=0 從 0.6941 變 0.6942，step=1 從 0.6940 變 0.6941。預測仍全 1，錯誤索引 0、2、4、6。
- 用畫素實測資料：
  - 頂邊是 [6,6,11,11,6,6,11,11]，左緣是 [4,8,12,4,8,12,4,8]。
  - 跨類矩形完全相同的配對只有 (4,1) 與 (6,3)。
  - 只看位置最多分對 6/8，只看高度 4/8。
  - 初始權重下，8 張的類別 1 機率都是 0.5215–0.5216。
  - 第 87、145、171 行的資料與斷言描述都正確，沒有把「至少一對」寫成逐對核對，也沒有說位置完全沒有資訊。
- 照頁面做自主練習：改成 `SmallCNN(width=8)` 後，先印出 parameters=4330、乘加數 1695776，接著在 `assert params == 1158 and macs == 479248` 出現 AssertionError；把斷言改成 4330／1695776 後跑完三步，exit 0。參考答案的逐層數字都核對過。
- 40 步腳本（不加 --record）exit 0：
  - 只寫入 artifacts/runs/learning/01-small-cnn/，網站紀錄與圖逐位元組未變。
  - 印出的 JSON 等於 report.json 去掉 loss_history。
  - 重產的圖是中文標題、單線、沒有圖例，和頁面描述一致。
  - 腳本直接呼叫 make_batch() 與 SmallCNN，亂數種子同為 7，起點與三步實驗相同。

**2. 先前審查意見與受程式改動影響的段落**
- 全部處理：第 87 行的補丁括號已刪；40 步段落移到練習之前，標題格式和 03、04 一致，#forty-steps 錨點存在；limits 的解釋已刪；收尾句已改；SVG 的 title、desc 和兩行可見文字已統一。
- 全頁沒有修訂或審查口吻，方向詞（上方、下方、前面）在搬移後都正確。
- 執行紀錄區塊逐字未變。

**3. 數字**
- 所有確定性數字都和程式一致（1158、479248、各層參數與乘加數、4330、1695776、感受野、6/8 等）。
- 沒有引入本機數字。本機新碼的值（首個 loss 0.694159、第 31 點起為 0、最小梯度約 1.4e-18、最大約 0.80）都沒有進頁面；保留的舊值都列在待重錄數值清單。

**4. 摘錄**
- 摘錄比對工具印出 []。
- SmallCNN 區塊去掉註解後，和程式第 29–42 行逐行相同。
- 訓練一步的區塊已寫明是改寫的示意；正文沒有程式行號。

**5. 可讀性**
順序合理；Adam、L2 長度都有解釋。只剩上面兩個建議。

**6. 渲染**
- zensical build --clean --strict exit 0，validate_site.py exit 0。
- validate_lessons 只因審查紀錄過期而失敗，屬發布前的預期狀態。
- docs/assets/diagrams/01-small-cnn.svg：viewBox、title、desc 都在，xmllint 通過；qlmanage 與 Chrome 都渲染乾淨。四個方塊照 3 單位／畫素換算，和 make_batch 一致，也和程式 PNG 的前四格相同。
- 和 01 相關的其他檔案都沒有變動：notebook、learning SVG、兩份紀錄、reviews、lesson_cases、腳本。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 必要 | docs/lessons/01-small-cnn.md 第 11 行（開場段「本節做兩個實驗，都在 CPU 上執行：完整程式只訓練 3 步，目的是確認形狀、梯度與參數更新都正確…」） | 這次編輯把原句「目的是確認形狀、梯度與參數更新」改成「…都正確」，誇大了程式實際檢查的範圍。lesson_cases/01-small-cnn.py 實際檢查的是：影像的 shape、dtype 與數值範圍；參數量與乘加數（間接核對各層 shape）；每一步首層權重的梯度是有限值（`torch.isfinite`）；訓練後首層權重有改變（`not torch.equal(before, …)`）。程式既沒有核對梯度數值算得對不對，也沒有核對更新量對不對。具體後果：讀者若把這三步當成自己模型的正確性測試，遇到符號錯、標籤對應錯這類會產生有限梯度、權重照樣會變的錯誤時，仍會以為「梯度已確認正確」。這句也和本頁〈核對與看錯誤〉的精確描述、第 157 行「三步只確認流程能跑」，以及本頁「參數更新成功與分類學好是兩件事」的主旨不一致。 |
| 2 | 建議 | docs/lessons/01-small-cnn.md 第 87 行「紅、藍兩類各有 2 張矩形畫得較高、2 張畫得較低（低 5 個畫素），所以只看高度猜不出類別」，以及同段斷言描述「每類都是 2 張高、2 張低」 | 這裡的「高／低」「高度」指的是矩形在圖上的上下位置（頂邊在第 6 列或第 11 列）。但第 11 行已說每個矩形都是 12×12，「矩形畫得較高」「只看高度」很容易被讀成矩形本身的高（尺寸）。照那樣讀，「只看高度猜不出類別」就成了理所當然、跟論證無關的話。舊句寫的是「位置…高 5 個畫素」，新句把「位置」兩個字拿掉了。 |
| 3 | 建議 | docs/lessons/01-small-cnn.md 第 196 行「執行時印出的 JSON 和 `report.json` 相同，只省略每一步的 loss。」 | 「JSON」是這次才進入正文的術語。本頁、暖身節與術語表都沒有解釋，之前只出現在自動產生的紀錄連結文字裡。目標讀者是程式新手，不知道 JSON 是什麼，就看不懂這句在比較哪兩樣東西。 |

修正必要問題時處理的項目（修正後由第 2 次複查確認）：

- 查核者的必要問題（第 11 行「都正確」誇大）：改寫成「目的是確認形狀正確、首層權重的梯度是有限值、這些權重確實有更新，不是訓練出可用的分類器」。只寫程式實際檢查的範圍：torch.isfinite(model.features[0].weight.grad)，以及 not torch.equal(before, model.features[0].weight)。
- 查核者的建議（第 87 行「高度」會被讀成尺寸）：改成上下位置的寫法：「畫在較高的位置／較低的位置：較高的頂邊在第 6 列，較低的頂邊在第 11 列，低 5 個畫素（列和程式一樣從 0 數起）。所以只看矩形的上下位置，猜不出類別」。斷言改寫成「每類矩形的頂邊都是 2 張在第 6 列、2 張在第 11 列」。SVG desc 的「畫得較高與較低」也一併改成「位置較高和位置較低」。
- 查核者的建議（第 196 行 JSON 未解釋）：在第一次出現處說明 JSON 是用純文字寫資料的格式，每項依序寫欄位名稱、冒號與值，例如 `"seed": 7`。再說腳本執行時印出同樣的 JSON，只省略每一步的 loss（已對照 main() 的 _history 過濾）。第 163 行的 `null` 補上「（沒有值）」。
- 先前審查意見第 3 行（版本標籤過期） Colab 按鈕 tag：保留。頁首已是 lessons-v0.4.0，由 build_lesson_notebooks.py --ref 改寫，不手改。
- 先前審查意見第 87 行（修訂說明）「紅矩形位置也都比藍矩形高 5 個畫素…本節沒有檢查」括號：刪除。改寫成 make_batch() 用 assert 核對的資料事實：每類頂邊各 2 張在第 6 列、2 張在第 11 列；1／4、3／6 位置相同、只差顏色；只看位置最多猜對 6 張；只有顏色能把 8 張完全分開。沒有寫「位置不帶任何資訊」，也沒有把「至少一對」寫成逐對核對。另把 GAP 參數比較拆成下一段。
- 先前審查意見第 178 行（段落重組） 40 步段落：整段移到〈核對與看錯誤〉之後、〈常見錯誤、自主練習與答案〉之前，成為本節第二個主要實驗。標題是「## 訓練 40 步：模型能學會這 8 張圖嗎 { #forty-steps }」，和 03、04 同格式。第 11 行開場交代兩個實驗並連到 #forty-steps；圖的 alt 改為「固定 8 張訓練圖、40 次更新的 loss」。
- 先前審查意見第 180 行（製作經過的敘述）「補充實驗」：改寫為「三步只確認流程能跑；這個實驗看模型能不能真的學會。它由 scripts/run_learning_extensions.py 執行，直接取用完整程式裡的 make_batch() 與 SmallCNN：同樣 8 張圖、同一種模型，亂數種子同為 7…」。SGD→Adam、步數和 optimizer 同時改變的句子保留。
- 先前審查意見第 201 行（其他）「仍保留它原本的錯誤」：改寫為「前面那張錯誤圖板是三步實驗的結果；這裡的全對是 40 步實驗的結果，兩者要分開看。」
- 先前審查意見第 218 行（修訂說明） limits 欄的解釋：刪除。「validation_accuracy 是 null」移到表下說明，現第 163 行，並加註「（沒有值）」。
- trace 01-small-cnn.svg 第 3 行 title／desc／可見文字：title 改為「固定 seed 7、只更新 3 步的小 CNN：四張圖全猜類別 1」。desc 交代比例（1 畫素＝3 單位）、這是程式 PNG 前四張的重畫、方塊左緣與頂邊位置、seed 7、SGD lr 0.1、三步後全猜類別 1。兩行可見文字統一成「只更新 3 步；模型全猜類別 1」與「類別 0＝紅；類別 1＝藍。真值（GT）與預測一起看。」；viewBox、title、desc 都保留。
- 受程式改動影響的段落第 36 行 step=0 0.6941 與「約 0.52」：數字保留，等重錄紀錄替換，列入待重錄數值清單。0.52 在暫存副本的初始權重下實測為 0.5215–0.5216。
- 受程式改動影響的段落第 87 行：同上方先前審查意見，括號已刪，換成 assert 核對的資料事實。
- 受程式改動影響的段落第 145 行（現第 147 行）圖 alt「…兩張紅矩形被錯分」：保留。它依賴紀錄的 predictions 全 1，列入待重錄數值清單。
- 受程式改動影響的段落第 147 行（現第 149 行）：改寫為「頁尾的執行紀錄裡，三步後模型全猜類別 1，錯誤索引是 0、2、4、6。上圖按比例重畫圖板的前四張（編號 0～3）…」。數值保留紀錄值。
- 受程式改動影響的段落第 180 行（現第 157 行）「0.694144…step=0 的 0.6941」：保留現值，列入待重錄數值清單。
- 受程式改動影響的段落第 184 行（現第 161 行）表格列：保留現值，列入待重錄數值清單。
- impact learning-plots 第 190 行英文對照清單：刪除。改成讀圖說明：每點在該次更新之前量；第 1 點未更新，第 40 點是第 40 次更新前；縱軸是 8 張圖交叉熵的平均。
- 受程式改動影響的段落第 197 行（現第 167 行）曲線形狀描述：保留現值，列入待重錄數值清單。
- 受程式改動影響的段落第 199 行（現第 169 行）梯度 L2 最小／最大：保留現值，列入待重錄數值清單。主詞改成 run_learning_extensions.py。
- 受程式改動影響的段落第 201 行（現第 171 行）40 步預測與 loss 0：保留現值，並加上「1 與 4、3 與 6 兩對只差顏色，模型對每一對都給了不同答案」。這句依賴紀錄的 predicted_classes，列入待重錄數值清單。
- 受程式改動影響的段落第 216 行（現第 186 行）「後段 loss 已是 0、梯度 L2 仍大於 0」：保留。它要等重錄後 loss_history 仍到 0 才成立，列入待重錄數值清單。
- impact learning-plots 第 218 行 limits：刪除（同先前審查意見第 218 行）。
- impact learning-plots 第 218 行建議補充：改成三行程式區塊，和 build_lesson_notebooks.py 產生的 notebook 相同。另寫明結果只寫到 artifacts/runs/learning/01-small-cnn/、不覆寫網站紀錄與圖，以及 report.json、learning.svg 各是什麼。--record 是錄製用的，頁面不提。
- 受程式改動影響的段落第 236 行執行紀錄區塊：未動，由 verify_curriculum.py 從紀錄重產；已確認和 HEAD 逐字相同。
- impact 01-small-cnn.svg 第 3 行 desc：改寫。寫入左緣第 4、8、12、4 欄，以及編號 0、1 頂邊在第 6 列、2、3 在第 11 列。
- impact 01-small-cnn.svg 第 17 行：編號 1（藍）y 從 128 改成 113（頂邊第 6 列）。
- impact 01-small-cnn.svg 第 18 行：編號 2（紅）y 從 113 改成 128（頂邊第 11 列）。渲染結果和程式 PNG 前四格一致。
- impact 01-small-cnn-learning.svg：未手改，由 run_learning_extensions.py --record 重畫。

### 第 2 次複查：通過

結論：通過，沒有必要問題。只有一條 should：第 163 行的「紀錄」沒說是哪一份，見下表。

所有程式都在暫存副本裡執行，repo 內沒有執行任何東西。

**1. 頁面對程式的描述：全部屬實**
- 課程程式 exit 0。輸出和紀錄 01-small-cnn.json 比，只有 step=0、step=1 的第 4 位小數不同（0.6942、0.6941），其餘逐行相同，跟受程式改動影響的段落預告的一致。
- 資料性質（畫素實測）：
  - 頂邊是 [6,6,11,11,6,6,11,11]，左緣是 [4,8,12,4,8,12,4,8]。
  - 跨類別、矩形完全重疊的配對正好是 (4,1) 和 (6,3)。
  - 只看高度最多分對 4/8，看完整位置最多 6/8。
  - 初始權重下，類別 1 的機率是 0.52148～0.52159。
  - 所以第 87 行的「最多猜對 6 張」「只看上下位置猜不出類別」「能把 8 張完全分開的線索只有顏色」，以及兩個斷言的描述，都和程式一致；頁面也沒有寫「位置不帶任何資訊」。
- 40 步腳本 run_learning_extensions.py：
  - 用的是 make_batch() 和 SmallCNN，seed 7 在建模型之前設定，起點和三步實驗相同。
  - Adam、lr 0.01、40 步。
  - 每步都檢查梯度是有限值、梯度的 L2 長度大於 0；權重確實有改變。
  - 不加 --record 時只寫入 artifacts/runs/learning/01-small-cnn/。
  - 印出的 JSON 和 report.json 比，只少了 loss_history。
  - 新產生的圖，軸名是「第幾次更新（每點在該次更新之前量）」與「訓練 loss（交叉熵）」，和頁面的讀圖說明一致。
- notebook：「本節可修改的完整實驗」那一格的「可選：40 步學習實驗」確實列了這三行。環境格確實會 clone 固定 tag、安裝指定版本的套件、切換到程式目錄。
- 練習照頁面實際做了一次：
  - 只改 width=8：先印出 parameters=4330 和 1695776，接著在那行斷言出現 AssertionError，exit 1。
  - 再把斷言改成 4330／1695776：跑完三步，exit 0。
  - 參考答案（各層參數、合計 4330、約 3.74 倍、MAC 合計 1695776）全部正確。
- 軸順序寫錯時的錯誤訊息，在 PyTorch 2.9.1 實測相符。

**2. 先前審查意見與受程式改動影響的段落項目：全部處理完**
- 三條補丁說明都已刪除：位置括號、limits 共用字串、英文對照清單。
- 40 步段落已移到練習之前，標題和 03、04 同格式，錨點 #forty-steps 有效。
- 「補充實驗」和「仍保留原本的錯誤」都已改寫。
- 全頁 grep 修訂、製作口吻的字眼，沒有發現。

**3. 數字**
- 確定性數字都和程式相符：1158／479248、各層參數與 MAC、3833984、128 KiB、589824、感受野 16 與角落 10×10、0.27～0.73、0.31。
- 沒有引入這台 Mac 跑出來的數字。
- 隨重錄而變的值都還是舊紀錄的值，編輯已完整列在待重錄數值清單。

**4. 程式摘錄**
- 摘錄比對工具對 docs/lessons/01-small-cnn.md 印出 []。
- SmallCNN 那一塊已標記 data-excerpt；訓練一步那一塊頁面寫明是改寫的示意。
- 正文沒有引用程式的行號。

**5. 渲染與改動範圍**
- 在暫存副本：zensical build --clean --strict exit 0，validate_site.py exit 0。
- 01-small-cnn.svg：
  - xmllint 通過，viewBox、title、desc 都在。
  - 座標換算正確，(72+200k+3x, 95+3y)。
  - qlmanage 渲染結果和程式 PNG 的前四格一致。
- 和第 1 章有關的檔案，只有 docs/lessons/01-small-cnn.md 和 docs/assets/diagrams/01-small-cnn.svg 被改；notebook、learning SVG、review 都沒有動。
- validate_lessons 只因全站審查紀錄還沒做而失敗，這是預期的；摘錄檢查在那之前已經通過。

**範圍外附註（不列為問題）**
- run_learning_extensions.py 產生的 learning SVG 只有 title 和 aria-label，沒有 desc。這屬於產生程式那個部分，不是這次頁面編輯的範圍。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/lessons/01-small-cnn.md 第 163 行（〈訓練 40 步〉表格下方的說明） | 「cnn 是本節模型在紀錄裡的名稱」和「紀錄裡的 `validation_accuracy` 是 `null`」是 40 步紀錄第一次出現。在這之前，頁面說的「紀錄」都是頁尾那份三步的執行紀錄（第 36、149 行）。這份 40 步紀錄是什麼、放在哪裡、JSON 又是什麼，要到第 196 行才說明並附上連結。讀者如果照字面去頁尾找，會找不到 `cnn` 和 `validation_accuracy`，要讀到 30 多行之後才知道指的是哪一份。 |

最後一次查核的建議事項，在下方〈定稿修正〉逐項處理。

## 讀者審查與技術查核

### 讀者審查（AI 以初學讀者身分閱讀、執行程式與練習）

方法：所有工作都在暫存副本進行，沒有寫入 repo 根目錄（git status 裡沒有任何 01 相關的變動）。

讀過的檔案：
- 審查用的事實與寫作規範清單
- docs/lessons/01-small-cnn.md 全文
- docs/glossary.md，以及前置的 docs/lessons/00-warmup.md
- lesson_cases/01-small-cnn.py、scripts/run_learning_extensions.py、miniyolo/figures.py、miniyolo/provenance.py
- notebooks/01-small-cnn.ipynb（各格內容），以及 scripts/build_lesson_notebooks.py 裡的相關字串
- artifacts/checks/curriculum/01-small-cnn.json 與 01-small-cnn-learning.json
- 為了對照用詞，另查 03-projection、04-localization、10-multiscale、11-csp、12-decoupled-head、14-feature-module、15-area-attention

建置與看頁面：
- 在暫存副本執行 `zensical build --clean --strict`，結果 No issues found。
- 讀 site/lessons/01-small-cnn/index.html。
- 另做一份把所有摺疊區打開的頁面副本，用 Playwright 快取的 chromium headless shell 截全頁圖，分段逐段看。MathJax、表格、摺疊區、程式區塊都正常渲染。

看過的圖：
- 用 qlmanage 渲染 docs/assets/diagrams/01-small-cnn.svg，以及 repo 裡的舊版 01-small-cnn-learning.svg（英文標籤）。
- 用 `PYTHONPATH=. OMP_NUM_THREADS=2 MPLBACKEND=Agg … scripts/run_learning_extensions.py --section 01-small-cnn` 重畫 artifacts/runs/learning/01-small-cnn/learning.svg，渲染後看了版面、標題、軸名、圖例與顏色。
- 完整程式輸出的 artifacts/01-small-cnn.png。

執行：
- 完整程式 exit 0。
- 用探針程式看初始 logits 與機率（類別 1 約 0.5215）、float64 初始 loss、head bias 與 GAP 特徵，並模擬「先跑完整程式格，再貼上頁面的訓練一步片段」，得到 NameError。

照頁面做的練習：
1. 自主練習：把 `model = SmallCNN()` 改成 `model = SmallCNN(width=8)`。程式印出逐層 shape 與 4330／1695776 後，在 assert 那一行出現 AssertionError。把 assert 改成 `4330 and 1695776` 後跑完三步。做完後把檔案還原，SHA-256 回到 f15df56…。
2. 40 步重跑：在本機執行 Colab 三行裡的腳本（等價做法），並渲染它畫的圖。本機沒有 IPython，display 兩行無法測；Colab 本身也沒有測。
3. 常見錯誤與其他宣稱：餵 `[8,32,32,3]`，得到的錯誤訊息和頁面引用的相同；另外核對了 1×2 permute／reshape 例子、softmax 做兩次（[0.9,0.1]→[0.69,0.31]，loss 最低 0.313）、標籤越界的訊息、uint8 輸入、float 標籤、對 4 軸 tensor 用 `permute(2,0,1)` 的錯誤，以及在 zsh 裡打 `!python` 的結果。

手算：參數、MAC、記憶體、輸出長度公式、感受野（逐層推出第 i 格看原圖第 4i−6～4i+9 個畫素），以及 width=8 的答案。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | 〈核對與看錯誤〉第 145 行「並把圖板寫入 `artifacts/01-small-cnn.png`……讀圖時先確認矩形顏色與 GT 一致」；〈自主練習〉參考答案第 221 行「觀察錯誤圖板可以」 | 完整程式用 Agg 後端存檔後就 `plt.close`，Colab 的輸出只有文字，不會顯示圖板；頁面也沒說怎麼打開這張 PNG。所以在 Colab 照頁面執行的讀者做不到「讀圖時先確認……再看預測」這一步；做完 width=8 練習後也看不到自己的新圖板（我在暫存副本跑 width=8，這台 Mac 上預測變成全猜 0、錯在 1、3、5、7，只有打開 PNG 才看得到）。頁面上的 01-small-cnn.svg 是紀錄那次前四張的重畫，不是讀者自己的結果。 |
| 2 | 建議 | 〈短程式與實際成本〉第 118–131 行「建立模型與 optimizer，再訓練一步」的示意程式 | 片段直接用 `images`、`labels`，頁面沒說它們從哪裡來。讀者把這段貼進 Colab 新格，即使先跑過完整程式，也會得到 `NameError: name 'images' is not defined`，因為這兩個變數只存在 `main()` 裡面。我在暫存副本模擬「先執行完整程式格，再貼這段」，確實得到這個錯；補一行 `images, labels = make_batch()` 後，三個 shape 依序是 [8,8,8,8]、[8,8]、[8,2]，和註解相符。 |
| 3 | 建議 | 第 87 行（編號 1 與 4、3 與 6 兩對圖位置相同）與第 147–149 行的圖 `01-small-cnn.svg`（只重畫編號 0～3） | 「只看位置最多對 6 張，只有顏色能把 8 張分開」這個論證要用到編號 4、6，但頁面上的圖只畫 0～3，Colab 讀者又看不到完整圖板。頁面也沒列出 8 張圖各自的矩形位置（程式裡的 `left, top = 4 + i % 3 * 4, 6 + (i // 2) % 2 * 5` 不在頁面上）。讀者只能相信，沒辦法自己對照。 |
| 4 | 建議 | 第 7 行「4×32×32＝4096 個輸出」、第 9 行「channel 數分別是 4 和 8」、第 15 行「NCHW：batch、channel、高、寬」 | channel 從第 9、15 行就開始用，到第 42、52 行才零散交代（「channel 是不同的特徵表示」「每個 channel 是一張 H×W 的圖」），始終沒有一句定義。術語表把 channel 的教學頁指向本節，本節卻沒寫出它的中文名「通道」，而 11.1 節的標題就用「通道」。第 7 行的「4×32×32」也沒說 4 是第一層卷積的輸出 channel 數，讀者在第二段就會卡在「4 從哪來」。 |
| 5 | 建議 | 第 93–116 行的模型定義與說明 | 讀者是程式新手，暖身節只預設會變數、函式、list；這段卻是 `class SmallCNN(nn.Module)`、`self.features = …`、`super().__init__()`。頁面只說 `__init__` 建層、`forward` 寫路線，沒解釋 class、`self`、`nn.Module`、`super().__init__()` 是什麼，讀者容易卡在「self 是誰」「為什麼要寫 super」。另外，第 116 行說明 `nn.Conv2d(3, width, 3, padding=1)` 的參數時沒提 stride。讀者要用第 62 行的公式找 s，在程式裡卻找不到；頁面只交代了 `MaxPool2d` 的預設 stride。 |
| 6 | 建議 | 〈常見錯誤〉第 202 行「直接輸入 `[8,32,32,3]`……看到它，先查軸的順序」 | 頁面只教了單張圖的 `permute(2,0,1)`（HWC→CHW），這裡的錯誤卻是整批的 `[8,32,32,3]`（NHWC）。讀者照前文把 `permute(2,0,1)` 用在 4 軸 tensor 上，會得到另一個錯誤（實測：`RuntimeError: … input.dim() = 4 is not equal to len(dims) = 3`），還是不知道正確改法。 |
| 7 | 建議 | 〈訓練 40 步〉第 159 行表頭「第 1 步（未更新）→ 第 40 次更新前的 loss」 | 術語表定義 step（一步）就是「參數更新一次」，所以「第 1 步（未更新）」自相矛盾。同一節的正文和圖用「第 1 點」「第幾次更新（每點在該次更新之前量）」，三步實驗又用從 0 起算的 step=0，讀者得自己對齊三套說法。 |
| 8 | 建議 | 第 72–79 行 shape 表最後一列「GAP、線性層｜`[8,8]` → `[8,2]`」 | 表上 GAP 之後直接是 `[8,8]`，但第 109 行程式註解寫 GAP 是 `[8,8,8,8] → [8,8,1,1]`，第 133 行又說 `flatten(1)` 把 `[8,8,1,1]` 變成 `[8,8]`。對照表格和程式的讀者會以為 GAP 直接輸出 `[8,8]`，看不出中間還有展平這一步。 |
| 9 | 建議 | 第 81–83 行感受野段落的「間距」 | 術語表 stride 的第 ② 義（特徵圖的 stride：相鄰特徵位置在輸入圖上相隔幾個畫素）把教學頁指向本節，但本節只用「間距」這個詞，從沒說它就是後面章節的 stride，例如第 10 章的「8×8（stride 8）」；術語表也查不到「間距」。讀者到第 10 章會接不上這裡學過的概念。 |
| 10 | 建議 | 第 42 行「kernel=3 指濾鏡的高、寬都是 3」、第 62 行「k 是視窗邊長（卷積就是 kernel）」、第 116 行「kernel（濾鏡邊長）」、第 135 行「k 是濾鏡邊長（kernel）」 | 本節把 kernel 當成「濾鏡邊長」的名字。但 3.2 節寫「P 的 kernel 是 1×1」「k 是 kernel 邊長」，11.1、12.2、15.2 節寫「卷積核」——在那些頁，kernel 和卷積核指的是濾鏡本身。術語表裡 filter、kernel、卷積核都沒有。讀者在本節學到「kernel＝邊長」「濾鏡＝filter」，之後讀到「kernel 邊長」「卷積核」，會以為是另一個東西。 |
| 11 | 建議 | 頁尾執行紀錄的輸出行 `Three-step smoke test only; accuracy here is not held-out performance.`；正文第 145 行只說明到 `error_indices` | 正文沒有解釋這行英文。smoke test 全站沒有定義，held-out 要到第 2 章才教；而且這行說「accuracy here」，程式卻沒印出任何正確率，讀者會去找一個不存在的數字。 |
| 12 | 建議 | 第 9 行〈歷史機制〉 | 一、「原版末端的大型全連線層也改用全域平均池化（GAP）取代」：第 7 行剛說全連線層就是線性層，第 40 行又說本節的 head 是一個線性層，讀者容易以為「全連線層已經被 GAP 取代了，為什麼還有 Linear」。二、最後一句 C 版「其中 3 層卷積用 1×1」對初學者是岔題，1×1 卷積本節也沒解釋。 |
| 13 | 建議 | 第 83 行「角落那格只看得到原圖 10×10；8×8 的 64 格中，只有中央 4×4 格的 16×16 整個落在原圖內」 | 前半段一步步算出 16，這兩個結論卻沒有推導。數學好的讀者想驗算時，不知道那 16 個畫素寬的範圍從哪裡開始，因為頁面只講範圍有多寬、沒講它在哪裡。（我逐層推過，結論正確。） |
| 14 | 建議 | 第 36 行「本節模型一開始給每張圖的兩類機率都約 0.5（類別 1 略高，約 0.52）」與第 149 行「為什麼全猜 1？」 | 頁面只說模型一開始就偏向類別 1，沒說為什麼 8 張不同的圖會得到幾乎一樣的分數，讀者容易以為模型「看過圖、覺得都是藍色」。我在暫存副本實測，初始 logits 每張都約 (0.157, 0.243)，紅圖和藍圖的 GAP 特徵幾乎相同，差別主要來自 head 的 bias（seed 7 時類別 1 的 bias 比較大）。 |
| 15 | 建議 | 第 171 行「最後的 float32 交叉熵顯示 0，是有限精度計算與過度自信造成的顯示結果」 | 「顯示 0」「顯示結果」會讓讀者以為真正的值是很小的正數，只是印成 0（像表上的 0.000000）；但第 167 行和摺疊說明都說 float32 算出來剛好是 0。兩種說法互相矛盾。 |
| 16 | 建議 | 第 167 行讀圖說明「前 8 點幾乎持平，第 10～23 點快速下降，從第 30 點起 loss 剛好是 0」 | 描述跳過了第 9 點和第 24～29 點。圖上曲線在第 30 點之前好幾點就已經貼在 0 軸上，讀者看圖會覺得「從第 30 點起才是 0」和圖不合，卻不知道中間那幾點是小到圖上看不出、但紀錄裡還不是 0。圖本身分不出「極小」和「剛好是 0」。 |
| 17 | 建議 | 第 188–196 行「想自己重跑」 | 這段只給 Colab 的寫法。本機讀者把 `!python scripts/run_learning_extensions.py --section 01-small-cnn` 照抄到終端機會失敗（實測 zsh 回報 `command not found: !python`；互動式 shell 還可能把 `!python` 當成歷史指令展開），後兩行 IPython 也只能在 notebook 裡用。頁面沒說本機要怎麼做。 |
| 18 | 建議 | 第 196 行「腳本執行時也會印出同樣的 JSON，只省略每一步的 loss」 | 印出的 JSON 很長，machine、code、dependencies_sha256 底下有十幾個 SHA-256 雜湊。頁面沒說哪幾個欄位對應上面的表格和正文，初學者很難從中找到表上的 loss、正確率、預測和梯度範圍。 |
| 19 | 建議 | 第 145 行「完整程式畫出 8 張圖後，先核對資料」 | 本頁的「畫」同時指產生資料（第 11 行「程式畫的 8 張圖」）和畫圖板（第 17 行「完整程式畫圖時」）。這句容易讀成「畫完圖板才核對資料」，但程式其實是一產生資料就核對，圖板到最後才畫。 |
| 20 | 建議 | 〈常見錯誤〉第 203 行「若 loss 報標籤越界，先確認類別 id 是 0／1」 | 第一條常見錯誤附了實際的錯誤訊息，這一條沒有。初學者實際看到的是英文 `IndexError: Target 2 is out of bounds.`（實測），不一定認得這就是「標籤越界」。 |
| 21 | 建議 | 摺疊說明〈loss 為什麼剛好是 0，梯度卻不是 0〉第 186 行「梯度則直接由錯誤類別的機率（約 2×10⁻⁹）算出來」 | 這句是整段摺疊說明的關鍵，卻只有結論。數學好的讀者會問「為什麼梯度正好是錯誤類別的機率」；CE 的梯度要到 7.3 節才推導，這裡少了一步。 |

### 技術查核（AI 對照原始論文、固定 commit 的官方程式、該節程式與手算）

方法：工作副本：rsync 到暫存副本（工作樹 3152741，排除 .git、site、.venv*、artifacts/runs、data/curated、data/downloads），所有程式都只在這份副本裡執行；另用 .../review/01-small-cnn-tech-work 放下載的來源與檢查腳本。

一手來源：
(1) Simonyan & Zisserman, Very Deep Convolutional Networks for Large-Scale Image Recognition, https://arxiv.org/pdf/1409.1556v6 ：用 macOS PDFKit 抽出全文。查了作者單位行（「Visual Geometry Group, Department of Engineering Science, University of Oxford」）；§2.1（「fixed-size 224 × 224 RGB image」「The convolution stride is fixed to 1 pixel … the padding is 1 pixel for 3× 3 conv. layers」「Max-pooling is performed over a 2× 2 pixel window, with stride 2」「three Fully-Connected (FC) layers: the first two have 4096 channels each, the third … 1000」）；§2.2 與 Table 1（A–E；C 有 conv1-256、conv1-512、conv1-512；D 共 16 個權重層＝13 層 conv3＋3 層 FC；「starting from 64 … increasing by a factor of 2 after each max-pooling layer, until it reaches 512」）；§2.3（「a stack of two 3× 3 conv. layers … has an effective receptive field of 5× 5」、27C² 對 49C²）；§3.1（momentum 0.9、batch 256、weight decay 5·10⁻⁴、「dropout … for the first two fully-connected layers (dropout ratio set to 0.5)」）；§3.2（測試時把 FC 轉成卷積並做空間平均）；§4 與 §4.5（「our “VGG” team」）；Table 9（「VGG Net-D (16 layers)」）。
(2) VGG 官方釋出的 16 層模型：gist https://gist.github.com/ksimonyan/211839e770f7b538e2d8 的 revision ded9363bd93ec0c770134f4e387d8aaaaa2407ce。VGG_ILSVRC_16_layers_deploy.prototxt 有 13 個 Convolution 層，全部 kernel_size 3、pad 1；pool1–pool5 都是 MAX；fc6、fc7 為 4096 並帶 dropout_ratio 0.5；fc8 為 1000。readme.md 寫「In the paper, the model is denoted as the configuration `D`」。
(3) torchvision v0.24.1（commit d801a34632023859a0a274803d6abaf0a45d77a5）torchvision/models/vgg.py：第 93 行 cfg \"D\"，第 433 行 vgg16 → _vgg(\"D\", …)。
(4) PyTorch v2.9.1（commit d38164a545b4a4e4e0cf73ce67173f70574890b6）：torch/nn/modules/pooling.py 第 64 行（ceil_mode 預設 False）與第 68 行（stride 預設等於 kernel_size）；torch/nn/modules/conv.py 第 388 行（valid 2D cross-correlation）；aten/src/ATen/native/cpu/LogSoftmaxKernelImpl.h 第 54–94 行（log_softmax＝x−max−log(Σexp(x−max))，以 scalar_t 累加，佐證 1＋2×10⁻⁹ 被捨入成 1）；torch/optim/adam.py 第 273–304 行（Adam 更新式）。
(5) Lin, Chen, Yan, Network In Network, https://arxiv.org/pdf/1312.4400v3 §3.2 Global Average Pooling（「we propose another strategy called global average pooling to replace the traditional fully connected layers」）。

讀過的 repo 檔案：docs/lessons/01-small-cnn.md、lesson_cases/01-small-cnn.py、scripts/run_learning_extensions.py、docs/assets/diagrams/01-small-cnn.svg（原始碼與座標）、docs/assets/diagrams/01-small-cnn-learning.svg、artifacts/checks/curriculum/01-small-cnn-learning.json、notebooks/01-small-cnn.ipynb（各格內容）、scripts/build_lesson_notebooks.py（grep）、docs/lessons/00-warmup.md、docs/lessons/04-localization.md、docs/lessons/05-assignment.md、docs/glossary.md（grep）、zensical.toml、miniyolo/__init__.py、figures.py、provenance.py 的 import、reviews/01-small-cnn.md、審查用的事實與寫作規範清單。

執行的程式與指令（都在工作副本內）：
- PYTHONPATH=. OMP_NUM_THREADS=2 MPLBACKEND=Agg .venv-model/bin/python lesson_cases/01-small-cnn.py：exit 0。shape 六行、parameters=1158、MAC 479248、predictions 全 1、error_indices=[0,2,4,6] 都與頁面一致。step 的 loss 小數會因電腦而異，依指示不比較。
- scripts/run_learning_extensions.py --section 01-small-cnn：exit 0，預測 [0,1,0,1,0,1,0,1]、train_accuracy 1.0、validation_accuracy null。
- 自寫 check1.py：初始 p(類別 1)≈0.5215，三步後≈0.518；label 為 int64，int32 會被拒；[8,32,32,3] 的錯誤訊息；標籤越界的 IndexError；uint8 與 float64 輸入都會報錯；permute 與 reshape 的例子；softmax 做兩次的界限 0.2689～0.7311、最低 loss 0.3133；分數差 20 時 loss=0.0、梯度 2.06e-9；分數差約 16.7 起 loss 變 0。
- 自寫 rf.py：用全 1 卷積與 avg pool 反傳，確認角落格 10 列、中央 2～5 格各 16 列。
- 把 width 改為 8 的兩份修改副本：只改寬度時印出 4330／1695776 後在斷言出現 AssertionError；同時改斷言後 exit 0；各層參數 224、584、1168、2320、34。
- 頁面第 120–131 行的示意程式：shape 依序為 [8,8,8,8]、[8,8,1,1]、[8,8]、[8,2]，與 model(images) 結果相同。
- 摘錄比對：[]（摘錄一致）。
- .venv-docs/bin/zensical build --clean --strict：exit 0、No issues found；再檢查產出的 HTML：6 個摺疊區塊、2 個表格、數學已轉換、#forty-steps 錨點存在。
- qlmanage -t -s 1200 渲染 repo 裡的兩張 SVG 與重畫的 artifacts/runs/learning/01-small-cnn/learning.svg 後實看。repo 裡是舊版英文圖；重畫的是中文標籤、沒有圖例，與頁面描述相符。另實看程式產生的 artifacts/01-small-cnn.png。
- gh api／curl 取得上列來源，swiftc 編出 PDFKit 抽字工具；grep 頁面查製作經過類用語（沒有命中）。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/lessons/01-small-cnn.md:9（「本節只保留『小卷積重複堆疊』的想法」）；相關：第 83 行感受野段 | 「只保留」低估了本節沿用的 VGG 設計，也把來源說錯了。VGG 論文 2.1–2.2 節的通用規則是：3×3 卷積、stride 1、padding 1（高寬不變），每層後接 ReLU；用 2×2、stride 2 的 max pooling；每次 pooling 後 channel 加倍（64→…→512）。本節這幾條全部照用。兩個 block（兩層卷積＋pooling，channel 4→8）的排法，和 VGG16 官方 prototxt 前兩組（conv1_1、conv1_2、pool1、conv2_1、conv2_2、pool2）一模一樣。照現在的寫法，讀者會以為 pooling 方式與 channel 加倍是本節自訂的。頁面也沒交代論文為什麼要堆疊 3×3。2.3 節的說法是：兩層 3×3 的有效感受野等於 5×5、三層等於 7×7，三層 3×3 只要 27C² 個權重，一層 7×7 要 49C²，而且層與層之間多了非線性。第 83 行算出的 1→3→5 正是這個論點，卻沒有和論文接起來。原版的尺度（輸入 224×224、五組、64～512 channel）也只用「縮得很小」帶過。 |
| 2 | 建議 | docs/lessons/01-small-cnn.md:9（「原版末端的大型全連線層也改用全域平均池化（GAP）取代」） | 原版末端是三層全連線：FC-4096、FC-4096、FC-1000，前兩層帶 dropout 0.5（論文 2.1、3.1 節；官方 prototxt 的 fc6/drop6/fc7/drop7/fc8）。本節換成的是「GAP＋一層 Linear(8,2)」，不是只有 GAP；head 是線性層這件事要到第 40 行才出現，兩處沒有接起來。GAP 也不是 VGG 的設計，出自 Network in Network（arXiv:1312.4400 第 3.2 節）。NIN 的原式是把各 feature map 平均後直接送進 softmax，完全沒有全連線層，本節的 GAP＋Linear 又和它不同。照現在的寫法，讀者可能以為 GAP 屬於 VGG，或以為本節已經沒有全連線層。 |
| 3 | 建議 | docs/lessons/01-small-cnn.md:171（「最後的 float32 交叉熵顯示 0，是有限精度計算與過度自信造成的顯示結果」）與第 177 行（「過度自信指正確類別的分數遠大於另一類」） | (a) 最後的 loss 是 float32 真的算出了 0.0，不是印出時四捨五入的「顯示」問題。紀錄的 last_pre_update_loss 就是 0.0；摺疊說明第 175、184 行也正確地寫「loss 算出來就是 0」。第 171 行的「顯示結果」和這兩處矛盾，會讓讀者以為真值只是小到印不出來。(b) 第 177 行把「過度自信」定義成「正確類別分數遠大於另一類」，這不是 ML 的用法。overconfidence 指預測信心高於實際正確率，是校準問題。這裡 8 張訓練圖全部分對，要說的其實是分數差（margin）極大、信心極高。初學者先記住這個定義，之後讀到校準相關內容會誤解。 |
| 4 | 建議 | docs/lessons/01-small-cnn.md:15（「應先轉 float 再除 255：模型權重是 float32，輸入也要是 float」） | 「float」有歧義。Conv2d 要求輸入和權重是同一種型別，也就是 float32。NumPy 的 astype(float) 或 Python 的 float 都是 64 位元，轉成 tensor 是 float64。在 PyTorch 2.9.1 實測，conv 會報 `Input type (double) and bias type (float) should be the same`。照字面「轉 float」去做的讀者仍可能失敗。 |
| 5 | 建議 | docs/lessons/01-small-cnn.md:145（「完整程式畫出 8 張圖後，先核對資料……然後印出表中的逐層 shape」） | 程式裡畫圖板是最後一步（lesson_cases/01-small-cnn.py 第 82–93 行），核對資料是在 make_batch() 產生圖片時就做。「畫出 8 張圖後」很容易讀成「先畫圖板再核對」，和同段後面「最後……把圖板寫入」的順序衝突。另外，「印出表中的逐層 shape」也不精確：程式只在每層 Conv2d、MaxPool2d 之後各印一行，共 6 行，不印輸入那一列，也不印 GAP、線性層那一列。 |
| 6 | 建議 | docs/lessons/01-small-cnn.md:167（「前 8 點……第 10～23 點……從第 30 點起 loss 剛好是 0」）；畫圖程式 scripts/run_learning_extensions.py:136–139（ax.plot 沒有點標記、縱軸是線性刻度） | 程式畫的是不帶點標記的連線，縱軸是 0～0.7 的線性刻度，10⁻³ 以下的值在圖上和 0 分不出來。在本機用程式重畫的 learning.svg 中，曲線大約第 25 點起就貼在 0 上，但 loss_history 要再晚好幾點才真的是 0.0。讀者照正文「從第 N 點起剛好是 0」去看圖，會看到曲線更早就到 0，也找不到標示「第幾點」的記號。「剛好是 0」只能從紀錄 JSON 的 loss_history 核對。這是正文描述與畫圖程式之間的落差，和各電腦算出的數值差異無關。 |
| 7 | 建議 | docs/lessons/01-small-cnn.md:42 與 44–48（卷積的定義與〈手算一次：濾鏡怎麼對邊緣起反應〉） | 頁面把「對應位置相乘再全部相加」叫做卷積。這和 PyTorch 的 Conv2d 一致，但 PyTorch 文件明說它算的是 cross-correlation（pytorch@d38164a5 torch/nn/modules/conv.py 第 388 行："where ⋆ is the valid 2D cross-correlation operator"）。數學上的卷積會先把濾鏡上下左右翻轉。數學好的讀者之後若照數學定義重算〈手算一次〉，[−1,0,1] 濾鏡會變成對「左亮右暗」起反應，和頁面的結論相反，卻不知道原因。 |
| 8 | 建議 | docs/lessons/01-small-cnn.md:81–83（「間距是目前特徵圖的一格相當於原圖幾個畫素」） | 這裡的「間距」就是術語表 stride 條目的第 ② 義「特徵圖的 stride」（docs/glossary.md 第 55 行，而且這個條目把本節列為出處），但本節從沒用過這個名稱。後面章節講的是 stride 8／16／32，讀者很難把它和這裡的「間距」對起來，也不符合審查用的事實與寫作規範清單「術語與術語表一致」的寫作規則。 |

各項的處理見下方〈定稿修正〉。

## 定稿修正

上面各項意見與先前查核留下的建議，由 AI 逐項核實後處理：必要問題全部修正，建議事項只在修正明確、範圍小時採用。

| # | 來源 | 意見 | 處理 |
|---|---|---|---|
| 1 | 先前查核 | 40 步紀錄第一次出現時沒說是哪一份紀錄 | 已修正：第一次提到時改成「這個實驗另存一份紀錄：[40 步結果](…/artifacts/checks/curriculum/01-small-cnn-learning.json)（JSON 檔，欄位怎麼對照見下方「想自己重跑」那段），cnn 是本節模型在裡面的名稱」，在這裡就直接附上 40 步紀錄的連結。後面也改寫成「這份紀錄」「這份 40 步紀錄」。 |
| 2 | 讀者審查 | Colab 看不到程式存出的圖板 | 已修正：已確認程式只存檔，沒有顯示圖板。補上看圖方法：Colab 用 IPython.display.Image 顯示 artifacts/01-small-cnn.png；本機用看圖程式打開這個檔案。notebook 會 chdir 到 repo 根目錄，所以相對路徑成立。 |
| 3 | 讀者審查 | 「訓練一步」示意程式貼上執行會出現 NameError | 已修正：實測重現了 NameError。片段第一行補上 images, labels = make_batch()，並說明要先執行完整程式那一格。照 notebook 的順序實跑，三個 shape 是 (8,8,8,8)、(8,8)、(8,2)。 |
| 4 | 讀者審查 | 圖只畫 0～3，讀者沒辦法驗證 1/4、3/6 兩對圖的位置相同 | 已修正：不改圖。照程式公式核對過 8 張矩形的頂邊列與左緣欄，用一句話列出來（6/4、6/8、11/12、11/4、6/8、6/12、11/4、11/8），讀者可以自己對照。 |
| 5 | 讀者審查 | channel 沒有定義，也沒寫出中文名；4×32×32 的 4 沒交代 | 已修正：NCHW 段的定義改成和術語表一致：「channel（通道）是圖的一層：每個位置在每個 channel 上各有一個數，channel 數就是同一個位置有幾個數」；RGB 的 3 個 channel 是 R、G、B 三層，每個輸出 channel 是一個濾鏡算出的一張圖。開頭第二段第一次出現時寫明第一層卷積輸出 4 個 channel、每個 32×32，並註明「channel 就是通道，下文說明」。 |
| 6 | 讀者審查 | class、self、super 沒有解釋；Conv2d 的 stride 預設值沒交代 | 已修正：模型說明段補兩句，交代 class/nn.Module、super().__init__()、self 屬性與 parameters() 的關係，並補上「沒寫 stride 時預設是 1，所以 s＝1」。 |
| 7 | 讀者審查 | 整批 NHWC 要怎麼改，頁面沒給 | 已修正：常見錯誤補上整批 [N,H,W,C] 要用 permute(0,3,1,2)。實測確認單張圖用的 permute(2,0,1) 用在 4 軸 tensor 上會報錯。 |
| 8 | 讀者審查 | 表頭「第 1 步（未更新）」和 step 的定義矛盾 | 已修正：表頭改成「loss：第 1 點（還沒更新）→ 第 40 點（第 40 次更新前）」，和正文、圖用的「點」一致。 |
| 9 | 讀者審查 | shape 表把 GAP 直接寫成 [8,8]，漏了展平這一步 | 已修正：最後一列拆成三列：GAP [8,8,1,1]、展平 flatten(1) [8,8]、線性層（head）[8,2]。 |
| 10 | 讀者審查 | 「間距」沒對應到術語表的「特徵圖的 stride」 | 已修正：感受野段補一句「後面章節把這個間距叫特徵圖的 stride」，並連到術語表（和技術 8 一起處理）。 |
| 11 | 讀者審查 | kernel 被當成「濾鏡邊長」的名字，和後面章節的用法不一致 | 已修正：第一次出現時寫成「濾鏡（filter），也常叫 kernel 或卷積核」，kernel=3 註明是 kernel_size。其餘各處改寫成「kernel 邊長」。術語表不在可修改的範圍內，沒有改。 |
| 12 | 讀者審查 | 輸出的英文 smoke test 那行沒有說明 | 已修正：補一句說明這行英文的意思（只是確認流程能跑的三步檢查，結果不代表在 held-out 上的表現），也說明 panel= 那行是圖板存檔的位置。 |
| 13 | 讀者審查 | 「全連線層被 GAP 取代」容易誤解；C 版那句是岔題 | 已修正：前半和技術 2 一起改：原版末端是三層全連線層，本節改成 GAP 再接一層很小的線性層。後半（C 版那句）沒動：技術查核確認內容正確，搬進摺疊區只是編排上的偏好。 |
| 14 | 讀者審查 | 角落 10×10、中央 4×4 這兩個結論沒有推導 | 已修正：已逐層推算核對，第 i 格看的是原圖第 4i−6 到 4i+9 個畫素。補一句推導：i＝0 時只有 10 個畫素在圖內，2≤i≤5 時整段都在圖內。 |
| 15 | 讀者審查 | 沒說為什麼 8 張圖的初始分數幾乎一樣、都偏向類別 1 | 已修正：實測初始 logits 每張都約是 (0.157, 0.243)，各張只在小數第 4 位不同；head 的 bias 是 (0.206, 0.270)，兩類分數差約 0.086，其中 0.065 來自 bias。據此補上定性的原因（8 張圖到 GAP 時幾乎一樣，分數差主要來自 head 的 bias），頁面沒有寫入這些實測數字。 |
| 16 | 讀者審查 | 「顯示 0／顯示結果」和摺疊說明的「算出來剛好是 0」矛盾 | 已修正：改成「用 float32 算出來剛好是 0，不只是印成 0」，原因是分數差極大，加上 float32 的捨入（和技術 3 一起處理）。 |
| 17 | 讀者審查 | 讀圖描述跳過部分點，圖上也分不出「剛好是 0」 | 已修正：改成「前 8 點幾乎持平，第 9～23 點快速下降，之後貼著 0。圖上分不出極小和剛好是 0；這份 40 步紀錄的 loss_history 裡，從第 30 點起剛好是 0」。已用紀錄核對：第 9 點是 0.664，已經開始下降。點號保留紀錄的現值，列入隨紀錄改變的數字（和技術 6 一起處理）。 |
| 18 | 讀者審查 | 重跑 40 步實驗只給了 Colab 的寫法 | 已修正：補上本機做法：在專案根目錄執行 python scripts/run_learning_extensions.py --section 01-small-cnn（不加 !），再用瀏覽器打開 learning.svg。腳本會自己 insert sys.path，所以不需要 PYTHONPATH。 |
| 19 | 讀者審查 | 印出的 JSON 沒有說哪幾個欄位對應表格 | 已修正：照紀錄與腳本實際輸出的欄位名稱，補一句對照：initial_loss／last_pre_update_loss、train_accuracy／validation_accuracy、predicted_classes、min／max_gradient_l2；machine、code、dependencies_sha256 可以先略過。 |
| 20 | 讀者審查 | 「畫出 8 張圖後，先核對資料」的順序有歧義 | 已修正：改成「用 make_batch() 產生 8 張圖時就先核對資料」（和技術 5 一起處理）。 |
| 21 | 讀者審查 | 標籤越界那條沒有附錯誤訊息 | 已修正：實測取得訊息，補上原文 IndexError: Target 2 is out of bounds.，並說明這表示標籤裡出現了 2。 |
| 22 | 讀者審查 | 「梯度由錯誤類別的機率算出」少了推導 | 已修正：補一行推導：令分數差為 d，loss＝ln(1+e^{−d})，對 d 微分得 −e^{−d}/(1+e^{−d})，正好是錯誤類別機率的負值。 |
| 23 | 技術查核 | 「只保留小卷積堆疊」低估了沿用的 VGG 規則，也沒接上論文 2.3 節 | 已修正：改成「沿用它的幾條設計規則（3×3 卷積重複堆疊、2×2 的 max pooling、每次 pooling 後 channel 加倍，下文都會說明；原版加倍到 512 為止）」。感受野段補一句論文的論點：兩層 3×3 等於一層 5×5、三層等於 7×7，27C² 對 49C²，層間多了 ReLU。 |
| 24 | 技術查核 | 末端換成的是 GAP＋Linear；GAP 出自 NIN，不是 VGG 的設計 | 已修正：改成「原版末端是三層很大的全連線層（輸出 4096、4096、1000），本節改成 GAP 再接一層很小的線性層」，並註明 GAP 出自 Network in Network（附 arXiv 連結）。 |
| 25 | 技術查核 | 「顯示 0」與「過度自信」用詞不精確 | 已修正：正文改成「算出來剛好是 0」。摺疊說明裡的「過度自信指…」改成「關鍵是正確類別的分數遠大於另一類」，不再沿用 ML 裡指校準問題的那個詞。 |
| 26 | 技術查核 | 「轉 float」應寫明 float32 | 已修正：改成「轉成 float32（.float()）」。實測 float64 輸入會報 Input type (double) and bias type (float) should be the same，所以頁面也補上「轉成 float64 一樣會報錯」。 |
| 27 | 技術查核 | 「畫出 8 張圖後」有歧義；「印出表中的逐層 shape」不精確 | 已修正：改成「產生 8 張圖時就先核對資料」，並寫明程式印出的是每層卷積與 pooling 的輸出 shape（共 6 行，對應表中第 2～5 列）。和實跑輸出的 6 行相符。 |
| 28 | 技術查核 | 曲線沒有點標記、縱軸是線性刻度，看不出從第幾點起剛好是 0 | 已修正：正文說明圖上分不出極小和剛好是 0，剛好是 0 要看紀錄的 loss_history。畫圖程式不在可修改的範圍內，程式畫的圖也不能改，所以只改正文。 |
| 29 | 技術查核 | PyTorch 的 Conv2d 算的是 cross-correlation，頁面沒說 | 已修正：在〈手算一次〉摺疊區末尾補一句：PyTorch 文件稱為 cross-correlation，數學課本的卷積會先翻轉濾鏡；權重是訓練出來的，所以不影響。 |
| 30 | 技術查核 | 「間距」沒對應到術語表的特徵圖 stride | 已修正：和讀者 9 一起處理：補一句「後面章節把這個間距叫特徵圖的 stride」，並連到術語表。 |
| 31 | 修正後的檢查 | （必要）「本節最後有連結」指向頁尾的三步紀錄，不是 40 步紀錄 | 已修正：確認頁尾〈實際執行紀錄〉只有三步實驗的 01-small-cnn.json。現在第一次提到 40 步紀錄時就直接附上 [40 步結果] 連結（artifacts/checks/curriculum/01-small-cnn-learning.json，和「想自己重跑」那段最後的連結是同一個網址），欄位對照則指向下方「想自己重跑」那段。頁面已經不再寫「本節最後」。 |
| 32 | 修正後的檢查 | channel 在定義之前就用了，定義的第一句也和術語表不一致 | 已修正：開頭第二段第一次出現時補上「channel 就是通道，下文說明」。NCHW 段的定義改成和術語表一致的「圖的一層：每個位置在每個 channel 上各有一個數，channel 數就是同一個位置有幾個數」。 |
| 33 | 修正後的檢查 | 讀圖說明仍跳過第 9 點 | 已修正：改成「第 9～23 點快速下降」。紀錄的第 9 點是 0.664，已經開始下降（第 8 點是 0.678）。點號列入隨紀錄改變的數字。 |
| 34 | 修正後的檢查 | VGG 的「每次 pooling 後 channel 加倍」少了上限 512 | 已修正：括號補上「原版加倍到 512 為止」，和論文 Configurations 一節的原文一致：after each max-pooling layer, until it reaches 512。 |

修正後由另一位 AI 檢查這一批頁面（`docs/lessons/00-warmup.md`、`docs/lessons/01-small-cnn.md`、`docs/lessons/02-diagnostics.md`）的改動，第 1 次：有必要問題。檢查內容：每項改動是否符合程式、紀錄與引用的來源（需要時重算或重跑），回報已修正的必要問題是否真的修好、沒改的理由是否成立，改動是否符合寫作規範，網站嚴格建置與程式摘錄比對是否通過。

| # | 嚴重度 | 位置 | 留下的意見 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | docs/lessons/01-small-cnn.md 第 168 行（〈訓練 40 步〉表格下方第一段） | 新寫的「這個實驗另存一份紀錄（JSON 檔，本節最後有連結）」指錯了位置。本頁的「本節」都指整節課（本節模型、本節卷積、完成本節）。頁面最後是〈實際執行紀錄〉，那裡唯一的 JSON 連結是三步實驗的 01-small-cnn.json，裡面沒有 cnn、validation_accuracy 或 loss_history。40 步紀錄的連結其實在〈想自己重跑〉說明的最後（「40 步結果」），它後面還有〈常見錯誤、自主練習與答案〉和〈實際執行紀錄〉兩節。讀者照這句話去頁尾找，會打開錯的紀錄，也找不到這段說的 validation_accuracy: null。這個問題本來就是要講清楚「是哪一份紀錄」，這樣改反而指向另一份。 | 由修正者再處理（上表來源為「修正後的檢查」的列） |
| 2 | 建議 | docs/lessons/01-small-cnn.md 第 7、9、15 行 | channel 在第 7 行就出現了（這次新加的「第一層卷積輸出 4 個 channel」），第 9 行又用了兩次，可是定義到第 15 行才給，中間也沒有「下文說明」之類的提示。另外，定義的第一句「channel（通道）是同一個位置上並排的幾個數」說的其實是同一個位置上所有 channel 的值，和術語表的「特徵圖的一層」不一致，也和同一句後半的「一個濾鏡算出的一張圖」不一致。 | 由修正者再處理（上表來源為「修正後的檢查」的列） |
| 3 | 建議 | docs/lessons/01-small-cnn.md 第 172 行（讀圖） | 原本的讀者發現明確指出描述跳過了第 9 點和第 24～29 點。這次修改用「之後貼著 0」補上了第 24～29 點，但「前 8 點幾乎持平，第 10～23 點快速下降」還是跳過第 9 點。紀錄裡第 9 點是 0.664，已經開始下降。 | 由修正者再處理（上表來源為「修正後的檢查」的列） |
| 4 | 建議 | docs/lessons/01-small-cnn.md 第 9 行 | 頁面把「每次 pooling 後 channel 加倍」當成沿用 VGG 的設計規則，但論文 2.1 節的原文是 max-pooling 後加倍「until it reaches 512」，也就是加倍到 512 就停（VGG16 最後兩組都是 512）。現在的寫法把論文的規則說得太寬了。 | 由修正者再處理（上表來源為「修正後的檢查」的列） |

修正後由另一位 AI 檢查這一批頁面（`docs/lessons/00-warmup.md`、`docs/lessons/01-small-cnn.md`、`docs/lessons/02-diagnostics.md`）的改動，第 2 次：通過。檢查內容：每項改動是否符合程式、紀錄與引用的來源（需要時重算或重跑），回報已修正的必要問題是否真的修好、沒改的理由是否成立，改動是否符合寫作規範，網站嚴格建置與程式摘錄比對是否通過。

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 2 輪：上一輪的處理與審查紀錄：通過

讀了紀錄並對照修正清單的 28 項：查核者的必要與 2 項建議的處理在前 3 項，有列出。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | 〈獨立查核〉第 1 次的「必要問題的修正」 | 只列 28 項中的前 12 項，沒有註明省略；清單多是內部項目（「trace 第 3 行 [stale-version] Colab 按鈕 tag」「impact 第 36 行 step=0 0.6941」），標題卻是「必要問題的修正」。 | 已修正：不再截斷，標題改成「修正必要問題時處理的項目」，列出全部項目。 |

### 第 3 輪：上一輪的處理與審查紀錄：通過

以腳本核對紀錄：第 1 次查核（1 項必要、2 項建議）的處理清單完整列出 28 項（上一輪要求不截斷，已做到），第 2 次複查通過；讀者審查 21 項、技術查核 8 項與修正後檢查 4 項，都在〈定稿修正〉34 列處理；批次檢查第 1 次的必要問題在第 2 次通過；來源（arXiv 1312.4400、1409.1556）都在技術查核方法段。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/01-small-cnn.md 第 61–69、144、308 行 | 處理清單用的是內部標籤：「checker must（第 11 行…）」「checker should（…）」「先前審查意見第 3 行 [stale-version]」「[patch-note]」「[restructure]」「[process-history]」「[other]」；第 144 行「這屬於產生程式那個頁面組」；〈上述處理〉摘要「對照 repair:01-small-cnn 的 28 項」。 | 未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 4 輪：上一輪的處理：通過

第 3 輪第 1 項：查核者的必要問題／should 已改成「查核者的必要問題／建議」（第 61 行起）；分類標籤改成中文（第 64 行「版本標籤過期」）；第 144 行「頁面組」改成「部分」；repair: 字樣已不在。處理說明屬實。

## 紀錄重產後的檢查

2026-10-05，另一位 AI 在獨立 rsync 副本（排除 `.git`、`site`、`.venv*`）重新檢查本節的全文、兩張 SVG、完整程式及直接／間接 repo 依賴；只借用既有的 Python 3.12.14／PyTorch 2.9.1+cpu 執行器，cwd 與 `PYTHONPATH` 均為副本。既有審查紀錄全文讀過，以下結果由本次執行、手算與一手來源獨立核對。

在副本執行 `PYTHONPATH=. /workspace/learn_to_yolo/.venv-model/bin/python lesson_cases/01-small-cnn.py` 與同執行器的 `scripts/run_learning_extensions.py --section 01-small-cnn`，兩項 exit 0。預設 stdout 與重產的 `01-small-cnn.json` 逐字相同：三步 loss 0.6942／0.6941／0.6940，預測全 1、錯誤索引 0／2／4／6。40 步重跑的 `report.json` 與紀錄全欄位相同：初始 loss 0.6941587328910828（六位小數 0.694159、四位小數 0.6942）、第 30 點仍為 5.9604641222676946e-8、第 31 點首次精確為 0，最小梯度 L2 1.4261925521200407e-18（約 1.4e-18）、最大 0.7967736124992371（約 0.80）；最末點是第 40 次更新前的 loss，正確率與預測則在第 40 次更新後評估，8 張訓練圖全對，`validation_accuracy` 仍是 `null`。正文更新後的數字、點號、四捨五入與訓練／獨立驗證範圍都相符。

照頁面把模型寬度改成 8，先在舊斷言得到預期的 AssertionError，再按參考答案改為 4330／1695776 後完成三步；逐層參數與 MAC、3.74 倍、記憶體、輸出 shape、感受野 16×16 與邊界 10×10 均重算一致。模型摘錄 AST 與完整程式一致，示意訓練格執行成功；圖片軸序、float64 輸入、long 標籤、標籤越界、permute／reshape、二次 softmax 與零 loss／非零梯度例子都實測符合敘述。另分開測量梯度及參數差：這次 40 個 Adam 步驟的參數差都大於 0；一般非零梯度本身不保證 float32 權重改變，腳本另比較訓練前後全部參數確認整體改變，未把梯度大小當成學習品質或泛化證據。

兩張 SVG 經 XML／座標檢查與 Chromium 截圖實看；三步圖與 default PNG 的前四張位置、GT／pred／紅色錯誤標題一致，40 步圖與本次重畫結果逐 byte 相同。嚴格 Zensical 建置與 `validate_site.py` 均 exit 0，Playwright／Chromium 的桌面與手機版面、兩個表格、六個摺疊區、MathJax、圖片與 `#forty-steps` 正常。notebook 最後程式格及可選 40 步命令也與程式／正文一致；本次未使用 Google Colab 託管 runtime。

本次重新打開 [VGG 論文 v6](https://arxiv.org/pdf/1409.1556v6) 的作者單位、§2.1～2.3／Table 1，及 [Network in Network 論文 v3](https://arxiv.org/pdf/1312.4400v3) §3.2，核對架構、GAP 與感受野說法；另讀 PyTorch 官方固定 tag [v2.9.1 的 Conv2d docstring](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.1/torch/nn/modules/conv.py)、[pooling](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.1/torch/nn/modules/pooling.py)、[loss](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.1/torch/nn/modules/loss.py)、[Adam](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.1/torch/optim/adam.py) 及 [CPU log-softmax 實作](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.1/aten/src/ATen/native/cpu/LogSoftmaxKernelImpl.h)，確認公式與有限精度說明。初學讀者的術語、說明順序、練習步驟與數據／圖的銜接未發現必要問題。

| 嚴重度 | 位置 | 發現與處理狀態 |
|---|---|---|
| 建議 | 40 步梯度段落「它只確認有在更新」 | 可把梯度與實際參數差的檢查分成兩句，避免將非零梯度本身讀成權重已更新的保證。現行上下文已提到權重改變，程式另有整體參數差檢查，不列為必要錯誤；由編輯者決定是否採用。 |

結論：重產後的本頁與兩份紀錄一致，沒有必要問題。完整程式 hash 為 `f15df5625a39f5aeb9bf85eff12a6fab32f23def8d9b89065fcf0bbd2a28ca16`，40 步腳本 hash 為 `b84e3317f95374df87d759e562db9b100ee14628b9cf5bb111238cd4a2e8e6cf`；本次沒有修改原 repo 或 review coverage。

### 修正後的獨立複查

2026-10-05 UTC，由與原審查者不同的 AI，在新建的 `/tmp/lessons-v0.4.0-reviews/fix-01/` 副本複查。已讀 `AGENTS.md`、`docs/preparation/publish.md` 第 7 步、原審查 `/tmp/lessons-v0.4.0-reviews/review-01.md`，並完整閱讀第 01 節正文至自動執行紀錄之前。另將原審查快照與修正後頁面逐行比較：差異只有第 174 行將「它只確認有在更新」改為明確區分梯度與實際參數差的三句。新副本頁面與作者工作目錄頁面逐 byte 相同。

**結論：修正通過；原建議已採納，沒有新增的必要問題或建議。** 新句「L2 長度只表示收到梯度；程式另外比較訓練前後的參數，確認 40 步整體有改變。梯度大小不代表學得好不好。」明確指出兩種檢查各自能證明什麼。整段中的「權重確實改變」有另外的參數差檢查支持，並未宣稱非零梯度能保證每一步的每個 float32 參數改變；「40 步整體」也準確保留程式的檢查範圍。前後仍清楚區分更新前 loss、40 次更新後的預測，以及同批訓練圖和獨立驗證；初學讀者不需先懂浮點數更新細節即可理解新句。

查核方法與實際結果：

- 閱讀 `scripts/run_learning_extensions.py:72`～109 的訓練與紀錄邏輯。程式先複製各參數為 `before`；每步在 `loss.backward()` 後檢查每個參數的梯度存在且全部有限，再算所有梯度平方和的平方根並斷言 `norm > 0`，然後才呼叫 `optimizer.step()`。第 40 步後另算 `changed = sqrt(sum((parameter_after - parameter_before)^2))` 並斷言 `changed > 0`，保存為 `weight_delta_l2`。這是訓練起點和終點的整體差，不是逐步差。原句的含混指代已解除，修正文字符合實作。
- 對照 `artifacts/checks/curriculum/01-small-cnn-learning.json`：`min_gradient_l2 = 1.4261925521200407e-18`、`max_gradient_l2 = 0.7967736124992371`，正文的約 `1.4×10^-18`／`0.80` 四捨五入正確；`weight_delta_l2 = 5.17251308976495 > 0` 支持 40 步整體參數有變。手算參數 `112+148+296+584+18=1158`，亦由建立模型後的 `numel()` 合計確認，故 L2 說明的 1158 維正確。另重算 `[3,4]` 的 L2 為 5，對照正文向量長度公式。
- 自行編寫並實際執行 `artifacts/runs/fix-review-01/check01.py`，最後 exit 0、`passed: true`。float32 參數 1、梯度 `1e-18`、SGD `lr=0.1` 的梯度 L2 實測為 `1.000000045813705e-18 > 0`，更新前後參數仍都是 1、參數差為 0；SGD 和 Adam 的 `lr=0`／梯度 1 也各得到非零梯度而參數差 0。手算理想 SGD 更新量約 `1e-19`，小於 float32 在 1 下方相鄰可表示數間距的一半；`torch.nextafter` 實測間距為 `2^-24 = 5.960464477539063e-8`，因此捨入後保持 1，反例符合預期。
- 重新讀取本環境官方 PyTorch **2.9.1+cpu** 套件中的 `torch/optim/sgd.py`，核對無 momentum、無 weight decay 時的 `_single_tensor_sgd` 使用 `param.add_(grad, alpha=-lr)`。本次執行的是固定版本套件本身；讀取的實作保存為 `torch-v2.9.1-installed-sgd-single-tensor.txt`。這也支持上述手算和小反例，而非沿用原審查者的測試結論。
- 核對學習紀錄列出的全部 12 個 repo 程式／依賴 SHA-256，以及兩張 SVG、notebook 和 40 步 JSON，均與原審查快照和作者目前檔案一致。程式未改，所以依本次修正範圍不重新訓練 40 步；本次的梯度極值與 40 步整體參數差是對現有原始 JSON 和實作獨立核對的結果。另實際執行預設 `lesson_cases/01-small-cnn.py`，exit 0，三步 loss 為 0.6942／0.6941／0.6940，參數 1158、MAC 479248，原斷言通過。
- 在副本實際執行 `zensical build --clean --strict` 與 `scripts/validate_site.py`，均 exit 0。使用 Playwright／`/usr/bin/chromium`，在自己啟動的 HTTP server 8841 上查看修正段落，1365 px 桌面與 390 px 手機寬度都各有兩個正常渲染的 MathJax 公式，文字沒有裁切，沒有 pageerror；兩張段落截圖均已用 `view_image` 實看。測試結束已停止自己的 server。沒有重做全站 MathJax／CSS 審查。

| 原發現 | 修正與複查處理狀態 |
|---|---|
| 建議：40 步段落的「它只確認有在更新」容易被讀成非零梯度即保證 float32 權重更新 | 已採納。新句明示非零梯度與訓練前後參數差各自的作用；查實作、JSON、手算與實際反例後通過，這項建議可結案。 |

新正文 coverage digest（使用既有 `review_coverage.digest()`，排除頁尾自動紀錄並正規化 Colab tag）為 **`d8d1c64d63712cc55709056bb08dc3881e6e6ea5a377f93fa8e05bb7a84e8430`**。本次檢查的快照：

| 檔案／範圍 | SHA-256 |
|---|---|
| 修正後完整 `docs/lessons/01-small-cnn.md` | `b4f2517768bd0883b0ce94a54a5abbb00123819021e5513540b0574ac5968fdb` |
| `lesson_cases/01-small-cnn.py` | `f15df5625a39f5aeb9bf85eff12a6fab32f23def8d9b89065fcf0bbd2a28ca16` |
| `scripts/run_learning_extensions.py` | `b84e3317f95374df87d759e562db9b100ee14628b9cf5bb111238cd4a2e8e6cf` |
| `artifacts/checks/curriculum/01-small-cnn-learning.json` | `624d4ad4237dedf2740064013234cbcc141d63b959bd808f18536599892bde7b` |
| `docs/assets/diagrams/01-small-cnn.svg` | `9d90438ef2d783ee3fec6e793b86cece537b928ae13bc33a8db91c65c8ed3bfb` |
| `docs/assets/diagrams/01-small-cnn-learning.svg` | `24cabd164475a801489559113fed68c510cd24e49eca80ee9cc65933cf9675b6` |

本次探針、完整 coverage digest、三個反例數值、瀏覽器檢查摘要與兩張截圖保存在 `/tmp/lessons-v0.4.0-reviews/fix-01/artifacts/runs/fix-review-01/`。這是針對文字修正的獨立複查，沒有重新執行未改的全部練習／40 步訓練，也沒有重查未改的 VGG／NIN 論文說法；沒有將舊審查的 pass 當成本次結論。全程未修改作者 repo、review coverage、Git 或 remote，未執行 GPU 工作。


## 2026-10-05 clear-tutorial 三輪重審

以上是原審查歷史；不追溯改成首次盲讀。這次由固定基線 `16f6910` 分段開放並保存當時理解，再修改、核技術及檢查銜接，詳見 [本輪方法與限制](clear-tutorial/16f6910/README.md)。

- 第一輪：[foundations當場閱讀原始紀錄](clear-tutorial/16f6910/first-read/foundations.jsonl)，基線來源與圖指紋保留；共享檔案系統不是技術隔離。
- 第二輪：[非作者技術／證據核對](clear-tutorial/16f6910/technical/foundations.md)，實際來源、數字及必要執行範圍見該報告。
- 第三輪：[另一位讀者前文→本節→後文複查](clear-tutorial/16f6910/transitions/foundations.md)，此輪完整頁閱讀非盲讀；受影響段落及圖另有delta核回。
- [原始卡點與具體處理](clear-tutorial/16f6910/decisions.json)保留未新增的選讀建議。原先前提包漏發及08提前brief的限制另列，沒有算成教材錯或冒稱08全程盲讀。

本輪修正後沒有未解的必要問題；這是AI閱讀／技術查核的實際範圍，不是學生學習成效驗收。全站實際Zensical桌面／手機、公式換頁與執行檢查見 [verification.json](clear-tutorial/16f6910/verification.json)，不以SVG檔存在或strict build取代視覺查核。

## 2026-10-05：v0.6.0 有界更新審閱

局部修正40步JSON對照：第1／40點loss、訓練8張正確率、validation null為未評；ViT首讀者及導讀審閱者已就地複查。DINO首讀者沒有複查此處，保持其原report限制。

本次僅重查以上變更，既有正文的歷史審閱保留；沒有把全頁或全書重新標成首次盲讀。[導讀／實際網站審閱](clear-tutorial/vision-v0.6.0/guide-visual-review.md)、[新增支線第三輪及實際前置](clear-tutorial/vision-v0.6.0/transitions-review.md)、[首讀修後複查](clear-tutorial/vision-v0.6.0/vit-recheck.md)記錄各自範圍。全版52本notebook的本機cell執行另見[實跑](clear-tutorial/vision-v0.6.0/local-notebook-runtime.json)；不是Google Colab登入執行。來源與圖／程式指紋更新於[coverage.json](coverage.json)。

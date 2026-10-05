# 術語快速查：卡住時回來看一眼

這是讀課文時的查表，不需先背。每節的數值例子與程式才是主要學習路線；這頁只幫你在卡住時快速想起一個詞的意思，再連回教它的那一節。

怎麼用這頁：

- 表格大致照課程順序分成七組。術語欄寫英文名稱，課文有中文名稱時一併寫出；用瀏覽器的頁內搜尋（Ctrl+F；Mac 是 ⌘+F）找中文或英文都可以。
- 「詳見」欄的數字是節次，和[完整閱讀路線](learning-path.md)上的編號相同；點下去會回到第一次教這個詞的那一節。有多個連結時，第一個以外的是補充的節：教得更詳細、教它另一種用法，或教同一列的另一個詞。
- 術語後面標 ※ 的，在書裡還有別的意思，見頁末[〈同名不同義〉](#homonyms)。

??? note "常用符號"

    同一個字母在不同節可能代表不同東西，以各節的說明為準。下面列出常見的意思，以及容易混淆的例外。

    - B：一個 batch 的資料筆數，通常是圖片張數。
    - C：channel 數，或類別數；看該節的說明。11.4 節的 C 不是個數，而是同時框住預測框 P 與真值框 G 的最小包圍框；同一節的小寫 c 是 C 的對角線長。
    - H、W：高、寬。3.1 節的 H(x) 則是一個函數的名字（理想的轉換），不是高。
    - N：一張圖的框（物件）數；12.4 節 `[N,K]` 裡的 N 是要監督的邊（距離）數；第 15 章是 token 數。NCHW 的 N 則是 batch 張數，就是 B。
    - S：grid 每邊的格數，例如 4×4 格時 S＝4。15.2 節摺疊區〈如果改成固定每區的 token 數呢？〉裡的 S 則是每區固定的 token 數。
    - A：第 9 章是每格的 anchor 數（也就是 slot 數）；15.2 節是 area（區域）數。
    - K：第 0 章是線性層的輸出數；12.4、16.1 節是 DFL 每條邊的 bin 數。attention 的 Q、K、V 裡，K 是 key。
    - P：12.1、12.3、13.1 節是候選（點）數；6.2 節表格裡的 P 是 precision，3.2 節的 P 是 projection shortcut。7.3 節 loss 式子 \(\sum_{i\in P}\) 裡的 P 是正格的集合；11.4 節的 P 則是一個預測框，不是個數。
    - G：12.3、13.1 節是 GT（真值物件）數；11.4 節的 G 則是一個真值框，不是個數。
    - M：6.2 節是一張圖的預測框數；15.2 節是 attention head 數。
    - D：第 0 章是每筆輸入的特徵數；15.2 節是每個 attention head 的 channel 數；第 19 章是這一幀的偵測框數。
    - T：第 19 章保留中的 track 數。
    - d：15.1 節是 query、key 向量的元素個數，也就是 attention 除以 \(\sqrt{d}\) 的那個 d；12.4 節的 d 是以格為單位的距離；12.1 節 Smooth L1 式子裡的 d 則是某一邊的誤差（預測−target），可以是負數，不是距離本身。
    - e：sigmoid、softmax 裡 \(e^{z}\) 的 e 是自然對數的底，約 2.718；16.3 節的 e 則是第幾輪（epoch），E 是總輪數；15.1 節的 E[·] 是平均（期望值）。
    - ln、log：以 e 為底的自然對數，例如 ln 2≈0.693；程式裡的 `log` 也以 e 為底，不是計算機上以 10 為底的 log 鍵。
    - σ：sigmoid，不是統計的標準差。
    - η：學習率，程式裡寫 lr。
    - t：公式裡多半代表 target，例如 BCE 式子裡的 t；但 tx、ty、tw、th 是模型四個框輸出的名字，不是 target；12.1、16.1、16.2 節 l、t、r、b 裡的 t 是 top 的縮寫，指候選點到框上邊的距離（見 ltrb）。
    - ⌊x⌋：不超過 x 的最大整數（floor），就是高中的高斯符號 [x]。

## 張量與訓練

| 術語 | 先用一句話記住 | 留意的地方 | 詳見 |
| --- | --- | --- | --- |
| tensor（張量）、shape（形狀） | tensor：裝數字的容器，可以有 0 個或多個軸；scalar（純量）沒有軸，shape 是 `[]`。shape：每個軸的長度 | 程式裡的 dim 就是軸。調換軸的順序要用 permute，例如把一般圖片的 HWC（高、寬、顏色）換成 CHW 是 `permute(2,0,1)`；reshape 只改形狀，會把顏色和位置混在一起 | [0](lessons/00-warmup.md)、[1](lessons/01-small-cnn.md) |
| batch（一批） | 同一次 forward（前向計算：把輸入算成輸出）一起處理的一組資料，例如一批圖片 | 同一批裡，每張圖的物件數仍可不同 | [0](lessons/00-warmup.md) |
| channel（通道） | 特徵圖的一層：每個位置在這層有一個值；channel 數就是同一位置有幾個特徵值（特徵維度） | RGB 輸入有 3 個 channel；中間層的 channel 不一定對應顏色 | [1](lessons/01-small-cnn.md) |
| gradient（梯度） | 某個數（參數或輸入）稍微改變時，loss 的變化率 | backward（反向傳播：從 loss 往回算）算出梯度，`optimizer.step()` 才更新參數 | [0](lessons/00-warmup.md) |
| optimizer（優化器）、SGD、Adam、learning rate（學習率，lr） | optimizer：拿梯度去修改參數的物件。SGD（stochastic gradient descent，隨機梯度下降）每步做 w←w−lr×梯度；Adam 依每個參數過去梯度的大小，自動調整每一步走多遠 | lr 太大會亂跳，甚至變成 NaN（Not a Number，算壞了的非數字）；太小幾乎不動。Adam 每步的改變量不等於 lr×梯度 | [0](lessons/00-warmup.md)、[1](lessons/01-small-cnn.md)、[2](lessons/02-diagnostics.md)、[7.4](lessons/07-training.md) |
| step（一步）、epoch（一輪） | step：參數更新一次，也就是呼叫一次 `optimizer.step()`，通常用一批資料。epoch：整份訓練資料都用過一次 | 資料只有一批時，一輪就是一步；例如 16.3 節只有 4 筆資料，30 輪就是 30 次更新 | [0](lessons/00-warmup.md)、[11.3](lessons/11-augmentation.md)、[16.3](lessons/16-training.md) |
| seed（亂數種子） | 決定亂數從哪裡開始；固定 seed，每次重跑的隨機初始權重（以及程式生成的資料）都一樣，結果才能重現 | 只跑一個 seed，分不出差異來自設定還是運氣；建模型的順序不同時，同一個 seed 也可能得到不同的權重（3.3 節）。換一台電腦或改了執行緒數，初始權重與資料仍相同，但加總順序可能不同，訓練後的小數就可能不同（7.4 節） | [1](lessons/01-small-cnn.md)、[3.3](lessons/03-comparison.md)、[7.4](lessons/07-training.md) |
| checkpoint（存檔） | 訓練時存下的檔案：模型的權重，常一併存設定與 optimizer 狀態 | 可以載回來推論或接著訓練；模型的 shape 改了（例如類別數不同），舊的 checkpoint 就載不進去 | [7.4](lessons/07-training.md)、[8.1](lessons/08-own-images.md) |
| overfit（過擬合）、泛化 | overfit：把訓練資料學到幾乎背起來，換新資料卻可能變差。泛化：對沒參與參數更新的資料也做得好 | 第 2 章會刻意先讓模型對少量資料 overfit，確認程式和模型學得動；7.4 節把它列為三步檢查之後的下一關，8.2 節在偵測模型上實際做這項檢查（結果見該節）。這項檢查通過，仍不保證泛化 | [2](lessons/02-diagnostics.md)、[7.4](lessons/07-training.md)、[8.2](lessons/08-own-data.md) |
| train／validation／test（訓練集／驗證集／測試集） | train 用來更新參數；validation 用來監測、挑設定；test 留到設定都定案後，最後評一次 | 反覆拿 test 挑設定，test 就間接參與了選擇，不再是獨立的 test，分數也會偏樂觀 | [2](lessons/02-diagnostics.md) |
| held-out（獨立資料） | 刻意保留、不參與參數更新的資料；validation 和 test 都屬於這類 | 同一個來源（例如同一段影片）的相似圖片要整組放在同一邊，否則評估分數會虛高 | [2](lessons/02-diagnostics.md)、[7.6](lessons/07-heldout.md) |
| 超參數（hyperparameter） | 學習率、步數、模型寬度這類由人事先決定、不靠梯度學出的設定 | 「調參」調的是超參數，不是模型權重；挑超參數用 validation，不用 test | [2](lessons/02-diagnostics.md) |
| 斷言（assert） | `assert 條件`：條件不成立就報錯停下（`AssertionError`），用來自動核對答案 | 做練習改了數字後出現 AssertionError，多半是斷言還在核對原題的答案（第 0 章練習 1），要把斷言裡的數字換成新的答案 | [0](lessons/00-warmup.md) |

## 網路結構

| 術語 | 先用一句話記住 | 留意的地方 | 詳見 |
| --- | --- | --- | --- |
| feature map（特徵圖） | 卷積層輸出的 `[C,H,W]` 數值；每個 channel 是一張 H×W 的圖，每個位置有 C 個特徵值 | 特徵圖的一格不是一個畫素（pixel）：64×64 的輸入算到 8×8 的特徵圖時，相鄰兩格在輸入圖上相隔 8 個畫素（見 stride） | [1](lessons/01-small-cnn.md) |
| stride（步幅） | ① 卷積或 pooling（池化）的 stride：視窗每次移動幾格，stride 2 讓長寬約減半。② 特徵圖的 stride（第 1 章算感受野時叫它「間距」）：相鄰特徵位置在模型輸入圖上相隔幾個 pixel，等於前面各層 stride 相乘；例如 64×64 經三次 stride 2 得到 8×8，stride 是 8 | pixel 距離除以 stride，才是特徵圖上的格數；用過 letterbox 時，要先換成輸入畫布的座標 | [1](lessons/01-small-cnn.md)、[10](lessons/10-multiscale.md) |
| backbone（主幹） | 模型前段負責從圖片提取特徵、輸出特徵圖的那一串層 | 只輸出特徵圖；框要用 xyxy 還是 ltrb 這類寫法，由 head 與 decode 決定 | [1](lessons/01-small-cnn.md) |
| neck | 夾在 backbone 與 head 之間，整理或融合特徵 | 小模型可省略獨立的 neck | [5](lessons/05-assignment.md)、[11.2](lessons/11-fusion.md) |
| head ※ | 把特徵轉成任務輸出（例如類別分數、框）的末端 | 最後每個 channel 代表什麼，必須和 target 對得上 | [1](lessons/01-small-cnn.md) |
| residual block（殘差區塊）、shortcut（捷徑） | 在主分支旁接一條 shortcut，把輸入加回主分支的輸出：\(y=x+F(x)\) | 相加前兩條路的 shape 必須完全相同；shape 不同時，先用 projection（可學的 1×1 卷積）對齊 | [3.1](lessons/03-identity.md)、[3.2](lessons/03-projection.md) |
| concat（串接）與 add（相加） | concat 沿 channel 把兩個 tensor 接起來（4＋4 變 8 個 channel）；add 逐個位置、逐個 channel 相加，channel 數不變 | 沿 channel concat 時，其他軸（B、H、W）要一樣長，否則 `torch.cat` 會報錯；add 要整個 shape 相同，但對不上的軸若都有一邊長度是 1，PyTorch 會 broadcast（廣播：把長度 1 的軸自動延伸成另一邊的長度），不報錯，所以要用 assert 核對兩邊的完整 shape（3.1 節練習 2）。shortcut 用的是 add | [3.1](lessons/03-identity.md)、[11.1](lessons/11-csp.md)、[11.2](lessons/11-fusion.md) |

## 框與座標

| 術語 | 先用一句話記住 | 留意的地方 | 詳見 |
| --- | --- | --- | --- |
| 框（box／bbox） | 物件的矩形框，常用四個數表示 | 原點在左上，x 向右、y 向下；本書用半開區間，寬＝x2−x1，所以不加 1 | [4.1](lessons/04-localization.md) |
| xyxy、cxcywh | xyxy：左上角與右下角 `[x1,y1,x2,y2]`。cxcywh：中心與寬高 `[cx,cy,w,h]`，cx＝(x1＋x2)÷2、w＝x2−x1 | 先寫清單位：原圖 pixel、輸入畫布 pixel（letterbox 後）、正規化（畫布座標除以畫布邊長，本書多半是 64；這是相對畫布的比例，不會自動對回原圖）。第 7 章的框 target 另有規定：xy 是格內偏移，wh 除以整張圖 | [4.1](lessons/04-localization.md)、[4.2](lessons/04-coordinates.md) |
| IoU（Intersection over Union，交併比） | 兩框的交集面積除以聯集面積；聯集＝兩框面積相加，再減掉交集 | 範圍 0 到 1：不重疊時是 0，完全相同時是 1 | [4.1](lessons/04-localization.md) |
| letterbox | 等比例縮放，再在空出來的地方補邊，湊成固定大小的輸入畫布 | 要存下實際縮放比例、補邊量與原圖尺寸才能還原；還原時先扣補邊、再除以比例 | [4.2](lessons/04-coordinates.md) |
| 框參數化、encode（編碼） | 框參數化：決定用哪幾個數、什麼公式描述一個框，例如第 7 章的格內偏移與整圖比例、第 9 章相對 anchor 的倍率、第 12 章的 ltrb。encode：把真值框換成模型該輸出的那幾個數；decode 反過來 | encode 和 decode 要用同一套公式與單位（例如 anchor 用 pixel 還是格），否則還原會錯 | [9.1](lessons/09-anchors.md) |

## 訓練目標與 loss

| 術語 | 先用一句話記住 | 留意的地方 | 詳見 |
| --- | --- | --- | --- |
| logits | 模型最後一層直接輸出、還沒經過 sigmoid、softmax、softplus（把數變成正數）等轉換的原始實數。類別與 objectness 的 logits 轉完是機率；框的 logits 轉完是描述框的數，例如座標比例、相對 anchor 的倍率或距離，看該節的框寫法 | `cross_entropy`、`BCEWithLogitsLoss` 都要接收 logits，內部自己轉換。sigmoid 的反函數也叫 logit：\(\mathrm{logit}(p)=\ln\bigl(p/(1-p)\bigr)\) | [1](lessons/01-small-cnn.md)、[7.3](lessons/07-loss.md) |
| sigmoid（σ） | \(\sigma(z)=1/(1+e^{-z})\)，把任意實數壓到 0 與 1 之間（數學上碰不到兩端；用 float32 計算時，極端的 logit 會捨入成 0 或 1，見 7.5 節），例如 \(\sigma(0)=0.5\) | 用在類別或 objectness 時，每個輸出各自是一個「是／否」機率，彼此不必加總為 1，可以同時接近 1；要在多類中只選一類，用 softmax。用在框時，壓出來的是座標比例，不是機率 | [4.1](lessons/04-localization.md) |
| softmax | 把一組分數換成加總為 1 的機率：第 k 類是 \(e^{z_k}/\sum_j e^{z_j}\) | 要沿正確的軸做：類別軸、DFL 的 bin 軸、attention 的 key 軸 | [1](lessons/01-small-cnn.md) |
| loss（損失） | 用一個能用 backward 算出梯度的數值，表示目前的預測離目標多遠；越小越好 | 不是 accuracy（正確率：答對的比例），也不直接等於 AP | [0](lessons/00-warmup.md) |
| MSE（mean squared error，均方誤差） | 每個誤差先平方，再取平均；誤差越大，罰得越重 | 第 7 章的框 loss 用正格座標的 MSE。它只比四個數各差多少，沒有直接衡量兩框重疊得好不好（見 IoU 類 loss） | [3.1](lessons/03-identity.md)、[4.1](lessons/04-localization.md)、[7.3](lessons/07-loss.md) |
| CE（cross entropy，交叉熵） | 多類分類的 loss：\(-\ln p\)，p 是 softmax 後正確類別的機率；p＝0.5 時約 0.693 | PyTorch 的 `cross_entropy` 內部先做 softmax，所以要傳 logits。7.3 節推出：CE 對類別 logits 的梯度是 softmax 機率減 one-hot（正確類別記 1、其他類別記 0 的向量）；先自己 softmax 再傳，等於做兩次，loss 可能看不出錯，這個梯度卻不對 | [1](lessons/01-small-cnn.md)、[7.3](lessons/07-loss.md) |
| BCE（binary cross entropy，二元交叉熵） | 是非題的 loss：\(-[t\ln p+(1-t)\ln(1-p)]\)，其中 \(p=\sigma(z)\)，目標 t 通常是 0 或 1 | `BCEWithLogitsLoss`（函式版是 `binary_cross_entropy_with_logits`）內部先做 sigmoid，所以也要傳 logits。t 也可以是 0～1 之間的小數，例如 12.2、12.3 節提到，YOLOv8 正樣本的類別 target 是依品質給的分數；這時式子照用，對 z 的梯度也照樣是 7.3 節推出的 \(p-t\) | [5](lessons/05-assignment.md)、[7.3](lessons/07-loss.md) |
| GT（ground truth，真值）、target（訓練目標）、監督 | GT：人工標的正確框與類別，也叫標註（annotation）或真值框。target：把 GT 整理成和模型輸出一一對得上、loss 能直接比對的答案。這種告訴模型每個輸出位置正確答案的資料，叫監督（supervision） | target 由 GT 轉換而來，格式不一定相同 | [4.1](lessons/04-localization.md)、[5](lessons/05-assignment.md)、[7.2](lessons/07-targets.md) |
| grid（格子、網格）、cell（格）、正格／負格 | 把輸入圖分成 S×S 格；第 5、7 章由物件中心落入的格當正格，其餘是負格，要學背景 | 中心剛好落在格線上時，floor（無條件捨去）會交給右邊或下面那格 | [5](lessons/05-assignment.md) |
| slot（槽） | 一個能輸出一組框與分數的位置；第 5、7 章每格 1 個，第 9 章每格 2 個（各配一個 anchor） | slot 數決定同一格最多放得下幾個物件；每個 slot 都輸出全部類別的分數，不分類別 | [5](lessons/05-assignment.md) |
| candidate（候選） | 模型可以輸出一個預測的位置或 slot；解碼後得到的預測框也常叫候選（候選框） | 候選數不是實際物件數 | [5](lessons/05-assignment.md) |
| objectness（物件分數） | 這個候選有沒有分到物件的分數（它的 logit 經 sigmoid） | 本書 grid 的目標是 0／1：正樣本 1、負樣本 0；第 9 章的 ignore 槽不算這項 loss，不是學 0。第 12、13、16 章的 head 沒有 objectness，改由類別 sigmoid 一起表達 | [5](lessons/05-assignment.md) |
| assignment（責任分配）※ | 訓練時決定哪個候選負責哪個 GT | 正樣本學框與類別，負樣本通常提供背景訊號。它和推論的 NMS、評估的 matching 是三件不同的事 | [5](lessons/05-assignment.md) |
| positive（正樣本）※、negative（負樣本）、ignore ※ | positive：分到物件、要學框與類別的候選。negative：要學背景的候選（grid 模型是 objectness 目標 0；第 12、13、16 章的 head 沒有 objectness，改成每個類別分數的目標都是 0）。ignore：某項 loss 完全不算的候選 | 空圖的格子都是負樣本、要學背景，不是 ignore；各版本的規則不同，本書到第 9 章才用到 ignore | [5](lessons/05-assignment.md)、[9.1](lessons/09-anchors.md) |
| mask（遮罩）※ | True／False 陣列，挑出要計入某項 loss 或要保留的元素，例如要算框與類別 loss 的正格 | 整批沒有任何正樣本時（例如全是空圖），對 0 個數取平均是 0÷0，得到 NaN；要改成回傳一個仍連著計算圖（PyTorch 記下的運算過程，backward 沿著它往回算梯度）的 0，不能刪掉空圖 | [5](lessons/05-assignment.md)、[7.3](lessons/07-loss.md) |

## 推論與評估

| 術語 | 先用一句話記住 | 留意的地方 | 詳見 |
| --- | --- | --- | --- |
| 推論（inference） | 拿訓練好的模型對圖片做預測，不更新參數 | 通常先 `model.eval()`，再放進 `torch.no_grad()`；兩個開關的作用不同（第 0 章） | [0](lessons/00-warmup.md)、[7.5](lessons/07-inference.md) |
| decode（解碼）※ | 把 head 輸出的數字換回畫素框與分數 | 不是模型再看一次圖，也不是評估 | [6.1](lessons/06-decode-nms.md) |
| score（分數） | 候選用來排序、篩選、做 NMS 的數；不是 precision，也不一定是校準過的機率（校準過：例如 score 約 0.8 的框，真的約有八成框對了物件） | 本書的 grid 模型（第 6～8、10 章與第 17～20 章）是 \(\sigma(\text{obj})\)×最大的類別 softmax 機率（例：0.9×0.8＝0.72）；第 12、13、16 章的 head 沒有 objectness，直接用類別 sigmoid | [6.1](lessons/06-decode-nms.md) |
| 門檻（threshold） | score 門檻拿候選自己的 score 去比，分兩種：顯示門檻決定畫出哪些框（本書常用 0.25）；候選截斷門檻在評估前先刪掉極低分的候選（例如 0.05、0.01；8.2 節用 0.1）。IoU 門檻比兩框的重疊，也分兩種：NMS 的 IoU 門檻比候選與候選；配對 IoU 門檻比預測與 GT，決定 TP 或 FP | 這四種門檻用途不同，要各自設定；兩種 IoU 門檻比的對象不同，都設 0.5 也不是同一個設定。訓練與追蹤另有自己的門檻，例如第 9 章 ignore 規則的尺寸 IoU 0.2、第 19 章配對 track 的 IoU 0.1 | [6.1](lessons/06-decode-nms.md) |
| NMS（Non-Maximum Suppression，非極大值抑制） | 同類候選依 score 排序，保留最高分的框，刪掉和它 IoU 超過門檻的，再對剩下的重複 | 比的是預測與預測，不看 GT；可能誤刪靠得很近的真物件 | [6.1](lessons/06-decode-nms.md) |
| matching（評估配對）※ | 評估時把預測框和 GT 配對，判定 TP 或 FP | 同圖、同類才比；每個 GT 最多成功配對一次；score 高的預測先挑 | [6.2](lessons/06-evaluation.md) |
| TP（正確偵測）、FP（誤報）、FN（漏檢） | TP（true positive，真陽性）：和還沒被配走的同類 GT 配對成功的預測。FP（false positive，假陽性）：沒配對成功的預測，包含背景框、IoU 不足的框、重複框。FN（false negative，假陰性）：沒被任何預測配到的 GT | 高分錯框仍是 FP。這裡的 positive 指模型畫了框，和訓練的正樣本無關；偵測不計 TN（真陰性），因為背景位置數不完 | [6.2](lessons/06-evaluation.md) |
| precision（精確率） | TP÷(TP+FP)：交給評估的預測框裡，對的占幾成 | 不是 score：score 是模型自評的排序分數，precision 要拿 GT 配對之後才算得出來 | [6.2](lessons/06-evaluation.md) |
| recall（召回率） | TP÷(TP+FN)：所有 GT 裡被找到的比例 | 提高顯示門檻可能讓 recall 下降：這裡指畫出的這組框的 recall，PR 曲線就是在模擬不同高度的顯示門檻。算 AP 時評估另用很低的候選截斷門檻，只改顯示門檻不會改變評估分數（第 17 章） | [6.2](lessons/06-evaluation.md)、[17](lessons/17-capstone.md) |
| PR 曲線（precision–recall 曲線） | 依 score 由高到低逐筆納入預測，每納入一筆記一個 (recall, precision) 點；橫軸 recall，縱軸 precision | 只報其中一個點，等於替模型挑好了門檻；原始的點會鋸齒狀上下跳 | [6.2](lessons/06-evaluation.md) |
| AP（average precision，平均精確率） | 每個 recall 位置，改取 recall 相同或更高的 PR 點裡最高的 precision，得到只降不升的階梯，叫 precision envelope（包絡）；AP 是包絡底下的面積（all-points 插值），也就是包絡的平均高度。例：6.2 節的包絡在 recall 0～2/3 高 0.5、2/3～1 高 0，所以 AP＝2/3×0.5＝1/3 | 不是把點連成折線算梯形面積，也不是單點 precision×recall | [6.2](lessons/06-evaluation.md) |
| AP50、mAP（mean AP） | AP50：配對 IoU 門檻 0.5 時的 AP。mAP：各類別 AP 的平均 | 本書的 mAP 只平均有 GT 的類別。AP50 只用一個 IoU 門檻，不是 COCO（常用的公開偵測資料集）在 IoU 0.50～0.95 共 10 個門檻上平均、也對各類別平均的 AP | [6.2](lessons/06-evaluation.md) |

## YOLO 演化機制

| 術語 | 先用一句話記住 | 留意的地方 | 詳見 |
| --- | --- | --- | --- |
| IoU 類 loss（GIoU、DIoU、CIoU） | 以 1−IoU 為主的框 loss，再加一項讓不重疊的框也知道往哪裡移：GIoU 加最小包圍框裡的空白比例，DIoU 改加兩個中心的距離，CIoU 再加寬高比的差異 | 兩框不重疊時 IoU 一直是 0，單用 1−IoU 得不到移動的方向；附加項就是補這一點 | [11.4](lessons/11-iou-loss.md) |
| decoupled head | 把 head 分成框、類別兩條分支，各自算輸出；前面的特徵仍然共用 | 第 7 章的 head 是 coupled：一層 1×1 卷積同時輸出框、objectness 與類別 | [12.2](lessons/12-decoupled-head.md) |
| anchor（錨框、先驗）※ | 事先給定、訓練中不更新的參考寬高；模型只學相對它要放大或縮小多少 | slot 和類別是獨立的軸：anchor 只管尺寸，不綁某一類 | [9.1](lessons/09-anchors.md) |
| anchor-free（不用 anchor） | 不使用 anchor 這種預設寬高模板；第 12 章讓每個候選點直接預測到框四邊的距離（ltrb） | 仍需要候選點，也仍需要 assignment 決定哪個候選負責哪個物件。第 7 章的 grid MiniYOLO 同樣沒有 anchor 尺寸，差別在框的寫法（12.1 節） | [12.1](lessons/12-anchor-free.md) |
| ltrb | 候選點到框左（left）、上（top）、右（right）、下（bottom）四邊的距離；本書多半以特徵格為單位，也就是畫素距離除以 stride（16.3 節的例子直接用畫素） | 和 xyxy 一樣是四個數，意思卻不同。例：點 (28,28)、stride 8 時，xyxy `[12,16,40,36]` 寫成 ltrb 是 `[2,1.5,1.5,1]`。點在框外時，至少有一邊是負的：12.1 節的 softplus 只輸出正數，表示不了；16.1 節的 YOLO26 輸出可正可負 | [12.1](lessons/12-anchor-free.md) |
| top-k | 依分數（或品質）由大到小排，只留前 k 名 | 只依每個候選自己的分數或品質取前 k 名，不比較候選框彼此的重疊，本身不保證每個物件只留一個框，不能叫 NMS | [12.3](lessons/12-assignment.md)、[13.2](lessons/13-nms-free.md) |
| DFL（Distribution Focal Loss）、bin | 把一條邊的距離拆給相鄰兩個整數刻度（bin）當目標：1.25 格→bin1 權重 0.75、bin2 權重 0.25，loss＝\(-0.75\ln p_1-0.25\ln p_2\)；解碼時取期望值 \(\sum_k k\,p_k\) | bin 是距離刻度，不是物件類別。K 個 bin 只能表示 0 到 K−1 格；Ultralytics 程式的 reg_max 就是 K，YOLO26 設 reg_max＝1，等於不用 DFL | [12.4](lessons/12-dfl.md)、[16.1](lessons/16-dfl-free.md) |
| one-to-many（一對多）、one-to-one（一對一） | 一對多：一個 GT 教好幾個候選，訓練訊號多，但推論時同一物件容易有多個高分框。一對一：每個 GT 只教一個候選，一個候選也最多負責一個 GT | YOLOv10 的 dual assignment（雙重分配）訓練時兩種 head 都接，推論只留一對一 head、不跑 NMS，這叫 NMS-free | [13.1](lessons/13-dual-assignment.md) |
| attention（注意力）、token、Q／K／V | 每個位置是一個 token（該位置各 channel 的值排成的向量）。拿 query（要找什麼）和每個 key（被比對的標籤）做內積、除以 \(\sqrt{d}\)，沿 key 軸 softmax 成權重，再把所有位置（含自己）的 value（被讀走的內容）加權相加 | 例：15.1 節左上位置的權重是 [0.3349,0.1651,0.3349,0.1651]，第一項就是讀自己。full attention 每張圖、每個 attention head 有 N² 個權重；Area Attention 分成 A 區後降為 N²/A | [15.1](lessons/15-attention-bridge.md)、[15.2](lessons/15-area-attention.md) |

## 影片、追蹤與部署

| 術語 | 先用一句話記住 | 留意的地方 | 詳見 |
| --- | --- | --- | --- |
| FPS（frames per second，每秒幀數） | 每秒幾幀。來源每秒產生幾幀叫來源幀率，例如 20 FPS | 來源幀率、程式每秒處理完幾幀（處理速率）、延遲，是三個不同的量 | [18](lessons/18-video.md) |
| latency（延遲）、throughput（吞吐量） | latency：一張圖或一幀從進來到出結果經過的時間；影片還要算排隊等待。throughput：每秒處理幾張或幾幀，例如 batch 2 時＝2÷batch latency（秒） | 湊成 batch 能提高吞吐量，但單張要等湊滿、等整批算完，延遲不一定變短 | [18](lessons/18-video.md)、[20](lessons/20-deployment.md) |
| tracking（追蹤） | 替每個物件發一個編號（track ID），跨幀（frame：影片裡的一張畫面）讓同一個物件一直拿同一個編號 | 偵測框正確，不保證 ID 不交換 | [19](lessons/19-tracking.md) |
| FP32、FP16 | 32 位元、16 位元浮點數。FP32 就是一般的 float32，約 7 位有效數字；FP16 比較省記憶體，在支援的 GPU 上常比較快，但只有約 3 位有效數字 | 換成 FP16 後要重新評估：分數靠近門檻時，可能改變框的去留 | [20](lessons/20-deployment.md) |

## 同名不同義：同一個詞在書裡的不同意思 { #homonyms }

表格裡標 ※ 的詞都整理在這裡；表格沒有單獨列出的 activation、padding 也一併收進來。看到這些詞時，先確認是哪一節、哪一種意思。

- **head**：偵測模型的 head 是把特徵轉成預測的末端，13.1、16.2 節的 many／one（一對多、一對一）head 也是兩個偵測 head。第 15 章 attention 的「單 head／多 head」是另一回事：把 channel 分成幾組，各算一份 attention 權重，和偵測 head 無關。
- **assignment**：第 5 章起（例如 12.3、13.1 節）多半指訓練時「哪個候選負責哪個 GT」。9.2 節 k-means 的 cluster assignment（歸群結果）是每個框的寬高分到第幾群；assignment problem（指派問題）是「一對一配對、讓總分最好」這類數學問題：13.1 節範例在訓練時的一對一配對就是在解它，第 19 章把偵測框配給 track 也寫成這類問題。9.2 節的歸群結果和第 19 章的配對，都不是訓練用的 assignment。
- **matching**：6.2 節的 matching 是評估時拿預測框和 GT 配對。第 19 章 tracking 的 association（關聯）是把這一幀的偵測框接到既有的 track 上，沒有 GT 參與，IoU 門檻也不同（本書用 0.1）。
- **anchor**：第 9 章的 anchor 是預設的框寬高。12.1 節提到 Ultralytics 程式裡的 `anchor_points`，只是候選點（參考點），沒有預設寬高。
- **mask**：訓練時挑出要算 loss 的格（例如正格）；11.3 節的 keep 挑出資料增強後要保留的框；12.3 節的資格遮罩挑出合格的候選。都是用來挑選的 True／False（或 1／0）表，挑的東西不同。
- **positive**：訓練時的正樣本（positive）是分到物件的候選；評估時 TP、FP 裡的 positive 指模型畫了框。兩者無關。
- **ignore**：第 5、9 章訓練時的 ignore 是「這項 loss 不算」；6.2 節提到的 COCO ignore 是「評估時不計分」的 GT 或預測。
- **decode**：第 6 章起的 decode 是把模型輸出換回框與分數；第 18 章的影片解碼，是把壓縮的影片檔還原成一張張畫素陣列。
- **activation**：第 3 章，以及第 14 章介紹官方 Conv 層時，指激勵函數（activation function，例如 ReLU）；11.2、12.2 節與第 14 章談記憶體時，指各層算出、要暫存在記憶體裡的中間張量。
- **padding**：第 1 章卷積的 padding（填充）是在輸入四周補值，例如補一圈 0；4.2 節 letterbox 的補邊（padding）是為了湊滿固定大小的畫布。

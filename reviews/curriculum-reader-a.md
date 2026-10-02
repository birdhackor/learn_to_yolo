# Curriculum reader A：第一輪獨立初讀

審查日期：2026-10-02。讀者假設：會基本 Python、PyTorch、NN／CNN，不熟大學數學。依指定順序閱讀八節的完整正文、完整 lesson case、notebook 四個 cell 與保存輸出、CPU curriculum JSON；有 learning 補充的三節也讀完整 JSON 與 `scripts/run_learning_extensions.py`。未讀舊 reviews、作者歷史或 Git 歷史來代替閱讀。

這是 `lessons-v0.2.0` 發布前教材的初讀；該 ref 尚未發布，因此沒有把目前 Colab 連結不能取得 ref 當作缺陷，也沒有宣稱已在全新 Colab 成功執行。八個 notebook 的環境 cell 完全一致，固定 PyTorch 2.9.1 與 CPU 範例；實驗 cell 逐字等於相對應 lesson case，保存 stdout 逐字等於 curriculum JSON，八個 JSON 的 case SHA256 也符合目前 case。這些核對不等於只看 exit code。

SVG 已讀原始文字／幾何並用本機 Chromium 渲染目視；渲染輸出只在 `/tmp/reader-a-*.png`。另目視三張已保存的 `artifacts/01-small-cnn.png`、`artifacts/04-localization.png`、`artifacts/04-coordinates.png`。只用 CPU 做兩個練習的獨立算式／模型檢查，未重跑會覆寫教材圖或 JSON 的 extension，未啟動 GPU、未對外傳訊，倉庫內只新增此報告。

主要需要修訂的是 02 與 04-coordinates 的練習操作提示。另有 01 程式片段的上下文、首次 L2 用詞、03-identity 訓練目標、04-localization accuracy 名稱四個清晰度問題。未發現正文主實驗、保存 stdout 或 SVG 的數值矛盾。

## 00-warmup

實際閱讀：`docs/lessons/00-warmup.md` 全文、`lesson_cases/00-warmup.py` 全文、`notebooks/00-warmup.ipynb` 四格及保存輸出、`artifacts/checks/curriculum/00-warmup.json` 全文。此節沒有引用 SVG，也沒有同名 learning 補充。

首次閱讀能接上「先算方向，再修改旋鈕」。正文不只是給導數答案，還從平方差展開說明 2e 的來源，對不熟微積分者有效。核對：輸入與權重 shape 都是 `[1,1]`，w=1、x=2、target=4 得 prediction=2、loss=4、gradient=-8；SGD lr=.1 得 w=1.8、prediction=3.6、new loss=.16，與 case／notebook／JSON 完全一致。`eval()` 沒有停止 autograd、replacement optimizer 指向 replacement weight 的檢查與解說也一致。

練習 lr=.25 的答案 w=3、prediction=6、loss=4 正確；正文已要求改相應 assertion，因此能操作。沒有必須修訂的問題。權重轉置在 scalar 暖身中較抽象，但已有 B／D／K 與內積的文字定義，未形成主流程阻礙。

## 01-small-cnn

實際閱讀：`docs/lessons/01-small-cnn.md` 全文、`lesson_cases/01-small-cnn.py` 全文、`notebooks/01-small-cnn.ipynb` 四格與 stdout、`artifacts/checks/curriculum/01-small-cnn.json`、`01-small-cnn-learning.json` 全文；`docs/assets/diagrams/01-small-cnn.svg`、`01-small-cnn-learning.svg` 原始內容與渲染圖；`artifacts/01-small-cnn.png`；extension 的 CNN 分支與共用訓練／作圖程式。

能理解 channels、RGB 跨 channel 乘加、pooling、GAP 與分類 head。shape 表與程式的六個輸出對上。獨立加總確認參數 112+148+296+584+18=1158；MAC 110592+147456+73728+147456+16=479248。float32 `[8,4,32,32]` 的數值大小為 131072 bytes，即 128 KiB。感受野 3→5→6→10→14→16 的每層增量符合 stride。width=8 練習的參數 4330、MAC 1695776 正確，且提示更新固定答案。

圖板前四張的矩形位置、顏色與 case 的 `left/top` 一致；8 張實際圖板都預測 1，錯誤是 0／2／4／6。三步 loss .6941／.6940／.6940 對上 JSON。40 步補充明确說明改用 Adam .01，首步 .6941443086、最後更新前 0、train accuracy 1.0、validation null 與 JSON 相符；曲線由 .694 降至 0，與數據對上。保存的最小整體梯度 L2 為約 1.15448e-17，雖很小仍大於 0，沒有把 CE 顯示 0 誤說成新資料都完美。

問題 A1（清晰度，`docs/lessons/01-small-cnn.md:38`）：短程式同時使用外部 `optimizer`／`images`／`labels` 與 `self.features`／`self.pool`／`self.head`，沒有說明這是方法內片段。按前面 main 風格直接貼入 code cell，`self` 沒有定義；初讀者容易以為前一節五行學習流程可以直接搬到這裡。建議改為 `model.features`／`model.pool`／`model.head`，或將 forward 方法與外部訓練步明確分成兩段。

問題 A2（首次用詞，`docs/lessons/01-small-cnn.md:80`）：這是指定順序中第一次提「L2 長度」，尚未像第 03-comparison 節那樣解釋「各梯度平方相加再開根號」。對不熟大學數學者，無法理解非零長度是在檢查什麼。建議在此首次出現加同樣一句定義；不用導入向量空間或範數理論。

## 02-diagnostics

實際閱讀：`docs/lessons/02-diagnostics.md`、`lesson_cases/02-diagnostics.py` 全文，`notebooks/02-diagnostics.ipynb` 四格與輸出，`artifacts/checks/curriculum/02-diagnostics.json` 全文。本節沒有 SVG 或 learning 補充。

四個檢查點有順序，detach／label 契約／held-out 各自回答的問題很清楚。程式確實只檢查斷圖時 head 梯度存在，沒有測量那個模型 loss 下降，正文已避免超出證據。無 bias 線性模型從零開始，訓練只有兩個不同點、各重複四次；本文有明說，因此不會把 8 筆誤讀成 8 種不同情況。核對更新後 step0／9／19 的 train loss .5945／.2244／.1236，validation loss .7937／1.5204／2.0153，accuracy 1.00／0.00。最終權重兩列為約 `[-.1950317,-.9751588]` 與 `[.1950317,.9751588]`，確實偏向較大 b 訊號。

問題 A3（需要修訂，`docs/lessons/02-diagnostics.md:57`、`lesson_cases/02-diagnostics.py:58`，notebook 實驗格同樣）：練習要求新增反背景的 8 點，卻未提示原本 `assert train_acc == 1.0 and val_acc == 0.0` 是原分佈的特定答案。用相同零初始化、SGD .2、20 步，CPU 獨立實跑新增 `[-.3,+1]`／`[+.3,-1]` 各四筆後，得到 train accuracy=1.0、validation accuracy=1.0，權重約 `[-.4708219,-.0133591]`／`[.4708219,.0133591]`；改動成功反而撞到原斷言。建議像其他節一樣明說應同步更改分佈特定的預期，提供此固定設定的可核對結果，保留梯度有限／訓練更新等檢查，不要讓初學者以為 validation 變好是程式錯誤。

## 03-identity

實際閱讀：`docs/lessons/03-identity.md`、`lesson_cases/03-identity.py` 全文，`notebooks/03-identity.ipynb` 四格與輸出，`artifacts/checks/curriculum/03-identity.json` 全文。本節沒有引用 SVG 或 learning 補充。

同 shape 逐值相加、與串接的差異、相加後 ReLU 不能保存负值，均能一次讀懂。無 bias 的兩個 4→4、3×3 卷積共 288 參數正確。人工 F=0 時 `[-2,-1,0,1]` 原樣通過，sum loss 得各輸入梯度 1，與程式／保存結果一致；全部分支權重零會令兩層權重梯度都零的說明正確，沒有誤當初始化建議。隨機 block 的 shape `[2,4,8,8]`、兩步 loss 1.0816→1.0724 對上 JSON。ReLU 練習的 y=`[0,0,0,1]` 與正／負值梯度答案正確；正文有提示改斷言，stride 練習明說將發生 mismatch。

問題 A4（訓練目標跳躍，`docs/lessons/03-identity.md:7`、`:54`，對應 `lesson_cases/03-identity.py` 的 `target=torch.zeros(...)`）：第二個隨機 block 的兩步實驗，正文只說可更新與「玩具 MSE」，未說它是拿隨機輸入、把整個 block 輸出往零 target 拉，也沒展開 MSE 意義。剛讀完「F=0 保存 x」的初讀者容易把 loss 下降理解成模型更會保持 x；實際此小實驗希望 F 往 -x 修正。建議補一句「第二部分 target 是同 shape 全零，用各輸出與零的誤差平方平均（MSE）檢查參數可更新；這個目標刻意不同於上面的 identity 算術測試」。無需增添新理論。

## 03-projection

實際閱讀：`docs/lessons/03-projection.md`、`lesson_cases/03-projection.py` 全文，`notebooks/03-projection.ipynb` 四格與保存 stdout，`artifacts/checks/curriculum/03-projection.json` 全文；`docs/assets/diagrams/03-projection.svg` 原始內容與渲染圖。本節沒有 learning 補充。

stage 在圖後立即定義。1×1 只限制空間、仍混合輸入 channel 的說明具體，321 的例子能手算。圖中主分支兩層順序、s／p、shape、18 個 projection 權重與 code 相符，畫面文字無重疊。核對主分支參數 6×3×9+6×6×9=486，P=18、总计504；P 每圖 MAC=2×2×6×3=72，加法24，均正確。人工輸入 shape `[1,3,4,4]` 經兩支得 `[1,6,2,2]`，第一 channel 四點321；隨機訓練 loss .3553→.3350 對上 JSON。

5×5 練習兩支都變3×3，正文精確指出要改的 shape 與 `torch.full` assertion；P stride=1 的預期 mismatch 正確。沒有必須修訂的問題。MSE 可沿用前一節若補充 A4 的定義，此節已明說零 target，不再跳過目標。

## 03-comparison

實際閱讀：`docs/lessons/03-comparison.md`、`lesson_cases/03-comparison.py` 全文，`notebooks/03-comparison.ipynb` 四格與保存輸出，`artifacts/checks/curriculum/03-comparison.json`、`03-comparison-learning.json` 全文；`docs/assets/diagrams/03-comparison-learning.svg` 原始內容與渲染圖；extension 的比較分支、共用訓練／評估程式。

對照條件交代完整，`load_state_dict` 與逐參數相等檢查確保同初始權重，兩個 optimizer 各自建立。Train top=3、validation top=5，圖片確實不同；但位置列表中每類只有少量不同樣本，本文的人工資料限制已足以避免當成真實泛化證據。主實驗資料／步數／LR 一致；L2 有明確簡單定義。

核對 112+3×288+10=986 參數，MAC=27648+221184+8=248840；shortcut 額外3×4×16×16=3072 次加法。plain loss .6938→.6936、stem 梯度約 .000987→.001014；residual .6922→.6855、stem 梯度約 .146834／.147695／.146226；兩者三步 validation 都 .50，與 notebook／JSON 相符。保存時間 .0124／.0081 秒被明說是單次 CPU 數值，沒有做速度優越結論。

40 步 plain 首步 .6937906742→最後更新前 .6929427385、train／validation .5／.5；residual .6921603084→.0307393558、1／1，均符合完整 JSON 與圖上曲線。1-block 練習參數410、MAC101384、額外加法1024正確，且要求改斷言及列印成本。沒有必須修訂的問題。

## 04-localization

實際閱讀：`docs/lessons/04-localization.md`、`lesson_cases/04-localization.py` 全文，`notebooks/04-localization.ipynb` 四格與保存輸出，`artifacts/checks/curriculum/04-localization.json`、`04-localization-learning.json` 全文；`docs/assets/diagrams/04-localization.svg`、`04-localization-learning.svg` 原始內容與渲染圖；`artifacts/04-localization.png`；extension 定位分支及 IoU 呼叫。

GAP 範例的1/16=.0625可以理解，本文也保留卷積邊界等可能帶位置資訊的限制，沒有絕對化。框格式、半開區間、不同尺寸除 W／H、sigmoid 四個比例与機率的差異均已說清。核對 `[4,6,16,18]`→`[10,12,12,12]`→`[.3125,.375,.375,.375]`，flatten 框 head 1024×4+4=4100，平均再 linear 4×4+4=20。固定人工 pred `[6,8,18,20]` 的 MSE .001953125、IoU100/188≈.5319149 正確；SVG 綠／黃線座標及比例也正確。

三步 class loss .7173→.7123、box loss .0185→.0091、total .8099→.7576，與 JSON 相符；model PNG 中首張 yellow pred 約 `[7.4886,7.1996,20.5954,19.9956]`，第二張約 `[9.9417,8.6842,23.8924,22.7757]`，對上列印值。網頁人工 SVG 與 model PNG 的不同被正文明确區分。右移6px的手算練習交集72／聯集216／IoU1/3、MSE .0087890625 正確，正文明确說不用改模型輸出，能操作。

40 步 Adam 補充首步 .8098831773→最後更新前 .4302131534，總參數4222，兩張訓練 IoU .6944889426／.7752397656，與 JSON 及曲線吻合，並未宣稱獨立定位評估。

問題 A5（指標名稱清晰度，`docs/lessons/04-localization.md:66`）：40 步表頭是「訓練 accuracy」，localizer=1.00。實際 `run_learning_extensions.py` 算的是 class logits argmax 與 labels 相等，即分類正確率；框仍只有約 .6945／.7752 的 IoU。初讀者容易把 1.00 解成整個定位任务全對。建議表頭明寫「訓練分類 accuracy」，並注明總 loss 是分類+5×框 MSE；已有模型 IoU 數字可直接保留。

## 04-coordinates

實際閱讀：`docs/lessons/04-coordinates.md`、`lesson_cases/04-coordinates.py` 全文，`notebooks/04-coordinates.ipynb` 四格與輸出，`artifacts/checks/curriculum/04-coordinates.json` 全文；`docs/assets/diagrams/04-coordinates.svg` 原始內容與渲染圖；`artifacts/04-coordinates.png`。沒有 learning 補充。

三個座標空間、順序先扣 padding 再除 scale，及 float 框 dtype 的理由清楚。80×40 原圖 stretch 的 [.8,1.6]、letterbox scale .8／top16，原框 `[10,5,50,25]`→`[8,20,40,36]`→原框均正確。正規化 `[.125,.3125,.625,.5625]`、錯誤直接乘原圖所得 `[10,12.5,50,22.5]` 也對。37×83 奇數例子 resize hw=(29,64)，實際 x scale64/83≈.7710843、y scale29/37≈.7837838，JSON 符合；正文明說这是依整數尺寸的教學契約。N=0 保留 `[0,4]` 的程式核對成立。

SVG 的原圖、縮放圖、補邊與綠框位置按相同比例繪製，PNG 的三張實際圖也與固定框對上。SVG 灰色用來識別 padding，程式實際用黑零值，正文已提醒需按訓練契約選 padding；沒有把灰色宣稱成模型前處理值。

問題 A6（需要修訂，`docs/lessons/04-coordinates.md:60`、`lesson_cases/04-coordinates.py:38`、`:43`、`:52`、`:67`，notebook 同樣）：直式練習只說改 H／W 與框，未列出要同步改的繪圖切片、固定 letterbox 答案、stretch 比例及圖標題。獨立呼叫 helper 驗證新框確實為 `[20,8,36,40]`、resize hw=(64,32)、padding `[16,0,16,0]`，且正確往返回 `[5,10,25,50]`。但是直接照頁面改 image／boxes 會先在舊 `[8,20,40,36]` assertion 失敗；即使換掉該答案，舊 stretch 比例 `[.8,1.6,.8,1.6]` 會把新框變成 `[4,16,20,80]`，甚至超出64畫布。原紅矩形切片也會與新框不同，圖標題還寫80×40。

建議给出像 projection 節那樣的具體改動清單：`image=torch.zeros(3,80,40)`、紅矩形 `image[0,10:50,5:25]=1`、boxes 改新框、letterbox assertion 改 `[20,8,36,40]`、stretch 因子改 `[1.6,.8,1.6,.8]`、original title 改40×80；stretch 框固定答案仍是 `[8,8,40,40]`，往返與空框檢查繼續保留。也可把這個練習明確做成新增的獨立 helper cell，避免修改原範例各處固定值。

## 初讀結論

八節的概念順序、人工驗證與模型輸出的區分大致清楚，核心數字都能追到算式、程式或保存紀錄。A3／A6 是會阻止讀者完成指定練習的實際問題；A1／A2／A4／A5 是首次閱讀時可用短補充消除的歧義。此報告保留第一輪發現，待根 agent 修訂後再另記複查結果。

## 獨立複查：2026-10-02

保留以上第一輪全文。本次重新閱讀八節目前的完整正文，逐項對照修訂；重新讀八個 notebook 的 markdown，核對完整 case cell／stdout／case hash。三個 40 步可選補充是在 notebook 原有的第 2 格 markdown 中，沒有新增可執行 cell；指示明寫完成完整案例後另開 code cell，與可選性相符。沒有因根 agent 宣告修好就預設通過。

所有必要實測以 `.venv-model/bin/python` 在 `/tmp/reader-a-recheck` 執行。A1 使用原 case 的模型與資料設定後，直接執行目前網頁的 Python 片段；A3 只在記憶體修改原 case 的訓練資料及該頁指定的新 assertion，再執行完整案例；A6 建立與 notebook 完整案例相同的全域 helper 定義後，直接執行目前網頁的新增獨立 cell。未修改 lesson case 或 notebook。七張目前 SVG 重新渲染後逐張目視；直式練習額外生成並目視的圖只存於 `/tmp/reader-a-recheck/portrait.png`。

| 第一輪問題 | 複查結果 | 具體證據 |
| --- | --- | --- |
| A1：CNN 片段混用 self | 已解決 | `docs/lessons/01-small-cnn.md:40` 改用 `model.features`／`model.pool`／`model.head`。按原設定實跑得到 feature `(8,8,8,8)`、summary `(8,8)`、logits `(8,2)`、loss `0.6941443085670471`，forward、backward、step 全部完成。 |
| A2：首次 L2 未定義 | 已解決 | `docs/lessons/01-small-cnn.md:80` 已說平方相加再開根號，且解释大於零表示至少一個非零梯度，不是品質分數。與 extension 記錄的整體梯度 norm 相符，沒有把「所有參數個別非零」混進結論。 |
| A3：新增反背景資料撞到舊 assertion | 已解決 | `docs/lessons/02-diagnostics.md:61` 明確指出舊答案只適用原題，並要求 train／validation 都為 1.0 的新 assertion。按此操作完整案例實跑：step0 train／validation loss `.6869/.6882`，step9 `.6343/.6498`，step19 `.5824/.6156`，最後 accuracy `1.00/1.00`；新 assertion 與原有有限梯度／修復更新檢查全部通过。 |
| A4：identity 隨機兩步目標不明 | 已解決 | `docs/lessons/03-identity.md:52` 已定義同 shape 全零 target、MSE 平方後平均，並說明 F 可能向 −x 修正、目標與第一部分保存 x 不同。與 case 的 `torch.zeros(2,4,8,8)` 及 `mse_loss(prediction,target)` 一致。 |
| A5：定位 accuracy 誤讀成框全對 | 已解決 | `docs/lessons/04-localization.md:66` 的兩個 accuracy 欄都明寫分類。正文仍分開提供模型框 IoU `.6945/.7752`、無獨立定位評估；總 loss 的分類+5×框 MSE 定義仍在前文，表格不再讓 1.00 冒充定位精度。 |
| A6：直式練習漏改多個固定值 | 已解決 | `docs/lessons/04-coordinates.md:62` 提供獨立 cell，不需修改原 main。原樣實跑四個 assertion 全過：resize hw `(64,32)`、mapped `[[20,8,36,40]]`、padding `[16,0,16,0]`、往返回 `[[5,10,25,50]]`、stretch `[[8,8,40,40]]`。紅矩形切片總和800，恰為20×40，與新框相符；目視直式原圖／letterbox 綠框都貼合紅矩形。另有修改原 main 時須同步改切片、答案、比例與圖題的提示。 |

`00-warmup`、`03-projection`、`03-comparison` 原先沒有必須修訂的问题，本次重新讀全文未見回歸。暖身仍為梯度 −8、w=1.8、新 loss=.16；projection 圖的兩支 shape `[B,6,2,2]`、18 個權重、321 手算與正文一致；comparison 成本986參數／248840 MAC／residual額外3072加法、三步各 .50 validation，以及40步 `.692943/.030739` 的兩條終點均保留一致。

目前七張 SVG 的文字、框位置與曲線再次目視通過：小 CNN 的紅色錯誤仍對应前四張的第0／2張；projection 路徑無文字重疊；人工定位圖仍明示是人工框，交集100／聯集188；letterbox 圖仍是橫式主例的上下補邊16，與新增直式練習的左右補邊16互不混用。40步 CNN、comparison、localizer 圖的曲線終點仍分别為0、約.693／.031、約.430。

三個相關 notebook 的可選 markdown 均使用正确的 `--section` id，並用 `IPython.display.SVG` 指向該節實際 SVG 路徑。每個命令啟動獨立 Python 程序，會依 extension 自己的 seed／設定訓練，文字也明說不取代三步檢查。此次沒有重跑這些會覆寫教材輸出的命令；驗證限於目前程式、路徑、保存結果與圖的對照。八個 notebook 的 case cell 與目前 case 逐字相等、stdout 與對應 JSON 一致、case SHA256 均相符，因此沒有引入 notebook／case／證據版本不一致。

複查未發現新阻礙；A1–A6 全數解決。結論是目前預備版本的這八節可以按正文完成原範例與指定練習，不代表未發布的 `lessons-v0.2.0` 已在全新 Colab 或真實影像任務完成驗證。

# Foundations：獨立第二輪技術與證據審查

審查者：`clear_first_evolution`。範圍是下列 11 頁現行工作稿；本審查者未撰寫或修改這些教材。基底 HEAD 是 `16f69103d1f216d1024a40610623083557324701`，工作稿含其後修正，不能把本報告當成舊 commit 的通過紀錄。新 `lessons-v0.5.0` 發布、notebook pins 與站台呈現另由協調者處理。

已完整閱讀實際 SKILL.md、review-protocol.md、11 頁正文、對應 lesson case 與 CPU 紀錄。本文是第二輪；先前 evolution 分段首次閱讀的 148 筆原始紀錄與 summary 保留原樣，沒有因本轮修正改寫成通過。

## 結論與實際方法

未發現尚未解決的 blocking 技術或數值錯誤。下列 11 個完整 case 在隔離 CPU 副本通過全部既有 assertions；選定練習與錯法探針也實跑。三個 40 步延伸實驗的全部 model 欄位（包括全部逐步歷史）與現有 JSON 完全一致。11 個案例 stdout 也一致，03-comparison 的單次秒數只核量測範圍，沒有要求機器負載下的秒數相等。

實驗使用現有 `.venv-model/bin/python`、PyTorch 2.9.1+cpu、2 執行緒，`PYTHONPATH=/tmp/clear-tutorial-foundations-tech-evolution`。完整複製 lesson_cases/miniyolo/scripts 到該目錄；執行前比對 Python 源碼，沒有需要更新的差異。所有圖片、JSON、模型更新與練習變體留在隔離副本，未覆寫原資料、權重或 evidence。沒有 GPU、安裝、commit 或 push。

實際腳本為 `/tmp/clear-foundations-cpu.py`、`/tmp/clear-foundations-cpu-extra.py`；結果在隔離目錄的 `cpu-checks.json`、`cpu-extra-checks.json`、`ce-sgd-checks.json`、`shared-checks.json`、`logs/` 與 `learning/`。後面保存教材和 evidence 指紋，核心查核結果也寫在本報告，避免只靠暫存 log。

本審查者逐一讀圖源與看過 17 張 SVG 的獨立 Chromium render：用 SVG 原文 `page.set_content`、按 viewBox 設定 viewport，顯式讓 SVG 填滿 viewport，排除 pt 與 CSS px 的換算造成預覽裁切。這是素材本身的檢查。**沒有在實際 Zensical 頁面做 desktop/mobile、數學排版、導覽切換或 inline 縮放檢查**；那些仍須以協調者的第三輪瀏覽器紀錄為準。沒有把 build 或 SVG 預覽冒充完整頁面驗圖。

## 逐頁查核

### 00-warmup

- 實際閱讀 scalar Linear case、完整 stdout、autograd/SGD/zero_grad 官方文件，並執行原 case、lr=0.25、略去 step 的練習變體（保留且改成相應已推導答案的 assertions）。
- x=2、target=4、w=1：prediction=2、MSE=4、gradient=-8；lr=.1 更新至 1.8、prediction=3.6、loss=.16。有限差分 w=1.01 的 loss=3.9204、斜率=-7.96 趨近 -8。lr=.25 得 w=3、loss=4；不 step 仍 w=1、loss=4，均符合正文。
- 實驗確實區別梯度累積、None 與零、alias/detach 共享 storage 與 clone 保存舊值、optimizer 持有原 Parameter。`eval()` 沒有關閉梯度，no_grad 的作用與位置一致。
- 無圖；純量任務和各數值已明列，不需自行猜圖或位置。未將此節聲稱成已教一般向量內積或 Linear `[K,D]` 權重矩陣。
- 疑點／建議：本頁無未閉合問題。來源支持見 D1/D2/D3。

### 01-small-cnn

- 完整讀模型、CPU 三步紀錄、40 步腳本與 JSON；實跑三步、width=8 練習、40 步 Adam。逐張查 `make_batch` 的非零畫素、channel、label，再對上 materials.svg；前四張三步圖板也查位置和 GT/pred。
- 八張框的 xyxy 依序為 `[4,6,16,18]`、`[8,6,20,18]`、`[12,11,24,23]`、`[4,11,16,23]`、`[8,6,20,18]`、`[12,6,24,18]`、`[4,11,16,23]`、`[8,11,20,23]`；各 144 畫素，偶數 channel0/label0、奇數 channel2/label1。SVG 黑圖 96×96、方塊 36×36，每原畫素 3 SVG 單位，所有位置與編號吻合。1/4、3/6 確實同位置異色。
- flow 的 NCHW、兩次 pooling、GAP、flatten/head 均對實際輸出。參數 112+148+296+584+18=1158；MAC/image 110592+147456+73728+147456+16=479248；batch=3833984；首層數值 128 KiB。width8 得 4330 參數、1695776 MAC/image，原 case 調整答案 assertions 後通過。
- 卷積是跨 channel 的 cross-correlation；掃窗圖明示只畫一 channel，同一濾鏡移位重用。反推每層索引並在每層有限特徵圖邊界裁切，最終第 i 格理論範圍 `[4i-6,4i+9]`；i=0 實際 `[0,9]`、i=3 `[6,21]`、i=7 `[22,31]`。所有 i 的實際範圍均吻合。r/j 逐層結果與 padding 說明正確；不是每個點都同樣影響。
- 三步的 loss .6942/.6941/.6940 都在更新前，更新後八張全 pred1；圖板只呈現前四張。初始藍機率範圍 .52148–.52159，正文約 .52 正確。另核 CE=`mean(-log p_y)`、logits 梯度 `(p-onehot)/8`，差異最大 7.45e-9；SGD 每個參數與 `old-.1*grad` 完全一致。雙 softmax [.9,.1]→[.689974,.310026]，最小 CE 約 .31326，正文約 .31 正確。
- 40 步全部逐點歷史重現：初始 .6941587328910828、第40次更新前 0、更新後 train accuracy 1、validation null。權重差 L2=5.17251308976495；梯度 L2 min=1.4261925521200407e-18、max=.7967736124992371。新 readable SVG 的 40 個點逐一對 JSON 映射，最大座標誤差 .0004872 SVG 單位（僅保留三位小數），來源 hash comment 正確。點數、1–40 時間軸與「更新前」標示正確。
- 捨入另實驗：logits [20,0]/label0，float32 CE=0、梯度 `[0,2.0611537e-9]`；float64 CE≈2.06115369e-9。實際第40點 finite logit gap 為 44.9486–47.8197，用穩定 double `mean(log1p(exp(-gap)))` 得 1.44265e-20，確認數學 loss 仍正、float32 報0合理。40步和三步同時改 optimizer/步數，正文正確限制因果；只學這八圖，未測新圖。
- 歷史 VGG D/C、13conv+3FC、512 上限、4096/4096/1000 head，及 NIN GAP 都查原論文 R1/R2。發現與核回見下文 F02/F03/F05，不能把這個小 CNN 當完整 VGG16 重現。

### 02-diagnostics

- 讀 body/head detach、label guard、人工分佈移轉案例，實跑原 case 與加入反向 b 關係的八筆練習。修復後 body 的梯度非零且權重改變；invalid label 是 preflight guard，沒有誤稱該段曾真的呼叫 CE 報錯。
- train 八筆只有兩種向量重複；validation 反轉 b。zero-init SGD20 的 train loss .5945→.2244→.1236、val .7937→1.5204→2.0153，accuracy 1/0 重現。權重兩列約 ±(.19503,.97516)，ratio5、分界 b=-.2a，由 collinear 資料及 SGD 推導成立，不泛用到 Adam。
- balanced 練習得到 train/val accuracy 1/1，權重約 ±(.47082,.01336)，支持 a 的規則占主導。圖中四點、實心/空心、兩軸比例、分界與上下箭頭，均對上原資料與分類結果。
- 正文要求多層、多批、多步核梯度，沒有把一次小梯度當梯度消失。它是人工 distribution shift，不能推論真實資料的泛化。
- 疑點／建議：無未閉合問題；與 03-comparison 的用詞經 F04 修正後一致。

### 03-identity

- 原 case 與 post-add ReLU、首卷積 stride2 變體實跑；查 R3/R4 與 torchvision 固定版本 D6。
- 單 channel 零分支把 `[-2,-1,0,1]` 原樣輸出，sum probe 的 input gradient 全1；這是已知答案和梯度路徑 probe，不是分類訓練。四 channel 隨機分支 MSE 兩步確有 nonzero branch gradient 和權重變化，shape、18/288 參數分別屬兩個不同測試。
- post-ReLU 得 output/gradient `[0,0,0,1]`，包含 PyTorch 在0的導數0。stride2 在首例2×2會把1×1分支廣播，不能靠「程式沒錯」認定 shape 合法；8×8對4×4會 RuntimeError，正文的 assert 建議成立。
- 原 ResNet 相加後 ReLU 與本節拿掉該 ReLU 的差異已在需要處與圖上明示；原始 Fig1 是 CIFAR plain56/20 的 train degradation。非負輸入的 identity 構造與任意含負輸入受 ReLU 限制分清。tensor 梯度應是 Jacobian，純量公式只作類比；不宣稱梯度必不消失。
- 全分支零化只是 probe，不能當 train-init 建議。torchvision zero_init_residual 實作是最後 BN scale=0，仍保留其他權重，支持正文。
- 疑點／建議：無未閉合問題。圖僅示第二個四 channel block，B 代表 batch，沒有畫成完整原版 ResNet。

### 03-projection

- 完整 case、321/probe、learned update、5×5 odd input、stride3 與 avgpool 變體實跑；手數有效/含 padding MAC，查 R3/R5/D4。
- F 參數486、P18、總504；MAC F648+1296=1944、P72，多約3.7%，相加24次；排除補零讀取時 F450/576。P 的跨 channel 內積321與錯序123 正確。stride2 probe `[0,2;20,22]`；stride3 `[0,3;30,33]`；avgpool版 `[5.5,7.5;25.5,27.5]`，shape/常數圖不能抓位置讀法，數字探針可以。odd5兩支都3×3。
- weights `[6,3,1,1]` 的四軸、Wv、stride2 只讀四位置與其餘位置只能經 F，現稿自足。圖 F/P 的6channel、2×2、486/18及無 ReLU shortcut吻合。
- 原 ResNet option A/B、1×1/stride2 與 ImageNet Table3 A25.03/B24.52 得到原論文支持；ResNet-D 的 avgpool2s2 + conv1s1 在 R5 Sec4.2 明載，探針不同不代表設計錯誤。
- 本案兩個失效暖身交叉引用已由作者修並實讀核回，見 F01。沒有宣稱 learned P 是 orthogonal projection、原樣 identity 或測到分類品質。

### 03-comparison

- 讀所有控制、暖機、cost、gradient、三步與40步結果；原case、one-block練習和40步SGD實跑。查 R3/D6/D7 的 BN/He 與 PyTorch default-init。
- 初始化 state_dict 相同、資料位置兩類完全平衡，4張val未與8張train重複；人工色塊、single seed、noBN/nopostReLU 是限制。3block兩者參數986、MAC248840；residual另3072 adds。oneblock為410/101384/1024。
- 原三步 plain stem norms .000984/.001001/.001011；residual .146701/.147590/.146128；兩者 val .5。oneblock plain .036030/.037604/.038848（遠超十倍），residual .160922/.157635/.153132，符合練習答案並保留 head 初始化隨消耗乱數次數變化的限制。
- 40步全部歷史和圖 bytes重現：plain .69379282→.69294828、train/val .5；residual .69213802→.03095774、train/val1。warmup深拷貝不改真模型；計時含三個 train_step與列印、排除 warmup/val/import，單次秒數不排名。
- default conv uniform±1/sqrt(fan_in) 得權重均方1/(3fan_in)，兩conv夾一ReLU 的約1/18是忽略相關性/對稱性等條件下的估算，正文明示未量每層。不把小例的問題當成原 ResNet degradation；論文 Sec4.1 強調 BN 下前/反向訊號正常，Sec3.4 有 BN及He-init參考。
- 用詞 F04 已修並核回：僅說 stem 梯度變小、尚未逐層診斷，與02的診斷規則一致。

### 04-localization

- 讀 mean class/flatten box architecture、手工框、三步與40步腳本；實跑原case/40step、獨立IoU/歸一化算術，查共用 geometry。
- features `[2,4,16,16]`；class head10、box head4100（總模型4222），sigmoid cxcywh以32正規化。4×4人工特征的位置交換有相同GAP .0625而flatten位置不同；只說平均操作不保留排列，不宣稱任何用GAP的模型絕不定位。
- GT `[.3125,.375,.375,.375]`／`[.6875,.5,.375,.375]`。手工pred `[6,8,18,20]` 兩坐標差2/32，MSE .001953125，IoU100/188=.531915；4×4同偏移IoU4/28=.142857；水平偏6 MSE.0087890625/IoU1/3。圖的物件、GT/pred、交集和單位都吻合，明示人工固定框。
- total=class+5box，三步列印更新前；.7576已是兩次更新後量，不是三次更新後量。權重5影響共享backbone梯度和box分支；不能從loss絕對值直接讀梯度強弱。
- 40步所有字段及SVG bytes重現：initial .80988318、lastpre .43021327、trainclass1/valnull；最終訓練框約 `[4.47785,4.77010,18.13039,18.93271]`、`[16.32700,10.58485,29.18213,23.11913]`，IoU .69448936/.77524126。颜色與位置绑定和兩圖訓練限制有明示，不能用来證明flatten胜過mean或新圖泛化。
- 疑點／建議：無未閉合问題；實際網頁上的雙圖縮放未由本審查者驗證。

### 04-coordinates

- 原case、portrait練習與錯切片、odd37×83、自共享geometry往返均實跑；核半開格線與補邊圖、rounding局部放大圖。
- 原W80/H40框[10,5,50,25]；stretch sx.8/sy1.6→[8,8,40,40]；letterbox64×32、top16→[8,20,40,36]，canvas normalize `[.125,.3125,.625,.5625]`，先減pad再除scale可回原圖。红像素閾值bbox與框一致；實際補0，灰色只用於解說。混用+1的861/800、IoU121/217/100/188均正確。
- odd W83/H37→64×29，sx64/83、sy29/37，top17/bottom18。actual底邊46、ideal45.53；各自往返都可能成立，须對齊實際整數縮圖尺寸，正文和放大圖有說明。Python ties-even 的28.5→28也符合實際。整數框轉float、空[0,4]與finite近似往返通過。
- 直式W40/H80正確框[5,10,25,50]→[20,8,36,40]；若仍錯用舊切片，實際红物件bbox[24,4,48,20]，檢查会失败。題目列出的圖/切片/GT/assert/stretch更新位置齊全。
- local與miniyolo letterbox在odd尺寸給出完全相同canvas及轉框；metadata命名不同但計算一致，undo結果誤差≤2e-6，非數學整數精確相等。錯把canvas normalized直接乘原W/H得到[10,12.5,50,22.5]。
- 疑點／建議：無未閉合问題。

### 05-assignment

- 原case、非對稱probe、collision、移藍框S4/S8及完整head S8練習實跑；與miniyolo.targets/losses/train實際實作對照，另查R6原YOLOv1。
- B2/S4一個slot/C2：兩positive/30negative/0ignore，首圖14negative；centers12/44→(0,0)/(2,2)，target均[.75,.75,.25,.25]。圖的中心、跨格框與每格16像素一致，圖只画首圖故negative14。
- asymmetric [2,4,22,16]→(0,0)/[.75,.625,.3125,.1875]；[36,12,52,28]→(1,2)/[.75,.25,.25,.25]，確能抓axis/rowcol錯誤。samecell同類第二物件raise明確，NMS刪候選無法補回輸出容量。
- 正格box/class、全格obj的mask與retain_grad檢查成立；背景class -1不被送到CE，默認ignore_index=-100不能自行忽略-1。共用miniyolo把背景class_ids填0但在positive mask外，正格box/mask/class完全一致。此case是random features下兩次head更新，非完整detector訓練。全空batch限制有交代，miniyolo.grid_loss另有connected-zero處理。
- 藍框移動到[20,36,36,52]後S4=(2,1)，S8=(5,3)，red S8=(1,1)，targets[.5,.5,.25,.25]。S8完整batch2产生126negative／首圖62，原odd/collision仍S4；按現文只改主例call、features和相應asserts即可全程通過。
- YOLOv1是B2、grid共享class、confidence含預測IoU、linear末端和平方loss；本教學sigmoid/softmax/BCE/CE及固定obj1不是原模型複現。R6支持區分；miniyolo.train確實每步build該batch的targets。
- 疑點／建議：無未閉合问題。

### 06-decode-nms

- 原case及其中四個閾值/跨類/空輸入練习實跑，與miniyolo.decode_grid同一人工logits比較，查D8嚴格IoU標准。
- row1col2/.25,.75,.25,.125解碼[28,24,44,32]，obj.9×class.8=.72。row1col1=[23.2,24,39.2,32]/.64；row3col0=[4,52,12,60]/.855。索引按mask的grid順序，不按fill順序；IoU7/13=.53846，NMS.5保留[2,1]。
- .75只留下背景誤報紅框；score.70留下blue+red但候選重新编号；NMS.6保留原[2,1,0]。同位置不同類兩框均留下，空NMS通過。共享decode最終boxes/scores/labels與case完全一致。
- SVG矩形位置按64×64→288×288縮放、ground truth與藍框重合、yellowduplicate偏4.8px、red背景框與閾值說明吻合。NMS不用GT，score不是precision且不是已校准成功概率。
- 當前letterbox逆變換blue→orig[35,10,55,20]是正文假想延伸，不伪稱該主case已經做過逆變換；共用geometry另已查。
- 疑點／建議：無未閉合问題；local單圖CPU函數沒有承诺GPU/多batch，工程版另做device/clip/degenerate處理。

### 06-evaluation

- 原case與其中閾值/刪FP/錯類/無GT例均實跑；獨立計算TP/FP/FN、envelope/allpoints面積；讀固定COCO源碼、VOC devkit實際匹配代碼，並用共享metrics跑重叠GT反例。
- 高分FP、TP、duplicate FP、TP→flags[0,1,0,1]，TP2/FP2/FN1、microP.5/R2/3、AP50=1/3；無GT類APNone不平均。PR圖精確坐標為x84+540r/y416−360p，原點、兩個重複recall的點、.5包络、2/3後0與面積均正確。
- 提高score threshold .85→AP1/6、R1/3；僅為解釋而oracle刪掉最高FP→AP5/9。VOC2007 11點插值7*.5/11=.31818，不能混成allpoints area；COCO101個recall、10個IoU閾值另有官方定義。AP輸入是排序候選，score本身不等於precision。
- VOCevaldet先在同圖同類全部GT求最大IoU，再檢查是否已占用；COCO普通非crowd匹配跳過已用GT。兩相互重叠GT＋兩相同預測時，本課及miniyolo flags[1,1]/AP1，而所讀VOC代碼給TP/FP；現文已准確限定协議差异，沒有宣稱等價VOC。
- 固定COCO源碼evaluateImg按img/category截maxDet，params maxDets[1,10,100]且useCats1；matching/crowd/ignore、AP插值與noGT邏輯逐段核過。正文正確把自己的single-IoU、無ignore/crowd/area/maxDet教學AP與COCO區分。
- 疑點／建議：無未閉合问題。沒有下載真實VOC/COCO資料或聲稱完成官方benchmark。

## 發現、修正與獨立核回

F01（03-projection，增加理解負擔）：原句把 (1,2,3)·(1,10,100) 的內積與 Linear `[K,D]` 當成暖身已經教過。實讀00只教scalar Linear(1,1)，沒有這兩項；讀者須借未教前提。作者改為當場由321定義「對應位置相乘後相加」及卷積四轴/Wv。審查者實讀修正文及00後，確認定義在需要處且不再借錯前文。已關閉。發現時未另存修改前整頁hash；上述原句記錄保留，最終指纹在後表，不能偽造舊hash。

F02（01，增加理解負擔）：page末歷史選讀重排後仍說block/GAP/channel「下文說明」。发現時page SHA=`5f67ca8179ed189b2d1f3bb23e0b198e7ca241e74fa3c375fabcf6ec3dfca739`。作者改「前文」，審查者實讀現note及相應前文，確認順序一致。已關閉。

F03（01，增加理解負擔）：原「GAP不是VGG的設計」範圍過廣；R1 Appendix C/PDF12確有GAPdescriptor，標准分類head則是FC。與F02同一发現時pagehash。作者改「GAP不是標準VGG分類head的設計」，並限定NIN替代FC的做法；審查者核回現文並對照R1 Sec2.1/Table1/AppendixC與R2 Sec3.2。已關閉。

F04（03-comparison，選讀改善）：「符合第2章說的梯度消失」容易超出只量stem的證據，而02明確要求逐層診斷。发現時page SHA=`36cad7348ab189652c46990abb0dd8e5c85b3b2ffc46068bc3b4ff97dfdb22de`。作者改「符合梯度變小的現象，尚未逐層診斷」；審查者實讀新句及02的診斷原則，確認不再越過證據。已關閉。

F05（01，選讀改善）：數學 `d=z_y-z_other` 微分 `-q` 與float32實際兩個logits梯度需要區分。實跑[20,0]得到float32 `[0,q]`，正確class `p_y-1` 已圓為0；整體梯度仍非零。已回報建議明示數學推導與實際反傳的差異。作者已補「未捨入的數學微分」及實際正確類0／錯誤類2.061e-9；審查者實讀現句，與CPU結果核對一致。已關閉。

F06（06-evaluation，結構delta核回）：作者把VOC／COCO官方細節放到AP50主例之後，把AP與最後P×R關係折疊為選讀。審查者實讀現稿的matching→四預測表→PR點／包絡／面積→AP50／mAP→官方比較，確認主線先建立本課規則，後續仍明示教學評測不等於官方工具；所有公式、閾值、反例及限制保留，沒有使讀者先套官方配對再錯解主例。只移動現有段落，不需要重跑訓練。已核回；最終hash見後表。

本輪修正的閉合只關乎本輪現稿與上述具體問題，不會自動把首次閱讀 raw issues 或第三輪未驗頁面改成通過。

## 已讀原始與官方來源

R1–R6為實際取得並讀支持段落的固定arXiv版本；D1–D8為版本化官方文件/官方源碼；E1/E2為固定評測實現。下載後不是只看摘要或記憶；本地PDF文字由pdftotext產生，PDF頁碼按form-feed重新定位，內容hash見後表。

|ID|來源與實際支持位置|
|---|---|
|R1|[VGG 1409.1556v6](https://arxiv.org/pdf/1409.1556v6)：Sec2.1/2.2、Table1（PDF2–3，小3×3、2×2s2、FC4096/4096/1000、D/C16層與channel512）；AppendixC（PDF12，GAP4096-D descriptor）。|
|R2|[NIN 1312.4400v3](https://arxiv.org/pdf/1312.4400v3)：Sec3.2，提出feature map平均後softmax，取代大型FC、空間信息平均。|
|R3|[ResNet 1512.03385v1](https://arxiv.org/pdf/1512.03385v1)：PDF1 Fig1/CIFAR20vs56、Sec3.1/3.2 Eq1/2與post-add ReLU、Sec3.3 optionA/B/1×1stride2、Sec3.4 BN/init、Sec4.1梯度正常與slow convergence、Table3 A25.03/B24.52。|
|R4|[Identity mappings 1603.05027v3](https://arxiv.org/pdf/1603.05027v3)：Sec2 Eq1–5，h/f都是identity時的forward/backward直接路徑，非一般post-ReLU無条件恒等。|
|R5|[Bag of Tricks 1812.01187v2](https://arxiv.org/pdf/1812.01187v2)：Sec4.2/PDF5，ResNet-D在shortcut1×1前加avgpool2s2、conv stride改1。|
|R6|[YOLOv1 1506.02640v5](https://arxiv.org/pdf/1506.02640v5)：Sec2/PDF2，中心歸格、B2、grid共享class、confidence=Pr(Object)*IoU、xy相對cell/wh整圖；Sec2.2/PDF3–4，linear輸出、平方loss與responsible maxIoU。|
|D1|[PyTorch2.9 autograd notes](https://docs.pytorch.org/docs/2.9/notes/autograd.html)：Locally disabling gradient/Evaluation mode；eval與no-grad orthogonal，BN/Dropout模式差異。|
|D2|[PyTorch2.9 SGD](https://docs.pytorch.org/docs/2.9/generated/torch.optim.SGD.html)：算法無momentum/weight_decay時θ←θ−lr·g。|
|D3|[PyTorch2.9 zero_grad](https://docs.pytorch.org/docs/2.9/generated/torch.optim.Optimizer.zero_grad.html)：set_to_none及沒收到gradient參數保持None，與zero的optimizer行為不同。|
|D4|[PyTorch2.9 Conv2d](https://docs.pytorch.org/docs/2.9/generated/torch.nn.Conv2d.html)：cross-correlation、NCHW、stride/padding與shape/weight定義。|
|D5|[PyTorch2.9 MaxPool2d](https://docs.pytorch.org/docs/2.9/generated/torch.nn.MaxPool2d.html)與[CrossEntropyLoss](https://docs.pytorch.org/docs/2.9/generated/torch.nn.CrossEntropyLoss.html)：floor/ceil_mode、logits、[0,C) class indices、mean和ignore_index。|
|D6|[torchvision v0.24.0 resnet.py](https://github.com/pytorch/vision/blob/v0.24.0/torchvision/models/resnet.py#L208)：208–223，He-init及zero_init_residual最後BN scale0。|
|D7|[PyTorch v2.9.1 conv.py](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/nn/modules/conv.py#L178)：178–188，默認uniform±1/sqrt(fan_in)，不是ReLU专用He-normal尺度。|
|D8|[torchvision0.24 nms](https://docs.pytorch.org/vision/0.24/generated/torchvision.ops.nms.html)：嚴格IoU greater-than閾值；同分CPU/GPU可能不一致，課內stable tie另有限定。|
|E1|[COCO api fixed cocoeval.py](https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/cocoeval.py)：evaluateImg 235–296；accumulate 367–408；params 498–514；skip used noncrowd、ignore排序、per-img/category maxDet、101recall/10IoU。|
|E2|[Oxford VOCdevkit_18-May-2011.tar](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar)：實際VOCcode/VOCevaldet.m，allGT ovmax循環後才檢查det(jmax)，+1 inclusive coordinates、difficult不計positive。只查源碼，沒有執行MATLAB或官方資料。|

<!-- fingerprints:start -->
## 最終來源與圖指紋

快照 UTC：2026-10-05T10:31:04.713987+00:00。以下全部是現稿SHA-256；頁面含正文、Colab pin與實測footer。11份case紀錄與3份learning紀錄的dependencies_sha256逐一對當下檔案核過，沒有不符。依賴指紋不是作者主張的替代品，數值已由上述CPU重跑核對。

|來源檔案|SHA-256|
|---|---|
|`docs/lessons/00-warmup.md`|`ead34cae1663945aafde6555a633d95006477ddace3406d100aa39ff8f285bac`|
|`lesson_cases/00-warmup.py`|`7929dea8fc669b1cf5eb3fa252f479b54857e824508d52b64f0a12fb5c1ed93b`|
|`artifacts/checks/curriculum/00-warmup.json`|`fcd3ab8f34d5dae7aeb7a6defb99ef6fb44281663e580eb2f07ebbaef6c61500`|
|`docs/lessons/01-small-cnn.md`|`434c49f8ecd00ae79cd7ee76db4ae33935957d3b936a223058958fe717a5a82c`|
|`lesson_cases/01-small-cnn.py`|`f15df5625a39f5aeb9bf85eff12a6fab32f23def8d9b89065fcf0bbd2a28ca16`|
|`artifacts/checks/curriculum/01-small-cnn.json`|`465e1390104c35c3dd208a3a0dc875f2c6202292bdb9d7823c661a3493075c98`|
|`docs/lessons/02-diagnostics.md`|`b85e5021148b455e736fc18211d7d2cc8bd25904ca042add39dce8d03b229f08`|
|`lesson_cases/02-diagnostics.py`|`37675b437fe818c4c4e56b87bb5dcf76a5defb649fa8d37a30acbcc1d651ead2`|
|`artifacts/checks/curriculum/02-diagnostics.json`|`3b1feca46c25b15ad87d2729cc80ce75dd8152c3863e0231656e667e4ac229b7`|
|`docs/lessons/03-identity.md`|`5af0147567fe9bfefb6424962c8a2dde6001f1f75ff09a41403112a146cf4555`|
|`lesson_cases/03-identity.py`|`5b90ff32f504ebb96a831727e2dcd10b98640398148146f301a5e51bab0fc26d`|
|`artifacts/checks/curriculum/03-identity.json`|`30fa4d9134563599205e417e1ebb4adebfaa23bf428de80a6d3965c008983eaf`|
|`docs/lessons/03-projection.md`|`883e15e9584884c4b19a1c37563a5c0052c6259f8cadfb85c55aed0755b249b6`|
|`lesson_cases/03-projection.py`|`7239239d4f1f464ea44e865d83129d0551830ed3bab84c20d0c18839754be2a1`|
|`artifacts/checks/curriculum/03-projection.json`|`1ee13de563c90b06d1fa98e5dc1f34ad06fdfa23a625a8243de8bec9634a638f`|
|`docs/lessons/03-comparison.md`|`2a87891b7d02e7257ce3eff2fce8617f71608fcce6bf17dc1255ab9994b410e8`|
|`lesson_cases/03-comparison.py`|`8acae191b43c671164496648ccfbef3950343b06fcfbbca5df6bb21141a7fc0b`|
|`artifacts/checks/curriculum/03-comparison.json`|`e7a0879e403217629ba4d3c28d0d354028f7418b40274dcafe5d97b78e931763`|
|`docs/lessons/04-localization.md`|`81f64274118c7325ef820b5c68d397d4a7ac65835ecd903798270a1f7c3cb13b`|
|`lesson_cases/04-localization.py`|`08848d3095ceb88d3c487caccb31672901bb8d45b20fcea720155473dc06b0f7`|
|`artifacts/checks/curriculum/04-localization.json`|`c8b66584a63a8d37b47bcad567442c62a8d21b7c8730fe859af4cab42d734157`|
|`docs/lessons/04-coordinates.md`|`40f0d10a836bf80d47e09a135dd8f14b891fd357106d34aecd6ca5907cf4fa56`|
|`lesson_cases/04-coordinates.py`|`17ff0de7342281381fa8610313a40bf00e6cb927dc0eb82ace04ff3f750c968b`|
|`artifacts/checks/curriculum/04-coordinates.json`|`7537b88d2696a360a04b96370077795f36cb9412f1cb844bda3e9a8e92fc0e6b`|
|`docs/lessons/05-assignment.md`|`6866e9afeab546775b54b3cca6220c371a649d4df86e45991da832bbde71dbbe`|
|`lesson_cases/05-assignment.py`|`2045c2e8c7e0bf6dc5ad2522790892bbf92d84ff364615fa7e6efc8bffcb33a1`|
|`artifacts/checks/curriculum/05-assignment.json`|`741f573b90af9a0494e5b7ba273a60a85f36f7d9285ae278045ba956a0a13ecc`|
|`docs/lessons/06-decode-nms.md`|`5a4ab36893f7e89bd5994c0dda82e927a10158069a1a56c3cec9ceaaf9442401`|
|`lesson_cases/06-decode-nms.py`|`a3bc11cbfa0ab9435b9184c6c309fcc10eb39509bbdd81f47a7fea3d104fb307`|
|`artifacts/checks/curriculum/06-decode-nms.json`|`1665e476d969407166a66185ae4765b04c62e4965d74988829cf0f40aa6bddeb`|
|`docs/lessons/06-evaluation.md`|`1242dc8db9d2e4a095ff3b5c5e6e1ac52ef1eba7721a1963d4fe5f9a17be5630`|
|`lesson_cases/06-evaluation.py`|`a6e079570ea20361b802aef294d045641e8052ce4b6cbfe1a0cad4f2d893f0f1`|
|`artifacts/checks/curriculum/06-evaluation.json`|`85ce2991f6829ddd0ba2819da46eb3a381826a87dd8de97502eb6ee993c9ee83`|
|`artifacts/checks/curriculum/01-small-cnn-learning.json`|`624d4ad4237dedf2740064013234cbcc141d63b959bd808f18536599892bde7b`|
|`artifacts/checks/curriculum/03-comparison-learning.json`|`06b1a3ed98a77af00a528dec9a240e42325bdea32ff6efc8ab6db38f5524533c`|
|`artifacts/checks/curriculum/04-localization-learning.json`|`2896307a4c01b6701d4ddd264562d2a329fe77c2d1bcad8b347d9481a8ab5a39`|
|`scripts/run_learning_extensions.py`|`b84e3317f95374df87d759e562db9b100ee14628b9cf5bb111238cd4a2e8e6cf`|
|`miniyolo/data.py`|`cccad00e2c4f96eb6567eafc9e12248379c6b715fc1790d75518a253baa6181d`|
|`miniyolo/geometry.py`|`6a6b57d3493888e99dae4a012dab78127b8b543a0e01d8107963a3d6e63d8483`|
|`miniyolo/targets.py`|`2c8e32f2845b3bf970c77304c5cca0f999083d36a4d5af2f75f14aa89b823f16`|
|`miniyolo/losses.py`|`81fa9331c9a2aebe2e9c6c453a5313e64566bb20c3fe77ca45ec9df53424d4e4`|
|`miniyolo/inference.py`|`995ac8f942d0c1e43d94f2efb3b7adcb91a9feb6b9bab923615209f0c190c313`|
|`miniyolo/metrics.py`|`53ce982e37cbd96c784f75c7d30faf99d52f79ab83ca7b8114eb21b4327330e0`|
|`miniyolo/train.py`|`aa5567f11be6ca7aee97bd082b9d5964414b12b73453763c915c7ffb409c4276`|

17張本頁使用的SVG均實際看過獨立render；01六張的label／圖位置／曲線點另逐項查數據。其他圖按逐頁紀錄核單位、數字、箭頭與框。完整Zensical desktop/mobile未驗證。

|圖檔|SHA-256|
|---|---|
|`docs/assets/diagrams/01-cnn-flow.svg`|`1eae350d8756238db0708a2865a13957dc13dd614a4e09d64f62b5f5e0f0bad5`|
|`docs/assets/diagrams/01-cnn-materials.svg`|`13e9618efcc9b18b43766dbf0c0872b14f05aca561e576d17c3107b95666a3f9`|
|`docs/assets/diagrams/01-convolution-window.svg`|`c8b8706ca476ea6725bea4fd32a6c293bf6a4e3aec5f1e10e8f9ae171b1bbc70`|
|`docs/assets/diagrams/01-receptive-field.svg`|`ae25c3c72023b673cbedc7911276a6c5cb192aa136264c27794b625153d48dae`|
|`docs/assets/diagrams/01-small-cnn-learning-readable.svg`|`33ff90b2fa990bfdc09b00f7378ef0a6ce956764c3eccd91706179ba3961463a`|
|`docs/assets/diagrams/01-small-cnn.svg`|`af5452951febc70c92c795372e4e1d5e37e1e6a66cccda9c403c4091dc3735a1`|
|`docs/assets/diagrams/02-diagnostics.svg`|`7e1e9773965b976e370f4dfbe5f4e46305d9c986f7cf661e6545f887c773b038`|
|`docs/assets/diagrams/03-comparison-learning.svg`|`c9a6051209f25ca5a2485eb85cc9bd83341292429f6e75b6988b8c00bf510552`|
|`docs/assets/diagrams/03-identity.svg`|`7d06f73907126b64113f69e28e64c2d62f0b5af6665cb31b1c5eafc68e803d4b`|
|`docs/assets/diagrams/03-projection.svg`|`257eb367cd3385b8364aea5b9dbda4d77f288c70c2d2c83a3c77eec764369b75`|
|`docs/assets/diagrams/04-coordinates-rounding.svg`|`883f55b09ba5bf1f9ef38a193ccdba89f2a8f4a339c9f8f9ad7516ffb20e4fef`|
|`docs/assets/diagrams/04-coordinates.svg`|`473e754e4a98108c0b34280886f25a3ad94b279830e2421a4a79480e53047acf`|
|`docs/assets/diagrams/04-localization-learning.svg`|`dc8c7a1d69ef0a22a6e8a399bed8349ddd5eacd7cef1f57a45c00cd2de388baf`|
|`docs/assets/diagrams/04-localization.svg`|`a4d35a3b11a2cae489f45a232a0343c73bfbb09aabbd1be4acacca9f567b700e`|
|`docs/assets/diagrams/05-assignment.svg`|`0bf373e60d870cd97fd1749ca0a93de204aeb0ba552f06fcfc3675f611925598`|
|`docs/assets/diagrams/06-decode-nms.svg`|`2e6d6cf9f26d95e6a3002b894596e11b8b1767211be038127acc73e39a82ff69`|
|`docs/assets/diagrams/06-evaluation.svg`|`8289113b070ec540923910c95227d2592d09545c0cb2e6ce215091b8ce8674fc`|

原始／官方來源下載內容指紋（實際讀的段落見R/D/E表）：

|本地來源檔|SHA-256|
|---|---|
|`vgg.pdf`|`83728f9efc21081792902b4c17a4657022d1e4a92ec81655c2427f4ef0755e50`|
|`resnet.pdf`|`1e0651b6810ecba34a3dbc5b5b0209226f889004607c1f203540a48d64e5a93a`|
|`nin.pdf`|`1fbe0f8b21cdbb83cc74bd96b54889171c13dfed1dcea89246baa00c3bb34f11`|
|`resnet-identity.pdf`|`676113014c130991d3a1203b151e1f2d4596f08898440c72270934a88d006c32`|
|`conv2d.html`|`27e6d332952cd1412babf32b0adaee1a95b606e54a0a193aac7f045a015a5a28`|
|`maxpool2d.html`|`8bd01dbd144aa807130bcf214abdb39220148fdb8528c542d4d3994d2a65500b`|
|`crossentropy.html`|`82d94828ca8f8403a8721d173aae9cb9f5fea5f6b27d5a28eddef9710b3f2f58`|
|`zero-grad.html`|`9bee03561d57f6fa1368cf918cfabdd589b0a16e7b1e89937216ce08d062b121`|
|`sgd.html`|`501c2cf704ace016366b9915a307915671d49883eb38f78d797c55026513b6c6`|
|`autograd.html`|`d0536f0e9a3679b027e3663405f0d3e2e3307f770a9c89320d79183aec65e15c`|
|`cocoeval.py`|`e514af401848d5a4cc5d3512bd93d2c317f92272c6d30f218e0f00b7b4110534`|
|`vocdevkit.tar`|`6101e33483e1f252821085f4b85634d334c1d44a0a5bc3921cd64320a40bd2cf`|
|`bag-of-tricks.pdf`|`5e07637c0f53db2e5a8671179c9442af21eb64fbfb58d1235d69c8b99dcb6744`|
|`yolov1.pdf`|`54bcd2dd05dc618849e8a94d8b88fe3eeb37f80e96e200600d38f1f733931678`|
|`torchvision-resnet.py`|`9faf2f321fdaf7edbd791aedc6e3610c7edf9c892112cbd75861beae1e9518d1`|
|`torch-conv.py`|`ba25b53dfc94b341b21d96ed628babe8204963454ca768c18d981452f2c61d9c`|
|`nms.html`|`430433bd22cd1de932944777cf819747e42bfe3b5e8638c15a29c4589720f686`|
|`VOCevaldet.m`（tar內實際檔案）|`c98716fd7f256af142d4362f3dc1d55f39339befa0c88ecf3aa83f786843bd78`|
<!-- fingerprints:end -->

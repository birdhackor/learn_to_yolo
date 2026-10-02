# 第一輪獨立初學讀者審查 E

審查日期：2026-10-02。讀者設定：只有基本 Python、PyTorch、NN／CNN 背景，不熟大學數學，對此專案沒有背景。依指定順序全文閱讀六節、各自完整 lesson case、notebook 的全部四格及已保存 output、各自 curriculum JSON；沒有閱讀舊 reviews 作為結論。唯一被這六頁引用的 SVG 是 `15-attention-bridge.svg`，已閱讀 SVG 原文並渲染檢視，其餘五頁沒有圖。

驗證使用既有 `.venv-model/bin/python`（Python 3.12.14、PyTorch 2.9.1+cpu）。六個原例重跑全部通過，stdout 與 JSON 完全一致；每個 case 的 SHA-256 與 JSON 紀錄相符，notebook 實驗格與 case 逐字相同，notebook 保存輸出與 JSON 相同。練習以記憶體內修改副本，在 `/tmp` 以 CPU 執行；沒有改教材、程式、圖或他人檔案，沒有使用 GPU、外部發訊或發布。notebook 環境格已逐行讀過，但沒有執行 git clone／pip；本審查不聲稱已驗證全新 Colab runtime 或尚待發布的 `lessons-v0.2.0` 標籤。

整體判斷：六節的 shape、數字和人工／短訓練證據界線可靠，指定練習都能完成。初學者主要會卡在術語入口、attention 的 softmax 中間步驟、參數計數和梯度手算；15.2 與 STAL 的空間分組需要圖。以下 P1 指對指定初學讀者值得本輪修正的理解缺口；P2 指可讀性改善，沒有發現 P0 數值或程式錯誤。六節各自的「前置」只列名稱，沒有可直接回補的教材連結；建議一律提供相關前節連結，並讓本節必要名詞用一句中文即可理解。

## 14-feature-module

讀過檔案：

- `docs/lessons/14-feature-module.md` 全文（1–71 行）。
- `lesson_cases/14-feature-module.py` 全文（1–76 行）。
- `notebooks/14-feature-module.ipynb` 全部 markdown、環境格、實驗格及 stdout。
- `artifacts/checks/curriculum/14-feature-module.json` 全部欄位。
- 本頁未引用 SVG，沒有可核對的路徑圖。

具體核對：輸入／輸出均 `[2,8,8,8]`；投影後 a、b 各 4 channels，b1、b2 各 4 channels；concat 為 `[2,16,8,8]`，融合 16→8。卷積含 bias 時，plain 為 `2×(8×8×3×3+8)=1168`；project 為 `8×8+8=72`，四個窄卷積為 `4×(4×4×3×3+4)=592`，fuse 為 `16×8+8=136`，總數 800。CPU 重跑 MSE `1.0722→1.0049`，直接 concat 槽梯度 L1 `[.2967,.3349,.3481,.2779]`；對應值／順序及每條總梯度斷言通過。

人工與实訓界線：`x.roll(1,-1)` 的單組合成特徵、30 次 SGD，不含圖片或框；plain 只計參數，沒有把未訓練 plain 當精度對照。b1 同時有直接 concat 和經 b2 的間接梯度，文件與程式分開檢查，這點清楚且正確。

練習實做：只改 `module = SplitAggregate(blocks=3)` 即通過；五份 4 channels，20→8，參數 1128，MSE `1.1248→1.0141`，五個槽的直接梯度檢查通過。不需要修改 assertion，操作說明可靠。

問題與建議：

1. **E14-1，P1，教材第5–7行／第11行。** residual、CSP、bottleneck、投影、hidden 一次出現，假設中的讀者只知道一般 CNN，無法確定「bottleneck」是什麼、為何這裡寬度是4。第11行直到開始算帳才説明兩個卷積和殘差。建議先補「residual＝保留原值再加上轉換結果；bottleneck＝本例的兩層窄卷積殘差小塊；hidden＝每條內部分支的channel數」，並連結 residual／CSP 前節。加一條 a、b、b1、b2 的路徑示意，尤其把 b 同時直接留下與送入第一個 block 畫出來。
2. **E14-2，P2，教材第38行／第50行。** 800／1128 的分項是對的，但沒有說 Conv2d 預設含 bias，讀者無法自行重算。加入通式 `out_channels×in_channels×kernel_height×kernel_width+out_channels`，再示範一個 4→4、3×3 的 `144+4=148`，即可補上算術缺口。
3. **E14-3，P2，教材第32行／第68行，case第54–64／72行。** `retain_grad()`、直接槽梯度、總梯度、L1 都是新層次；英文輸出 `gradient L1` 容易被誤讀成新增的 L1 loss。補「retain_grad讓中間tensor也保存反傳梯度；此处L1是把梯度絕對值加總，只用來確認非零，不是訓練loss」，並把進階路徑診斷放成可選閱讀。

## 15-attention-bridge

讀過檔案：

- `docs/lessons/15-attention-bridge.md` 全文（1–69 行）。
- `docs/assets/diagrams/15-attention-bridge.svg` 全文、渲染圖。
- `lesson_cases/15-attention-bridge.py` 全文（1–54 行）。
- `notebooks/15-attention-bridge.ipynb` 全部四格與 stdout。
- `artifacts/checks/curriculum/15-attention-bridge.json` 全部欄位。

具體核對：兩個 2×2 channels 攤平為左上 `[1,0]`、右上 `[0,1]`、左下 `[1,1]`、右下 `[0,0]`；shape `[1,2,2,2]→[1,4,2]→[1,4,4]→[1,2,2,2]`。第一query與keys的內積 `[1,0,1,0]`，除 √2 後為 `[.7071068,0,.7071068,0]`；softmax 分母 `2×exp(.7071068)+2=6.0562300`，得到 `[.3348808,.1651192,.3348808,.1651192]`，output `[.6697615,.5]`。SVG 各框 shape、來源／接收軸、每列加總1與教材一致；渲染可讀，箭頭沒有遮住字。

人工與实訓界線：identity初始化 Q/K/V、固定四tokens，MSE reconstruction target 即輸入feature；僅一次 SGD，loss `.1795` 是更新前計算值，程式只確認三份投影的非零梯度與整份權重改變。沒有偵測精度主張，界線清楚。

練習核對：案例已在 optimizer step 前用副本將第四token改 `[2,0]`。新softmax分母 `2×exp(.7071068)+1+exp(1.4142136)=9.1694803`；weights `[.2211810,.1090574,.2211810,.4485805]`、output `[1.3395231,.3302385]`，與文件和輸出四位小數相符。原資料 assertions 保留，練習可執行。

問題與建議：

1. **E15B-1，P1，教材第17–31行。** 這節承諾「一列一列算」，但由內積直接跳到softmax答案，只給 PyTorch 語法。數學不熟的讀者難以手算 `.3349`，也難理解練習為何必須改所有分母。建議補「內積＝對應元素乘後相加」的 `[1,0]·[1,1]=1×1+0×1=1`，及「softmax＝每個分數先取exp，再除這列exp總和」的上述分母／一項算式。說明 d＝每個query/key的特徵數，本例2。
2. **E15B-2，P2，教材第7／13／17／37／41行。** token、head、identity、row-major、affinity、FlashAttention 仍須讀者猜語意。至少補「token＝一個空間位置的channel向量；single head＝只做一組讀取；identity＝輸入值原樣輸出的線性初值；row-major＝逐列由左到右；affinity在此指每對位置的權重表」。FlashAttention 可標記為延伸名詞，並連結softmax／矩陣乘法回補內容。
3. **E15B-3，P2，教材第47行／notebook第3格。** 自主練習已把修改、答案、斷言全部加入原例；讀者按Run All就能看到答案，實際上沒有自己的操作。建議把「先算新第一列、再揭示答案」獨立成一個markdown練習提示，明確指出修改的是 `exercise_tokens[0,3]`，且必須在step前執行。現在的程式沒有可操作性故障，只是預測環節容易被跳過。

## 15-area-attention

讀過檔案：

- `docs/lessons/15-area-attention.md` 全文（1–66 行）。
- `lesson_cases/15-area-attention.py` 全文（1–55 行）。
- `notebooks/15-area-attention.ipynb` 全部四格與 stdout。
- `artifacts/checks/curriculum/15-area-attention.json` 全部欄位。
- 本頁未引用 SVG；前節 SVG 沒有顯示area分組。

具體核對：4×4位置按row-major编号0–15；A=4時 `[1,16,2]→[4,4,2]`，每區是一條水平帶。full weights `[1,16,16]` 共256，area `[4,4,4]` 共64，通式 `A×(N/A)²=N²/A` 相符。token0的Q是零，所有分數零，所以來源均勻；full平均 `7.5/16=.46875`，第一區平均 `1.5/16=.09375`。改token15每channel加5，full首token變化應為 `5/16=.3125`，area不變；程式斷言通過。原例一次SGD的MSE `.0048`。

人工與实訓界線：配對路徑共用同份identity QKV，區域範圍是唯一差異；沒有位置卷積、輸出投影、CNN對照或真實偵測訓練。文件沒有把64／256當完整模型延遲或GPU benchmark，也沒有把單層隔離說成整個YOLOv12不跨區。

練習實做：A=2得到 weights `[2,8,8]`、128 pairs、首token `.21875`（stdout `.2188`），改token15仍不影響首區。A=1得到256 pairs、首token `.46875`、干預影響full和area。A=4且changed_index=3時兩路都受影響。三種副本均不需修改assert，全部通過。

問題與建議：

1. **E15A-1，P1，教材第13–15／34–36／46行。** 本節最重要的「水平帶而非四象限」只靠文字，讀者要自己把0–15排成圖，再推斷區界與token15／3的位置。建議加4×4編號小圖，A=4用每列不同色，標出query0與changed15／3；另用A=2兩条8token帶，直接對照練習。這同時能核對reshape是否真的保留位置順序。
2. **E15A-2，P2，教材第9行。** M heads與 `[B×A,M,N/A,D]`、√D 在理解單head區域機制前就一次出現，且這份實驗沒有M軸。建議把本例 `[1,16,2]→[4,4,2]` 先講完，再放「多head延伸」；首次以中文說明每個head只是把channels分成多組，與area切位置是兩個不同軸。
3. **E15A-3，P2，教材第34行／case第31–38行。** 「query為零所以均勻」可以再补一行：四個來源分數全0，exp(0)=1，所以每個權重1/4。干預只印True/False；建议同时印full首值 `.46875→.78125` 与area `.09375→.09375`，让读者核对干预的数值效果，而不必只信assert。

## 16-dfl-free

讀過檔案：

- `docs/lessons/16-dfl-free.md` 全文（1–81 行）。
- `lesson_cases/16-dfl-free.py` 全文（1–67 行）。
- `notebooks/16-dfl-free.ipynb` 全部四格與 stdout。
- `artifacts/checks/curriculum/16-dfl-free.json` 全部欄位。
- 本頁未引用 SVG。

具體核對：K=16、座標0–15，四邊64 logits；全零logits的每bin機率1/16，平均 `(0+…+15)/16=7.5`。真值 `[1.25,2.5,18,3]` 格，stride8對應 `[10,20,144,24]` 像素，候選 `(80,80)` 解碼 `[70,60,224,104]`；18超過15。DFL對1.25的 `.75/.25` 權重期望值正確。Smooth L1初項 `.75+2+17.5+2.5=22.75`，平均5.6875；平均後梯度−.25，SGD `.0−.5×(−.25)=.125`。300步學到指定四數、loss<1e-6；帶符號 `[-1,2,3,4]` 解碼 `[88,64,104,112]`，寬16、高48，合法。raw輸出6600／600，float32每值4bytes，26400／2400。

人工與实訓界線：四個自由參數，不含feature/CNN head；未訓練DFL只演示初始化及有限表示，不是精度對照。Smooth L1不是官方完整CIoU＋正規化L1。正確強調DFL-free仍需定位監督，並把DFL-free與NMS-free分開。

練習實做：紙筆right=8落在[0,15]範圍。為驗證修改程式的可行性，副本同時把target18→8、初loss第3項17.5→7.5、超界assert改為涵蓋範圍、decode x2由224→144；全部通過，初loss3.1875，學得 `[1.25,2.5,8,3]`，框 `[70,60,144,104]`。帶符號例子原例已驗證。

問題與建議：

1. **E16D-1，P1，教材第5／11–15行。** 假設讀者無專案背景，DFL、bin、ltrb、stride、xyxy的必要語意沒有集中交代；逐邊解碼數字正確，但「格」指什麼仍可能不清。建議連結第12章DFL／距離解碼，並先加一行「ltrb＝從候選點到左、上、右、下邊界的距離；一格此例等於8像素；xyxy＝左上x,y和右下x,y」，搭候選點與四邊小圖。帶符號例子也用同圖標候選點在框外。
2. **E16D-2，P2，教材第30行。** 梯度−.25的結果可重現，但對不熟微分的讀者，為何負號、為何除4仍需猜。補「distance小於target時增加distance會使loss下降，所以導引為負；四邊mean使每項係數除4；SGD用舊值减lr×梯度」。這比只列函數分段式更能連上第一步 `.125`。
3. **E16D-3，P2，教材第53行／case第15／19／40／46行。** 修改target的練習已提醒同步assert，比盲改好，但沒有列出確切四處和新答案。建議列right=8的初項7.5／初loss3.1875／x2=144，以及 `target.max() <= k-1`；或者將它限定為紙筆練習，明確不要求改原例。其餘CIoU、正規化L1與候選資格擴張可標為官方延伸查證，不要讓讀者誤以為都得先會才可跑主例。

## 16-inference-head

讀過檔案：

- `docs/lessons/16-inference-head.md` 全文（1–78 行）。
- `lesson_cases/16-inference-head.py` 全文（1–82 行）。
- `notebooks/16-inference-head.ipynb` 全部四格與 stdout。
- `artifacts/checks/curriculum/16-inference-head.json` 全部欄位。
- 本頁未引用 SVG。

具體核對：输入 `[2,3,64,64]`，backbone `[2,8,4,4]`，兩head各 `[2,6,4,4]`，raw候選 `[2,16,6]`；部署top3框 `[2,3,4]`，score／label各 `[2,3]`。Conv3→8參數 `8×3×3×3+8=224`，head各 `6×8+6=54`，train332、deploy278，差54。stride16候選中心8／24／40／56。人工点 `(24,40)` 配 `[1,.5,2,1.5]` 解碼 `[8,32,56,64]`，sigmoid(0)=.5，sigmoid(ln3)=3/4，winner label1／score.75。保留one raw與reference allclose通過，deploy無many參數。

人工與实訓界線：一次raw平方mean更新没有物件標註／assignment；detach只是切断one到backbone的梯度。deepcopy保留更新後one與backbone，輸出相等不代表NMS-free模型學到唯一偵測。top3會輸出背景候選、官方雙階段top-k與本例單標籤有所不同，文件明確，沒有誤導。

練習實做：副本把 `scores.topk(3,dim=1)` 改5，主shape assertion同步 `(2,5,4)`／`(2,5)`，得到boxes `[2,5,4]`、scores／labels `[2,5]`；參數仍332／278，其餘斷言通過。128尺寸練習未執行，紙筆重算stride32與中心16／48／80／112可知原stride16不適用。

問題與建議：

1. **E16I-1，P2，教材第13／42–44行，case第34／49行。** softplus是此例距離的必要運算，卻沒有定义；人工raw透過 `log(expm1(distance))` 產生，又讓不熟數學的讀者多遇到一個反函數。建議補「softplus把任意raw數轉為正距離，公式log(1+exp(raw))；人工raw只是倒算，確保softplus後正好得到指定距離，不需把倒算公式當成模型學得的東西」，並讓該技術段可展開閱讀。
2. **E16I-2，P2，教材第56行／case第40–41／72行。** top5練習可做，但`gather`、`ids[...,None].expand`是在選框的重要代码，沒有解释。补shape帳本：ids `[B,k]`、扩张为 `[B,k,4]`，在候选轴取每框四座標；再明列需要改topk與shapeassert兩处。這能讓讀者理解選的是哪個維度，不只是改3為5。
3. **E16I-3，P2，教材第9／24／26–38行。** 雙head、detach、複製、部署只保留one的关系全靠散落段落。加简单圖「backbone→many（反傳到backbone）；backbone→detach→one（只改one）；deploy=backbone+one」，并标明raw比對发生于解碼前，可降低将train dict和deploy tuple混读的门槛。

## 16-training

讀過檔案：

- `docs/lessons/16-training.md` 全文（1–84 行）。
- `lesson_cases/16-training.py` 全文（1–72 行）。
- `notebooks/16-training.ipynb` 全部四格與 stdout。
- `artifacts/checks/curriculum/16-training.json` 全部欄位。
- 本頁未引用 SVG。

具體核對：E=30、e=0–29，many／one從 `.8/.2` 到 `.1/.9`；e=15為 `.437931/.562069`，符合「中間約.438/.562」。features `[4,1]`、targets `[-1,1,3,5]`。one初w=.3184233904、bias=.3137805462，forward四數與表一致；errors `[.9953572,−.6862195,−2.3677961,−4.0493727]`，MSE5.866377；已乘.2之dw=−1.146190、db=−.610803，lr.05更新至 `.375733/.344321`。30次完成後重forward，fixed MSE `.663073`、progressive `.016285`。STAL真值 `[7,7,9,9]` 是2×2，中心8／8，擴為边长16資格框 `[0,0,16,16]`，四格心 `(4,4),(12,4),(4,12),(12,12)` 的资格0→4；回归GT不变。

人工與实訓界線：两个Linear独立、相同target、没有shared backbone或assignment，纯SGD里可解释为不同有效learning rate；不会证明YOLO26 AP增益。STAL只检查资格，不训练；MuSGD完全没运行，官方预训练／内部recipe也没复现。此三种范围在正文与stdout均正确声明。

練習實做：只改 `main()` 的 final_many `.1→.3` 即全部通过；首步相同，末many／one `.3/.7`，one MSE `.043999`，仍低於fixed `.663073`。一对一top1的资格池后每GT至多1个，不能由四点资格回答4；文件答案正确。

問題與建議：

1. **E16T-1，P1，教材第15–17／32–42行。** 首步表让读者核对数值，却从MSE跳到dw/db，非数学背景无法自行推出梯度。建议先中文解释「b是给one loss的音量旋钮」，并展開一次：`dw=.2×(2/4)×sum((prediction−target)×x)=−1.146190`，`db=.2×(2/4)×sum(prediction−target)=−.610803`。解释2来自平方loss、4来自四笔平均、.2来自分支权重，再接现有SGD更新。`∂L/∂θ`可作为同一句的数学记法，不要先用记号当说明。
2. **E16T-2，P1，教材第48–52行。** STAL资格、GT、stride门槛、ranking、top-k、冲突连在同段，且关键0→4没有空间图。建议用像素图同时画原2×2真值、16×16资格框和四个格心，标「进入候选池，尚未分配」。首次解释GT＝标注真值框；ranking＝按预测品质排序；后续选择／冲突解决不在本例执行。加图后仍保留原回归框不变的检查。
3. **E16T-3，P2，教材第56–60行。** Muon、矩阵更新几何、Newton–Schulz、参数分组、权重衰减、Objects365／COCO等远超本节可动手验证的范围。已有「未执行」声明应保留；建议把该段標为延伸查证，可跳过而不影响主例，并先给直白目标：「除了沿梯度走，还会整理矩阵权重的更新方向；本节只介绍区别，不教实现」。不需要为了填满篇幅加入未经验证的缩版优化器。

## 本輪修正後建議複查的項目

优先补 E15B-1 的softmax算式、E15A-1 的分区图、E16T-1 的首步梯度、E16T-2 的资格图，以及 E14-1／E16D-1 的入口定义与前节链接。若新增SVG，逐值核对token编号、区域边界、GT/资格框和格心；若改case，重新核对notebook完整实验格、JSON case hash及输出。不要把当前六个CPU原例通过当作Colab标签或新runtime通过，待本轮教材修正与读者複查完成再发布 `lessons-v0.2.0`。

## 修訂後獨立複查（2026-10-02）

保留以上第一輪初讀，不以作者的「已補」通知當作通過。重新全文讀六個lesson頁，重新核對六個case／notebook／JSON；`15-area-attention.py` 的變更全文重讀並CPU重跑。其餘五個case雜湊與第一輪相同，沿用第一輪完整閱讀與實跑結果，另檢查當前notebook實驗格逐字相同、已存output與JSON相同、JSON雜湊指向當前case。全部一致，沒有新數值或shape錯誤。六個notebook的非實驗格也重新讀過；沒有執行Colab初始化／發布。

新增SVG原文及目視渲染均已核對：`14-split-paths.svg`、`15-area-layout.svg`、`16-distance-decode.svg`、`16-head-paths.svg`、`16-stal-candidates.svg`。渲染副本只寫在`/tmp/reader-e-*.png`，沒有改正式圖。原`15-attention-bridge.svg`仍與第一次核對的四位置shape路徑一致。

逐項複查：

| 節次 | 修訂位置／核對內容 | 初讀問題的狀態 |
| --- | --- | --- |
| 14 | 教材第5行加residual／CSP連結及bottleneck／hidden定義；第36行含bias計數公式、retain_grad和gradient L1的解釋，4→4卷積148、800／1128總參數仍正確。新split圖顯示b先留下也送進F1、b1先留下也送進F2，四份4ch→16ch→8ch。 | E14-1主要入口已補、E14-2／3已補；新圖有一項槽順序歧義，見R-E1。 |
| 15 bridge | 第19行定義token／single head／identity／row-major／affinity；第23行展開內積、d與exp分母6.0562，2.0281／6.0562=.3349、1／6.0562=.1651正確；第53行指明exercise_tokens[0,3]、step前及先手算。 | E15B-1／2已補；E15B-3操作已明確，答案仍先直接寫在第49行，可再把先預測提示放到答案前，屬P2編排改善。 |
| 15 area | 新圖A4逐列0–3／4–7／8–11／12–15，A2為0–7／8–15；query0、changed3／15框線及來源區域吻合。第52行補零query→exp(0)=1→各權重1/4，full .46875→.78125、area .09375不變；多head shape移到第54–56行可跳讀折疊。 | E15A-1／2／3已補，新增干預輸出與實跑一致。 |
| 16 DFL | 第5行補四邊距離／DFL連結及ltrb／xyxy／格單位；距離圖的藍框寬308、高88，畫面每單位對應2倍像素，l/t/r/b分段為20／40／288／48，正好是10／20／144／24px，點及框數字吻合。第59行補負梯度、mean除4及SGD方向。 | E16D-1／2已補；E16D-3的right8預期數字正確，但四處修改清單尚須與實際程式對齊，見R-E2。 |
| 16 inference | 第11行softplus公式、人工raw倒算而非學得結果、ids[B,k]→[B,k,4]沿候選軸gather與top5 shape說明正確。新图many路徑可回傳backbone、one路徑有detach，deploy只複製backbone＋one再decode/top-k。 | E16I-1／2／3已補。圖的前向箭頭和反傳文字不混淆，框、分數、類別的回傳介面未變。 |
| 16 training | 第42行dw=.2×(2/4)×sum(error×x)、db=.2×(2/4)×sum(error)與首步−1.146190／−.610803吻合；第48行GT／ranking釋義。STAL圖的紫框256×256對16×16、綠框32×32對2×2，中心與四格心相對位置正確，明標進池未分配。Muon第58行改可跳讀、第60行說明不需實作。 | E16T-1／2／3已補；沒有把STAL候選池數量當正樣本數量，沒有新增MuSGD驗證主張。 |

變更case的實跑與練習：`15-area-attention`預設全部assert通過，stdout與2026-10-02T17:08:10的JSON完全相同，hash為`d50234ad372f98f5bbdb784e7bb38afae26055ff8eedf762872f317a8edee71a`。新增stdout的full=.7812、area=.0938與未四捨五入的.78125／.09375相符。再次用記憶體副本做A2、A1、changed_index3均通過：A2干預後full .78125／area .21875；A1兩路皆 .78125；idx3且A4時full .78125／area 1.34375（第一區增加5/4），符合來源個數。

複查新增或仍需處理的項目：

1. **R-E1，P2，`docs/assets/diagrams/14-split-paths.svg`第7／12–15行。** 四個入concat箭頭由上到下是a、b1、b2、b；實際`paths=[a,b]`再append b1、b2，槽順序是a、b、b1、b2。這張圖是路徑而非正式槽表，因此不構成參數／shape錯誤，但新手可能沿高度讀出錯順序。建議加「concat槽順序a,b,b1,b2；箭頭高度只表示路徑」或明列槽，避免與正文的逐槽順序檢查混淆。
2. **R-E2，P2，`docs/lessons/16-dfl-free.md`第59行。** 「同步四處」列入了自動mean產生的平均loss3.1875，卻把必改range assertion放到後一句；反引號寫`K`而case變數為小寫`k`。建議四處明列：target第三項8、initial_terms期望第三項7.5、`target.max()<=k-1`、decoded x2=144；平均loss3.1875是自動計算的核對值。公式層面大寫K本身沒錯，但這裡是可修改程式清單，应與code變數一致。
3. **殘餘P2編排建議。** bridge的先預測提示第53行、area的零query手算第52行、DFL的負梯度第59行目前皆在來源查覈段之後；讓讀者先遇到答案再遇到引導。可把提示移到首次練習答案之前、數學補充移到對應計算段。其餘四頁的前置仍只有名詞，若方便可補前節連結；這些不影響本次已核對的數字與操作。

本次結論：第一輪P1理解缺口已實質補齊，新增圖與數值能支持基本CNN背景讀者理解機制。沒有新增P0／P1內容錯誤；建議完成R-E1／R-E2的兩個P2小修再收束本輪。全新Colab與標籤發布仍未由本審查驗證。

## 定點關閉複查（2026-10-02）

只追加本段，不覆寫兩次閱讀紀錄。獨立重讀兩處修訂並核對實際檔案：

- **R-E1已關閉。** `docs/assets/diagrams/14-split-paths.svg`第17行明列「串接槽順序：a、b、b1、b2；入線高度不代表槽順序」，與case的paths順序相符。重新渲染並目視：底部文字完整可讀，未被裁切或箭頭遮住；四條路的拓撲及4×4→16→8數字維持正確。
- **R-E2已關閉。** `docs/lessons/16-dfl-free.md`第59行列出target[0,2]→8、initial_terms期望第3項→7.5、範圍assert改`target.max()<=k-1`、decoded x2→144；另說mean自動變3.1875，與case第15／19／40／46行逐項對齊，變數使用小寫k。按這四處在記憶體副本修改，於`/tmp`用既有CPU環境執行，全部assert通過；初loss3.187500、學得右距離8、解碼框[70,60,144,104]。未修改正式case、notebook、圖或他人檔案。

本審查已無需本輪修正的未解決項目；先前列的提示放答案前／補前置連結仍是可選P2編排建議。這個關閉狀態只涵蓋教材可讀性、圖與數值及本地CPU練習，不等於全新Colab或標籤發布驗證。

# 第 11～13 章新讀者初讀報告（reader D）

審閱日期：2026-10-02。讀者假設：只懂基本 PyTorch、NN、CNN，數學不熟，不知道這個專案的歷史。依指定順序完整閱讀八節正文，再逐節對照 case、notebook 全部四個 cell、保存的輸出和 curriculum JSON。未用既有 reviews 作結論。`lessons-v0.2.0` 視為準備發布的版本，沒有把尚待發布當教材錯誤。

本文只記初读所得與修改建議，沒有修改教材、程式、notebook、SVG 或 artifacts。相關 SVG 只有 `11-iou-loss.svg`；讀了完整 SVG 原始碼，並以本地 Chromium（禁用 GPU、外部請求）渲染後目視。

## 執行與共同核對

- 使用現有 `.venv-model/bin/python`、PyTorch `2.9.1+cpu`，八個原始 case 均 exit 0；本次 stdout 與各自 `artifacts/checks/curriculum/<id>.json` 逐字一致。沒有 GPU 訓練或外部發訊息。
- 八份 JSON 的 `case_sha256` 均與目前 case 相符。八份 notebook 的實驗 cell 3 均逐字等於 case，保存的 stdout 也逐字等於 JSON。不能把這種一致性解讀為在新 Colab runtime 完成驗證。
- 完整讀過各 notebook 的環境 cell 1：固定 `REF='lessons-v0.2.0'`、跳過 LFS smudge、檢查 exact tag、Torch `2.9.1`、numpy/pillow/matplotlib 版本、`os.chdir` 和 `sys.path`。這輪沒有執行遠端 clone/install，也沒有實測 Colab。
- 初次誤用系統 Python 因未裝 torch 而無法執行；換現有 CPU venv 後全部成功。這不是教材錯誤。
- 自主練習以 `/tmp/curriculum-reader-d` 為工作目錄、`PYTHONPATH=/workspace/learn_to_yolo`，在記憶體中改寫 case source 後執行，沒有覆寫專案檔案。
- 沒發現八節原題數學答案或已保存輸出錯誤。最需要修的是三節「照文字修改輸入卻撞到原題 assertion」：`12-decoupled-head`、`12-dfl`、`13-nms-free`。其他建議以補讀者橋接和明確界線為主。

## 1. 11-augmentation

實際讀過：`docs/lessons/11-augmentation.md` 全文（1～59 行）、`lesson_cases/11-augmentation.py` 全文（1～61 行）、`notebooks/11-augmentation.ipynb` 的四格及保存輸出、`artifacts/checks/curriculum/11-augmentation.json` 全部欄位。本節沒有引用 SVG。

具體核對：

- 圖像 `[3,64,64]`、框 `[1,4]`、labels 為 long `[1]`。紅色畫素是 `image[0,12:28,8:24]`；flip 後像素區域和框都落在 `[40,12,56,28]`，雙 flip 恢復。
- Crop offset `[16,8,16,8]`：先得到 `[-8,4,8,20]`，再裁到 `[0,4,8,20]`。面積 `8×16=128`，原面積 `16×16=256`，比例 `.5`；`.5` 留下 class0、`.6` 留下 boxes `[0,4]` 和 labels long `[0]`。
- Flip 後中心 `(48,20)`，64/4=16 畫素一格，`floor(48/16)=3`、`floor(20/16)=1`，正文的 cell `(x3,y1)` 正確。
- 空圖確實用全零圖搭配空框、空 labels，不是有紅物件卻漏標註。沒有 backward 或 detector 訓練，正文已正確說明。

問題與修改建議：

1. **建議補可直接執行的初始化**：正文 13～17 行用了 `new_boxes`，卻沒有建立它；新讀者複製會 `NameError`，也看不到如何避開第 42 行提醒的原地改寫錯誤。可在 snippet 加 `W=image.shape[-1]` 和 `new_boxes=boxes.clone()`。27～32 行的 `clip_to_crop` 在實際 case 並不存在，應標「示意碼」或把兩軸 clamp 寫出來。
2. **建議補前置詞的最短解釋**：第 7 行「bag of freebies」「Mosaic」「mixup」「固定 budget」，第 34 行「letterbox」「targets」對只懂 CNN 的讀者沒有入口。尤其 letterbox 影響框的單位，建議寫「等比例縮放後補邊，框也乘縮放比例並加補邊偏移」，並把資料契約/target 建立處連結附在該句旁。
3. 第 42 行自主題的幾何答案正確，但沒說如何驗證整張框；在 notebook 的 `main()` 外另呼叫 `horizontal_flip(image, torch.tensor([[0.,0.,64.,64.]]))` 比改原題 `boxes` 更容易完成，也避免原題像素/框 assertion 干擾。這是操作性建議，不是答案錯誤。

## 2. 11-iou-loss

實際讀過：`docs/lessons/11-iou-loss.md` 全文（1～64 行）、`lesson_cases/11-iou-loss.py` 全文（1～77 行）、`notebooks/11-iou-loss.ipynb` 四格及輸出、`artifacts/checks/curriculum/11-iou-loss.json`；完整讀 `docs/assets/diagrams/11-iou-loss.svg` 並看過本地渲染圖。

具體核對：

- G/P 各 16×16、面積256、union512；C 是40×16、面積640。`L_GIoU=1+128/640=1.2`；左移1 px後 `2-512/624=1.179487179…`。
- `L_DIoU=1+576/1856=1.310344827…`。本例比例相同所以 `v=0`、CIoU 同 DIoU。xyxy MSE=`(576+576)/4=288`，是有尺度的數字，不宜與無單位比例直接比效果。
- GIoU x 中心梯度 `.02`；SGD lr50 使40→39。DIoU x 梯度約 `.0124851367`，lr100 使中心到 `38.751486…`，CPU float32 輸出 `38.7514877319` 合理。左移4 px的 `1+400/(36²+16²)=1.257731959…` 正確。
- SVG 紅/藍框寬高各160顯示單位、相距80；C 寬400、高160、中心相距240，與圖上「10倍 pixel 間距」相符。ρ、c²、union、C面積和左移1 px後數字都對上，字無截斷。
- 真實更新的是人工框的兩個中心參數，寬高固定，不是 CNN detector。正文 7、40、44 行有清楚界線。

問題與修改建議：

1. **建議補公式閱讀橋接**：第 25～27 行第一次出現 `ρ²`、`c²`、`αv`、`atan`，数学不熟的讀者能代入576/1856，卻不容易知道每個符號在測哪件事。可先逐字說「ρ是兩中心直線距離、c是C的對角線長度；平方後相除使比例不受像素尺度影響」，再說 `atan(w/h)` 把長寬比變成角度來比較。圖中可把 c² 附上 pixel²，與正文單位一致。
2. **建議把第 27 行 detach 的作用說成一條梯度路徑**：目前只說常見實現選擇；可補「本次把α當固定權重，仍對v和中心項求梯度」。本例 `v=0`，所以其實沒有驗證非零長寬比項的梯度，應避免讓讀者以為四種loss的所有部分都已測到。
3. **建議明寫合法框前提**：第 46 行「沒有epsilon導致零框除0」容易讓讀者認為本函式已處理任意退化框。case 18 行的 `twh[0]/twh[1]` 沒有保護真值高度0；本例只使用正寬高有效框。可在 `losses()` 前說「輸入GT必須已驗證x2>x1、y2>y1；clamp是數值保護，不能修復無效標註」，不需把本節擴成通用幾何庫。
4. 第 46 行兩題可直接透過 `losses(target,target)`、`losses(box_from_center(torch.tensor([36.,20.])),target)` 操作；原 case 也已驗證兩題，操作性良好。可把呼叫方式附在題旁。

## 3. 12-anchor-free

實際讀過：`docs/lessons/12-anchor-free.md` 全文（1～74 行）、`lesson_cases/12-anchor-free.py` 全文（1～43 行）、`notebooks/12-anchor-free.ipynb` 四格及輸出、`artifacts/checks/curriculum/12-anchor-free.json`。本節沒有 SVG。

具體核對：

- stride8 cell `(2,2)` 中心 `(20,20)`；人工點 `(24,24)` 不是此cell中心，正文已明確指出，沒有混淆。
- 點到真值 `[12,16,40,36]` 的 px 距離 `[12,8,16,12]`，格單位 `[1.5,1,2,1.5]`；乘8並減左上/加右下，完全解回原框。
- 完整 `[B,P,4]` 契約與case實際 `[P,4]=[1,4]` 已區分。框外点 `(48,24)` 的右距離 `40−48=-8`，與正距離表示衝突。
- 原題 loss `0.376236→0.000012`。依第54行完整修改：point `(20,20)`、stride4、target `[2,1,5,4]`、所有decode同步，80步即通過；實測 loss `1.866909→0.000013`、學得框約 `[12.01,16.03,39.98,35.99]`。

問題與修改建議：

1. **建議補 softplus/Smooth L1 的最短定義**：第34～44行直接從「logits從0」跳到「距離可學」，新讀者不知道0為何變成.693、Smooth L1對什麼做平均。可補 `softplus(z)=log(1+exp(z))`，是可微且正的轉換；本例 Smooth L1 比較四個「格單位距離」，不是直接比較像素框。
2. **建議更直接明寫更新对象**：第34行「四個可學logits」與前面的密集head契約相接，讀者可能以為正在訓練影像模型。加一句「本次raw就是四個獨立nn.Parameter，沒有輸入影像/CNN；這只證明表示可被loss拉向一個target」，能讓“真更新”和“真偵測訓練”邊界更易辨認。
3. 第54行自主題對所有需同步修改的位置交代完整，實測可操作；此節沒有需修的練習障礙。

## 4. 12-decoupled-head

實際讀過：`docs/lessons/12-decoupled-head.md` 全文（1～70 行）、`lesson_cases/12-decoupled-head.py` 全文（1～55 行）、`notebooks/12-decoupled-head.ipynb` 四格及輸出、`artifacts/checks/curriculum/12-decoupled-head.json`。本節沒有 SVG。

具體核對：

- x `[2,3,4,4]`→共享特徵 `[2,8,4,4]`→box `[2,4,4,4]`、class `[2,2,4,4]`；raw box經softplus才是正距離。所有位置都是人工正樣本、無assignment，正文已有說明。
- 含bias：backbone224、兩支卷積各584、box投影36、class投影18；合計 `224+1168+36+18=1446`。直接1×1 `8→6` 為54個head參數，不與整個模型1446混算。
- 分別backward的路徑與第三次共享梯度相加符合計算圖；原題cosine `-.0073`接近0，不能據此證明任務衝突或效果提升。

問題與修改建議：

1. **需修自主題操作指示（第50行）**：「只將分類loss乘2，再驗證總backbone gradient」若只改case42行為 `(box_loss+2*cls_loss).backward()`，會在43行仍比較 `gbox+gcls` 而 `AssertionError`。我已實做重現。請明列「總backward改為Lbox+2Lcls、assert改為box_backbone_grad+2*class_backbone_grad，印出文字也同步改」；如果先把 `cls_loss` 本身乘2再測分項，則保存的gcls已含2，須說清楚採哪一种。
2. **建議補與耦合基線的詞義**：第50行「對照coupled head」沒有定義其結構。可說「共享最後一串卷積，再一次輸出box和class」；目前“先決定對齊成本”很合理，但只懂基本CNN的讀者缺少可開始寫出的基線。
3. 第36行的cosine解釋已給±1和0的直觀意義，做得足夠；可再說 `.flatten()` 是把共享權重的各元素排成向量，避免把這個值誤看成分類輸出cosine。

## 5. 12-assignment

實際讀過：`docs/lessons/12-assignment.md` 全文（1～70 行）、`lesson_cases/12-assignment.py` 全文（1～70 行）、`notebooks/12-assignment.ipynb` 四格及輸出、`artifacts/checks/curriculum/12-assignment.json`。本節沒有 SVG。

具體核對：

- 品質/IoU表 `[G,P]=[2,4]`，不是class輸出的 `[P,C]`；正文說明score已按GT class索引，且是獨立sigmoid分數。
- p1對A：交256、union384，IoU=2/3；對B：交192、union448，IoU=3/7。p2對A `96/(320+288-96)=.1875`，對B `.9`，幾何全部相容。
- A品質 `.9×.5²=.225`、`.7×(2/3)²=.311111…`；B `.8×(3/7)²=.146938…`、`.7×.9²=.567`。p2對A雖非零IoU，點在A外，mask後品質0。
- top2後p1衝突依IoU交給A，owner `[0,0,1,-1]`；mean BCE梯度 `[−.125,−.125,−.125,+.125]`。空GT回傳全−1，沒有假裝測了空正樣本框loss。
- 第49行干預p1框→GT B時，IoU對A=.25、對B=1，品質 `.04375/.8`，owner `[0,1,1,-1]`；case結尾已跑此題，操作性良好。

問題與修改建議：

1. **建議補新讀者最關鍵的mask步驟**：第20行雖已列資格和數字，但可直接寫「p2框和A有重疊，參考點卻在A外，所以A的p2品質被置0」。這會把“點的資格”和“預測框的IoU”兩套幾何連在一起，而不是讀者自行比對表格。
2. **建議連結兩種衝突規則**：這節第24行按最高IoU解衝突，下一節13-dual用最高quality。兩者都是明寫的教學選擇、不是數值錯誤；可在此或下一節提醒「下節的quality表實驗改用品質解衝突，不是直接沿用本函式」，避免讀者以為assignment有唯一通用argmax規則。
3. 第37行已清楚揭示固定score與可學foreground logits是兩组數，這一點非常必要。仍可補 `.125` 更新後的logit數字（lr1：正為+.125、背景−.125），讓“target→梯度→更新”一路不需額外猜。

## 6. 12-dfl

實際讀過：`docs/lessons/12-dfl.md` 全文（1～62 行）、`lesson_cases/12-dfl.py` 全文（1～51 行）、`notebooks/12-dfl.ipynb` 四格及輸出、`artifacts/checks/curriculum/12-dfl.json`。本節沒有 SVG。

具體核對：

- 10px/stride8=`1.25`格；bin1/2權重`.75/.25`，加權平均1.25。四bin初始均`.25`、期待值1.5、loss=`ln4=1.386294…`、gradient=`[.25,−.5,0,.25]`。
- 四邊head `[B,P,4K]` reshape `[B,P,4,K]`，softmax最后bin軸；K16×stride8的最高刻度15格=120px，raw輸出600對6600個數，計算正確。有限softmax logits只逼近範圍端點；正文用“最多”是合理上界。
- 兩分佈 `[0,.75,.25,0]` 與 `[.375,0,.625,0]` 期待值皆1.25；不表示它們的DFL相同，也沒有冒充校準好的不確定性。
- 400步實測原題probs約`.0025,.7475,.2474,.0025`，期待值約1.25格、9.9999px。它是在直接更新單邊四logits，不是完整detector。

問題與修改建議：

1. **需修自主題同步修改（第42行）**：只把case24行target改2.6，case30行仍用原梯度assert而失敗。我已重現。即使改wanted_grad，case40行仍比較1.25，41行仍要求bin1>.73，也會再失敗。請列明新wanted_grad `[.25,.25,−.15,−.35]`、期待值比較2.6、改查bin2/3權重；可用目標權重的容差驗證，不讓讀者自行猜所有舊常數。
2. 同步修正後，400步得到prob約`[.0025,.0025,.3975,.5975]`、期待距離`2.5899`格=`20.7192`px，距2.6小於.02。最後仍印「different distributions ...1.25」是獨立反例，不是當前target的結果；建議替這行加“另一个固定反例”，避免做完練習以為還有漏改。
3. **建議修術語**：第7行「分散式四邊回歸」易被讀成distributed computing。改為「以分佈表示的四邊回歸」。第一次bin可寫「距離刻度（bin），0、1、2、3各代表0、1、2、3格」，把“距離不是物件類別”提前到公式之前。
4. 第24行「p−target分佈」是核心推理；可直接列target分佈`[0,.75,.25,0]`再做逐項相減，讓數學不熟的讀者也能自己重建gradient，而不只是信任答案。

## 7. 13-dual-assignment

實際讀過：`docs/lessons/13-dual-assignment.md` 全文（1～70 行）、`lesson_cases/13-dual-assignment.py` 全文（1～61 行）、`notebooks/13-dual-assignment.ipynb` 四格及輸出、`artifacts/checks/curriculum/13-dual-assignment.json`。本節沒有 SVG。

具體核對：

- 品質 `[2,3]`；many top2的A→p0,p1、B→p0,p2，p0按quality交A，owner `[0,0,1]`。
- 不同欄配對共6種，總品質依序可為1.00、1.10、1.73、1.05、.98、.20。最優A→p1/B→p0=`1.73`，owner `[1,0,-1]`；指定貪心路徑為1.10。
- 輸入 `[3,4]`→features `[3,4]`→兩個logits `[3,2]`；two-class targets與owners一致。這裡GT id剛好等於class id，正文已說兩GT恰好兩類，不能推廣成GT index就是類別。
- `features.detach()`只切斷one分支到backbone的梯度，one head仍更新；many更新backbone。這是真實小網路一步更新，不是實際YOLOv10 detector訓練或NMS-free效能。
- 按第50行完整操作改B p2=.95、one owner `[0,-1,1]`、最優與貪心相等assert後實測通過；兩者1.85，many仍 `[0,0,1]`。練習比其他節清楚。

問題與修改建議：

1. **建議減少global matching對主題的干擾**：第7、27行已多次說不是官方assigner，但第27行「一般需要真正的最優指派solver」仍可能讓新讀者以為放大YOLOv10就需要這種solver。改為「若要把本例『總品質最大』問題放大，需要最優指派solver；YOLOv10採另一種top1規則，沒有這段全域搜尋」。
2. **建議加一行consistent的範圍**：第7行說“相容的品質排序”，本文卻大篇幅比較global optimum和greedy。可說「本例共用同一quality表；它只能示意一致的評分來源，不重現官方指數相容條件」。這能對上第46行歷史設計的動機。
3. **建議在code前列出六種配對或一句排列解釋**：第23行 `permutations` 到第27行 `P!/(P−G)!` 是数学跳躍。可先寫「A有3個欄可選，B只剩2欄，因此3×2=6」，再把階乘公式列作延伸。大多數新讀者理解dual分支不需先懂Hungarian。
4. 第40行「一次step更新兩個head」在case中僅明確assert了one head權重變化，many head雖確實在optimizer且有loss路徑，未逐值檢查。若要把“兩個head均驗證更新”當執行證據，需另外保存並比較many head；否则可縮句為“總loss送入optimizer；case核對one head确實更新”。

## 8. 13-nms-free

實際讀過：`docs/lessons/13-nms-free.md` 全文（1～67 行）、`lesson_cases/13-nms-free.py` 全文（1～56 行）、`notebooks/13-nms-free.ipynb` 四格及輸出、`artifacts/checks/curriculum/13-nms-free.json`。本節沒有 SVG。

具體核對：

- p0/p1交90、union110，IoU=`9/11=.818181…`；一對多target `[1,1,1,0]`，一對一 `[1,0,1,0]`，100步相同SGD後scores正約.959、負約.041。
- threshold .5：many無NMS `[0,1,2]`；加NMS剩2框且當前輸出 `[0,2]`；one無NMS `[0,2]`。人工scores top2 `[0,1]` 確實漏B。
- 手算TP/FP序列TP、FP、TP，precision `1,.5,2/3`、recall `.5,.5,1`；all-points interpolated AP=`.5×1+.5×(2/3)=5/6`。這段AP是人工別組scores的手算，程式沒有呼叫AP計算，正文也已區分。
- 本節僅四個直接可學二元logits、已知固定框，不接CNN、沒有雙head特徵共享或assignment選擇；正文第7行對界線說得清楚。

問題與修改建議：

1. **需修自主題操作指示（第47行）**：把one-target改`[0,1,1,0]`，實測在case44行仍要求one_ids `[0,2]`而失敗；新答案應為 `[1,2]`，要同步改assert。one高分框p1對A仍有IoU `.8182`，不是完全對齊GT，題目的結論正確。
2. **需修threshold題操作指示（同第47行）**：把case39、40行`.5`改`.99`，得到空候選列表，但case43行仍要求many為3/2框而失敗。應把score threshold集中為變數，再告訴讀者同步檢查三條路徑均空；不要改 `nms(...threshold=.5)` 的IoU threshold，兩個門檻意義不同。在這個固定100步實驗是確定全部過濾，不只是“可能”。
3. **建議精確對齊“檢查”的宣稱**：第31行「本例檢查框數和物件覆蓋」；case43行對many/NMS只有框數assert，未直接驗證B的p2留下、A的p0/p1至少留其一。演算法和當前輸出確實有覆蓋；若要聲稱程式核對覆蓋，可補 `2 in many_after_nms` 與 `{0,1}`至少一個被保留，而非只靠2個框。
4. **建議把評估門檻名稱寫全**：第13行NMS IoU threshold、第31行score threshold、第35行GT matching IoU threshold同為.5，容易讓新讀者以為一個`.5`共用所有步驟。可用三個明確名稱或小表列「刪低分／刪重複／判TP」；這直接有助於完成第47行練習。

## 初讀後的修訂優先順序

1. 補齊三節自主練習的同步assert/輸出修改，讓讀者照題目操作可到達新答案。
2. 補softplus、DFL bin/target向量、IoU符號的短橋接；避免僅有答案和公式而缺可重建的步驟。
3. 把global matching、人工直接參數、退化框前提和“程式已檢查”的範圍寫得更精確。現有原題數字、shape和歷史/效果界線已大致一致，不需推翻教材主線。

## 獨立修訂複查（2026-10-02 17:13 UTC）

以上初讀內容完整保留。此段是讀取根 agent 修訂後的當前檔案，再獨立執行練習的結果；不把「已寫補充說明」直接當作練習已通過。這輪重新讀了八節當前全文，逐項回看原報告的建議，讀了變更的 NMS case/JSON 及其他相关 case，重新核對八份 notebook code、保存輸出、JSON/hash。SVG 沒有本輪修改，仍以初讀時完整原始碼與目視結果為準。

### 同步與實跑結果

- 八節當前 case hash 均等於各自 JSON；notebook cell 3 均等於當前 case，notebook stdout 均等於 JSON。`13-nms-free.json` 的新增執行時間為 `2026-10-02T17:08:08.382146+00:00`，新的覆蓋斷言已同步入 notebook；它沒有改變原題 stdout。
- 在 `/tmp/curriculum-reader-d`、CPU PyTorch `2.9.1+cpu` 實跑全部八節對應的自主題。除下述 NMS score `.99` 題外，依当前文字同步修改後均通過。
- Augmentation 整張框 flip 仍為 `[0,0,64,64]`；IoU 相同框四項皆0、左移4px的DIoU為 `1.2577319145`。
- Anchor-free 新point `(20,20)`、stride4，target `[2,1,5,4]`，loss `1.866909→0.000013`，解碼框约 `[12.01,16.03,39.98,35.99]`。
- Decoupled 总loss改 `Lbox+2Lcls`、assert改 `gbox+2gcls`、輸出文字同步，通過。DFL target2.6及三項預期同步，通過；初始gradient `[.25,.25,−.150000095,−.349999905]`、prob约 `[.0025,.0025,.3975,.5975]`、期待值 `2.5899`格=`20.7192`px。
- Assignment p1框改GT B，IoU欄 `[.25,1]`、quality欄 `[.04375,.8]`、owner `[0,1,1,-1]`。Dual B/p2品質 `.95`，many owner `[0,0,1]`、one owner `[0,-1,1]`，optimal/greedy均1.85。
- NMS 修訂原題及one-target `[0,1,1,0]`、one_ids `[1,2]`均通過；新增覆蓋斷言也通過。没有啟動GPU訓練、远端clone/install或改教材檔案。

### 仍使練習失敗的新缺項

**`13-nms-free.md` 第51行的score `.99` 指示漏了新增覆蓋斷言。** 我照文字只改case39～40行score比較、43行many/NMS數量為0、44行one列表為空，會在新增case45行失敗：

```python
assert 2 in many_after_nms and any(i in many_after_nms for i in (0, 1))
```

`.99`時 `many_after_nms=[]`，本來就沒有A/B覆蓋。請在練習指示補「本題也把覆蓋斷言改為 `assert many_after_nms == []`」，原題`.5`仍保留覆蓋斷言。補這一項後我獨立重跑通過，三路均為空。這是更新預期，不是把正確的原題覆蓋檢查刪掉。

另第47行仍寫「這次兩分支**可能**全被過濾」，第51行才寫「**確定**為空」；可將第47行同步修為確定，直接把更改步驟併入自主題。這個固定100步案例確實全被過濾，沒有隨機不確定性。

### 原報告逐項處理狀態

以下編號對應各節初讀「問題與修改建議」，未補的可讀性建議與會使練習失敗的缺項分開判斷。

| 節 | 初讀項目 | 複查判斷 |
| --- | --- | --- |
| 11-augmentation | 1：snippet初始化／示意碼 | 未補。當前13～17行仍沒建立W/new_boxes；在已有image、boxes、W的獨立呼叫中，仍 `NameError: new_boxes`。27～32行 `clip_to_crop`仍未定義。建議加clone或標示示意碼；完整case本身正常。 |
| 11-augmentation | 2：bag of freebies／mixup／budget／letterbox／target詞義 | 未補。仍建議優先解釋第34行letterbox如何同時改框座標，其餘歷史名詞可附短註或連結。 |
| 11-augmentation | 3：整張框自主題入口 | 題目答案與獨立函式呼叫實測通過；尚未把呼叫方式加入題目，是操作性建議。 |
| 11-iou-loss | 1：ρ、c、atan、α的閱讀橋接 | 已補第19行，單位、角度和係數用途清楚。SVG的c²標字仍未另附pixel²，但正文已交代，屬可選補強。 |
| 11-iou-loss | 2：detach作用／非零v的驗證界線 | 已補「反傳把係數當常數」；第29行仍明寫本例v=0。若要再清楚，可補「本例未驗證非零長寬比項梯度；v的路徑未detach」，不阻礙目前原題。 |
| 11-iou-loss | 3：合法框前提 | 未補。當前case仍接受未驗證的GT；我以GT `[8,12,8,12]`實測CIoU为NaN。請註明輸入必須正寬高，clamp不能修復無效標註；本節有效GT案例全部正常，不必擴成通用框庫。 |
| 11-iou-loss | 4：兩個自主題函式入口 | 答案及獨立實跑通過，原case已有對應檢查；具體呼叫方式仍未加入題旁，可選。 |
| 12-anchor-free | 1：softplus／Smooth L1 | 第34行已補公式、正距離、大小誤差及格單位；主要橋接已解決。仍未明說預設mean平均四邊，屬小補強。 |
| 12-anchor-free | 2：raw直接自由參數、無CNN | 第34行已明說nn.Parameter、沒有圖片/CNN，解決。 |
| 12-anchor-free | 3：自主題同步stride等修改 | 指示仍完整，本輪在/tmp重跑通過。 |
| 12-decoupled-head | 1：乘2練習同步assert／輸出 | 第54行已補兩處改法與不先乘gcls，依文操作通過。 |
| 12-decoupled-head | 2：coupled head詞義 | 第54行已定義共享末端卷積及一次輸出，解決。 |
| 12-decoupled-head | 3：flatten與gradient cosine | 既有±1/0解釋仍正確；flatten如何將權重排成向量未補，屬可選讀碼橋接。 |
| 12-assignment | 1：p2與A重疊卻被mask的兩種幾何 | 第53行已具體補點資格對比框IoU，解決。 |
| 12-assignment | 2：與下一節quality衝突規則的差異 | 第53行已明說不是沿用同assigner，解決。 |
| 12-assignment | 3：BCE→lr1更新數值 | 原第37行界線仍清楚；正logit+.125、背景−.125的更新數字未補，是可選數字旅程補強。 |
| 12-dfl | 1：2.6題三處assert | 第46行已完整交代wanted_grad、期待值和bin2/3概率容差，依文操作通過。 |
| 12-dfl | 2：a/b 1.25是獨立反例 | 第46行已明說不與2.6訓練target混用，解決。 |
| 12-dfl | 3：分散式用詞／bin定義 | 第7行改為以分佈表示，解決。bin定義加在第46行，內容正確但首次公式出現前仍看不到；建議移到第7～11行附近。 |
| 12-dfl | 4：gradient前列target分佈 | 第24行仍未直接列 `[0,.75,.25,0]`後逐項相減；第30行才有該向量，原讀者橋接建議未補。 |
| 13-dual-assignment | 1：solver限定為本例總品質最大問題 | 第27行已限縮，解決。 |
| 13-dual-assignment | 2：consistent只示意共用quality、不重現官方條件 | 第54行已補，解決；可移到第7行或quality表之前，讓首次閱讀先有範圍。 |
| 13-dual-assignment | 3：permutations的3×2=6橋接 | 第54行內容正確，已補。建議移到第27行階乘公式之前，而不是來源之後。 |
| 13-dual-assignment | 4：兩個head均更新的執行證據 | 未改case，只assert one_head變化，當前第40行仍說一次step更新兩個head。實際圖與optimizer支持兩者更新，但若要聲稱逐值檢查兩者，仍應加many_head比較或精確改句。 |
| 13-nms-free | 1：one-target→one_ids同步 | 第51行已補 `[1,2]`，實跑通過。 |
| 13-nms-free | 2：score .99步驟／確定空 | 已補原有數量assert和不要改NMS IoU，但新coverage assert仍使題目失敗，見上方精確重現；第47行的“可能”也尚未同步。 |
| 13-nms-free | 3：覆蓋檢查的宣稱 | case45行新增p2及p0/p1覆蓋assert，原题、notebook、JSON同步，本輪原題實跑通過，已解決。 |
| 13-nms-free | 4：三個.5門檻的語義 | 第51行明確分开score比較與NMS IoU門檻，重要操作混淆已改善。GT matching门槛仍只在第35行說明，可選再用“評估配對”短註與另外兩者並列。 |

修訂没有引入新的原題數字、shape或輸出不一致。最優先只剩NMS `.99`題漏同步新增覆蓋assert；其餘表中未補項是原先讀者橋接與證據措辭建議，應按教材希望讓基本CNN讀者自主完成的程度處理。

## 最終定點複查（2026-10-02 17:21 UTC）

保留前兩輪紀錄。根 agent 再修訂後，本輪重讀指定修改位置與相關case，於 `/tmp/curriculum-reader-d` 使用CPU獨立驗證。**先前會使操作失敗或與驗證宣稱不符的缺項均已解除；沒有發現新的數字、shape或執行問題。**

- **Augmentation snippets與詞義：通過。** 從當前頁面直接擷取兩段Python執行，不另補W、新框副本、helper或面積變數。Flip得到`[40,12,56,28]`，原boxes保持`[8,12,24,28]`、副本不共用資料指標，紅色pixel與新框吻合；crop得到`[0,4,8,20]`、class0、visible128/original256。當前第43行已補letterbox比例/補邊與框同步、target建立連結、Mosaic/mixup/bag of freebies/budget定義。
- **IoU合法GT前提及梯度範圍：通過。** 當前第48行已明說有限座標、正寬高、epsilon不修標註，以及非零CIoU長寬比梯度未測。case6行在計算前檢查GT；我獨立傳入0×0、零寬、零高、反序、NaN、inf六種GT，均收到明確AssertionError，沒有繼續算出NaN。有效框四項loss與一步更新仍通過，stdout逐字等於新JSON。
- **DFL四槽target橋接：通過。** 當前第26行已直接列`[0,.75,.25,0]`，與uniform`[.25,.25,.25,.25]`逐槽相減為`[.25,-.5,0,.25]`，補足之前只能相信gradient答案的跳躍。2.6題三處預期指示仍保持前輪已實跑通過的內容。
- **Dual兩head的執行證據：通過。** 當前case50行檢查many/one梯度均非零，52行保存many權重，54～55行驗證兩head在step後均改變。獨立重跑原題通過、stdout不變；再依自主題改B/p2=.95及相應owner/optimal預期，新增兩head斷言也通過，品質仍1.85/1.85。
- **NMS score `.99`練習：通過。** 當前第47行已改“確定全被過濾”；第51行明确要求把新增coverage預期改為`many_after_nms==[]`。我完全照本輪文字修改score比較、數量/列表與coverage預期，保留NMS IoU門檻`.5`，獨立執行得到many/no-NMS、many/NMS、one/no-NMS三者皆`[]`，全部斷言通過。前輪唯一仍會失敗的題目已解除。
- **最新證據同步：通過。** 八節case hash、notebook實驗cell及保存stdout仍與JSON一致。`11-iou-loss`新JSON執行時間為`17:20:00.575866 UTC`、SHA=`ce2b65ec6ac63cc78cf79cf7e73fbb33edef067817c760e6050b9baf9df82949`；`13-dual-assignment`為`17:20:02.411994 UTC`、SHA=`b535b08c8ff2e9ed3ad6678f4089f7ebc6e21f977ce9a331a8602d79f9ebe96f`。這輪也親自執行了這兩份當前case，stdout逐字一致。

前輪表中其他「可選」入口示例、mean/flatten短註和把bin/3×2說明移到首次使用處，仍屬版面與教學補強，沒有改列為本輪阻礙。這輪只更新本review；未改教材、程式、notebook、圖或artifacts，沒有GPU訓練、外部發訊息或Colab遠端runtime驗證。

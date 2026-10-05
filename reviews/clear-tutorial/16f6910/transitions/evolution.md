# evolution：第 9–12 章非作者銜接閱讀

審查者：/root/clear_first_reference；未參與本組正文、圖或第二輪技術審查。時間：2026-10-05 10:44:00 UTC。依 .agents/skills/clear-tutorial/SKILL.md 與 references/review-protocol.md 第三輪執行。

本輪實際先完整讀 07-targets、07-inference、08-own-images，再依序完整讀 9.1、9.2、10、11.1–11.4、12.1–12.4 共 11 個 target 頁，最後讀 13.1 第 1–65 行開場與配對邊界。先記自己的理解與有意義的小變化，全部正文讀完才查 first-read/evolution.jsonl 的原問題及 git diff 16f69103d1f216d1024a40610623083557324701。沒有將原 issues 當成閱讀答案，也沒有改原 raw、decisions、coverage 或教材。

這是跨頁完整銜接閱讀，不是逐段盲讀；共同檔案系統沒有隔離全文。先前本人核非 lesson 的版本對照來源時已接觸官方 assignment／DFL 機制，因此不稱本組機制首次接觸。沒有重讀全書 1–6／完整 7-loss，沒有把這些頁冒稱本輪已讀前文。小變化以下皆為讀者的手算預測與理由，沒有執行改版教材或產生新訓練數據。實作與數據第二輪範圍另見 technical/evolution.md、technical/modern-applications.md；本檔不代替那些技術查核。

本輪沒有新 blocking 或未解決 burden 銜接問題。原 target 14 件 issues 為 11 burden、3 optional；11 件 burden 與 fusion 的 1 件 optional 在現文核回，decoupled／DFL 的 2 件圖解建議保留 optional，理由見末表。這是閱讀與局部 SVG 呈現的結論；全站 Zensical、MathJax、頁面切換及桌面／手機正文內圖片呈現尚未由本輪驗證，不能據此宣布新版發布通過，也沒有真人學生理解證據。

## 實際前文與跨章主線

07-targets 教的是物件中心取 floor 後的一個責任格：xy 是格內比例，wh 是全圖比例，正格才學框／類別，沒有物件的格仍學背景；同格兩物件是容量／規則衝突。07-inference 把未轉換輸出 decode 成 pixel xyxy，再以 objectness×類別機率篩分數、分類別 NMS；手填框是管線核對，不是學習結果，NMS 也不是評估時與 GT 配對。08-own-images 以實際 sx、sy 及 padding 保存座標回程、模型 config 與類別順序；三步舊紀錄只支持整條管線可跑，改類別名稱不會學新類別。

以 07 固定紅框 [8,12,24,28] 為前文小變化：只平移 x+16，中心從 (16,20) 到 (32,20)，gx 從 1 到 2、gy 仍 1，格內 xy 仍 (0,0.25)，wh 仍 (0.25,0.25)。理由是整格平移只改存放位置。score 篩掉的框不能由後面的 NMS 救回；名稱替換只改 metadata。這些區別支撐下列新規則。

整組不是逐章把同一個已訓練模型全部改裝：9.1 直接更新 raw 槽；9.2 只算 train 寬高統計；10 換 TwoScale；11.1／11.2 使用隨機特徵測元件；11.3 只變圖與框；11.4 只動預測中心；12.1 只學四個距離；12.2 是人工全正樣本的圖片小模型；12.3 使用手填配對材料與另一組前景 logits；12.4 只更新單邊 bin logits。現文都有在用到這些材料前界定範圍，不把它們串成公平的版本效能對照。

## 逐頁實際理解、變化與銜接

### 9.1 09-anchors

任務是把第 7 章直接 wh 比例換成 anchor×exp 修正量，並把每格一槽變成兩槽。64×64、4×4 格，anchor 為 16×16／8×8；紅 GT [8,12,24,28] 中心 (16,20) 屬格 (1,1)。尺寸 IoU 是 1／0.25，選槽 0 正樣本、槽 1 ignore，其他 30 槽背景。學的是 224 個直接 Parameter，不是從圖片產生輸出的 CNN。訓練中心答案 (0,0.25) 與 raw 中心反解分開，wh 答案 log 比值為 (0,0)；中心 clamp 只令 decode 往返近似。

小變化：只把第 0 個 anchor 高改 16→8，GT 與位置不改。兩尺寸 IoU 變 0.5／0.25，仍選槽 0、槽 1 仍 ignore；target 為 [0,0.25,0,ln2]。中心／槽數不變，height raw 需從 0 往上學，因起點高只有 8。不是將 anchor 本身更新成 GT。

前後銜接：保留 07 的格中心責任與 pixel 框，明說第 7 章 wh sigmoid 改成 raw log 修正；兩槽不等於兩類，ignore 不等於推背景。9.2 只處理這些尺寸起點從哪份資料選，不改框位置。自定尺寸 ignore>0.2 與官方 predicted-box IoU 規則有範圍區分。圖與 loss 表足以回答槽狀態；raw 往返選讀不再打斷訓練 target 主線。

### 9.2 09-anchor-clustering

任務是用 6 筆 train 寬高，按中心對齊尺寸 IoU 分兩群，再用群平均更新起點。材料 [8,8]、[9,8]、[8,9]、[32,16]、[30,16]、[32,18]，初值取首／末，分群 000111，均值 (25/3,25/3)、(94/3,50/3)。圖的點在寬高平面，不在原圖 pixel 位置；平均最佳尺寸 IoU 是 anchor 覆蓋尺寸程度，不是 AP。算術平均只是 IoU 距離的 heuristic，正文以反例明說不保證目標單調下降；median 也非一般解。

小變化：只把第二筆 [9,8] 改 [11,8]。它對初值 8×8 的 IoU 為 8/11，對 32×18 為 88/576，仍屬群 0；小群新均值 (9,25/3)，寬高大群不變。這是新 train 統計，不足以推論 AP 改善。

前後銜接：承接 9.1 的「只比大小」與 log(w/anchor)，明示這裡弱 baseline [16,16]×2 與 9.1 非同設定，另列 9.1 起點比較，避免把數值差當訓練增益。只用 train、預處理後 pixel 尺寸、類別／前處理／anchors 保存共同契約；10 的多尺度處理特徵解析度，不是重新聚類或把輸入圖縮兩次。

### 10 10-multiscale

任務是讓相同 pixel GT 在 stride 8／16 的 head 上各有自己的 xy offset，再合併 pixel 候選。材料改成一張 64 圖，紅小框 [5,5,13,13]、藍大框 [32,32,56,56]；只分紅給 fine、藍給 coarse。TwoScale 的 fine_features 是 NCHW，fine_pred 是 NHW7；不是沿用 GridDetector 只多加一個 head，也明說不沿用第 9 章 anchors。兩頭 wh 都除 64，xy 各除 stride；未分配物件在另一頭的中心格是背景，沒有 ignore。

小變化：紅框只往右移 8 pixel，中心 (9,9)→(17,9)，fine 的 (gx,gy) 為 (2,1)、offset (0.125,0.125)；coarse 為 (1,0)、offset (0.0625,0.5625)。wh 仍 (0.125,0.125)，仍只分 fine，不能因粗頭能解相同框就也填其 target。理由是尺度改變責任格與局部 offset，不改全圖寬高比例。

前後銜接：各 decode 後才有共同 pixel 座標系，才能串接候選並跨 head 做 NMS。手填重複候選的 2→1 是管線核對；40 步實測另標 genuine 模型預測，class0 小框 IoU0.44 是 FP／FN、class1 TP 對 AP=1，加總 mAP50=0.5。低 loss 不代表小框通過定位門檻，也沒有 heldout 或公平單尺度效能對照。下一章的 fusion 改 fine 特徵來源；IoU loss 另解釋寬高 MSE 與幾何對齊的差距。

### 11.1 11-csp

任務是測通道切兩路再融合的元件，輸入是隨機 [B,8,8,8]，不是圖片或 detector。前 4 channel bypass，後 4 經兩個 3×3+ReLU，concat 成 8 後 1×1 fuse。Full 1240 參數對 CSP 368，bias 都計入；concat 不等於 add，block 的輸出也不是原值。輸入非零梯度不能單獨證明卷積支路真的使用，正文還分參數梯度與 fuse 更新。

小變化：只把兩個 3×3 的 bias 關掉，保留 fuse bias。Full 減 16 參數得 1224，CSP 減 8 得 360；shape、兩路與 concat 規則不改。少參數不構成 AP 或實際速度提升證據。

前後銜接：讀者帶第 1 章卷積／通道與第 3 章反傳即可追本節；10 head 仍是另一個完整模型，本元件沒接入它。先看 block 主例與圖，歷史 DenseNet／C3 比較已移主例之後選讀。11.2 借用串接概念，但輸入改成深淺兩張不同網格，必須再對空間大小；14 才延伸切分／融合元件。

### 11.2 11-fusion

任務是沿深→淺追 1×1 reduce→nearest→concat→3×3 mix。隨機 shallow [1,8,8,8]、deep [1,16,4,4] 互不相關，不含圖片、backbone 或 head。減 channel 後 [1,8,4,4]，上取樣到 8×8，concat 成 16 channel，再 mix 為 8。concat 只要求 B/H/W 一樣，C 不必相同；reduce 是降低成本而非合法 concat 的必要條件。nearest 複製沒有新細節，「每來源梯度4」限定於輸出總和 loss。

小變化：channels 不變，只令 shallow 空間為 9×9、deep 為 5×5。size=shallow.shape[-2:] 能得到 concat [1,16,9,9]、輸出 [1,8,9,9]，參數仍 1296，單張 float32 concat 是 16×9×9×4=5184 bytes；scale_factor=2 變10×10便不能接9×9。shape 一樣仍不保證原圖位置對齊。

前後銜接：10 的深淺 head 問題是起點，但 channels 已縮為8／16，1296不是直接加在10模型的成本；接回其16／32特徵另列5152。圖中的深／粗在上、淺／細在下，所以 top-down 確實向下；lateral橫接只到本節concat，PAN另選讀。nearest 是主例，bilinear 的 -.25 等來源位置與平方 loss 梯度有獨立選讀圖，不與AP插值混稱。11.3 接着換資料，不延續隨機特徵實驗。

### 11.3 11-augmentation

任務是對固定圖與框同步 flip／crop，不是訓練或量增強AP。64圖紅框 [8,12,24,28] 水平flip成 [40,12,56,28]；半開框用 W−右／W−左，而 pixel 索引用 W−1−index。crop(left16,top8,size32) 先平移成 [-8,4,8,20]、clip至 [0,4,8,20]，可見比例0.5；keep≥0.5保留、0.6刪。刪框但保留紅像素會形成未標註前景，不能隨意教全背景；labels要套同一keep。

小變化：只令 crop left16→12，top8、size32、門檻不變。平移框 [-4,4,12,20]、clip [0,4,12,20]，可見比例0.75，0.5與0.6門檻都保留。若再圖框×2回64，新框 [0,8,24,40]，中心 (12,24)、格 (0,1)、target [0.75,0.5,0.375,0.5]；原物件中心不能沿用。

前後銜接：從特徵元件切換到資料明確；裁框／篩labels及剩餘監督／回64重建targets拆成三段，32與64單位各自有圖。回64是手算延伸，程式輸出仍是crop32，不假稱新增完整resize實驗。07-targets 只收框且預設64，若送crop32須image_size=32；08實際meta／04座標轉換可供letterbox。下一IoU loss是定位監督的新問題，不把圖框增強當網路已學會。

### 11.4 11-iou-loss

任務是固定G [8,12,24,28] 與P [32,12,48,28]，只學預測中心。兩16²框不相交，IoU loss1、中心梯度0；C40×16、union512、空白128使GIoU1.2、cx梯度0.02。DIoU是1+576/1856=1.310345，cx梯度約0.012485；這裡aspect差v0，所以CIoU同DIoU，不能驗其額外寬高比項。正文明說MSE也可能有方向，IoU plateau不是所有回歸loss都失靈。

小變化：P中心只改cx40→44，cy20、wh16不改。P變 [36,12,52,28]，仍無重疊；C寬44，GIoU=2−32/44≈1.272727，cx梯度32/44²≈0.016529仍正，lr50左移約0.826而非原1pixel。loss更大但梯度可更小；aspect仍相同，CIoU仍退化DIoU。

前後銜接：10小框loss低而IoU失敗提供問題動機；本節不是重訓那個模型。DIoU標題先說距離相對包圍尺度，並在介紹處先提醒不保每步ρ變短，三位置表證明往下移ρ²576→577而分母1856→1889使loss下降，y轉折點autograd0與位置有影響不矛盾。替換07 loss時需在保留計算圖的positive xyxy上算，不能拿no_grad decode/NMS回訓練。12.1會換幾何表示，但四邊連續值的SmoothL1是另一教學選擇。

### 12.1 12-anchor-free

任務是從參考點量四邊距離，不用預設anchor尺寸。64輸入／8×8網格只用來說stride8；點列3欄3中心(28,28)，GT [12,16,40,36] 中心(26,26)不須重合。pixel ltrb (16,12,12,8) ÷8得格單位 (2,1.5,1.5,1)，decode乘stride一次回pixel。框外點(44,28)右距離負，不能由softplus正距離表達，仍要assignment。80步只更新4raw距離，不是圖片CNN定位。

小變化：只改參考點為左鄰欄2、列3的 (20,28)，GT不動。pixel距離 (8,12,20,8)、target (1,1.5,2.5,1)，decode仍回同GT [12,16,40,36]。理由是量距起點改變，物件不會跟點移動；此點仍在框內可作正樣本。

前後銜接：正文承認第7章grid也不使用anchor尺寸，只是xywh與ltrb不同；anchor_point命名不等於尺寸先驗，point pixel／distance格／decode pixel有同一旅程。11.4官方CIoU與本例SmoothL1分清；12.2會增加分類與圖片特徵，12.3解決哪些點可用，12.4再改每邊分佈，anchor-free沒有推成NMS-free。

### 12.2 12-decoupled-head

任務是用兩份人工4×4圖片feature測框與分類分支梯度。backbone輸出 [2,8,4,4]，box [2,4,4,4] 是4個raw ltrb；class [2,2,4,4] 是兩類獨立sigmoid logits，沒有第7章objectness。材料所有位置都是class0、距離1.5格，沒有實際GT配對或背景篩選。box-only只進box/shared，class-only只進class/shared，合loss在shared梯度相加；cos≈−0.0073接近正交，不是強衝突或準確度改善證據。

小變化：只令類別數C2→3，新增第3類target0、其他channels／候選不改。最後class 1×1由18參數到27，多9，全模1446→1455，class shape [2,3,4,4]，box不改，P仍16。類別多了不等於格點變多；新的梯度cos數值須重算，不能沿用舊值。

前後銜接：12.1四個直接Parameter→現在有CNN與兩分支，範圍切換明示；從softmax改各類BCE的理由是背景需兩類都0。1446不能與54／638不同容量coupled頭直接判勝。三次backward表、None說明與g_box+g_cls公式已足以追梯度；12.3再談把候選連回GT，13.1才有意用detach讓one-to-one只學自己的整套頭，不與本節意外切共享梯度混淆。

### 12.3 12-assignment

任務是手填四個point、兩GT的class score與預測框IoU，依「嚴格在框內→score×IoU²→每GT top2→選中GT間以IoU解衝突」得到owner [0,0,1,−1]。owner是GT索引不是類別；點是否inside與預測框IoU不同，p3高分但不在框內不獲資格。後續只對另一組四個前景raw logits示範BCE，target [1,1,1,0]、初梯度 [−.125,−.125,−.125,+.125]，不是接12.2模型繼續訓練。

小變化：只把A-p0的score .9改0。metric0不具資格，A只剩p1；B仍p1/p2，p1按IoU仍歸A，所以owner [−1,0,1,−1]，前景 [0,1,1,0]、初梯度 [+.125,−.125,−.125,+.125]。top2沒有填滿兩個的保證；p0梯度方向翻轉，總前景數可變。

前後銜接：先以四列橋接owner→GT類別→兩類硬target [1,0]/[1,0]/[0,1]/[0,0]→單前景欄，再明示本例只實作最後一欄；正式品質軟target另述。與7靜態責任相比現在依當前預測動態選，target配對用no_grad不代表loss輸出detach。訓練一GT可多正樣本、評估一GT最多一TP分清；官方CIoU與所有GT衝突邊界有選讀，13.1用教學品質解衝突有回連，避免同名assignment偷偷改規則。12.4先有owner才能只取正樣本邊。

### 12.4 12-dfl

任務是只把一條邊10pixel／stride8=1.25格換成K4的bin0..3分佈。這是新數字，不沿用12.1同框距離。target [0,.75,.25,0]、bin軸softmax、weighted CE−.75lnp1−.25lnp2、expectationΣkpk；400步只更新四個raw logits。均勻p.25時期望1.5、loss ln4、梯度p−t=[.25,−.5,0,.25]；DFL最小是target熵約.562而非0，同期望a/b反例區分分佈形狀與框幾何。完整頭 [B,P,4,K]、正樣本四邊攤N，mean的梯度要÷N。

小變化：只把target1.25改0.75格，K4、stride8、零logits不變。t=[.25,.75,0,0]、初始loss仍ln4、梯度 [0,−.5,.25,.25]，目標期望0.75格即6pixel。bin0當下梯度0不保證之後一直0，softmax分母隨其他logits更新而變。這不推出機率已校準或AP提升。

前後銜接：12.1 ltrb標量→每邊Klogits；12.2只改box分支輸出，12.3決定正樣本邊；DFL與期望IoU loss分工而非互相替代。target須0≤d<K−1以免右bin越界，官方clamp可截短距離；K16/stride8的120pixel是單邊而非總框寬，reg_max跨repo定義有提醒。13.1再按assignment目的拆兩個完整head，不能將box/class分支當雙頭。文字、四元素反例和梯度已可完成理解；柱圖保留可選需求。

## 下一節 13.1 的實際過渡

實讀開場第1–65行，沒有宣稱整頁13審完。從12.3一GT多正候選切入重複框與去NMS動機；從12.2接入每個head仍各含box/class，從梯度例接有意detach的one-to-one路。top1初選、品質衝突表 [0,−1,−1]、全域toy最佳 [1,0,−1] 都明示不同規則，前一節的IoU衝突與官方CIoU對所有GT比較有直接回連。新版正文還先界定第三GT可在衝突後拿兩個正點，因此不能把top1名稱當最終每GT≤1保證。same predictions／比例指數只支持排名，不保全程相同owner，例子範圍清楚。

13-dual-head-flow 的1280／390獨立render已實看：實線特徵、虛線梯度，one輸入detach停回共享特徵、one頭自身仍學；many回shared，官方推論留one→全圖top-k→score門檻→無NMS。圖把官方top10/top1與toy top2/全域最優分開，未把toy分類梯度當偵測成績。這個過渡不需借後文才能釐清12章哪一個東西改了。

## 原問題修正後核回

原 first-read/evolution.jsonl 的14個 target issues 保留不動；下表是追加的現文複查，不回寫成原讀者pass。其 prerequisite 04／05／06／07問題不屬本報告target closure，也不以本輪漏讀替它們消失。decisions.json現仍pending是協調者彙整流程，不是我改檔的授權。

| 原 ID | 原 severity | 現文位置與實際結果 |
| --- | --- | --- |
| evolution-0065-issue-1 | burden | 9.1 在手算前先分loss target與raw，六步後立即給loss答案小表；encoded/clamp往返移選讀。能在需要時答「訓練比較offsets而非logit反解」；關閉。 |
| evolution-0066-issue-1 | burden | 9.1 改直接摘完整程式，wh/offsets/regression/objectness/classification一致，表連數學與資料來源；無須在兩套alias中追。關閉。 |
| evolution-0079-issue-1 | burden | 10 開頭只說教學範圍，YOLOv3歷史差異在主例／實測後的選讀；主線不先經Darknet、多標籤／NMS反例。關閉。 |
| evolution-0080-issue-1 | burden | 10 coarse/fine分圖、各自只標自身正負，所有白格負也有圖註；相同GT坐標可跨圖對照，不再同區正/負疊色。獨立render實看；關閉。 |
| evolution-0081-issue-1 | burden | 10 _features/_pred固定資料角色，分支圖標NCHW→NHW7、64／16候選與loss匯總；原命名另選讀。能追共享early和兩頭，不須由shape猜語義；關閉。 |
| evolution-0083-issue-1 | burden | 10 新直式實測圖加#0 FP／小GT FN、#1 TP、#2 FP，圖內score/IoU，正文緊鄰表列配對。#2 y64完整顯示；生成原圖保留選讀。獨立render實看；關閉。 |
| evolution-0085-issue-1 | burden | 11.1 主例／兩路圖先於CSPNet歷史選讀，bypass／branch／concat／fuse有具體channels與參數；不靠C3/DenseNet背景才能讀block。關閉。 |
| evolution-0095-issue-1 | burden | 11.2 深粗在上、淺細在下，top-down箭頭向下，實線只畫本程式四步；backbone/head/PAN不在主例冒充已做。獨立render實看；關閉。 |
| evolution-0098-issue-1 | optional | 11.2 主線nearest先完成，bilinear另選讀，來源坐標圖對-.25及兩端對齊有實際位置。選讀改善已處理；關閉。 |
| evolution-0107-issue-1 | burden | 11.3 拆裁框keep／labels與未標前景／32→64重建targets；兩站圖分開單位、數字與手算範圍。能回答當前在哪個畫布；關閉。 |
| evolution-0115-issue-1 | burden | 11.4 DIoU標題改比例尺度，介紹前提醒不是每方向ρ縮短；三位置表讓ρ²及c²同步改變可核。現圖仍只示水平GIoU，沒有垂直DIoU圖，但當下表與敘述已能回答反直覺方向，沒有必須猜的因果；關閉burden，垂直對比可選。 |
| evolution-0130-issue-1 | optional | 12.2 沒有三路backward圖；三次反傳表、None的位置、shared相加公式與retain_graph完整說明已可追方向。保留optional，若新增圖應沿同forward對照三backward，但目前不是主要概念／必要圖缺漏。 |
| evolution-0138-issue-1 | burden | 12.3 loss前新增四列owner→兩類硬target→前景target表，緊接说明只實作前景與正式品質軟target；與12.2無objectness头的接口不需自行補。關閉。 |
| evolution-0147-issue-1 | optional | 12.4仍無bin柱圖／數線；四bin監督、uniform梯度、學得分佈、同期期望a/b與loss熵下限都是明確數列與算式，可以按順序理解。保留optional；若補圖應比uniform／target／學得並區分期望與分佈形狀，不能只畫一個漂亮峰。 |

## 正文及圖的版本指紋

以下完整頁面SHA包含Colab版本連結與保存紀錄；正文SHA是檔案從開頭至 curriculum-evidence:start 前的原始UTF-8 bytes（包含該段前換行，沒有正規化）。13頁完整SHA只定位檔案版本；實讀範圍另列1–65行SHA，不代表完整13已審。這些不是coverage指紋。

| 實讀頁面／範圍 | 完整頁面 SHA-256 | 正文 SHA-256 |
| --- | --- | --- |
| 07-targets.md | `a3816d6f82e0bb9859557d6771ec2978d66304ee4f2a2f6670d8e4514421c205` | `586e9f871b1ab6dc1144fc99522a85592be11338675a1b2e94fe470d11dcfc8b` |
| 07-inference.md | `37e293a9a25016581b92fc216eb3d868b1c80c828c074ee6593a61006b289bfc` | `f30cd709e2d14ebe952c4d62018797fbb131ff29872262c8b007c51829d3b009` |
| 08-own-images.md | `332da2c487b9e3591b61dbb02fc182bbaa1f4afc20461e2c05da332d7f822006` | `1566e446a33c6bd664ca3b4fa095dc1852bb54010d112132520a1050a8d32184` |
| 09-anchors.md | `7604bfaecb11e818a45d2a745c8450801e73c9541b351540ffc7f479080b3cea` | `76cf654880cee90c125012d84f7255b1a0408f090756ff8d2e4a74d3dcf696ac` |
| 09-anchor-clustering.md | `3fc8b91b87bfde4827f7a90949262fdb657560680215fc1b5f39223caba3824e` | `c2a02fb708bbf01b681257a760e8cd4edb1fe0202de840e8980eb51ad0e197f6` |
| 10-multiscale.md | `f8c07a9dde22911e407fa18c5b8ecd9c30451f8f91821f990314573b0c9e4578` | `4b2c869a2b3df05d10d514ddb38be72a206ae644dbef66d36a2835a465c60fb8` |
| 11-csp.md | `40058c08d8d01a1f9ef4ce11d79ac9e89a6603470fc41ea2c08d25ccba476270` | `38f6774337e87395dde2e4a3006158e1a6ca38fc2228bf735388a22fae133898` |
| 11-fusion.md | `e5adc94096402154bae57fcd25306db268043d6dbb532eb5b3523eb4d4a138b7` | `dd257e5983ef8cb52c32e5c0e72d2a9fa6cce8b4a3ff97b872c7b89f8dd46e6d` |
| 11-augmentation.md | `912b5cec4f0ae0e17b6cbbfe0a24d1b2082f690950770b74c113efa18ade6fe6` | `a2d0350d4563da549896ada2e4fbf509b30574357dd7c928fee7b04099bfd44a` |
| 11-iou-loss.md | `e6668a24141a24376dc455cb66cd941f9abc48ad6787007a51e4569e32dff7b4` | `ddcee1e12bdd224c0bbf708da0825a357eaa2f8eed52079bb7e7f130e7ca7b84` |
| 12-anchor-free.md | `6874048fb9cb6f72b024a9c10af98afbd0c7d4a8fce63391a1e81efde25321b9` | `4c2502ce8f400c0081081f7db7af88dd84cf47d9a9c48aeb804d6852dd6c2804` |
| 12-decoupled-head.md | `926dfafe157605f5cb41fd7a84db1ff8c13bf66ad02b28b6359f8c5903786570` | `d4007db080ebcd670bdb6c338e3c683896bcadbfb132fda30cb042c801c769db` |
| 12-assignment.md | `819308138126eef1825a41685d19a79d75ca6af50a30fd9c5747e539344731a5` | `01a2bc21d12bb2ccab144f6da5d9213537e65889c0c3777738c6c9abeda8c893` |
| 12-dfl.md | `670eb809234ae66e1d1590142fd8ec0cf8b15aa1045c0720ad40a8627420a256` | `5994a6d930b8483cb11647e0e24abbbc511925af69f5bef9436d22439744df4c` |
| 13-dual-assignment.md | `23d265ae9a2236832e4fca60cdcddfe27972dda8fd4d32d184588ed206df97b6` | `bfed914525f9c363b0dee640d094c2cfb50aa7f25cba09bad4aed2a89ea6a362` |

13.1 實讀第1–65行 SHA-256：`35871d19f2ad7425e0a8739b69ca4dd80b98d420757f4700df1369f17637788e`。
原 first-read/evolution.jsonl SHA-256：`ad9e5a0482b866609467da22ab2e888ad92ea238a498bc6877e923beddeb103a`。

圖SHA與本輪實際呈現範圍：

| 圖 | SHA-256 | 本輪呈現 |
| --- | --- | --- |
| 09-anchors.svg | `93c73f27386a64d9282413f080db87a2891c249115cb165622cb78c13ba3f4e5` | 1280／390 獨立 SVG render 實看 |
| 09-anchor-clustering.svg | `58a4bd6ea6a5a636e83f03bdbe276ec94d786195c4344d9e88ee1520dc348b13` | 1280／390 獨立 SVG render 實看 |
| 10-multiscale-coarse.svg | `a2cb024b373a8e9cef993dddffc1e433fbb744e8cd25708bd7af0aad1cf7fab4` | 1280／390 獨立 SVG render 實看 |
| 10-multiscale-fine.svg | `a768a4d7dad9f1e5cc062d685e1a642dbaff14a89c8d14b2c38e9320031a3dc5` | 1280／390 獨立 SVG render 實看 |
| 10-multiscale-branches.svg | `ed6adbab0e7222d493f9a84ceb3d935edb2456e1cae4e47c2c864907a00a13ce` | 1280／390 獨立 SVG render 實看 |
| 10-multiscale-observed-predictions.svg | `f64ab6a0957ee14c2ed28a08d50a272f90065a03c671ab8bd38917c228e85c95` | 1280／390 獨立 SVG render 實看 |
| 10-multiscale-learning.svg | `4c985b8824c35b49b6ca1a88df45d7980e60176906c51dff86feab779aaabd17` | 1280／390 獨立 SVG render 實看 |
| 11-csp.svg | `64377bcc3357725640c000e0ac13ef9582578af3643ab9ade1622be84851c5df` | 1280／390 獨立 SVG render 實看 |
| 11-fusion.svg | `7619f421c06e28a559933091f9d15e215c4fd61f79afbd5c9fc90374b195affc` | 1280／390 獨立 SVG render 實看 |
| 11-bilinear-coordinates.svg | `4f87851588f338428e6e72f80eb4db0c345aefb83ebb2eefb5c299f14d0a3a8b` | 1280／390 獨立 SVG render 實看 |
| 11-augmentation.svg | `46a835957fbdf9f3b3baa9a5299e7205a294e15e9a38d071780d9a5b2c8c4086` | 1280／390 獨立 SVG render 實看 |
| 11-crop-resize-target.svg | `74ea43b6fced49b79a6a09bf16aed6b4c52c4f045d19270aecc6f5059344d7ab` | 1280／390 獨立 SVG render 實看 |
| 11-iou-loss.svg | `b9ea80169cfe1940bb9627d5841367f4989d17cfef8d07b5155ce24493e8c3d5` | 1280／390 獨立 SVG render 實看 |
| 12-anchor-free.svg | `0f43b6c74f090473582866c4dc56949c8fd55267e4a5b99f2e499f5382774548` | 1280／390 獨立 SVG render 實看 |
| 12-assignment.svg | `4a3817c274012a7838e95b9b0a9d8188259eea8f5a23869b819df21722e700b2` | 1280／390 獨立 SVG render 實看 |
| 13-dual-head-flow.svg | `43f3c73fc6ddeb833ffcf7413a379d2dcc26f5abb0b29d2def4f4bbde633e3dc` | 1280／390 獨立 SVG render 實看 |


## 圖的實際呈現方法與限制

使用現有 Playwright／/usr/bin/chromium，將當前SVG直接放到本地HTML；viewport1280×800及390×844，body margin16，svg width100%、max-width980、height auto。圖寬分別980與358px；逐張實看兩種screenshots，沒有由XML通過或bbox無溢出代替目視。結果存在 /tmp/clear-reference-evolution-renders/；這是可重建的個人審閱暫存，不是共享實驗證據，也不是網站截圖。

- 10-multiscale-coarse.svg：1280與390獨立render實看：紅小物件中心(0,0)斜線負格、藍大中心(2,2)綠正格，其他白格也負；黑點與gx/gy讀得到。與fine分圖，沒有同格同時兩head狀態的猜測。
- 10-multiscale-fine.svg：1280與390獨立render實看：同64圖、紅小中心(1,1)正、藍大中心(5,5)負；框跨格但中心唯一責任。分圖可與coarse對照，同圖坐標一致。
- 10-multiscale-branches.svg：1280與390獨立render實看：early三stride2得到fine_features，再分fine_head與deep→coarse_head；_features NCHW vs _pred NHW7、64/16候選與兩loss相加都在圖上。不是fusion，fine/deep含共享路徑。
- 10-multiscale-observed-predictions.svg：1280與390實看：GT綠虛線、預測橙，#0 FP/紅小GT FN、#1 TP、#2 FP，class/score/IoU與門檻0.5可讀。#2下邊完整至pixel y64，不被SVG裁界。報告列號非score排名有圖註。
- 10-multiscale-learning.svg：1280可讀loss下降與右側實測框；390縮圖的字太小，不當手機主線圖。正文已將原生成圖收選讀，直式實測圖與數值表負責主線；不要求手改生成證據圖。
- 11-fusion.svg：1280與390實看：Deep4x4在頂、Shallow8x8在底；藍top-down向下經reduce/nearest/concat/mix，綠lateral橫接。輸入是隨機張量，未做backbone/head/PAN的圖註可讀；shape與通道串接有實際路徑。
- 11-bilinear-coordinates.svg：1280與390實看：三列共同來源座標軸，來源0/1、False對应-.25/.25/.75/1.25、True對应0/1/3/2/3/1；虛線端點對照，False超界取邊界值註。回答選讀插值位置疑問，非本節程式nearest主線。
- 11-crop-resize-target.svg：1280與390實看：crop32與resize64兩站等寬呈現，各用自己的pixel單位；框[0,4,8,20]→[0,8,16,40]、新中心8,24與格0,1對得上。箭頭明說以下手算，未混稱完整程式輸出。
- 11-iou-loss.svg：1280與390實看：G/P/C未相交、rho24、C40x16空白128及紫對角c平方1856可讀。下面只示水平左移1pixel GIoU1.2→1.179487；沒有垂直DIoU圖，正文三位置表已區別比例下降與距離增加。
- 09-anchors.svg：1280／390最終render實看：紅實GT與紫虛anchors中心對齊，gx1/gy1淡格、x16格線的floor歸右有圖註；兩槽16²positive1、8²ignore1與其餘30negative分三卡，objectness與其他loss差異直接可讀。
- 09-anchor-clustering.svg：1280／390最終render實看：①兩個中心對齊交集例IoU.25/.5；②寬高平面六點不是圖片位置；③④各群放大、初始紫虛線圈→最後紫菱形，8/9與30/32/16/18座標及25/3、94/3、50/3均值可讀。正文①與②～④方向詞已實讀核一致。
- 11-csp.svg：1280／390最終render實看：Full與CSP各一圖板，8→兩個4→concat8→fuse8，bypass／branch箭頭合流；參數1168+72=1240、296+72=368與bias說明可讀。不是相加、方框大小不表示參數、有無BN/ReLU的範圍有註。
- 11-augmentation.svg：1280／390最終render實看：①原64圖、②由①flip64、③由①crop32，沒有把crop誤接flip後；框、紅pixels半開範圍、原/新中心與責任格1,1→3,1、裁去灰半與可見比例128/256=.5都可讀。三圖同pixel比例與crop來源①清楚。
- 12-anchor-free.svg：1280／390最終render實看：GT12,16,40,36、候選28,28與GT中心26,26不是同點，point至四邊ltrb箭頭、pixel刻度與row/column0..7分開；16/12/12/8除stride8成2/1.5/1.5/1可讀，框外44,28的r=-4px=-.5格明示不能負責。
- 12-assignment.svg：1280／390最終render實看：GT A0..20、B12..32及四參考point8/16/24/40同x數線，p1在兩GT內、p3兩者外；下一圖板四預測框另列不能把point與box混稱，p3高度10不同於其他16且圖註說長條高度不代表框高。所有座標可與正文IoU/inside材料對照。
- 13-dual-head-flow.svg：1280／390實看：訓練兩head、實線特徵／虛線梯度、detach停shared回傳、many學shared、one只學自己；官方top10/top1與toy top2/global分開。推論one→全圖topk→score門檻→無NMS，toy只驗梯度不驗偵測成績有註。

10-multiscale-learning 是原始生成雙欄圖，桌面可讀；390縮圖字太小，不能當手機主線基本解說。正文將它放選讀，而直式實測圖及數值表已提供主線判斷，因此沒有要求手改會被重產覆蓋的生成圖。12.2與12.4無SVG，明確保留可選圖需求，不假稱看過圖。

除下方追加的09圖片局部呈現外，本輪未做全組 Zensical 實際CSS／載入／正文版寬、MathJax公式、導覽切換、圖片放大操作、Colab或GPU檢查。全站browser由另輪處理；本報告只完成所列正文銜接、原問題核回與獨立SVG呈現。若正文／圖SHA再變，須對改動範圍追加複查，而不能沿用本表指紋。


## 最終六圖與09站台局部追加核回（2026-10-05 10:47:13 UTC）

六圖作者最後補 width=480 與相應 height，協調者另外把09-anchors圖板標題baseline 128→110，與上方column索引分開。已重新讀這些屬性／位置、重新render六圖的1280／390，共12張。09-clustering／11-csp／11-augmentation／12-anchor-free／12-assignment各兩張PNG的SHA與先前實看版本逐一相同，確認補intrinsic尺寸沒有改獨立render像素；09-anchors兩張像素不同，已再次實看修改後的整張圖，標題、column刻度、框中心、positive／ignore／negative卡均可讀。上方圖表已換成本輪最終SVG SHA；09-anchors採當前 `93c73f27386a64d9282413f080db87a2891c249115cb165622cb78c13ba3f4e5`，不是作者manifest較早的c429bcd5…版。根據最終檔案核回，不因作者稱final便推定完成。

另外以 Playwright 實際開 http://127.0.0.1:8767/learn_to_yolo/lessons/09-anchors/，1280×800與390×844，確認page／SVG response均200、載入SVG SHA與上述當前檔案完全相同，頁面沒有橫向溢出。正文內圖的實際寬度為480px／314.859375px，natural size480×1180；不是沿用獨立HTML的980／358px。實看兩寬度component screenshot；桌面component capture因長圖滾到中央會捕捉固定header疊在上半部，另以正常頁面滾動擷取上／下兩個viewport，實看可讀的標題／格線與三種槽卡。手機實際圖不需點擊放大可讀。暫存 `site-09-anchors/results.json`、`figure-1280.png`／`figure-390.png`及桌面上／下viewport，在 /tmp/clear-reference-evolution-renders/。

這項追加只核09當前圖片在站台內的局部呈現，沒有將它擴稱11頁browser、MathJax或整站導覽已通過；也沒有改原首次閱讀紀錄。

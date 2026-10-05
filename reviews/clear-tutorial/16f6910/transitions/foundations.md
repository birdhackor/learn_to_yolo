# 第三輪：基礎14頁非作者銜接複查

完成時間：2026-10-05T10:37:02.630459+00:00。審查者 applications；以 `/workspace/learn_to_yolo` 目前工作樹為準，基礎 HEAD 為 `16f69103d1f216d1024a40610623083557324701`，預計發布 tag 尚未作為已發布證據。

這一輪是非作者、非盲讀的前後銜接複查。我未撰寫、未做第二輪技術審查這14頁；我已讀過後段教材與部分前置，不能當成零背景首次閱讀。逐頁實讀現文全部內容（包含展開區源文），沿 index→learning-path→00→01→02→03.1→03.2→03.3→04.1→04.2→05→06.1→06.2→07.1，另完整讀07.2作最後一頁的下一節銜接。每頁先記當場理解、小變化預測，再讀 canonical foundations 原始 issue；沒有改寫或覆蓋舊first-read。

本輪沒有重跑課程基線或GPU。小變化是從已教公式自行推算的理解證據，不聲称已執行其訓練結果。圖只讀取SVG來源的title/desc/文字與正文對應；**14頁桌面／手機 browser、公式render與圖中文字擁擠都未驗**。沒有把SVG存在或source解析當成視覺通過。Root另做全站browser。

結果：原10個 issue 中9個在现稿閉合，1個保留 optional；本輪新增低度 T-F01 圖中 channel/layer混詞，由root修後已實讀核回。沒有新阻斷。這只覆蓋下列14頁與07.2銜接，不能等同全書驗收。

## 本輪新問題與核回

- **T-F01，low，closed**：`01-cnn-flow.svg` 原GAP方框「每層空間取平均」與本節已教channel／卷積層兩詞混淆。提出時仍為原文；root改「GAP：各 channel 取平均」後，實際重讀完整SVG，第27行已正確，`[8,8,8,8]→[8,8,1,1]→[8,2]`未變。只關閉來源語義，不聲稱已看browser。

## 原first-read逐ID closure

Canonical來源 `first-read/foundations.jsonl` 共110個target單位（foundations-0000–0109），都保留原文。下表只列原有issue的10個ID；其餘單位本來沒有issue，逐頁範圍與目前理解在後文記錄。

|原ID|狀態|實際核回與保留理由|
|---|---|---|
|foundations-0002|closed|首頁拿掉尚未教的 grid／loss／NMS／IoU 長管線，改以分類與偵測問法、同圖示意及四個閱讀問題引路；learning-path 保留閱讀依賴，主入口不需先背推論細節。|
|foundations-0012|closed|00 主線先以 x=2 與兩倍誤差=-4 相乘得到 -8；有限改變與略去平方項的 Δ 推導位於「選讀：需要時再回來查」。已能自行預測 eta=.05 的 w/loss。|
|foundations-0013|closed|00 先走五步、輸出、更新表及練習，再進選讀的 None／零梯度、別名、detach／clone／item、eval/no_grad 與 optimizer 身分。必要的 backward≠step、每步清梯度留在主線。|
|foundations-0019|closed|01 開場先 8 張實際材料與紅0／藍1、同位置不同色的辨識問題，僅短述 3×3 堆疊和四層小模型；VGG C/D、512 通道、13+3 與 GAP 出處全部在頁尾摺疊。|
|foundations-0021|closed|01 感受野大小16留在主線；-6～9 的端點、逐層反推表、4i-6～4i+9 與邊角精算位於選讀。新增掃窗、CNN shape flow 及感受野三位置圖；已讀 i=0 的表能追出偏移，而不是猜 -6。|
|foundations-0039|closed|03.1 在 Δy/Δx 前明示一維簡化、x/y 各為單一數；返回圖張量時解釋卷積跨元素影響與 Jacobian、多路梯度相加。人工零分支才使 sum loss 的每元素梯度=1，非一般梯度保證。|
|foundations-0054|closed|03.3 主文保留手算986參數、248840 MAC、3072加法及程式核對；完整 forward hook、modules／handle／旗標說明現在摺疊。讀者可先理解控制變因與結果。|
|foundations-0058|closed|03.3 分布平方平均、交叉項與1/18幅度推算移為選讀；主線用實測 stem 梯度表與40步 loss/accuracy，明說未逐層量特徵、推算不能當成已測數據。沒有偽造逐 block 實测圖。|
|foundations-0068|retained-optional|04.1 尚未展示兩張真正40步 GT／pred 框疊圖；已有兩組預測 xyxy、IoU=.6945/.7752、訓練集限制及分項 loss 圖。人工 IoU 圖明標非模型輸出，不能用它假稱此建議已完成。正文可逐值理解，但一眼看出偏移的改善仍可保留。|
|foundations-0081|closed|05-assignment.svg 已只列正格2／負格14，移除提前出現的 ignore0；正文「正、負、ignore 與 mask」先解釋本節無 ignore 與 loss 差異，再提三狀態。|

另外，foundations-0067／0070 曾建議讓網頁可見3步真實疊圖，與0068同類；目前也未新增，仍保留圖解改善，沒有當成完成。0012／0013的數字箭頭及計算圖可選建議，已藉主線重排與摺疊降低負擔；讀者現在能追 -8 與五步，不需新增圖才能走主線。0039反向路徑標圖也非阻断，現稿一維前提與Jacobian範圍已補。0058沒有要求作者捏造未量過的逐block圖，而是把其推算留選讀并限制主張。

## 逐頁現稿、理解與前後接點

### docs/index.md

原單位：foundations-0000–foundations-0003（4個）。現頁 SHA256：`dc8947e5d0d7baf3ef68a74e6631d788344b5891c50f9cea9701e099a501f6a6`。

當場理解：分類只回答整圖類別；偵測多回答框位置。藍框是人定答案；MiniYOLO是教學小模型，CPU合成資料，非照片品質承諾。Colab各節獨立可只讀網頁。

小變化預測（紙筆／推理）：把同一紅方塊向右移，分類答案仍紅；希望偵測框必須一起右移，圖示答案不能當成實際模型預測。

前文→本節→下一節：index → learning-path：入口已指出暖身及完整路線，後頁補實際先備與各節先後。

材料／答案：同張黑底紅塊，分類答案紅；偵測多一個位置藍框。文字與 SVG desc 都明示是任務答案示意，不是模型已測結果。

圖來源核對：分類／偵測同材料、藍框答案示意標示一致。

|目前圖檔|SHA256|
|---|---|
|classification-vs-detection.svg|`a050400b9fe7e35a06073ca97ab15ee893ce1fef62bc57236707539a00d1c90e`|

原issue：foundations-0002 → closed。具體依据見上表。

### docs/learning-path.md

原單位：foundations-0004–foundations-0009（6個）。現頁 SHA256：`7008aa1575e26490fd7d731450975fc6b67fd8ad5bbc9988a5b1b5f00714f825`。

當場理解：從0一次更新→CNN→診斷→ResNet→分類加位置→多物件責任→人工decode/AP→可訓練grid；9–16是機制小實驗不全疊。17可由0–7接入，18/20另先讀8.1，19先18。C有channel/class雙義由節內限定。

小變化預測（紙筆／推理）：RGB批8張若改64×96，形狀為[8,3,64,96]；只把第17結業當目標，可以暫略9–16，但不能因此省略0–7的訓練與AP。

前文→本節→下一節：learning-path → 00：頁面承諾只有Python必要、梯度由0數字建立；00要實際補導數與更新，不可在主線默認熟PyTorch。

先備：Python 必要，梯度、tensor軸、CNN各在0／1教；17可由0–7接入，18／20另先8.1，19先18。VGG／ResNet只給導覽意義，不把歷史細節當預先精通。

本頁無內嵌圖檔；不要求讀者由不存在的圖才能完成主線。

原first-read本頁沒有issue。這輪實讀未發現必要前置藏入API或後文的缺口；未把「原無issue」作為略過現稿的理由。

### docs/lessons/00-warmup.md

原單位：foundations-0010–foundations-0018（9個）。現頁 SHA256：`ead34cae1663945aafde6555a633d95006477ddace3406d100aa39ff8f285bac`。

當場理解：固定x=2、y=4、w=1，loss=4；局部斜率-8由2倍誤差乘x得到。backward只寫grad，SGD以減eta×grad更新w，eval/no_grad不是step。主線五步及必要API都有當場解釋；其餘別名、物件身分、累加、浮點、Colab重跑細節摺疊。

小變化預測（紙筆／推理）：若eta改0.05，w由1到1.4，pred=2.8，新loss=1.44；只刪step則grad仍-8而w、loss不變。

前文→本節→下一節：00→01：以同樣zero_grad/forward/loss/backward/step骨架換成圖片batch與CNN；00已教更新方向，不需要讀者先懂autograd。

材料／答案：x=2、y=4、w=1；更新前loss4、grad-8，lr=.1後w1.8、pred3.6、loss.16。練習明示只改主optimizer而非可選後段，失敗assert與修法能對回當步。沒有必要內容藏在API補充中。

本頁無內嵌圖檔；不要求讀者由不存在的圖才能完成主線。

原issue：foundations-0012 → closed；foundations-0013 → closed。具體依据見上表。

### docs/lessons/01-small-cnn.md

原單位：foundations-0019–foundations-0026（8個）。現頁 SHA256：`434c49f8ecd00ae79cd7ee76db4ae33935957d3b936a223058958fe717a5a82c`。

當場理解：RGB/NCHW/channel、logit/argmax/交叉熵、卷積權重共用、ReLU、pool、GAP與backbone/head從具體8張材料逐一教。3步SGD全猜藍是流程測試；40步Adam由初始重跑8/8只是訓練集結果，不能把optimizer和步數兩變更當單因果。GAP收位置，為第4節留問題。

小變化預測（紙筆／推理）：把RGB同位置的紅圖換成藍，位置不變但答案0→1；width4→8時後續卷積輸入/輸出channel都倍增，成本約平方，最後仍8×2 logits而不是8×4。

前文→本節→下一節：00五步訓練骨架已可套CNN；01的finite梯度/參數更新≠學會分類，順接02依更新前數值、梯度、前後權重逐層診斷。

材料／答案：8張32×32，紅偶數／藍奇數，1與4、3與6相同位置不同色，不能用位置答對全部。shape／filter／channel逐一教。3步與40步、SGD／Adam、從初始重跑及無held-out的界線清楚。T-F01圖上layer/channel混詞已核回。

圖來源核對：6張SVG的title／desc／文字內容對回材料、shape、三步錯誤圖、40點loss與感受野；圖1/4、3/6配對與0/2錯、1/3對一致。此為來源語義核，不是render核。

|目前圖檔|SHA256|
|---|---|
|01-cnn-materials.svg|`13e9618efcc9b18b43766dbf0c0872b14f05aca561e576d17c3107b95666a3f9`|
|01-convolution-window.svg|`c8b8706ca476ea6725bea4fd32a6c293bf6a4e3aec5f1e10e8f9ae171b1bbc70`|
|01-cnn-flow.svg|`3be7ea88f32a75002ba0d9d09e4629f03de3ef2111922356c297af4fbf4dc965`|
|01-small-cnn.svg|`af5452951febc70c92c795372e4e1d5e37e1e6a66cccda9c403c4091dc3735a1`|
|01-small-cnn-learning-readable.svg|`33ff90b2fa990bfdc09b00f7378ef0a6ce956764c3eccd91706179ba3961463a`|
|01-receptive-field.svg|`ae25c3c72023b673cbedc7911276a6c5cb192aa136264c27794b625153d48dae`|

原issue：foundations-0019 → closed；foundations-0021 → closed。具體依据見上表。

### docs/lessons/02-diagnostics.md

原單位：foundations-0027–foundations-0034（8個）。現頁 SHA256：`b85e5021148b455e736fc18211d7d2cc8bd25904ca042add39dce8d03b229f08`。

當場理解：四步查資料/標籤→逐段梯度和參數更新→少量擬合→held-out；detach錯置可讓head學但body不學；人工二維SGD零初始共線資料使b權重5倍a，反轉b的validation全错，不是影像效能測試。

小變化預測（紙筆／推理）：不改模型、只把validation的b恢復訓練符號，原學到分界同時把train/validation答對；再訓練更久原反轉資料仍錯，因權重比例條件不變。

前文→本節→下一節：01更新≠學會，02把這個警告變成可核程序；02最佳化/泛化分開之後03.1說plain更深連訓練錯誤也更高，有正確問題類型。

例子轉換：01圖片CNN→02兩個數a/b的刻意失敗，開場即明說非影像效果。標籤long與BCE float、detach與保存副本兩用途有即時解釋；20步每次更新後同權重重算train/val，沒有偷偷對比不同時點。

圖來源核對：SVG描述a/b同尺度、訓練實心／validation空心、類別色、b=-.2a與翻b箭頭，對回人工兩種點與分界。

|目前圖檔|SHA256|
|---|---|
|02-diagnostics.svg|`7e1e9773965b976e370f4dfbe5f4e46305d9c986f7cf661e6545f887c773b038`|

原first-read本頁沒有issue。這輪實讀未發現必要前置藏入API或後文的缺口；未把「原無issue」作為略過現稿的理由。

### docs/lessons/03-identity.md

原單位：foundations-0035–foundations-0042（8個）。現頁 SHA256：`5af0147567fe9bfefb6424962c8a2dde6001f1f75ff09a41403112a146cf4555`。

當場理解：教學block相加後無ReLU且不含BN；零分支只驗identity/輸入sum梯度1，負數不截，全部權重零不適合正式學習；隨機4channel另跑2步MSE向零只驗branch更新。一維梯度例與張量Jacobian限制分明。

小變化預測（紙筆／推理）：零分支仍維持不變，但把loss從sum改mean，x.grad為每元素1/4而非1；加post-ReLU會截負數、零處PyTorch梯度0。

前文→本節→下一節：02最佳化問題→03.1直接梯度路；兩路shape相同才用identity，stride/channel改動引出03.2projection；與原版ReLU/BN差異主線已給。

材料分兩組：1channel2×2手設零分支、4channel8×8隨機分支MSE向零；第一組負值保留與sum梯度1不外推成第二組總梯度1。原版post-ReLU／BN簡化主線說清，0初始化僅機制檢查且不建議正式訓練。

圖來源核對：兩路shape及post-add ReLU本節拿掉的圖注，與第二組4channel材料一致，非第一組1channel。

|目前圖檔|SHA256|
|---|---|
|03-identity.svg|`7d06f73907126b64113f69e28e64c2d62f0b5af6665cb31b1c5eafc68e803d4b`|

原issue：foundations-0039 → closed。具體依据見上表。

### docs/lessons/03-projection.md

原單位：foundations-0043–foundations-0050（8個）。現頁 SHA256：`883e15e9584884c4b19a1c37563a5c0052c6259f8cadfb85c55aed0755b249b6`。

當場理解：stage交界主支3×3/s2/p1與P1×1/s2/p0都4→2且3→6，P的W6×3是可學channel轉換，非保留原值。321驗channel順序；10row+col位置探針另驗取樣0/2，same shape不能替代數值來源。兩部分手設與隨機訓練分清，下一節比較用identity而非projection。

小變化預測（紙筆／推理）：只把P stride2改3：4×4仍輸出2×2，常值image仍321，但位置探針改為[[0,3],[30,33]]；shape assert不能抓，位置assert可抓。

前文→本節→下一節：03.1同shape相加→03.2學P對齊且梯度乘權重；03.3返回同shape identity做條件對照，頁尾明說非本節projection，無默默變例子。

材料分常值image、位置probe、另建隨機更新三者；321算channel混合，0/2取樣probe抓stride3假同shape。stage、projection非正射影、權重輸出／輸入軸與圖片NCHW之別在使用前教。練習5×5時兩支3×3與固定probe/目標不跟著改的原因可追。

圖來源核對：F486／P18權重、3→6、4→2、兩路相加與圖上stride/padding一致。

|目前圖檔|SHA256|
|---|---|
|03-projection.svg|`257eb367cd3385b8364aea5b9dbda4d77f288c70c2d2c83a3c77eec764369b75`|

原first-read本頁沒有issue。這輪實讀未發現必要前置藏入API或後文的缺口；未把「原無issue」作為略過現稿的理由。

### docs/lessons/03-comparison.md

原單位：foundations-0051–foundations-0061（11個）。現頁 SHA256：`2a87891b7d02e7257ce3eff2fce8617f71608fcce6bf17dc1255ab9994b410e8`。

當場理解：same初始權重需state_dict複製不靠連續seed；唯一shortcut差、ReLU兩側同置、固定8train/4獨立位置變val，3步驗stem梯度與更新40步看固定設定能否學。BN/初始化原版差異與150倍梯度非泛化證明有主線限制。

小變化預測（紙筆／推理）：若在同樣3block權重令所有F=0，plain head只看bias，各圖同logit；residual把stem特徵送到head，但是否分對仍需訓練而非shortcut保證。

前文→本節→下一節：03.2末尾明示本節返回identity，03.3用控制變因收束分類機制；下一04由GAP分類改加保位置boxhead，任务转换已在01預告。

先備與設計：03.2最後明示本節回到同shape identity，3.3重述F與x+F；state_dict複製初值、兩個optimizer、暖機副本、公平ReLU位置與固定train/val皆留在主線。與原論文退化問題不等同的限制沒有因減負被拿掉。

圖來源核對：實測曲線來源文字寫更新前、8固定圖、seed7、40次，非新圖效果；無逐block幅度圖，不把估算當圖證。

|目前圖檔|SHA256|
|---|---|
|03-comparison-learning.svg|`c9a6051209f25ca5a2485eb85cc9bd83341292429f6e75b6988b8c00bf510552`|

原issue：foundations-0054 → closed；foundations-0058 → closed。具體依据見上表。

### docs/lessons/04-localization.md

原單位：foundations-0062–foundations-0070（9個）。現頁 SHA256：`81f64274118c7325ef820b5c68d397d4a7ac65835ecd903798270a1f7c3cb13b`。

當場理解：單物件同backbone分class平均head與box展平head；xyxy半開px→cxcywh比例，sigmoid四數是座標不是class概率。class+5MSE分記、IoU幾何與MSE各回答不同事；3步、40步、人工IoU分清，40步只有兩訓練圖且顏色可記位置的限制有交代。

小變化預測（紙筆／推理）：GT與12×12pred只平移x=3px，IoU=(9×12)/(144+144-108)=0.6；中心MSE=(3/32)^2/4=0.002197265625；輸入64×64後box flatten維度失配，class GAP仍可。

前文→本節→下一節：01 GAP丟位置已教；04把框單位、半開、sigmoid和IoU從頭補，接04.2非正方形更容易抓寬高/列欄錯，05再解單圖多物件而非默認04涵蓋。

材料／答案：兩張32×32與兩個GT，框head展平1024→4、類別GAP4→2。四個sigmoid數是座標比例；半開／xyxy／cxcywh／px／比例都有首次定義。人工偏移IoU、3步框、40步框及class accuracy分清。保留0068與0067／0070相關圖建議，沒有用loss圖代替定位成果圖。

圖來源核對：IoU圖有人工固定框標示、100/188及半開單位；雙刻度分項曲線標明total=class+5box、右圖未乘5。實測框疊圖建議仍未實現。

|目前圖檔|SHA256|
|---|---|
|04-localization.svg|`a4d35a3b11a2cae489f45a232a0343c73bfbb09aabbd1be4acacca9f567b700e`|
|04-localization-learning.svg|`dc8c7a1d69ef0a22a6e8a399bed8349ddd5eacd7cef1f57a45c00cd2de388baf`|

原issue：foundations-0068 → retained-optional。具體依据見上表。

### docs/lessons/04-coordinates.md

原單位：foundations-0071–foundations-0078（8個）。現頁 SHA256：`40f0d10a836bf80d47e09a135dd8f14b891fd357106d34aecd6ca5907cf4fa56`。

當場理解：原圖px、畫布px、畫布比例三系區別；stretch兩比例、letterbox縮0.8加上方16，inverse先扣padding再除scale。往返成功≠resize像素對齊，奇數37×83用實際29/37與64/83，整数dtype先浮點、empty[0,4]主例/練習清楚。

小變化預測（紙筆／推理）：同框由原80×40改投128×128時s=1.6、top32，框[16,40,80,72]，畫布正規化數仍[.125,.3125,.625,.5625]；直接乘原WH仍錯。

前文→本節→下一節：04.1半開xyxy/px/比例已有教；04.2補非正方圖的轉換，不把32×32定位模型直接當64×64可跑；05多框任務需另建K輸出與配對。

04.1固定32定位→本節用64畫布只做幾何，並沒有宣稱32模型已可吃64。原80×40、畫布64×64、畫布正規化三系明列；奇數取整實際scale與往返成功≠內容對齊被分開；empty[0,4]在第7節會再用。

圖來源核對：往返圖原80×40→64×64、s.8/pad16；rounding圖45.53/46與實際29/37吻合。

|目前圖檔|SHA256|
|---|---|
|04-coordinates.svg|`473e754e4a98108c0b34280886f25a3ad94b279830e2421a4a79480e53047acf`|
|04-coordinates-rounding.svg|`883f55b09ba5bf1f9ef38a193ccdba89f2a8f4a339c9f8f9ad7516ffb20e4fef`|

原first-read本頁沒有issue。這輪實讀未發現必要前置藏入API或後文的缺口；未把「原無issue」作為略過現稿的理由。

### docs/lessons/05-assignment.md

原單位：foundations-0079–foundations-0086（8個）。現頁 SHA256：`6866e9afeab546775b54b3cca6220c371a649d4df86e45991da832bbde71dbbe`。

當場理解：variable N標註轉固定[B,S,S,7]，單slot中心floor責任格；xy格內、wh整圖比例。正負mask逐項，負格只obj、ignore本例無；output grad0≠共用head不更新，head只用人工features。中心碰撞明報容量不足，NMS不能補第二GT；整批空圖NaN限制指07-loss。

小變化預測（紙筆／推理）：框[16,16,32,32]中心24/24由r1c1負責，target[.5,.5,.25,.25]；改S8責任r3c3且xy=0（中心落格線），wh仍.25。空圖B1時本節box/class mean產NaN，不可硬稱已支援全空batch。

前文→本節→下一節：04教px/center比例，05重新定義xy的格內單位避免與04整圖cx混；05建立target規則，06.1以反向decode人工輸出取回同主例框，是可驗雙向接點。

04單框→本節多框head的新固定輸出與可變N，是新任務明示。slot非類別、正格非圖片、背景obj非第三類皆已定義；symmetric主例另有asymmetric核對，可抓row/col。人工features更新及全空batch NaN限制清楚，不把這當訓練好的detector。

圖來源核對：中心12/12與44/44、r0c0/r2c2及target相同原因對回正文；現圖只先正2／負14，沒有未教ignore。

|目前圖檔|SHA256|
|---|---|
|05-assignment.svg|`0bf373e60d870cd97fd1749ca0a93de204aeb0ba552f06fcfc3675f611925598`|

原issue：foundations-0081 → closed。具體依据見上表。

### docs/lessons/06-decode-nms.md

原單位：foundations-0087–foundations-0094（8個）。現頁 SHA256：`5a4ab36893f7e89bd5994c0dda82e927a10158069a1a56c3cec9ceaaf9442401`。

當場理解：人工logits依05定義解碼xy格尺寸、wh整图；三新候選黃.64/藍.72/背景紅.855。score=obj×maxclass不是precision或校準概率；classwiseNMS候選對候選、不讀GT，藍黃IoU7/13刪黃但保高分FP。四種門檻與還原座標需分開。

小變化預測（紙筆／推理）：score門檻降到.60仍留下同三框，NMS仍[2,1]；把黃框類別單改1後，即使幾何不變classwiseNMS不再藍黃互刪，最後三個都留。

前文→本節→下一節：修正前頁當場预期：06不是沿用05那兩個GT，而是清楚說另造logits/唯一[28,24,44,32]GT的新反例；定義/單位不變。06.1不含真實模型效能，06.2另接pred→GT matching補precision/AP。

05→06保留head數字定義，換成三個人工logits反例，開場及NMS段明示唯一新GT。我的05當場預期以為會沿用同一主例，讀06後已校正：不需要沿用同個框，資料切換有交代。NMS候選↔候選、matching預測↔GT、assignment訓練前區別在同一頁列清。

圖來源核對：人工3候選索引、框座標與.64/.72/.855、keep[2,1]、唯一GT與背景FP描述一致。

|目前圖檔|SHA256|
|---|---|
|06-decode-nms.svg|`2e6d6cf9f26d95e6a3002b894596e11b8b1767211be038127acc73e39a82ff69`|

原first-read本頁沒有issue。這輪實讀未發現必要前置藏入API或後文的缺口；未把「原無issue」作為略過現稿的理由。

### docs/lessons/06-evaluation.md

原單位：foundations-0095–foundations-0102（8個）。現頁 SHA256：`1242dc8db9d2e4a095ff3b5c5e6e1ac52ef1eba7721a1963d4fe5f9a17be5630`。

當場理解：每class跨圖score排、同圖同類未配GT最高IoU≥.5，FP重複/背景、FN漏GT；P預測分母R固定GT分母。四rank先table→PR點→右側最大包络→recall寬×右端高度AP1/3；補點/無GT/候選截斷都教。官方差異完整移主例後摺疊，AP≠P×R另摺疊，不擋初學主線。

小變化預測（紙筆／推理）：把最低.70真TP score改.99（超過背景.95），排序TP/FP/TP/FP，包络0→1/3高1、1/3→2/3高2/3，AP5/9；最終TP2 FP2 FN1和P/R不變，證實排序影响AP。

前文→本節→下一節：06.1score不是precision由06.2GT配對具体回答，候選/GT IoU不再混NMS；2圖3GT是另造人工評分材料且開場明示，07將把這些定義接到真正可學圖像資料。

06.1→06.2換成2圖3GT4人工pred，開場先交材料與目的；score .95背景誤報再次說明score≠precision。主例rank→P/R→PR→envelope→AP不依赖後面的官方段。原first-read無issue；新增burden delta另實讀：官方配對大段已在AP主例後摺疊，P×R延伸另摺疊。

圖來源核對：PR軸recall/precision、四rank點、rank3被包絡抬高但判定仍FP、2/3→1高度0/AP1/3皆可對回表。

|目前圖檔|SHA256|
|---|---|
|06-evaluation.svg|`8289113b070ec540923910c95227d2592d09545c0cb2e6ce215091b8ce8674fc`|

原first-read本頁沒有issue。這輪實讀未發現必要前置藏入API或後文的缺口；未把「原無issue」作為略過現稿的理由。

### docs/lessons/07-data.md

原單位：foundations-0103–foundations-0109（7個）。現頁 SHA256：`f086a8694e46320912d340add2c1339c2a04fedc061c014092962d839ef97730`。

當場理解：資料契約imageCHW float32 RGB0–1、box[N,4]半開px float、labels[N]long；B影像stack但variableN標註list。空[0,4]非假零框、背景非第三class；class1藍畫channel2。固定红[8,12,24,28]像素/標註逐值核與生成0–2框雜訊不同，矩形8–15整格内→中心严格格内→xy≠0已補。

小變化預測（紙筆／推理）：只把蓝色像素放channel1而labels仍1，shape/框范围合法但expected_pixels按label映射Bchannel逐值核会失败；红x2增4且切片同步增4，红sum320、旧256切片断言仍过，因此必须查整块而非旧窗口。

前文→本節→下一節：06是人工输出评估，07.1明确尚无模型只检查inputGT；07.1固定场景红中心(16,20)在格线，生成材料中心非格线，两者不同性质已有区分，需实际读07.2看它如何解释端点。

06人工pred評分→07建立真正訓練前input/GT契約，明說尚未建立模型。固定[8,12,24,28]紅框會接下一節；固定中心在格線與生成資料中心非格線不同。ShapeDataset 8–15px整格內→中心嚴格格內→xy不為0的理由已在本頁，不需翻source猜。空框、variableN list、RGB/class編號及重畫核對都能從本節學到。

圖來源核對：SVG顯示固定紅/藍px框與RGB/channel順序、半開兩端及紅中心16/20；它不是帶雜訊生成資料，正文已分清。

|目前圖檔|SHA256|
|---|---|
|07-data.svg|`7c45416667c1e130c9ef4433a600435edd4f2f8de0c20869487e3f7e0429c1ab`|

原first-read本頁沒有issue。這輪實讀未發現必要前置藏入API或後文的缺口；未把「原無issue」作為略過現稿的理由。

## 07.1 → 07.2 的最後銜接

實際完整讀07-targets現稿作下一節context，不把它算成本輪第15個審查頁：固定紅中心(16,20)、藍中心(48,44)，依floor得到gy/gx=(1,1)/(2,3)，xy=(0,.25)/(0,.75)；主文先教格線左閉右開，摺疊才談sigmoid端點只能逼近。07.1生成材料寬高8–15並整格內的句子，已足以理解為何真正訓練的ShapeDataset不會有xy端點0；固定場景與生成材料的差異不是矛盾。

07.1的target是原始逐圖dict，07.2明示原始標註list→build_targets→固定格target的格式轉換；NCHW的N=batch與每圖框數N同字母不同用途在07.1表內已限定。05負格class填-1、07.2填0的變更也在07.2先交代，不能沿舊值判正負，需positive。

## 保存內容與範圍限制

- `foundations-current-read.jsonl`：14頁當場閱讀紀錄，各有實際時間、非盲讀旗標、初讀頁SHA、理解／小變化／過渡。05對下一例的預期被06實讀校正，保留原紀錄與後續校正，沒有重寫成事先知道。
- `foundations-inputs.json`：最終14頁與內嵌圖完整SHA／文字source摘要、07.2context SHA及canonical原始紀錄SHA。
- 本報告未修改docs、coverage、code、notebook、artifacts或舊reviews；只新增本輪報告，並依root要求在自己第二輪probe目錄補原官方固定commit LICENSE及來源說明。
- Browser未測；實際新圖可見性、手機字級、公式版面、頁面連結點擊由root全站browser補驗。
- 04.1實測框疊圖仍retained-optional。來源有兩組框與IoU，但缺一眼可比的疊圖不被改標已完成。

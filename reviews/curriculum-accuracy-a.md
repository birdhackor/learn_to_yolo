# 獨立技術正確性審查 A

審查日期：2026-10-02。範圍依序為 00-warmup、01-small-cnn、02-diagnostics、03-identity、03-projection、03-comparison、04-localization、04-coordinates，目標版本為待發布的 lessons-v0.2.0。本 reviewer 重新讀原始材料並獨立驗算，未讀取或引用既有 reader reviews。

**結論：八節的核心算術、模型 shape、梯度機制、訓練數值及歷史歸屬均一致。沒有發現 P0／P1 的數學或實作錯誤。需修正一項驗證紀錄過度宣稱，以及三項低嚴重度的適用條件／文字精確性問題。**

嚴重度：P0＝會使主要結論失效的重大錯誤；P1＝會教錯核心機制或令主要案例失敗；P2＝證據敘述或次要技術結論需修正；P3＝限定條件或文字精確性的小修正。未發布 tag、人工資料小、未復現完整原版模型，不列為技術錯誤。

## 查核方式

八節各自的 docs/lessons/&lt;id&gt;.md、lesson_cases/&lt;id&gt;.py、notebooks/&lt;id&gt;.ipynb **所有四個 cells（含環境格、markdown、完整案例及已存 stdout）**、artifacts/checks/curriculum/&lt;id&gt;.json 均完整讀取。補充 JSON／SVG 的具體清單見逐節內容；沒有補充產物的節不以別節替代。另完整讀取 scripts/run_learning_extensions.py，並讀取補充定位實驗所用的 miniyolo/geometry.py。

在 /tmp/accuracy-a/&lt;id&gt;/ 隔離重跑全部八個短案例，使用專案 .venv-model/bin/python、PyTorch 2.9.1+cpu。八個 subprocess 均 exit 0；stdout 與相應 JSON 完全一致，僅 03-comparison 的實測秒數依本次執行而不同。八個 case_sha256 皆與現有案例檔吻合；八個 notebook 的完整案例 cell 皆等於案例檔，已存 stdout 皆等於 JSON。未在原 checkout 重跑會覆寫產物的命令。

三個 40 步補充實驗在 /tmp/accuracy-a/extensions/ 使用相同小模型與固定資料重算：01-small-cnn、03-comparison、04-localization 的**完整報告 dictionary 與現有 learning JSON 全部相等**，包含每一步 loss、梯度 norm 的 min／max、權重變化、分類結果，以及定位框與 IoU。這些是極小資料的 CPU 驗算，沒有 GPU、長訓練或完整架構效能實驗。另逐點解析三張 learning SVG 的全部 40 個曲線點（plain／residual 各一條），確認與 JSON 的線性座標映射一致，最大圖座標誤差小於 0.000001；圖例、刻度、標題與可及性文字均讀取。

Notebook 環境格是靜態查核，沒有執行未發布 tag 的遠端 clone／Colab 安裝。本 review 不宣稱已驗證發布後 Colab 網路流程；也不將待發布 tag 視為技術缺陷。

## 已實際讀取的權威來源

### VGG 原始論文

讀取官方取得的全文 /tmp/yolo_sources/1409.1556.txt，包括摘要、§2.1 Architecture、§2.2 Configurations、Table 1／2、§2.3 Discussion。URL：[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)，[PDF](https://arxiv.org/pdf/1409.1556)。

§2.1 原文：「the padding is 1 pixel for 3 × 3 conv. layers」及「Max-pooling is performed over a 2 × 2 pixel window, with stride 2」。同節明說原版有三個 FC，前兩個各 4096，最後 1000 類。§2.3 明說兩個不插入 pooling 的 3×3 卷積有 5×5 感受野，三個有 7×7。§2.2／Table 1 顯示通道從 64 起、最高 512，配置 D 為 16 個有權重層。這些實際段落支持小卷積堆疊的歷史歸屬，也證明教材的 4／8 channels、兩 blocks、GAP 是明確的教學縮減。

### ResNet 原始論文

讀取官方取得的全文 /tmp/yolo_sources/1512.03385.txt，包括 Introduction 的 degradation 論證、Fig. 2、§3.1–3.4、Eq. (1)／(2)、§4.1 Plain／Residual Networks 與 Identity vs. Projection Shortcuts 的相關段落／Table 2／3。URL：[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)，[PDF](https://arxiv.org/pdf/1512.03385)。

§3.1 定義 F(x) := H(x) − x；§3.2 Eq. (1) 是 y = F(x, {Wi}) + x，同節原文「We adopt the second nonlinearity after the addition」，不能忽略原版相加後 ReLU。Eq. (2) 是 y = F(x, {Wi}) + Ws x，同節明說 x 與 F 的 dimensions 必須相等。§3.3 的 option B 用「1×1 convolutions」匹配 dimensions，跨兩種 feature-map 尺寸用 stride 2；§3.4 明說 BN 在卷積後、activation 前。

§3.2 對照條件是相同參數數／深度／寬度／計算成本，並明確排除額外 element-wise addition。§4.1 同時討論 training error 與 validation，且不把 degradation 簡化成必然的梯度消失；原文「backward propagated gradients exhibit healthy norms with BN」。教材沒有宣稱 toy norm 可以重現這一歷史結果。

### PyTorch 2.9 正式定義

以下頁面均於本輪下載並閱讀相關定義，文字副本僅存 /tmp/accuracy-a/：

- [Linear](https://docs.pytorch.org/docs/2.9/generated/torch.nn.Linear.html)：y = x A^T + b；weight 為 [out_features, in_features]；bias=False 才省略 additive bias。
- [Conv2d](https://docs.pytorch.org/docs/2.9/generated/torch.nn.Conv2d.html)：輸出為 bias 加上跨輸入 channels 的加總；實際運算是 cross-correlation；groups=1 時所有輸入連到所有輸出。Shape 完整式包含 dilation*(kernel_size−1)；weight 為 [Cout, Cin/groups, kH, kW]。
- [MaxPool2d](https://docs.pytorch.org/docs/2.9/generated/torch.nn.MaxPool2d.html)：每 channel 區域取最大；預設 ceil_mode=False；ceil_mode=True 改用向上取整並有邊界規則。
- [AdaptiveAvgPool2d](https://docs.pytorch.org/docs/2.9/generated/torch.nn.AdaptiveAvgPool2d.html)：輸出目標 H×W、保留輸入 channel 數；本課 output_size=1 對應每 channel 1×1。
- [ReLU](https://docs.pytorch.org/docs/2.9/generated/torch.nn.ReLU.html)：逐元素 max(0,x)、輸出 shape 不變。
- [CrossEntropyLoss](https://docs.pytorch.org/docs/2.9/generated/torch.nn.CrossEntropyLoss.html)：input 為「unnormalized logits」；類別索引 target 在 [0,C) 且 dtype long；此模式等價於 LogSoftmax 接 NLLLoss。教材說內部處理 softmax 是可接受的概念簡化，並非建議先手動 softmax。
- [MSELoss](https://docs.pytorch.org/docs/2.9/generated/torch.nn.MSELoss.html)：mean 對**所有元素**取平均，不是只除 batch 數。
- [Autograd mechanics](https://docs.pytorch.org/docs/2.9/notes/autograd.html) 的 Setting requires_grad、No-grad Mode、Evaluation Mode：backward 累加 leaf 的 .grad；no-grad 不記錄 backward graph；原文「Evaluation mode is not a mechanism to locally disable gradient computation」，train／eval 與 no-grad 是不同開關。
- [SGD](https://docs.pytorch.org/docs/2.9/generated/torch.optim.SGD.html)：沒有 momentum／weight decay 時更新為 θt = θt−1 − γ gt。
- [Optimizer.zero_grad](https://docs.pytorch.org/docs/2.9/generated/torch.optim.Optimizer.zero_grad.html)：set_to_none=True 後，沒有收到梯度的參數 .grad 保持 None。
- [interpolate](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.interpolate.html)：4D 圖片採 NCHW；align_corners 的幾何把 pixels 看成 squares，False 以角落邊界對齊，支持本課的連續 pixel 邊界縮放契約。

## 00-warmup

**逐節來源**：完整讀取該節 MD／PY／notebook cells 0–3／00-warmup.json。本節沒有 learning JSON 或 SVG。另讀上述 Linear、SGD、autograd、zero_grad 正式段落。

**核實**：

- [1,1] 輸入及 weight、[B,D] @ [K,D].T → [B,K]、scalar loss [] 正確。bias=False 與 fill_(1) 對應手算設定；Linear 預設初始化不是固定1。
- x=2、w=1、y=4 得預測2、loss4；鏈式法則 2(wx−y)x=−8；lr=.1 得 w=1.8、預測3.6、loss=.16。lr=.25 得 w=3、預測6、loss4。手算及 CPU 重跑一致。
- backward 與參數更新、梯度累加與清除、detach／item 斷開原图、clone 副本、eval 不停梯度均正確。Optimizer 引用傳入參數物件，換 model 需重建 optimizer，與本例 assertion 一致。
- 每步 loss 不保證都下降、此線性問題不能代表 CNN 或泛化，是合理限制。

**需修正 A1，P3，定義的限定條件**：MD:11 將「一般線性層」概括為只有乘權重後加總。PyTorch 的一般 Linear 是 affine xW^T+b，內積只對應未加 bias 的部分。MD:33 已有 bias=False，所以**本例數學與程式沒有錯**；建議第11行加「本例沒有 bias；一般線性層還可以加每個輸出的 bias」，避免把一般 Linear 定義成純內積。

## 01-small-cnn

**逐節來源**：完整 MD／PY／notebook cells 0–3／01-small-cnn.json／01-small-cnn-learning.json／01-small-cnn.svg／01-small-cnn-learning.svg；另讀 learning extension 程式、VGG §2.1–2.3／Table 1–2 與上述卷積、pool、GAP、ReLU、CE API。

**核實**：

- 32×32×3=3072；HWC 與 NCHW、permute 與 reshape 的區別、float32 0–1 及 long 類別 [8] 正確。
- groups=1 的 RGB 3×3 輸出 channel 用27個值與27個權重，加 bias；卷積共享位置權重，輸出 channels 不必等於顏色／物件類別。ReLU 與 2×2 stride-2 max pooling 定義正確。
- 表中 shape 全部正確：[8,3,32,32] → [8,4,32,32] → [8,4,16,16] → [8,8,16,16] → [8,8,8,8] → [8,8] → [8,2]。
- 理論感受野依序1→3→5→6→10→14→16，且 pooling 後相鄰特徵間距增加，推導正確。GAP 保留 channel、移除直接的位置排列，並未宣稱所有 CNN 都有完美平移不變性。
- 四層 conv／head 參數為112、148、296、584、18，合計 **1158**；MAC 為110592、147456、73728、147456、16，合計 **479248／圖**。float32 [8,4,32,32] 為131072 bytes＝128 KiB。MAC 不含 bias、activation／pool／搬移的範圍有說清楚。
- width=8 練習參數224、584、1168、2320、34＝**4330**；MAC 221184、589824、294912、589824、32＝**1695776**；皆獨立重算，3.74倍說法正確。
- 三步全猜1，錯誤索引0／2／4／6；SVG前四張矩形座標、色彩及標題與人工資料一致，紅色錯誤標題對應正確。
- 40步改Adam lr=.01，完整報告重算相等：初始 .6941443085670471、最後更新前0、8張訓練全對，沒有validation；最小gradient norm約1.15e−17仍非零。有限精度的loss=0不代表新資料完美，敘述正確。
- VGG歷史歸屬成立；原版大FC被GAP替代、通道／深度縮小均明說，是合理簡化，不要求補成VGG16。

**需修正 A2，P3，公式的適用範圍**：MD:21 稱 floor((H+2p−k)/s)+1 是「一般」輸出長度，實際只適用 dilation=1，pooling 另須預設 ceil_mode=False。保留簡式並加條件即可；MD:48 參數式另加「本節 groups=1、有 bias、方形 kernel」。不需要引入 grouped conv 的完整教學。**本節所有尺寸、參數量、MAC 與練習數值都正確**。獨立反例：H=8、k=3、p=0、s=1、dilation=2 的 Conv2d 實得4，簡式得6；H=5、k=s=2、ceil_mode=True 的 pooling 實得3，簡式得2。

## 02-diagnostics

**逐節來源**：完整 MD／PY／notebook cells 0–3／02-diagnostics.json。本節沒有 learning JSON 或 SVG。另讀 zero_grad／autograd／CE 官方定義。

**核實**：

- body 2→3、head 3→2 與 head(body(x).detach()) 切圖正確。head有梯度而body.weight.grad=None，修復後body非零梯度與更新均重跑確認。None與很小數值不同；教材沒有宣稱量測了斷圖模型的loss下降。
- [0,2] 不符合兩類logits的0／1類別索引契約，調learning rate不會修復它。argmax不能代替CE的logits輸入，正確。
- 訓練8筆只有兩個不同點；validation反轉b且未參與更新。無bias、全零初始化、SGD lr=.2、20步與正文一致。
- 更新後loss時間點及step19＝第20次更新正確。三個記錄點train／validation loss .5945/.7937、.2244/1.5204、.1236/2.0153，最後權重兩列約±[.1950317,.9751588]，train=1、validation=0均再現。
- 新增8筆練習亦獨立跑20步，train／validation都是1；新a=±.3並非把validation的a=±.2同點放進訓練。人工分佈限制及test不應用來挑超參數正確。
- 小資料overfit是管線／容量線索，與held-out泛化不同；一次小norm不足以判斷vanishing gradients，準確。

**需修正**：未發現技術錯誤。數值型背景／物件訊號只是明說的人工比喻，不要求另建真實圖片實驗。

## 03-identity

**逐節來源**：完整 MD／PY／notebook cells 0–3／03-identity.json。本節沒有 learning JSON 或 SVG。另讀ResNet §3.1、§3.2／Eq. (1)、Fig. 2與ReLU／autograd API。

**核實**：

- y=x+F(x) 是同位置／channel相加，非concatenation；兩支[B,4,8,8]輸出仍同shape。4-channel兩個無bias的3×3卷積有 **288** 參數，shortcut沒有參數。
- 人工零分支保留[-2,-1,0,1]，sum loss對各輸入梯度為1。全零兩層卷積與中間ReLU下，兩層weight gradients都是0，亦獨立檢查。另建隨機分支訓練，沒有推薦死分支初始化。
- 加post-add ReLU後y=[0,0,0,1]，獨立autograd得輸入梯度[0,0,0,1]；正文對正／負值說明正確，零點採PyTorch的0梯度約定。
- 隨機block的MSE目標是整個輸出接近零，F可往−x修正，與人工F=0測試不同。[2,4,8,8]、兩步loss 1.0816→1.0724、非零末層梯度及更新均再現。
- 原版相加後ReLU為原文明寫設定，本節省略且指出負值差別，屬正確且必要的簡化。梯度直接項存在不等於總梯度永遠1或沒有最佳化問題，限制正確。

**需修正**：未發現技術錯誤。

## 03-projection

**逐節來源**：完整 MD／PY／notebook cells 0–3／03-projection.json／03-projection.svg。本節沒有 learning JSON。另讀ResNet §3.2／Eq. (2)、§3.3 option B、§3.4與Conv2d官方式。

**核實**：

- [B,3,4,4]主分支3→6、3×3／s2／p1，再6→6、3×3／s1／p1，得[B,6,2,2]；shortcut 1×1／s2／p0同形。P(x)+F(x)與SVG分支、shape、s／p圖例正確。
- P不是逐值複製；1×1只限空間1個位置，仍混合全部输入channels。1*1+2*10+3*100=321、第一輸出channel四點皆321正確。此p0設定stride2取座標0／2，不是區域平均池化。
- P weight [6,3,1,1]＝**18** 參數，主分支162+324＝486，block＝**504**。P每圖6×2×2×3＝**72** MAC、另24個逐值相加，正確。
- 隨機block [2,3,4,4]→[2,6,2,2]、零target MSE、兩步 .3553→.3350、P非零梯度與更新都再現。
- 5×5練習兩支輸出3×3，獨立forward確認；P改stride1成5×5而主分支仍3×3，assert應失敗。
- 1×1 learned projection與跨尺寸stride2有原論文支持；省略BN／post-add activation、沒有架構優劣結論，均合理。

**需修正**：未發現技術錯誤。

## 03-comparison

**逐節來源**：完整 MD／PY／notebook cells 0–3／03-comparison.json／03-comparison-learning.json／03-comparison-learning.svg；另讀learning extension、ResNet §3.2對照條件、§4.1／Table 2–3。

**核實**：

- stem、3個F、head結構相同；F=Conv–ReLU–Conv，只改是否加x，兩邊沒有末端ReLU，避免混進activation變因。
- 複製state_dict後逐參數初值相等；分開optimizer；資料／SGD lr=.1／步數相同。Validation位置與訓練不同，有逐圖unequal檢查；未用validation更新。
- stem112＋3×288＋head10＝**986**；MAC 27648＋6×36864＋8＝**248840／圖**；residual另3×4×16×16＝**3072**加法。1-block練習410參數、101384 MAC、1024加法正確。
- 記錄是更新前loss及stem weight L2 gradient norm、更新後validation。plain三步norm約 .000987/.001004/.001014；residual約 .146834/.147695/.146226；兩邊三步validation=.5均再現。秒數只是本次CPU小循環觀察，非效能基準，教材與JSON timing scope一致。
- 40步完整報告再現：plain .693790674→.692942739、train／val=.5；residual .692160308→.030739356、train／val=1。四張shifted validation仍是很小人工分佈，未被說成ImageNet結論。
- SVG兩條曲線40點均吻合各自loss_history。norm大不等於梯度品質／泛化較好；參數數相同不等於函式相同，正確。
- 原版對照研究為歷史來源；明說省略BN／深度／相加後activation，無須補完整benchmark。原論文也不支持ResNet永遠更準，教材沒有此錯。

**需修正**：未發現技術錯誤。曲線橫軸可選改成「training iteration（loss measured before its update）」以更直接呼應更新前loss，但表格已明確標示，不列為確定錯誤。

## 04-localization

**逐節來源**：完整 MD／PY／notebook cells 0–3／04-localization.json／04-localization-learning.json／04-localization.svg／04-localization-learning.svg；另讀learning extension、box_iou實作及CE／MSE／GAP API。

**核實**：

- 兩個4×4單一非零值feature map平均同為 **1/16=.0625**。位置資訊說法有邊界／padding例外，沒有誇大成任何GAP模型都不能定位。
- backbone [B,3,32,32]→[B,4,16,16]；分類head [B,4]→[B,2]；flatten 1024→4框head [B,4]，shape一致。
- 半開xyxy、cxcywh轉換、按W／H分別正規化、sigmoid是座標比例而非類別機率正確；正寬高不保證框在圖內，限制正確。
- [4,6,16,18]→[10,12,12,12]→[.3125,.375,.375,.375]。人工pred [6,8,18,20]兩项中心誤差 .0625，MSE=2*.0625²/4=**.001953125**，獨立算得一致。MSE對batch×四座標所有元素平均，符合API。
- 兩框各144、交集100、union188，IoU=**.5319149**；SVG用9倍縮放正確畫出框，並明說人工。右移6pixel練習交集72／union216＝**1/3**，MSE=**.0087890625**，獨立算得一致。
- 三步分類 .7173→.7123、框 .0185→.0091、total .8099→.7576及兩個模型pixel框全部再現。同一訓練資料、沒有獨立定位評估的限制清楚。
- flatten框head=4*(1024+1)＝**4100**；GAP後linear框head=4*(4+1)＝**20**；完整Localizer=112+10+4100＝**4222**。固定尺寸與位置對應的代價正確。
- 40步Adam完整報告再現：total .809883177→.430213153、train分類=1、沒有validation；模型框IoU **.694488943／.775239766**也按geometry重算相同。SVG40點吻合JSON；沒有把人工IoU圖當成模型成果。
- 明說是每圖一物件雙head模型，没有完整YOLO歷史版本歸屬。省略背景與可變物件數，是合理階段範圍。

**需修正**：未發現技術錯誤。

## 04-coordinates

**逐節來源**：完整 MD／PY／notebook cells 0–3／04-coordinates.json／04-coordinates.svg。本節沒有 learning JSON。另讀interpolate官方align_corners幾何，獨立算整數、奇數、空框及直式練習。

**核實**：

- [3,40,80]為CHW；原框[10,5,50,25]寬40／高20。整數框先轉float32再建立scale，避免 .8 被整數dtype截成0，正確。
- Stretch sx=.8／sy=1.6得[8,8,40,40]。Letterbox s=.8、resize高32／寬64、上下各16，得[8,20,40,36]；正規化[.125,.3125,.625,.5625]正確。
- 逆轉先減padding再除scale；y1=(20−16)/.8=5；錯序20/.8−16=9。直接乘原圖W／H得[10,12.5,50,22.5]確實錯，數字正確。
- 37×83 resize29×64，實際sx=64/83≈.7710843、sy=29/37≈.7837838；保存兩軸實際比例及整數top／left的契約正確。承認取整後非完美同倍率，不把理想s混用。
- [N,4]與空框[0,4]廣播、每圖metadata、裁切和逆轉不同，敘述正確；SVG的content、padding與框比例全部一致。
- 直式練習resize高64／寬32、左右各16、框[20,8,36,40]；undo還原[5,10,25,50]、stretch [8,8,40,40]，已独立執行全部斷言。
- 只驗幾何、不宣稱模型準確的限制合理。往返是框座標而非原圖pixels重建，沒有誤稱圖片插值可逆。

**需修正 A3，P2，驗證紀錄超出實際 assertion**：PY:60–63只對空框呼叫letterbox並檢查transformed shape，沒有將empty metadata交給undo；但stdout（同步進notebook／JSON／MD）稱「odd-size and empty-box round trips passed」。這是**確定的證據敘述錯誤，不是變換公式錯誤**。最小修正：保存空框metadata、實際呼叫undo並assert回傳shape (0,4)，再保留現在紀錄；或改成「odd-size round trip and empty-box forward shape passed」。本 reviewer另做空框undo得到[0,4]且finite，補檢查不需改幾何設計。修正後須重存JSON與notebook相同輸出／digest。

**需修正 A4，P3，字串不一致**：MD:76叫讀者同步修改圖題 Original 40x80；實際PY:67與notebook完整案例是 Original 80x40。本例W80／H40，程式圖題沒有錯；將練習文字改成實際字串，或寫「同步修改原圖圖題為新的W×H」即可。

## 修訂清單

| ID | 節／位置 | 嚴重度 | 最小修正 | 類型 |
| --- | --- | --- | --- | --- |
| A1 | 00-warmup MD:11 | P3 | 說明一般Linear含bias，本例bias=False | 定義範圍未寫完整；本例正確 |
| A2 | 01-small-cnn MD:21、48 | P3 | 簡式加dilation=1／ceil_mode=False；參數式加groups=1、有bias、方形kernel | 合理簡式需限定；現有數字正確 |
| A3 | 04-coordinates PY:60及同步stdout | P2 | 補真正empty undo assertion，或縮小round trips宣稱 | 確定的驗證紀錄過度宣稱 |
| A4 | 04-coordinates MD:76 | P3 | 圖題引用與實際case一致 | 確定的文字不一致 |

除這份review外，沒有修改教材、程式、notebook、artifact。已有工作樹變更均保留；所有驗算輸出在 /tmp。修訂後可針對四點及受影響的來源／產物做獨立複查。

## 修訂後獨立複查：最終狀態

複查時間：2026-10-02 17:36 UTC之後。以上初審發現保留原狀，下表是對修訂後實際檔案的重新查核，不是依修訂者回報直接關閉。

| ID | 實際複查內容 | 最終狀態 |
| --- | --- | --- |
| A1 | 重新讀00 MD:11，已補「一般Linear還會加上每個輸出的bias（可學的常數偏移）」並指出本例bias=False、只有w | 已修正；符合官方Linear的affine定義 |
| A2 | 重新讀01 MD:21、48，輸出簡式限定dilation=1／ceil_mode=False並說改設定需改公式；參數式限定groups=1、有bias、方形kernel | 已修正；原有shape／參數／MAC數值不變 |
| A3 | 重新讀04 PY:60–63與notebook完整case，保存empty_metadata、實際undo(empty_boxes, empty_metadata)、assert restored shape(0,4)及finite均已加入 | 已修正；round trips紀錄現有實際執行證據 |
| A4 | 重新讀04 MD:76與PY:69，文字明說原圖題Original 80x40（寬×高）、直式改Original 40x80 | 已修正；文字與case一致 |

04-coordinates修訂案例再次在 /tmp/accuracy-a/04-coordinates/ 實跑exit 0，stdout與最新JSON完全一致；最新case SHA256為028f44fe462b589165a2e8b3609051eae9ea924a51a4c2c3c1ed881dad245137，與JSON吻合；notebook完整案例與PY吻合、儲存stdout与JSON吻合。另在/tmp用wrapper記錄main實際執行的undo輸入shape，得到[(1,4),(1,4),(1,4),(0,4)]，確認空框逆轉路徑確實被執行，不只是新增未用變數。

同時複查額外採納的曲線標示改善：scripts/run_learning_extensions.py:100的橫軸已改Training step (pre-update loss)；01／03／04三張learning SVG均含此標示，三節MD均解釋第1點未更新、第40點是第40次更新前，更新後預測另算。最新三份learning JSON與本 reviewer先前在/tmp獨立重算的完整報告仍全部相等。新SVG每一條曲線的全部40點再次與JSON比對，最大y座標誤差仍小於0.000001；模型、optimizer、loss數值没有被曲線文字修改改變。

**最終結論：A1–A4全部關閉，八節在所述教學範圍內沒有尚待修正的技術錯誤。** 初審的完整逐節證據、合理簡化與未執行的發布後Colab流程界線仍適用。

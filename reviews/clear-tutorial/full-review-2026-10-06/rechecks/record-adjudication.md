# 本輪首次閱讀紀錄的規則符合性裁定

本裁定由未參與教材改寫的 AI 查核者作成。這是可回看凍結全文的**非盲讀紀錄查核**，不是新的首次閱讀，也不代表真人學生研究。只讀本輪 frozen 教材、manifest、first-read、reports、decisions 與本輪使用的 SKILL／protocol；沒有搜尋作者對話、舊試驗或任意 repository 歷史，沒有修改教材、skill、其他報告或原始 first-read，沒有訓練。共享檔案系統是指令與順序 gate 的隔離，不是技術存取隔離。

凍結來源 commit：`64a25d4fbcff5577965c29efbbcb5d9898ba95d9`。本文件的正文引文一律指向 `frozen/docs/`；工作樹修正不會倒填凍結版。結案時52個target頁已讀完，逐頁覆蓋與實際查核範圍列在文末。

## 規則與判定層次

- R1（四題焦點）：SKILL「四題固定檢查本節主要概念或整個方法；子步驟可補充，不能只答眼前小操作就判整體介紹完整。」
- R2（逐項證據）：protocol「每題的每個核心主張，都要引用當下已讀原文或指出圖中具體依據，不能只在整段答案後附一段概括引文。」
- R3（命名方法）：protocol「上位類別已定義，不等於新的命名方法已介紹」；首次正文教學需名稱、當下所需含義、整體角色與理由。
- R4（未知與時序）：protocol「到主要概念已被用來解釋機制、推理或實作時……必須將缺項具體列入問題，不能保留未知卻判整體通過。」合理待教須說明目前不需要它；後文只追加澄清，不改寫早期判斷。
- R5（分級）：protocol「重要縮寫未展開仍須記錄，但若角色與含義已清楚，不因名稱一項就升為主要概念阻礙」；有部分說明時只指出仍缺銜接。

本裁定分開 **紀錄／規則執行問題** 和 **凍結教材問題**。前者表示保存的證據不足以支持嚴格規則符合性，不能據此說正文完全沒教。後者才是正文在當下缺少必要介紹。紀錄問題的 severity 說明其查核影響，並不直接成為學生閱讀障礙。原有當場答案、問題及後來澄清全部保留。

## 已確認問題

### A01：22.1 已知 DINO 的名稱／角色仍不清楚，卻只以自監督與 view 操作完成作結

- 類型：命名方法檢查漏項；教材介紹／教學時序。
- Severity：**增加理解負擔**；完整名稱本身另分為**選讀改善**。不裁定整個 DINO、view 或自監督完全未教。
- 當場記錄：`first-read/vision.jsonl` 第126行，`22-views/03`，首次使用四題Q1回答「DINO是後續兩模型／投影頭比較分佈的具體方法，此時只概述」，unknown 是「DINO 名稱、兩模型的角色及投影分佈算法明示留22.3，尚未操作」，但 `issues=[]`。第128行頁末四題的主焦點仍是「跨view自監督方法」，未知仍含「DINO的兩模型、分佈／loss／center／溫度細節明示留後續」。
- 當時正文：22-views 第35行已教「把資料本身製造的關係當成訓練目標，叫 self-supervised learning（自監督學習）」；第37行已用命名方法說明「DINO 用兩份模型產生目標和回答，並把 CLS 經一個投影頭轉成分佈再比較。投影頭是讓特徵變成訓練用輸出的可學層」。這裡已有方法與部分角色，不只是頁末連結。
- 已讀背景：index 第27行「DINO（2021）是學影像特徵的方法，與同名的 DETR 系列偵測器不同」；learning-path 第86行已定位為2021自監督方法，且說第23章會接單物件定位；21-training 第124行已說下一節「保留它的讀圖路徑，換掉……學習訊號」。不能抹去這些背景，也不能把導覽定位當成 DINO 的正文介紹已完整。
- 已交代：無人工紅藍label的問題、同圖關係作目標、ViT產生CLS、希望保留共同內容、兩模型產生目標／回答、投影頭為可學輸出轉換、這一頁程式只生成view不更新ViT。
- 仍缺：DINO／無標籤自蒸餾的名稱意義，與上位「自監督」的關係；兩份模型產生目標與回答的分工如何構成此方法；ViT 是讀圖模型而 DINO 是訓練方法，這次要留下什麼表示供何種後續使用。角色已有概述，但還未連成命名方法的「是什麼→為何學→所得表示及用途」。完整loss、center、溫度公式及更新係數可後教，不要求22.1一次教完22.3。
- 本裁定只要求當下夠用的整體角色與預期表示用途，不要求22.1產生訓練好的checkpoint、提供optimizer／梯度數據，或證明DINO優於其他方法。原文本頁「先只生成view」的範圍是合理的。
- 規則依据：R1、R3、R4、R5。四題以 view／自監督為主並非全部答錯；問題在重要新命名方法沒有單獨辨識缺項，必要未知被一概當成細節待教，局部資料操作完整不能替它結案。
- 來源歸屬的具體斷點：第37行「其詳細機制留到22.3」指投影頭的詳細機制，不是DINO名稱意義與兩模型基本角色全數延期。當場unknown把三者合稱「明示留22.3」，把局部細節預告擴成整體介紹免查；這個擴張不由所引原文支持。
- 後來何處澄清：22-distillation 第9行師生分工、第11行投影用途，第106行才展開 `self-distillation with no labels`；22-distillation 第96行附近才區分完整續訓狀態與只取 teacher backbone 特徵。本查核將這些只記為後來澄清，不提升22.1當場理解。`22-distillation/01`已另列optional名稱問題，是後來成功記錄，不消除22.1漏抓。
- 處置：根據凍結原文的已知／缺項補一則獨立裁定；工作樹修正另做受影響單元複查。不能回填原始22.1筆記成為當場成功發現。
- 最終報告核對：`reports/vision.json` 的結論寫「唯一主要時序負擔是21.2……其他為名稱與小型算式／操作補充」。其全章復述可作讀完後理解證據；該「唯一」不足以涵蓋22.1的必要命名方法漏項。不能用後來22.3完整復述正確抵銷本輪22.1當場未列問題。`decisions.json` R003已正確註明本輪沒有成功獨立偵測，應保留此區別。

### A02：有 claims 陣列，不代表四題每個核心主張已有逐項引文

- 類型：**紀錄／規則執行問題**。
- Severity：**增加理解負擔（紀錄核對層）**。當場復述多數能從凍結正文核對，不據此判成正文缺教；嚴格R2符合性不足。
- 例一：`first-read/vision.jsonl` 第122行，`21-training/05` 頁末Q4答「train CE下降且128/128、新val/test各64/64、梯度及參數更新支持……；續訓以模型／optimizer差0、loss及RNG一致支持」。claims僅兩項引文：「這是一個 seed 的受控顏色任務，沒有架構對照」和「沒有重現原論文的 ImageNet／JFT 預訓練或自然照片成績」。兩句都支持結論限制，**不能支持學習、參數更新與續訓一致的核心證據**。
- 凍結正文其實有支持：21-training 第63–66行給train／val／test答對數；第68行說「每一步的整體梯度長度介於 0.071033～15.675162，沒有全為0」與「六組都確實改變」；第100–104行給模型參數差0、optimizer差0、接續loss及RNG相同。問題是當場答案漏附，並非教材漏寫。
- 例二：`first-read/foundations.jsonl` 第58行，`03-comparison/07` 頁末Q3答包含16×16紅藍材料、8train／4下移val、SGD.1、3步梯度與更新、40步學習、986參數／248840MAC及多3072次加法。claims僅引 `residual.load_state_dict(plain.state_dict())` 權重複製與40步結果表。**控制材料、訓練條件、成本與額外加法**未逐項引；它們在已讀正文03-comparison第2–3個單元本來有說明。
- 例三：`first-read/detector.jsonl` 第120行，`07-inference/03` 頁末Q1答框逆target、score=obj×最大class、score過濾、同類NMS；claims只引「decode_grid 回傳長度B的list」及評估／畫圖共用decode。兩引文支持介面與共用，不能支持**逆座標、計分及篩選順序**。已讀07-inference/01與/02有那些支持，當場未列。
- 例四：`first-read/evolution_a.jsonl` 第154行，`09-anchor-clustering/08` 頁末Q1說以1−IoU歸最近代表、每群mean迭代、k為群／槽數、cluster assignment與正負責任不同。claims只引「anchor只管寬高……換成wh再聚類」與mean更新是heuristic。它們支持材料與最優性限制，未支持**歸群／更新程序及兩種assignment分工**，而已讀/01、/02本來有完整教學。
- 例五（狀態精確性，severity為選讀改善）：foundations第11行 `00-warmup/00` Q3的核心主張「調後的w仍用於模型回答」標 `explicit_body`，只引「把輸入乘上w」。由本單元的「模型以w相乘」與「訓練根據錯誤調w」可以合理推出更新後仍按同一模型規則回答，理解本身成立；但這是兩句相連的推論，應列兩句與推論關係，而非單句已明說訓練後使用流程。此例只是紀錄狀態需更精確，不能據此說暖身缺少必要介紹。
- 規則依据：R2；每個核心主張須逐項引，不是每題任選一兩句。上述是核心方法或結果，不只是修辭和不影響結論的數字細節。
- 處置：原始答案作為當場理解證據保留；本文件列出來源可支持但當場未引用的主張。補充查核只能稱非盲讀证据補足，不能改名為最初已符合逐項引文。

### A03：BatchNorm 缺名曾漏抓，真首次位置的敘述需保留選讀註解的實際閱讀

- 類型：紀錄漏項／名稱時序。
- Severity：**選讀改善**。BatchNorm角色已清楚，不是主概念阻礙。
- 凍結原文：00-warmup 第205行（`00-warmup/07`）已教「BatchNorm（用平均與變異數把數值標準化的層）」及train／eval用批次統計／累積統計的分工。3.1第11行後來再次教channel尺度角色。
- 當場記錄：foundations `00-warmup/07` 的四題聚焦eval與no_grad，未列BatchNorm英文全名缺項；直到 `03-comparison/00` 第51行才列optional。foundations最終報告已承認「更早暖身/07已有行為簡介但當時未列名稱問題」。
- 裁定：這是**承認後來漏抓**，沒有以後文回改早期記錄，符合保存原始判斷的方向。查核初期decisions R009的「真正首次正文3.1」不能無說明排除本輪確實已讀的0章選讀註解；若專指主正文，須明列這個範圍。結案時R009已改成「真正最早出現在00暖身的eval/no_grad選讀註解」，並保留3.1／3.3同名說明；來源位置的文字已校正，早期首讀漏抓仍保留。
- 規則依据：R3、R5；選讀註解可不作主線必備，但本輪閱讀包保留並已讀，不應消失。不能以僅缺英文全名說BatchNorm用途完全沒教。

### A04：23.1 的 iBOT 選讀命名方法仍漏列名稱補充；KoLeo名稱來源另待查證

- 類型：紀錄漏項／名稱補充。
- Severity：**選讀改善**。不要求本頁教會選讀演算法的內部計算，也不判定DINOv2／Gram主概念阻礙。
- 凍結原文：23-dino-versions 第25行「iBOT的patch任務：把student輸入的部分patch藏起來，讓它預測teacher在那些位置的輸出分佈；teacher看未遮住的圖」；第29行「KoLeo：對圖級特徵加上鼓勵分散的約束，避免它們全擠在一起」。這兩個命名方法已有具體角色和目的，iBOT的英文完整名稱没有交代。KoLeo的名稱來源亦未交代，但本次沒有外查其正式命名，不把未補名稱來源直接視為必須展開縮寫的已確認違規。
- 當場記錄：vision 第150行 `23-dino-versions/01`，Q3已引用iBOT的功能，但 `issues=[]`。Q1未知只記「選讀SK／KoLeo等算法未教且明示不實作，未擅補」。第155行頁末仍只保留Gram anchoring中文名optional，說SK／KoLeo選讀算法未完整教。
- 裁定：完整演算法後教／不實作可以合理，iBOT的名稱補充仍需記。不能把不實作的範圍說明等同重要縮寫已展開。原文已有用途、材料／目標層級的說明，這裡僅為名稱漏抓，不把它擴大成「選讀方法全未介紹」。KoLeo名稱來源可另作optional查證／改善，不能為所有原生類別名或品牌名杜撰英文展開。
- 規則依据：R3、R5。後文沒有補iBOT全名；另由協調者查證正式名稱後決定補寫位置，不回填第一次筆記。

## 已核對而不升級為問題的情況

- `00-warmup/00` 尚未知更新算式：當下是建立手算任務與參數角色，下一單元接loss、再接梯度，未拿未知算式作推理。是合理待教。
- `01-small-cnn/00` 尚未知卷積細算：已有VGG小卷積堆疊與分類任務，緊接教共享局部窗口；不要求開場先懂卷積。
- VGG的完整英文名稱與經典CNN定位在已讀learning-path第26行已有，正文1章第13行又交代借用3×3堆疊／縮高寬，不能把未在本頁重複全名說成VGG完全無介紹。
- `07-targets/00` 與 `07-loss/00` 使用scope_overview：中心分配與三項loss已在實際前文05-assignment教過，套到本頁實際材料不須假設零背景。新梯度／全空分支另有first_use四題，頁末回到整個方法。沒有只因timing標籤不是first_use而自動判失敗。
- `21-attention/03` 完整Transformer／LN的排列性質缺銜接：原讀者已列burden，且明說attention本身已教；是合比例的時序問題，不能說QKV全未教。後文21.3的澄清只追加，不倒填21.2。
- `22-collapse/04` 未能重算分佈CE公式：當下可以根據作者明示的手工表格判定存在「低loss而跨圖特徵全同」的反例。完整軟分佈CE可下一節教；不自動要求本頁完整實作DINO或更新optimizer。若要求讀者重算表格才是需要提前補足的另一目標；本輪未把不會重算偽稱已知。
- CNN／ReLU／Adam／MAC、ResNet、ViT中文名／CLS等：原紀錄多將名稱補充分為optional，保留已有角色／機制。這些分級符合R5，不從名字一項推成整體沒有教。
- `23-dino-versions/03` 的Gram anchoring中文標籤：完整英文名稱已給，且第74行用中文解釋「用參考模型的patch關係約束student」及同view對應位置，中文含義已有。另加固定中文譯名可作optional的可讀性改善；它不是縮寫，最新SKILL沒有要求所有完整英文方法名稱都必須另造正式中文譯名，故不能把此批評算成必要含義完全未教。
- 原紀錄的網站、手機、外部來源與實作均大致明列未驗證；本次紀錄查核沒有做渲染／執行檢查，不能替它們勾選已驗證。

- 最後完成的 `12-assignment`、`12-dfl`、`16-dfl-free`、`16-inference-head`、`16-training`：主首次與頁末仍圍繞整個分配／DFL／直接距離／部署轉換／訓練策略，能分清材料、方法角色、產物、用途及實測限制；没有新增可支持的必要角色缺項。例如12.3把分配detach與原預測loss梯度分開；DFL先教softmax／監督／期望三步，沒有拿單邊參數更新當完整detector AP；16.3分清loss權重、STAL候選資格及MuSGD更新器，明說未執行完整MuSGD。這是所核對答案的判斷，不是每一附加主張已逐條證明。
- `16-dfl-free/01` 的STAL全名：原讀者已列optional；正文當場已說小物件擴選格子可產生負GT距離，基本角色已有。16-training後來展開 Small-Target-Aware Label Assignment，是後來釐清，不能取消首次名稱缺項。`reports/evolution_b.json`同時列target original optional=3及STAL later clarified，保留了這個區別；最後尚未釐清數2不等於最初只有2項。

## 最終裁定與查核限制

**全52個target頁的紀錄覆蓋已完成，但不能統稱最新規則下的嚴格首次盲讀合格。** 原始順序紀錄是當場理解的重要證據；A01顯示必要命名方法未知漏列，A02顯示逐項引文與來源狀態執行不足，A03／A04另有合比例的名稱漏抓。已教內容和合理待教均保留，沒有把上述問題擴成全書或整個方法完全沒教。這份非盲讀裁定也不能替代新的首次閱讀。

人工語意查核的明確範圍是52頁每頁至少一份主首次使用／scope overview四題及一份頁末四題，共104份檢查（416份答案）；另外查讀重要新名稱、unknown／issues、後來澄清及相關前文，必要時讀凍結全文。每頁在下表列出所核對的首份與頁末位置；同頁另有多份first_use，表中首份不是保證每個新概念的真正首次時點都獨立通過。A01另核對22-views/03等表外時點。查核也讀了六組報告的結論／範圍與相關問題、decisions中相關來源及處置，沒有將讀完後的正確復述倒填成早期已知。

機器核對範圍是全部995個保存單元及995個disclosure、全部212個checkpoints／848份答案與1680個claims：四題數、claims存在、允許的狀態值與inference理由、引用unit當時已可見、引文字句逐字位於其閱讀包，另去除空白／粗體與反引號後對照凍結來源的單元行範圍、disclosure指紋和先記錄再開下一段的時間。143份manifest来源SHA另與frozen逐一比對；六組gate最終cursor與manifest及實際行數相符，awaiting均為false。**這些只驗證格式、來源位置、聲稱狀態的結構及順序，沒有逐條證明1680個claim的引文語意支持，也沒有逐一證明explicit_body／prior_or_navigation／inference的語意分類正確。** A02的具體例正說明字句存在仍不足以支持其他核心主張；未列問題的補充checkpoints也不能據此一律宣稱R2通過。

報告的完成單元／目標頁範圍與保存紀錄相符；各組首次閱讀、前文範圍、未做技術／網站驗證的聲稱均作了核對。vision的「唯一主要時序負擔」需要A01補充，不代表本查核認可。evolution_a的零issues是其當場紀錄，不能解讀為A02引文完備。結案核對初期decisions R003的`current_round_detection`已承認本輪漏抓，`source`字串尚留「current vision first-read still pending」而vision已完成161單元；這項來源狀態文字已回報，協調者已同步完成／漏抓與A01獨立裁定來源。狀態校正不改原始漏抓判斷，也不是新的教材缺教裁定。A03所指出R009首次位置文字亦已校正。

本次未執行課程程式、重做訓練、外查原論文／名稱來源或渲染網站，因此不能裁定數據重現、實作正確、手機可讀或工作樹修正已通過。共享FS與時間gate不提供技術存取隔離，也無法僅由時間戳證明讀者未曾違反指令；六組access declarations是聲明而非技術監控。本裁定是AI紀錄查核，不代表真人目標學生研究。原始first-read沒有改寫；修正後複查、非盲讀證據補足及全新讀者首讀必須各自標明實際性質。

## 52頁覆蓋與結案快照

以下位置均為相應組 `first-read/<group>.jsonl` 的一基行號。每頁已核對首份主檢查及頁末四題；檢查數是該頁保存的所有checkpoints數，不是逐条語意證明數。

<!-- record-snapshot:start -->

結案快照時間：2026-10-06T05:34:42.697112+00:00。

|組別|保存單元|target單元|完成target頁|checkpoints|claims|gate|
|---|---:|---:|---:|---:|---:|---|
|foundations|58|48|6|29|223|cursor=58, awaiting=false|
|detector|140|82|13|55|464|cursor=140, awaiting=false|
|evolution_a|213|73|11|44|299|cursor=213, awaiting=false|
|evolution_b|254|41|8|26|226|cursor=254, awaiting=false|
|applications|169|29|4|17|201|cursor=169, awaiting=false|
|vision|161|63|10|41|267|cursor=161, awaiting=false|

|target頁|組別|首份主檢查／記錄行|頁末／記錄行|檢查數|
|---|---|---|---|---:|
|00-warmup|foundations|00-warmup/00 (first_use), 11|00-warmup/09, 20|6|
|01-small-cnn|foundations|01-small-cnn/00 (first_use), 21|01-small-cnn/08, 29|7|
|02-diagnostics|foundations|02-diagnostics/00 (first_use), 30|02-diagnostics/06, 36|5|
|03-identity|foundations|03-identity/00 (first_use), 37|03-identity/06, 43|4|
|03-projection|foundations|03-projection/00 (first_use), 44|03-projection/06, 50|3|
|03-comparison|foundations|03-comparison/00 (first_use), 51|03-comparison/07, 58|4|
|04-localization|detector|04-localization/00 (first_use), 59|04-localization/07, 66|6|
|04-coordinates|detector|04-coordinates/00 (first_use), 67|04-coordinates/06, 73|3|
|05-assignment|detector|05-assignment/00 (first_use), 74|05-assignment/05, 79|4|
|06-decode-nms|detector|06-decode-nms/00 (first_use), 80|06-decode-nms/05, 85|5|
|06-evaluation|detector|06-evaluation/00 (first_use), 86|06-evaluation/06, 92|6|
|07-data|detector|07-data/00 (first_use), 93|07-data/05, 98|4|
|07-targets|detector|07-targets/00 (scope_overview), 99|07-targets/04, 103|2|
|07-loss|detector|07-loss/00 (scope_overview), 104|07-loss/06, 110|5|
|07-training|detector|07-training/00 (scope_overview), 111|07-training/05, 116|4|
|07-inference|detector|07-inference/00 (scope_overview), 117|07-inference/03, 120|3|
|07-heldout|detector|07-heldout/00 (scope_overview), 121|07-heldout/05, 126|3|
|08-own-images|detector|08-own-images/00 (scope_overview), 127|08-own-images/07, 134|4|
|08-own-data|detector|08-own-data/00 (scope_overview), 135|08-own-data/05, 140|6|
|09-anchors|evolution_a|09-anchors/00 (first_use), 141|09-anchors/04, 145|4|
|09-anchor-clustering|evolution_a|09-anchor-clustering/00 (first_use), 146|09-anchor-clustering/08, 154|5|
|10-multiscale|evolution_a|10-multiscale/00 (first_use), 155|10-multiscale/04, 159|3|
|11-csp|evolution_a|11-csp/00 (first_use), 160|11-csp/08, 168|3|
|11-fusion|evolution_a|11-fusion/00 (first_use), 169|11-fusion/10, 179|5|
|11-augmentation|evolution_a|11-augmentation/00 (first_use), 180|11-augmentation/04, 184|4|
|11-iou-loss|evolution_a|11-iou-loss/00 (first_use), 185|11-iou-loss/06, 191|4|
|12-anchor-free|evolution_a|12-anchor-free/00 (first_use), 192|12-anchor-free/04, 196|4|
|12-decoupled-head|evolution_a|12-decoupled-head/00 (first_use), 197|12-decoupled-head/05, 202|5|
|12-assignment|evolution_a|12-assignment/00 (first_use), 203|12-assignment/04, 207|4|
|12-dfl|evolution_a|12-dfl/00 (first_use), 208|12-dfl/05, 213|3|
|13-dual-assignment|evolution_b|13-dual-assignment/00 (first_use), 214|13-dual-assignment/04, 218|4|
|13-nms-free|evolution_b|13-nms-free/00 (scope_overview), 219|13-nms-free/03, 222|2|
|14-feature-module|evolution_b|14-feature-module/00 (first_use), 223|14-feature-module/03, 226|3|
|15-attention-bridge|evolution_b|15-attention-bridge/00 (first_use), 227|15-attention-bridge/05, 232|4|
|15-area-attention|evolution_b|15-area-attention/00 (first_use), 233|15-area-attention/04, 237|2|
|16-dfl-free|evolution_b|16-dfl-free/00 (first_use), 238|16-dfl-free/06, 244|3|
|16-inference-head|evolution_b|16-inference-head/00 (first_use), 245|16-inference-head/04, 249|3|
|16-training|evolution_b|16-training/00 (first_use), 250|16-training/04, 254|5|
|17-capstone|applications|17-capstone/00 (first_use), 141|17-capstone/05, 146|3|
|18-video|applications|18-video/00 (first_use), 147|18-video/06, 153|5|
|19-tracking|applications|19-tracking/00 (first_use), 154|19-tracking/06, 160|4|
|20-deployment|applications|20-deployment/00 (first_use), 161|20-deployment/08, 169|5|
|21-patches|vision|21-patches/00 (first_use), 99|21-patches/06, 105|4|
|21-attention|vision|21-attention/00 (scope_overview), 106|21-attention/04, 110|4|
|21-transformer|vision|21-transformer/01 (first_use), 112|21-transformer/05, 116|4|
|21-training|vision|21-training/02 (first_use), 119|21-training/05, 122|3|
|22-views|vision|22-views/01 (first_use), 124|22-views/05, 128|3|
|22-collapse|vision|22-collapse/01 (first_use), 130|22-collapse/05, 134|4|
|22-distillation|vision|22-distillation/01 (first_use), 136|22-distillation/06, 141|6|
|22-features|vision|22-features/01 (first_use), 143|22-features/06, 148|4|
|23-dino-versions|vision|23-dino-versions/01 (scope_overview), 150|23-dino-versions/06, 155|5|
|23-detection-bridge|vision|23-detection-bridge/02 (first_use), 158|23-detection-bridge/05, 161|4|

合計完成 **52/52個target頁**；995個保存單元包含336個target單元及659個各組實際導航／先備單元，跨組先備重複不算額外target頁。212個checkpoints為149個first_use、11個scope_overview及52個page_end；848份答案內有1680個保存claims，其聲稱狀態為explicit_body 1424、prior_or_navigation 242、inference 14。

結構／引用可見性flags：`[]`；順序gate時間flags：`[]`；manifest凍結來源指紋flags：`[]`。沒有flags不撤銷A01–A04，也不表示每個主張已有充分的逐項引文。

<!-- record-snapshot:end -->

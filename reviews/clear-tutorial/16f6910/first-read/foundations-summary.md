# foundations 首次閱讀摘要

本組只透過協調者 gate 逐單位閱讀凍結版本 `16f69103d1f216d1024a40610623083557324701`。已完成 `foundations-0000` 至 `foundations-0109`，110/110 單位，14 個 target 頁面；gate 最後回覆 `READING COMPLETE`。實際範圍包含首頁、閱讀路線、第 0–6 章各節，以及 7.1 資料頁。沒有補讀其他未揭露來源，也沒有執行程式、training、GPU、Colab、網站或手機排版測試。

每單位均當場寫 understanding、elements、next_change_and_basis、issues、needed_visuals、visual_status、variation，並先 record 再 next。原始紀錄在 `/workspace/learn_to_yolo/reviews/clear-tutorial/16f6910/first-read/foundations.jsonl`；`/tmp/clear-tutorial-foundations-note.json` 僅是最後一單位的暫存筆記，不是完整紀錄。完整路徑由協調者提供，盲讀期間未自行搜尋 reviews。

目前記錄 8 項 burden、2 項 optional、0 項 blocking。所有疑點都仍待作者修正及首次讀者複查，沒有任何一項可稱 closed。

## 每頁問題與值得保留的部分

| target 頁面 | 單位 | 首讀問題清單 | 應保留的好處 |
| --- | --- | --- | --- |
| `docs/index.md` | 0000–0003 | **0002 burden**：負責格計算後，首頁又密集教 loss、分數、NMS、IoU、獨立資料評估。術語雖有短解釋，仍需回讀才能知道哪些只需先有印象。建議先保留訓練／推論／評估用途，NMS 完整刪框步驟放後節或明示選讀。 | Python class 與物件類別分清；人工 GT 與模型預測分清；中心壓格線歸右／下格、座標與索引順序說明很具體；CPU 與真實資料驗證範圍清楚。 |
| `docs/learning-path.md` | 0004–0009 | 無當場記錄問題。後半術語算法目前未知，但此頁是選讀導覽，沒有把尚未教授的算法判為主線缺漏。 | 各版本節不是持續疊在同一模型；人工框是代替模型輸出；主課與選修依賴及 Colab 可獨立執行／知識仍有前置的區分清楚。 |
| `docs/lessons/00-warmup.md` | 0010–0018 | **0012 burden**：兩段變化率接合後密集使用 Δ、略去平方項及極限，需把 e=-2、x=2 反覆代回。建議先給 w→預測（×2）→loss（×-4）的數字鏈，把极限推導選讀。**0013 burden**：五行主流程後集中 None／0、alias、detach／clone／item、部分斷圖仍可 backward 等陷阱，主線被長段沖淡。建議先展示一步結果，進階細節放選讀或診斷章。 | 一旋鈕從 w=1 到 1.8 的手算；backward 算梯度、step 才改參數；學習率不是實際步伐；改題需更新舊斷言而非刪掉；保存輸出與本人執行輸出有區分。 |
| `docs/lessons/01-small-cnn.md` | 0019–0026 | **0019 burden**：卷積尚未正式示範，歷史段先塞 VGG C／D、原版層數、512 通道、GAP 出處等，帶離紅藍分類問題。建議歷史細節選讀。**0021 burden**：只教感受野範圍與間距，突然出現原圖區間 `4i−6` 到 `4i+9`，偏移來源需猜。建議回推圖／小表或邊角精算選讀。 | 每層 shape 與參數／MAC 可代公式；permute 與 reshape 用實際小數列区分；錯誤圖板分圖編號與類別；3 步工程檢查與 40 步學習、單 seed 與訓練圖泛化限制清楚。 |
| `docs/lessons/02-diagnostics.md` | 0027–0034 | 無當場記錄問題。 | 四個檢查步驟與三個失敗的不同排序明示；None 不是小梯度；body／head 分開檢；a／b 人工資料與反轉 b 的 validation、數值大小與穩定關係都有定義；全對不等於有信心。 |
| `docs/lessons/03-identity.md` | 0035–0042 | **0039 burden**：x 已是多維圖張量，卻直接使用 Δy／Δx 與 scalar a／b 鏈乘，未先說明是一維直覺；不清楚改整圖或單值。零分支單值例後面可核，但一般分支推理範圍仍需猜。建議先明示單數簡化，或用 scalar F 例子。 | 加法與串接；shortcut identity 與整 block identity；後 ReLU 改負值；人工零分支與另建隨機 block 的兩個實驗目的；零初始化不能直接當訓練建議；broadcast 通過不保設計正確。 |
| `docs/lessons/03-projection.md` | 0043–0050 | 無當場記錄問題。 | 321 核通道混合、位置探針核取樣、shape 核兩路對齊，三種證據互補；1×1 只不看空間鄰居、仍跨所有通道；實驗材料／模型另建、奇數 H、不同設計與寫錯區分清楚。 |
| `docs/lessons/03-comparison.md` | 0051–0061 | **0054 burden**：公平比較與手算成本已建立，完整 forward hook／modules／handle／旗標實作支線帶離捷徑是否有用的問題。建議 hook 選讀。**0058 burden**：分布平方平均、交叉項抵銷、ReLU 半量及開根號幅度在一長段推導；a²／3 只說可積分，需停下回算。建議逐 block 特徵／梯度實測小表或圖先行，機率推導選讀。 | state_dict 複製相同起點，seed 不是自動同權重；紅藍同位置避免偷猜；副本暖機不先改正式模型；3 與 40 步證據、梯度大小與準確率、此例梯度消失與論文退化区分完整。 |
| `docs/lessons/04-localization.md` | 0062–0070 | **0068 optional**：真正 3／40 步框僅給小數座標及存圖路徑，網頁可見圖是人工框和 loss 曲線；要在腦中疊回才能了解 IoU 約 .69／.78 偏在哪。建議在 40 步段並排真正 GT／pred 疊圖，保留人工手算圖角色。 | 半開框、xyxy→cxcywh→正規化、兩 head、sigmoid 四座標不是四類機率；分項 loss 和單位；IoU 與 MSE 對小框不同；兩張圖顏色可背框所以不能證明 flatten 優於平均，限制說清。 |
| `docs/lessons/04-coordinates.md` | 0071–0078 | 無當場記錄問題。 | 原圖／畫布／畫布正規化三座標系並表；先扣 padding 再除 scale；空框 shape；round 與實比例；能往返不等於貼內容；像素內容檢查可抓切片錯位；各圖 metadata 與裁切後還原區分清楚。 |
| `docs/lessons/05-assignment.md` | 0079–0086 | **0081 optional**：責任圖先出 `ignore 0`，正文下一單位才教；當時要猜第三種格的意思。0082 已首次釐清，原始未知記錄保留，沒有回寫。建議圖註「本例無忽略格，下段介紹」或先只列正負格。 | 格／slot／候選此模型同義；正樣本指格非圖片；框內 xy 與整圖 wh 尺度；GT、target、人工特徵角色；負格不學框／類、共用權重仍可變；同格容量不由 NMS 解決；不對稱例抓顛倒很有用。 |
| `docs/lessons/06-decode-nms.md` | 0087–0094 | 無當場記錄問題。 | 非對稱解碼；人工 logits 倒推；score 是排序訊號非 precision／校準概率；三比較四門檻；候選索引按格、NMS 按分數、重新篩選重編號；提高門檻只留高分錯框的反例有效。 |
| `docs/lessons/06-evaluation.md` | 0095–0102 | 無當場記錄問題。 | 同圖同類、全圖 score 排名、GT 只配一次；刪框需重配；P／R 分母；PR 點與右側包絡圖；AP 不是通用 P×R；all-points、11 點、COCO 定義差異與 None 處理；用 GT 手動删錯框不能當部署策略。 |
| `docs/lessons/07-data.md` | 0103–0109 | 無當場記錄問題。 | 回到首頁同一紅框有助串主線；藍是類 1 卻通道 2；target 變數此處原 GT 與後續每格 target 區別明說；影像 stack／標註 list；空圖不是零框占位；固定逐值重畫與合法範圍框的語意錯誤；固定局部和仍通過不證整物件正確。 |

## 主要閱讀斷點與圖解需求

最常見斷點是主線已可手算，接著一段載入很多歷史、程式機制或數學推導。這不是缺名詞定義，而是先看懂現在做什麼後，還要重新找回那個問題。暖身、CNN 與比較章可先給核心數字鏈／結果，再將 None 與斷圖變體、hook、歷史版本及尺度統計推導放選讀。

空間概念的圖需求集中在 CNN 卷積掃窗、感受野與邊角回推，以及逐 block 特徵／梯度變小；目前該幾頁可見圖多是結果圖板或 loss 曲線，沒有直接回答上述空間或機制問題。這些列為減負或精算說明需求，未判 blocking。定位章補真正模型框疊圖會讓成果更可見。

本書較強的部分是材料角色與核對目的：原 GT／每格 target／人工模型輸出／真正模型預測多數明說；圖索引與類別編號、xy 與 row／col、pixel 與比例、更新前 loss 與更新後 accuracy 都有具體例子。非對稱資料、位置探針與重畫像素檢查比只驗 shape 更能讓首次讀者明白「為什麼這個證據能抓到錯」。

## 實際視覺範圍

我使用 `view_image` 實際看過下面 14 個 gate 提供 PNG preview。這些是原 SVG 的靜態瀏覽器呈現，沒有做 Zensical、實際網站、手機排版或互動測試。每張必要文字、框、軸與圖例在本次工具顯示都可見，沒有記錄裁字或看不見的必要框。

| 頁面／圖 | preview 檔名 | 閱讀效果 |
| --- | --- | --- |
| 首頁中心責任格 | `0420f5474adeda77.png` | 壓格線中心歸右格、座標／索引清楚。 |
| CNN 三步前四圖 | `2a7a319bcaa9499a.png` | 圖編號、GT／pred、錯誤色清楚。 |
| CNN 40 步 loss | `242ee1b4499ea43a.png` | 時點與下降趨勢可讀。 |
| 診斷 a–b 平面 | `f2d871f65f501a94.png` | 實心訓練／空心驗證、反 b 箭頭、分界線有效。 |
| identity 兩路 | `4bbb2a7ace4779e6.png` | 兩支 shape、加號與省略後 ReLU 的位置清楚。 |
| projection 兩路 | `4676372541e9b5dd.png` | F／P 設定、權重與對齊 shape 清楚。 |
| plain／residual 40 步 | `64baa3616300d728.png` | 兩條 loss 線與圖例可比較。 |
| 定位人工 IoU | `e8b2a5f8c2e41f3b.png` | 紅物件、綠 GT、黃人工 pred、交集／聯集清楚。 |
| 定位分項曲線 | `f814f5a5079aa329.png` | 左總與分類／加權框、右框獨立刻度有效。 |
| 座標往返 | `02646adeba1ea876.png` | 灰色示意补邊與真正补 0、原圖内容区清楚。 |
| 第 5 章中心分配 | `d34fb5ca03d504a5.png` | 中心／正格／target 有用；ignore 早於正文定義。 |
| NMS 前後 | `6d5fb8195571c3a7.png` | 留錯紅、删重複黃、正確藍與 GT 重合可見。 |
| PR 包絡面積 | `ec7046c52826453e.png` | 點號／排序與包络、原折線非面積、漏檢段有效。 |
| 7.1 半開資料框 | `b2bcab88a92808d4.png` | 左上／右下边界、RGB／類別与索引說明清楚。 |

preview 都在 gate 提供的 `artifacts/runs/clear-tutorial-16f6910/figure-previews/` 下。未看程式另外存出的 `artifacts/01-small-cnn.png`、`04-localization.png`、`04-coordinates.png`，未讀連結 JSON、源碼、notebook、外部解釋或未揭露章節。

## 原始紀錄的閱讀者自我更正

0100 的「下一動作」筆記曾無依據猜下一練習會補 GT2；0101 當場發現實際題是刪高分 FP，已明說更正而保留 0100 原始猜測。這是閱讀者預猜失誤，沒有列為作者問題。少數早期原始筆記混有簡體用字；本摘要以繁體中文整理，未回寫覆蓋原始當場紀錄。

未做任何技術執行或網站排版測試；對程式正確性、Colab 可跑性、模型效果、官方評分一致性及真實圖片泛化都屬未驗證。

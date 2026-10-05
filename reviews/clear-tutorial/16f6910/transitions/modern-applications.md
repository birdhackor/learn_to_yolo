# 第三輪前後銜接審查：modern／applications 全 12 頁

審查者：`clear_first_evolution`。日期：2026-10-05。基底 commit：`16f69103d1f216d1024a40610623083557324701`；實際審查的是修改後工作樹，內容指紋見末段。我沒有撰寫這 12 頁，也沒有擔任這組的第二輪技術審查者；第二輪負責的是 foundations 組。本輪是可以回查全文的第三輪，不是首次盲讀，不把第一輪紀錄改寫為通過。

結果：在本輪實讀與獨立 SVG 預覽範圍內，12 頁沒有未閉合的主要概念阻礙或必要圖解缺漏。原始三個 burden 已閉合；一個 optional 已在近圖數值表處理，逐框直接加 TP／FP 字樣仍可選。新發現第 16 章兩處回顧把 YOLOv10 初選與 YOLO26 最終篩選混在一起的 burden，作者修正後已實際重讀核回。

## 實際方法與未驗證範圍

先完整讀現稿 `12-assignment`、`12-dfl`，再依實際導覽順序讀 13.1、13.2、14、15.1、15.2、16.1、16.2、16.3、17、18、19、20。每頁當場記任務、材料、單位、意義改變，以及自己提出的小變化預測；這些筆記先於閱讀 raw issues，另存 `modern-applications-read-notes.md`。完成全部現稿後，才讀 `modern.jsonl`／`applications.jsonl` 的 target issues 與圖解需求；沒有先用原問題或作者交付稿套答案。

12 張 SVG 均以 Chromium 的 `page.set_content` 讀入實際 SVG 原文、依 viewBox 設 viewport 產生預覽後逐張實看。這是獨立圖預覽，**沒有驗證實際 Zensical desktop／mobile 頁面、MathJax、導覽或手機縮放**。第 18 章小字放大入口只核了 Markdown 連結與圖本身，沒有實際點擊網站。

本輪沒有重跑 CPU 訓練、ORT、影片檔／相機或 GPU。逐頁的小變化都是依已讀材料提出的預測與推理，不冒稱新實驗。另用唯讀腳本核對 12 頁頁尾 stdout 與對應 CPU evidence JSON 完全相同，12 份 evidence 的 `case_sha256` 均對應當前 case，且紀錄的 `passed` 為 true；這只證明材料對應，不能取代第二輪的獨立技術驗證。沒有改 docs、coverage、舊 raw 或舊報告，沒有 commit／push。

## 已讀前文是否真的提供背景

`12-assignment` 實際提供 owner 是 GT 清單索引、與 class ID 分開，以及嚴格 inside → `score×IoU²` → top-k → 衝突的順序。它明說教學規則與官方 CIoU 解衝突不同；target 生成的 detach 不等於把供 loss 用的模型輸出也 detach。第 13 章承接後，不需猜 owner 的方向或把獨立 sigmoid 兩類誤當總和為 1 的 softmax。

`12-dfl` 實際提供相鄰 bin 加權 CE、`sum(k*p_k)` 期望距離、距離格乘 stride 解碼、K 個 bin 最大 K−1、非零 target 熵下限和 raw 輸出數。第 16 章直接距離與 reg_max=1 的 identity 因而有可回查的基礎；「單 bin softmax 得零」與「直接輸出一個 signed 距離」已能分辨。

## 逐頁獨立理解與小變化預測

### 13-dual-assignment（原 target `modern-0061`～`modern-0065`）

任務是理解雙重監督怎樣連到兩個 head，先看官方 top-1 初選和衝突，再切到分類梯度 toy。兩個完整 head 各自還有分類／框分支，並非把第 12.2 的兩分支改名。many loss 更新共享特徵；one 輸入 detach，但自己的 head 仍學。官方 CIoU 解衝突與品質表示意、toy 全域最優、官方軟 target 都在需要的位置限定。官方反例 `[2,2]` 使 C 有兩個正樣本，已明說不能由名稱保證 final 每 GT 最多一個；第 16 章規則不同也有提示。

小變化：只把 A 的 p1 品質 .85 改 .92，B 列不動，toy 全域最優是 A→p1、B→p0，品質 1.80，owner `[1,0,-1]`；品質 top-1 示意此時也沒有衝突。一對多 top-2 的 p0 仍因 A .90>B .88 歸 A，owner `[0,0,1]`。由此可分辨「排名公式一致」與「每候選 target 一致」。

圖實看：`13-dual-head-flow.svg` 的前向、反向、detach 停點和只留 one 的推論能沿箭頭追；官方／toy 差異也有標。原 `modern-0062`／`0064` 及 `0063` 圖需求核回見下表。沒有新未閉合問題。

### 13-nms-free（`modern-0066`～`modern-0070`）

任務由網路與 assignment 轉成兩套獨立的四個可學 logits，固定框、不接 CNN、不學框、不重現 assigner，入口已說清楚。GT A／B、四候選、p1 duplicate、唯一 target 差異、score／NMS／matching 三門檻、top-k 的重複反例與人工 AP 都有材料。AP 5/6 是手算，不是 held-out 模型成績。

小變化：把 B 及 p2 一起由 x=20～30 移到 2～12，p0／p2 是兩個真物件卻有 IoU=2/3>.5。原 many 高分加 NMS 會錯刪一個、recall=.5；固定 one target 的 logit toy 可以留下兩個，但不保證真模型會如此。相交真物件不能等同重複候選。

圖實看：四固定框、GT 區間、兩套 target 和 .959／.041 分數能相互對照。raw 本頁沒有 issues，本輪也沒有需要猜的任務切換。

### 14-feature-module（`modern-0071`～`modern-0075`）

任務從候選去重轉為同 shape 的特徵模組，材料是隨機 `[2,8,8,8]` 與左鄰 roll target，非偵測框。project→split a／b→b1／b2→concat→fuse 的形狀閉合；concat 段的直接梯度與特徵總梯度有近處定義，避免用 b1 總梯度誤證保留路徑。循環 roll 與零 padding、窄通道容量、zero 輸出 baseline 都已交代；MSE 下降不能冒稱學會平移，更不能比較 AP。

小變化：channels=8、blocks=2 不動，只把 hidden=4 改 2。project 為 8→4，四份各 2 channel、concat=8、fuse=8→8；參數 `36+4×38+72=260`。理論感受野仍 1／1／5／9；通道變窄不能推出 accuracy 更好。

圖實看：a／b／b1／b2 分段和 concat 箭頭可追。raw 無 issues，本輪無未閉合問題。

### 15-attention-bridge（`modern-0076`～`modern-0082`）

任務改成內容決定的空間讀取。2×2 四 token 是明確材料；attention head 與偵測 head 在第一次使用時分開，Q／K 決定讀取比例，V 提供內容。source 軸 softmax、內積非 cosine、token 回圖順序、12 個可學矩陣元素與 16 個當次 affinity 中間值均有定義。channel 1×.5 是人工重建梯度 probe，非偵測表現；N² 配對數與 FlashAttention 儲存差異也有限定。

小變化：K 投影設 0，Q／V 保持 identity。所有相似度為 0，每個 query 均勻讀四份，各 1/4，輸出 token 全是 `[.5,.5]`。均勻性來自 K，並非先對 V 做 softmax。

圖實看：第一個 query 指向四來源的比例與加權結果可以追。raw 的新 token 重畫屬可選，現有完整算式已足夠；無必要圖缺漏。

### 15-area-attention（`modern-0083`～`modern-0088`）

任務從 full 改為 area，改的是可配對位置，不是再分 channel 的 attention heads。row-major 4×4、A=4 是四橫帶，不是四象限；位置卷積／BN 拿掉後才做跨區干預，理由在實驗前說明。q0=0 讓權重均勻，即使修改 token15 同時改 K，q0 內積仍為 0。單層 area 不響應不能推成完整網路永不跨區，後面的卷積／BN／full attention 有另行說明。

小變化：A=8 時，每區 2 token、pair=`16²/8=32`；token0 只讀 0／1，原輸出 .03125。改 token15+5 不影響它；改 token1+5 則增 2.5。可學參數仍 12，速度不能據此保證變成 1/8。

圖實看：A=4／A=2 橫帶、query 與改動 token 的位置清楚。raw 無 issues，本輪沒有依賴未揭露概念。

### 16-dfl-free（`modern-0089`～`modern-0096`）

任務比較直接距離與 DFL，再區分 softplus toy 與官方 signed 距離。reg_max=1 的 identity、K=16 最大 15 格、四距離格乘 stride、右距離 18 超限、其他 stride 可表示同一個 144 px 都明確。SmoothL1 toy 不冒充官方 CIoU／normalized L1。外點 signed 負距離先連到 STAL，框合法仍須 l+r、t+b 正；不能由較少輸出數推論 latency 或 AP。

小變化：GT `[74,64,228,108]` 不動，候選中心由 `(84,84)` 改 `(116,84)`，stride 8 的 target 變 `[5.25,2.5,14,3]`，全部進入 K16 範圍，raw 數量不變。超限也取決於候選位置。

圖實看：中心 `(84,84)`、ltrb 格／畫素、x=204 的 15 格邊界和 x=228 真框對得上。raw 提到的負 l 外點圖是可選；近處已有數值與合法寬條件，無必要圖缺漏。

### 16-inference-head（`modern-0097`～`modern-0102`）

任務是真正移除 many，並核對部署保留的權重和 raw；不是只讀 `['one']`。332→278 參數、已更新 one 的 deepcopy、same-PyTorch `torch.equal` 和三個輸出 shape 有材料。人工解碼 probe 不證明真實候選排序；平方 loss 不教 assignment 或唯一物件；single-label top3 不冒稱官方 box／class two-stage top-k。官方 API 預設 many+NMS 與論文 one 預設分開說明，NMS-free 需 `nms=False`。

小變化：只把輸入寬改 128、高仍 64，pool4 使 raw shape 不變，但 x 格心變 `[16,48,80,112]`、y 格心 `[8,24,40,56]`，距離須乘 `[32,16,32,16]`。硬設單一 stride16 會讓 x 座標縮半且不報 shape 錯。raw 等價不能代替幾何契約。

圖實看：訓練梯度停在 detach、部署沒有 many、解碼前逐值比對的位置明確。新 T01 已作者修正並重讀：YOLOv10 初選非 final 保證；本章 YOLO26 topk2=1 最終至多一、可零。

### 16-training（`modern-0103`～`modern-0108`）

任務先分三項技巧，只實驗兩個獨立 Linear head 的 loss 權重，STAL 只算資格、MuSGD 只說明。progressive／固定 .8/.2 同時差 sum b 和排程，等總量 .45/.55 對照把兩者拆開。首步 forward／MSE／已乘 b 梯度／SGD、更新前 history 和更新後評值都有數值；eigen／H 推導標選讀，非主例先決。STAL 擴大資格框、不改 GT，signed 距離接回 16.1；topk2 最終篩選也與第 13 章分開。

小變化：總 loss 乘 2，同時 SGD lr 由 .05 改 .025。此無 momentum／weight decay 的 toy 每步更新仍相同，`.025×2×grad=.05×grad`，首步與最終 MSE 不變；總 loss 與已加權梯度會乘 2。不能照搬到 Adam／MuSGD。

圖實看：原 `[7,7,9,9]` GT、資格框 `[0,0,16,16]`、四格心與影像 y 向下、0→4 只是進池都有標。T01 在主實驗動機入口修正後已核回。

### 17-capstone（`applications-0065`～`applications-0071`）

任務回到可交付的靜態偵測需求，明確使用第 7 章 grid MiniYOLO，沒混 DFL／attention。32／16／16 split、seed、box5→10、validation 差 .01 保留規則、chosen test 只評一次有完整協議。coverage 不是 TP／recall、四種 FP、四門檻和各 report 欄位有近處定義。1.03／1.12 秒只計 fit loop；2.96 ms 含 uint8 轉換、模型、NMS、放大與 GT／預測畫框，不是純 forward。正式兩次只差一項，總 loss 大小不可直接比較；單次小資料不能證明穩定改善。

小變化：只把 display threshold .25 改 .5，evaluation cutoff .05、訓練、NMS 不動，權重、mAP、keep_change、chosen test 不變；圖少畫 .25～.5 候選，畫框計時可能變。畫面乾淨不等於 precision 提升。

圖實看：#4／#14 高分定位錯誤與 #10 低分背景 FP 可以對上近圖表；後者評估 cutoff .05 與上列顯示 .25 分開。原編號 burden 已閉合，optional 表已處理，但沒逐框直接寫 TP／FP。

### 18-video（`applications-0072`～`applications-0081`）

任務把同一模型放進逐幀管線，新增 Frame 時間戳／RGB、來源寬圖 letterbox、queue 和釋放。generator 先用 count_up 定義；lazy 來源與小示範 list、no_grad 裝飾器、finally／closing、runpy／session 差異都在程式前說清楚。來源 FPS、處理速率、擷取到完成的延遲三者分開；假來源不真等 50 ms、8 幀無框是漏檢而非掉幀。影片像素 decode 與框 decode、FFV1 檔案證據及未測相機／VFR／PTS都有界線。

小變化：保持 12 幀與畫面，只標成 25 FPS，最後時間 .44 秒、總播放 .48 秒、GIF 每幀 40 ms、每秒位移 100 px。boxes 不變，模型 ms 不由 timestamp 決定。若是真的 25 FPS 來源而每幀處理 50 ms，無限排隊第 k 幀延遲為 `50+10k` ms；現有 lazy 示範沒量這種排隊。

圖實看：第 0／5／11 幀、來源 0／.25／.55 秒、有框／無框及單次 ms 對得上。原尺寸圖放大是現稿明示的可用讀法，網站點擊未驗。raw 無 issues，本輪無必要材料缺漏。

### 19-tracking（`applications-0082`～`applications-0089`）

任務從偵測框轉到跨幀身份。track 四欄位、update 五步、矩陣列欄與 ID、門檻 .1、max_age=2 只容許連漏一幀、速度乘幀差、列舉先 pair 數再 IoU 都能手算。主例是人工同類框及一次人工漏檢，11/12 detector recall 不隨 tracker 改；3→0 switches 只來自配對。接第 18 章真模型前又明示改任務、只篩 class0、空幀更新、篩選後 boxes／IDs 同步；無 GT 不報正式 MOT 指標。ByteTrack 低分輪只接仍 Tracked 的未配 track，Lost 不進，現稿已限定。

小變化：只反轉第 2 幀 detection 順序，框與真實身份評估同步，motion 的回傳 IDs 成 `[2,1]`，但 B 是 ID2、A 是 ID1，物件 ID 沒換。不能把 list 順序變化算成 ID switch；若依舊 A／B 索引評分才會人工算錯。

圖實看：六幀／兩方法、A／B 分上下只為標籤，實際 y 相同的圖注，以及顏色代表 ID 非 class，均清楚。raw 無 issues，本輪無未閉合問題。

### 20-deployment（`applications-0090`～`applications-0099`）

任務改成 backend 匯出一致性，明示回到第 7 章 GridDetector 一步 SGD 固定權重，非 16.2 one head、非品質評估。ONNX 只包模型；不同原圖的 metadata 分別保留；checker／raw／還原框三層和防空對空的框數 assert 有材料。動態 batch 非動態空間，80×80 拒絕及 AdaptiveAvgPool 追蹤限制清楚；raw／整段 scope、paired 差、湊 batch 的服務延遲分開。trtexec 未執行、另一次 L4 Python 實測、allowed FP16 非全層 FP16、CPU／GPU 權重不同、無 AP 都直接寫明。

小變化：匯出與 raw 不動，若 ORT 後處理誤用第一張 metadata 還原第三張 48×80 直圖，checker 和 raw allclose 仍過，但框座標比較會失敗；補邊方向與縮放不同，labels／scores 卻仍可相同。三層檢查各抓不同的錯。

圖實看：model-only 藍區、metadata 旁路、各 backend 的匯出／build 虛線與三層紫色位置均對到正文。raw 無 issues，本輪沒有由任務切換而必須猜的條件。

## 原始 first-read 問題逐 ID 核回

以下是現稿的複查結果，不改原始 severity／原文，也不把 first-read 單位改成 pass。

| 原 ID／原嚴重度 | 原問題與當時理解 | 現稿實際核回位置／方法 | 結果 |
| --- | --- | --- | --- |
| `modern-0062`／burden | 「本例的一對一要求每個 GT 恰好配一個候選」先講 exact／greedy／Hungarian、後見官方 top-1，需記兩套定義。 | 重讀 13.1「官方的一對一」→「程式 toy」→選讀。官方先走 `[0,-1,-1]`／衝突；toy 入口明說改全域最優、用途為 target 與梯度；列舉／貪心／Hungarian 折疊選讀。獨立小變化也能同時預測兩套結果。 | burden 閉合；可只讀官方與梯度主線而跳過求解算法。owner 對照表足以辨方向，額外 owner 箭頭仍可選。 |
| `modern-0064`／burden | 常見錯誤「圖上有兩個物件，就該有兩個正樣本」像再保證官方 final 恰一個。 | 重讀 13.1 常見錯誤：toy 僅候選足夠時各配一；官方 top-1 限制初選，衝突改 final 數量，不能推出兩正／推論兩框。主文另明確三 GT 的 `[0,0,2]` 限制。 | burden 閉合；沒有把另一個「至多一個」錯誤保證代替舊句。 |
| `applications-0067`／burden | class0「#4 那兩個高分定位 FP」不知 # 是圖片或候選。 | 重讀 17「由失敗提出可測的假設」現稿明寫 **validation 圖片 #4**，緊接「#4 是圖片編號，不是預測編號」。圖各欄也寫驗證圖 #，report 明細 image=4／prediction=0、1。 | burden 閉合；回查位置不需猜。 |
| `applications-0068`／optional | 比較圖不能直接看出 #4／#14 是否過 IoU .5，希望至少示例補 TP／FP／IoU。 | 實看 SVG，再讀緊接圖的數值表：#4 baseline .4918／.3034；#14 baseline .2275、改動 .4917，均判定位 FP，並說 #14 看似接近仍未過 .5。 | optional 已在近圖說明處理；沒有逐框直接加 TP／FP 標籤，仍可作可選圖強化，不是必要圖缺漏。 |

`modern-0063` 雖沒有正式 issue，當時圖解需求是雙 head、detach 和推論保留 one。現稿新增 `13-dual-head-flow.svg` 放在入口，我實看箭頭與解說，需求已在當下提供；不用等第 16 章才看到。`modern-0095` 的 signed 負 l 外點圖、`applications-0068` 的逐框標籤皆保留可選，不冒稱作者已加入圖。

## 新發現 T01：版本回顧需要區分初選與最終篩選

位置是 16.2 回顧 13.1 與 16.3 Progressive Loss 開頭。修前原句分別為「一對一（one-to-one）讓每個物件只分配一個正候選」和「13.1 說過…one-to-one 讓每個真值物件最多只選一個正候選」。當場已讀 13.1 的 YOLOv10 final 可能 `[0,0,2]`，因此不能把這兩句理解為相同實作的最終保證；須猜它是初選、設計目標，還是 YOLO26 的後篩選。後文 16.3 STAL 的 topk2=1 才釐清 YOLO26，但對 16.2 的讀者太晚。

嚴重度：增加理解負擔。建議是在回顧當處寫 YOLOv10 top-1 初選／減重複目標，以及本章 YOLO26 衝突後 topk2=1、最終至多一個且可能零個。

作者修後，我實際重讀當前 16.2 回顧段（當前第 20 行）與 16.3 第 29 行，並對照已讀 13.1 的反例和 16.3 STAL 完整段。兩處現在都明說版本與規則不同、YOLO26「至多」一個且可零個；沒有再宣稱恰一個。T01 閉合。這是對修改後原句的核回，不以作者回報代替閱讀；沒有重跑無關訓練或改 first-read raw。

## 內容指紋

正文 SHA-256 的定義：原始 UTF-8 bytes 中 `<!-- curriculum-evidence:start -->` 之前的全部內容，不 trim、不格式化。全頁 SHA 包含頁尾 evidence。以下快照取自最終核回後當前工作樹；指紋是本輪實讀來源，不把 base commit 當成修改後內容。

快照 UTC：`2026-10-05T10:41:09.224763+00:00`。

| 頁面 | 正文 SHA-256 | 全頁 SHA-256 |
| --- | --- | --- |
| `12-assignment` | `01a2bc21d12bb2ccab144f6da5d9213537e65889c0c3777738c6c9abeda8c893` | `819308138126eef1825a41685d19a79d75ca6af50a30fd9c5747e539344731a5` |
| `12-dfl` | `5994a6d930b8483cb11647e0e24abbbc511925af69f5bef9436d22439744df4c` | `670eb809234ae66e1d1590142fd8ec0cf8b15aa1045c0720ad40a8627420a256` |
| `13-dual-assignment` | `bfed914525f9c363b0dee640d094c2cfb50aa7f25cba09bad4aed2a89ea6a362` | `23d265ae9a2236832e4fca60cdcddfe27972dda8fd4d32d184588ed206df97b6` |
| `13-nms-free` | `80f0220e7cfe9c9c5ad5d6734e83610faec1e7559a143fa59b649dd22be00dbb` | `023323d044c1253c07a4adc24a420107d8851de3a4674a9d71cf2704868a10ec` |
| `14-feature-module` | `c48c2362d3aad2cf701336250f8773c4b500cefedab7982ae875962aaabf1f77` | `f9a8f4240d4f095f15507f3a3e607e5c7362f106ce77540f89254a448649ca29` |
| `15-attention-bridge` | `fd343012264c8daf4b5ebc34f3621802e78135667682fa2e95fd991c764463e0` | `eba51744673bf59c6c1e9cc913d3bb2d83faa73ed7463465b4f0728682f4cf8e` |
| `15-area-attention` | `cf5ba48a0c0837b2938022137f60b53a4a2dcb821fc1f6f71a8b4653144c8d00` | `1870d643f21078b9c40238f3c4d1eedb3fa3ee147897b6ef7927bb549a6e6436` |
| `16-dfl-free` | `2899ada67ed83a6464e8bdc0ec7c250c1c76932c91e843a3087ffaf8e3e7cf21` | `8243eb1124300e1bc1aec55272418dfc3b9452dbc8cd9c105f62fd10e3518ce0` |
| `16-inference-head` | `cb8a76113bc8c999295ab167cbab45896c937495245758b732c16ab4564838e2` | `4251983ae7b3f6685b6d8d6071d345308e6079f4b5681c571f00ef9083ff129c` |
| `16-training` | `2253a37ac8c65a5ac030c6d763d3d5dab7b5d919e55ca728125c077fdd786fee` | `c84b993d07e157ce382e42d5635bc3532b84b853d955695de55c1dabce2ea37a` |
| `17-capstone` | `c99d7d3e0a7255148383d716c4ef0a66c3d4c714e5ce705218689871fc09ed7a` | `25c66ab672af41f9c08226a89b5f17c193b6c0ef26a84d583612349f0fe4445d` |
| `18-video` | `66553aff4a6d5f61e1ad6182dfe60432277968a2111e67067f82d77290c22813` | `aa37572bcc725c62156a6624fd32a30249eecb26d14faf842b222eb41aecc269` |
| `19-tracking` | `497782300cf46a9e56d6328518562274b0978c226349a2539627e6034d55914c` | `7d8ba461b6d0a132a111ceafcb812a6761eccf57046df1f329b3950c2822db23` |
| `20-deployment` | `0564f2e79e19c0b6ff4ae1a839174fb0b7e3041a6bb163fdc2e93c9b8260d563` | `28e152b135fb2af9310e78d0cc3cea7880713cf0b10a8591fc4af01ad6b1825a` |

| 實看圖檔 | SVG SHA-256 |
| --- | --- |
| `docs/assets/diagrams/13-dual-head-flow.svg` | `43f3c73fc6ddeb833ffcf7413a379d2dcc26f5abb0b29d2def4f4bbde633e3dc` |
| `docs/assets/diagrams/13-nms-free.svg` | `1913ed758c16038685859224ee59b8ea78984fa1ffaa9fd5c2d84d2c3d33b831` |
| `docs/assets/diagrams/14-split-paths.svg` | `3dcd7a8b95cd422632559f86b3b91d573a3cf6fd19a944e291c12130c44ca299` |
| `docs/assets/diagrams/15-attention-bridge.svg` | `1f946ed74133526da208c15fc1ac52b33b8576bae8ea68ebf1f9efd6324f205b` |
| `docs/assets/diagrams/15-area-layout.svg` | `cc93577143d427cd8c089d8dd8e4c12b8039e38be11a17d7135cbd5f45f0e89d` |
| `docs/assets/diagrams/16-distance-decode.svg` | `fb41cf83ae19ea8c490e0ae2af3f61563251c6021e51ed7178a9d47b0a2d01f9` |
| `docs/assets/diagrams/16-head-paths.svg` | `1acac44ac36b5b847e55a22e24d2a97fc2095a25800e3ed0df71e75dbcebf2c4` |
| `docs/assets/diagrams/16-stal-candidates.svg` | `04d26cccdc1c78a9f72ce64550a6c156d0f9bccef78b895ff3080fa67c7fc49f` |
| `docs/assets/diagrams/17-capstone.svg` | `d86dca51873ecaa58df51d8388a704005699c20820d16854d3814c7b8735e9ce` |
| `docs/assets/diagrams/18-video.svg` | `9c2bf9984e04e64f5a1f788ea774a74ad74799cb2732b0ce353369ae1d8f3040` |
| `docs/assets/diagrams/19-tracking.svg` | `b49961cef2c248e1b163e652d77c89ff9e7670c5ce0a5f164ffeae36740bafe3` |
| `docs/assets/diagrams/20-deployment.svg` | `45f53f5c670301be885cbc89767ee1195e8fbe908741eeec467b7f49ea6efda4` |

對應 case／CPU evidence 指紋（只核材料對應，本輪未重跑）：

| 頁面 | case SHA-256 | evidence JSON SHA-256 |
| --- | --- | --- |
| `13-dual-assignment` | `5ba8cda913f25bde1c82d93d91330b7db4a57f497e8305663aa3c0b3f62fa4d5` | `20d1fefd17f382db2e43fe44eb0fc59362649d55459f000f2c6530f10fb74e85` |
| `13-nms-free` | `7697044f6ea966ef1bbf1200a39a6d65d2d160ee186810a1f479e433a551c8be` | `9f18ed0065d0386e3fd36b25ba3e79a5d81ef8018b717799e2ca1dbd0537d38f` |
| `14-feature-module` | `5fceb84d094061f30744f2ac138702d55f24e746fe5780e3ca653e56843fac6c` | `05bc8b0e823bac904f8d18ce9f7ab30ccd2079fb306138a6ffe3c3a3b99843c4` |
| `15-attention-bridge` | `9823e1f3b30b6feb92d96fe5a9dd74a315500eaa31987f89c2cdea4b686aed06` | `67b0336c1add8a7fb1727c473b9e3d42a46c4c06200848eead39127353785f53` |
| `15-area-attention` | `eb26ab33ea5718820880820062c1664945a8bb591fc3a7d80e04a71860854041` | `604bb16d155348b43af65be5b0da3ff3fb8abd738fb6512c1e97d8ec4206e114` |
| `16-dfl-free` | `fc9aeedfab08771310905ea48dd78ea64f0b5cd76569b956e8a09c01de4c3a04` | `a7c1e7884b269918371d47dfc04566fccb0d1813d1ead5e9f37b885e4a7c3f8e` |
| `16-inference-head` | `e76f4949c12b64c8f1470fda2a8b23efff198eaa342950aaba9cd3dd924c9c3d` | `95a8dd773033000bb327f849348ded72bcf2350afdd13de124ddc1ea8af144ae` |
| `16-training` | `b50fb995486aa0ff042edb1486de27605fc0c3ec28f201ecb290b97a733fe31e` | `286c24f3ece775b4a60df2afa9b9f373c335f08eb2c8386a583c6e10d1b18e38` |
| `17-capstone` | `a5969a35c9409bae299c070229defef69a40e340d53c3aecd563f087fea12cf8` | `aefc743cfd67dccfb5867c377fe0063622aeb5288d6043797371afdd1a7119bf` |
| `18-video` | `0aa6c1713098e244dd3651c6f319844ad0e17b10ee28a187e8a224ee6e9024a0` | `f7d0eb05bad4a736b561c75463049d794d829a09397b19b526782e0e838bb609` |
| `19-tracking` | `e1fc77ce9e6e8e09a34f4dedf8543155154ab58709eaf1ab61756793f7a14efa` | `0b331433e387dcc0f037634b6df18fc7667432fa09167064128fa9a2267cd306` |
| `20-deployment` | `bf01b4ab68543d80e4f4c0ac27740d87852b9095b7b6b005b9d2dcbaf7157cd3` | `48b1cd9f50e8e6df94bf6acb1fc49beed817e55da1f94ea40dd297f99f981145` |

原始 raw 檔保持原內容：

- `reviews/clear-tutorial/16f6910/first-read/modern.jsonl`：`9880617d78f18b3cfe019c556aa144c630f9326e9cc75d55b6c0340f9091bf29`。
- `reviews/clear-tutorial/16f6910/first-read/applications.jsonl`：`b664962da744055b8362d27d70344ead87c54f8be4d7df02c2a88667c511d116`。

原問題的凍結單位指紋由 raw 直接讀出，未重建內容：

| ID | 凍結 page SHA-256 | 凍結 unit SHA-256 |
| --- | --- | --- |
| `modern-0062` | `a0c9137cf03000eb723dee18cd8864c309ac35ea2231d2f64e2f1abf0382a9c5` | `23be45a0f177e733171a5c907c019f3f353d2617246579d566b15a1d50786029` |
| `modern-0064` | `a0c9137cf03000eb723dee18cd8864c309ac35ea2231d2f64e2f1abf0382a9c5` | `865623ad249597a17ac67639161cfbd78ab03a8ee95d6557e789e038f7372735` |
| `applications-0067` | `22aaaadaa43ca7cc1519899ff07a4d06000b0acfa7b45406526883e893ce56ac` | `b7ac6d1830994833f1309ea8022ac8fd9cf9d54c8440540a5ca10ca1770a6958` |
| `applications-0068` | `22aaaadaa43ca7cc1519899ff07a4d06000b0acfa7b45406526883e893ce56ac` | `8130d0e6a1221c73d294c8595da43079610f9865d2ca82b1ae257a43af618c25` |

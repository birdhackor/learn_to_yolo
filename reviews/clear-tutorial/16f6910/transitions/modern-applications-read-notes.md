# 第三輪當場筆記（先現稿，尚未對照 raw issues）

前文 12-assignment／12-dfl 已完整實讀。已知 owner 是 GT 清單索引，與 class id 分開；12.3 的 inside→score×IoU²→topk→IoU 衝突與13的品質toy不同；target生成 detach不等於整個輸出detach。DFL 以相鄰bin加權CE監督，期望距離格×stride解碼，K與K−1、raw輸出數和熵下限皆有前文支撐。

## 13-dual-assignment

當場理解：官方每GT top1初選／top10多樣本、CIoU解衝突／軟target，toy則品質衝突＋列舉全域optimal；不把toy owner冒充官方。兩個完整head不是分類/框兩分支；many訓練shared features，one輸入detach但自己的head仍學。官方一對一名稱是目標及初選，三GT反例可能C最後兩positive，現文把限制放在推理處。

自己小變化：若僅把A對p1品質由.85加至.92（B列不改），toy全域選A→p1、B→p0，sum1.80、owner[1,0,-1]；品質top1也無衝突地相同；top2仍因p0 A.90>B.88而many owner[0,0,1]。因此公式／品質表一致仍不等於兩head逐候選target一致。

當場疑點：尚無blocking或需猜的owner方向；forward/detach/toy/official界線自足。品質表沒有幾何材料，現文明說不能推出官方CIoU owner。圖尚待独立SVG預覽查看，真Zensical browser未做。

## 13-nms-free

當場理解：從13.1的head/assignment變成兩套獨立可學logits，明示不接CNN、不學框、不重現assigner。材料A/B四固定框、p1duplicate、唯一target差、训练分數、score/NMS/matching三門檻、top-k反例與人工AP都自足。新logits target的任務变化在程式前說清楚。

自己小變化：若把B及p2一起從x20–30移到2–12，p0與p2屬兩真物件、IoU2/3大於.5。相同高分一對多+NMS會錯刪一個真物件而R=.5；固定target的一對一logittoy仍能同時保留p0/p2，但不能把此結果推成真模型保證。说明NMS-free目标不等于所有重叠框互删。

图已实际看独立SVG，GT x区间、四候选与两套target/0.959/0.041对应明确；沒有真实页browser。无blocking。

## 14-feature-module

當場理解：从候选去重转为features同shape模组，开头提出保留不同转换深度，材料是随机[2,8,8,8]与roll target，不是框。SplitAggregate project→split a/b→b1/b2→concat→fuse；direct concat-segment gradient与总梯度分工有近处定义，不能用b1总grad证明concat保留。官方C3k2/C2f/Bottleneck差异限定清楚。

自己小變化：保持channels8/blocks2，将hidden4改2：project8→4、四份各2、concat8、fuse8→8，输出仍8ch；params36+4×38+72=260。理论RF依然1/1/5/9，窄路径传左邻容量更小，不能据参数减少预言roll loss或AP提高。

图已实际看独立SVG，a/b/b1/b2段与箭头能追；无blocking。roll循环边界与零padding、信息瓶颈、zero输出baseline均足以阻止把loss下降当学会任务。

## 15-attention-bridge

當場理解：14同shape卷积特征转到内容决定的空间读取，2×2明确素材排成tokens；attention head与检测head第一次出现就区分，Q/K定读取比例、V作内容，内积与cosine不同、source轴softmax、归图顺序、12矩阵可学权重与16 affinity中间值皆有定义。channel1×.5是人为reconstruction gradient probe，非检测成绩。

自己小變化：若只把K投影设0，Q与V维持identity，所有分数0、每个query读四份各1/4，全部输出token均[.5,.5]。这不是四份V先softmax；V仍保留token内容，均匀性来自K。无需假定权重表是可学Parameter。

当场无blocking，完整手算和小图回答读谁、读多少、输出什么；affinity会随输入重算的边界清楚。独立SVG预览已看，真网页未验。

## 15-area-attention

當場理解：从full改area只限制位置配对，head分channel、area分位置。row-major不经window重排所以4×4/A4为四横带，q0=0让权重均匀可手算，干预token15时K也变但q0内积仍0；BN/位置卷积拿掉的原因在实验前定义。因此跨区不响应仅该单层toy，不等于完整网络。

自己小變化：A=8时每区两token，pair=16²/8=32；token0只读0/1，输出(.0+.0625)/2=.03125。改token15+5仍不同区无影响；改token1+5会增加5/2=2.5。参数仍12、无法预言整模型速度1/8。

当场无blocking；长条布局图中query/change token框及A4/A2清楚。独立SVG看过，未做真网页browser。

## 16-dfl-free

當場理解：12.4 的 softplus 直接距離、DFL，與官方 YOLO26 的 signed 直接距離分開；reg_max=1 的 identity 不等於單 bin softmax。四距離以格為單位乘 stride 解碼，右邊 18 超出 K16 的 15 格，但別的 stride 可表示相同 144 px。SmoothL1 toy 與官方 CIoU 加 normalized L1 明示不同。signed 負值與 STAL 擴充候選範圍先預告，合法框仍須 l+r、t+b 正。

自己小變化：保持 GT [74,64,228,108]，只把候選中心由 (84,84) 移到 (116,84)，stride 8 的 target 變 [5.25,2.5,14,3]。同一框這時四方向都在 K16 可表示範圍內，raw 數量不變。DFL 是否超限也取決於候選位置，不能只看框寬或 stride。

當場無 blocking，數值、物理單位與 official/toy 差異都在相關段落自足。圖待看，真網頁未驗。

## 16-inference-head

當場理解：從訓練雙 head 轉到真正移除 many 的 DeployToy，不只切換讀取；332→278 參數、deepcopy 已更新 one、同一 PyTorch 的 torch.equal 檢查及輸出格式有材料。人工 ltrb probe 與 raw equality 都明示不能證明真實候選排序／精度。官方 default many + NMS 與論文 one 推理路徑分別說明，single-label top3 toy 與官方 two-stage box/class topk 分開。

自己小變化：若只把输入寬改為 128、高仍 64，固定 pool4 保持 raw shape；候選中心 x 變 [16,48,80,112]、y 為 [8,24,40,56]，stride_x=32、stride_y=16，距離需乘 [32,16,32,16]。繼續硬設單一 stride16 會只把 x 方向座標縮半而不報 shape 錯誤。這是幾何契約，不是刪 head 的數值等價保證。

當場疑點：開頭無限定句「一對一（one-to-one）讓每個物件只分配一個正候選」可能重建 13.1 已排除的 final owner 恰一個保證。需交作者確認初選／目標與最終衝突結果的措辭。圖待看，真網頁未驗。

## 16-training

當場理解：本節先分清三項技巧，只對 loss 權重跑獨立 Linear head，STAL 只查資格、MuSGD 只說明。progressive 與固定 .8/.2 不只排程還差 sum b；增加等總量 .45/.55 把效果分開。首步 MSE、已乘 b 梯度與 SGD 手算，更新前 history 與更新後評值分開。可跳讀 eigen/H 推導不是做主例的先決。STAL 資格框不是回歸 GT，signed 負距離接回 16.1；YOLO26 topk2=1 與 13.1 官方 YOLOv10 不同。

自己小變化：把兩 head 的總 loss 同時乘 2 而 SGD lr 改成 .025，這個無 momentum／weight decay 的 Linear toy 每步更新完全相同：.025×2×梯度=.05×梯度，所以首步參數和三組最終 MSE 不變；記錄的加權梯度和總 loss 會加倍。此等價不能直接套到 Adam/MuSGD 或新增動量。

疑點：開頭『13.1 說過…最多只選一個』需區分 YOLOv10 初選與 YOLO26 final topk2。已回報 root。STAL 圖已看，軸向、原 GT 及擴張資格框 0→4 皆清楚。

## 17-capstone

當場理解：從第 16 章機制回到一個可交付需求，明示使用第 7 章 grid MiniYOLO 而不混 DFL/attention。32/16/16 split、seed、box5→10、val 規則 .01、只測 chosen test 一次，coverage 非 TP/recall、四種 FP、四門檻及兩種計時範圍都自足。當次 1.03/1.12 秒只計 fit loop、2.96 ms 含 uint8 tensor/模型/NMS/診斷画框，並非純 forward，也不支援速度因果。練習交付格式和 report 欄位有近處定義，不需猜作者標準答案。

自己小變化：若將畫圖 display threshold .25 改 .5，保持評估 cutoff .05、所有訓練與 NMS 相同，模型權重、baseline/changed mAP、keep_change 與 chosen test 都不變；圖會少畫 score .25–.5 候選，畫框計時可能改變。不能因畫面更乾淨宣稱 precision 提升。若改的是 cutoff，則不保證上述不變。

當場無 blocking；圖中 #4/#14 的高分定位錯誤與 #10 低分背景 FP 可以直接對到正文。已實看獨立 SVG，未驗 Zensical desktop/mobile。暖機失敗處置雖長，錯誤箭頭與正式訓練判別有具體動作，無需猜。

## 18-video

當場理解：17 靜態圖 → 18 同模型逐幀跑，新增 Frame index/source time/RGB、來源寬圖 letterbox、queue、釋放。generator 以 count_up 實例定義後才用，lazy 來源與 list 小示範分開；裝飾器 no_grad 暫停恢復與 finally/closing、runpy/session 差異皆在相關程式前交代。FPS 來源／處理速率／擷取到完成的延遲三者分開。假來源不真等50 ms，框是模型實算，無框的8幀不是串流掉幀；FFV1 adapter 記錄、未測camera/VFR/PTS和有損codec均明示。

自己小變化：不改畫面和模型，只把來源時間戳標成 25 FPS（count12），最後時間 11/25=.44 秒、總播放 .48 秒、GIF 每幀40 ms、每秒移動100 px。逐幀 boxes 不變，模型計算 ms 不由時間戳決定。若真正來源25 FPS而處理50 ms，無限排隊第k幀延遲=50+10k ms；與預設 lazy 示範不同。

當場無 blocking。獨立SVG看過三幀0/5/11與來源0/.25/.55、預測/無框、單幀ms都有標。小字需點原圖放大現文已有入口，真網頁手機大小未驗。

## 19-tracking

當場理解：偵測框→跨幀身份，ID與矩陣row/col/類別明分，主例人工完美框加1漏檢不是跑模型。track四欄位、update五步、max_age=2只能連漏1幀、速度乘幀差、門檻.1、列舉先pair數再IoU都自足。六幀材料支持手算第2幀交換與第5幀舊ID仍在但IoU0、3→0 switches；11/12 detector recall 和tracker結果分開。真實18輸出接線只留class0、空幀update、篩選後boxes與ids一起對應，無GT所以不宣稱正式MOT分數。ByteTrack低分輪僅仍Tracked未配上的track，Lost不進，現文限定明確。

自己小變化：只把某一幀 detections 的順序反轉，保持框和 GT 對應身份評估同步，IoU矩陣欄也反轉；最佳配對及物件實際ID不變，update回傳list順序隨輸入變。例如motion第2幀若輸入B,A，IDs會[2,1]，不能把這個數字表面變化數成兩次ID switch。若仍以原A,B索引評估則會人為算錯。

當場無 blocking，尚待獨立SVG看圖。真頁browser未驗。

## 20-deployment

當場理解：從跨幀狀態轉到 backend 匯出一致性，明示換回第7章 GridDetector 一步SGD固定權重，不是16.2 one head也不是偵測品質任務。ONNX/model-only邊界、ORT/TensorRT、opset/session/engine、input raw layout、不同圖metadata、checker/raw/還原框三層及防空對空的框數assert都有定義與材料。batch1/2/3不是動態空間，80拒絕與pool追蹤限制解釋足夠。paired時間差、raw/整段scope、等batch服務延遲與throughput分開。未執行trtexec、作者另跑L4 Python、allowedFP16不等於各層FP16、CPU/GPU權重不同和無AP均直接標示，讀者不需猜哪項證據支持哪個宣稱。

自己小變化：若保持匯出和raw完全不動，但ORT後處理錯拿第一張metadata還原第三張48×80直圖，checker及raw allclose仍通過，最後框座標比較會失敗（兩張letterbox補邊方向／比例不同）；labels和scores仍可相同。因此三層不是重複同一檢查，不能只報raw誤差小。

當場無 blocking。独立SVG看過metadata旁路、model-only藍區、三backend事先匯出／build虛線與三層圓點，與正文可對；未做真頁browser。自己的小變化是預測／推理，沒有新跑ORT或GPU。

## 作者修正後實際核回（與原始 first-read 分開）

所有現稿及12張SVG預覽讀完後，才讀 modern.jsonl 與 applications.jsonl 的 target issues。modern-0062 官方流程改先行，toy改用全域optimal在小節入口明說，enumeration/greedy/Hungarian改選讀；modern-0064 常見錯誤明確區分toy候選足夠／官方初選／推論不保證兩框。applications-0067 明寫validation圖片#4；applications-0068 近圖新增#4/#14 IoU數值與定位FP判定表，optional要求的至少示例補說明已處理，未逐框直接加TP/FP字樣，保留為可選圖強化。

新增銜接T01：16.2與16.3回顧13.1的兩句曾重新說成每GT恰一／最多一。root修後已實際重讀當前16.2第21行、16.3第29行：YOLOv10 top-1限制初選；YOLO26另在衝突後topk2=1，最終至多1、可能0。與13.1及16.3 STAL完整段相容，疑點閉合。未改原始raw，也未將作者報告當核回證明。

16.1/16.2/19三張先前註記待圖者已全看完成，共12SVG。所有圖是Chromium page.set_content的獨立source預覽，沒有実際Zensical desktop/mobile或MathJax排版驗證。全12頁的小變化是自己的推理預測，這輪沒有新CPU訓練／ORT／camera／GPU執行。

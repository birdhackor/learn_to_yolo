# 第二輪獨立技術與證據審查：modern / applications

審查者：`clear_first_applications`。日期：2026-10-05。基準 checkout：`16f69103d1f216d1024a40610623083557324701`；查的是目前工作樹的 12 頁修訂稿，不是既有發布 tag。預計 `lessons-v0.5.0` 尚未建立。本審查者撰寫的是 07/09/10，不曾撰寫本報告的 12 頁；本輪問題由 root 修正文，審查者只核回並新增本報告與 probes，未自行修改這 12 頁、程式、coverage 或舊 reviews。

結果：找到 1 個 blocking（M13-01）與 2 個 medium（M13-02、M19-01），均經 root 實修、本人讀現稿核回。目前沒有剩餘技術阻擋。這不包含全站／桌面／手機視覺驗收：本輪只核 SVG 內容、箭頭來源、座標、嵌圖和數據；12 頁的實際 Zensical 瀏覽器可讀性全部標為**未驗**，交 root 的全站 browser。也沒有獨立重跑 L4／TensorRT；其结論只限保存紀錄與程式的證據一致性。

## 實際方法與原始證據

完整讀取各頁正文與 case；工程頁另讀 `miniyolo/{data,targets,losses,models,metrics,inference,geometry,deployment_gpu}.py`、`scripts/{verify_video_file,modal_deployment}.py` 和 deployment workflow。各頁 `artifacts/checks/curriculum/*.json` 的 case 與全部 bound dependencies SHA-256 均與目前檔案一致。影片與 GPU 的額外記錄／依賴清單見 [engineering-provenance.json](probes/engineering-provenance.json)。舊數字未被重寫。

在 `/tmp/modern-applications-review/repo` 複製 `lesson_cases`、`miniyolo`、`scripts`，使用本環境 `.venv-model/bin/python`（PyTorch 2.9.1+cpu）執行，所有圖、ONNX、影片、報告都寫隔離副本／`/tmp`。12 個基準 case 退出碼皆 0；16 個有針對性的練習變體，成功或预期失敗的位置皆與正文一致。原始 stdout/stderr 見 [cpu-runs](probes/cpu-runs)，執行及改動清單見 [baselines.json](probes/baselines.json)、[variants.json](probes/variants.json)。未跑 GPU、未下載 checkpoint、未 push／commit，也未重新記錄 shared evidence。

兩個官方 assignment probe 保留 plain JSON 和完整固定上游源碼，可離線重跑：

- [top1-conflict.py](probes/top1-conflict.py)／[結果](probes/top1-conflict.json)／[URL、SHA、執行方法](probes/top1-conflict-provenance.json)：原版 `bbox_iou` 與完整 `TaskAlignedAssigner` 的 AST 節點原封編譯，僅排除無關 imports；呼叫原版 get_pos_mask 與 forward。CPU，無 GPU。JSON 包含全部 GT／預測框、anchor、類別、分數、mask、topk/alpha/beta/eps 與輸出。
- [v26-post-conflict.py](probes/v26-post-conflict.py)／[結果與證明](probes/v26-post-conflict.json)：原版 `select_highest_overlaps` AST，檢查同一種衝突 mask 在 conflict 後的 topk2=1 篩選。只驗此方法與數量界線，不冒充完整 YOLO26 訓練。

[external-sources.json](probes/external-sources.json) 保存本次實際 GET 成功的固定 URL、commit／論文版本與完整 SHA。主要支持版本：

| 來源 | 固定版本與支持位置 |
| --- | --- |
| YOLOv10 作者論文／repo | arXiv `2405.14458v1` §3.1、Fig.2、matching-metric appendix；THU-MIG `453c6e38a51e9d1d5a2aa5fb7f1014a711913397`：tal.py L90–125、L232–260；loss.py L719–720；head.py L496–534；ops.py L851–864；predict.py L8–31 |
| YOLO11／YOLO26 官方 repo | Ultralytics `441632cdfd19e22e60a4b1b1999d46326ca51ec4`：C2f/C3k2/C3k、Detect、BboxLoss/E2ELoss、TAL、MuSGD、trainer、yolo11/yolo26 YAML 與 training recipe，逐頁列明方法 |
| YOLOv12 作者論文／repo | arXiv `2502.12524v1` §3.1–3.4、area cost、R-ELAN 與 speed/ablation；作者 `2abab7153a065fb2925e8088e9ca2b19016ab7d6` block.py AAttn/ABlock/A2C2f L1176–1385 與模型 YAML |
| Attention／YOLO26 論文 | `1706.03762v7` §3.2.1 公式(1)、variance footnote；`2606.03748v1` §3.2/3.3、STAL公式(5)、§4.1、MuSGD §4.3.2 |
| 部署官方文件 | PyTorch 2.9 ONNX API（頁面連到 v2.9.1）、ONNX v1.19.1 checker.py；TensorRT `b8db91e15be2cae4465ac17fab19e0f969e45407` trtexec README、sampleOptions.cpp 與 NvInfer.h |
| 追蹤作者源碼 | SORT `2236dff5019565958b84df7d871d41cc1db58ac7` associate_detections_to_trackers；DeepSORT `f08cf1dc470eeb1cd2add1cbf077d95ac6c48aab` Tracker._match；ByteTrack `d1bf0191adff59bc8fcfeaa0b33d3d1642552a99` byte_tracker.py L177–269、matching.py L39–49 |
| 正式追蹤指標 | TrackEval `12c8791b303e0a0b50f753af204249e622d0281a` identity.py L47–86/132–134（全序列指派、IDF1）；hota.py L52–103/170–173（association、DetA 與 HOTA） |

## 問題與修正核回

**M13-01 — blocking，已關閉。** 原稿把固定 YOLOv10 的 topk=1 說成「最終每 GT 至多一個正樣本」，另以此推導軟 target 一定等 CIoU。原版 tal.py 的 conflict 在全 GT 上取最大 CIoU，不限已 top-k 選它的 GT。實跑：GT A=[0,0,9,10]、B=[1,0,10,10]、C=[0,0,10,10]；p0/p1 的預測皆 [0,0,10,10]，anchor=(4,5)/(6,5) 全 inside；三類 scores p0=[.9,.8,1e-6]、p1=[1e-4,1e-4,.9]。初選 A/B→p0、C→p1；p0 被 conflict 改給 C，最終 owner=[2,2]、GT 正數=[0,0,2]，C 的軟 target=[.00105409,1]，均為正。root 改 13.1/13.2，現稿分清 top-1 初選、最終每候選只歸一個 GT、toy 的嚴格配對與理想推論數量，且折疊反例回指 probe。本人已重讀受影響全文及圖；13-flow 圖只標官方 top-1 選法，未宣稱最終 GT 數量。

**M13-02 — medium，已關閉。** 原修稿仍說「單正且 m 非零就約等於CIoU」。eps 可主導極小 m，精確是 `m*CIoU/(m+eps)`。root 現改為精確式，僅 m 遠大於 eps 時近似；本人實讀現稿 L46 核回。

**M19-01 — medium，已關閉。** 原稿說 ByteTrack 首輪高分配「所有 track」，其次所有 unmatched 配低分。作者固定版初輪是 confirmed-active + lost；低分輪只有首輪 unmatched 且 state==Tracked；unconfirmed 另配剩高分。root 現稿已逐項收窄，並保留低分不開新 track；本人讀現稿與固定源码 L193–269 核回。未執行完整 ByteTrack，審核的是此處算法敘述。

## 逐頁查核

以下每頁都核對本文、程式、保存輸出與必要圖；完整頁／case／圖／evidence SHA 在各節末，全部是核回後的工作樹內容。`No issue` 表示本輪技術範圍未發現剩餘問題，不代表真實學生理解、完整模型品質、GPU 重現或 browser 通過。

### 13-dual-assignment

**判定：M13-01/M13-02 已關閉；其餘 no issue；browser 未驗。**

實讀 2×3 品質表、全域列舉／貪心、owner→分類 target 與 detach；CPU baseline 重現 [0,0,1] / [1,0,-1]、1.73/1.10、BCE .6855/.7065。B→p2=.95 練習按正文三項斷言改動，實跑得到全域＝貪心1.85、owner=[0,-1,1]。候選不足時排列列舉為空，GT id 直接當 class 只在2GT/2class toy適用，正文限制已明示。

固定官方 loss 確為 tal_topk=10/1，head 在 one 的特徵輸入 detach；論文 §3.1 的 Hungarian替代 top1、相同比例 α/β 的排名推導有固定來源支持，但初始相同預測的假設不是訓練全程保證。完整官方 probe 揭露 conflict 最終GT數量例外，root修後成立。SVG雙head前向／回傳阻斷與部署路徑對得上源碼，toy只分類且沒有AP/速度誇稱。數據依據：13-dual-assignment.json 與本次 stdout、官方 probe。

内容指紋（SHA-256）：

```text
docs/lessons/13-dual-assignment.md  23d265ae9a2236832e4fca60cdcddfe27972dda8fd4d32d184588ed206df97b6
lesson_cases/13-dual-assignment.py  5ba8cda913f25bde1c82d93d91330b7db4a57f497e8305663aa3c0b3f62fa4d5
artifacts/checks/curriculum/13-dual-assignment.json  20d1fefd17f382db2e43fe44eb0fc59362649d55459f000f2c6530f10fb74e85
docs/assets/diagrams/13-dual-head-flow.svg  43f3c73fc6ddeb833ffcf7413a379d2dcc26f5abb0b29d2def4f4bbde633e3dc
```

### 13-nms-free

**判定：M13-01 連帶文句已核回；其餘 no issue；browser 未驗。**

四個框皆pixel xyxy；重算 p0/p1 IoU=90/110=.8182。四個獨立logit從0，以 mean BCE、SGD lr1更新100步，實跑正=.959/負=.041；初步梯度±.125及logit更新方向正確。穩定排序的many/NMS保留[0,2]，one/noNMS[0,2]；人工.92/.90/.80/.05的top2確為[0,1]漏B。SVG以60+8x呈現0–50位置，GT/候選/table列與target逐項一致。

手算 all-points AP 的TP/FP/TP為5/6；尾部背景FP只降末端precision，不降此AP。owner換p1與score .99練習實跑，分別[1,2]及全空，原IoU門檻保持.5。固定官方 v10postprocess 是兩階段候選／class-pair topk，predict只拿one、再score filter，max_det300預設／export用法正確；正文已分清toy指定target、真head可能漏檢／重複，未把框少當準確率。數據依據：13-nms-free.json。

内容指紋（SHA-256）：

```text
docs/lessons/13-nms-free.md  023323d044c1253c07a4adc24a420107d8851de3a4674a9d71cf2704868a10ec
lesson_cases/13-nms-free.py  7697044f6ea966ef1bbf1200a39a6d65d2d160ee186810a1f479e433a551c8be
artifacts/checks/curriculum/13-nms-free.json  9f18ed0065d0386e3fd36b25ba3e79a5d81ef8018b717799e2ca1dbd0537d38f
docs/assets/diagrams/13-nms-free.svg  1913ed758c16038685859224ee59b8ea78984fa1ffaa9fd5c2d84d2c3d33b831
```

### 14-feature-module

**判定：no issue；browser 未驗。**

讀SplitAggregate及固定官方 C2f L291–327、C3k2 L1069–1107/C3k與yolo11 YAML。toy含bias無BN，project/fuse無activation，hidden bottleneck e=1 vs官方C3k2預設e=.5，與整版YOLO11收益的比較邊界已列明。a/b/b1/b2四份各4ch、concat16→8和SVG四個入線順序一致；total path gradient與direct concat slice gradient確實不同、case分別核對。

手算參數72+592+136=800 vsplain1168；卷積MAC每張49152 vs73728（不含bias），blocks3=1128／blocks4=1456，不能外推split皆省。CPU基準MSE1.0722→1.0049；全零參考約.97，正文未宣稱學會roll。3blocks練習實跑concat20→8、1128及動態assert成立。RF四路為1/1/5/9（相對此8×8 module輸入），zero-padding與rollwrap限制正確。數據依據：14-feature-module.json；無完整YOLO11/AP比較。

内容指紋（SHA-256）：

```text
docs/lessons/14-feature-module.md  f9a8f4240d4f095f15507f3a3e607e5c7362f106ce77540f89254a448649ca29
lesson_cases/14-feature-module.py  5fceb84d094061f30744f2ac138702d55f24e746fe5780e3ca653e56843fac6c
artifacts/checks/curriculum/14-feature-module.json  05bc8b0e823bac904f8d18ce9f7ab30ccd2079fb306138a6ffe3c3a3b99843c4
docs/assets/diagrams/14-split-paths.svg  3dcd7a8b95cd422632559f86b3b91d573a3cf6fd19a944e291c12130c44ca299
```

### 15-attention-bridge

**判定：no issue；browser 未驗。**

核 CHW flatten(2).transpose(1,2) 的row-major tokens [[1,0],[0,1],[1,1],[0,0]]；QKV identity只是初值，raw affinity /sqrt2、沿key軸softmax後加權V，公式與Attention論文§3.2.1/variance footnote一致。CPU基準重現權重.3349/.1651/.3349/.1651、首token[.6698,.5]；SVG箭頭線寬20×權重、來源與receiver方向明示、V绕过softmax。

四個token改[2,0]內建練習得到.2212/.1091/.2212/.4486與[1.3395,.3302]。target逐channel*.5的view避免broadcast到寬軸；改target=feature實跑如正文預告於Q/K allclose否定斷言失敗，已印前六行但不印第七行。分開Q/K/V梯度及一次step、實際權重改變檢查成立。N=HW成平方、邊长翻倍pair16倍的範圍正確；FlashAttention說明限精确attention的I/O算法，不冒稱pair變線性。v12 AAttn/ABlock有位置卷積、残差與多head，不等於本toy。數據依據：15-attention-bridge.json。

内容指紋（SHA-256）：

```text
docs/lessons/15-attention-bridge.md  eba51744673bf59c6c1e9cc913d3bb2d83faa73ed7463465b4f0728682f4cf8e
lesson_cases/15-attention-bridge.py  9823e1f3b30b6feb92d96fe5a9dd74a315500eaa31987f89c2cdea4b686aed06
artifacts/checks/curriculum/15-attention-bridge.json  67b0336c1add8a7fb1727c473b9e3d42a46c4c06200848eead39127353785f53
docs/assets/diagrams/15-attention-bridge.svg  1f946ed74133526da208c15fc1ac52b33b8576bae8ea68ebf1f9efd6324f205b
```

### 15-area-attention

**判定：no issue；browser 未驗。**

固定作者 AAttn L1206–1261 是連續token reshape [B*A,N/A,C]；4×4/A4為水平帶，不是四象限。作者模型YAML仍有area1層；AAttn先对整圖V做5×5 depthwise PE，Conv的train BN也可跨區，故toy的完全隔離不可外推整版YOLOv12。論文§3.2写7×7與此fixedcode5×5差異、R-ELAN梯度穩定主張和純pair成本／總速度比較邊界已明示。

CPU基準full256pair／area64pair、token0 .46875/.09375；改15+5後.78125/.09375。A2、A1、改token3三練習都實跑：128/.21875；256/.46875→.78125；64/.09375→1.34375。另autograd查token0對token15：area梯度[0,0]、full [.0625,0]；只是這個無BN/PE、sharedQKVtoy。N²/A、head/batch乘數、learnable參數12不變正確；不把4倍pair節省叫全網4倍快。SVG0–15分區、红query0/紫3與15均對上索引。數據依據：15-area-attention.json、numeric-probes.json。

内容指紋（SHA-256）：

```text
docs/lessons/15-area-attention.md  1870d643f21078b9c40238f3c4d1eedb3fa3ee147897b6ef7927bb549a6e6436
lesson_cases/15-area-attention.py  eb26ab33ea5718820880820062c1664945a8bb591fc3a7d80e04a71860854041
artifacts/checks/curriculum/15-area-attention.json  604bb16d155348b43af65be5b0da3ff3fb8abd738fb6512c1e97d8ec4206e114
docs/assets/diagrams/15-area-layout.svg  cc93577143d427cd8c089d8dd8e4c12b8039e38be11a17d7135cbd5f45f0e89d
```

### 16-dfl-free

**判定：no issue；browser 未驗。**

固定yolo26 YAML reg_max1、Detect L125–139之4輸出距離/Identity及_decode没有softplus；loss.py BboxLoss L133–154實為CIoU+帶符號距離正規化L1，target/pred先乘stride再除WH、品質權重、dfl gain保留，與toy SmoothL1清楚分開。bbox2dist不給reg_max時不clamp，所以STAL框外anchor需要負距離的解释成立。

K16期望只能0–15，18stride8=144px本層放不下，但stride32=4.5格可放下，正文不外推DFL不能處理大框。CPU300步重現1.25/2.5/18/3、loss5.6875→0、初步grad-.25/距離.125；右邊改8并按正文四處同步改，實跑loss3.1875、x2=148。重算signed [-1,2,3,4]→[92,68,108,116]與合法l+r/t+b条件。SVG依(3.25x−50,3.25y−104)核框[74,64,228,108]、anchor84/84、DFL界x204及3格差；raw6600/600=26400/2400bytes只算head輸出，不称總memory/latency。數據依據：16-dfl-free.json。

内容指紋（SHA-256）：

```text
docs/lessons/16-dfl-free.md  8243eb1124300e1bc1aec55272418dfc3b9452dbc8cd9c105f62fd10e3518ce0
lesson_cases/16-dfl-free.py  fc9aeedfab08771310905ea48dd78ea64f0b5cd76569b956e8a09c01de4c3a04
artifacts/checks/curriculum/16-dfl-free.json  a7c1e7884b269918371d47dfc04566fccb0d1813d1ead5e9f37b885e4a7c3f8e
docs/assets/diagrams/16-distance-decode.svg  fb41cf83ae19ea8c490e0ae2af3f61563251c6021e51ed7178a9d47b0a2d01f9
```

### 16-inference-head

**判定：no issue；browser 未驗。**

讀DualToy/DeployToy深複製、one特徵detach與一次SGD，推論无many，raw用torch.equal逐值相等不是拿top3比。基準参数332→278差54（16.27%只限toy）、output B2/3框/4coords，raw差0。手算候選(24,40)、distance[1,.5,2,1.5]*16→[8,32,56,64]；softplus逆raw、sigmoid[0,ln3]=.5/.75/class1一致。top5練習實跑shape[2,5,4]/[2,5]/[2,5]、參數不變；固定adaptivepool4的128輸入不能沿用stride16幾何，正文要求兩處32并列新中心。

官方Detect L196–202训练one detach；L266–289两階段class-pair topk，与toy每候選argmax单类不同；fuse L290–295依end2end移除不用分支，BaseModel.fuse卷積BN不是同件事。官方recipe L26明确目前API默认many+NMS、nms=False才选无NMS，与论文设计路径分开。SVG比對線连兩个one raw，前向/梯度stop與部署相符。toy不稱完成assignment或TensorRT、品質無证据。數據依據：16-inference-head.json。

内容指紋（SHA-256）：

```text
docs/lessons/16-inference-head.md  bcd753feb76eb77021582f6990354cb49e6cc087fde1f25c06947076f0faeb53
lesson_cases/16-inference-head.py  e76f4949c12b64c8f1470fda2a8b23efff198eaa342950aaba9cd3dd924c9c3d
artifacts/checks/curriculum/16-inference-head.json  95b2f301b0b45e22ba7ec1b1aff8f4a21d7d2f5aae8e8706f00fb571c8ccb4cb
docs/assets/diagrams/16-head-paths.svg  1acac44ac36b5b847e55a22e24d2a97fc2095a25800e3ed0df71e75dbcebf2c4
```

### 16-training

**判定：no issue；browser 未驗。**

固定E2ELoss L1322–1354：many.8→.1，one.2→.9；trainer L639在epoch末update，所以官方epoch不是toy每次SGD。CPU30步重現b總量6/16.5/16.5與MSE .663073/.016285/.016882，首步dw/dbias與權重逐值對上。final_many=.3練習實跑總量13.5、配對固定(.55,.45)、首步仍相同。两个head独立，所以不能由toy推「早期many帮one」；H=[[3,1],[1,2]]正特徵值與各步交換、固定總量正數AMGM的推導正确，正文坦明非倒序實跑。

STAL图按16SVG units/pixel核[7,7,9,9]／[0,0,16,16]及4/12格心，y向下；只是资格0→4未ranking不是4正，三层stride16原有(8,8)資格。固定TAL `<stride_val=16`与论文公式(5) `<smin=8`不同，正文已說；回歸target保留原GT。YOLO26 top7→conflict→topk2=1原版方法probe/逐行證明最終每GT至多1，沒有外推v10反例。

讀官方MuSGD u-muon.py、trainer参数2D/4D分组、另存SGD momentum；用原版Newton–Schulz5 CPU實算diag(3,.1)→diag(.6875,1.1328125)，float32 .6970/1.1288，吻合正文.69/1.13和.70；不是全MuSGD训练。论文章3.3.1/4.3.2的500ep47.4 vsSGD600ep47.0确有來源，發布Objects365预训练和实验branch内部参数来自固定recipe，均非本toy测得。數據依據：16-training.json、numeric-probes.json。

内容指紋（SHA-256）：

```text
docs/lessons/16-training.md  b73abb0c11697febbbf753ba4a36746a2171d6febfee5f69db1bbea84c5d001c
lesson_cases/16-training.py  b50fb995486aa0ff042edb1486de27605fc0c3ec28f201ecb290b97a733fe31e
artifacts/checks/curriculum/16-training.json  fd0152b46b05f3ba20c8fd15330ff5b8aa5558d99889438b26450cfdb739bc38
docs/assets/diagrams/16-stal-candidates.svg  04d26cccdc1c78a9f72ce64550a6c156d0f9bccef78b895ff3080fa67c7fc49f
```

### 17-capstone

**判定：no issue；browser 未驗。**

讀最终稿、完整fit/evaluate/error_diagnostics/save_panel/timing與MiniYOLO依賴。train32/val16/test16 seeds1100/2200/3300；独立生成图SHA集两两无重合、GT33/17/13，见split-hashes。兩次fit都重设模型seed7、batch8固定輪轉、160步Adam.01，仅box权重5→10；额外10步warmup模型丟棄。选型只看val差>=.01，源码chosen后只调用一次test，warmup和总loss不能互比限制明確。

隔離CPU實跑重現全部非計時report字段：baseline AP0=.3333/AP1=.5556/map.4444、TP9/FP5/GT17；changed .625/.7778/map.7014、TP12/FP3；coverage9/4/4→12/2/3。FP #4 IoU .49179366/.30342045、#14 baseline .22750938／changed .49168783、#10 class1 score .06797741 IoU0，都和正文表／图相符；背景FP在class1 TP之后，不改变此AP但precision9/14→9/13是对的。error匹配与正式AP同图同类greedy1:1规则一致，coverage不充当recall。

选定改動test map.445238、precision7/11、recall7/13；没有baseline test也不拿val.4444对test.4452。baseline與changed總loss权重不同不可直接比較，AP改善也不能全算定位、单seed切分小波動等限制保留。所生成17 SVG與頁面圖**完整byte-exact**（含9張嵌圖与GT/预測框），选择图14/5/4/7与score .25／背景例.05一致。CPU计时查源码为3warmup+12median，uint8转tensor→model→decode/NMS→draw（含畫GT、resize192），无fileIO；loop计时不含模型/target準備。练习boxweight2不改assert实跑通过，以当次报告决策。數據依據：17-capstone.json、独立stdout与split-hashes；不声称统计或真实照片收益。

内容指紋（SHA-256）：

```text
docs/lessons/17-capstone.md  25c66ab672af41f9c08226a89b5f17c193b6c0ef26a84d583612349f0fe4445d
lesson_cases/17-capstone.py  a5969a35c9409bae299c070229defef69a40e340d53c3aecd563f087fea12cf8
artifacts/checks/curriculum/17-capstone.json  aefc743cfd67dccfb5867c377fe0063622aeb5288d6043797371afdd1a7119bf
docs/assets/diagrams/17-capstone.svg  d86dca51873ecaa58df51d8388a704005699c20820d16854d3814c7b8735e9ce
```

### 18-video

**判定：no issue；browser 未驗；實體相機／有損MP4／VFR未測。**

讀Frame／synthetic_frames／run_stream／opencv_frames／queue simulation及verify_video_file。來源12幀64×96RGB,uint8,20FPS，帧11时间.55s、總长.6s；不是实时播放节拍。letterbox64宽64高43，sx2/3、sy43/64、top10/bottom11正确；prediction还原xyxy并draw原图。CPU160步一次warmup+12正式重現框数1/1/0/0/0/0/1/1/0/0/0/0；8次漏檢不是掉幀，跨格只是假說且5/11反例保留。

独立實跑FFV1/AVI驗證：VideoCapture确读文件、RGB／prediction／overlay全等、來源SHA与保存纪录相同；EOF／提前close／openfailure资源release三项通过，tracking输入原图class0与ID亦完全重現。原頁SVG3張embedded PNG與独立case逐張完全相同；整SVG不相同只因本次耗時标签不同，未覆写原图。sample IoU约.504/.473/.304/.461，故箱数不充当品质。@torch.no_grad generator另stub probe：model内False，每次yield外与close后True，避免梯度状态泄漏。

計時scope逐项核为preprocess/model/post/draw总perf interval，不含capture/decode/queue；median分项加总不是total median。另video-file wall包含open/read/decode及collection、排除train/encode/fileoutput；与39.2s云启动不是同scope。queue30FPS/50ms例的90帧3s与最后处理delay1550ms推算吻合；dropold只示意队列，不是硬体实测。adapter BGR→RGB、恒定FPS index timestamp、VFR/PTS限制及相机未测明示。數據依據：18-video.json、video-file.json、video-file-independent.json、numeric-probes.json。

内容指紋（SHA-256）：

```text
docs/lessons/18-video.md  aa37572bcc725c62156a6624fd32a30249eecb26d14faf842b222eb41aecc269
lesson_cases/18-video.py  0aa6c1713098e244dd3651c6f319844ad0e17b10ee28a187e8a224ee6e9024a0
artifacts/checks/curriculum/18-video.json  f7d0eb05bad4a736b561c75463049d794d829a09397b19b526782e0e838bb609
docs/assets/diagrams/18-video.svg  9c2bf9984e04e64f5a1f788ea774a74ad74799cb2732b0ce353369ae1d8f3040
```

### 19-tracking

**判定：M19-01 已關閉；其餘 no issue；browser 未驗。**

讀Tracker update、exact_gated_matching、run/IDswitch/evaluate_detections/figure。lexicographic最大匹配数后IoU总和、gate>=.1、全列举2×2七种而非greedy；10×10部分配对234662231种说明不可扩展，CPU numeric probe验数。四coord速度单位pixel/frame、间隔乘速度／除gap更新与last_frame删除条件>max_age正确。

CPU基準raw IDs [[1,2],[1,2],[2,1],[2,1],[2],[2,3]]／switch3，motion维持IDs／switch0；门槛.2同物体相邻IoU与f2反向位置错配吻合，f5 B恢复raw IoU0需要新ID，而velocity仍保留。max_age1練習实跑motion末帧[1,3]/switch1，raw仍3。GT detectionTP11/12 FP0独立于tracker、沒有score故不是AP；本例交叉另一GTIoU≤.2，精确match与评分greedy等价。

图两种规则同一份人工boxes、color=ID，A/B分垂直两条仅便于标签、实际y一样，六帧数字源码对齐。真实detector bridge先筛class0再用筛后boxes对应ids，每空帧update支持过期，失联过长不能靠速度修漏检；独立FFV1run重現0/1帧ID1、6/7帧ID2，无GT身份评估。SORT/DeepSORT/ByteTrack范围已查固定作者源码，M19-01修后符合；IDF1/HOTA概念通过固定TrackEval源码核查，未报这些正式指标。數據依據：19-tracking.json、video-file.json。

内容指紋（SHA-256）：

```text
docs/lessons/19-tracking.md  7d8ba461b6d0a132a111ceafcb812a6761eccf57046df1f329b3950c2822db23
lesson_cases/19-tracking.py  e1fc77ce9e6e8e09a34f4dedf8543155154ab58709eaf1ab61756793f7a14efa
artifacts/checks/curriculum/19-tracking.json  0b331433e387dcc0f037634b6df18fc7667432fa09167064128fa9a2267cd306
docs/assets/diagrams/19-tracking.svg  b49961cef2c248e1b163e652d77c89ff9e7670c5ce0a5f164ffeae36740bafe3
```

### 20-deployment

**判定：no issue；browser 未驗；TensorRT独立GPU重跑未做，INT8未测。**

完整CPUcase实际导出opset17 ONNX、checker与ORT CPU执行，torch2.9.1/onnx1.19.1/ORT1.23.2。动态只B，输入[B,3,64,64] raw[B,4,4,7]；三來源3份metadata、batch1/2/3实比值，error1.19e-7/4.77e-7/4.77e-7与保存相同。还原框/score/label逐值全等，每来源16个并强制非空；padding候选可退化、不是16正确检测，正文已说明。80空间真的ORT拒绝；fixedpool导出不能仅标动态axes解决。oneSGDstep不当质量证据。

三層检查scope正确：checker默认结构（full_check额外shape，不做numericalparity），raw值、shared外部pre/post+undo后的框。图metadata绕过ONNX，PyTorch/ORT/TRT三backend与事先export/build、每图数据流分开。三项故障练习实跑：删除第三源B3 shape断言，preprocess80于B1形状断言，score.07第三层空结果比较全通过但B1非空断言失败，与答案一致。

计时源码raw3warmup20median，E2E3warmup40 paired alternatingorder；0.2547/.1123ms只raw，3.2786/3.0431ms含前后处理但不含I/O/queue/display，paired差.2490不等两median差；批2 .0989ms的images/s不可视服务延迟。不同环境重新数值会变，未把本轮时间替换原证据。

TensorRT固定官方sampleOptions说明 --fp16仅允许FP16加FP32、--noTF32及batchprofile1/2/4；10.12deprecated/10.13还能用有source支持。查deployment_gpu.py:40Adam固定训练图、清TF32、build序列化/反序列化、setshape/address、asyncv3后同步、batch1–4 raw/box/score/label非空比较，3warmup20 timing包括setup/output allocation/sync但无copy/prepost。保存L4FP32/允许FP16记录最大raw7.6294e-6／pixelbox3.8147e-6、两者约.147ms，非每层FP16或加速证明。modelweights与CPUoneSGD不同不可跨两记录比较。Modal源码与workflow核真实GPU1次／新CPU容器读volume验证三个文件SHA，保存记录report证明一致；39.20s包括build/startup/transfer。所有绑定依赖hash相符，但本审查未取远端engine重跑，验收仅保存的流程证据及限制。

内容指紋（SHA-256）：

```text
docs/lessons/20-deployment.md  28e152b135fb2af9310e78d0cc3cea7880713cf0b10a8591fc4af01ad6b1825a
lesson_cases/20-deployment.py  bf01b4ab68543d80e4f4c0ac27740d87852b9095b7b6b005b9d2dcbaf7157cd3
artifacts/checks/curriculum/20-deployment.json  48b1cd9f50e8e6df94bf6acb1fc49beed817e55da1f94ea40dd297f99f981145
docs/assets/diagrams/20-deployment.svg  45f53f5c670301be885cbc89767ee1195e8fbe908741eeec467b7f49ea6efda4
```

## 尚未驗證與後續交接

全12頁實際browser／手機／數學排版可讀性未驗，由root統一做；SVG source與圖數字通過不能代替這件事。L4／TensorRT沒有獨立重跑、未逐layer審計precision；實體相機／有損影片／VFR／正式MOT／真實圖片或多seed完整YOLO品質未測，本文均保留界線。上述限制不是新增阻擋，也不是把外部論文結果當本repo實測。

本輪已關閉技術問題後，可交第三輪前後銜接讀者。若上述指紋的正文／图／程式再變，需局部重查受影響內容；本報告不能為其他工作樹內容背書。

## 最終 evidence 更新核回（2026-10-05）

root 重產 curriculum CPU 紀錄後，本人再讀 17 現稿：loop 時間更新為1.03／1.12秒，chosen診斷端到端2.96ms，與新record的1.031913963／1.121707580秒、2.962395476ms一致；全部非計時report字段與本人的隔離CPU實跑逐項完全相同。新文也明示draw包含192×192放大、GT與顯示門檻.25預測，符合源碼，不能把它當純推論時間。19單節重產仍exit0、case與依賴SHA一致。重新核過12頁case/dependency绑定及全部現文/圖/evidence SHA，更新上列指紋；未因時間變化重跑GPU。M19-01的confirmed/lost/unconfirmed新範圍亦已讀現文核回。

## 第三輪 T01 技術 delta 核回（2026-10-05）

第三輪非作者銜接讀者在 16.2 回顧 13.1、16.3 Progressive Loss 開頭發現兩句仍把「一對一」說成最終每 GT 一正。第二輪已核對 YOLO26 的 `topk2` 界線，卻未抓到這兩個回顧句；此處追加原疑點與修後核回，不改先前 first-read raw，也不把當時的 no issue 改寫為已查到。問題沿用 [transitions 的 T01](../transitions/modern-applications.md#新發現-t01版本回顧需要區分初選與最終篩選)，屬 M13-01 的跨頁文句遺漏；現已關閉。

本人完整實讀當前 16.2／16.3，當處已分清：固定 YOLOv10 的 top-1 限初選，衝突後每 GT 數量不由此保證；本章固定 YOLO26 在衝突後再篩 `topk2=1`，最終每 GT **至多一個，也可能沒有**。16.2 在名詞出現處先說「衝突後再篩一次」，無須先猜下一節的 `topk2`。16.3 STAL 的資格 0→4、先 7 名→衝突→最後至多 1 個與練習第 4 題相容，未把資格點當最終正樣本，未承諾恰一個。

固定 Ultralytics commit `441632cdfd19e22e60a4b1b1999d46326ca51ec4` 的官方 `loss.py` L1327–1328 是 many `tal_topk=10`、one `tal_topk=7, tal_topk2=1`；L374–381 把參數原樣傳入分配器。官方 `tal.py` L158–173 的順序為初選 mask → `select_highest_overlaps` → targets；L364–371 先解衝突，L373–380 再把品質乘當前 mask、每 GT row 取 `topk2`、以乘法保留該 subset。因此 row-wise 遮罩最多只有一個非零，零值並列也不會把先前已刪掉的點復活，最終可能零個。這是數量上界的原碼證明，不把「前 7 名」誤當最終數量保證。

本次把既存未修改的官方 AST probe 與固定 `u-tal.py` 複製到 `/tmp/16-topk2-delta/`，CPU 重新執行，assert 全過，結果與留存 JSON 完全相同：三 GT 衝突輸入後最後正數 `[0,0,1]`。僅重播解衝突方法，不冒充完整 YOLO26 forward／訓練，沒有 GPU、160／1600 步重訓或改 evidence。固定来源 URL、完整 SHA、當次命令及 plain JSON 結果見 [16-topk2-delta.json](probes/16-topk2-delta.json)，原 [probe](probes/v26-post-conflict.py) 與 [結果](probes/v26-post-conflict.json) 保留。

同時讀當前兩張 SVG 原文：16-head-paths 的 one 梯度只到 detach，部署只有 backbone／one，紫線比的是解碼前 raw；16-stal-candidates 的 `[7,7,9,9]`／`[0,0,16,16]` 與 (4／12) 格心仍按 16 SVG 單位／畫素一致，圖明說只是候選池。此處沒有做 Zensical browser 驗證，source 核對不充當手機可讀性驗收。兩個 case／兩張 SVG 的 SHA 和第二輪相同；root 在 10:40 重產的兩份 execution JSON 因執行時間等 metadata 更新而 SHA 改變。本人實讀新 JSON，核其 case／dependency 綁定、CPU、exit0／passed，並逐字確認 stdout 等於第二輪本人的隔離 CPU 輸出（數值、shape、首步紀錄都相同）。本次沒有因純文字修正重跑這兩個 case。

本次 delta 指紋（SHA-256；以本段快照為準，保留上方第二輪歷史快照）：

```text
docs/lessons/16-inference-head.md  4251983ae7b3f6685b6d8d6071d345308e6079f4b5681c571f00ef9083ff129c
docs/lessons/16-training.md  c84b993d07e157ce382e42d5635bc3532b84b853d955695de55c1dabce2ea37a
docs/assets/diagrams/16-head-paths.svg  1acac44ac36b5b847e55a22e24d2a97fc2095a25800e3ed0df71e75dbcebf2c4
docs/assets/diagrams/16-stal-candidates.svg  04d26cccdc1c78a9f72ce64550a6c156d0f9bccef78b895ff3080fa67c7fc49f
lesson_cases/16-inference-head.py  e76f4949c12b64c8f1470fda2a8b23efff198eaa342950aaba9cd3dd924c9c3d
lesson_cases/16-training.py  b50fb995486aa0ff042edb1486de27605fc0c3ec28f201ecb290b97a733fe31e
artifacts/checks/curriculum/16-inference-head.json  95a8dd773033000bb327f849348ded72bcf2350afdd13de124ddc1ea8af144ae
artifacts/checks/curriculum/16-training.json  286c24f3ece775b4a60df2afa9b9f373c335f08eb2c8386a583c6e10d1b18e38
```

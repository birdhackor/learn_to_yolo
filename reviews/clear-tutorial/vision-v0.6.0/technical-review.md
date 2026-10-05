# ViT／DINO v0.6.0 獨立技術與證據審閱

最終結論：**本次10篇ViT／DINO教材的第二輪技術與證據審閱通過，沒有未解決的必修技術問題。** 2026-10-05 12:14 UTC 在協調者通知首次閱讀修正完成後，重新讀取10篇當前正文、核對修改的圖及實作／證據fingerprints。先前引文定位與notebook入口問題均有實際修後複查。此結論只涵蓋下述技術與證據範圍，不替代首次分段閱讀、瀏覽器視覺驗收或第三輪銜接結論。

審查者：`vision_technical`，未參與教材或實作撰寫。先讀 `.agents/skills/clear-tutorial/SKILL.md` 與 `references/review-protocol.md`，未讀作者自查或其他審閱報告。此輪可以讀完整頁面、實作與原始來源，因此是技術審閱，不是分段盲讀。初始 Git HEAD 為 `823232bf9eba4074b4db4c99b6b076feaa1a9992`；新教材尚在工作樹，精確版本以 [初始 SHA manifest](technical-initial-sha256.json) 為準，不能只用 HEAD 代替。

範圍：10 個 `21-*`、`22-*`、`23-*` lesson cases 及相應正文／圖，`vision_transformer.py`、`vision_data.py`、`self_distillation.py`、`run_dino_pretrained.py`，10 份 curriculum execution JSON 與 `dinov2-pretrained.json`。沒有修改程式／教材，沒有 commit／push，也沒有重新啟動旧 GPU 實驗。

## 初步問題

1. **引文位置需更精確，非實作問題。** `22-features.md`〈原論文怎麼用 backbone 特徵〉與 `23-detection-bridge.md`〈這個 bridge 與兩個 DINO 名稱〉只指向 DINO §3.2 來支持取 backbone 特徵。實際 DINO `2104.14294v2` PDF p4 左欄的 §3.1〈Network architecture〉才有 “The features used in downstream tasks are the backbone f output.”；§3.2 PDF p5 的〈Evaluation protocols〉支持 linear／k-NN 凍結評測。建議分別指明兩個位置。已回報協調者，待修後複查。

   **本項修後實際複查：已通過。** 最終22-features註解已分別指向§3.1〈Network architecture〉及§3.2〈Implementation and evaluation protocols〉；23-detection-bridge也寫成§3.1取backbone／§3.2評測協議，與直接查閱的paper位置一致。

2. **Notebook 執行相容性，修正後需獨立再驗。** 協調者後續以實際notebook cell執行發現 `21-training.py` 的 `repo_dependencies(__file__)` 在無`__file__`的cell失敗，且`parse_args()`會解析kernel的`-f` argv。協調者正在修正此case並重產該份證據。本審查先前23個tests及獨立數值核查沒有覆蓋actual notebook entrypoint，不能拿它們宣稱這項已通過；待修後將不注入假`__file__`、保留kernel-style argv獨立核cell。

   **本項修後實際複查：已通過。** 修後case SHA-256為`6c9593ba8ef5fb0e4a1f445f0ccd22ce237a5766d87682dea78af8f12ef44322`；修前`fbf17b83a9daaf98c1b3e29cc54852369ea0379b95cd42bd9da3e3eec0ce4e16`保留在initial manifest。直接讀取同步notebook的第3格，bytes與case完全相同；以`__name__='__main__'`且沒有`__file__`的namespace執行，前後`__file__`均不存在，kernel-style `['ipykernel_launcher','-f','/tmp/vision-technical-kernel-connection.json']` argv亦保留。完整60步及30步checkpoint還原實際完成。另依repo既有`PYTHONPATH=.`約定執行一般CLI預設與既有`--output`，均完成且產物是本次新寫。三路loss history、梯度統計、heldout分數、weight-change與resume結果全與更新formal record相同；model／optimizer error0、suffix loss／RNG精確相同。詳見[entrypoint核查](technical-vit-entrypoints.json)及三份raw JSON。初次直接CLI漏設documented `PYTHONPATH=.`而得到`ModuleNotFoundError: miniyolo`，補齊正確呼叫方式後通過，沒有為此改任何程式。

本輪最終沒有發現阻礙核心模型／資料／分數結論的問題；修前問題與修後核查均保留於此報告。

## 原始來源直接核查

均透過 TLS 從下列 URL 實際取得，保存完整檔案 SHA-256 於 [來源 manifest](technical-original-sources.json)。PDF 的頁碼以下以實際 PDF page 計數；沒有只憑論文名稱或記憶認定。

|來源|實際取得版本與具體支持位置|核查結果|
|---|---|---|
|[ViT](https://arxiv.org/pdf/2010.11929)|取得 `2010.11929v2`，§3.1 PDF pp3–4、式 (1)–(4)；§4.1 Table 1 PDF p5；Appendix A PDF p13、D.4|支持非重疊 patch 共用線性投影、learned CLS／position、Pre-LN、兩條 residual、GELU、最終 CLS LN。原預訓練 head 為含隱藏層 MLP；教材明示單一線性 head 及 tiny 規模的差異。|
|[DINO](https://arxiv.org/pdf/2104.14294v2)|`2104.14294v2`，§3.1／Algorithm 1 PDF p3、〈Teacher network〉與〈Network architecture〉PDF p4；§3.2 PDF pp4–5；§5.3 PDF pp8–9|支持跨 view CE、排除同 view 配對、teacher stop-grad、student 後的 teacher EMA、raw logits 的 center EMA、teacher 低溫與 centering 的互補。teacher 沒有事先受監督訓練。官方 multi-crop 與本例兩個 global views 的差異已揭露。|
|[固定官方 DINO code](https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/main_dino.py)|commit `7c446df5b9f45747937fb0d72314eb9f7b66930a`；`DINOLoss.forward` lines 380–404、`update_center` lines 407–418；`train_one_epoch` optimizer／EMA lines 330–351；`DataAugmentationDINO`|官方 forward 使用舊 center、detach target、跳過同 view，然後更新 center；optimizer 後做 teacher EMA。教材本例把 center 更新放在 optimizer／teacher EMA 後，仍使用事先算好的 raw teacher logits，對下一步的 center 等價，沒有以 EMA 後 teacher 重算當步目標。|
|[DINOv2](https://arxiv.org/pdf/2304.07193v2)|`2304.07193v2`，§4 PDF pp5–6，〈Patch-level objective〉、〈Untying head weights〉、〈Sinkhorn-Knopp centering〉、〈KoLeo regularizer〉|student 的 masked patches 對齊 teacher 未遮住的對應位置；DINO 與 iBOT heads 分開；teacher 採 SK、student softmax；KoLeo 鼓勵 batch 特徵分散。教材只解說這些設計，没有宣稱 tiny trainer 已實作。|
|[Registers](https://arxiv.org/pdf/2309.16588)|實際取得 `2309.16588v2`；§2.2、Figure 6 PDF p5|register 是 patch embedding 後加的額外 learned tokens，沒有直接對應圖片 patch。它是後續修改，不能當成所有 DINOv2 baseline 原有設定。固定官方 hub 同時提供 `dinov2_vits14`（0 register）與 `dinov2_vits14_reg`（4 registers）。|
|[DINOv3](https://arxiv.org/pdf/2508.10104v1)|`2508.10104v1`，§4.2 PDF p12、式 (2)／(3)；§4.3 PDF p13；Appendix C PDF p58|Gram 以 L2-normalized patch features 的 pairwise dot products 約束關係。§4.2 用 earlier teacher，1M 後啟動 refinement，每 10k 將 Gram teacher 更新成 current EMA teacher；Appendix C 指最多三次更新。§4.3 用兩倍解析度 teacher、bicubic downsample 到 student grid。現文避免永久 initial snapshot 的誤解。手工 MSE 明示不是完整官方 loss／訓練。|
|[2022 DINO detector 官方 README](https://github.com/IDEA-Research/DINO/blob/d84a491d41898b3befd8294d1cf2614661fc0953/README.md)|commit `d84a491d41898b3befd8294d1cf2614661fc0953`，README line 5、citation|名稱是 “DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection”，對應 `2203.03605`，與 2021 自監督 DINO 分開。教材沒有把 teacher／center／SSL projector 當成 DETR detector 機制。|

官方 DINOv2 實跑固定 commit `e1277af2ba9496fbadf7aec6eba56e8d882d1e35`；直接讀核對過 SHA 的 `dinov2/hub/backbones.py`、`dinov2/models/vision_transformer.py`、`dinov2/layers/patch_embed.py` 及 README License。`vit_small` 確為 D=384、depth=12、6 heads，patch=14；`dinov2_vits14` 預設 0 register。官方 patch embed 是 Conv2d 後 `flatten(2).transpose(1,2)`，sequence 為 row-major；`x_norm_*` 是 LayerNorm 輸出，cosine 還需另外 L2 normalization。code 與 model weights 的 Apache-2.0 授權說明來自 fixed README lines 592–594。

## 逐項證據與獨立核查

10 份 curriculum JSON 均有實際 `exit_code=0`、`passed=true`；11 份紀錄全部 `dependencies_sha256` 與核查時工作樹一致，沒有發現 stale source hash。退出碼僅證明程式完成；下列資料另外核對機制、更新與數據支持範圍。

|教材／實驗|實作與已保存數據支持|獨立核查與結論範圍|
|---|---|---|
|21-patches|32×32 RGB、P=8、16 patches、raw192、D32、CLS index0、17 tokens；patch 內容 C／row／col，patch 次序 row-major。手工192→3投影與 TinyViT32維分開。|另用任意 RGB pixels 比較 Conv2d 與 flatten 後的等價 Linear，最大差 `7.152557e-7`。`[row=1,col=2]` 對 patch index6。並非分類訓練或偵測。|
|21-attention|4 heads、每 head8維、attention `[B,4,17,17]`；return attention 在 dropout 前，每列和1。手工小 attention 做1次 SGD，TinyViT swap 部分沒有訓練。|讀核 Q/K/V reshape、query/key軸與輸出投影；固定位置／完整向量重排區分正確。17²→65²是14.619倍 score count，不是 runtime 倍數。|
|21-transformer|兩個 Pre-LN blocks、兩條 residual、LN／GELU／Linear head，CLS與patch保留，23,970參數。manual residual 與 block一致。|獨立參數計數23970；模型架構和 ViT式(1)–(4)一致於已明示的 tiny simplification。尚無 optimizer，不以 shape 或位置改變宣稱學會分類。|
|21-training|完整 TinyViT所有參數入 AdamW；CPU、dropout=.1、60步、lr=.003、wd=.01。train128 seed101，val/test64 seeds202/303；固定步數、不按 validation 選 checkpoint。|讀核所有參數有限且非零總梯度、optimizer更新、六組權重變化、eval固定train loss與heldout64/64。既有測試也實跑包含短訓練、model／optimizer／RNG精確接续。結論只支持同生成規則的新圖。|
|22-views|生成1張 seed101圖與 view seed22，crop原圖xyxy，resize／flip／brightness；scale是面積且取整。原框不當變換後答案。|SSL函式只收images；crop右下exclusive。大crop仍可能漏物件；教材明示 crop風險與保留色彩的設計，人為增強選擇不等於 label進loss。此case只是資料操作。|
|22-collapse|手寫常數features/logits；uniform CE≈2.7726，peaked CE≈6.18e-7，std0，常數紅預測4/8。|沒有optimizer或訓練、沒有center ablation；低CE與有用features分開。不是原DINO防塌縮效果的實測。|
|22-distillation|兩global views、同初始化student/teacher；head CLS32→64→16，bottleneck L2，最後16無bias logits；原 supervised head不用且frozen。只AdamW student，teacher no-grad／EMA，rawcenterEMA。|以非預設參數實際重算一步：CE=`2.0031199455`完全相同，center公式最大差`7.45e-9`、teacherEMA最大差`1.19e-7`。另跑3+2步、dropout=.1、非預設LR／温度／EMA／crop的完整checkpoint續跑，student／teacher／optimizer／center、step、config、torch/python/view RNG、nextviews全相同。|
|22-features|凍結teacher CLS `[N,32]`；train-only reference bank與標準化；random baseline來自同一次initial backbone；相同seed900 head、Adam .02、120 steps。|從保存的SSL checkpoint重算features、直接off-diagonal pair cosine、獨立訓練probe。兩套1-NN均val/test64/64、probe train128/128與val/test64/64。std/cosine：SSL `.3295054/.8559933`，random `.0842096/.9917242`。均未改backbone。相等得分沒有支持SSL收益。|
|23-dino-versions|三個指定normalized向量：共同旋轉featureMSE1、GramMSE0；常數GramMSE≈.2603495。選讀官方DINOv2 frozen inference使用另一份模型／權重。|原始來源支持版本差異。saved NPZ sha吻合；重新cosine得到CLS `.9850406647`，query index68 `(4,4)`／pixel`[56,56,70,70]`，top1 index187 `(11,11)` `.8331384063`、top2 index68 `(4,4)` `.7685461640`，全部top8索引吻合。此相似度沒有類別機率／IoU意義，沒有自然圖品質排名。|
|23-detection-bridge|自己從零160步SSL→frozen teacher patches `[B,16,32]`→row-major `[B,32,4,4]`；train-only每位置/channel統計；512→64GELU→class2+box4，head監督200步。CE+10 SmoothL1(normalizedxyxy)。|從saved backbone/head與獨立生成split重建全部推論，train mean/std逐位吻合。用Python scalar面積公式另算IoU，train/val/test均值`.8593764/.6160325/.6056988`、joint張數128/49/54、顏色全對。4個test展示IoU`.4459248/.3522003/.5610500/.6325744`吻合。一次預測按圖片配一真值，沒有AP/mAP；沒有random框baseline，不能推SSL/CNN排名。|

資料本身也另做逐pixel SHA比對：train／val／test三組兩兩完全相同圖片的交集皆0，類別數是64/64、32/32、32/32。這與不同seed、同分布小型合成例相符，不支持自然照片泛化。位置／尺寸／亮度的抽樣不依label；兩類仍共享矩形形狀，顏色才是答案。

checkpoint核查還確認student與teacher state、optimizer參數／動量與步數、loss state中的center、seed、initial baseline、backbone/config、view／torch／Python RNG均有保存。現在是CPU固定安排，沒有NumPy、CUDA或可變scheduler參與本trainer；optimizer state保留weight_decay=.01，clip norm=3與view flip/photometric為已記source中的固定程式設定。核查不宣稱跨硬體／不同PyTorch的逐位重現。

## 圖與保存輸出的一致性

已核對 DINO 一步SVG的1舊center→2student optimizer→3teacher EMA→4rawlogits centerEMA，與實作一致；修後只把圈號改成普通數字，公式／箭頭資料來源與更新順序沒有改變。官方DINOv2 SVG query `(4,4)`與top1 `(11,11)`、top2 `(4,4)`皆與數值及row-major pixel映射相符。

修後`22-features.svg`另外直接解析160個曲線點，與實際完整history的step／loss線性座標映射逐點吻合，最大x／y誤差`4.96e-7／5.13e-7`SVG units；loss實際範圍`1.2611022～2.7611330`，全部點都在plot clip rectangle內。raw、完整loss序列與source history SHA見[修後曲線核查](technical-final-curve.json)。這項只核資料座標與clip邊界，不代替畫面可讀性檢查。

`23-detection-predictions.svg` 另外用XML解出四張embedded PNG：與seed303 test0..3的RGB逐pixel相同（round(`float*255`)量化）；八個真值／預測SVG rectangles均與versioned JSON中的xyxy縮放相符，最大座標差小於`0.000005` SVG units。第一次核查草稿錯誤假定floor量化，診斷為至多1個uint8值之差，改用round後完全吻合，沒有修改任何repo圖。

此技術審閱沒有做桌面／手機瀏覽器實際呈現檢查，文字可讀性、font與手機排版仍由相應視覺／首次閱讀修後複查負責，不能以XML正確或build成功代替。

## 可追溯執行紀錄

- `.venv-model/bin/python -m pytest tests/test_self_distillation.py tests/test_vision_transformer.py -q`：**23 passed in 4.29s**。
- `.venv-model/bin/python /tmp/vision-technical-sources/independent_check.py`：actual CPU獨立patch／一step公式／非預設checkpoint／saved bridge／DINOv2 arrays核查完成；同內容腳本保存於ignored `artifacts/runs/vision-technical/independent_check.py`，SHA-256 `bb6a7bfec519f800c7e0db647da5e10e298ce12ff02eac9320305d9c122d283e`。raw結果見 [independent checks](technical-independent-checks.json)。
- probe獨立重算raw見 [probe recompute](technical-probe-recompute.json)：保存SSL權重重新取train/val/test features，兩套特徵各以train統計標準化、seed900的Linear(32,2)、Adam=.02／120次full batch，另直接對off-diagonal cosine矩陣取平均。
- 圖座標／embedded pixels核查raw見 [SVG coordinate recompute](technical-svg-coordinate-recompute.json)。
- 修後21-training actual notebook cell、CLI default及`--output`三路60步核查見[entrypoints](technical-vit-entrypoints.json)，原始輸出保存於[notebook cell](technical-vit-kernel_no_file-raw.json)、[CLI default](technical-vit-cli_default-raw.json)、[CLI --output](technical-vit-cli_output-raw.json)。kernel bootstrap完整原文保存於ignored `artifacts/runs/vision-technical/kernel-vit-check.py`，執行時以`python -c`讀取actual notebook cell而非執行這個bootstrap的`__file__`。
- 完整首次local SHA見 [initial manifest](technical-initial-sha256.json)；paper／fixed-source downloaded SHA見 [original sources](technical-original-sources.json)；執行狀態與命令見 [metadata](technical-check-metadata.json)。

未重跑160步SSL／200步bridge完整訓練，相關更新證據來自source hash一致的實際紀錄與現碼assert；上述saved-weight獨立推論核查補足分數及座標核驗。未執行新的官方DINOv2 forward，重算使用SHA吻合的NPZ；官方archive、weights、每個Python與LICENSE實體檔的SHA另外逐一吻合實跑紀錄。它們支持保存結果的可核對性，不能取代尚未做的自然圖像效能評測。

## 修後複查

已在首次閱讀修正完成通知後複查全部10頁。修後主要差異是：21.1明確定位dropout=0的手工實驗；21.2更清楚說明跨patch的用途及完整TinyViT最終CLS的觀察點；21.3把兩條Pre-LN公式分行；22.1提前交代crop座標、分清另生成1張紅圖與前章128張資料；22.4提前定義random／SSL，並修正DINO引文；23.2修正相同引文。資料切分、模型配置、訓練參數、label使用、評分口徑及限制沒有改動。

新增21.2素材敘述另做相稱的CPU資料核查：seed101、128張資料的train3為紅色，bbox=`[4,9,19,25]`；閱讀用patch1 `[0,0,8,8]`物件交集0，patch6 `[8,8,16,16]`物件交集56pixel²，支持「patch1只有背景、patch6已有紅色」的新句。raw見[素材核查](technical-final-local-material.json)。未為文句／排版修正重跑既有同版本probes或完整SSL。

修後唯一程式差異是21-training的notebook入口相容性，已用actual cell／CLI三路60步實跑核查；其新formal紀錄的數值與先前相同。其餘9個case、3個model/data/SSL檔及官方CLI source SHA與初始版本相同。全部11份evidence的dependency SHA再次逐一匹配當前source。最終manifest在讀完10頁後再比對，沒有閱讀中途變動；詳見[最終SHA manifest](technical-final-sha256.json)與[版本差異](technical-version-delta.json)。

|最終核心source|SHA-256|
|---|---|
|`miniyolo/vision_transformer.py`|`212c8fcd59df0abd2ff9d0e545dbee051bbbe306089f5e0973d47783090fa394`|
|`miniyolo/vision_data.py`|`92b1cc3885a0b897eba2a24ac1cd604c55b5ea903a1886904e64b9364c67dde1`|
|`miniyolo/self_distillation.py`|`15d60e80fe19a14d980cd44999143a4da18847f19737fb1d762f85c90f8f11d0`|
|`scripts/run_dino_pretrained.py`|`4cb7a5903e6ba358e3a03974f824a4f940c972578a5aa6cfbd732676c1582cb3`|
|`lesson_cases/21-training.py`|`6c9593ba8ef5fb0e4a1f445f0ccd22ce237a5766d87682dea78af8f12ef44322`|

教材的分數能由指定資料／權重／統計量重算，核心公式與simplification可由original papers及fixed official code支持；沒有把手算、完成backward或成功退出改稱模型能力，也沒有把相等random/SSL分數、兩張controlled drawings或單物件IoU擴大成自然影像／架構優劣主張。此範圍的第二輪技術審閱通過；本紀錄沒有改標為首次盲讀通過。

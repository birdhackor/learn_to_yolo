# Evolution 非作者技術審查（第 2 輪）

審查者：modern；日期 2026-10-05；基底 HEAD `16f69103d1f216d1024a40610623083557324701`，檢查工作樹終稿。這 11 頁均非本審查者撰寫；03/04/05 自己的作者改稿不在此報告範圍。依 repo clear-tutorial skill 檢查「輸入／輸出契約→小算例→實際程式→原始機制／證據限定」，未採作者自查作獨立證據。計畫 release 為 lessons-v0.5.0，檢查時 tag 尚待主代理建立，不把尚未存在的 Colab tag 當已執行證據。

目前沒有 blocking issue。確定發現的 E-01（官方全零衝突邊界描述，low）已由 root 修正並經實際核回。V-01 是手機縮圖字小的視覺觀察，已回報、等 6 張專用圖更新後追加獨立複核；不拿 SVG set_content 預覽冒稱實際網站全站 browser 驗證。

## 方法、證據與未驗範圍

- 逐頁讀完整正文、完整 lesson case、所有公布 evidence 欄位（環境、source/dependencies hash、完整 stdout、stderr、exit、計時範圍）。所有宣告 dependency_sha256 都逐檔比對。11 个主程式在 /tmp 隔離副本、PyTorch 2.9.1+cpu／2 threads 重跑，exit0且stdout與公布紀錄逐字一致。最新 JSON 中只刷新 metadata/timing 的記錄亦已讀回核一致；以下hash對應報告寫入時工作樹。
- 10章额外40步学习也在 /tmp 复制的原脚本／原依赖重跑，未传 --record，loss_history／prediction／parameter／weight_delta／metric与公布报告 exact。自己重新以完整未舍入框求IoU/AP，沒有用作者摘要代替。
- 13項額外CPU檢查含完整練習正／負結果、bilinear梯度、IoU補充梯度／位置表、DFL兩邊加權、官方原method全零同分，以及保存prediction的metrics。預期失敗的80步大框與3.2格range assert被視為符合正文預測，不列為教材故障。
- 原始論文PDF為可固定的 arXiv版本，實際下載並讀相关章／式／圖；原碼使用下列固定commit，所有快取原碼重新計sha對上下載索引。並非僅凭印象或讀二手摘要。
- 全15張SVG在系统Chromium set_content，以1000/390 viewport、body边距16、SVG width100%獨立截圖及經CTM換回viewBox的文字BBox檢查：無越出viewBox的text。實際目視14張390預覽及生成learning圖1000預覽，核座標、方向、units、圖例、數字。此為圖檔預覽，沒有對實際網站CSS、折疊／sticky header／全部desktop/mobile頁面作完成聲明，主代理另負責全站browser。
- 未做GPU、部署、commit/push；未修改共享 lesson_cases／miniyolo／scripts／教材／已有reviews／coverage或shared evidence。此報告與 /tmp 個人核對資料是本輪產物。

重跑輸出：`/tmp/clear-tutorial-evolution-tech/cpu-case-results.json`；補充：`extra-check-results.json`；40步：`learning-reproduction/report.json`與`learning-reproduction-comparison.json`；圖：`svg/audit.json`及15×2 screenshots。/tmp 不是長期發布證據，下文保留可复查的方法、数值、完整hash與source版本。

## 逐頁結果與完整 SHA-256

### 09-anchors

方法與範圍：完整閱讀正文、`lesson_cases/09-anchors.py:size_iou/main`、4 行公布輸出與完整 JSON，手算中心所在格、尺寸 IoU、log 寬高和 mask，再在隔離副本執行完整程式。看過圖中的中心、同中心尺寸框、兩槽與其餘 30 個負槽。原論文／原碼核對用 YOLOv2 v1 §2 的 Dimension Clusters／Direct location prediction、Darknet 固定 region_layer.c 236–306 與 yolov2-voc.cfg 243–258。

數值與機制：GT `[8,12,24,28]` 中心 `(16,20)`，格 `(1,1)`、比例 `[0,.25]`，兩 anchor 尺寸 IoU `[1,.25]`，tw/th `[0,0]`。中心的 loss target 是比例，反解有限 logit 才 clamp 到 `1e-4`；decode 的 x 偏差 `.0016` 因而正確，不能把 −9.21024 當 center MSE target。32 槽＝1 positive＋1 ignore＋30 negative；31 個有效 obj 槽平均 BCE，ignore 不計框、obj、class。原 Darknet ignore 用當下 decoded 框對 GT 的 IoU，`.6`；本例用同格的尺寸 IoU `>.2`，已明寫不同。VOC 原 cfg `rescore=1/softmax=1` 亦不等於本例所有訓練細節；頁面沒有宣稱重現原訓練。224 個可學 raw 數、一次更新、沒有 CNN／AP 的限定保留。练习32×16的 IoU `.5/.125` 与 log `[ln2,0]`／`[ln4,ln2]` 可手算一致。

結論：無阻塞或新增數值／契約問題。SVG 縮圖可追中心與 slot，但小標籤有 V-01 的手機縮放觀察；實際頁 CSS 待主代理查。

| 檢查檔案 | SHA-256 |
| --- | --- |
| `docs/lessons/09-anchors.md` | `7604bfaecb11e818a45d2a745c8450801e73c9541b351540ffc7f479080b3cea` |
| `lesson_cases/09-anchors.py` | `7a7e8c431340e33f55fc8afc90136b225d18387940d6677eab9e8aecacd31469` |
| `artifacts/checks/curriculum/09-anchors.json` | `2157849a3e3b4d33880cb9c1ded9887f04d4cfe818c5d836d00dd6b6d169399c` |
| `docs/assets/diagrams/09-anchors.svg` | `f161b7a31f424183494b2da33f6623501d2dcd84bd3eaae63d1f3acf2ce41e7c` |

### 09-anchor-clustering

方法與範圍：完整讀正文、`size_iou/cluster/main`、完整公布 JSON，核六筆固定尺寸的每群平均／中位數、弱基線和新形狀，隔離執行。讀 YOLOv2 v1 §2 Dimension Clusters、Figure 2、Table 1；沒有把本例平均更新冒稱原文唯一算法。

數值與機制：歸群 `[0,0,0,1,1,1]`，平均 anchors `[25/3,25/3]`、`[94/3,50/3]`；平均最佳尺寸 IoU `.3817→.9119`，同一歸群改逐維中位數 `[8,8]/[32,16]` 得 `.9340`；新來源 `[4,40]/[40,4]` 平均 `.1975`。上一頁手填16×16＋8×8約 `.7093` 是另一個手算對照，未冒列為執行 stdout。共同縮放2倍的 IoU 不變，但固定 anchor 而只放大物件，IoU會變，兩者有分開。1−IoU 是共中心尺寸相似度，沒有位置也不是 AP。平方歐氏距離的平均最小化推導成立；IoU 距離改平均無單調保證的反例 `[1,10]/[10,1]`，總距離 `.9474→1.6835` 正確。只用 train、依模型實際 resize/letterbox 後尺寸收集、固定初始化與空群留舊中心的限定保留。

結論：無阻塞或新增技術問題；不對這個固定兩群例子推廣全域最佳解／偵測品質。圖中 IoU 面積、六點及均值移動一致；V-01 縮圖字小待新版。

| 檢查檔案 | SHA-256 |
| --- | --- |
| `docs/lessons/09-anchor-clustering.md` | `482f74afd9761b0a40d4f9b56e60d19e964529c23d13b10d8c0ad1496d1cd3cf` |
| `lesson_cases/09-anchor-clustering.py` | `1262a5b3c925d9eb2e9c03cbe70279355a0575d3e0c8d7f28522883e692221a4` |
| `artifacts/checks/curriculum/09-anchor-clustering.json` | `a9829cc97234435855e30fecfa2c5272b31304a8b25da754f52f7c8d0109070d` |
| `docs/assets/diagrams/09-anchor-clustering.svg` | `37df7a2a58f68d14885bfb5a2cfc983ec3a9366f7e3af693f0c6a5d361efc029` |

### 10-multiscale

方法與範圍：完整讀 `TwoScale/main`、正文與兩份完整公布 JSON；讀 `miniyolo.targets:build_targets`、`losses:grid_loss`、`inference:decode_grid`、`geometry:nms/box_iou`、`metrics:evaluate_ap/_interpolated_ap`；完整讀 `scripts/run_multiscale_learning.py`。核兩尺度 target、候選數、感受野、重複框與 merge/NMS，隔離跑主例，再在 /tmp 複製 scripts/miniyolo/lesson_cases，原樣 CPU 重跑40步，未用 `--record`。核 YOLOv3 v1 §§2.1–2.4、Darknet固定 yolo_layer.c 83–107／159–223、yolov3.cfg 三個 yolo/upsample/route 段；YOLOv2 v1 Multi-Scale Training 核輸入尺寸隨機變更與本節多 head 的差別。

數值與機制：三次 stride2 得 fine8×8、16C、理論RF15；再一次得 coarse4×4、32C、RF31。small target 在 coarse `[.5625,.5625,.125,.125]`、fine `[.125,.125,.125,.125]`；大框只由 coarse 负责，因此另一路中心負格不是 ignore。總候選80、raw元素560，fine1×1額外119參數，全模型8702。原 YOLOv3 三尺度、每尺度3anchor、獨立 sigmoid 類別、跨9anchor挑一個最好尺寸、其餘符合 prediction-IoU條件可ignore；本例兩尺度、無anchor、類別softmax、人選尺寸責任、無ignore，已明寫。paper ignore `.5` 与固定 cfg `.7` 已區分。

40步 evidence：同一張人工訓練圖、seed7、Adam `.01`，loss 在第1／40次更新前 `3.6593661308/.0703368783`，40步後權重L2差 `6.5035628999`。独立重跑的40點loss_history、三個prediction、參數、delta、AP/precision/recall與報告逐值 exact。由保存未四捨五入框再算，#0對紅GT IoU `.441043`、#1對藍GT `.862990`、#2 `.299122`；#1/#2互IoU `.278879`，所以NMS留#2合理。class0 AP=0、class1 AP=1，mAP50=.5、precision=1/3、recall=.5；#2低分FP在TP之後不降低all-points插值AP。畫框表和新增直式圖的 `.763/.836/.131`、`.44/.86/.30` 都對。#0小框FN，未把loss低／mAP .5說成小物件找到。生成 learning 圖與新圖數據一致。

結論：無阻塞或新增技術問題。只驗同訓練圖學習，沒有 held-out小物件AP，也沒有單尺度公平比較；兩段NMS在重疊鏈未必等於一次全局NMS的限定保留。四張新增圖 mobile 縮圖約13px以上可讀；原生成雙欄圖縮圖小，正文已有清楚直式實測圖。

| 檢查檔案 | SHA-256 |
| --- | --- |
| `docs/lessons/10-multiscale.md` | `f8c07a9dde22911e407fa18c5b8ecd9c30451f8f91821f990314573b0c9e4578` |
| `lesson_cases/10-multiscale.py` | `add9aefa8efec7180d06ab255807ac4fa6847c490c8447774191f0a6566ed6a6` |
| `artifacts/checks/curriculum/10-multiscale.json` | `6ffea3254bce8b21b1aa2eaa0a0165641f7776748474d5e5ee706c2599204688` |
| `docs/assets/diagrams/10-multiscale-coarse.svg` | `a2cb024b373a8e9cef993dddffc1e433fbb744e8cd25708bd7af0aad1cf7fab4` |
| `docs/assets/diagrams/10-multiscale-fine.svg` | `a768a4d7dad9f1e5cc062d685e1a642dbaff14a89c8d14b2c38e9320031a3dc5` |
| `docs/assets/diagrams/10-multiscale-branches.svg` | `ed6adbab0e7222d493f9a84ceb3d935edb2456e1cae4e47c2c864907a00a13ce` |
| `docs/assets/diagrams/10-multiscale-observed-predictions.svg` | `f64ab6a0957ee14c2ed28a08d50a272f90065a03c671ab8bd38917c228e85c95` |
| `docs/assets/diagrams/10-multiscale-learning.svg` | `4c985b8824c35b49b6ca1a88df45d7980e60176906c51dff86feab779aaabd17` |
| `artifacts/checks/curriculum/10-multiscale-learning.json` | `b825d38aa28ad98d0c84fe38f8be79431fae3f13d18f2690e5028da6f122ec09` |
| `scripts/run_multiscale_learning.py` | `db5480b8b226793092a6c59b4c79197ad0d33e7327a384f5b9a46eb573413e89` |

### 11-csp

方法與範圍：完整讀正文、`CSP/Full/main`、完整 JSON；手算channel與參數、對照每個參數／兩組輸入梯度斷言與一次SGD。隔離執行。原典讀 CSPNet v1 摘要、§3／Figures2–4（transition→concat→transition與兩種fusion次序）；原碼讀 YOLOv5 v6.0 commit956be... common.py `Conv/Bottleneck/BottleneckCSP/C3` 37–47／95–136 與 yolov5s.yaml backbone/head。另讀 YOLOv4 v1 §3與固定Darknet cfg backbone。

數值與機制：本例chunk前4bypass、後4走兩層3×3/ReLU，再concat8C後1×1fuse；不是兩路相加，也不是F=0時整塊identity。Full `2×584+72=1240`、CSP `2×148+72=368`；容量不同、參數比例不等於延遲／AP。Full輸入前後4C梯度L1 `.033093/.029087`，CSP `.255476/.019409` 与完整輸出逐字一致；兩模型各重新抽x，不能跨模型比品質。每個參數非None、有限、非全0以及fuse更新確實被斷言檢查；僅輸入兩半梯度非零不足排除branch漏接。原文所稱duplicate-gradient問題與本例梯度L1大小不同，已分開。原CSPDenseNet的partial transition并不等同toy concat後單fuse；YOLOv5 C3是兩路1×1投影不是chunk、Bottleneck可有shortcut、Conv無bias/BN/SiLU，backbone/head shortcut配置差異可核。

結論：無阻塞或新增確定技術問題；初始化梯度解說只限本例，不應拿來證明所有卷積層一律衰減。圖的Full/CSP線路、8→4/4→8、concat與參數一致。V-01 mobile縮圖字小待新版。

| 檢查檔案 | SHA-256 |
| --- | --- |
| `docs/lessons/11-csp.md` | `40058c08d8d01a1f9ef4ce11d79ac9e89a6603470fc41ea2c08d25ccba476270` |
| `lesson_cases/11-csp.py` | `9710906769f4d757331a7c99d2ba89d8d45c40d52c76f1d2a32a510e13a4f556` |
| `artifacts/checks/curriculum/11-csp.json` | `d376a3741add2932b78363be8c898c548ba608d0f961408915f26ecbae672abc` |
| `docs/assets/diagrams/11-csp.svg` | `88f5986dbbcbe84a6c11d867df37879ca018e4163960c085668c0afff676bf3b` |

### 11-fusion

方法與範圍：完整讀 `Fusion.forward/main`、正文、完整 JSON、主圖和選讀雙線性座標圖。隔離執行主程式，再CPU核2×2補充插值／平方平均梯度。原典核 FPN v2 §3（nearest、lateral1×1、add、合併後3×3減aliasing）、PANet v2 §3.1／Figure2（bottom-up 3×3 stride2＋add＋3×3，以及<10層捷徑）；YOLOv4 v1 §3.3／Figure6改concat；YOLOv3 v1 §2.3与固定cfg route；YOLOv5 v6.0 yaml head。

數值與機制：本程式沒有圖／backbone／head，隨機兩輸入 `[1,8,8,8]`、`[1,16,4,4]`。reduce→nearest→concat→mix為 `[1,8,4,4]→[1,8,8,8]→[1,16,8,8]→[1,8,8,8]`。concat只要求B/H/W同，不要求C同；unreduced16C可接成24C，源碼真assert此契約。參數136+1160=1296；不reduce為1736；按第10章16/32C真正接入neck時528+4624=5152，不是直接1296。concat單activationfloat32為4096B；不含其他訓練記憶體。100輸入三次stride2→13、deep7，用scale_factor2→14會不配，而size13合法；shape相同仍不保證pixel格點對齊，已限制隨機tensor例。

補充數字：nearest源梯度全4；預設bilinear第一列 `[1,1.25,1.75,2]`、True為 `[1,4/3,5/3,2]`。輸出平方平均的nearest源梯度 `.5/1/1.5/2`，False bilinear `.78125/1.09375/1.40625/1.71875`，CPU相符。1×1與nearest複製的代數交換、在小圖先reduce乘加2048而非8192成立。程式每個reduce/mix參數梯度非None／有限／非全0以及mix更新均有斷言。

結論：無阻塞或新增技術問題。主圖top-down真向下、lateral真同尺度、輸出標為feature且未畫未實作PAN/head；本文沒把隨機shape實驗當融合AP收益實测。兩圖 mobile 縮圖清楚。

| 檢查檔案 | SHA-256 |
| --- | --- |
| `docs/lessons/11-fusion.md` | `e5adc94096402154bae57fcd25306db268043d6dbb532eb5b3523eb4d4a138b7` |
| `lesson_cases/11-fusion.py` | `4de6c1260d6f76f55013b4d17d935dbff46f4c1ec21975cbae5b065cd887af63` |
| `artifacts/checks/curriculum/11-fusion.json` | `9c6c6a611744a0381a1aa802cdbce3bef9cfde712d45c79d5be7aa87e51ed946` |
| `docs/assets/diagrams/11-fusion.svg` | `7619f421c06e28a559933091f9d15e215c4fd61f79afbd5c9fc90374b195affc` |
| `docs/assets/diagrams/11-bilinear-coordinates.svg` | `4f87851588f338428e6e72f80eb4db0c345aefb83ebb2eefb5c299f14d0a3a8b` |

### 11-augmentation

方法與範圍：完整讀 `horizontal_flip/crop/main`、正文、完整 JSON与兩圖；逐值對照預期整張圖斷言而非只對box/doubleflip；隔離執行。原典核 YOLOv4 v1 §2.2／§3.3 Mosaic四圖與bag of freebies；固定YOLOv5 v6.0 augmentations.py `random_perspective/box_candidates/mixup/letterbox`、datasets.py `__getitem__/load_mosaic`、hyp.scratch.yaml fliplr `.5`。讀ShapeDataset.__getitem__確定每索引重設自己的Generator。

數值與機制：框是連續半開邊界，W−x；pixel編號W−1−x。GT `[8,12,24,28]` 翻成 `[40,12,56,28]`，紅畫素x40…55、y12…27；doubleflip必要但不足，可逆的錯誤公式仍會还原，源碼另核整圖。中心 `(16,20)→(48,20)` 責任 `(1,1)→(3,1)`。crop原點 `(16,8)`、32×32，框先 `[-8,4,8,20]` 再 `[0,4,8,20]`，可見128/256=.5，`.5>=`保留、`.6`刪且labels同keep。0門檻仍加visible>0避免零面積；完全內部crop契約已明寫。空圖是零pixel與Float[0,4]/Long[0]；刪GT但保留紅pixel是漏標前景，不能當空圖。resize×2手算 `[0,8,16,40]`、新中心 `(8,24)`、新格 `(0,1)`；沒有冒稱主程式實作此resize。藍框練習clip `[24,28,32,32]`、32/256=.125，雙框keep `[True,False]`、label0，手算對。

來源細節：v5 bbox random-perspective候選filter寬高>2、面積比>.1、aspect<20且整列cls/box一起篩；原碼segment路徑另用.01，本文bbox例不混用。mixup全labels concatenate；fliplr確實在datasets而非augmentations，normalized cx變1−cx和本例一致。seed固定不代表同圖每epoch同變換；ShapeDataset既有每索引Generator若再用它抽增強會固定，文章提示合理。

結論：無阻塞或新增技術問題；主圖有V-01小字觀察，新的crop→resize→targets圖mobile清楚，沒有把手算說成已訓練或已做預算比較。

| 檢查檔案 | SHA-256 |
| --- | --- |
| `docs/lessons/11-augmentation.md` | `912b5cec4f0ae0e17b6cbbfe0a24d1b2082f690950770b74c113efa18ade6fe6` |
| `lesson_cases/11-augmentation.py` | `5665b9ddc7a920ec45041987096e49c931515a3f62c76e5e774c7d8031b1ddde` |
| `artifacts/checks/curriculum/11-augmentation.json` | `68d166c4cdff4d1f091de32b76e90f001a1844fbfacf743488b85571056e42df` |
| `docs/assets/diagrams/11-augmentation.svg` | `202b88c0da8cf4d3be37282491d7a805f3e8e6720f71640df1f3ca6ea202b156` |
| `docs/assets/diagrams/11-crop-resize-target.svg` | `74ea43b6fced49b79a6a09bf16aed6b4c52c4f045d19270aecc6f5059344d7ab` |

### 11-iou-loss

方法與範圍：完整讀 `losses/box_from_center/main`、正文、完整 JSON与圖；原樣CPU重跑，再令center與wh可學核補充gradient、DIoU位置表、兩題loss。核GIoU v2 §§3–4／Algorithm2与IoU=0時2−U/Ac；DIoU/CIoU v1 Eqs6–12／Figure2與論文省略v梯度分母的限定；YOLOv4 v1使用CIoU，YOLOv5 v6.0 loss.py135–136与metrics.py bbox_iou223–227 no_grad α。

數值與機制：G `[8,12,24,28]`、P `[32,12,48,28]`，面积各256、union512、C40×16=640、ρ²576/c²1856。loss1／1.2／1.310345／1.310345；中心梯度 IoU[0,0]、GIoU[.02,0]。独立lr50把x40→39、C624、loss1.179487；lr100 DIoU x38.7515、loss1.294497。圖的新1.179487正確（已關閉原modern0027圖1.174987）。不可微相接、上下緣相同與PyTorch取值說明保留，沒有把y梯度0解讀為完全無y影響。DIoU表右下移1pixelρ²577/c²1889→loss1.305453，距离增加而比值下降成立；不是第三次SGD更新。包含框例GIoU=.75、DIoU=.765625亦正確。

補充核算：可學w時GIoU dw=−.015、DIoU dw≈−.006688；center固定尺寸CIoU/DIoU中心梯度一致與αdetach成立。練習中心36 DIoU1.257732/GIoU1.111111；宽32框v≈.041956、α≈.040267、DIoU1.264865、CIoU1.266554，CPU對照。GT finite/正寬高是assert，pred倒置只clamp不会报错、epsilon不修標註，已明寫。MiniYOLO仍MSE；若自己改需正格同座標xyxy且保留計算圖，不可直接用no_grad decode_grid/NMS。

結論：無阻塞或未關閉數值問題。原modern0033練習也明確區分IoU求梯度、GIoU/DIoU只求loss。圖C/ρ/c、单位pixel／pixel²及移一步數字對；mobile圖清楚。没有推廣單框實驗到AP必升。

| 檢查檔案 | SHA-256 |
| --- | --- |
| `docs/lessons/11-iou-loss.md` | `e6668a24141a24376dc455cb66cd941f9abc48ad6787007a51e4569e32dff7b4` |
| `lesson_cases/11-iou-loss.py` | `ddce126434321bddf95543a02f21b082602024b3ba0ef6992cf8b2be70647325` |
| `artifacts/checks/curriculum/11-iou-loss.json` | `ce0124fad2b98449d926e1969ed28f9f59bb868460a33b49a87e2e2c6044edab` |
| `docs/assets/diagrams/11-iou-loss.svg` | `b9ea80169cfe1940bb9627d5841367f4989d17cfef8d07b5155ce24493e8c3d5` |

### 12-anchor-free

方法與範圍：完整讀 `decode/main`、正文、完整 JSON與帶格點圖；隔離跑完整例，再按正文修改清單跑stride4／大框80與100步。核FCOS v1 §3.1／Eq1的point與LTRB，再讀固定Ultralytics441... tal.py `make_anchors/dist2bbox`、loss.py v8DetectionLoss `bbox_decode/get_assigned_targets_and_loss`，確認官方grid點單位与toy pixel點單位的差別。

數值與機制：point(28,28)、stride8、GT `[12,16,40,36]`，pixelLTRB16/12/12/8，cell距離2/1.5/1.5/1；decode只把距離乘stride一次。point并非GT中心(26,26)，因此左右交換可被往返断言抓。outside(44,28)要r=−.5cell，toy softplus不能精確表示。四可學raw不是CNN；SmoothL1 beta1 mean4初loss .376236→80步 .000012，初梯度−.125/−.1009/−.1009/−.0384，不由target對稱與否決定。stride4点22得到 `[2.5,1.5,4.5,3.5]`，正文列出的decode三處／target断言同步后80步确实pass；y2改62为10cell时80步在after<before/100失败、100步通过。练习限制学习率／步数的解说成立。

結論：無阻塞或新增技術問題。anchor_points只是參考點而非預設anchor尺寸；并不保证NMS-free；DFL K16的距離刻度上限與候選密度／assignment代價分開。圖每點與GT、pixel/cell距離、r負值正确；V-01縮圖軸字小待新版。

| 檢查檔案 | SHA-256 |
| --- | --- |
| `docs/lessons/12-anchor-free.md` | `6874048fb9cb6f72b024a9c10af98afbd0c7d4a8fce63391a1e81efde25321b9` |
| `lesson_cases/12-anchor-free.py` | `db623f3fedbbb57779640ef0b14cd897309ba728dc43bd96cc1f0fdb2b88b08c` |
| `artifacts/checks/curriculum/12-anchor-free.json` | `80aec56742ce823f1cc990cbb6f3927e85fd85cdaec377b1e1835dec08eb4ad0` |
| `docs/assets/diagrams/12-anchor-free.svg` | `8b7a5abd2f0c2d2bc65a3a35388b14578f585346a85ddf617e174dd4a342ac80` |

### 12-decoupled-head

方法與範圍：完整讀 `DecoupledHead/main`、正文、完整 JSON；追三次清梯度反傳、None跨支路與shared梯度加總，隔離主例和分類權重×2练习。核YOLOX v2 §2.1／Figure2，固定Ultralytics441... head.py Detect cv2/cv3、tasks.py parse_model legacy以及yolov8.yaml C2f配置，未只看Detect類別default legacy=False就誤判v8走DWConv新版。

數值與機制：共享backbone3→8；box/class各3×3→ReLU→1×1，shape `[2,4,4,4]`／`[2,2,4,4]`。独立sigmoid类别，无obj；所有位置同class0且距離1.5是人工全正例，不是圖片中的真实物件、没有assignment。box-only与class-only跨支路.grad None、shared总grad=gbox+gclass；cos−.0073≈90.4°只近乎正交，没说大冲突／解耦解决全部共享冲突。参数224+2×584+36+18=1446；两branch1222、coupled1×1=54、coupled含一层3×3=638，容量未对齐不能直接把收益归因分工。只把输出层拆开仍是等价权重行分组，无新增任务专用中间层。

练习Lbox+2Lcls三处改动确实pass，第二次单独cls不乘2，否则reference gradient会先被加权。set_to_none=False与detach错误的具体失败点符合源码。固定v8实际为legacy两层3×3＋1×1／DFL box64C，toy每branch只有一层3×3且直接四连续距離，差异已列。

結論：無阻塞或新增技術問題。本頁沒有SVG，ASCII路徑、shape與成本表已核；實際HTML表格与手机布局未驗。

| 檢查檔案 | SHA-256 |
| --- | --- |
| `docs/lessons/12-decoupled-head.md` | `926dfafe157605f5cb41fd7a84db1ff8c13bf66ad02b28b6359f8c5903786570` |
| `lesson_cases/12-decoupled-head.py` | `38f6e64626db6a0324a23bc3d05e3c0d892df9e7a54423f98ff343abd6b5dbfc` |
| `artifacts/checks/curriculum/12-decoupled-head.json` | `2428eb47a55f5a3904f345907c97d3d7fcdd7231c2483a3e3542f8ec75d2634b` |

### 12-assignment

方法與範圍：完整讀 `box_iou/assign/main`、正文新owner→兩類target→toy前景桥表、完整 JSON与圖；逐點核框內資格、score×IoU²、topk、IoU解衝突、empty与BCE。隔離執行主例及score改.1／metric解衝突练习。读TOOD v1 §3.2／Eqs9–12（t=s^αu^β、实例质量normalize）；固定Ultralytics441... tal.py get_pos_mask/get_box_metrics/select_highest_overlaps/get_targets与normalize、loss.py v8DetectionLoss中的detach、α.5/β6/topk10、BCE正负与框lossfg。

數值與機制：A class0/B class1；p1對A IoU2/3、對B3/7，品質 .3111/.1469；p2與A框雖IoU.1875但reference point在A外所以遮0，p3虽B score .9但outside/IoU0。k2双方有p1、比IoU歸A，owner `[0,0,1,-1]`；k1与scorealone对照揭示排序才有机会改变入选。移p1框到B得A IoU.25／B1，owner `[0,1,1,-1]`，总positive仍3但每GT数变；.1score而保持IoU冲突owner不变，改metric冲突owner变，练习同步断言真实pass。

新桥表闭合：owner是GT编号，A/B class0/1是假設；对照的兩類hardtargets为 `[1,0],[1,0],[0,1],[0,0]`，不是owner值直接填class。程式实际只训练另建foreground4logits、targets `[1,1,1,0]`；hand scores[G,P]不是这四logits也没训练，独立BCE mean4初grad `[-.125,-.125,-.125,.125]`、lr1后logits `[.125,.125,.125,-.125]`。正文清楚说官方无objectness并在同head未detach输出计算loss、quality缩放target而非这张hard1表，没有冒称衔接12.2图片模型。官方当前版本CIoU clamp0、跨stride合并、STAL边长floor16与toy strictinside＋metric>0不同，限定保留。

問題 E-01（low，已關閉）：旧选读把官方冲突限成『所有框内eligible GT』；实际对完整overlaps表max，外部为0。抽出固定method原样CPU输入mask_pos `[0,1,1]`／overlaps全0，返回owner0，可转给原不eligible GT（tie）；这不影响本文两GT正值例。root已改成全GT表max、資格外0、非限topk、全零同分依fixed max；实际读回line132并核源代码，closure成立，未自己改正文。

結論：無剩餘阻塞或數值問題；圖兩GT／四点／预测box是两组独立材料，未把predicted box中心当referencepoint。V-01手機縮圖字小待新版。

| 檢查檔案 | SHA-256 |
| --- | --- |
| `docs/lessons/12-assignment.md` | `819308138126eef1825a41685d19a79d75ca6af50a30fd9c5747e539344731a5` |
| `lesson_cases/12-assignment.py` | `c5a878e08249a513b5a5f5432615f714522f5fbaec92c4484e05bfb75b9249bb` |
| `artifacts/checks/curriculum/12-assignment.json` | `2918296eab727ad6f145905e0520f9c0151b72879ed277a5e920f44737689dd6` |
| `docs/assets/diagrams/12-assignment.svg` | `1300badfe1205713e497f74847c41b09654e89a66ce11541c3466cd9f422dff7` |

### 12-dfl

方法與範圍：完整讀 `dfl/expected_distance/main`、正文、完整 JSON；逐条核相邻bin权重、initialCE gradient、softmax期望、entropy下限与两个同expectation分布，隔離跑主例及2.6／3.2练习，另CPU计算两条边 reduction。原典读GFL v1 §3.2／Eq5–6／Figure5與统一GFL Eq7（DFL仅CE部无额外focal调制）；固定Ultralytics441... block.py DFL、head.py Detect.reg_max/no、loss.py DFLoss/BboxLoss与tal.py bbox2dist。核mmdetection v3.3.0 commit44ebd17... gfl_head.py Integral23–55／输出4*(reg_max+1)。

數值與機制：1.25cell、stride8=10pixel、K4，targetprob `[0,.75,.25,0]`，uniformE1.5、DFLln4=1.386294、gradient `[.25,-.5,0,.25]`；bin2初梯度0不代表永远不学。400SGD.5输出probs `.0025/.7475/.2474/.0025`、E≈1.25且仍DFL非零。entropy下限 `−.75ln.75−.25ln.25=.562335` 正确；概率单纯形上的理想分布与finite logits只趋近的差别保留。target分布不是物件定位的不确定性已校准指标；不同分布 `[0,.75,.25,0]`／`[.375,0,.625,0]` E同1.25而后者bin1=0导致DFL无限，expectation本身不够验DFL。

范围／成本：toy断言0<=d<K−1避免right索引溢出，不默默clip；fixed官方两处夹到K−1−.01，K16=14.99不是16个最大值16。mmdet reg_max16含0…16共17bin，命名差异真实。100候选C2直接4+C=600raw，而DFLK16为6600raw；这是元素成本不是完整延迟/显存benchmark。两边CE先逐边加权再mean的 `.841478`，错先mean再乘得 `.982304`，CPU相符。

练习2.6的bin2/.4、bin3/.6与梯度 `.25/.25/−.15/−.35`、四行修改实际pass（期望2.5899在.02容差内）；3.2在range assert失败，增加bins或换适当尺度都须重新满足契约。官方还用IoU与quality weighted/reg前景mask；本文单边logits实验没有声称完整v8loss／AP效果。

結論：無阻塞或新增技術問題。本頁無SVG，bin與softmax軸／两边加权表及數字已核；HTML手機表格与展开呈现未驗。

| 檢查檔案 | SHA-256 |
| --- | --- |
| `docs/lessons/12-dfl.md` | `670eb809234ae66e1d1590142fd8ec0cf8b15aa1045c0720ad40a8627420a256` |
| `lesson_cases/12-dfl.py` | `2f3ec8899cb25eb7df5d1118f57a715de935de9cc7569152e542d4563fa1e1c1` |
| `artifacts/checks/curriculum/12-dfl.json` | `eddf4637869043c40f89deeffeacf1761b959ed0058fb154ecdbd00d89dd7233` |

## E-01 的關閉與視覺觀察

E-01 — low／已關閉，12-assignment 選讀官方衝突範圍。固定441... `get_box_metrics`先建全0 overlaps、只在eligible位置寫CIoU；`select_highest_overlaps` line364用`overlaps.max(1)`而未再mask成eligible。當eligible兩GT的CIoU皆clamp0，原method可在全零tie取GT0，即使GT0不是eligible。獨立抽出原method、原body，`mask_pos=[0,1,1]`／`overlaps=[0,0,0]`／topk=topk2=1得到owner0、resolved `[1,0,0]`。主例兩GT正值與bridging假設不受影響。root把「框內GT」描述改成「全GT重疊度表取最大，資格外0，非限topk，零值同分依固定max」。已實際重讀修改句而非只採「已修」訊息，語義符合固定碼，closure成立。

V-01 — 非阻塞觀察／待獨立補核6張新版圖。原SVG自動縮成358px內容寬時，09-anchors／11-csp／12-assignment最小字約8.45px，09-anchor-clustering／12-anchor-free約6.96px，11-augmentation約7.96px；實際目視可識別幾何但小標籤吃力。這是上述set_content預覽條件，實際站頁CSS／放大操作未驗，不能將數字當網站字級測試結果。已請主代理查mobile呈現並更新6張專用SVG。10章四張新增直式圖最低約12.79–13.77px，11-fusion／bilinear／crop／IoU四圖約16.11–17.90px；主線圖清楚。生成10-learning雙欄圖字小（縮圖約4.97px），主文已有獨立直式實測圖與數值表，未要求改生成圖／證據。

## 固定原典與原碼來源

論文位置以原PDF章、式、圖為準：YOLOv2 `1612.08242v1` §2 Dimension Clusters／Direct location prediction／Multi-Scale Training；YOLOv3 `1804.02767v1` §§2.1–2.4；CSP `1911.11929v1` §3与Figures2–4；FPN `1612.03144v2` §3；PANet `1803.01534v2` §3.1／Figure2；GIoU `1902.09630v2` §§3–4／Algorithm2；DIoU/CIoU `1911.08287v1` Eqs6–12／Figure2；TOOD `2108.07755v1` §3.2／Eqs9–12；GFL `2006.04388v1` §3.2／Eqs5–7／Figure5；FCOS `1904.01355v1` §3.1／Eq1；YOLOX `2107.08430v2` §2.1／Figure2；YOLOv4 `2004.10934v1` §§2.2、3.3与Figure6。

原碼版本：pjreddie/darknet `f6afaabcdf85f77e7aff2ec55c020c0e297c77f9`；Ultralytics YOLOv5 v6.0 `956be8e642b5c10af4a1533e09084ca32ff4f21f`；Ultralytics當前固定版本 `441632cdfd19e22e60a4b1b1999d46326ca51ec4`（含STAL，不能無限定當作歷史2023v8）；mmdetection v3.3.0 `44ebd17b145c2372c4b700bfb9cb20dbd28ab64a`；用来核v4配置的AlexeyAB/darknet固定 `59596d7880f6504768df41d6daa586f5cb2b932f`（只核當前保存的cfg，不宣稱2020原始訓練完整復現）。

以下只列成功下載／獨立hash驗證的原典。cache本地路徑均在`/tmp/clear-tutorial-evolution-tech/sources/`；正文引用的URL／commit／symbol位置与hash可独立追溯。

| 原典檔名 | 固定下載來源 | SHA-256 |
| --- | --- | --- |
| `yolov3.pdf` | https://arxiv.org/pdf/1804.02767v1 | `37049049b5e06f67c6cd22b72f7b9352914b0eb3a03e99abf54594c1005a8468` |
| `csp.pdf` | https://arxiv.org/pdf/1911.11929v1 | `34911bf18e60291c9d5769303b7754902f4064019fe9d49f1dcdd98b934fb0e4` |
| `fpn.pdf` | https://arxiv.org/pdf/1612.03144v2 | `3a33e7ad38c17037624defde57f380f49dee918c1c14abd5dbcad032bae91428` |
| `pan.pdf` | https://arxiv.org/pdf/1803.01534v2 | `c77f080f627789c1b3a4103b05efb0f08a1e346eccef5f3572f834274329a2f0` |
| `giou.pdf` | https://arxiv.org/pdf/1902.09630v2 | `56fceedbbe952fd132eb75a6e6bc637028c365fa5a8c6b584e4020ddf01f291e` |
| `diou.pdf` | https://arxiv.org/pdf/1911.08287v1 | `ba108d72fb9782fd2fc96253fb901fb488c257c07642ae1e6ea36d325d0d3429` |
| `tood.pdf` | https://arxiv.org/pdf/2108.07755v1 | `1c94cc4f94940e7974cbce0e92c75bf925f83396e9ea591e130eed03918b9c01` |
| `gfl.pdf` | https://arxiv.org/pdf/2006.04388v1 | `04a56cf0bd1c144ff940ad12a4134b75090843994bbfc22b84a04c1e5ed68a54` |
| `fcos.pdf` | https://arxiv.org/pdf/1904.01355v1 | `a6a6a23b919e912f4f809c0462925a8fd0d4cd3750a40a307f252408e6d49fdc` |
| `yolox.pdf` | https://arxiv.org/pdf/2107.08430v2 | `8b3d8689b65887885b5f6f1061ff9c278a59bcdfa2063ca0a0484d00040e8889` |
| `darknet-region.c` | https://raw.githubusercontent.com/pjreddie/darknet/f6afaabcdf85f77e7aff2ec55c020c0e297c77f9/src/region_layer.c | `9cdcf276a19510a660b41a188eab615ea4f24dd45ef428f6ef7fada87b359f4f` |
| `darknet-yolo.c` | https://raw.githubusercontent.com/pjreddie/darknet/f6afaabcdf85f77e7aff2ec55c020c0e297c77f9/src/yolo_layer.c | `97968f8e58b7aaba5bcbbdc4070b8a360871d8ab5ae411d1fa8ab87cbbc145c4` |
| `darknet-yolov2-voc.cfg` | https://raw.githubusercontent.com/pjreddie/darknet/f6afaabcdf85f77e7aff2ec55c020c0e297c77f9/cfg/yolov2-voc.cfg | `ecf8b24ca01bdedc5c569300a6e99120ca5e251108eaa973b9cb19e28ed93352` |
| `darknet-yolov3.cfg` | https://raw.githubusercontent.com/pjreddie/darknet/f6afaabcdf85f77e7aff2ec55c020c0e297c77f9/cfg/yolov3.cfg | `22489ea38575dfa36c67a90048e8759576416a79d32dc11e15d2217777b9a953` |
| `darknet-detector.c` | https://raw.githubusercontent.com/pjreddie/darknet/f6afaabcdf85f77e7aff2ec55c020c0e297c77f9/examples/detector.c | `f80353470eac9d94cd2fae44bf4d6b52685bc550fbbb4a87deec576531dd8125` |
| `ultralytics-tasks.py` | https://raw.githubusercontent.com/ultralytics/ultralytics/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/nn/tasks.py | `4f54fd2fd743d364e25124597dbfd74746e1edf62ab7bf85f27251c5334a4880` |
| `ultralytics-yolov8.yaml` | https://raw.githubusercontent.com/ultralytics/ultralytics/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/cfg/models/v8/yolov8.yaml | `405a3c302eb6096103a99af4d82c1d2d8b2bbff469a228c55aac944b2c19672a` |
| `mmdet-gfl-head.py` | https://raw.githubusercontent.com/open-mmlab/mmdetection/v3.3.0/mmdet/models/dense_heads/gfl_head.py | `0cedb6467bccf8cf2368de9647aac6b43d6a65cae75800fb47ab026613f24b89` |
| `yolov2.pdf` | https://arxiv.org/pdf/1612.08242v1 | `6c5e99e00874eeebb58de6a6676c98a6e650b80e4a867a64ea1e7cd748ac738a` |
| `yolov4.pdf` | https://arxiv.org/pdf/2004.10934v1 | `531196e36fc7e5e4190328039882671aba0ec1c6e2518102183eaf926635f502` |
| `yolov5-common.py` | https://raw.githubusercontent.com/ultralytics/yolov5/956be8e642b5c10af4a1533e09084ca32ff4f21f/models/common.py | `71d6e15aee84e8cde2a0c64920f7421fdc6beaff4cd26f64dd82ec358b5f2fb0` |
| `yolov5s.yaml` | https://raw.githubusercontent.com/ultralytics/yolov5/956be8e642b5c10af4a1533e09084ca32ff4f21f/models/yolov5s.yaml | `d17de90c1a69a54d95b56d710d3688557b1ca7817558931630c6c7052c47f258` |
| `yolov5-augmentations.py` | https://raw.githubusercontent.com/ultralytics/yolov5/956be8e642b5c10af4a1533e09084ca32ff4f21f/utils/augmentations.py | `8f4a615bd6ba9fbd65d591fdb9209945a45147c15f88c5736dc382685b83def7` |
| `yolov5-datasets.py` | https://raw.githubusercontent.com/ultralytics/yolov5/956be8e642b5c10af4a1533e09084ca32ff4f21f/utils/datasets.py | `e9a66c0c6eac420277094a224f12a100dca63c89fad9ee93aa641df45cb18d67` |
| `yolov5-hyp.yaml` | https://raw.githubusercontent.com/ultralytics/yolov5/956be8e642b5c10af4a1533e09084ca32ff4f21f/data/hyps/hyp.scratch.yaml | `be4307dc0cfee94f4b4613a83e7b828db7a89a5c2cd3885d4fb1108a5c32aed1` |
| `yolov5-loss.py` | https://raw.githubusercontent.com/ultralytics/yolov5/956be8e642b5c10af4a1533e09084ca32ff4f21f/utils/loss.py | `610b4e3ef4fcff0f2b7eaaf98a5532076397a30d399fa05e021fc94b42a67843` |
| `yolov5-metrics.py` | https://raw.githubusercontent.com/ultralytics/yolov5/956be8e642b5c10af4a1533e09084ca32ff4f21f/utils/metrics.py | `137f499b43119bfeb3470de576460b8669b3e93ab74380a1e0d3604cba9b306d` |
| `ultralytics-block.py` | https://raw.githubusercontent.com/ultralytics/ultralytics/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/nn/modules/block.py | `3be57fed557f10f76701f66cf91788bc6ec9ce1683f1a34b407aba65847c744f` |
| `ultralytics-head.py` | https://raw.githubusercontent.com/ultralytics/ultralytics/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/nn/modules/head.py | `6b118882874ee2550b2c5c819a712774e1c729faf9fdc5ae6f068996c9893d02` |
| `ultralytics-loss.py` | https://raw.githubusercontent.com/ultralytics/ultralytics/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/utils/loss.py | `8eb4e089b7c36cef13bb872f9aa37e346da5c022bcfd979e11e7a57a0666b106` |
| `ultralytics-tal.py` | https://raw.githubusercontent.com/ultralytics/ultralytics/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/utils/tal.py | `c69ec67990777a4d13df219ef9b254085b2df283fbe5fd283245b76d6b461b3c` |
| `yolov4.cfg` | https://raw.githubusercontent.com/AlexeyAB/darknet/59596d7880f6504768df41d6daa586f5cb2b932f/cfg/yolov4.cfg | `a6d0f8e5c62cc8378384f75a8159b95fa2964d4162e33351b00ac82e0fc46a34` |
| `ultralytics-metrics.py` | https://raw.githubusercontent.com/ultralytics/ultralytics/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/utils/metrics.py | `a41b9c09c19f5c41990b00d323efc3ef657d391ad0a1a7eb1774b5556d2fd37e` |

## Repo imported dependency hashes

所有宣告依賴逐檔驗證，以下列出10章主例／40步腳本用到的額外來源（各lesson_case已在逐頁hash表列出）。

| Repo檔案 | SHA-256 |
| --- | --- |
| `miniyolo/__init__.py` | `785b058b2b011124243b06ee59df8e59dfa75b77f050b897479e5be636763fc9` |
| `miniyolo/data.py` | `cccad00e2c4f96eb6567eafc9e12248379c6b715fc1790d75518a253baa6181d` |
| `miniyolo/figures.py` | `155222c5bb0d6942bcc231d5c32c6f1a4db662293c8cf909560f1f9b75a3ea15` |
| `miniyolo/geometry.py` | `6a6b57d3493888e99dae4a012dab78127b8b543a0e01d8107963a3d6e63d8483` |
| `miniyolo/inference.py` | `995ac8f942d0c1e43d94f2efb3b7adcb91a9feb6b9bab923615209f0c190c313` |
| `miniyolo/losses.py` | `81fa9331c9a2aebe2e9c6c453a5313e64566bb20c3fe77ca45ec9df53424d4e4` |
| `miniyolo/metrics.py` | `53ce982e37cbd96c784f75c7d30faf99d52f79ab83ca7b8114eb21b4327330e0` |
| `miniyolo/models.py` | `49d029ea4ba2650ce8933cf97e3d25dc7aff2ca4e972eec19cdda17b0f4900e6` |
| `miniyolo/provenance.py` | `21769813c36b45f513c9f0d6125194f3baa5ae265278ffd44a796d298d124ed3` |
| `miniyolo/targets.py` | `2c8e32f2845b3bf970c77304c5cca0f999083d36a4d5af2f75f14aa89b823f16` |
| `scripts/run_learning_extensions.py` | `b84e3317f95374df87d759e562db9b100ee14628b9cf5bb111238cd4a2e8e6cf` |


## 六張新版 SVG 的獨立核回（V-01 已關閉）

核回時間：2026-10-05T10:53:05.778959+00:00。這是初稿 V-01 的後續 closure；前面的舊圖指紋與「待新版」保留作歷史，**截至本次，V-01 已關閉，沒有 blocking issue**。沒有採作者自查或作者 manifest 作通過證據。

方法：直接讀六個當前 SVG XML、各頁相鄰圖說及相應實際 lesson case；在隔離預覽做 390/1000px 的 text BBox 檢查，全部 text 留在 viewBox 內，六圖均有 viewBox/title/desc。再用系統 Chromium 訪問 root 的實際 Zensical HTTP 站頁，於 390px viewport 查看六圖。HTTP 回傳 SVG bytes 逐位元等於工作樹圖，manifest 綁完整 SHA、實際圖片尺寸、最小字級與 PNG SHA；沒有把作者截圖當獨立證據。這六頁的 desktop 實際站頁本輪未補核；原圖檔 1000px 預覽與 root 全站檢查另有範圍。

实际六頁手機內容圖寬為 314.86px（09anchors 在清單內）或342.98px（其餘）；最小基本字 22 經 CSS 縮放後約14.43px或15.72px，數字、索引、IoU/參數式可讀。原始 context PNG 保留網站 UI。高圖的 locator screenshot 可能把固定 header 拍到圖中，另存 clean PNG，捕捉時只隱藏站頁 sticky header/tabs 等匹配 UI，不改圖或圖的尺寸；目視核查也含此乾淨補圖。這是六張圖的可讀性與技術內容核圖，不宣稱全站或所有表格/折疊已驗。

| 圖 | 獨立核的數值、幾何與讀者用途 |
|---|---|
| 09-anchors | 圖到像素映射 x=80+5.5x、y=152+5.5y：紅GT [8,12,24,28] 與16×16 anchor相同；8×8 anchor 同中心(16,20)。藍責任格為(1,1)，x=16格線經floor歸右格；1 positive/1 ignore/30 negative、IoU 1/.25與>.2自訂ignore均與case一致。root 把「64×64圖」標籤 baseline 128→110後，本人重新讀屬性及看圖，標籤不再撞上頂部0–3欄索引，幾何未變。 |
| 09-anchor-clustering | 六尺寸點 [8,8]/[9,8]/[8,9] 與 [32,16]/[30,16]/[32,18] 非圖片位置；①的同中心IoU .25/.5 正確，②–④的群組與均值 [25/3,25/3]、[94/3,50/3]、初始化→紫菱形箭頭正確。只保留本例平均更新，沒有從圖推廣全域最優。正文①／②～④新引用亦實讀，避免新直式圖沿用左右方向。 |
| 11-csp | 8channel full的兩3×3與1×1，對比4channel旁路＋4channel兩3×3、concat成8再1×1；H/W不變，無誤畫add。含bias、無BN的手算1240=2(8×8×9+8)+72，368=2(4×4×9+4)+72，與actual類別一致。方框不是品質證明，圖例保留這個限定。 |
| 11-augmentation | 所有畫素同4×比例。原框[8,12,24,28] flip成[40,12,56,28]、中心16→48且責任格(1,1)→(3,1)；crop窗口[16,8,48,40]由①原圖出發，非②後續，切成32×32，框[0,4,8,20]，可見8×16/16×16=.5。紅畫素半開範圍與被移除灰區一致。 |
| 12-anchor-free | 實際rect/circle按 x=64+5.5x、y=184+5.5y 回算：GT[12,16,40,36]、候選(28,28)、中心(26,26)、外點(44,28)。ltrb=16/12/12/8畫素，除stride8=2/1.5/1.5/1格；外點r=-4畫素=-.5格；候選在第(3,3)格中心、框外点第(5,3)，沒有錯把候選說成GT中心。 |
| 12-assignment | 同一x軸上的 A[0,0,20,16]／B[12,0,32,16]、重疊12–20、參考點x8/16/24/40（y8）和四pred [0,0,10,16]/[4,0,24,16]/[14,0,32,16]/[40,0,50,10] 的實際線段/矩形位置一致。圖明說只畫x、長條高度不代表框高，GT資格由reference point、IoU由prediction box，沒有把toy順序圖冒作官方TAL。 |

六圖內容沒有改變公開數值或 lesson case，因此没有另做訓練。再核全部11個 current case SHA、每個公開 dependency SHA，以及先前獨立 CPU stdout 對 current 公布 stdout，均一致；若 evidence 全檔 SHA因 metadata更新而變，以下仍列此刻完整SHA。上文固定論文/原碼位置及簡化限定仍適用，這次没有新增原典復現聲稱。

### 核回六圖與當前頁面指紋

正文SHA定義為全文中 curriculum-evidence:start 註記之前的原始 bytes；全檔SHA仍含現行v0.5 Colab/footer等。它們是此次核回時的指紋，不覆寫初讀時hash。

| 頁 | 全文 SHA-256 | 正文 SHA-256 | 新圖 SHA-256 |
|---|---|---|---|
| `09-anchors` | `7604bfaecb11e818a45d2a745c8450801e73c9541b351540ffc7f479080b3cea` | `76cf654880cee90c125012d84f7255b1a0408f090756ff8d2e4a74d3dcf696ac` | `93c73f27386a64d9282413f080db87a2891c249115cb165622cb78c13ba3f4e5` |
| `09-anchor-clustering` | `3fc8b91b87bfde4827f7a90949262fdb657560680215fc1b5f39223caba3824e` | `c2a02fb708bbf01b681257a760e8cd4edb1fe0202de840e8980eb51ad0e197f6` | `58a4bd6ea6a5a636e83f03bdbe276ec94d786195c4344d9e88ee1520dc348b13` |
| `11-csp` | `40058c08d8d01a1f9ef4ce11d79ac9e89a6603470fc41ea2c08d25ccba476270` | `38f6774337e87395dde2e4a3006158e1a6ca38fc2228bf735388a22fae133898` | `64377bcc3357725640c000e0ac13ef9582578af3643ab9ade1622be84851c5df` |
| `11-augmentation` | `912b5cec4f0ae0e17b6cbbfe0a24d1b2082f690950770b74c113efa18ade6fe6` | `a2d0350d4563da549896ada2e4fbf509b30574357dd7c928fee7b04099bfd44a` | `46a835957fbdf9f3b3baa9a5299e7205a294e15e9a38d071780d9a5b2c8c4086` |
| `12-anchor-free` | `6874048fb9cb6f72b024a9c10af98afbd0c7d4a8fce63391a1e81efde25321b9` | `4c2502ce8f400c0081081f7db7af88dd84cf47d9a9c48aeb804d6852dd6c2804` | `0f43b6c74f090473582866c4dc56949c8fd55267e4a5b99f2e499f5382774548` |
| `12-assignment` | `819308138126eef1825a41685d19a79d75ca6af50a30fd9c5747e539344731a5` | `01a2bc21d12bb2ccab144f6da5d9213537e65889c0c3777738c6c9abeda8c893` | `4a3817c274012a7838e95b9b0a9d8188259eea8f5a23869b819df21722e700b2` |

### 全11頁當前 source/evidence 指紋核對

| 頁 | 全文 SHA-256 | case SHA-256 | 公開 JSON SHA-256 |
|---|---|---|---|
| `09-anchors` | `7604bfaecb11e818a45d2a745c8450801e73c9541b351540ffc7f479080b3cea` | `7a7e8c431340e33f55fc8afc90136b225d18387940d6677eab9e8aecacd31469` | `2157849a3e3b4d33880cb9c1ded9887f04d4cfe818c5d836d00dd6b6d169399c` |
| `09-anchor-clustering` | `3fc8b91b87bfde4827f7a90949262fdb657560680215fc1b5f39223caba3824e` | `1262a5b3c925d9eb2e9c03cbe70279355a0575d3e0c8d7f28522883e692221a4` | `8df633a0412b0475037f736f91614b58e78e1b0d3d81886e4406275c48531767` |
| `10-multiscale` | `f8c07a9dde22911e407fa18c5b8ecd9c30451f8f91821f990314573b0c9e4578` | `add9aefa8efec7180d06ab255807ac4fa6847c490c8447774191f0a6566ed6a6` | `6ffea3254bce8b21b1aa2eaa0a0165641f7776748474d5e5ee706c2599204688` |
| `11-csp` | `40058c08d8d01a1f9ef4ce11d79ac9e89a6603470fc41ea2c08d25ccba476270` | `9710906769f4d757331a7c99d2ba89d8d45c40d52c76f1d2a32a510e13a4f556` | `d376a3741add2932b78363be8c898c548ba608d0f961408915f26ecbae672abc` |
| `11-fusion` | `e5adc94096402154bae57fcd25306db268043d6dbb532eb5b3523eb4d4a138b7` | `4de6c1260d6f76f55013b4d17d935dbff46f4c1ec21975cbae5b065cd887af63` | `9c6c6a611744a0381a1aa802cdbce3bef9cfde712d45c79d5be7aa87e51ed946` |
| `11-augmentation` | `912b5cec4f0ae0e17b6cbbfe0a24d1b2082f690950770b74c113efa18ade6fe6` | `5665b9ddc7a920ec45041987096e49c931515a3f62c76e5e774c7d8031b1ddde` | `68d166c4cdff4d1f091de32b76e90f001a1844fbfacf743488b85571056e42df` |
| `11-iou-loss` | `e6668a24141a24376dc455cb66cd941f9abc48ad6787007a51e4569e32dff7b4` | `ddce126434321bddf95543a02f21b082602024b3ba0ef6992cf8b2be70647325` | `ce0124fad2b98449d926e1969ed28f9f59bb868460a33b49a87e2e2c6044edab` |
| `12-anchor-free` | `6874048fb9cb6f72b024a9c10af98afbd0c7d4a8fce63391a1e81efde25321b9` | `db623f3fedbbb57779640ef0b14cd897309ba728dc43bd96cc1f0fdb2b88b08c` | `61495eafd3110734d8046d2ca92d5c36b2ff8915db13e530c70ad346796104ca` |
| `12-decoupled-head` | `926dfafe157605f5cb41fd7a84db1ff8c13bf66ad02b28b6359f8c5903786570` | `38f6e64626db6a0324a23bc3d05e3c0d892df9e7a54423f98ff343abd6b5dbfc` | `2428eb47a55f5a3904f345907c97d3d7fcdd7231c2483a3e3542f8ec75d2634b` |
| `12-assignment` | `819308138126eef1825a41685d19a79d75ca6af50a30fd9c5747e539344731a5` | `c5a878e08249a513b5a5f5432615f714522f5fbaec92c4484e05bfb75b9249bb` | `2918296eab727ad6f145905e0520f9c0151b72879ed277a5e920f44737689dd6` |
| `12-dfl` | `670eb809234ae66e1d1590142fd8ec0cf8b15aa1045c0720ad40a8627420a256` | `2f3ec8899cb25eb7df5d1118f57a715de935de9cc7569152e542d4563fa1e1c1` | `eddf4637869043c40f89deeffeacf1761b959ed0058fb154ecdbd00d89dd7233` |

本輪新核圖本地證据：`artifacts/runs/clear-tutorial-16f6910/grid-transition-browser/manifest.json`，SHA `20457bc601c47edda10b451e039c31d4b755b764cdc67fdb5f694c4bfebb4d5a`；其 rows 保留每個 SVG/served SVG/原始 PNG/clean PNG 完整SHA，六圖獨立站頁均390px。預覽幾何/BBox細節在 `/tmp/clear-tutorial-evolution-tech/svg-final/audit.json`、`geometry-checks.json`，不是永久發布證据。未驗範圍仍是其餘頁面最新HTML折疊/表格、全站desktop/mobile與GPU；交root最終全站browser。

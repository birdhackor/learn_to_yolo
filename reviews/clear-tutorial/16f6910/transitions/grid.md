# 第三輪：Grid 到自己的資料的順序銜接複查

審查者：`clear_first_modern`；日期：2026-10-05。七頁均非本審查者撰寫，也未由本審查者做過第 7–8 章第二輪完整技術審查。本輪先完整讀現稿 `06-decode-nms`、`06-evaluation`、`07-data`，依序讀 `07-targets` → `07-loss` → `07-training` → `07-inference` → `07-heldout` → `08-own-images` → `08-own-data`，最後讀 `09-anchors` 開場至第 95 行。再開其他讀者的原始 first-read 紀錄，而非採作者摘要當讀者答案。

這是**非首次盲讀的独立順序複查**，沒有逐段 gate，也不聲稱全新上下文。本審查者先前现代組实际 prerequisites 包含部分 07 與 11–12，曾做 09 開場的非作者技術核查；此前沒有讀過兩個 08 頁全文或其他讀者的 08 原始答案。另曾撰寫 03identity/comparison、04coordinates、05assignment，這些頁不在本輪獨立驗收範圍。root 的任務 brief 提及待查限制與新圖，所以保留此上下文披露，不能把本輪改稱 blind。

每頁當場理解、材料、小變化預測、問題與當時全文/圖 SHA，先存 `/tmp/clear-tutorial-grid-current-read.jsonl`，讀完兩頁 08 後才開相關 raw/issues；原樣保存為 [grid-current-read.jsonl](grid-current-read.jsonl)（11 筆）。這是修正原 grid 讀者接收作者 brief 後再讀 08 的限制的另一位讀者順序證據，不覆寫歷史 raw。

目前正文銜接没有 blocking 卡點。已確認的手機視覺問題 G-V01：07 的原生成曲線/圖板文字在 390px viewport 的 SVG 預覽太小，已回報 root，待新版可讀圖完成後實際核回。08 两張新增窄版 evidence 圖已實看：主線字約 13.77–15.15px，圖說的 score、IoU、TP/FP/FN 可追。實際 Zensical desktop/mobile 全站由 root 檢查，本報告不冒稱已驗。

## 方法、執行與範圍

1. 先讀上述現稿並保存當場理解，才查 `first-read/grid.jsonl`、`applications.jsonl`、`evolution.jsonl`、`modern.jsonl`、`foundations.jsonl`、`reference.jsonl` 中本組相關原始 issues。只查原紀錄，不讀作者的問題摘要作理解證據。
2. 讀實際八個 `lesson_cases/07*` / `08*`、`miniyolo/custom_data.py`、`scripts/detect_image.py`。另實讀 `scripts/run_custom_data_learning.py` 110–255、520–632、645–735 的來源、配對、fresh model、prior-only-report、固定切分、最後一次 test 與固定取圖流程；其餘 miniyolo data/targets/loss/model/geometry/inference/metrics 在前一輪已有本審查者實讀，核指紋與這次公開依賴一致。
3. `/tmp/clear-tutorial-grid-cpu/` 是隔離副本。PyTorch 2.9.1+cpu 下重跑 07data 加七個目標頁小程式，八個皆 exit 0、stderr 空、stdout 與當前公布 JSON **逐字一致**；case 與全部依賴 SHA 一致。未寫回 shared source/evidence、未 GPU。
4. 相稱的變化檢查：去掉 07training `optimizer.step()`，三次輸出完全相同且最後「參數已改變」assert 按預測失敗；08own-data 改 A1 split 為 validation 仍印 source leakage；加第四類並同步 shape assert 後完整小程式通過，head `[2,4,4,9]`。完整結果 `/tmp/clear-tutorial-grid-cpu/checks.json`、`evidence-checks.json`。
5. 核公開 `grid-learning.json` 與兩份 custom-data JSON 的設定、loss、指標、GT/預測、前160步逐值一致，以及 11/15/15 個公布 source 依賴 SHA，全部一致。**没有重訓 160 或1600步**；沒有驗 CPU 續訓、GPU 或照片品質。對這些實驗，只在本輪確認公布數字、固定程式語義與銜接限定。
6. SVG 用 Playwright / 系統 Chromium 在 1000/390px viewport、body margin16、width100% `set_content()`，計算 text BBox、查看截圖；不是站頁 CSS 驗證。實看 main 七種不同 SVG 的 mobile（object-journey、兩07生成圖、07data、08letterbox、兩08窄版），其中兩08新圖另看 desktop。07 原生成圖數值仍可信，但手機字小不等於主圖可讀。

## 逐頁當前理解與閉合
### 07-targets

讀時全文 SHA：`a3816d6f82e0bb9859557d6771ec2978d66304ee4f2a2f6670d8e4514421c205`。報告時全文 SHA：`a3816d6f82e0bb9859557d6771ec2978d66304ee4f2a2f6670d8e4514421c205`；報告時正文（執行紀錄區之前）SHA：`586e9f871b1ab6dc1144fc99522a85592be11338675a1b2e94fe470d11dcfc8b`。

當前任務：把07data兩圖原始GT list轉成B2固定4×4格目標，讓每head值有正確答案與loss mask

材料與例子切換：同上一頁scene紅/藍＋empty；紅center16,20→[0,.25,.25,.25]索引[0,1,1]；藍48,44→[0,.75,.25,.25]索引[0,2,3]；正2/負30

小變化預測：換box[40,2,48,10]，center44,6→gx2/gy0、target[.75,.375,.125,.125]，positive.nonzero[0,0,2]；類別增2→3只7→8outputs，same-cell容量仍1不能解碰撞

實際閉合：原始標註清单→build_targets→固定格子target橋，未要求记targets/target拼写；正mask与objectness答案两用途、负class0与红class0需positive区分；框/cls仅正、obj全32；red endpoint0仅逼近，生成框center内侧明确。B2两框回到上一页材料，图只画red也说明浅蓝为责任格非blue物件

當場觀察：無新卡點；下一頁需明确B从2→1且只取red，有15背景；形状与defaultclass0 vs05的-1差别都闭合。

本頁連結圖的報告時 SHA：

| 圖 | SHA-256 |
|---|---|
| `docs/assets/diagrams/object-journey.svg` | `a80d4e43b9e2dc76d13f76137b0689d6700a2a2ee54faa502607e3401e629d78` |

### 07-loss

讀時全文 SHA：`2b3a4f33c0e12feab2c8f52f97fee44bfdbfcaa8e8329ff44811c4720288ae52`。報告時全文 SHA：`2b3a4f33c0e12feab2c8f52f97fee44bfdbfcaa8e8329ff44811c4720288ae52`；報告時正文（執行紀錄區之前）SHA：`34c549ea9fe70c0a36d42d47d118b30779b68a232f8dd45659509e11067b882f`。

當前任務：用人工零logits把目標→三項mean loss→哪些logit收到何方向梯度手算，保護全空batch

材料與例子切換：從上一頁B2/紅藍+空圖改成明標B1、僅紅框唯一正格；prediction[1,4,4,7]全0無CNN；target[0,.25,.25,.25]class0

小變化預測：加空圖B2：三项loss不變且box/class梯度不變，obj每logit±1/64；複製red圖B2：mean值不變所有每logit梯度减半，共享相同CNN权重總grad不變。class兩logits共同+10 CE和gradient不變

實際閉合：box=.109375、obj/cls=ln2、total=1.933169；obj±.03125与cls[-.5,.5]不同分母清楚；負框/cls零；空batch pos.any else pred*0.sum仍在圖，loss有限0+.693+0。沒有真的optimizer更新，下一頁才CNN。15背景output-gradient量合计.46875不是CNN权重grad，需要各输出对权重变化率乘后带方向相加，新修句正确

當場觀察：没有未解技术卡点。全0logits是验证fixture并非07training模型初始实际输出；进入training后应重新区分材料/输出来源。公式多但推导可折叠，主线手算已有小数与梯度可闭合。

本頁連結圖的報告時 SHA：

| 圖 | SHA-256 |
|---|---|
| 無專用 SVG；表格/數字例為主 | — |

### 07-training

讀時全文 SHA：`8fa15d5a6bfd8917f52e947a95218e1d07b8e476b8234b5a294ffbaf694db72d`。報告時全文 SHA：`8fa15d5a6bfd8917f52e947a95218e1d07b8e476b8234b5a294ffbaf694db72d`；報告時正文（執行紀錄區之前）SHA：`54f1d743439d7c134654b64279e44cebe471ba3781f24b21f519590e392b73ec`。

當前任務：把已核過的資料、格子目標、loss 接到真實 CNN 與 Adam；三步只證明訓練迴圈會更新，再看另一個 160 步的小資料實驗。

材料與例子切換：三步改用 seed 7 的四張生成圖，每張恰好一框，共四個正格與六十個背景格；不是前頁的全零 logits，也不是固定紅藍圖。模型 wh/objectness bias 為 -1.8/-2。160 步另重設模型，訓練 32 張、驗證 16 張、測試 16 張，含空圖；batch 8，三個 seed、閾值與 160 步預先固定。

小變化預測：拿掉 optimizer.step() 時 shape、有限 loss 與梯度檢查仍能過，三次列印保持相同，最後參數已改變的 assert 會失敗。三步把候選門檻從 .25 降到 .05 可能顯示框，但不能由此推出定位正確。

實際閉合：現稿清楚交代 3 步、四圖過擬合建議與正式 160 步是不同操作。列印的是該步更新前 forward；零梯度清除不等於重設參數。三步 pos/neg objectness 約 .12，4/64 正格使常數最優值 .0625，共享 bias 初期下降有可追的原因。160 步 mAP、precision、recall 對全 16 張計算；高分 .980 但 IoU .47 的框仍是 FP，且 .05 候選門檻與先前 .25 範例明示切換。CPU 1.18 秒只指訓練迴圈；資料簡化與單 seed 限定保留。

當場未記正文問題；後續核原始 ID 的處理列於下表，不以這句取代逐項閉合。

本頁連結圖的報告時 SHA：

| 圖 | SHA-256 |
|---|---|
| `docs/assets/diagrams/grid-learning-curve.svg` | `818baea73d025d34b16eb67e16947d46b7615f49b6d4618f3c3f0567c120682d` |
| `docs/assets/diagrams/grid-learning-predictions.svg` | `8eb7ba62adfa0df235a3530982ed6c73cdec7ced2ae47fcceabb1eac8b2b0bd4` |

### 07-inference

讀時全文 SHA：`79ae6881d2ff9144a905d7f559679e416c2bdf7226d8e4212a816df23dbd1beb`。報告時全文 SHA：`37e293a9a25016581b92fc216eb3d868b1c80c828c074ee6593a61006b289bfc`；報告時正文（執行紀錄區之前）SHA：`f30cd709e2d14ebe952c4d62018797fbb131ff29872262c8b007c51829d3b009`。

當前任務：把真實模型輸出接到共用 decode_grid，先確認格式，再用已知答案的人工 logits 核對數字，不把人工框當學習證據。

材料與例子切換：64×64、4×4×7 head；返回固定紅藍場景，batch 三張為紅藍、只有紅、全黑。未訓練模型不是前頁三步模型；fixture 只借 raw 的形狀，不使用畫素。sigmoid clamp 對紅框的 x offset=0 造成 .0016 pixel 右移。

小變化預測：另建 obj=.8、class=.9 的 score=.72 fixture；score 門檻 .70 保留，.75 刪除，再調 NMS 無法救回已刪候選。NMS IoU=1 不刪 IoU=.998 的同類框，改 .5 剩高分一個。

實際閉合：xy 以格寬 16 為尺，wh 以輸入 64 為尺，接回 07-targets 一致。未訓練模型 scores 約 .06 因 .25 門檻全空不代表 decoder 壞了。紅藍圖與負責格圖各自用途明示，責任格淺藍不當藍物件。推論每圖的 [N,4] / [N] / [N] 契約包含空與 batch=1。NMS 預測對預測與 heldout 預測對 GT 的 .5 門檻分開解釋。160 步圖的 .05 評估門檻與本頁 .25 顯示門檻明示差異；非方形原圖需先 undo_letterbox，第 8 章再接。

當場 API 疑問已做最小查證並由 root 修句、實際重讀關閉，見 G-T01；非推論流程卡點。

本頁連結圖的報告時 SHA：

| 圖 | SHA-256 |
|---|---|
| `docs/assets/diagrams/07-data.svg` | `7c45416667c1e130c9ef4433a600435edd4f2f8de0c20869487e3f7e0429c1ab` |
| `docs/assets/diagrams/object-journey.svg` | `a80d4e43b9e2dc76d13f76137b0689d6700a2a2ee54faa502607e3401e629d78` |

### 07-heldout

讀時全文 SHA：`e107fe133e6d000025fe0dfb22ebb15a3cfcde4da7bf6c6715d912a605a93445`。報告時全文 SHA：`e107fe133e6d000025fe0dfb22ebb15a3cfcde4da7bf6c6715d912a605a93445`；報告時正文（執行紀錄區之前）SHA：`743c70acc651ee8f4d6fd3f996f8cabc3eeec076060bfa51a14880553849aa2a`。

當前任務：分清人工評估器答案、三步真模型 held-out 管線冒煙測試與同一次 160 步學習證據，知道評估協議和資料切分要保存。

材料與例子切換：人工例子另有兩張圖、各一個紅類 GT，位置借用紅／藍固定框但第二個仍標紅；A 的三個預測直接給評估器，不做 NMS，B 漏掉。三步 train/heldout seed=7/901 各四圖每圖一物件、候選 .01；160 步仍是 32/16/16、seed 7/700/7000、候選 .05，與訓練頁同一次實驗。

小變化預測：刪掉排在 TP 後面的 .8/.7 FP，AP 維持 .5，precision 1、recall .5；將背景 FP 提至 .95 排到 TP 前面，AP 降為 .25。測試 19 GT 多漏一個使 recall 約降 .053。

實際閉合：人工 AP=.5 不是末 precision 1/3 或 P×R=1/6。原始 GT list 交 evaluate_ap，不是固定格子 target。三個門檻表清楚區分 score、預測對預測 NMS、預測對 GT matching。四張 held-out 只核管線，160 步另有 validation/test，test 最後一次且未回頭調設定。與前頁 NMS / 顯示閾值切換交代完整。圖板 4 張不等於 mAP 全 16 張，16 個預測的 precision 分母不是圖片數；高分 .980 / IoU .47 為 FP 和 FN。seed 獨立只能支持同生成规则，不外推照片，真影片須按來源分組。

當場未記正文問題；後續核原始 ID 的處理列於下表，不以這句取代逐項閉合。

本頁連結圖的報告時 SHA：

| 圖 | SHA-256 |
|---|---|
| `docs/assets/diagrams/grid-learning-curve.svg` | `818baea73d025d34b16eb67e16947d46b7615f49b6d4618f3c3f0567c120682d` |
| `docs/assets/diagrams/grid-learning-predictions.svg` | `8eb7ba62adfa0df235a3530982ed6c73cdec7ced2ae47fcceabb1eac8b2b0bd4` |

### 08-own-images

讀時全文 SHA：`332da2c487b9e3591b61dbb02fc182bbaa1f4afc20461e2c05da332d7f822006`。報告時全文 SHA：`332da2c487b9e3591b61dbb02fc182bbaa1f4afc20461e2c05da332d7f822006`；報告時正文（執行紀錄區之前）SHA：`1566e446a33c6bd664ca3b4fa095dc1852bb54010d112132520a1050a8d32184`。

當前任務：不增加類別，先把非方形圖片讀取、letterbox、載入同架構 checkpoint、decode、還原與輸出接通；照片能輸入不代表紅藍模型會認汽車。

材料與例子切換：新建與第 7 章同架構的兩類 GridDetector，CPU 更新三步再保存並載回；另畫 120×80 黑底紅框 [20,10,60,30] PNG。這不是第 7 章 160 步 checkpoint；另提供 CLI 操作才產生該 160 步模型。人工已知框只驗座標，exact logits 驗載入。

小變化預測：整張原框 [0,0,120,80] letterbox 為 [0,10,64,53]；還原輸入 [32,32,48,48] 得約 [60,40.93,90,70.70]。將兩類名稱換成 car/person 通過 shape/config 檢查，但只改標籤，沒有學新類。CLI .25 改 .05 會讓初始化約 .06 候選出現，不能拿框數判載入成功。

實際閉合：承接 07-inference 像素輸出但新增原圖空間：HWC→CHW→letterbox→batch→model→decode→undo。size 欄位 H/W 和 scale/pad 欄位 x/y 順序清楚；sy=43/80 與理想 scale 64/120 的差異有具體錯誤 10.08/30.23 與原圖下緣 80.6，接回 04-coordinates 的真正 resize 像素邊界目的。每圖 meta、GT 超界不遮掩、padding-only 預測還原後面積零刪除可追。評估 .05 和 CLI 顯示 .25 明示，不混為 AP 改善。state_dict/config/class order、只接受本書權重、Colab 路徑和輸出檔都有具體操作。

當場觀察：checkpoint 與 CLI 細節在主文占比高，首次到第 8 章需要較長閱讀；但各段用途與檢查對象可以分辨，未形成必須猜答案的卡點。

本頁連結圖的報告時 SHA：

| 圖 | SHA-256 |
|---|---|
| `docs/assets/diagrams/08-own-images.svg` | `237df7f604a77f6818a697ae2cff0ad2b39cf28d6e3722acad6d22dbcf32e1e5` |

### 08-own-data

讀時全文 SHA：`51fd720683e2e36df9ec5b85618caedd6a2f75c4c29e70aa23fbd5c5c5d2f671`。報告時全文 SHA：`51fd720683e2e36df9ec5b85618caedd6a2f75c4c29e70aa23fbd5c5c5d2f671`；報告時正文（執行紀錄區之前）SHA：`dfb2869d43e3b5c685d48003b39af26bf1c7a337b852e761fe1f7d10ab3d9f87`。

當前任務：真的增加黃矩形 class 2，以自訂原圖 pixel xyxy JSON、整組 source_id 切分、同步 letterbox、重建三類 model/target/optimizer 來訓練；區分格式管線的一步 fixture 與完整學習結果。

材料與例子切換：主例六張人工 PNG：三來源各一黃一空，背景與位置不同，各 split 兩圖；只有 train 一步更新，[2,4,4,8]。完整實驗另外 48 張，24/12/12，三類各有 GT、每 split 三空、每四圖一來源，共 12 來源；64 letterbox、batch 8、Adam .01、width 8、1600 步，model/train seed 7、validation 700、新 test 7001。

小變化預測：同來源 A1 從 train 改 validation 仍被拒；新增第四類使 head 軸為 9、原 shape assert 必須同步，但沒有綠 GT 就不能學綠。用評估候選 .1 畫圖比 CLI 預設 .25 多框，紅圖黃類錯誤框 .16 在 .25 顯示會消失，AP 並不因此改善。只填框結構合法但未貼齊畫素會通過驗證，需疊圖目視。

實際閉合：raw GT 經 Dataset 同步 letterbox、collate 後才 build_targets，接回 07 的原始標註→固定格子答案。bool labels、空 shape、尺寸與範圍是程式能查的；漏標錯位仍需自己看。source_id/path 檢查不讀畫素，另比較完全相同 RGB/PNG，二者能力界線說清。160 步末 loss .16977、mAP .0068 與中途波動有區分；1600 是同初始化重新訓練，prior 只讀報告核條件，不載 160 權重/Adam。train 1.0，validation .296296、新 test .777778，後兩者各九 GT，非穩定泛化排名。曲線每批 loss 起點約 1.56 與整個 train 表格 1.55017 不同分母；四張圖固定選類別第一張與空圖，不以效果挑。圖 score/IoU/TP/FP/FN 可按前文配對規則理解。最後保持 val/test 同格 GT，只拒 train 同格，不為模型容量刪評估物件；自然接下一章 anchors。

正文完成後實際看兩張新版 SVG：第160步紅線、紅 TP/額外黃 FP、藍/黃 FP+FN、空圖零預測均能按已讀配對規則解釋；尚未查看圖的當場註記只描述讀時狀態，沒有事後覆寫。

本頁連結圖的報告時 SHA：

| 圖 | SHA-256 |
|---|---|
| `docs/assets/diagrams/08-custom-loss-readable.svg` | `a4d3ce1757586212317e71d15b8212a0780ee30e502e5e28c39959be628b11ed` |
| `docs/assets/diagrams/08-custom-predictions-readable.svg` | `0f935cc449e1f52af61c02a9897e41c68bac866848f008680d325c2a5a1ec157` |
| `docs/assets/diagrams/08-custom-learning.svg` | `83036ad187af44e64bf54200e47b47b414dfab96ef03b4e5db79321d187f09a7` |

## 前文與下一節的銜接

- 06decode 的已知假輸出與 NMS，06evaluation 另換兩圖三 GT 四預測；07 開始重新建立固定兩色資料，不能把 06 的分數當模型學會。
- 07data 明示固定紅藍圖與生成圖不同。固定紅中心在格線上、格內 x=0；生成矩形完整在格內、中心嚴格內部、xy 不為0。此差異在開始讀 target 前已知道。
- 07target 的 B=2（紅藍+空）→07loss 的 B=1單紅→07training 的 B=4生成且每圖一物件→160步的32圖含空，各頁都先交代材料。原始 GT list →固定格子答案→loss 只正格框/類、全部格obj→decode 不依賴 GT，是可重述的單向資料流。
- 07inference 再回固定人工紅藍，未訓練模型只核接線，已知 logits fixture核 decoder；07heldout 的三步模型用未更新 seed901四圖核整條評估管線，160步是另一份真學習資料。人工 AP=.5 與160步 mAP並不跨實驗比較。
- 08images 每圖 letterbox metadata 把 decode 的64輸入框回到原圖，exact logits核載入；模型類別仍紅/藍。08own-data 才增加黃類、改 head、同步重建 optimizer與target，原图JSON→letterbox→GT batch→grid target路徑接得上。
- 08 六PNG一步→48PNG 160/1600重新訓練，score截止變.1、三類、非方形來源，全部明示；與07的.05/兩類/32張資料不同，不能跨頁比AP。新test seed只是合成資料便利，不被當成真照片test替代品。
- 08 尾段 train 同格拒絕、val/test 保留所有GT；09開場返回單紅手算、每格兩anchor槽與尺寸exp，loss權重另不同，明示機制實驗沒有品質證據。新增類別不增加槽，anchor也不保證解決所有容量衝突。

前文實讀全文 SHA：

| 頁 | 讀時 SHA-256 |
|---|---|
| `06-decode-nms` (完整272行) | `5a4ab36893f7e89bd5994c0dda82e927a10158069a1a56c3cec9ceaaf9442401` |
| `06-evaluation` (完整176行) | `1242dc8db9d2e4a095ff3b5c5e6e1ac52ef1eba7721a1963d4fe5f9a17be5630` |
| `07-data` (完整190行；06decode/eval實際補讀完) | `f086a8694e46320912d340add2c1339c2a04fedc061c014092962d839ef97730` |
| `09-anchors` (開場至第 95 行：問題、簡化範圍、同一紅框 target、兩槽与三種狀態) | `7604bfaecb11e818a45d2a745c8450801e73c9541b351540ffc7f479080b3cea` |

## 原始 first-read IDs 的具體處理

以下原始問題均留在原 JSONL，不覆寫為 pass。這裡記現稿能在本輪閉合的內容；若原紀錄是當時 prerequisite 未揭露，只能關閉本輪顺读條件，不回改那次 gate 的限制。

| 原始 ID | 原始問題 / 當時理解 | 本輪實際現稿與閉合 |
|---|---|---|
| grid-0066 | 未揭露07data，只有第5章格式 | 本輪先完整讀07data，先知道GT/畫素/CHW與空圖，再讀target；該前提閉合。本輪不宣稱原gate曾正確揭露。 |
| grid-0069、grid-supplied-07-data-04 | target0與生成資料中心不碰線只靠推論 | 07data生成規則現在直接說中心嚴格格內、xy不為0，早於07target使用；固定紅框端點仍保留作sigmoid極限例。 |
| evolution-0042 | 原始targets/target只差s、多套命名負擔 | 07target現稿以原始標註清單→build_targets→固定格子答案說明；表列欄位、shape與哪項loss使用，actual excerpt名字仍原樣但不靠記s差異推理。 |
| evolution-0053 | 15背景logit梯度加總被當CNN權重梯度 | 07loss第191與220行先後明示每logit梯度乘其對權重變化率再帶方向相加；.46875只是輸出信號量，本輪理解與手算不把它等同某权重梯度。 |
| grid-0088、evolution-0061 | 160步結果、圖、計時與重跑檔案混為長節 | 結果、固定協議、圖板與限定先完成；『想重跑或查檔案』獨立標題與折疊把subprocess/報告欄位後移。主要比較前提仍在正文。圖手機可讀性另列G-V01，文字閉合不等於圖已過。 |
| applications-0029、applications-0037 | AP50 prerequisite實際未讀，all-points難重述 | 本輪实际先读完整06evaluation，能手算1/3包絡面積；07training回鏈並用一句重述排序→IoU配對→all-points→有GT類平均；07heldout人工表可算.5。仅閉合這位順序讀者，不修正原讀者當時缺AP前文的事實。 |
| applications-0061 | 折疊『接續1600』誤像載160權重再跑 | 現稿兩次從同seed/初始化獨立訓練；表上方已先說不是載160checkpoint，折疊標題也說獨立重新訓練總共1600；`prior`僅讀report核對，不接Adam，实际source597–600新建model/optimizer，published first160 loss逐值相同。 |
| applications-0062 | 非零梯度宣稱每一步參數更新 | 表與後段明示有限非零反傳信號，仍看step和參數差值；保存值weight_delta L2>0只核訓練前後，不冒稱逐步每參數都有改變，CPU續訓未驗。 |
| grid-0120 | 比較前提後先插SHA/seed/命令長核對才見曲線 | 主文保留相同初始化、train/val與固定設定、新test、最后checkpoint后直接結果與窄版曲線；重建舊資料/SHA/prior/输出差異集中選讀。現場讀完無需回讀猜實驗是續訓。 |
| grid-0121 | 圖的TP/FP/FN被两程序CLI/JSON與checkpoint計時打斷 | 图解后直接四张固定取样≠整组AP提醒与真test限定；兩次CLI與JSON核對、梯度/存載、計時各選讀。圖/AP64座標和CLI原圖座標差異保留在必要位置。 |
| evolution-0022 | NMS練習插長容差公式 | 當前06decode練習保留allclose但詳細容差在選讀；本輪能先預測.6保留三框與.75只高分FP，再選擇核數值。 |
| evolution-0026、evolution-0027 | 官方規則與AP不等式打斷主例 | 06evaluation先完成四框表→PR→包絡面積，再放官方差異與P×R選讀；能由主文重述AP1/3。原建議額外GT空間示意未新增，本輪由坐標表/相對位置文字可追，不能聲稱該圖已補。 |
| evolution-0065、evolution-0066 | anchor訓練target與raw反解混用、多套命名 | 09開場先明說訓練中心比例vswh raw修正，正文target四值及小表與optional反解分開；source名稱表對actual名字。這是下一節過渡的有限核讀，詳細技術依另份evolution.md。 |

本輪抓取本組相關原始 issues 與當時理解/圖需求，沒有逐單位重新閱讀所有無 issues 的原始紀錄，也沒有用作者摘要替代。本輪八小程式執行核對不是新增首次讀 pass。

## 新觀察、root 修正與核回

**G-T01（low，已關閉）**：07inference 第93行原說 inference_mode 新tensor『之後不能再拿去參與要算梯度的計算』。PyTorch 2.9.1官方 `torch/autograd/grad_mode.py` 同樣採保守總括，但實跑outside `x+a` backward可行、`x*a`需要保存inference tensor則RuntimeError。root已改成『接回某些需要梯度的運算時會報錯』，本例只推論、要接回訓練改用no_grad；本審查者實際重新讀該句，符合CPU事實，且不增加無關長推導。原始當場問題保留。

官方固定版本位置：[PyTorch v2.9.1 grad_mode.py, class inference_mode](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/autograd/grad_mode.py)；下載文件SHA `84d64257ccd56bd1af2994d84e34f4f4b04e11a90b8b82edccfe27ce210e7318`。沒有由此宣稱兩API速度比較。

**G-V01（medium，待核回）**：07training/heldout現行生成圖在390px viewport、358px內容寬時，grid-learning-curve最小字6.36px、grid-learning-predictions5.97px。圖板基本位置可識別，必要score/IoU與TP/FP/FN標籤吃力，不能靠文字解釋代替可讀圖。已回報root；applications會只從原JSON重排新窄版圖，保留舊生成圖補充，不重訓。待实际新版SVG與段落修改後追加closure/hash，当前不宣称此项通过。

08两张新图相同預覽條件最小字15.15/13.77px；實看可讀，无BBox超viewBox。08letterbox圖8.95px與07data7.96px也有小座標字，但主要解碼/坐標數值完整在近鄰正文，07責任格圖14.04px可读；將实际站CSS/这些小字的最终mobile验收留root，不把放大预览当通过证据。

## 執行 source 與公開 evidence 指紋

以下是報告時的內容指紋；source base commit不是唯一指紋，实际未提交文件以SHA核对。沒有改shared code、evidence、舊reviews或coverage。

| 檔案 | SHA-256 |
|---|---|
| `lesson_cases/07-data.py` | `3e82d40e0dafb7c07813e8521472bc6dc220eefae4ebd576353fa3daa3a24215` |
| `lesson_cases/07-targets.py` | `95804f80642c0f83830c7f88de9e3a4aa540582e67698af1cc8edab3bf2cd9c0` |
| `lesson_cases/07-loss.py` | `1fa1c9ace4eb1399130d340267059a00d204cf682bb56f33ca2f0a11a3ad84d0` |
| `lesson_cases/07-training.py` | `fe1037df4ece00bef8810d5d3a31184c1a862b926306688759bcb5cb748afc99` |
| `lesson_cases/07-inference.py` | `75011612e6f4c8aee853cdcaec7e7d64d44499e70464cdbb108d98e06c937c10` |
| `lesson_cases/07-heldout.py` | `ada566e4f1c294156116f7771c2bf31dfc7d23770a7e76a56331a595cf888568` |
| `lesson_cases/08-own-images.py` | `d812cdb29d585e4c6ea6be4a3e1e502259be7bd48547cda749378a89a8aff65a` |
| `lesson_cases/08-own-data.py` | `d3be8a09b62a77c542ba48a93e10c36f5a8a2f34e587b2e207c4d7e41c8f300a` |
| `miniyolo/data.py` | `cccad00e2c4f96eb6567eafc9e12248379c6b715fc1790d75518a253baa6181d` |
| `miniyolo/targets.py` | `2c8e32f2845b3bf970c77304c5cca0f999083d36a4d5af2f75f14aa89b823f16` |
| `miniyolo/losses.py` | `81fa9331c9a2aebe2e9c6c453a5313e64566bb20c3fe77ca45ec9df53424d4e4` |
| `miniyolo/models.py` | `49d029ea4ba2650ce8933cf97e3d25dc7aff2ca4e972eec19cdda17b0f4900e6` |
| `miniyolo/geometry.py` | `6a6b57d3493888e99dae4a012dab78127b8b543a0e01d8107963a3d6e63d8483` |
| `miniyolo/inference.py` | `995ac8f942d0c1e43d94f2efb3b7adcb91a9feb6b9bab923615209f0c190c313` |
| `miniyolo/metrics.py` | `53ce982e37cbd96c784f75c7d30faf99d52f79ab83ca7b8114eb21b4327330e0` |
| `miniyolo/custom_data.py` | `7dd9999f03dbc32ce1ec951c58121c8f746d381b0244e4c2a82cca59e7ca7e55` |
| `scripts/detect_image.py` | `69738c487cfad24bea2009384712ca0ba958ac1fd394e2226ae33d7de9682c6a` |
| `scripts/run_custom_data_learning.py` | `b8b3b442e4e2349173a2eb8198b33f966d1e75d80c8b1edb07d1af4d3b5e99e5` |
| `miniyolo/train.py` | `aa5567f11be6ca7aee97bd082b9d5964414b12b73453763c915c7ffb409c4276` |
| `artifacts/checks/grid-learning.json` | `c66887b113691433f4f8787cb55fbbb6f4dacf26d635565dd259b5988417a298` |
| `artifacts/checks/curriculum/custom-data-160-step.json` | `d1b5582921555b93338f26088ceb2d4e3a84b8a4b04266965df86a4ea37b0f96` |
| `artifacts/checks/curriculum/custom-data-learning.json` | `fb22ce8356fbd747f967c3b82df53bd55de840d826d44fc73abfdba315f8e584` |
| `artifacts/checks/curriculum/07-data.json` | `f9acacbb73cb8f6edafc383c2f5be0ba215a0c56db8f14cc2aca5711d31b946a` |
| `artifacts/checks/curriculum/07-targets.json` | `26cf6375a7747f2190d273e222610b534d1b5f094f52e404112334c02dace320` |
| `artifacts/checks/curriculum/07-loss.json` | `27c8020d6b08fa20d74921d8373d6d5d3d01ef8105d34f891acd2a01f65100ad` |
| `artifacts/checks/curriculum/07-training.json` | `13f6792e07b1d98882c67b8978f85a328bcbfd4ba454cacccf2c0b0cfa1dbd4d` |
| `artifacts/checks/curriculum/07-inference.json` | `d82f3587c30c4d84b5987a0dc51d992f3a446abbd90f9b7ff1457336fa1d1157` |
| `artifacts/checks/curriculum/07-heldout.json` | `38186fc33a15af761c5e4bc5baf00a047cf4009afbb4b46967eb66add6c5c332` |
| `artifacts/checks/curriculum/08-own-images.json` | `74941dd06348105327dcb18b1828531137222cafc31570a7edd3d140ac32373c` |
| `artifacts/checks/curriculum/08-own-data.json` | `096eb292085f114c186ed269e8391ff056496af404468b93acf2890be4bf2d26` |


## 新版 07 主圖及圖說核回（G-V01 已關閉）

核回時間：2026-10-05T10:53:05.778959+00:00。初稿 G-V01 的舊圖字級與當時「待核」保留作歷史，**本次新圖可讀性已實際核回，G-V01 已關閉**。G-T01 的 narrowing 句仍已關閉，当前七頁正文銜接無 blocking 問題。

先讀當前07training/heldout圖說，再直接核兩新SVG與原 grid-learning.json，沒有先讀圖作者報告，也沒有採作者自查作通過證據。兩圖 data-source-report 指紋均對上同一原JSON，沒有重新訓練、改選驗證圖或改生成圖。獨立核對：

- loss圖四條polyline各160點，逐點對照total/box/objectness/classification，用x=64+i×426/159、y=438−loss×224還原，差在SVG小數輸出精度內（≤.00051 SVG單位）。紅box尚未乘5，total=5×box+obj+cls；第1點是第1次更新前。圖没有把訓練loss當validation品質。
- 圖板內四個embedded PNG的bytes逐一等於既存生成圖同index0–3；GT與prediction的每個rect按220/64縮放，位置/寬高逐一對原JSON（1e-6精度）。保留原四圖、原排序、原框。
- 由完整未四捨五入框自行算IoU並依同圖同類、score排序、未配對GT執行一次配對。圖片0藍#0 score1.000/IoU.623834→TP，紅#1 .980/.473420→FP，紅GT→FN；圖1藍1.000/.729035、紅.998/.753848均TP；圖2紅1.000/.771366與圖3紅1.000/.899224均TP。標籤的.62/.47/.73/.75/.77/.90均符合捨入。四張圖不能替代全部16張的AP，正文提醒保留。
- 当前07training第166行為各圖下方兩行資料及FN説明；07heldout的最新「#k 紅／藍 score s」、紅=class0/藍=class1、1.000僅四捨五入、第0圖「#1 紅 score0.980」已由本人實際重讀，與SVG一致。没有留下要求讀者找不存在的class c標籤。

實際圖可讀性：訪問root共享的Zensical HTTP 07training與07heldout，390px與1280px viewport。390px圖實寬342.98px，最小22基字換成約14.51px；1280px實寬687.98px，換成約29.11px。兩張新圖的主要數字、#k、score、IoU與TP/FP/FN可讀；手機頁document.scrollWidth375≤390，這個範圍沒有圖造成的水平溢出。served SVG bytes均與當前工作樹SVG逐位元相同。原始頁context截圖與每img截圖保留，另外把高圖capture中可能混入的sticky header隱藏後存clean補圖；未改SVG/layout，也没有把補圖冒稱整頁UI測試。

這次实际站頁snapshot建立時，root剛修的07heldout紅/藍圖說尚未被sharedsite重建，因此page SHA只表示截圖時工作樹Markdown指紋，不聲稱它就是served HTML內容。**圖說文字closure依current Markdown實讀，新圖closure依相同SHA的actual served SVG/PNG**。最終重建後圖說的HTML、折疊/全頁desktop/mobile由root全站browser承接，未在本報告冒稱已驗。此限制已寫入manifest。

同時修正本報告初稿「當場觀察」逐字分號的呈現錯誤：它來自把字串當list做join，只恢復同一原筆記完整句，沒有改當場理解或新增答案。保存的grid-current-read.jsonl與/tmp原檔逐位元一致，11筆、非gate盲讀聲明保持。

### 最終核回指紋

正文SHA為 curriculum-evidence:start 註記前原始bytes；讀時指紋仍在上文與原筆記，不以此表取代。

| 檔案 | 全文 SHA-256 | 正文 SHA-256（若適用） |
|---|---|---|
| `docs/lessons/07-targets.md` | `a3816d6f82e0bb9859557d6771ec2978d66304ee4f2a2f6670d8e4514421c205` | `586e9f871b1ab6dc1144fc99522a85592be11338675a1b2e94fe470d11dcfc8b` |
| `docs/lessons/07-loss.md` | `2b3a4f33c0e12feab2c8f52f97fee44bfdbfcaa8e8329ff44811c4720288ae52` | `34c549ea9fe70c0a36d42d47d118b30779b68a232f8dd45659509e11067b882f` |
| `docs/lessons/07-training.md` | `444cfaaad5a648dcd95147beedbc4d167f95812956f042983171b82f491d429a` | `73efbb950aa626e4a87795f8db5c517bd6eb28775eb6d329b37521aec5a81d4a` |
| `docs/lessons/07-inference.md` | `37e293a9a25016581b92fc216eb3d868b1c80c828c074ee6593a61006b289bfc` | `f30cd709e2d14ebe952c4d62018797fbb131ff29872262c8b007c51829d3b009` |
| `docs/lessons/07-heldout.md` | `24d6e1216e4845d996f54b568b7b84c03e0c2868f3da0d538bd30122584c741e` | `c44faea8f42d92075cc7e94a4fb59ba955a958b7bdfe19c3401bfaa8dfbd2948` |
| `docs/lessons/08-own-images.md` | `332da2c487b9e3591b61dbb02fc182bbaa1f4afc20461e2c05da332d7f822006` | `1566e446a33c6bd664ca3b4fa095dc1852bb54010d112132520a1050a8d32184` |
| `docs/lessons/08-own-data.md` | `51fd720683e2e36df9ec5b85618caedd6a2f75c4c29e70aa23fbd5c5c5d2f671` | `dfb2869d43e3b5c685d48003b39af26bf1c7a337b852e761fe1f7d10ab3d9f87` |
| `docs/assets/diagrams/07-grid-loss-readable.svg` | `8975637e5d4912b68f9bbd9a02dd092c31e9d0f029cc5f6268f730d3fe14adec` | — |
| `docs/assets/diagrams/07-grid-predictions-readable.svg` | `ac0aef6ee5429918f840f78711b29e6abeebc48235456d0e6fb8b151f0c0ecc3` | — |
| `artifacts/checks/grid-learning.json` | `c66887b113691433f4f8787cb55fbbb6f4dacf26d635565dd259b5988417a298` | — |
| `reviews/clear-tutorial/16f6910/transitions/grid-current-read.jsonl` | `0c214d888fd40f73f5c4e925fd4a2386c0120d47a21626e03af8fd36410aa6dc` | — |
| `artifacts/runs/clear-tutorial-16f6910/grid-transition-browser/manifest.json` | `20457bc601c47edda10b451e039c31d4b755b764cdc67fdb5f694c4bfebb4d5a` | — |

### 本地 actual-page browser 範圍

| 頁面/圖 | viewport | 實際圖寬 px | 最小rendered字 px | SVG SHA-256 |
|---|---|---|---|---|
| `07-training/07-grid-loss-readable` | 390 | 342.98 | 14.51 | `8975637e5d4912b68f9bbd9a02dd092c31e9d0f029cc5f6268f730d3fe14adec` |
| `07-training/07-grid-predictions-readable` | 390 | 342.98 | 14.51 | `ac0aef6ee5429918f840f78711b29e6abeebc48235456d0e6fb8b151f0c0ecc3` |
| `07-heldout/07-grid-predictions-readable` | 390 | 342.98 | 14.51 | `ac0aef6ee5429918f840f78711b29e6abeebc48235456d0e6fb8b151f0c0ecc3` |
| `07-training/07-grid-loss-readable` | 1280 | 687.98 | 29.11 | `8975637e5d4912b68f9bbd9a02dd092c31e9d0f029cc5f6268f730d3fe14adec` |
| `07-training/07-grid-predictions-readable` | 1280 | 687.98 | 29.11 | `ac0aef6ee5429918f840f78711b29e6abeebc48235456d0e6fb8b151f0c0ecc3` |
| `07-heldout/07-grid-predictions-readable` | 1280 | 687.98 | 29.11 | `ac0aef6ee5429918f840f78711b29e6abeebc48235456d0e6fb8b151f0c0ecc3` |

PNG個別全SHA與來源對應保留在 `artifacts/runs/clear-tutorial-16f6910/grid-transition-browser/manifest.json`（SHA見上表）；本輪同一manifest的其餘六SVG核圖詳technical/evolution.md追加closure。自行重算配對為 `/tmp/clear-tutorial-grid-cpu/new-figure-matching.json`。未把/tmp暫存當永久發布證据；公開source/evidence及本報告指紋讓核對範圍可追。

# modern 首次閱讀總結

本組從未讀過本書開始，依 gate 實際先補讀 61 個 prerequisite，再讀 48 個 target；共 109/109，gate 已顯示 READING COMPLETE。這不是從首頁順讀全書的聲明。全部凍結 source_commit 為 16f69103d1f216d1024a40610623083557324701。只透過 gate next/record 取得單位，每單位由模型當場閱讀與寫記錄，未搜索或讀未揭露教材/實作。未修改教材、未執行訓練或程式測試、未使用 GPU，未驗證 Zensical 排版、瀏覽器頁面或手機頁面。

原始當場記錄由 gate 保存；同內容逐單位副本在 `/tmp/clear-tutorial-modern-original.jsonl`（109 筆，modern-0000～0108）。每單位的理解、元素、下一變化依據、issues、圖需求、視覺狀態及variation均有記錄。下列問題尚未經作者修正複查，全數保持待處理，沒有 closed。

## 目標頁逐頁

### 13.1 docs/lessons/13-dual-assignment.md（modern-0061～0065，5 單位）

- **burden，modern-0062**，原文「本例的一對一要求每個 GT 恰好配一個候選」：主題是 YOLOv10，主例卻先大段介紹全域最佳、貪心、排列、Hungarian，以及與 many 相反的 owner，最後才看官方每 GT top-1。讀者需同時保留 toy「每 GT 恰好一個」與官方「至多一個且衝突可丟 GT」兩套規則。建議先在同一品質表走 top-1 與衝突，作主線；exact/greedy/Hungarian 作獨立選讀，說明用途。
- **burden，modern-0064**，原文「圖上有兩個物件，就該有兩個正樣本。」：此前已說官方 top-1 的兩 GT 搶同一候選可讓 B 沒有正樣本，此句又像保證兩 GT 有兩正。建議限定本例全域配對且候選足夠，或改成每 GT 至多一正並保留衝突可能丟 GT。
- 應保留：owner 與 best 方向相反的解釋、GT 索引不等類別、兩 head 不等 box/class 分支、one 輸入 detach 仍訓自身、品質一致保證的假設與限制、推論 top-k 與訓練 top-k 區別。
- 圖：本頁無實圖。modern-0063 記錄可用雙 head 訓練/推論路徑圖統整主線；直到 modern-0098（16.2）實看雙 head 圖才得到相同視覺統整。保留早期需求，未回改原記錄。

### 13.2 docs/lessons/13-nms-free.md（modern-0066～0070，5 單位）

- 原始 issues 空；沒有必須猜才能继续的卡點。
- 應保留：兩套只差 p1 target 的對照，p1 分數真正由訓練降 .959→.041；三種數值同 .5 的門檻分別比什麼；NMS stable tie 與局部 IDs 映回；top-2 兩框皆屬 A 仍漏 B 的反例；score .99 無框但 recall=0；每 GT 覆蓋不能只靠框數。
- 本組未實读第 6 章評估前頁，但本頁 note 的 TP/FP 表、precision/recall 公式與兩段包絡面積補足了 AP=5/6 的手算；未冒稱評估前頁已審。
- 視覺：實看 `2d108579bdcbdfe9.png`，比例 x 範圍與 p1 黃底 target/分數對照清楚。頁面排版未驗證。

### 14 docs/lessons/14-feature-module.md（modern-0071～0075，5 單位）

- 原始 issues 空；沒有必要卡點。CSP 原頁不在本組補讀範圍，開場提供的對比足以理解本頁。
- 應保留：四路 a/b/b1/b2 全部留下的圖；1×1 投影先混合才切半；channel 帳本與理論感受野/實際 padding 範圍；直接 concat 梯度與 path 總梯度兩個不同檢查；plain 只比較成本而非未訓 loss；roll 非通用位移的兩限制及全 0 基準，清楚限制「loss 下降」的意義；blocks 變多不普遍省參數。
- 視覺：實看 `05e9d224ecc50116.png`，四路箭頭、concat channel 段與 fuse 16→8 清楚。

### 15.1 docs/lessons/15-attention-bridge.md（modern-0076～0082，7 單位）

- 原始 issues 空；沒有必要卡點。
- 應保留：row-major tokens 與錯 reshape 的實值對照；Q/K 決讀比例、V 是內容；identity 只是初值；第一列 softmax 的分子/分母/輸出完整手算；內積不同於 cosine；來源軸 softmax 每列 1；√d 的可選推導與飽和例；target 的 broadcast、梯度 allclose 與權重 equal 各自用途；2×2 例不顯示距離收益；權重不是最終因果贡献。
- 視覺：實看 `b30fab1e6c2b15e2.png`，shape 路徑及第一接收位置指向來源的箭頭有图例，字與数值清楚。

### 15.2 docs/lessons/15-area-attention.md（modern-0083～0088，6 單位）

- 原始 issues 空；沒有必要卡點。
- 應保留：area 分位置、head 分 channel；A4 橫帶非四象限；reshape 的各份獨立矩乘；共享 QKV 只比較讀取範圍；N²/A 減中間表而不減參數、不等於整體速度四分之一；token15 干預前後可手算；官方 BN/位置卷積與其他尺度可跨區，本小例移除它們有明确理由；候選3同區干預變化。
- 視覺：實看 `8da4c9ad7640a2ef.png`，A4/A2 编號與邊界、query0/changed15/3 清楚。

### 16.1 docs/lessons/16-dfl-free.md（modern-0089～0096，8 單位）

- 原始 issues 空；沒有必要卡點。
- 應保留：reg_max=1 是 Identity 跳 DFL，不是唯一 bin0；18 格例只比較 stride8，其他 stride 可涵蓋同一 144 畫素；負距離有實際 STAL 用途但 l+r/t+b 必須正；Smooth L1 首步梯度/步數與真模型训练速度不同；未訓 DFL 不拿來比準；raw 輸出成本不是整模型；官方 CIoU+正規化 L1 及 loss_dfl/dfl 舊命名；DFL-free 與 NMS-free 是兩個設定。
- 視覺：實看 `943836f3d0ed7172.png`，點84、四距離與 x204 的 DFL 右界清楚。

### 16.2 docs/lessons/16-inference-head.md（modern-0097～0102，6 單位）

- 原始 issues 空；沒有必要卡點。
- 應保留：只讀 dict one 仍白算 many；deepcopy 裁成真正無 many 模組；fake loss 的目的只使 one 離開初始權重；raw 完全一致與人工 decode 两种驗證各驗什麼，人工例未驗候選排列的边界明确；eval 不是删除，BN 模式與融合分清；單標 top3/官方候選類組合 top-k 区別；硬 stride 換128時所有assert仍可通過但框錯一半；論文預設與 API nms=False 路徑不同。
- 視覺：實看 `b7c9790c7b715b0b.png`，前向、梯度阻斷與裁剪後 raw 逐值比對清楚；此圖也補了13.1早期的雙 head 視覺需求，不能因此抹掉早期負擔。

### 16.3 docs/lessons/16-training.md（modern-0103～0108，6 單位）

- 原始 issues 空；沒有必要卡點。長篇 eigenvector 推導为可跳讀補充；主要首步与等總量對照可跟。
- 應保留：三技巧作用/驗證程度分開；三组同seed/資料/步數與等總 b 對照；首步 weighted SGD 逐值表；獨立 toy 中 a 不影響 one，不能支持前期 many 好處；b 顺序不影响此线性二次例是推導而非跑過的反排對照；STAL 原 GT 与资格框分開、四點只進候選池、不同stride及小邊各自判断；MuSGD矩阵/bias分組、未跑与發布配方限制。
- 視覺：實看 `ae97fe447833c8a4.png`，原绿2×2、紫資格16×16與四蓝点、影像y向下都清楚。

## 實際补讀背景及其問題

實際補讀頁面為：07-data（0000～0006）、07-targets（0007～0012）、07-loss（0013～0023）、11-iou-loss（0024～0034）、12-anchor-free（0035～0040）、12-decoupled-head（0041～0047）、12-assignment（0048～0053）、12-dfl（0054～0060）。未實讀它們連結到的其他章節，沒有從第1～12章全面順讀的聲明。

- **burden，modern-0027，11-iou-loss**：正文「一步剛好左移 1 pixel，中心從 40 到 39，可以直接對照上面手算的 1.179487。」但 modern-0024 實看图底寫 `GIoU loss 1.2 → 1.174987`。正文手算及 modern-0034 執行輸出皆 1.179487。需要統一圖底數字；待修正複查。
- **optional，modern-0033，11-iou-loss**：第2题先问 IoU loss 与中心梯度，随后「\(L_{\text{GIoU}}\)、\(L_{\text{DIoU}}\) 呢？」讓我预期也求其梯度，答案仅列G/D loss。需说明只问loss或補梯度；待處理。
- 其他背景原始 issues 空。資料契約、target mask、全空batch、梯度方向、anchor-free ltrb、共享分支及动态 assignment/DFL的當場理解都已逐單位保存，不以術語清單代替實際閱讀。

## 視覺與驗證範圍

共實看12張 gate 指定的静态 PNG preview：5張 prerequisite（data `b2bcab88a92808d4.png`、object journey `0420f5474adeda77.png`、IoU `bfbde9baa2c597b7.png`、anchor-free `a2f1561370e731e4.png`、assignment `9d3e243c946e2cc9.png`）及上述7張 target。全部可見必要字/框；IoU preview 的數字差已記錄。

preview 為原 SVG 的靜態瀏覽呈現；未看原 SVG 源碼、未验证 Zensical 400宽图/手機縮放或排版。沒有執行程式、訓練、GPU、官方模型/外部来源、AP、held-out、匯出或 latency 測試。作者提供的執行紀錄只作阅读材料，沒有說成我實測。

主閱讀斷點集中13.1 toy exact配對與官方top1主線的切換；必要背景到各目標的共同例子/單位/shape、大部分具體數值和圖解可理解。建议保留嚴格區分「可計算、能反傳、實際更新」與「偵測效果/版本收益」的界線及可預測小變化練習。

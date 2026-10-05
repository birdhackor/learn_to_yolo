# DINO 四節與偵測橋接：作者自查與凍結材料

日期：2026-10-05。這份紀錄是作者自查，不是獨立首次閱讀、技術審查或銜接通過。已讀真實前置 15.1、21.1～21.4，並遵循 `.agents/skills/clear-tutorial/SKILL.md`、`references/review-protocol.md` 和根目錄 `AGENTS.md`。

## 可供分段揭露的閱讀單位

### 22-views.md

## 一張矩形，做出兩份看法
## 要保留的共同內容，決定了 view 怎麼做
## 讓同圖兩份表示互相接近
## 程式先只生成 view
## 改一件事，先預測

### 22-collapse.md

## 先看四張 view，和兩種回答
## 一個零 loss，不能替兩個模型分高下
## 下游任務會看見這個漏洞
## 執行這個失敗例
## 改一件事，先預測

### 22-distillation.md

## 兩份 ViT，先從相同權重開始
## 先用舊 center 和兩個溫度，算出這一步的答案
## 只讓 student 朝跨 view 的目標移動
## 再更新 center 和 teacher，供下一步使用
## 跑一次真更新，並檢查能否接續
## 改一件事，先預測

### 22-features.md

## 保留相同材料，把訓練與評分分開
## 第一種讀法：找最像的訓練圖
## 第二種讀法：只訓練一個線性分類頭
## 必須和同起點的隨機特徵一起看
## 曲線和特徵統計，各補充一件事
## 改一件事，先預測

### 23-detection-bridge.md

## 仍是一個矩形，多了一份位置答案
## 用 patch 特徵，把空間排列拿回來
## 一個新 head，讀出顏色與四個框參數
## 一張圖一個框，直接按圖片配對評分
## 改一件事，先預測

## 實測及來源範圍

- 實作 agent 保存的五個實際 CPU case JSON/stdout 位於 `artifacts/runs/dino/`；`case-runs.json` 保存 case source SHA、指令、exit status 與耗時。本文讀取最終 JSON，沒有自行再訓練。case 的詳細獨立技術驗證仍交由協調者安排。
- 22-views/22-neighbors/23-detection-predictions 直接以相同資料規則、sample count、seed 與 view RNG 重建像素並 embed PNG；views 的 crop 與 JSON 用 assert 比較。預測框直接讀 23.2 JSON，保留 test0、1 未達 IoU0.5 的失敗及 test2、3 達門檻例子。
- 22-features 曲線用 Matplotlib 讀 22-distillation 的逐步 actual history；test 特徵 std/cosine、最近鄰索引讀 22-features JSON。來源是相同固定設定的兩次 case，沒有把示意或手工向量當成實測特徵圖。
- 22-collapse 是二維平方差手算 SVG，正文再接 K16 固定分佈交叉熵的實際計算；沒有模型更新，沒有 center ablation 結論。
- teacher stop-gradient、raw logits center、student step 後 teacher EMA 與特徵用途的來源位置核對 `reviews/vision-branch/sources/primary-facts.md`：DINO2104.14294v2 §3.1/Alg.1/§3.2/§5.3；固定 main_dino.py 7c446df 的 L318–319、L384–416、L326–350。
- SSL/random 的 1NN 和相同 seed900/120step linear probe 均 validation/test64/64。教材明寫沒有 SSL 改善證據。Bridge test cls64/64、meanIoU.606、joint54/64，按圖一框配對，沒有 AP、架構排名或自然照片效果結論。

## 圖檔自查與尚未驗證

- 22-collapse.svg: XML valid; width520; minimum explicit/inherited font 24; 17 text nodes
- 22-distillation.svg: XML valid; width520; minimum explicit/inherited font 23; 25 text nodes
- 22-features.svg: XML valid; width520; minimum explicit/inherited font 22; 25 text nodes
- 22-neighbors.svg: XML valid; width520; minimum explicit/inherited font 23; 10 text nodes
- 22-views.svg: XML valid; width520; minimum explicit/inherited font 23; 11 text nodes
- 23-detection-bridge.svg: XML valid; width520; minimum explicit/inherited font 22; 34 text nodes
- 23-detection-predictions.svg: XML valid; width520; minimum explicit/inherited font 23; 16 text nodes

每圖只有520px intrinsic width，font最小22。已核XML、來源／shape／座標與arrow文字；普通Chromium截圖未完成，沒有將此記為已完成圖像或網站視覺檢查。actual Zensical 桌機／手機、公式及分段首次閱讀交由 root 後續檢查。

## 凍結內容指紋

| 檔案 | SHA-256 |
| --- | --- |
| `docs/lessons/22-views.md` | `7d06f54b7c87e88fce8bfa3026de4bad0f6c936e31152bbf0340e3d340d56461` |
| `docs/lessons/22-collapse.md` | `100ff5a7d28060227b4ea0639662b32997d0cb5e8cbf71abe87a12f601d8649e` |
| `docs/lessons/22-distillation.md` | `1b87c4b08551c1ac0a0d539724e13cc0caa0c9683de96a1d9ca2558ed4609adf` |
| `docs/lessons/22-features.md` | `2be7f771354a000dc18c6c47ca014a365616911a775e3fa6dfd1323965186f54` |
| `docs/lessons/23-detection-bridge.md` | `0e2718966db4e6799ddc292dceb204413f47562b13b3435ab39aad605184db8b` |
| `docs/assets/diagrams/22-collapse.svg` | `75355d38843db229681ee0358fb2b552a5b20773844462e426ffd8e611b331d8` |
| `docs/assets/diagrams/22-distillation.svg` | `d1e86428f886504bef67b72069caf90da9df89646564863fc8988958b0c5736f` |
| `docs/assets/diagrams/22-features.svg` | `e72e90a03c2ae05b307ef206bd5130753bb92d8d60cd51520b87eda2d86a4f40` |
| `docs/assets/diagrams/22-neighbors.svg` | `951440b4c4ceeccea25d66f9bd17c4cec44ef7ba7fc16497e1d390a9b838f73a` |
| `docs/assets/diagrams/22-views.svg` | `c6ac553db3b63d61ac75d38b7cb98640b671434e71e1781c108815b3a24a8d31` |
| `docs/assets/diagrams/23-detection-bridge.svg` | `acf675cb2992480924913bb7ac24d240182ce282946ab5ea9c3918c2a8731dca` |
| `docs/assets/diagrams/23-detection-predictions.svg` | `a52635e89a4ef0975e796fb596c25ea0aebf5e43612042946ab812feab7baaa3` |

# 閱讀路線：先追一個問題，再加下一個機制

<details class="chapter-a-toc">
<summary>本頁目錄</summary>
<ul>
<li><a href="#a-cnn-03">A. 從神經網路到能學的 CNN（第 0–3 章）</a></li>
<li><a href="#b-48">B. 從分類走到可評估的偵測（第 4–8 章）</a></li>
<li><a href="#c-yolo-916">C. YOLO 的演化機制（第 9–16 章）</a></li>
<li><a href="#d-1720">D. 整合與應用（第 17–20 章）</a></li>
<li><a href="#e-attention-vitdino-2123">E. 從 attention 分岔：ViT／DINO（第 21–23 章，選讀）</a></li>
<li><a href="#_2">遇到卡住的地方</a></li>
</ul>
</details>

全書安排 52 節：第 0–20 章的 42 節主線，加上第 21–23 章的 10 節 ViT／DINO 選讀支線。先讀[全書開頭](index.md)的影像辨識故事，再從 A.0 依序讀即可。路線先回答「網路怎麼從資料學」，再問「除了類別，怎麼回答位置與多個物件」，之後才看 YOLO 的不同設計。每節配對一個獨立的 Colab notebook；只想先讀網頁時，不必連上 Colab 的執行階段（runtime：Colab 替你開的雲端機器）。第一次遇到或想複習術語時，都可以查[術語快速查](glossary.md)。

**開始前需要會什麼？**

- **必要：**會基本 Python：變數、if／for、函式、list 與 dict，也看得懂 class 的寫法（`class`、`self` 與方法；本書的模型都寫成 class，這個 class 和物件的「類別」是兩回事）。還不會的話，可以先讀 [Python 官方教學的繁體中文版](https://docs.python.org/zh-tw/3/tutorial/)第 3–5 章與第 9 章（class）。
- **有了比較輕鬆：**照著範例跑過一次 PyTorch（Python 的深度學習函式庫），聽過神經網路與卷積的概念。A.0 先教神經元、線性與非線性，再用一個參數示範梯度下降；A.1 接到卷積與圖片分類。這些經驗不是省略基礎介紹的先決條件。
- **數學：**高中程度的函數、指數與對數、向量與矩陣乘法即可。課文的 ln 是以 e≈2.718 為底的自然對數（例如 ln 2≈0.693），A.1 解釋 softmax 與交叉熵時就會用到。導數（變化率）與連鎖律，第 0 章會用數字例子從頭說明。數學不需要熟背；公式會連到數字與程式。

??? note "預先查詢形狀記號（A.1 本文也會教）"

    後面常用 `[B,C,H,W]` 描述一批圖片的形狀。例如一次送 8 張 64×64 的彩色圖，形狀寫成 `[8,3,64,64]`：B=8 是這一批（batch）的圖片張數；C=3 是 channel（通道）數，也就是每個位置有幾個數值：RGB 彩色圖每個位置有紅、綠、藍 3 個值，各自排成一層，就是 3 個 channel；H、W 分別是高度、寬度。C 有時代表類別（class）數，各節會寫明是哪一種。

**依目標選讀：**

- **先取得一個成果：**第 0–7 章完成單張圖片的物件偵測流程；第 8 章接自己的圖片與資料。
- **完整演化：**再讀第 9–16 章。
- **整合與應用：**第 17 章是結業任務，讀完第 0–7 章就能做，不必先讀第 9–16 章。第 18–20 章是選修，可按需求挑讀，但第 18、20 章要先讀 8.1 節（非正方形圖片的補邊與框還原），第 19 章要先讀第 18 章。
- **ViT／DINO 選讀支線：**先讀第 1、3 章與 15.1 attention，再從 21.1 開始。這條支線沿分類、影像特徵與單物件定位前進，不改主線順序，也不要求先讀完所有 YOLO 版本。數學仍從向量、shape 與實際數字說起。

第 9–16 章（版本節）不會把每一代 YOLO 的改動依序全部疊到同一個模型上。每節都寫明自己的起點（從哪個簡化設定出發）和這一節改了什麼；和起點還有其他差異時，該節也會列出，例如 9.1 節同時改了寬高的寫法和每格的槽數，第 10 章的兩尺度模型是另外寫的小網路。這些小實驗用來看懂機制怎麼算、多了哪些成本，不是公平的效果比較；哪些結果有實測，見[驗證範圍](status.md)。

## A. 從神經網路到能學的 CNN（第 0–3 章）

這一階段從最小的計算器開始：先看權重與 bias 怎麼計算，再看為什麼有些分類題需要非線性，最後用梯度下降訓練。接著把輸入換成圖片，學 CNN 怎麼看局部、保留通道差異並產生類別答案。當網路沒有學好，先診斷資料與更新，再看 ResNet 怎麼以捷徑改變學習路線。

| 順序 | 這節回答的問題 |
| --- | --- |
| [A.0 神經網路與第一次學習](lessons/00-warmup.md) | 神經元怎麼算？一條直線分不開的答案，非線性如何幫忙？梯度下降如何改參數？ |
| [A.1 CNN 如何讀圖](lessons/01-small-cnn.md) | 局部卷積、RGB、池化與感受野怎麼接到分類？為什麼用 softmax 與交叉熵？ |
| [A.2 訓練診斷](lessons/02-diagnostics.md) | 有更新、學會訓練題、答對新題，如何分開檢查？ |
| [A.3.1 原樣捷徑](lessons/03-identity.md) | 輸入已有的部分與共同修正，如何分給兩條路？捷徑如何影響前向與反向？ |
| [A.3.2 投影捷徑](lessons/03-projection.md) | 高寬或通道改變，怎麼相加？1×1 為什麼仍能混合通道？ |
| [A.3.3 對照實驗](lessons/03-comparison.md) | 如何固定條件，判斷捷徑在這個設定中的效果與成本？ |

讀完 A 章，你應能沿著「輸入 → 特徵 → 分數 → loss → 梯度 → 更新」解釋一個小分類器，也知道一份好看的訓練結果還需要哪些驗證。B 章才把位置框加入答案。

## B. 從分類走到可評估的偵測（第 4–8 章）

分類只交出一個類別；偵測還要回答物件在哪裡，以及一張圖裡有幾個物件。這一階段先讓模型多輸出一個框，再處理多物件的答案分配。接著用人工指定的框練習刪除重複預測與評分，把規則看清楚後，才接成能訓練、推論及評估的 Grid MiniYOLO，最後換成自己的圖片與資料。

- 4.1 [單物件分類與定位](lessons/04-localization.md)：類別之外，還要回答在哪裡
- 4.2 [座標轉換與還原](lessons/04-coordinates.md)：框跟圖片一起移動
- 5 [多物件輸出與責任分配](lessons/05-assignment.md)：哪個預測負責哪個物件
- 6.1 [人工框解碼與 NMS](lessons/06-decode-nms.md)：少一個框不一定更好
- 6.2 [人工框評估與 AP50](lessons/06-evaluation.md)：把預測逐筆算成證據
- 7.1 [Grid MiniYOLO 資料](lessons/07-data.md)：先讓資料可以被檢查
- 7.2 [Grid MiniYOLO targets](lessons/07-targets.md)：把框變成訓練目標
- 7.3 [Grid MiniYOLO loss](lessons/07-loss.md)：loss 必須能手算
- 7.4 [三步訓練與診斷](lessons/07-training.md)：先只更新 3 次參數，確認程式能跑；同頁再看 160 步（160 次參數更新）的學習實驗
- 7.5 [完整圖片推論](lessons/07-inference.md)：把輸出接回圖片
- 7.6 [獨立資料評估](lessons/07-heldout.md)：用沒參與訓練的圖片評估模型
- 8.1 [自己的圖片推論](lessons/08-own-images.md)：先把座標換算和類別順序弄對
- 8.2 [自己的類別與資料](lessons/08-own-data.md)：類別、標註與來源切分（同一個來源的圖片，只能整批分到訓練、驗證或測試其中一組）

## C. YOLO 的演化機制（第 9–16 章）

各章對照的版本：第 9 章 YOLOv2、第 10 章 YOLOv3、第 11 章 YOLOv4／v5、第 12 章 YOLOv8、第 13 章 YOLOv10、第 14 章 YOLO11、第 15 章 YOLOv12（15.1 節先講 attention 的基本算法）、第 16 章 YOLO26。

- 9.1 [YOLOv2 anchor 與框參數化](lessons/09-anchors.md)：anchor 是尺寸起點
- 9.2 [尺寸聚類](lessons/09-anchor-clustering.md)：先驗（anchor 的預設尺寸）由哪一份資料決定
- 10 [YOLOv3 多尺度](lessons/10-multiscale.md)：同一個 pixel 框看兩種尺度
- 11.1 [CSP](lessons/11-csp.md)（Cross Stage Partial）：分一部分通道走較短的路
- 11.2 [特徵融合](lessons/11-fusion.md)：把深層資訊送回細網格
- 11.3 [圖與框同步增強](lessons/11-augmentation.md)（資料增強：訓練時把圖片翻轉、裁切等，當成新的訓練樣本）：畫素怎麼變，框就怎麼變
- 11.4 [IoU 類 loss](lessons/11-iou-loss.md)：沒有重疊時還能往哪裡移
- 12.1 [Anchor-free](lessons/12-anchor-free.md)：從候選點（可以各自輸出一個框的位置）量出到框四條邊的距離
- 12.2 [Decoupled head](lessons/12-decoupled-head.md)（head：模型最後把特徵轉成預測的部分）：分類和定位在哪裡分工
- 12.3 [Sample assignment](lessons/12-assignment.md)：哪個候選值得被教
- 12.4 [DFL](lessons/12-dfl.md)（Distribution Focal Loss）：把一條邊距離學成分佈
- 13.1 [YOLOv10 dual assignment](lessons/13-dual-assignment.md)：訓練時多教，推論時少重複
- 13.2 [NMS-free 推論](lessons/13-nms-free.md)：拿掉 NMS 前，重複候選學會了什麼
- 14 [YOLO11 特徵模組](lessons/14-feature-module.md)：拆路徑、保留中間成果、再融合
- 15.1 [Feature map 到 attention](lessons/15-attention-bridge.md)：四個位置怎麼互相讀取
- 15.2 [YOLOv12 Area Attention](lessons/15-area-attention.md)：互動範圍是一筆預算
- 16.1 [YOLO26 DFL-free](lessons/16-dfl-free.md)：移除 bins（DFL 替一條邊的距離準備的 0、1、2、… 格這些整數刻度），仍要把框學好
- 16.2 [YOLO26 推論 head](lessons/16-inference-head.md)：把訓練用的分支從部署模型（實際交付使用時的模型）真正拿掉
- 16.3 [YOLO26 訓練補強](lessons/16-training.md)：訓練中把 loss 的比重從一對多分支逐步移到推論用的一對一分支（Progressive Loss）；另外說明照顧小物件的候選分配（STAL）與 MuSGD 優化器，只對第一項做實驗

## D. 整合與應用（第 17–20 章）

- 17 [靜態偵測結業任務](lessons/17-capstone.md)（靜態＝單張圖片，不是影片）：用一次有理由的改動交付結果
- 18 [影片串流](lessons/18-video.md)：處理每一幀，並分清 FPS（frames per second，每秒幀數）與延遲
- 19 [簡易 tracking](lessons/19-tracking.md)（追蹤：在影片裡跨畫面維持同一個物件的編號）：框很準，ID 仍可能換人
- 20 [ONNX／TensorRT](lessons/20-deployment.md)（ONNX：Open Neural Network Exchange，一種通用的模型格式；TensorRT：NVIDIA 的推論加速工具）：匯出後先證明同一個輸入得到同一個結果

## E. 從 attention 分岔：ViT／DINO（第 21–23 章，選讀）

第 21 章讓紅／藍矩形圖走過「切成 patch → 交換資訊 → CLS 分類」；第 22 章再問，訓練時不看類別答案，如何學影像特徵？這裡的 DINO 指 2021 年的自監督方法，與同名的 DETR 系列偵測器不同。第 23 章區分後續版本，再把逐 patch 特徵接回單物件定位。21、22 章不需要先會偵測的 assignment 或 AP；23.2 會用到 [4.1 的框與定位](lessons/04-localization.md)及 [6.2 的 IoU](lessons/06-evaluation.md)。

- 21.1 [圖片切成 patch](lessons/21-patches.md)：追蹤每塊圖的順序、位置與 CLS
- 21.2 [Patch 如何交換資訊](lessons/21-attention.md)：沿用 15.1 的 Q／K／V，接上多頭 attention
- 21.3 [組成 tiny ViT](lessons/21-transformer.md)：LayerNorm、兩次殘差相加與 MLP 如何組成一個 block
- 21.4 [訓練、評估與恢復 ViT](lessons/21-training.md)：真正更新全部參數，用獨立圖評估，並恢復 optimizer 與 RNG
- 22.1 [沒有標籤的兩種視圖](lessons/22-views.md)：同一張圖的不同裁切如何成為訓練材料
- 22.2 [一致但沒有資訊：collapse](lessons/22-collapse.md)：兩個輸出一樣，為什麼還可能沒學到可用特徵
- 22.3 [從零實作 DINO 核心](lessons/22-distillation.md)：teacher、student、停止梯度與 teacher 更新各負責什麼
- 22.4 [特徵有沒有用：近鄰與 linear probe](lessons/22-features.md)：凍結特徵，再與同起點的隨機特徵比較
- 23.1 [DINO 版本與官方預訓練特徵](lessons/23-dino-versions.md)：分清版本來源與可選的權重下載操作
- 23.2 [凍結 patch 特徵接回定位](lessons/23-detection-bridge.md)：保留 patch 順序，讓小 head 學一個框與類別

預設實驗只用 CPU 和固定 seed 的合成資料，不下載權重；官方預訓練操作另選。小 ViT 與 DINO 核心實作縮小了原版規模，這次沒有新增 GPU 訓練或 DINOv3 實作。支線的分類及定位數字只支持受控紅／藍矩形任務；在這次簡單分類中，隨機特徵與自監督特徵都得到 64／64，不能說自監督比較好。實驗程式與 notebook 固定到 `lessons-v0.6.1`，網頁敘述與圖解持續修訂；公開入口驗證以保存紀錄中的 tag 與結果為準。

## 遇到卡住的地方

以下建議在讀到第 7 章之後最有用：

1. 挑同一個物件，對照三樣東西：標註的框、轉成模型要學的數值（target）、從模型輸出換回來的框（解碼結果）。看座標與 tensor 的形狀有沒有對上。
2. 不要只看一個總 loss。總 loss 通常由幾項誤差組成（第 7 章是框、有沒有物件、類別三項），要分開看。
3. 算式看不懂時，先把正在讀的那一節例子裡的具體數字代進去算一遍，觀察形狀（shape）與單位。
4. 程式報錯停下時，先保存完整的錯誤訊息，再看最後一行（錯誤的種類與說明）。例如在自己的電腦上出現 `ModuleNotFoundError: No module named 'miniyolo'`，表示執行時沒加 `PYTHONPATH=.`（見第 0 章〈在自己的電腦執行〉）；做練習時出現 `AssertionError`，多半是斷言還在核對原題的答案（見第 0 章練習 1）。程式跑得完、但 loss 不降或結果不對時，照[訓練診斷](lessons/02-diagnostics.md)的順序排查。

哪些結果有實測、哪些事沒有驗證，見[驗證範圍](status.md)。

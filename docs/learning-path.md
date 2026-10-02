# 閱讀路線：先追一個問題，再加下一個機制

從第一節依序讀即可。你會走過小 CNN、ResNet、單物件定位、可訓練的 grid detector，再用小實驗理解 YOLO 的不同設計。每個網頁都附獨立 Colab；想先閱讀時不用建立 runtime。

先有 basic Python／PyTorch／CNN 就夠。B 表示 batch 圖片數、C 表示 channel／class 數（各節會明說是哪種），H/W 表示高度／寬度。數學不需要熟背；公式會連到數字與程式。

**先取得一個成果：**第 0–7 章完成靜態偵測管線；第 8 章接自己的圖片。**完整演化：**再讀 9–16 章。**應用選修：**17 章整合，18–20 章可按需求挑讀。不是把所有歷史改動一口氣疊到同一個模型；版本節會說明共用基線與本次單項實驗。

## A. 先讓 CNN 學得動

- [一次學習的超短暖身](lessons/00-warmup.md)
- [VGG 風格小 CNN](lessons/01-small-cnn.md)
- [訓練診斷](lessons/02-diagnostics.md)
- [ResNet identity shortcut](lessons/03-identity.md)
- [ResNet projection shortcut](lessons/03-projection.md)
- [Plain／residual 對照](lessons/03-comparison.md)

## B. 分類走到可評估的偵測

- [單物件分類與定位](lessons/04-localization.md)
- [座標轉換與還原](lessons/04-coordinates.md)
- [多物件輸出與責任分配](lessons/05-assignment.md)
- [人工框解碼與 NMS](lessons/06-decode-nms.md)
- [人工框評估與 AP50](lessons/06-evaluation.md)
- [Grid MiniYOLO 資料](lessons/07-data.md)
- [Grid MiniYOLO targets](lessons/07-targets.md)
- [Grid MiniYOLO loss](lessons/07-loss.md)
- [少量 overfit 與偵測診斷](lessons/07-training.md)
- [完整圖片推論](lessons/07-inference.md)
- [獨立資料評估](lessons/07-heldout.md)
- [自己的圖片推論](lessons/08-own-images.md)
- [自己的類別與資料](lessons/08-own-data.md)

## C. YOLO 的演化機制

- [YOLOv2 anchor 與框參數化](lessons/09-anchors.md)
- [尺寸聚類](lessons/09-anchor-clustering.md)
- [YOLOv3 多尺度](lessons/10-multiscale.md)
- [CSP](lessons/11-csp.md)
- [特徵融合](lessons/11-fusion.md)
- [圖與框同步增強](lessons/11-augmentation.md)
- [IoU 類 loss](lessons/11-iou-loss.md)
- [Anchor-free](lessons/12-anchor-free.md)
- [Decoupled head](lessons/12-decoupled-head.md)
- [Sample assignment](lessons/12-assignment.md)
- [DFL](lessons/12-dfl.md)
- [YOLOv10 dual assignment](lessons/13-dual-assignment.md)
- [NMS-free 推論](lessons/13-nms-free.md)
- [YOLO11 特徵模組](lessons/14-feature-module.md)
- [Feature map 到 attention](lessons/15-attention-bridge.md)
- [YOLOv12 Area Attention](lessons/15-area-attention.md)
- [YOLO26 DFL-free](lessons/16-dfl-free.md)
- [YOLO26 推論 head](lessons/16-inference-head.md)
- [YOLO26 訓練補強候選](lessons/16-training.md)

## D. 整合與應用

- [靜態偵測結業任務](lessons/17-capstone.md)
- [影片串流](lessons/18-video.md)
- [簡易 tracking](lessons/19-tracking.md)
- [ONNX／TensorRT](lessons/20-deployment.md)

## 遇到卡住的地方

先回到同一個物件的原圖框、target 和解碼結果，看座標與各軸有沒有一致；不要只看一個總 loss。算式不懂時先代入本節的小數值，觀察 shape 與單位。實驗失敗則保留原始錯誤，照[訓練診斷](lessons/02-diagnostics.md)的順序排查。

只需要複習一項術語時可看[快速查表](glossary.md)。哪些結果已實測、哪些仍待 GPU 或真實資料，見[驗證範圍](status.md)。

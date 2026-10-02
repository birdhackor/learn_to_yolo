# 從一個小 CNN，走到你看得懂的 YOLO

假如你會寫一點 PyTorch，卻常在偵測模型的 `target`、head、loss 與 NMS 之間迷路，這份教材就是從這些接縫開始。先讓簡單 CNN 做分類，再加 ResNet shortcut，最後把「整張圖一個答案」改成「這裡有物件，它的框在這裡」。每次只引入一個主要變化。

[從第一節開始](lessons/00-warmup.md){ .md-button .md-button--primary }
[檢視完整閱讀路線](learning-path.md){ .md-button }

## 這裡怎麼學

42 個小節都有完整網頁與獨立 Colab notebook。網頁包含直覺、數值例子、圖、關鍵程式、可核對結果與練習答案；只看網頁也能學習。想動手時再點該節的 Colab 按鈕，不需要先執行上一節。你不必先熟記微積分，公式旁會說明符號、shape 與單位。

全書的主角是 **MiniYOLO**：為理解機制而設計的小型 PyTorch 模型。早期版本研究 grid、anchor、多尺度；現代分支研究 anchor-free、DFL、配對、attention 與推論 head。YOLO 是不同團隊的分支家族，版本號不代表每次改動都划算。我們每次都問：原來哪裡卡住、改了什麼、得到什麼，以及付出什麼。

![同一個物件走過資料、監督、loss、解碼與評估](assets/diagrams/object-journey.svg)

上圖的紅框中心是 `(16,20)` 畫素。在 64×64 圖片的 4×4 grid 中，每格 16 畫素，這個中心交給第 1 列、第 1 欄（從 0 起算）。先把一個物件追到底，比一開始背完整架構容易；[target](lessons/07-targets.md)、[loss](lessons/07-loss.md) 與 [推論](lessons/07-inference.md) 會逐步拆解它。

## 先了解這版的證據

人工框與小張量實驗用已知答案驗證機制；CPU 範例檢查 forward、backward 和參數更新。另有小型合成資料的短訓練，作為「這條偵測管線確實能學動」的早期證據。這些結果不代表真實照片的辨識品質，也不拿來替原版 YOLO 排名。

本版不要求 GPU，也不要求下載大型資料。真實資料的長訓練、不同架構的 AP 對照與 GPU／TensorRT 效能，會在安排硬體後補上；[驗證範圍與後續實驗](status.md) 列出已做與待做的專案。教材程式與 Colab 固定到同一個 release tag，避免內容跟程式各用一版。

## 已備妥的資料與操作

- [資料來源、授權與下載](preparation/data.md)：Fashion-MNIST 已查覈並發布至 LFS；偵測資料的使用與再散佈條件分開列出。
- [GitHub Pages／Colab／LFS 操作](preparation/publish.md)：維護網站與下載資料的步驟。
- [公開課程研究](planning/course-research.md)與[讀者回饋](planning/feedback.md)：這份課綱吸收了哪些安排。
- [Colab 環境檢查](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/main/notebooks/00_environment_check.ipynb)：只檢視 Python、PyTorch 與 GPU 資訊。

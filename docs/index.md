# Learn to YOLO：前置準備

這個專案規劃用小型 PyTorch 模型，從 CNN、ResNet、圖片定位到 YOLO 演化，研究每項變化的動機、好處與代價。目前提供大綱、資料規劃與基礎配置，正式教材尚未開始。

- [教學大綱](planning/outline.md)：閱讀順序、章節目標與完成條件。
- [資料規劃](preparation/data.md)：可用來源、規模、標註、授權與下載入口。
- [發布操作步驟](preparation/publish.md)：GitHub Pages、Colab 與 Git LFS。
- [內容規格](preparation/architecture.md)：每小節如何同時供純閱讀與 Colab 執行。

## 先確認 Colab 環境

[在 Colab 開啟環境檢查](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/main/notebooks/00_environment_check.ipynb){ .md-button }

這份 notebook 只顯示 Python／PyTorch／GPU 是否可用，不訓練模型。notebook 已推送至 GitHub。免費 Colab 可用 CPU；GPU 型號、額度與執行時間依當時分配，無法預先保證。

未來各小節的網頁會包含完整說明、圖與已保存的結果；Colab 用於自行重做與修改實驗。讀者只看網頁也能學習，不必先連到 runtime。

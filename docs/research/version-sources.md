# 現代版本機制的原始來源

本頁列出第 12–16 章對照的 YOLO 各版本官方程式碼、文件與論文：程式碼固定在哪個 commit、核對了哪些內容，以及教材採用到什麼範圍。

查核日期：2026-10-02。本頁的程式碼、設定檔與官方說明文件連結都固定到特定 commit；若連到 main，讀者日後打開時可能看到另一份程式結構。第 12–16 章的實驗都是小型教學實驗；表中的版本名稱說明機制出自哪裡，不代表本書重現了那個版本。

手機上可左右滑動表格，查看完整欄位。

| 來源與固定版本 | 核對內容 | 教材採用範圍 |
| --- | --- | --- |
| [Ultralytics repository 441632cdfd19e22e60a4b1b1999d46326ca51ec4](https://github.com/ultralytics/ultralytics/tree/441632cdfd19e22e60a4b1b1999d46326ca51ec4) | `nn/modules/head.py` 的分類／回歸分支、DFL／identity、雙 head、detach、推論時的輸出選擇、`postprocess`、`fuse`；`utils/tal.py` 的候選點、距離解碼與 task-aligned 選擇（TAL）；`utils/loss.py` 的 `DFLoss` | 第 12 章只示範各機制。固定版本的 TAL 程式含有 YOLO26 的改動（例如小物件候選擴張），本書不把這些改動當成原始 YOLOv8 的設定 |
| [YOLOv10 作者 repository 453c6e38a51e9d1d5a2aa5fb7f1014a711913397](https://github.com/THU-MIG/yolov10/tree/453c6e38a51e9d1d5a2aa5fb7f1014a711913397) | 一對多／一對一的 consistent dual assignments（一致的雙重分配）、head 特徵的 detach、推論只用一對一 head 且不以 IoU 挑框；[論文](https://arxiv.org/abs/2405.14458) | 13.1 節以列舉求全域最佳的一對一配對，只是通用的教學工具；YOLOv10 不這樣配對，也不用匈牙利演算法（Hungarian algorithm）。13.2 節固定框位置的分數實驗不是完整的偵測器 |
| [YOLO11 官方配置](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/cfg/models/11/yolo11.yaml)（Ultralytics 同一 commit） | C3k2 用在哪些層；同一 commit 的 `block.py` 中 C3k2／C2f／Bottleneck 的定義 | 第 14 章採用受 C3k2 啟發的 split／aggregate 結構（`SplitAggregate`），省去 BN 與可選的 C3k 內部路徑；不把結果歸因到整版 YOLO11 的精度 |
| [YOLOv12 作者 repository 2abab7153a065fb2925e8088e9ca2b19016ab7d6](https://github.com/sunsmarterjie/yolov12/tree/2abab7153a065fb2925e8088e9ca2b19016ab7d6) | `nn/modules/block.py` 中 AAttn 的 Q／K 與 V 投影、把連續 token 分成 area、不用 FlashAttention 時的 CPU 算法、逐 channel 的位置卷積（depthwise）；[論文](https://arxiv.org/abs/2502.12524) | 第 15 章先講 full attention，再用同一份 QKV 權重對照 area attention；不宣稱 FlashAttention 的速度，也不宣稱完整 YOLOv12 的 AP |
| [YOLO26 官方配置](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/cfg/models/26/yolo26.yaml)（Ultralytics 同一 commit）與[原論文](https://arxiv.org/abs/2606.03748) | `reg_max: 1` 與 `end2end: True`；同一 commit 中 `BboxLoss` 的 CIoU 與正規化 L1、`E2ELoss` 的一對多／一對一（many／one）loss 權重 0.8／0.2→0.1／0.9、TAL 的小物件候選擴張（STAL）與 `topk2`、`optim/muon.py` 的 MuSGD | 第 16 章分開驗證直接距離（16.1 節）、NMS-free 模式下移除輔助 head（16.2 節）與 Progressive Loss 的權重排程（16.3 節）；STAL 只算一個例子的資格遮罩；MuSGD 只做說明，不執行未經驗證的簡化版 |

[YOLO26 訓練配方（recipe）](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/docs/en/guides/yolo26-training-recipe.md)寫明：官方發布的模型先在 Objects365 預訓練、再用 COCO 微調，部分內部超參數只能在實驗用的 Git 分支上設定。本書各節的實驗都只用 CPU、不使用預訓練權重，偵測實驗也只用合成圖或小型自製資料，所以結果不能和官方 checkpoint 的數字做公平的性能比較。第 12–16 章各節都在頁內寫明該節實驗驗證了什麼、沒有驗證什麼。

YOLO26 論文和固定 commit 的程式碼，在推論預設與 STAL 門檻上並不一致。本書逐項標明依據的是論文還是程式碼，不把兩者混在一起：

- **推論預設**：論文（arXiv 第 1 版）§3.2.1 把 one-to-one head 標為預設（default）；固定 commit 的預測函式 `model.predict()` 卻預設走 many 加 NMS，要設 `nms=False` 才走 NMS-free 的 one 分支（[16.2 節](../lessons/16-inference-head.md)）。
- **STAL 門檻**：論文的 STAL 公式 (5) 是邊長小於 8 才替換成 16；固定程式碼則是小於 16 就替換成 16（[16.3 節](../lessons/16-training.md)）。16.3 節 2×2 畫素的例子同時滿足兩個條件。

DFL 的相鄰距離 bin 監督另對照 [Generalized Focal Loss 原論文](https://arxiv.org/abs/2006.04388)（[12.4 節](../lessons/12-dfl.md)）；一般 attention 的算法引用 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)（[15.1 節](../lessons/15-attention-bridge.md)）。

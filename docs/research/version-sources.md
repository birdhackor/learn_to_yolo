# 現代版本機制的原始來源

查覈日：2026-10-02。本表記錄第12–16章作者實際閱讀的原始source與官方文件。教材的小模型皆是MiniYOLO教學實驗；版本名稱表示研究動機，不代表完整復現。固定commit可避免讀者後來開啟main，卻看到另一份結構。

| 分支與固定版本 | 核對內容 | 教材採用範圍 |
| --- | --- | --- |
| [Ultralytics 441632cdfd19e22e60a4b1b1999d46326ca51ec4](https://github.com/ultralytics/ultralytics/tree/441632cdfd19e22e60a4b1b1999d46326ca51ec4) | `nn/modules/head.py`的分類／回歸分支、DFL/identity、雙head、detach、推論選擇、postprocess、fuse；`utils/tal.py`候選點、距離解碼、task-aligned選擇；`utils/loss.py`DFLoss | 第12章只示範各機制，不把現行TAL包含的YOLO26改動追溯成原始YOLOv8設定 |
| [YOLOv10作者 453c6e38a51e9d1d5a2aa5fb7f1014a711913397](https://github.com/THU-MIG/yolov10/tree/453c6e38a51e9d1d5a2aa5fb7f1014a711913397) | 一對多／一對一consistent dual assignments、head特徵detach、推論one-head與非IoU的選擇；[論文](https://arxiv.org/abs/2405.14458) | 第13章的枚舉全域匹配是通用一對一教學工具，不宣稱官方採Hungarian；固定框的分數實驗不是完整detector |
| [YOLO11官方配置](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/cfg/models/11/yolo11.yaml) | C3k2採用位置；同commit `block.py`內C3k2/C2f/Bottleneck | 第14章採split/aggregate啟發結構，省去BN與可選C3k內部路徑，不歸因整版精度 |
| [YOLOv12作者 2abab7153a065fb2925e8088e9ca2b19016ab7d6](https://github.com/sunsmarterjie/yolov12/tree/2abab7153a065fb2925e8088e9ca2b19016ab7d6) | `nn/modules/block.py` AAttn的QK/V投影、連續token分area、CPU fallback、位置depthwise卷積；[論文](https://arxiv.org/abs/2502.12524) | 第15章先full attention再同權重area對照，沒有FlashAttention速度或完整YOLOv12 AP宣稱 |
| [YOLO26官方配置](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/cfg/models/26/yolo26.yaml)與[原論文](https://arxiv.org/abs/2606.03748) | `reg_max:1`與`end2end:True`；同commit `BboxLoss`的CIoU與正規化L1、`E2ELoss` .8/.2→.1/.9；TAL小物件候選擴張／topk2；`optim/muon.py`的MuSGD | 第16章直接距離、NMS-free模式下移除輔助head與progressive權重分開驗證；STAL只算資格mask；MuSGD不執行未驗證縮版 |

[YOLO26訓練recipe](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/docs/en/guides/yolo26-training-recipe.md)明述Objects365預訓練、COCO微調及部分內部參數需要實驗分支。本書從零CPU的小矩形資料不能與其checkpoint數字作公平性能比較。各頁面保留本節已驗證範圍和未驗證條件。

YOLO26論文v1 §3.2.1把one-to-one稱為default，但本書固定commit的高階預測API預設many加NMS，須選`nms=False`才走NMS-free的one分支。論文STAL公式(5)使用邊長小於8→替換16；固定程式碼則小於16→替換16。本書逐項標明依論文還是程式碼說明，不把兩者的預設與門檻混寫；2×2畫素的教學例同時满足兩個條件。

對DFL另核對[Generalized Focal Loss原論文](https://arxiv.org/abs/2006.04388)的相鄰距離bin監督；對一般attention引用[Attention Is All You Need](https://arxiv.org/abs/1706.03762)。第20章ONNX／ORT已在CPU實際匯出和執行，另以單張L4建立並執行TensorRT engine；[部署章](../lessons/20-deployment.md)分別列出兩次實測範圍與精度限制。

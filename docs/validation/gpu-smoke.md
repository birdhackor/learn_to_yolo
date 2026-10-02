# 小規模 GPU 與 checkpoint 實測

這個入口驗證本專案的 `GridDetector`、`ShapeDataset`、target、loss 與真實 optimizer 更新。它共用 `miniyolo.train.optimizer_step`，使用 8 張固定合成矩形、64×64 RGB、width=8、兩類、4×4 grid；不是另寫一個無關的 PyTorch 探測模型，也不衡量照片辨識品質。

## 執行限制與路徑

GitHub Actions 的 **GPU training and checkpoint smoke test** 僅接受 `workflow_dispatch`。一般 push、PR 不會啟動 GPU。工作流程逐一呼叫 Modal：

1. 單張 L4 跑不中斷的 40 步對照，再從相同 seed 跑 20 步並儲存中途 checkpoint。
2. 另一個 CPU container 重新讀取已 commit 的專案 Volume，校驗 checkpoint 與設定檔的 SHA-256；HF 設定具備時，傳到私有 checkpoint repo，依上傳 commit 下載並再次校驗。
3. 新的單次使用 L4 container 從下載檔（HF 未設定時從 Volume 檔）續訓至第 40 步，對照模型、Adam 狀態、StepLR、loss 軌跡與 RNG。

合計 80 次 optimizer 更新。GPU 函式固定 `gpu="L4"`、`max_containers=1`、`timeout=600`、`retries=0`、`single_use_containers=True`。Actions 以專案 concurrency group 排隊，呼叫採同步順序；結束時查本次 Modal app 為 stopped 且 tasks=0。程式不修改帳號費用上限。

Volume 預設 `learn-to-yolo-checkpoints`，採 `projects/learn-to-yolo/runs/github-<run_id>-<attempt>-<commit>/` 專用路徑。HF 也使用相同專案與工作路徑，避免覆寫其他專案；不發布公開模型。

## 沿用現有設定

- `MODAL_TOKEN_ID` 優先取 Repository variable，亦支援 Repository secret；`MODAL_TOKEN_SECRET` 取 secret。
- `MODAL_ENVIRONMENT` 取 variable，預設 `main`。
- HF token 僅注入 Modal CPU container；先選指定的 `MODAL_HF_SECRET_NAME`（也支援 `MODAL_SECRET_NAME`／`HF_SECRET_NAME`），再找 `codex_cloud` 或其他現有 Secret 的 `HF_TOKEN`。只回傳 key 是否存在，不回傳值。
- `HF_CHECKPOINT_REPO`／`HF_RELEASE_REPO` 取 variables。本次不用 release repo。checkpoint repo 未指定但 HF token 可用時，建立 token 所屬帳號的私有 `learn-to-yolo-checkpoints` repo；已存在的公開 repo 會拒絕上傳。
- `MODAL_CHECKPOINT_VOLUME`／`MODAL_VOLUME_NAME` 可指定既有專案 Volume。

Actions 使用 `requirements-modal.txt`，不安裝 PyTorch。GPU image 使用從現有 `requirements-model.txt` 產生的完整 `requirements-gpu.lock`，固定 `torch==2.9.1+cu128` 並驗證套件 hashes；其他模型依賴維持原有版本。GPU 回傳 JSON 字串，`torch.__version__` 明確轉成 `str`。

## checkpoint 與比較

保留原本的 `model_state_dict`、`optimizer_state_dict`、`config`、`class_names`、`steps_completed`，再加入格式版本、scheduler、Python／NumPy／PyTorch CPU／CUDA RNG。可用 `torch.load(weights_only=True)`，既有圖片推論入口仍能讀取。舊 checkpoint 可做推論，但缺少 RNG 的檔案不能宣稱精確續訓。

固定 seed，關閉 TF32 與 cuDNN benchmark，要求可用的 deterministic kernels。原模型的 AdaptiveAvgPool2d CUDA backward 若不提供 deterministic 實作，會保留模型並記錄警告。模型、optimizer 與浮點 loss 容許 `rtol=1e-5, atol=1e-6`；scheduler 與 RNG 狀態及 probe 值則必須一致。這個檢查不能保證不同 GPU、不同 PyTorch 版本都逐位相同。

每步檢查有限且非零的 gradient norm，並核對 GPU 上的模型、圖片和 targets、權重變化、Adam step 與固定資料 loss 下降。保存 preprocessing 的 RGB／NCHW／float32／[0,1] 契約、資料 seed／fingerprint、類別順序與 code commit；視覺模型沒有文字 tokenizer。

## 計時與驗證結果

訓練計時在頭尾同步 CUDA，包含前向、反向、optimizer／scheduler、梯度校驗、RNG probe、純量記錄與 console logging；不含模型／資料建構、存檔、Volume commit、container 啟動、image build 或傳輸。存檔、commit、HF 上下載另記時間。client wall time 包含啟動與傳輸；這次小測試不是正式吞吐量或 GPU 效能基準。

## 本次實際結果

[Actions 工作 37032967155](https://github.com/birdhackor/learn_to_yolo/actions/runs/37032967155)與其中的 **gpu-smoke-37032967155-1** artifact 均已完成。測試程式 commit 是 `eaa0cdf073bec49b09402bfbed40db9b0648b425`。本機前置測試 24／24 通過，原訓練 CLI 的新版 checkpoint 亦可由既有圖片推論入口讀取。

| 項目 | 實測結果 |
| --- | --- |
| GPU／PyTorch | 單張 NVIDIA L4；2.9.1+cu128，CUDA build 12.8 |
| 模型 | 原專案 GridDetector，15,511 個參數；模型、圖片與 targets 均在 CUDA |
| 訓練量 | 8 張固定合成圖；40 步不中斷，另 20+20 步恢復；共 80 次更新 |
| 固定資料 loss | 1.04003835 → 0.06170251；前 5 步均值 0.92157216、末 5 步均值 0.06433246 |
| 梯度／權重 | 80 步梯度均有限且非零，L2 範圍 0.152426–1.594808；權重改變 L2 約 8.388245 |
| Volume | 明確 commit 後，另一 CPU container 讀取 6 個檔案，全部 SHA-256 相符 |
| HF | 沿用 Modal Secret `codex_cloud` 的 HF_TOKEN；checkpoint 與設定／來源／code commit 上傳私有 repo |
| HF 下載 | 固定上傳 commit 重新下載中途 checkpoint，SHA-256 與 Volume 相符；實際從此下載檔續訓 |
| 恢復 | 新 L4 container 從第 20 步續訓到 40；Adam state 也是 40，LR 恢復為 .008、結束為 .0064 |
| 對照 | 模型、optimizer 與 loss／gradient 軌跡的最大差異皆 0；scheduler、Python／NumPy／CPU／CUDA RNG 與 probe 相符 |
| 工作結束 | Modal app `ap-IvP1WCRTduPs9ZTJKY8aG1`，API 狀態 stopped，running tasks=0 |

HF 私有 repo 是 `birdhackor/learn-to-yolo-checkpoints`，路徑為 `projects/learn-to-yolo/runs/github-37032967155-1-eaa0cdf073be`。上傳 commit：`24f89355835dd12fe77451911c813e98d310efdc`。中途 checkpoint SHA-256：

```text
15550f659bf0dc18b3fb85e2cc99d4f6a87f83eea997911637ad265aa8f80824
```

Volume 使用相同的專案工作路徑，沒有覆寫其他專案檔案。完整純 JSON 結果見 [repo 驗證紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/gpu-smoke-37032967155.json)，Actions／artifact 識別與本機檢查見 [證據索引](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/gpu-smoke-evidence.json)。權重只存於 Modal Volume／私有 HF repo，不進普通 Git 或網站。

### 本次時間

| 範圍 | 秒 |
| --- | ---: |
| 不中斷 40 步，頭尾 CUDA 同步 | 1.556944 |
| 中斷前 20 步，頭尾 CUDA 同步 | 0.113483 |
| 新 container 恢復後 20 步，頭尾 CUDA 同步 | 1.958308 |
| baseline／中途 checkpoint 存檔 | 0.017145／0.012785 |
| 第一輪 checkpoint 與設定的 Volume commit | 3.454562 |
| 恢復後 checkpoint 存檔／首次 commit | 0.014411／0.916128 |
| HF 上傳 checkpoint＋設定／下載 checkpoint | 2.457110／1.875531 |
| GPU producer／resume 呼叫 wall time | 17.441722／15.346392 |
| client 整段 wall time | 172.790510 |

前面三個數字是含檢查與記錄的訓練計時，會包含第一次使用某些 kernels／optimizer 的初始化成本；不中斷、暖機後及新 container 的時間不可直接當成速度排名。Volume 欄位是被明確計時的首次資料 commit，後續保存 JSON 報告的 commit 包含在呼叫 wall time 中。client 整段包含 image build、CPU probe／校驗、啟動與傳輸，後續 Actions 的停止確認與 artifact 上傳不在這個 client 計時內。

實際收到 AdaptiveAvgPool2d CUDA backward 缺少 deterministic implementation 的警告，已保存於 JSON。本次仍量得對照差異為 0，不能因此保證其他硬體／版本逐位一致。尚未做長訓練、真實照片品質、正式 throughput、混合精度或多 GPU 的 checkpoint 對照。本次checkpoint測試沒有執行TensorRT；後續獨立L4部署實測已完成，見[部署章的紀錄](../lessons/20-deployment.md)。這次已有設定全部足夠，不需要補 token 或改帳號費用上限。

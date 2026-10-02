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

每步檢查有限且非零的 gradient norm，並核对 GPU 上的模型、圖片和 targets、權重變化、Adam step 與固定資料 loss 下降。保存 preprocessing 的 RGB／NCHW／float32／[0,1] 契約、資料 seed／fingerprint、類別順序與 code commit；視覺模型沒有文字 tokenizer。

## 計時與驗證結果

訓練計時在頭尾同步 CUDA，包含前向、反向、optimizer／scheduler、梯度校驗、RNG probe、純量記錄與 console logging；不含模型／資料建構、存檔、Volume commit、container 啟動、image build 或傳輸。存檔、commit、HF 上下載另記時間。client wall time 包含啟動與傳輸；這次小測試不是正式吞吐量或 GPU 效能基準。

目前已完成本機 CPU 前置檢查；GPU、Volume 與 HF 的結論待 Actions 實際工作完成後填入。完整結果將保存為 `gpu-smoke-<run_id>-<attempt>` Actions artifact，整理出的 JSON 存在 `artifacts/checks/`。權重只存於 Modal Volume／私有 HF repo，不進普通 Git 或網站。

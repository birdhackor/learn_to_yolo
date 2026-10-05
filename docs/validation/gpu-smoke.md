# 小規模 GPU 與 checkpoint 實測

這項測試在雲端的一張 NVIDIA L4 GPU 上檢查一件事：訓練中途存成 checkpoint（存檔），換一個全新的 container（容器：打包好程式與執行環境的獨立執行單位）讀回來接著練，結果是否和一次練完的訓練相同。訓練的是本教材的 `GridDetector`（第 7 章的偵測器），不是另寫的測試用模型；資料、target 與 loss 也都用教材本身的程式（`ShapeDataset`、`build_targets`、`grid_loss`）。每一步都呼叫 `miniyolo.train.optimizer_step` 真的更新參數，這也是訓練 CLI（命令列程式：在終端機打指令執行的程式，這裡指 `python -m miniyolo.train`）用的函式。資料是 8 張固定的合成矩形圖：64×64 RGB、兩類；模型 width=8、4×4 grid。

它只檢查訓練、存檔與續訓的流程，不衡量模型辨識照片的品質，也沒有涵蓋長時間訓練、正式的吞吐量（throughput，每秒能處理多少資料）、混合精度（部分運算改用較低精度的數值格式）或多 GPU 的 checkpoint 對照。這項測試不執行 TensorRT（NVIDIA 的推論加速工具）；TensorRT 在 L4 上的核對是另一項實測，見[第 20 章的 L4 結果](../lessons/20-deployment.md#l4-results)。

## 執行限制與路徑

測試由 GitHub Actions（GitHub 的自動執行服務）的工作流程 **GPU training and checkpoint smoke test**（`.github/workflows/gpu-smoke.yml`）執行。它只能手動啟動（觸發條件只有 `workflow_dispatch`），一般的 push（上傳提交）或 PR（請求合併修改）不會啟動 GPU。工作流程在 Modal（租用雲端 GPU 的服務）上依序執行三段：

1. 在一張 L4 上先不中斷地練 40 步，存下結果當作對照；再從相同的 seed（亂數種子）重新練 20 步，每步的 loss 等數值都和對照的前 20 步比對，再存下中途的 checkpoint。這一段結束前，checkpoint 與 JSON 檔都已明確 commit 到專案的 Volume（Modal 的雲端儲存空間）；commit 是正式寫入，其他 container 才讀得到。
2. 另一個只用 CPU 的 container 重新讀取 Volume，核對兩個 checkpoint 與四個 JSON 檔（設定、前處理、資料來源、程式版本）的 SHA-256（由檔案內容算出的指紋，內容改一點就會不同）。有 Hugging Face（HF，存放與分享模型檔的網站）的 token（存取金鑰）時，再把中途 checkpoint 與這四個 JSON 檔上傳到私有的 checkpoint repo，然後依這次上傳產生的 commit（HF repo 的版本）把中途 checkpoint 下載回來（JSON 檔不下載），核對它的 SHA-256，通過後存進 Volume。
3. 一個新開、只用一次的 L4 container 讀取這份下載的 checkpoint（沒有 HF 下載檔時改讀 Volume 裡原本的中途 checkpoint），從第 20 步接著練到第 40 步，再和對照比較模型、Adam 的狀態、StepLR（學習率排程：每隔固定步數把學習率乘上固定比例）、每步的 loss 與梯度大小（L2 norm：把所有參數的梯度元素平方後加總，再開根號），以及 RNG（亂數產生器）。

三段合計 80 次 optimizer 更新。GPU 函式固定 `gpu="L4"`、`max_containers=1`（同時最多一個 container）、`startup_timeout=600`（container 啟動最多 600 秒）、`timeout=600`（啟動後每次呼叫的執行最多 600 秒）、`retries=0`（失敗不自動重試）、`single_use_containers=True`（每個 container 只處理一次呼叫）；整個 Actions 工作最多 25 分鐘。工作流程屬於 `learn-to-yolo-gpu-smoke` 這個 concurrency 群組（第 20 章的 `deployment-gpu.yml` 也在這一組）：同一時間只跑一個，進行中的不會被取消；排隊等待的最多只有一個，又有新的執行排進來時，原本在等待的那一個會被取消。`scripts/record_evidence.py --gpu` 等前一個工作流程跑完才啟動下一個，不受這點影響。三段依序呼叫，前一段結束才開始下一段。工作流程結尾的檢查步驟不論前面成功與否都會執行：確認這次執行在 Modal 上建立的 app（上面各段函式所屬的應用程式）已停止（stopped），而且執行中的任務數為 0（tasks=0）。連查三次仍未停止，它會停止這個 app（只停這次的，不動其他 app）再繼續查；最後仍確認不了，這一步就失敗。程式不修改帳號的費用上限。

Volume 預設是 `learn-to-yolo-checkpoints`。每次執行寫進自己專用的目錄 `projects/learn-to-yolo/runs/github-<run_id>-<attempt>-<commit>/`（`<commit>` 是程式 commit 的前 12 碼；這裡的 commit 是 Git 的版本編號）；目錄已存在時程式會停下，不覆寫。HF 上也用相同的專案與路徑，不會覆寫其他專案的檔案。checkpoint 只上傳到私有 repo，不發布公開模型。

## 需要的設定

工作流程從 repository 的 variables 與 secrets（GitHub 上存放設定值與加密金鑰的地方）讀取下列設定：

- `MODAL_TOKEN_ID` 優先取 Repository variable，也可以放在 Repository secret；`MODAL_TOKEN_SECRET` 取 secret。兩項缺一，程式就停下，不送出任何雲端工作。
- `MODAL_ENVIRONMENT` 取 variable，預設 `main`。
- HF token 只注入 Modal 上只用 CPU 的 container，不進 GPU container。程式依序在這些 Modal Secret（存在 Modal 上的金鑰設定）裡找 `HF_TOKEN`，用第一個找到的：variable `MODAL_HF_SECRET_NAME`（也可用 `MODAL_SECRET_NAME` 或 `HF_SECRET_NAME`）指定的 Secret、`codex_cloud`、Modal 環境裡其他的 Secret。無法列出 Modal 環境裡的 Secret 時，只試指定的那一個；沒有指定就試 `codex_cloud`。檢查只回傳 `HF_TOKEN` 是否存在，不回傳它的值。
- `HF_CHECKPOINT_REPO`（variable）指定要上傳的 HF repo；沒有指定時，用 token 所屬帳號的 `learn-to-yolo-checkpoints`。repo 不存在時，程式把它建成私有 repo；已存在而且是公開的，程式不上傳。沒有任何 Secret 含 `HF_TOKEN` 時，程式跳過 HF 的上傳與下載，這次執行仍可通過；有 token 卻遇到公開的 repo 或 HF 這一段出錯時，第 3 段照樣執行，但整次執行的狀態記為 partial（部分完成），不算通過。`HF_RELEASE_REPO`（variable）只在紀錄裡註明是否設定，這個工作流程不使用 release repo。
- `MODAL_CHECKPOINT_VOLUME`（或 `MODAL_VOLUME_NAME`）指定要用的 Volume，沒有設定時用 `learn-to-yolo-checkpoints`；Volume 不存在時會自動建立。

Actions 端執行 `scripts/modal_gpu_smoke.py`（這個檔案在 GitHub Actions 上呼叫 Modal，也定義了上面三段在 Modal 上執行的函式），只安裝 `requirements-modal.txt`，也就是 Modal 的用戶端套件，不安裝 PyTorch；工作流程會先確認 Actions 端沒有 PyTorch。GPU container 的 image（映像檔：預先裝好套件的執行環境）安裝 `requirements-gpu.lock`。它由 `requirements-gpu.in` 產生：`requirements-model.txt` 再加上 CUDA（NVIDIA 的 GPU 運算平台）版的 `torch==2.9.1+cu128`，連同它們依賴的套件全部固定版本，安裝時核對每個套件檔的 SHA-256。所以 GPU 上的 PyTorch 是 2.9.1+cu128（針對 CUDA 12.8 編譯的版本），其他模型依賴的版本與 `requirements-model.txt` 相同。GPU 函式把結果轉成 JSON 字串再傳回，`torch.__version__` 也先轉成一般字串，所以傳回 Actions 端的資料裡沒有 PyTorch 的物件。

## checkpoint 與比較

中途與最後的 checkpoint 都由 `miniyolo/checkpoint.py` 的 `save_checkpoint` 存成同一種格式：一個 dict（字典：用鍵名取值），含 `model_state_dict`（權重）、`optimizer_state_dict`（Adam 的狀態）、`config`、`class_names`、`steps_completed`（已完成的步數）、格式版本 `format_version`（值為 2）、scheduler 的狀態，以及 Python、NumPy、PyTorch CPU、CUDA 四種 RNG 的狀態。裡面只有 tensor、數字、字串、list、dict 這類單純的資料，所以可以用 `torch.load(weights_only=True)` 載入。訓練 CLI（`python -m miniyolo.train`）存的 checkpoint 也是這個格式，只是它不用 scheduler，這一項存成 `None`；圖片推論入口 `scripts/detect_image.py` 只需要其中的權重、`config` 與 `class_names`，兩者都能讀。沒有 `format_version`＝2 或沒有 RNG 狀態的 checkpoint 仍可用來推論，但續訓用的 `load_checkpoint` 會拒絕它。少了 RNG 狀態，續訓後的亂數序列就接不上；訓練若用到亂數（例如隨機打亂資料順序或隨機增強），結果也會和不中斷的訓練不同。

同一套中斷與續訓也能在 CPU 上執行：`tests/test_checkpoint.py`（`.venv-model/bin/python -m pytest tests/test_checkpoint.py`）用 CPU 跑同樣的流程並比對，也檢查 `scripts/detect_image.py` 讀得到這種 checkpoint，以及沒有 `format_version` 與 RNG 狀態的 checkpoint 仍能由推論入口載入、但 `load_checkpoint` 拒絕用它續訓。CPU 上的結果不能代替 GPU 的紀錄。

兩次訓練要能逐項比較，所以程式固定 seed，關閉 TF32（NVIDIA GPU 上較快、但精度較低的矩陣運算格式）與 cuDNN（NVIDIA 的神經網路運算函式庫）的 benchmark 模式（自動試出最快的演算法，每次挑的可能不同），並要求 PyTorch 使用 deterministic（同樣的輸入每次都得到同樣結果）的實作。GridDetector 裡的 AdaptiveAvgPool2d 在 CUDA 上的 backward（反向傳播）沒有 deterministic 實作；程式設定 `warn_only=True`，遇到這種運算只記錄警告，模型照常使用這一層。比較時，模型、optimizer，以及每步的 loss、梯度大小與學習率，允許 `rtol=1e-5, atol=1e-6` 以內的浮點誤差（相對誤差約十萬分之一、絕對誤差約百萬分之一）；scheduler、RNG 狀態與 RNG probe 都必須完全相同。RNG probe 是每步結束後從四種 RNG 各抽一個數，用來確認亂數序列接得上，不影響圖片或 loss。這個檢查不能保證不同 GPU、不同 PyTorch 版本都逐位相同。

每一步都檢查 loss 是有限值、每個參數都有梯度而且數值有限，梯度大小也不為 0。另外也核對：看得到的 GPU 只有一張，而且是 L4；模型、圖片與 targets 都在 GPU 上；訓練後權重確實改變；用這 8 張固定圖算的 loss 降到起始值的 80% 以下；續訓結束時，Adam 對每個參數記錄的更新次數（step）都是 40。checkpoint 旁邊另存幾個 JSON 檔，記錄前處理規格（RGB、float32、數值範圍 [0,1]、維度順序 NCHW，也就是 batch、channel、高、寬）、資料的 seed 與 fingerprint（由圖片與標註算出的 SHA-256）、類別順序與程式的 commit。這是影像模型，沒有語言模型那種要一起保存的 tokenizer（把文字切成小單位的工具）。

## 計時與驗證結果

下面的結果來自 GitHub Actions 的[執行 37217100342](https://github.com/birdhackor/learn_to_yolo/actions/runs/37217100342)。紀錄存在 repo 的 [`artifacts/checks/gpu-smoke.json`](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/gpu-smoke.json)，狀態為 passed（通過）；執行的程式 commit 是 `463d3f59fb8042f8edc07fcc18581fab2c6bb418`。

這份紀錄用 SHA-256 綁定 `miniyolo/gpu_smoke.py`（訓練、存檔與比對的程式），以及它直接或間接 import 的每個 repo 模組（列在紀錄的 `dependencies_sha256`）。`scripts/modal_gpu_smoke.py` 不在其中：GPU 函式外層的 Volume reload／commit、`HF_TOKEN` 的檢查，以及整個第 2 段（Volume 的 SHA-256 核對、HF 的上傳與下載）都寫在這個檔案裡，在 Modal 上執行；`requirements-gpu.lock` 與 `gpu-smoke.yml` 也不在其中。綁定的檔案都沒變，紀錄就一直有效；任何一個改了，紀錄才算過期，要用 `scripts/record_evidence.py --gpu <分支>` 重跑。它會在 GitHub Actions 上啟動這個工作流程，跑完後下載結果、存成新的紀錄，步驟見〈[發布與帳號設定](../preparation/publish.md)〉。只改了沒有綁定的檔案時，紀錄不會過期，`record_evidence.py --gpu` 也會跳過它。要重新驗證時：

1. 先把修改 commit 並推到 `<分支>`（工作流程跑的是推上去的程式），再手動啟動這個工作流程：`gh workflow run gpu-smoke.yml --ref <分支>`。
2. 用 `gh run list --workflow gpu-smoke.yml` 找到這次執行的編號；跑完後用 `gh run download <編號> --dir artifacts/runs/gpu-smoke-download`，把結果下載到不進 git 的資料夾。
3. 確認下載的 `result.json` 裡，`status` 是 `passed`、`modal_stop_confirmation.verified` 是 `true`，再把它存成 `artifacts/checks/gpu-smoke.json`。
4. 執行 `python3 scripts/validate_curriculum_evidence.py --scope gpu`，通過才 commit。

網站建置時，同一支檢查會確認紀錄對得上目前的程式、狀態為 passed，而且 Modal app 已確認停止。

### 驗證結果

| 項目 | 實測結果 |
| --- | --- |
| GPU／PyTorch | 單張 NVIDIA L4；PyTorch 2.9.1+cu128，CUDA build 12.8 |
| 模型 | 教材的 GridDetector，15,511 個參數；模型、圖片與 targets 都在 CUDA 上 |
| 訓練量 | 8 張固定合成圖；不中斷 40 步，另有中斷前 20 步與續訓 20 步；共 80 次更新 |
| 固定資料 loss | 訓練前 1.04003835 → 訓練後 0.06170251；不中斷的 40 步中，前 5 步的 loss 平均 0.92157216，最後 5 步平均 0.06433246 |
| 梯度／權重 | 80 步的梯度都是有限值，梯度大小都大於 0，範圍 0.152426–1.594808；權重改變量（L2 norm）約 8.388245 |
| Volume | 明確 commit 後，另一個 CPU container 讀取 6 個檔案，SHA-256 全部相符 |
| HF | 使用 Modal Secret `codex_cloud` 裡的 `HF_TOKEN`；中途 checkpoint 與設定、前處理、資料來源、程式版本四個 JSON 檔上傳到私有 repo |
| HF 下載 | 依上傳 commit 重新下載中途 checkpoint，SHA-256 與 Volume 裡的相符；續訓讀的就是這個下載檔 |
| 續訓 | 新的 L4 container 從第 20 步練到第 40 步；Adam 的 step 也是 40；學習率恢復為 0.008（StepLR 每 20 步乘 0.8），結束時為 0.0064 |
| 對照 | 模型、optimizer，以及每步 loss、梯度大小與學習率的最大差異都是 0；scheduler、Python／NumPy／CPU／CUDA RNG 與 probe 都相同 |
| 工作結束 | Modal app `ap-38NEukrgml6Zn6I6vXRoiM` 狀態 stopped，執行中任務數 0 |

兩個 GPU 階段都收到 AdaptiveAvgPool2d 的 CUDA backward 沒有 deterministic 實作的警告，原文保存在紀錄裡。這次執行的對照差異仍是 0，但不能因此保證其他硬體或版本也逐位相同。

HF 私有 repo 是 `birdhackor/learn-to-yolo-checkpoints`，路徑是 `projects/learn-to-yolo/runs/github-37217100342-1-463d3f59fb80`，上傳 commit 是 `79ece12ba0f45c3a7a644dc353515c9514e0e02f`。中途 checkpoint 的 SHA-256：

```text
15550f659bf0dc18b3fb85e2cc99d4f6a87f83eea997911637ad265aa8f80824
```

Volume 也用同一個專案工作路徑，沒有覆寫其他專案的檔案。權重只存在 Modal Volume 與私有 HF repo，不放進 Git repo，也不放上網站。

### 各段時間

訓練時間在開始與結束時都先同步 CUDA（等 GPU 把排進去的運算做完），量到的才是實際完成的工作。它包含前向、反向、optimizer 與 scheduler 更新、梯度檢查、RNG probe、記錄數值與印出 log；不含建立模型與資料、存檔、Volume commit、container 啟動、image build 與網路傳輸。存檔、commit 與 HF 上傳下載各自另外計時。wall time（實際經過的時間）則包含啟動與傳輸。這項小測試不是正式的吞吐量或 GPU 效能基準。

| 範圍 | 秒 |
| --- | ---: |
| 第 1 段：不中斷 40 步的訓練 | 1.494028 |
| 第 1 段：中斷前 20 步的訓練 | 0.140341 |
| 第 3 段：新 container 續訓 20 步 | 1.750403 |
| 第 1 段：對照／中途 checkpoint 存檔 | 0.025053／0.014336 |
| 第 1 段：checkpoint 與 JSON 檔的 Volume commit | 4.430447 |
| 第 3 段：續訓後 checkpoint 存檔／第一次 Volume commit | 0.012611／0.901127 |
| 第 2 段：HF 上傳 checkpoint 與 JSON 檔／下載 checkpoint | 1.505219／1.032223 |
| 第 1 段／第 3 段：GPU 呼叫的 wall time | 35.186691／19.436598 |
| Actions 端整段的 wall time | 72.374291 |

前三列是含檢查與記錄的訓練時間。第一次用到某些 kernel（在 GPU 上執行的運算程式）或 optimizer 時有初始化成本：不中斷的 40 步與新 container 的 20 步都包含這類成本，中斷前的 20 步則是在同一個 container 暖機之後才跑，所以這三個數字不能直接拿來比快慢。兩個 Volume commit 的數字，是明確計時的第一次資料 commit；之後保存 JSON 報告的 commit 算在 GPU 呼叫的 wall time 裡。Actions 端整段的時間從檢查設定開始，到第 3 段結束、這次的 Modal app 關閉為止，包含 image 的準備與建置、尋找含 HF token 的 Modal Secret、三段的執行、container 啟動與傳輸；之後 Actions 的停止確認與 artifact 上傳（把結果 JSON 存成 Actions 的附件）不在這段時間內。

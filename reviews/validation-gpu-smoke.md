# 審查紀錄：GPU／checkpoint 實測

審查範圍：`docs/validation/gpu-smoke.md`。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面改寫後，由另一位 AI 獨立查核：對照 repo 的程式、指令、紀錄與頁面引用的來源，實際執行頁面上的部分指令與步驟，並檢查與其他頁的說法是否一致。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：通過

通過，沒有必要問題，只剩兩項建議。

1. 先前審查意見與敘事：8 條先前審查意見都已處理。全頁沒有製作或修訂經過的敘述，也沒有「待補／之後會」這類計畫式的範圍說明。剩下的「這次」都是指紀錄裡那一次執行，審查用的事實與寫作規範清單允許這種用法。
2. 事實正確：頁面上每個結果、ID、commit、hash 與計時都對照了 artifacts/checks/gpu-smoke.json，包括我從 80 步自行算出的前／後 5 步平均與 L2 範圍，全部相符。程式行為對照了 gpu-smoke.yml、deployment-gpu.yml、scripts/modal_gpu_smoke.py、miniyolo/gpu_smoke.py、checkpoint.py、train.py、models.py、scripts/detect_image.py、evidence_records.py、record_evidence.py、validate_curriculum_evidence.py 與各 requirements 檔：預設值、上限、Secret 尋找順序、私有 repo 規則、鎖定檔來源都正確。這些檔案與執行時的 commit 463d3f5 完全相同。
3. 我在暫存副本實測：
   - tests/test_checkpoint.py：2 passed。
   - 訓練 CLI 存的 checkpoint 是 format_version 2，load_grid_checkpoint 與 load_checkpoint 都讀得到。
   - validate_curriculum_evidence.py --scope gpu 通過，紀錄對得上目前的程式。
4. 沒有遺漏：舊頁的範圍限制與安全事項都還在（不覆寫、只傳私有 repo、不改費用上限、結尾的停止確認、不保證跨硬體逐位相同、不執行 TensorRT）。刪掉「本機前置測試 24／24」是對的，新紀錄裡沒有這項資料。
5. 建置：在暫存副本執行 zensical build --clean --strict 與 validate_site.py，兩者都通過。產生的頁面裡兩個表格、清單、程式區塊與四個連結都正常，其中 #l4-results 錨點在 HEAD 就已存在。這次查核的頁面只改了 docs/validation/gpu-smoke.md。

兩項 should：第 12 行沒寫明從 HF 下載回來的只有中途 checkpoint；第 13／37／60 行逐步比對的「梯度」其實是梯度的 L2 norm。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/validation/gpu-smoke.md 第 12 行（〈執行限制與路徑〉步驟 2） | 「再把中途 checkpoint 與這四個 JSON 檔上傳到私有的 checkpoint repo，依這次上傳產生的 commit（HF repo 的版本）下載回來，再核對一次」最自然的讀法是五個檔案都下載回來並核對過。實際上 scripts/modal_gpu_smoke.py 的 transfer 只用 hf_hub_download 取回 midpoint.pt，紀錄裡也只有 storage.hf.sha256 這一個值（中途 checkpoint 的 SHA-256）。同頁表格的「HF 下載」列寫的是正確範圍（「重新下載中途 checkpoint」），兩處說法不一致。 |
| 2 | 建議 | docs/validation/gpu-smoke.md 第 13、37、60 行 | 逐步比對的「梯度」其實是每一步梯度的 L2 norm：history 裡的 gradient_l2 欄位，由 compare(history, baseline_history[20:]) 比對，並不是整組梯度本身。第 13 行「每步的 loss、梯度（gradient）」、第 37 行「每步的 loss、梯度允許…浮點誤差」、第 60 行「loss／梯度軌跡的最大差異都是 0」，都會讓讀者以為每一步的整組梯度都逐一比對過。L2 norm 的定義要到第 39 行才出現。 |

### 第 2 次查核：通過

結論：通過。沒有必要問題，只有一條 should：第 15 行 timeout 的括號說明不夠精確。

一、遺留項目與受程式改動影響的段落
- gpu-smoke 兩條都已正確修正：
  - 步驟 2 現在寫明只下載中途 checkpoint（JSON 檔不下載），和 transfer 裡 hf_hub_download 只取 midpoint.pt 一致。
  - 「梯度」第一次出現時改成「梯度大小（L2 norm…）」，第 37、39、55、60 行也都跟著改。L2 norm 的定義和 optimizer_step 的算法相同：所有梯度元素平方和開根號。
- env-notebook 一條已修正：連結文字改成導覽名稱「完整閱讀路線」。
- 受程式改動影響的段落對這兩個檔案沒有條目。editor 判斷 train-render 那段備註已過時，判斷正確：gpu-smoke.json 的 dependencies_sha256 有 12 個檔，等於 bound_code，含 train.py 與 provenance.py。

二、對照程式與紀錄
- 程式：逐句對照了 miniyolo/gpu_smoke.py、scripts/modal_gpu_smoke.py、gpu-smoke.yml、deployment-gpu.yml、checkpoint.py、train.py、detect_image.py、tests/test_checkpoint.py、evidence_records.py、record_evidence.py、validate_curriculum_evidence.py、requirements-gpu.in 與 .lock。lock 的版本和 requirements-model.txt 逐項相同。git diff 463d3f5 確認這些 GPU 相關程式在執行之後都沒改過。
- 數字：頁面上所有數值都從 gpu-smoke.json 重算過，四捨五入後一致，包括前 5／後 5 步平均、80 步梯度大小範圍、9 項計時、app id、HF commit 與 SHA。
- 確定性數值：在 CPU 上重算，也一致：15,511 個參數、學習率 0.008／0.0064、format_version 2、6 個檔。
- 頁面沒有加入在 Mac 上量到的數字。editor 列出的「依紀錄而定的數值」清單是完整的。

三、照著頁面的說明實際操作過
1. pytest tests/test_checkpoint.py：2 項通過。
2. validate_curriculum_evidence.py --scope gpu：通過。
3. 在另一份副本改 miniyolo/losses.py，gpu-smoke.json 變成過期；只改 scripts/modal_gpu_smoke.py，紀錄仍然有效。這和頁面說的綁定範圍一致。
4. record_evidence.py 的列表模式沒有列出任何 GPU 紀錄。
5. 用假的 modal CLI 模擬 confirm_stopped：前三次查詢 app 都沒停時，才送出一次 modal app stop，而且只針對這次的 app；總共查 8 次仍沒停，就回報確認不了，這一步失敗。
6. 訓練 CLI 存的 checkpoint：format_version 是 2、scheduler 是 None，detect_image 載入得了。

四、一致性
- 頁面用現在式寫成，沒有寫修訂或審查經過。範圍限制寫成現況。
- 和審查用的事實與寫作規範清單、status.md、index.md、README、curriculum.md、20-deployment.md 都沒有矛盾。
- 術語說明和術語表、20-deployment 的寫法一致。中英文間空格與全形標點沒有問題。
- 執行連結旁的日期是紀錄值，審查用的事實與寫作規範清單那條規則是針對 CPU 紀錄頁，這裡可以接受。

五、建置與檔案
- 我的副本裡 zensical build --clean --strict、validate_site.py、validate_preparation.py 都通過。頁面連結指向 #l4-results 與 publish.md，渲染正常。README 的相對連結都存在。43 本 notebook 都是合法 JSON。
- 環境 notebook 的 JSON 格式（indent=2、ensure_ascii=False、結尾換行）和三格的 id、類型、行數都不變，只改了 next-steps 格的一行。環境格確實印出 REF（教材版本）。
- 沒有發現 editor 改到這兩個檔以外的檔案。

副本
log

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/validation/gpu-smoke.md 第 15 行（〈執行限制與路徑〉第二段）「`timeout=600`（每次呼叫最多 600 秒）」 | 這段括號說明是這次編輯新加的，把上限說小了。scripts/modal_gpu_smoke.py 的 GPU 函式同時設了 `timeout=600` 和 `startup_timeout=600`。依 Modal 文件，從 v1.1.4 起（requirements-modal.txt 固定 modal==1.6.0），只要設了 `startup_timeout`，`timeout` 就只管每次呼叫的執行時間；container 啟動時間另外由 `startup_timeout` 限制。所以一次 GPU 呼叫最長可能是啟動 600 秒再加執行 600 秒。讀者照「每次呼叫最多 600 秒」估算單次呼叫的最長時間或費用，會少算一半。不過整份工作另有 25 分鐘上限，實際影響不大。 |

最後一次查核的建議事項，在下方〈定稿修正〉逐項處理。

## 讀者審查與技術查核

### 事實查核（AI 對照 repo 的程式、紀錄與頁面引用的來源）

方法：Repository: https://github.com/birdhackor/learn_to_yolo, branch release/lessons-v0.4.0, commit 31527417d328e18418c438cb76b4630a8c695f33 (clean working tree), rsynced to the 暫存副本. Files read: docs/validation/gpu-smoke.md; 審查用的事實與寫作規範清單; .github/workflows/gpu-smoke.yml, deployment-gpu.yml, pages.yml; scripts/modal_gpu_smoke.py; miniyolo/gpu_smoke.py, checkpoint.py, train.py, models.py, data.py, provenance.py; scripts/evidence_records.py, record_evidence.py, validate_curriculum_evidence.py, detect_image.py; tests/test_checkpoint.py; requirements-modal.txt, requirements-model.txt, requirements-gpu.in, requirements-gpu.lock (header and pins; shasum = the record's model_lock_sha256 9ef8e0ec…); artifacts/checks/gpu-smoke.json (every field; Python recomputation of the first-5 and last-5 step averages 0.92157216 and 0.06433246, the 80-step gradient range 0.152426–1.594808, and baseline vs interrupted and resume max differences of 0); docs/preparation/publish.md (§6 steps 3–4 and 219); docs/lessons/20-deployment.md (#l4-results anchor); docs/lessons/07-training.md and 07-inference.md (GridDetector belongs to chapter 7); docs/glossary.md; zensical.toml (nav titles). git diff --stat 463d3f5..HEAD on the GPU-related files: miniyolo/, modal_gpu_smoke.py, both workflows and the lock are unchanged; only evidence_records.py, record_evidence.py and validate_curriculum_evidence.py changed (Fashion-MNIST lines). git ls-files: no .pt/.onnx weights tracked. GitHub (read-only): gh run view 37217100342 --repo birdhackor/learn_to_yolo --json … (workflow_dispatch, headSha 463d3f5…, 2026-10-04, success) and --log (validate-only output, objects created, step lines, passed status, confirm-stopped JSON); gh api contents of artifacts/checks/gpu-smoke.json (404 on main today; on release/lessons-v0.4.0 the blob df81766… matches the working tree). External sources: Modal https://modal.com/docs/sdk/py/latest/App (App.function: timeout, startup_timeout, retries, max_containers, single_use_containers, serialized); https://modal.com/docs/guide/timeouts (timeout vs startup_timeout); https://modal.com/docs/guide/volumes (Commits and reloads, background commits, create_if_missing); https://modal.com/docs/sdk/py/latest/Secret (Secret.objects.list); PyTorch https://docs.pytorch.org/docs/2.9/generated/torch.use_deterministic_algorithms.html (warn_only, the AdaptiveAvgPool2d CUDA backward listing, CUBLAS_WORKSPACE_CONFIG); GitHub Docs https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency (one running and one pending; an existing pending run is canceled; queue option); Hugging Face API https://huggingface.co/api/models/birdhackor/learn-to-yolo-checkpoints (401 unauthenticated, consistent with private) and …/learn-to-yolo-models (public, contains only .gitattributes, so no public model). Commands run in the scratch copy: PYTHONPATH=. OMP_NUM_THREADS=2 MPLBACKEND=Agg .venv-model python -m pytest tests/test_checkpoint.py -p no:cacheprovider (2 passed), and the page's documented form without PYTHONPATH (2 passed); a CPU produce()/resume() run giving parameter_count 15511, lr 0.008/0.0064, Adam step [40,40], model/optimizer/history errors 0, the 6-file list, checkpoint keys, the four RNG keys, the scheduler state and the preprocessing/data-source/model-config JSON; python -m miniyolo.train --steps 2 --samples 8 --device cpu (checkpoint has format_version 2 and scheduler_state_dict None; load_grid_checkpoint and load_checkpoint both accept it); scripts/detect_image.py run on the CPU smoke midpoint.pt (it loads and detects); python3 scripts/validate_curriculum_evidence.py --scope gpu (2 GPU records match the current code) plus a per-file binding diff (none); .venv-model python scripts/record_evidence.py (listing mode; no GPU record listed as out of date) and --help; .venv-docs zensical build --clean --strict (no issues; rendered page has 2 tables, links and anchors resolve); python3 scripts/validate_site.py (passed); curl against the HF API. The page has no SVG, so nothing was rendered with qlmanage.

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 必要 | 〈計時與驗證結果〉第二段（第 45 行）：「這份紀錄用 SHA-256 綁定在 Modal 上執行的實驗程式 miniyolo/gpu_smoke.py……（……Actions 端的 scripts/modal_gpu_smoke.py 不在其中）。這些檔案都沒變，紀錄就一直有效，不必重跑」；相關說法在第 29 行「Actions 端（在 GitHub Actions 上呼叫 Modal 的程式 scripts/modal_gpu_smoke.py）」 | 頁面把 scripts/modal_gpu_smoke.py 說成只在 Actions 端執行的程式，讀者會以為「在 Modal 上跑的程式都受紀錄綁定，只有 Actions 端的呼叫程式沒綁」。實際上這個檔案的 build_app() 定義了三個 Modal 函式，設了 serialized=True，函式本體會送進 Modal container 執行，包括：(1) GPU 函式的外層：volume.reload／commit 與 commit 的計時、收集 AdaptiveAvgPool2d 警告、寫 producer-committed.json；(2) 檢查 HF_TOKEN 的 probe_hf；(3) 整個第 2 段 transfer()：6 個檔案的 SHA-256 核對、「必須是不同 container」的檢查、HF repo 是否私有、上傳、依上傳 commit 下載並核對 SHA-256。scripts/evidence_records.py 的 GPU_RECORDS 只把 miniyolo/gpu_smoke.py 當入口，所以紀錄的 dependencies_sha256 不含這個檔案，也不含 requirements-gpu.lock 與 gpu-smoke.yml（紀錄雖存了 model_lock_sha256，但沒有任何程式檢查它）。結果是：維護者改了第 2 段的核對邏輯或 GPU 環境的 lock，validate_curriculum_evidence.py 仍判紀錄有效，record_evidence.py --gpu 也會因 is_current 為真而跳過；表格裡「Volume」「HF」「HF 下載」幾列就可能在描述已經不存在的程式。這部分沒有任何守衛。 |
| 2 | 建議 | 〈執行限制與路徑〉第 15 行：「工作流程屬於 learn-to-yolo-gpu-smoke 這個 concurrency 群組（第 20 章的 deployment-gpu.yml 也在這一組）：同一時間只跑一個，進行中的不會被取消。」 | GitHub Docs〈Control the concurrency of workflows and jobs〉寫明：同一群組預設最多一個執行中、一個等待中；有新的執行排進來時，原本等待中的那一個會被取消，由新的取代。cancel-in-progress: false 只保護執行中的那一個。頁面只講了執行中的情況。舉例：deployment-gpu 正在跑時，維護者連續手動啟動兩次 gpu-smoke，第一次啟動的會被取消；照頁面的說法，他會以為兩次都會依序執行。 |
| 3 | 建議 | 〈計時與驗證結果〉第 43 行：「下面的結果來自 GitHub Actions 的執行 37217100342（2026-10-04，UTC）。」 | 審查用的事實與寫作規範清單的寫作規則要求頁面正文不要寫死紀錄日期，重跑後以紀錄 JSON 為準。這個日期和執行編號、commit、app ID、HF commit、SHA-256 一樣，GPU 紀錄重產時都會變。publish.md 第 6 節也說明這一頁不會因紀錄重產而被 review_coverage 列出，只能靠人工搜尋更新，多寫死一個值，就多一個會過期而沒人提醒的地方。 |

各項的處理見下方〈定稿修正〉。

## 定稿修正

上面各項意見與先前查核留下的建議，由 AI 逐項核實後處理：必要問題全部修正，建議事項只在修正明確、範圍小時採用。

| # | 來源 | 意見 | 處理 |
|---|---|---|---|
| 1 | 事實查核 | 把 modal_gpu_smoke.py 說成只在 Actions 端執行，綁定範圍說錯 | 已修正：已查證：gpu()、probe_hf()、transfer() 都設了 serialized=True，在 Modal 上執行；GPU_RECORDS 只以 miniyolo/gpu_smoke.py 為入口。第 29 行改成：這個檔案也定義了在 Modal 上執行的函式。第 45 行改成照實描述：只綁 gpu_smoke.py 與它 import 的模組，modal_gpu_smoke.py（Volume commit、HF_TOKEN 檢查、整個第 2 段）、requirements-gpu.lock、gpu-smoke.yml 都不在其中；也補上只改這些檔案時，要手動用 gh workflow run／gh run download 重產的作法。守衛缺口本身在 scripts/evidence_records.py，不在我能改的範圍。 |
| 2 | 事實查核 | concurrency 等待中的執行會被取消，頁面沒說 | 已修正：補一句：排隊等待的最多只有一個，有新的執行排進來時，原本在等待的那一個會被取消。也補上 record_evidence.py --gpu 會等前一個跑完才啟動下一個，不受影響（已對照 gh run watch 的流程確認）。 |
| 3 | 事實查核 | 正文寫死紀錄日期 | 已修正：刪掉「（2026-10-04，UTC）」，日期由執行連結與紀錄提供。 |
| 4 | 先前查核 | timeout=600 的括號把上限說小了 | 未改：已經修好：目前第 15 行分開寫 startup_timeout=600（container 啟動）與 timeout=600（啟動後每次呼叫的執行）。 |

修正後由另一位 AI 檢查這一批頁面（`docs/validation/curriculum.md`、`docs/validation/gpu-smoke.md`、`docs/preparation/data.md`、`docs/preparation/publish.md`、`docs/preparation/architecture.md`、`docs/planning/outline.md`、`docs/planning/course-research.md`、`docs/planning/feedback.md`、`docs/research/foundation-data.md`、`docs/research/detection-data.md`、`docs/research/lfs.md`、`docs/research/pages-colab.md`、`docs/research/version-sources.md`、`README.md`）的改動，第 1 次：通過。檢查內容：每項改動是否符合程式、紀錄與引用的來源（需要時重算或重跑），回報已修正的必要問題是否真的修好、沒改的理由是否成立，改動是否符合寫作規範，網站嚴格建置與程式摘錄比對是否通過。

| # | 嚴重度 | 位置 | 留下的意見 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | docs/validation/gpu-smoke.md 第 45 行，新增的手動重跑步驟「跑完後用 `gh run download <編號>` 下載結果，把其中的 `result.json` 存成 `artifacts/checks/gpu-smoke.json`」 | 這段沒說只有通過的結果才能存成紀錄。工作流程失敗時也會上傳 result.json，而且裡面已經寫上綁定程式的 SHA-256。維護者照這段把它存進去並 commit 後，`record_evidence.py --gpu` 會把它當成現行紀錄而跳過，Pages 建置則會因為 status 不是 passed 而一直失敗。〈發布與帳號設定〉第 186 行對 `--gpu` 的路徑有同樣的警告，這條手動路徑卻沒有。 | 已修正；這項修正由下方〈後續編輯的檢查〉核對 |

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 1 輪：獨立查核之後的編輯

對照 scripts/modal_gpu_smoke.py 第 33–35 行（gpu=L4、timeout=600、startup_timeout=600、retries=0、max_containers=1、single_use_containers=True）與 gpu-smoke.yml（timeout-minutes 25、concurrency）。Modal 1.6.0 wheel 的 docstring 與官方 timeouts 說明確認：timeout 限制每個 input 的執行時間，不含啟動；startup_timeout 限制 container 啟動，且優先於 timeout。所以括號說明正確。concurrency 最多一個執行、一個等待，新排入的會取消舊的等待者，敘述正確；record_evidence.py --gpu 逐一 `gh run watch`，不受影響。綁定範圍的說明（Volume reload／commit、probe_hf、transfer 都在 modal_gpu_smoke.py）正確。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | 〈計時與驗證結果〉第二段手動重跑：「跑完後用 `gh run download <編號>` 下載結果，確認紀錄的 `status` 是 `passed` 後，再把其中的 `result.json` 存成 `artifacts/checks/gpu-smoke.json`；沒有通過就不要 commit。」 | result.json 的 status 在最後一步 `--confirm-stopped` 失敗時仍是 passed：modal_gpu_smoke.py 只把 modal_stop_confirmation.verified 寫成 false，然後以 1 結束。照這步存檔並 commit，Pages 建置時 validate_curriculum_evidence.py 會因為 app 沒有確認停止而失敗。另外，`gh run download <編號>` 沒加 `--dir` 時，會在目前目錄（repo 根目錄）建立 gpu-smoke-<run>-<attempt>/；這個目錄不在 .gitignore 裡，〈發布與帳號設定〉的 `git add -A` 會把它一起 commit。 | 已修正：改成下載到不進 git 的 `artifacts/runs/gpu-smoke-download`，確認 `status` 是 `passed`、`modal_stop_confirmation.verified` 是 `true`，存檔後先執行 `validate_curriculum_evidence.py --scope gpu`，通過再 commit。 |

### 第 2 輪：上一輪的處理與審查紀錄：通過

核對手動重跑那句：gpu-smoke.yml 只有 workflow_dispatch，artifact 名為 gpu-smoke-<run>-<attempt>、內容是 result.json；`gh run download <編號> --dir …` 會放在該資料夾下的 artifact 子目錄；.gitignore 含 artifacts/runs/；modal_gpu_smoke.py --confirm-stopped 把 modal_stop_confirmation 寫回 result.json；validate_curriculum_evidence.py --scope gpu 檢查 is_current、status==passed、gpu_calls_finished 與 modal_stop_confirmation.verified；record_evidence.py --gpu 逐一 gh run watch、用 rglob 找 result.json。gpu-smoke.json 確有這兩個欄位。句子屬實，修好上一輪的兩點（停止確認、下載位置不進 git）；concurrency 句與 deployment-gpu.yml 同組也正確。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | 〈計時與驗證結果〉第二段手動重跑那一句 | 五個步驟塞在一句裡；「確認紀錄的 `status`」指的其實是剛下載、還沒存成紀錄的 result.json；也沒說 `<編號>` 從哪裡查。 | 已修正：改成四個編號步驟，寫明確認的是下載的 `result.json`，並用 `gh run list --workflow gpu-smoke.yml` 找 run 編號。 |
| 2 | 建議 | reviews/validation-gpu-smoke.md〈獨立查核〉 | 兩次查核的順序顛倒：「第 1 次查核」寫「一、遺留項目與 impact - gpu-smoke 兩條都已正確修正」，指的卻是列在「第 2 次查核」的兩項 should。「修正後由另一位 AI 檢查改動」的內容是別批頁面（「指派的 14 頁裡…」）。 | 已修正：各輪依實際執行時間排序；修正後的檢查只掛在這一批真的有改動的頁面。 |

### 第 3 輪：上一輪的處理與審查紀錄：通過

〈計時與驗證結果〉的四個手動重跑步驟逐項對照：gpu-smoke.yml 只有 workflow_dispatch，artifact 名稱是 gpu-smoke-<run_id>-<attempt>，內容是 result.json；gh workflow run gpu-smoke.yml --ref、gh run list --workflow、gh run download <id> --dir 的寫法正確（沒指定 -n 時，每個 artifact 會解到子資料夾，「下載的 result.json」這個說法仍成立）；artifacts/runs/ 在 .gitignore；modal_gpu_smoke.py 只在 gpu_calls_finished 之後才把 status 設成 passed，--confirm-stopped 會把 modal_stop_confirmation 寫回 result.json；validate_curriculum_evidence.py --scope gpu 檢查 is_current、passed、gpu_calls_finished 與 verified（實跑 exit 0），record_evidence.py --gpu 也是用 rglob 找 result.json 再存檔。內容屬實，也解決了上一輪的發現（步驟拆開、確認的是下載的 result.json、寫明編號從哪裡查）；渲染成 4 項 ol，網站建置通過。紀錄的各輪已依時間排序，修正後檢查只掛在有改動的頁。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | docs/validation/gpu-smoke.md 第 47 行，手動重跑第 1 步 | 沒提醒先把修改 commit 並推到 <分支>。這條手動路徑正是給「只改了沒有綁定的檔案」（modal_gpu_smoke.py、requirements-gpu.lock、gpu-smoke.yml）的情況用的。維護者若只在本機改了 modal_gpu_smoke.py 就照做，工作流程跑的是遠端的舊版；結果照樣 passed，--scope gpu 也會通過（這些檔案不在綁定範圍），存進 repo 的紀錄卻沒有驗證到這次修改，沒有任何檢查抓得到。record_evidence.py --gpu 會比對 ls-remote 擋下這種情況，手動路徑沒有這道保護。 | 已修正：第 1 步先提醒把修改 commit 並推到 `<分支>`，工作流程跑的是推上去的程式。 |
| 2 | 建議 | reviews/validation-gpu-smoke.md 第 65–66、78 行 | 路徑清理留下的殘句：「副本：暫存副本」「log：暫存副本」，以及英文方法段的「rsynced to the 暫存副本Files read:」（少了標點，兩句黏在一起）。 | 未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 4 輪：上一輪的處理：有必要問題

手動重跑第 1 步「先把修改 commit 並推到 <分支>（工作流程跑的是推上去的程式）」對照 .github/workflows/gpu-smoke.yml（只有 workflow_dispatch、actions/checkout@v4 不指定 ref，跑的是 --ref 分支上推上去的 commit）與 record_evidence.py --gpu 的 ls-remote 比對：屬實，回應第 3 輪第 1 項，和第 4 步的「通過才 commit」不衝突，讀來通順、合寫作規則。B：讀紀錄全文核對第 3 輪兩項處理；第 1 項屬實，第 2 項不實。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | reviews/validation-gpu-smoke.md 第 78 行（〈事實查核〉方法段開頭），對應第 3 輪第 2 項處理 | 處理寫「點名的路徑殘句與內部用語已清理或換成白話」，但點名的「rsynced to the 暫存副本Files read:」只變成「rsynced to the 暫存副本 read: docs/validation/gpu-smoke.md; …」，發現指出的「少了標點、兩句黏在一起」仍在，還少了 Files 一字。 | 已處理：第 3 輪的處理說明改成統一的說明。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |
| 2 | 建議 | reviews/validation-gpu-smoke.md 第 65–66 行 | 點名的「副本：暫存副本」「log：暫存副本」刪掉路徑後，只剩孤立的「副本」「log」兩行，沒有意義。 | 已處理：第 3 輪的處理說明改成統一的說明。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 5 輪：上一輪的處理：有必要問題

第 3 輪第 2 項、第 4 輪第 1 項屬實：第 78 行已經是「rsynced to the 暫存副本 Files read:」。第 4 輪第 2 項不實。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | 第 4 輪第 2 項處理欄（第 142 行） | 發現點名的是：刪掉路徑後只剩孤立的「副本」「log」兩行。這兩行仍在第 65、66 行，處理欄卻寫「已修正：點名的文字已不在紀錄裡」。產生器略過了 4 個字以下的引文，只比對了早已不在的「副本：暫存副本」「log：暫存副本」。 | 已處理：用詞類的處理說明改成統一的說明（紀錄保留查核者的原文，只統一替換路徑與內部名稱），不再逐句計數。 |


## 2026-10-05 clear-tutorial 三輪重審

以上是原審查歷史；不追溯改成首次盲讀。這次由固定基線 `16f6910` 分段開放並保存當時理解，再修改、核技術及檢查銜接，詳見 [本輪方法與限制](clear-tutorial/16f6910/README.md)。

- 第一輪：[reference當場閱讀原始紀錄](clear-tutorial/16f6910/first-read/reference.jsonl)，基線來源與圖指紋保留；共享檔案系統不是技術隔離。
- 第二輪：[非作者技術／證據核對](clear-tutorial/16f6910/technical/reference.md)，實際來源、數字及必要執行範圍見該報告。
- 第三輪：[另一位讀者前文→本節→後文複查](clear-tutorial/16f6910/transitions/reference.md)，此輪完整頁閱讀非盲讀；受影響段落及圖另有delta核回。
- [原始卡點與具體處理](clear-tutorial/16f6910/decisions.json)保留未新增的選讀建議。原先前提包漏發及08提前brief的限制另列，沒有算成教材錯或冒稱08全程盲讀。

本輪修正後沒有未解的必要問題；這是AI閱讀／技術查核的實際範圍，不是學生學習成效驗收。全站實際Zensical桌面／手機、公式換頁與執行檢查見 [verification.json](clear-tutorial/16f6910/verification.json)，不以SVG檔存在或strict build取代視覺查核。

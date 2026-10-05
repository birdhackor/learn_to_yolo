# 審查紀錄：發布與帳號設定

審查範圍：`docs/preparation/publish.md`。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面改寫後，由另一位 AI 獨立查核：對照 repo 的程式、指令、紀錄與頁面引用的來源，實際執行頁面上的部分指令與步驟，並檢查與其他頁的說法是否一致。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：有必要問題

不通過：有 1 個必要，另有 4 個建議。

對象是 docs/preparation/publish.md。

1. 先前審查意見與寫法：19 條先前審查意見加上總評都處理了。全文沒有製作或修訂經過、沒有計畫語氣（待補、之後會），也沒有撰寫平台的細節（秘密欄位、聊天、S3 501）。

2. 事實核對：大多正確。我逐一對照了 scripts/github_auth.py、四個 workflow（pages.yml 依序五項檢查與權限、prepare-lfs.yml 的步驟與 contents: write、兩個 GPU workflow 都只有 workflow_dispatch）、record_evidence.py、evidence_records.py、verify_curriculum.py（--check、--render-only）、validate_curriculum_evidence.py、review_coverage.py、validate_lessons.py、validate_preparation.py、build_lesson_notebooks.py（recorded_output）、verify_remote_lfs.py、download_data.py、package_fashion_mnist.py、data/manifest.json 的 lfs_assets、data/local-lfs-verification.json（26,421,880 與 53,723,336 bytes）、data/remote-lfs-verification.json、artifacts/checks/mathjax-vendor.json、.gitignore、.gitattributes、tests/test_notebook_bootstrap.py 的五個分支測試，以及 zensical 的 --clean／--strict。

3. 唯一的 must：第 6 節第 3 步寫「GPU workflow 失敗後，修好再重跑這一步」，但照做不會真的重跑。原因是 `is_current()` 只比對 `dependencies_sha256`，不看 `status`；失敗結果開跑前就寫好了目前程式的 SHA-256，所以 `--gpu` 會把這份失敗紀錄當成現行紀錄而跳過。我在暫存副本模擬確認過，repo 裡的檔案沒有動。

4. 刪掉的內容：沒有遺失唯一的範圍限制或操作說明。Colab 的範圍限制 docs/status.md 也有寫；docs/research/lfs.md 引用的「Git LFS 一節」仍然存在。

5. 渲染：在暫存副本重新同步到最新版 validate_site.py 後，`zensical build --clean --strict` 結果是 No issues found，`validate_site.py` 也通過（新版已包含 numbered lists 檢查），`validate_preparation.py` 同樣通過。第 6 節渲染成一個 10 項的 ol，內含 8 個 code block。

6. 檔案範圍：這次查核的頁面的 diff 只有 publish.md，而且和編輯者暫存副本裡的版本完全相同。repo 裡其他未提交的修改來自同時進行的其他部分，我無法逐一確認它們不是這次查核的頁面改的。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | docs/preparation/publish.md 第 6 節第 3 步（重產 GPU 紀錄）：「workflow 失敗時…這份失敗的結果通不過檢查，不要 commit，修好後重跑這一步。」 | 照這句做，多數情況下不會真的重跑。`record_evidence.py --gpu` 只靠 `scripts/evidence_records.py` 的 `is_current()` 決定要不要重跑（`scripts/record_evidence.py` 第 104–106 行），而 `is_current()` 只比對 `dependencies_sha256`，不看 `status`。`scripts/modal_deployment.py`（第 74–75 行）和 `scripts/modal_gpu_smoke.py`（第 207–208 行）開跑前就把目前程式的 `dependencies_sha256` 寫進 result.json，失敗時照樣上傳；`--gpu` 把它複製成紀錄檔以後，這份失敗紀錄就被當成「現行」。所以只要失敗原因不在綁定的程式裡（例如 Modal 的 `MODAL_TOKEN_SECRET` 沒設、Modal 暫時出錯，或修的是沒被綁定的 `scripts/modal_deployment.py`），重跑 `--gpu` 都會跳過這份紀錄、不啟動 workflow。最後只會在 `validate_curriculum_evidence.py` 的 `status == "passed"` 斷言失敗，第 2 步的列表也不會再把它列為過期，維護者會卡住。我在暫存副本模擬過：把一份 status=failed、dependencies_sha256=bound_code(...) 的結果寫進 `artifacts/checks/curriculum/deployment-gpu.json` 後，`is_current()` 回傳 True，`--gpu` 的待跑清單裡就沒有它了。v0.4.0 第一次產生 GPU 紀錄時（Modal 設定還沒好）就會遇到這個情況。 | 已修正：見下方修正項目，第 2 次複查確認。 |
| 2 | 建議 | docs/preparation/publish.md 第 6 節第 4 步：「在那台電腦 checkout 同一個分支」與 `--push <分支>` | 「建立分支並推上 GitHub」只寫在第 3 步，而沒有過期的 GPU 紀錄時第 3 步要跳過。跳過第 3 步、又要到另一台電腦產生 CPU 紀錄的人，讀到「同一個分支」會找不到它指的是哪一個。頁面也沒提醒要先把第 1 步的修改 commit 並推送，記錄用的電腦才拿得到這些修改。 | 已處理：目前的第 3 步把修改推到一個分支，並說明在另一台電腦記錄時也要從 GitHub 取得；第 5 步寫明 checkout 第 3 步的分支。最後一輪檢查逐項核對過。 |
| 3 | 建議 | docs/preparation/publish.md 第 6 節第 10 步（驗證公開網站） | 有兩個缺口。(a) 這一步要人在 Colab 開新 tag 的 notebook、執行環境格，但要寫進 `curriculum-publication.json` 的欄位清單（tag 與 commit、部署 commit、workflow run、逐頁比對）裡沒有 Colab 的結果。審查用的事實與寫作規範清單說 Colab 上的實際執行要看發布驗證紀錄，照這份清單做，Colab 的結果就不會留下紀錄。(b) repo 裡沒有任何腳本會產生或比對這份紀錄（全 repo 只有本頁提到 `curriculum-publication`），可是頁面要人確認「每一頁的內文都和本地 `site/` 相同」，網站有 60 個 HTML 頁面，卻沒說要怎麼比，維護者容易卡在這裡。 | 已處理：發布後的驗證改成手動啟動的 Verify published lessons 工作流程，再用 `scripts/verify_release.py save` 存成紀錄，並寫明不經過 Colab。最後一輪檢查核對過。 |
| 4 | 建議 | docs/preparation/publish.md 第 5 節〈首次使用遠端 LFS〉第 5 步：「在 `data/manifest.json` 的 `lfs_assets` 記下它的路徑、大小與 SHA-256」 | 第 6 步接著要人執行 `scripts/verify_remote_lfs.py`，這支腳本第 31 行會讀 `asset["id"]`。如果只照頁面記下路徑、大小與 SHA-256，沒有 `id`，執行時就會出現 KeyError: 'id'。另外，記大小的欄位名稱是 `bytes`。 | 已處理：〈首次使用遠端 LFS〉第 5 步列出 `id`、`path`、`bytes`、`sha256` 與 `license`。最後一輪檢查核對過。 |
| 5 | 建議 | docs/preparation/publish.md 第 6 節第 6 步（審查改過的頁面） | 審查排在產生紀錄之後，但頁面沒說審查後如果又改了頁面或程式，要回到哪一步。如果改的是 `lesson_cases/` 或它 import 的模組，notebook 要重建（第 1 步），紀錄也會過期（第 2–5 步）。這時 `validate_lessons.py` 的 notebook 比對是一個不帶訊息的 assert，失敗時讀者看不出原因。 | 已處理：第 7 步末段說明審查之後又改了程式要從第 1 步重來；只改頁面文字或圖時，重審 `review_coverage.py` 列出的頁面。最後一輪檢查核對過。 |

修正必要問題時處理的項目（修正後由第 2 次複查確認）：

- line 3 「目前教材（lessons-v0.3.0）的公開驗證見」: rewrote — intro names no tag; public verification is step 11 (artifacts/checks/curriculum-publication.json, whose source_ref/release_commit record the tag)
- line 22 「這是待 owner 決定的選項，本輪未替你宣告」: rewrote — 「repo 沒有 LICENSE 檔。…由 owner 決定…決定後在 repo 根目錄加上 LICENSE。」
- line 24 「## 2. 把修訂推到 GitHub」: rewrote — 「## 2. 推送到 GitHub」
- line 28 「本環境使用你提供的 GIT_LFS_AUTHORIZATION 秘密欄位」: restructured — 〈推送認證〉 leads with the maintainer's own SSH/credential helper/fine-grained PAT plus permission checks; scripts/github_auth.py is an optional helper described per its code; the 'push permission is not LFS proof' rule is kept; push uses git push origin HEAD:main
- line 39 「本輪實測：LFS batch upload 認證回 200，但…S3…回 501」: deleted the debugging history; the workflow is described in 〈重新發布 LFS 資料〉
- line 45 「### 推送修訂」: rewrote — 「### commit 與 push」
- line 73 「你看到的雲端環境「儲存並發布」不是 GitHub Pages 的發布流程。」: deleted
- line 84 「每節網頁另有固定 lessons-v0.3.0 的 Colab 入口」: rewrote — 「每節網頁的 Colab 按鈕開啟發布 tag 的 notebook」 plus which checks cover notebooks and that Colab itself is untested by them
- line 86 「## 5. Git LFS：已準備與你需要設定的部分」: rewrote — 「## 5. Git LFS」
- line 98 「此外，Fashion-MNIST 完整封裝已透過 GitHub Actions 真正上傳…」: rewrote into one present-tense paragraph (remote upload/download record + local 26,421,880 / 53,723,336 bytes test; Penn-Fudan local only)
- line 105 「不必為前置工作開通付費」: rewrote — 「先決定預算；在包含額度內使用 LFS，不必先開通付費。」
- line 109 「目前只是準備規則，不將完整公開 dataset 全量上傳。」: rewrote — 「LFS 只放精選封裝，不全量上傳完整的公開 dataset。」
- line 111 「### 本雲端需要重新發布資料時」: rewrote — 「### 重新發布 LFS 資料」
- line 113 「直接從此雲端 PUT 到 S3 的路徑曾回 Transfer-Encoding 501…」: rewrote — workflow steps per .github/workflows/prepare-lfs.yml, GITHUB_TOKEN, contents: write, no PAT secret
- line 117 「不是本輪新增的模型」: rewrote — the hypothetical checkpoint example is gone; 〈Colab 只取得指定 LFS asset〉 uses the real data/curated/fashion-mnist-v1.tar
- line 128 「正式小節再替換成固定的 release／commit 與確實存在的 asset」: rewrote — 「教材的 notebook 要用時，改成 clone 固定的發布 tag（git clone --branch <tag>）…完整的來源 dataset 走下載器或官方下載，不走 Pages。」
- line 143 「本輪教材與42份Colab固定為 lessons-v0.3.0；已發布的 lessons-v0.1.0、lessons-v0.2.0 保留不動。」: restructured — section moved before the references as 「## 6. 發布新版教材」; opening uses section-map.json source_ref; no tags listed
- line 145 「下一版例如使用 build_lesson_notebooks.py --ref lessons-v0.4.0 配對；這會重建notebook，須重新保存實際輸出」: rewrote — steps 1–6 follow the current code (recorded_output keeps stdout for unchanged cases; record_evidence.py lists, --gpu, --run/--push/--bundle; verify_curriculum.py --render-only); this round adds step 3 and the GPU-failure handling
- line 147 「例如 git tag lessons-v0.4.0；不要再建立或移動已發布的0.1.0、0.2.0、0.3.0」: rewrote — step 9 uses git tag -a <新 tag>; 「已發布的 lessons-v* tag 不重建、不移動」
- verdict (restructure; build checks also block on evidence/validate_site): done — section 2 rebuilt around the maintainer's own credentials; section 3 lists all five build-job checks with 「任何一項失敗都不會部署」; release section is numbered §6 with placeholders

### 第 2 次複查：有必要問題

結論：不通過，有 1 個 must（第 6 節第 6 步）和 3 個建議。

**must：第 6 步的例子會讓網站在新輸出旁邊放舊圖。** 例子說「只取回了 artifacts/checks/ 的 JSON」時跑 `--render-only` 就好，但 `--render-only` 不會重畫圖。記錄用的電腦重畫的圖不在 JSON 裡，例如 17／18／19 節的圖，以及補充實驗的 `*-learning.svg`、grid-learning、`08-custom-*` 等圖。有 10 個課程頁會因此顯示舊圖，而且沒有任何檢查會發現。`--push` 與 `--bundle` 會把這些圖一起帶回，沒有這個問題。

**should（3 項）：**
- 第 7 步沒寫要做哪兩種審查（docs/status.md 對外說明的模擬初學讀者 AI 審查與 AI 技術審查）。
- `validate_site.py` 的說明漏了「編號清單」與「被誤當成 HTML 標籤的文字」兩項檢查。
- 刪掉平台說明時，連「認證不要寫進 remote URL、repo 檔案或 Git config」這句也一起刪了。

**其餘都通過：**
- **traces**：全部處理完，全文找不到製作經過、計畫語氣或平台細節。
- **事實**：逐條對照 scripts/record_evidence.py、evidence_records.py、verify_curriculum.py、build_lesson_notebooks.py、review_coverage.py、四個 validate_* 腳本、github_auth.py、verify_remote_lfs.py、package_fashion_mnist.py、四個 workflow 檔，以及 data/manifest.json、data/*-lfs-verification.json、artifacts/checks/curriculum-publication*.json、mathjax-vendor.json、tests/test_notebook_bootstrap.py，指令、旗標、路徑、權限與流程都正確。上一輪的 must（GPU workflow 失敗後要先換掉存下的結果）描述正確：`is_current` 只比對 SHA-256，而 GPU 腳本一開始就把 `dependencies_sha256` 寫進結果。
- **沒有遺失重要內容**：與先前審查意見所依據的版本相比，只少了上面第 7 步那一項。
- **渲染**：在暫存副本跑 `zensical build --clean --strict` 回報 No issues found；`validate_site.py` 與 `validate_preparation.py` 都以 exit 0 通過。第 6 節渲染成一個 11 項的 `<ol>`，第 11 步內有巢狀 `<ul>`。
- **只改了列出的檔案**：工作區裡其他修改屬於同時進行的其他部分，沒有跡象顯示這個部分改了 publish.md 以外的檔案。

在工作區跑 `validate_lessons.py`／`validate_curriculum_evidence.py` 會失敗，原因是其他部分正在改 `lesson_cases/`，不是這一頁造成的。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 必要 | docs/preparation/publish.md 第 6 節第 6 步（第 192–198 行） | 例子「只取回了 `artifacts/checks/` 的 JSON」讓人以為只帶回 JSON、再跑 `--render-only`，網站就和紀錄一致。實際上 `verify_curriculum.py --render-only` 只呼叫 `attach()`，只改頁尾的執行紀錄區塊和 notebook 最後一格。第 5 步在記錄用的電腦上重畫的圖都不會更新：`verify_curriculum.py` 的 `SITE_FIGURES` 把 17-capstone.svg、18-video.svg、19-tracking.svg 從 git 不追蹤的 `artifacts/lesson-*/` 複製進 `docs/assets/diagrams/`；補充實驗腳本也會寫 `*-learning.svg`、`grid-learning-*.svg`、grid-validation 的 PNG 和 `08-custom-*.svg`。有 10 個課程頁（01-small-cnn、03-comparison、04-localization、07-training、07-heldout、08-own-data、10-multiscale、17、18、19）顯示這些圖。`validate_curriculum_evidence.py` 不檢查圖；圖沒被帶回，檔案就沒變，`review_coverage.py` 的 SVG 雜湊也就不會要求重審。所以照這個例子做，網站會在新輸出旁邊放舊圖，而且沒有任何檢查會發現。 |
| 2 | 建議 | docs/preparation/publish.md 第 6 節第 7 步（第 200 行） | 原本的步驟寫明要做「陌生讀者審查」，現在只寫「審查改過的頁面」、審查涵蓋哪些內容、紀錄放在哪裡，沒有寫要做哪一種審查。docs/status.md 第 83 行對外說明，教材的審查是兩種 AI 審查：模擬初學讀者的審查，以及對照論文與固定版本官方程式的技術審查，並記下每個發現和處理方式。維護者照第 7 步做了別種審查，`review_coverage.py --write` 照樣記錄、`validate_lessons.py` 照樣通過，status.md 的說法對這些頁就不再成立，也沒有檢查會發現。 |
| 3 | 建議 | docs/preparation/publish.md 第 3 節 build job 清單的 `validate_site.py` 項（第 70 行） | 清單列的 Markdown 檢查只有「表格／摺疊區塊／數學／連結」。但 scripts/validate_site.py 還會檢查編號清單，以及被 Markdown 誤當成 HTML 標籤的文字；它自己的輸出寫的是 "tables, numbered lists, admonitions, math, links and no text parsed as tags"。建置因為這兩項失敗時，維護者對照這份清單會找不到原因。 |
| 4 | 建議 | docs/preparation/publish.md 第 2 節〈推送認證〉（第 23–34 行） | 原文有一句「不寫進 repo、remote URL、Git config」，這是全頁唯一講認證要怎麼保管的句子，卻跟平台秘密欄位的說明一起被刪掉了。現在只說明 helper 本身不會把 header 寫進任何地方，沒有提醒維護者自己不要把 PAT 或 Authorization header 寫進 remote URL（例如 `https://<token>@github.com/…`）、repo 裡的檔案或 Git config。 |

建議事項的處理（處理後由下一次複查檢查）：

- must, §6 step 6 (--render-only and figures): fixed. Step 6 now says `--render-only` rewrites only the 42 evidence blocks, the notebooks' last-cell outputs and `artifacts/checks/curriculum/index.json`, and never redraws or copies a figure (verified: the render-only branch calls only attach() and write_index()). It also says `--push` and `--bundle` carry the whole `Changed files:` list (record_evidence.py OUTPUTS covers artifacts/checks, docs/assets/diagrams, docs/lessons, notebooks), and that any other way of bringing records back must include the listed records and figures. It names the consequence: old figures next to new output, and no check catches it (validate_curriculum_evidence never reads figures, validate_lessons checks only SVG viewBox/title, and review_coverage sees no change). The example is now 「只帶回了紀錄與圖，或合併時保留了這邊的頁面」. Step 5 now names which figures it redraws: `verify_curriculum.py` copies the chapter 17–19 figures from git-ignored artifacts/lesson-*/, and every supplementary script except the video one writes its figure into docs/assets/diagrams/.
- should, §6 step 7 (which reviews): applied. Lesson pages get two AI reviews. One simulates a beginner reader (high-school level, strong at math, new to programming) and checks understanding and whether text, figures and numbers agree. The other is a technical review against the original papers, pinned official code and the calculations. Each review record must log every finding and how it was handled. The step links to 〈驗證範圍與後續實驗〉 and says that a different review method means updating the site's description too (worded so it does not depend on which pages carry that description).
- should, §3 validate_site.py item: applied. Added 編號清單 and 有沒有文字被誤當成 HTML 標籤 (both match the script's own report string and checks). While editing the item I also added the Zensical-version-matches-requirements-docs.txt check and spelled out 'artifact 邊界' as no symlinks, no LFS pointers and no .pt/.pth/.onnx/.ipynb/.zip/.tar/.gz files. That suffix list is exactly validate_site.py line 244, so it does not overclaim.
- should, §2 推送認證: applied. Added: PAT 與 `GIT_LFS_AUTHORIZATION` 的值都不要寫進 remote URL（例如 `https://<token>@github.com/…`）、repo 裡的檔案或 Git config, with the reason (plaintext in config files; repo files may be published with a commit). The alternatives (SSH key, credential helper, the env-var helper) stay in the same section.
- extra, not from the checker, §6 step 4: fixed a sentence a concurrent script change made false. scripts/record_evidence.py (untracked; another agent changed it at 00:05 along with validate_curriculum_evidence.py) now runs `validate_curriculum_evidence.py --scope gpu` after `--gpu` and `--scope cpu` after `--run`. That makes 「另一種紀錄還過期時這項檢查會失敗」 false. The paragraph now says the two kinds are checked separately, a stale record of the other kind does not fail either check, and all records are checked together only in step 8. I deleted the ordering sentence that relied on the old reason; the step order and step 5's 「做了第 4 步時，要包含推上去的 GPU 紀錄」 still carry the order. Step 5 now names `--scope cpu` and the `Changed files:` output.
- Tests: in a scratch copy, `zensical build --clean --strict` exited 0 and `scripts/validate_site.py` exited 0, both run after the final edit. The 11 release steps render as one list, and the status-page link resolves.

### 第 3 次複查：通過

沒有必要問題，判定通過；另有 5 項建議。

上一輪的檢查結果：
- 唯一的 must（`--render-only` 與圖）已解決。第 6 步寫明 `--render-only` 只呼叫 attach() 與 write_index()，只改執行紀錄區塊、notebook 最後一格與 index.json，不重畫也不複製任何圖；`--push`／`--bundle` 帶回整份 `Changed files:` 清單（與 record_evidence.py 的 OUTPUTS 一致）；舉例已改成「只帶回了紀錄與圖」。第 5 步說明了哪些圖會重畫：verify_curriculum.py 的 SITE_FIGURES 複製第 17–19 章的圖，影片紀錄以外的補充腳本都會寫 docs/assets/diagrams/。
- 三項建議都已採用，內容分別與 docs/status.md 第 83 行、validate_site.py（第 244、251 行）、github_auth.py 相符。沒有被駁回的建議。

這一輪的查核範圍：
- 逐句對照了：record_evidence.py（`--scope gpu/cpu`、check_pins、run_gpu 比對 ls-remote、失敗時仍保存 result、`--push`／`--bundle`）、evidence_records.py、miniyolo/provenance.py、兩個 Modal 腳本（開始時就寫入 dependencies_sha256）、verify_curriculum.py、build_lesson_notebooks.py、review_coverage.py、三個 validate_* 與 validate_site.py、四個 workflow 的權限與觸發方式、manifest 與 LFS 紀錄、verify_remote_lfs.py、tests/test_notebook_bootstrap.py 的五個分支、zensical.toml 的導覽名稱、Zensical CLI 的 `--clean`／`--strict`。
- 結果：沒有不實的指令、flag、預設值或路徑；沒有過度宣稱 Colab、GPU 或審查範圍；沒有修訂經過的敘述，也沒有計畫式語氣。gpu-smoke.json 與 reviews/coverage.json 目前不在工作樹，屬於審查用的事實與寫作規範清單列出、發布時才會補齊的暫時狀態，所以不算問題。

建置與檢查：在暫存副本（rsync，排除 .git、.venv-docs、.venv-model）執行 `zensical build --clean --strict` 與 `python3 scripts/validate_site.py`，兩者都是 exit 0。渲染後各編號清單的項目數是 6／5／6／11，第 6 節的 11 步是同一個清單。

5 項 should（細節見下表）：
1. 補充紀錄重產後，非課程頁正文引用的數字沒有任何檢查會抓到，例如 status.md 的第 8 章數字。
2. 第 11 步「改名保留舊的發布驗證紀錄」和審查用的事實與寫作規範清單的發布狀態相反。
3. 本頁沒寫 python3 要哪一版，但檢查腳本用到 tomllib，需要 3.11 以上。
4. `--bundle` 用 `tar -xzf` 解開時會直接覆寫同名檔，不像 `--push`＋`git pull` 會合併。
5. `review_coverage.py --write` 的路徑沒有完整例子，傳 docs/ 底下的路徑會丟 ValueError。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/preparation/publish.md 第 6 節第 5–7 步（第 186–212 行） | 第 5 步重產補充紀錄後，數字可能改變（例如程式改了）。有些頁面在正文直接引用這些紀錄的數字，卻沒有顯示會重畫的圖。例如〈驗證範圍與後續實驗〉（docs/status.md 第 78 行）寫了第 8 章的 loss 1.55017→0.16921、train AP50 0.00680、validation 0.388889、test 0.666667，這些數字來自 custom-data-160-step.json 與 custom-data-learning.json。非課程頁的審查只涵蓋頁面文字和它顯示的 SVG，而 status.md 沒有任何 SVG，所以 review_coverage.py 不會列出它；validate_curriculum_evidence.py 也不檢查非課程頁的正文。照第 6、7 步做，這些頁面會在新紀錄旁邊繼續顯示舊數字，沒有任何檢查會發現。課程頁不受影響：紀錄一改，程式或圖的 SHA-256 也會變，review_coverage.py 會要求重審。 |
| 2 | 建議 | docs/preparation/publish.md 第 6 節第 11 步（第 245 行） | 「現有的這份是上一版的驗證，先改名為 curriculum-publication-v<上一版的版本號>.json 保留」和審查用的事實與寫作規範清單定的發布狀態相反：沒有頁面引用的舊驗證紀錄不留在工作樹，要看就去 git 歷史。改名後的檔案沒有任何頁面引用：目前只有 docs/status.md 引用發布驗證紀錄，它連的是 curriculum-publication.json。同類的 curriculum-publication-v0.2.0.json 也沒有頁面引用。照這一步做，每次發布都會在工作樹多留一份沒有頁面引用的舊紀錄。 |
| 3 | 建議 | docs/preparation/publish.md 第 1 節（第 7–15 行）與第 6 節開頭（第 145 行） | 本頁有三個指令會 import 標準庫的 tomllib：`python3 scripts/validate_lessons.py`（經由 review_coverage.py）、`python3 scripts/validate_site.py`、`python3 scripts/review_coverage.py`。tomllib 要 Python 3.11 以上，Pages workflow 用的是 3.12，但本頁沒寫 python3 要哪一版。Zensical 0.0.67 只要求 3.10 以上，所以在 python3 是 3.10 的系統（例如 Ubuntu 22.04）上，第 1 節的安裝與建置都會成功，第 7、8 步卻會停在 `ModuleNotFoundError: No module named 'tomllib'`，訊息看不出原因。 |
| 4 | 建議 | docs/preparation/publish.md 第 6 節第 5 步的 `--bundle`（第 192 行）與第 6 步（第 194 行） | 第 6 步說「`--push` 與 `--bundle` 帶回的就是這整份清單」，讀起來兩者等價，其實不是。`tar -xzf` 會直接覆寫同名檔，而清單裡的頁面與 notebook 是記錄用那台電腦上的版本。平常編輯的電腦若在第 3 步之後改過其中一頁（例如等紀錄時修了錯字），解開後那個修改會被蓋掉，而且沒有任何提示。`--push` 加上 `git pull` 則會合併，或提示衝突。 |
| 5 | 建議 | docs/preparation/publish.md 第 6 節第 7 步（第 204–208 行） | `review_coverage.py --write <頁面路徑>` 要的是從 repo 根目錄算起的路徑（docs/…）。但同一段前面寫「其他頁把路徑的 / 換成 -，例如 reviews/preparation-publish.md」，那裡的「路徑」指的是 docs/ 底下的路徑。讀者照這個意思傳 `preparation/publish.md`，script 會在 relative_to 丟出 ValueError traceback（已在暫存副本實測），訊息看不出該怎麼改。 |

### 第 4 次查核：有必要問題

結論：有 1 個必要，所以這次不通過。第 7 步把〈驗證範圍〉說成「只說明課程頁的兩種 AI 審查」，但 docs/status.md 第 83–86 行其實同時說明了課程頁和其他 17 頁的審查。另外有 3 個 should：第 6 步對 git pull 行為的描述、第 5 步 Fashion-MNIST 下載的寫法、第 11 步沒通過時的處理。其餘修改都核對無誤。

核對過的項目：
- 指令與選項：record_evidence、verify_curriculum、build_lesson_notebooks、review_coverage、validate_curriculum_evidence、validate_site、download_data、verify_release 都在暫存副本跑過 --help。gh workflow run -f 與 gh run download --dir 也核對了。五個 workflow 都只有 workflow_dispatch，Pages 用 Python 3.12，權限與頁面描述相符。
- 第 2 步：在暫存副本執行 `record_evidence.py` 列表，前後所有檔案的 SHA-256 都沒變，確實只列出。grid 圖的比對只在 grid 紀錄沒過期時才做，renderer 不在紀錄綁定的程式裡，都和頁面相符。
- 第 5 步：COMMANDS、download_data.py 的目的地與沿用條件、30.88 MB 都正確；影片檔與 Fashion-MNIST 的腳本確實不寫 docs/assets/diagrams。
- 第 6 步：Changed files 只涵蓋 OUTPUTS 四個路徑，--bundle 的 arcname 不帶 ./。`tar --exclude='docs/lessons/*' --exclude='notebooks/*'` 在 bsdtar 3.5.3 與 GNU tar 1.35（容器）都實測過，只解出紀錄與圖。--render-only 遇到過期紀錄會停下且不寫任何檔，也不碰圖；validate_curriculum_evidence 不檢查圖。
- 第 7 步：review_coverage 的命名規則、可一次給多頁、涵蓋範圍都相符。`--write docs/preparation/publish.md` 可以寫入；`preparation/publish.md` 出現 ValueError；審查檔不存在時是 AssertionError。課程頁的紀錄與審查綁定同一組程式；status、data、gpu-smoke 三頁都沒有 SVG，也都引用紀錄數字。
- 第 9、11 步逐項對照 verify_release.py 與 verify-release.yml，包括網站檢查、四種情況加第 20 章、README 指令、save 的寫法，以及少了 curriculum-publication.json 時會在讀檔中止，都相符。lessons-v0.3.0 裡的這份檔案 source_ref 是 lessons-v0.2.0，所以頁面寫「新 tag 裡是上一版的驗證」是對的；查核指示說它「留在上一版 tag 裡」不正確，編輯照實寫是對的。沒有程式讀 release-bootstrap 紀錄。
- 第 1 節：四支腳本在 Python 3.9 都停在 tomllib 的 ModuleNotFoundError，而且都能用 3.10 的語法解析，所以在 3.10 上也是同樣錯誤。Zensical 0.0.67 要求 Requires-Python>=3.10。

遺留與指示的處理：5 條遺留項目都已正確修正。受程式改動影響的段落沒有本頁的條目。額外指示都已處理。編輯提出的 status.md:29 跨頁矛盾已不成立，status.md 現在寫的是 Linux runner。

其他檢查：
- 沒有新加入 Mac 實測的數字，新數字都是固定值；沒有敘述修訂經過。
- 重新 rsync 後，zensical build --clean --strict、validate_site.py、validate_preparation.py 都 exit 0。
- 第 6 節渲染成單一 ol、11 項；README 的相對連結都能解析；43 本 notebook 都是合法 JSON。
- 有一點無法完全核實：工作樹裡有其他部分同時在改檔，「只改了列出的檔案」無法逐一歸屬；publish.md 本身和暫存副本一致。

相關路徑：
- 本頁：docs/preparation/publish.md
- 對照頁：docs/status.md
- 腳本與 workflow：scripts/verify_release.py、scripts/record_evidence.py、.github/workflows/verify-release.yml
- 暫存副本與紀錄

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 必要 | docs/preparation/publish.md 第 6 節第 7 步第一段末句（第 209 行）：「網站在〈驗證範圍〉向讀者說明的是課程頁的這兩種 AI 審查，改用別的方式審查時，那裡的說明也要跟著改。」 | 這句和 docs/status.md 對不上。status.md 第 83–86 行同時說明兩類頁面的審查：「42 節課程頁」有兩位（模擬初學讀者、技術審查），「其他 17 頁」也有兩位（以該頁讀者身分閱讀、事實查核）。本頁卻寫成〈驗證範圍〉只說明課程頁的兩種審查，而且這句緊接在描述其他頁審查的句子後面，等於暗示其他頁的審查不在那裡。照本頁做的維護者改了非課程頁的審查方式時，會以為〈驗證範圍〉不用改，status.md 第 86 行就會對讀者說錯，也沒有任何檢查會發現。 |
| 2 | 建議 | docs/preparation/publish.md 第 6 節第 6 步第一段（第 196 行）：「這台電腦在第 3 步之後若又改過同一個檔，git 不會直接蓋掉，而是合併兩邊的修改；合併不了就停下，列出有問題的檔案。」 | 平常編輯的電腦在第 3 步之後若已經 commit 了修改，兩邊的分支就分岔了。git 沒設定 pull.rebase 或 pull.ff 時（也就是預設），`git pull` 不會合併，而是停在 `fatal: Need to specify how to reconcile divergent branches.`，也不會列出檔案。這是在暫存副本用 git 2.56.0、空的 global config 實測的結果。只有修改還沒 commit 時，才會照頁面說的停下並列出檔案（`Your local changes to the following files would be overwritten by merge`）。所以「不會直接蓋掉」成立，但「而是合併兩邊的修改」在預設設定下不成立。 |
| 3 | 建議 | docs/preparation/publish.md 第 6 節第 5 步說明段（第 194 行）：「Fashion-MNIST 的紀錄要先用 `scripts/download_data.py` 把 30.88 MB 的原始資料下載到 `data/downloads/fashion-mnist/`；…就直接沿用。」 | 句子沒寫是誰去下載，讀起來像要讀者在 `--run` 之前自己先執行 download_data.py。實際上，record_evidence.py 的 COMMANDS 在重產這份紀錄時，會自動先跑 `download_data.py fetch fashion-mnist`，再跑 run_fashion_cnn.py。讀者若照字面自己去跑、又沒加參數，會被 argparse 擋下，還得自己找出要加 `fetch fashion-mnist`。頁面也沒說記錄的電腦要能連網，以及快取裡有不符的舊檔時，download_data.py 會停下（Existing file failed verification）。其餘內容都正確：30,878,645 bytes＝30.88 MB（manifest 用十進位顯示）、下載目的地、沿用條件。 |
| 4 | 建議 | docs/preparation/publish.md 第 6 節第 11 步最後一段（第 261 行）：「沒通過時，紀錄裡逐項記著結果與每個指令的輸出，可以從中找出…。commit 並推到 main；…」 | 頁面說了沒通過時去哪裡找原因，下一句卻直接寫「commit 並推到 main」，沒說沒通過時該怎麼辦：失敗的結果要不要 commit、修正後要重跑哪一步。第 4 步對失敗的 GPU 紀錄有明確指示（看完後換掉、不要 commit），這一步沒有。而且 bootstrap 檢查的是 tag 裡的 notebook 與 README，這類問題在 main 修好後重跑同一個 tag 也不會通過。〈驗證範圍〉與〈全套實驗與審查〉又把這兩份檔案當作發布驗證紀錄連給讀者。讀者拿到 passed=false 時，不知道該照哪條路走。 |

修正必要問題時處理的項目（修正後由第 5 次複查確認）：

- 檢查者 must（第 7 步只說〈驗證範圍〉說明課程頁的審查）：已修正。改成〈驗證範圍〉與〈全套實驗與審查〉都說明課程頁與其他頁各自的兩種 AI 審查，README.md 也寫了；改審查方式時這幾處都要改。已核對 status.md 第 83–86 行、validation/curriculum.md 第 88 行、README 第 7、78 行；其他非課程頁沒有描述審查方式。
- 檢查者 should（第 6 步 git pull 在預設設定下不會合併）：已修正。指令改成 `git pull --no-rebase` 並說明它的作用。在暫存副本實測過分岔、衝突、未 commit 修改，以及 pull.ff=only／pull.rebase=true 的情況，頁面描述都成立。
- 檢查者 should（第 5 步 Fashion-MNIST 下載像是要讀者自己跑）：已修正。改成 `--run` 會先執行 `download_data.py fetch fashion-mnist`，並寫明連網條件、沿用條件，以及遇到不符的舊檔會停下（訊息指名檔案，移走後重跑）。
- 檢查者 should（第 11 步沒寫沒通過時怎麼辦）：已修正。寫明兩份都 passed=True 才 commit；有一份沒通過就都不 commit，並附理由與還原指令；接著分成 tag 以外的原因（重跑 workflow 再 save）和 tag 內容的原因（改用新 tag 從第 1 步重來，第 9 步的 git rm 不必做）。
- 上一輪第 1 項建議（重產紀錄後，非課程頁引用的紀錄數字不會被 review_coverage 列出）：已在先前修好。第 7 步第 4 段點名〈驗證範圍〉、〈資料規劃〉、〈GPU／checkpoint 實測〉與 README，要求搜尋後逐一核對並重審。已確認這三頁都沒有 SVG，而且都引用紀錄數字。
- 上一輪第 2 項建議（改名成 curriculum-publication-v<版本>.json）：已在先前修好。第 11 步改成直接覆寫，上一版用 `git log -p` 查，沒有改名的指示。
- 上一輪第 3 項建議（python3 要用哪一版，tomllib 需要 3.11 以上）：已在先前修好。第 1 節第 2 段與第 6 節開頭寫明 Python 3.12，並說明 tomllib 的原因。已確認 review_coverage、validate_site、verify_release 直接 import tomllib，validate_lessons 經由 review_coverage 間接用到。
- 上一輪第 4 項建議（--bundle 解開時會覆寫同名檔）：已在先前修好。第 6 步第 2 段說明 tar 會覆寫，改成只解開紀錄與圖，再執行 `--render-only`。
- 上一輪第 5 項建議（review_coverage --write 的路徑要從 repo 根目錄算）：已在先前修好。第 7 步附了 `--write docs/preparation/publish.md` 的完整例子，並說明不加參數列出的路徑可以直接用。
- 受程式改動影響的段落：沒有 docs/preparation/publish.md 的條目，不需處理。讀過程式修改清單裡和 record_evidence／render_learning 相關的部分，都不影響本頁的敘述。
- 額外指示（第 11 步改成覆寫並刪掉改名指示）：已在先前完成，這次又核對了一次。查核指示單說上一版的驗證「留在上一個 tag 裡」，這點不採用，因為不正確：驗證在建立 tag 之後才 commit，所以 lessons-v0.3.0 裡的是 v0.2.0 的驗證。頁面維持正確的寫法：「新 tag 裡的這個檔案也是上一版的驗證」，上一版可以從 git 歷史查到。
- 額外指示（核對每一步的指令、選項與路徑）：已完成。在暫存副本對所有相關腳本跑過 --help，也讀了 pages／prepare-lfs／deployment-gpu／gpu-smoke／verify-release 五個 workflow。除了上面四項修正，沒有其他不符。
- 額外指示（第 5 步 Fashion-MNIST 下載要對照 record_evidence.py）：已核對並改寫。COMMANDS 先跑 download_data.py fetch fashion-mnist，再跑 run_fashion_cnn.py；30,878,645 bytes 等於 30.88 MB，下載目的地是 data/downloads/fashion-mnist/，也就是 loader 的預設資料夾。

### 第 5 次複查：通過

結論：通過，沒有必要問題，只有 3 個建議。

**建置與渲染**
- 在暫存副本用自建的 Python 3.12 與 Zensical 0.0.67 環境，`zensical build --clean --strict`、`validate_site.py`、`validate_preparation.py` 都 exit 0。
- README 的 9 個相對連結都存在，43 本 notebook 都是合法 JSON。
- 渲染出的第 6 節是一個 ol、共 11 項；第 6、11 步新增的段落、程式區塊與清單都在各自的項目裡。
- 這個部分只有 docs/preparation/publish.md 有 diff。repo 裡沒有多出測試用的檔案，只有其他程式部分新增的 verify_release.py 與 verify-release.yml。
- `validate_lessons.py` 與 `review_coverage.py` 在暫存副本 exit 1，原因是 59 頁都還沒有審查。審查用的事實與寫作規範清單寫明審查到發布時才定稿，所以這是目前的暫時狀態，和本頁無關。

**實際照做過的步驟**
- 第 6 步的 `git pull --no-rebase`：用 git 2.56.0、空的 global config 試了七種情況。兩邊改不同的行會自動合併；同一行衝突會列出 CONFLICT 檔名；這台電腦有未 commit 的修改時會列出檔案並中止；設了 pull.ff=only 或 pull.rebase=true 仍然會合併；只有一邊有新 commit 時直接 fast-forward。plain `git pull` 遇到兩邊分岔會停在 fatal。頁面的描述都成立。
- 第 6 步的 tar 指令：用 record_evidence 同樣的 tarfile 寫法打包，再用 bsdtar 3.5.3 執行頁面的 `tar --exclude=... -xzf`。頁面與 notebook 沒有被解開，紀錄與圖都解開了。
- 第 7 步的 `review_coverage.py --write`：給 `docs/preparation/publish.md` 會正確記錄；給 `preparation/publish.md` 會丟出 ValueError，和頁面要求「從 repo 根目錄算起」一致。
- 第 5 步的資料下載：放一個不符的舊檔再執行 `download_data.py fetch`，會停下，訊息會指名那個檔。
- 第 11 步的還原：`git restore --staged --worktree` 能還原成上一版；對已經不在 git 裡的檔執行 `git rm` 會 fatal，所以頁面寫「第 9 步不必再刪」是對的。

**對照程式核對過的敘述**
- 七支相關腳本都跑過 --help，五個 workflow 都讀過，頁面用到的選項都存在。
- 用到 tomllib 的是 review_coverage.py、validate_site.py、verify_release.py，validate_lessons.py 經由 review_coverage 間接用到；Zensical 只要求 Python 3.10 以上；用系統的 Python 3.9 實測，四支都停在 tomllib 的 ModuleNotFoundError。
- 五個 workflow 都只能手動啟動。
- verify_release.py 的兩項檢查、save 的行為、核對舊 tag 的方式，以及「少記或記錯舊 tag」的兩種情況，都和頁面相符。
- 第 20 章的環境格確實會安裝 ONNX 套件。
- record_evidence.py 先執行 `download_data.py fetch fashion-mnist`，下載到 data/downloads/fashion-mnist/。
- 哪些腳本會重畫 docs/assets/diagrams/ 的圖，和頁面所寫的一致；Pages 建置前的檢查都不比對圖的內容。
- 沒有程式讀 curriculum-release-bootstrap.json；而且在這個檔已刪除的暫存副本上，validate_site 仍然通過。
- 上一版的 curriculum-publication.json 欄位格式與 verify_release.py 相容。

**遺留項與額外指示**
- 遺留的 5 個建議在目前的頁面上都已修好。
- 受程式改動影響的段落沒有本頁的條目。
- 額外指示都已處理：第 11 步改成覆寫、刪掉改名指示；各步的指令與選項已核對；Fashion-MNIST 的下載與 record_evidence.py 相符。
- 編輯者不採用「上一版的驗證留在上一個 tag 裡」，這個判斷正確：用 git show 查過，lessons-v0.3.0 裡的是 lessons-v0.2.0 的驗證。

**數字與行文**
- 新加的數字只有 30.88 MB：manifest 四個檔共 30,878,645 bytes，是固定值。沒有引入在 Mac 上量到的數字；待重錄數值清單是空的，這樣正確。
- 沒有敘述修訂經過。內容和審查用的事實與寫作規範清單、status.md、validation/curriculum.md、README 一致；README 的發布步驟較簡略，但把細節導向本頁，沒有矛盾。
- 頁面標點與中英文之間的空格都符合慣例。

**判定不需修改的一點**
編輯者提過：某個 job 中途當掉、沒有產生結果檔時，save 會停在 AssertionError。頁面寫「兩份都印出 passed=True 才 commit」，這時不會成立，讀者不會因此 commit，所以不列為問題。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/preparation/publish.md 第 6 節第 7 步第 4 段（第 219 行） | 這段把 review_coverage.py 漏掉的頁面寫成「不是課程頁、也沒有顯示重畫的圖的頁面」，讀起來像課程頁一定會被列出。其實課程頁的審查只綁定該節程式和它 import 的模組，所以引用其他紀錄的課程頁也可能漏掉。例子：18-video.md 引用 video-file.json 算出的 IoU 0.50、0.47、0.30、0.46，19-tracking.md 引用同一份紀錄的 ID 序列，20-deployment.md 引用 deployment-gpu.json 的 loss 0.9848→0.0710 與 39.20 秒。我用 repo_dependencies 比對過：scripts/verify_video_file.py 和 miniyolo/deployment_gpu.py 不在這三節的審查綁定裡。只改這兩支程式時，兩份紀錄會重產，但三頁的審查不會失效，它們顯示的 18-video.svg、19-tracking.svg、20-deployment.svg 也不是由這兩份紀錄重畫的。段末要求在 docs/ 與 README.md 搜尋引用處，這個做法本身已經涵蓋這三頁，所以不會讓人做錯事；只是分類寫得不準。 |
| 2 | 建議 | docs/preparation/publish.md 第 6 節第 7 步第 1 段（第 209 行） | 「〈驗證範圍〉與〈全套實驗與審查〉……README.md 也寫著這些審查方式；改用別的方式審查時，這幾處的說明都要跟著改」這份清單少了一處：〈課程大綱〉（docs/planning/outline.md 第 156 行，〈課程刻意不做的事〉）也寫著「教材的審查者都是 AI」。有人照這句改審查方式（例如加入真人審查）時，只會改這三處，課程大綱那句就變成錯的。而且它的文字沒有變，review_coverage.py 不會列出它，沒有任何檢查會發現。編輯者報告說「其他非課程頁沒有描述審查方式」，這點不完全正確。 |
| 3 | 建議 | docs/preparation/publish.md 第 6 節第 1 步（第 149–155 行） | 發布新 tag 時，build_lesson_notebooks.py 只改 42 本 notebook、各節頁面的 Colab 連結和 section-map.json 的 source_ref。但有幾頁在正文直接寫出目前的 tag `lessons-v0.4.0`：README.md 第 7 行，docs/status.md 第 29、51、90 行，docs/validation/curriculum.md 第 114 行。這些正文沒有檢查會看：validate_lessons.py 只比對 Colab 連結與 notebook metadata，validate_site.py 只比對 Colab 配對，第 11 步比對的是公開網站和 tag 的建置，兩邊的舊字樣相同，所以也不會發現。本頁沒有提醒，下一版照這份流程發布時，網站與 README 會繼續寫著舊 tag。這不是本輪修改造成的，但它是本頁所寫的發布流程裡沒人守的一段。 |

最後一次查核的建議事項，在下方〈定稿修正〉逐項處理。

## 定稿修正

上面各項意見與先前查核留下的建議，由 AI 逐項核實後處理：必要問題全部修正，建議事項只在修正明確、範圍小時採用。

| # | 來源 | 意見 | 處理 |
|---|---|---|---|
| 1 | 先前查核 | review_coverage 會漏列的頁面分類不準 | 未改：已經修好：第 219 行已寫出引用別份紀錄的課程頁（第 18、19、20 章）也不會被列出。 |
| 2 | 先前查核 | 描述審查方式的頁面清單少了〈課程大綱〉 | 未改：已經修好：第 209 行已列入〈課程大綱〉。 |
| 3 | 先前查核 | 新 tag 發布時，正文寫死的 tag 沒有提醒要改 | 未改：已經修好：第 155 行已寫明 README、〈驗證範圍〉、〈全套實驗與審查〉要搜尋 lessons-v 手動改。〈全套實驗與審查〉裡的 tag 也保留，所以這份清單仍然成立。 |

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 1 輪：獨立查核之後的編輯

核對第 1 步與 scripts/build_lesson_notebooks.py：它只用 re.sub 改 Colab URL 和 section-map 的 source_ref，最後一格依紀錄填輸出，缺紀錄的節列在輸出最後。核對第 7 步的審查方法、列出的說明頁（status、curriculum、README、outline 第 156 行），以及「紀錄重產後 review_coverage 不會列出哪些頁」：18/19 的依賴不含 scripts/verify_video_file.py，20 的依賴不含 miniyolo/deployment_gpu.py；status、data、gpu-smoke 三頁都沒有 SVG；圖重畫後內容變了才會讓顯示它的頁被列出。這段推論與程式一致。第 9、11 步與 verify_release.py 的 save（下載 site.json／bootstrap.json，存成兩份紀錄並印出 passed）、verify-release.yml 相符。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | 第 7 步第一段：「頁面引用原始論文、官方程式或函式庫文件的說法，另對照頁尾〈參考來源〉連到的固定版本。」 | 這是給審查者照做的程序，但 29 節課程頁沒有這種頁尾行（3.3 的 ResNet §4.1、6.2 的 VOCevaldet、9.1 的 region_layer.c、10 的 yolov3.cfg 等），00、02、6.1、7.4 的文件連結又是 stable。照字面執行，這些頁的論文或官方程式說法不是沒有對照依據，就是會拿會變動的 stable 文件去對照，結果也和 status.md 對讀者說的不一致。 | 已修正：改成可照做的規則：論文看原文，官方程式看固定的 commit 或 tag（頁面連到固定版本時就用那一版），函式庫看官方文件，並在審查紀錄列出查閱的來源與版本。 |
| 2 | 建議 | 第 1 步：「正文裡直接寫出 tag 的地方不會跟著改：`README.md`、〈驗證範圍〉與〈全套實驗與審查〉都寫著目前的 tag」 | 清單漏了 docs/lessons/00-warmup.md〈第一次用 Colab〉第 3 步的「等環境格印出「固定教材版本： lessons-v0.4.0」這一行」。build_lesson_notebooks.py 只改 Colab URL，所以這行不會更新。只照清單改的維護者，會讓新版第 0 章叫讀者等一個舊 tag 的訊息。後面雖然有「搜尋 lessons-v」，但列舉看起來像是完整清單。 | 已修正：清單加上第 0 章〈一次學習的超短暖身〉環境格印出的版本，並說明審查涵蓋只略過 Colab 連結裡的 tag，改了那一句要重審那一頁。 |
| 3 | 建議 | 第 7 步：「其他頁對照 repo 的程式、指令、紀錄與頁面引用的來源查核。」 | status.md 與 curriculum.md 告訴讀者其他 17 頁「並實際執行頁面上的部分指令」，這一步卻沒有要求執行指令。這一步自己也說說明與方法要一致；照它做出的審查，會讓那兩頁的說法變成不實。 | 已修正：補上「頁面上有指令的，也實際執行其中一部分」。 |

### 第 2 輪：上一輪的處理與審查紀錄：有必要問題

核對第 1 步與第 7 步：build_lesson_notebooks.py 只改 Colab 網址與 section-map 的 source_ref；在 docs 與 README 搜尋 lessons-v0.4.0，Colab 連結以外只出現在 README 第 7 行、status 第 29／51／92 行、00-warmup 第 17 行、curriculum 第 114 行，清單完整；review_coverage.py 的 digest 只把 Colab 連結裡的 tag 正規化。第 7 步的審查方法與 status／curriculum／README、outline 第 156 行一致，可照做。三項發現都已修好，合規。另讀了本頁的審查紀錄。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | 第 1 步「審查涵蓋只略過 Colab 連結裡的 tag，所以改了第 0 章這一句，那一頁要重新審查。」 | 〈驗證範圍〉與〈全套實驗與審查〉也在 Colab 連結以外寫著 tag（status 第 29、51、92 行，curriculum 第 114 行），改了同樣要重審；這句只點名第 0 章，維護者可能以為另外兩頁不用。第 7 步的 review_coverage.py 仍會列出它們，所以不會悄悄漏掉。 | 已修正：改成這幾頁正文裡的 tag 改了，除了 `README.md`，那幾頁都要重新審查。 |
| 2 | 必要 | reviews/preparation-publish.md〈獨立查核〉第 1 次與第 4 次 | 紀錄的問題：第 1 次查核後面是空白的「必要問題的修正：」；第 1 次的 4 項 should（第 4 步分支、第 10 步驗證、LFS 第 5 步、第 6 步審查順序）在紀錄裡找不到處理（resolve:publish 的 findings 沒被收入）；第 4 次的修正清單只列前 12 項（共 13 項），沒有註明省略。和「每份紀錄列出…每一項的處理」不符。 | 已修正：產生器讀入修正時處理的項目與建議事項的處理（處理後由下一次複查檢查），不再輸出空的清單，每項發現都有對應的處理。不再截斷修正清單。 |
| 3 | 建議 | reviews/preparation-publish.md 第 4 次查核的修正清單 | 內部用語：「派工單說上一版的驗證「留在上一個 tag 裡」」「impact.json：沒有 docs/preparation/publish.md 的條目，不需處理。讀過程式修改清單裡和 record_evidence／render_learning 相關的單元」「遺留 1…遺留 5」。 | 未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 3 輪：上一輪的處理與審查紀錄：有必要問題

第 1 步末句對照 scripts/review_coverage.py：digest 只用 COLAB_TAG 正則，把 Colab 連結裡的 tag 正規化，並切掉執行紀錄區塊。在 docs 與 README 搜尋 lessons-v：Colab 連結以外只有 README 第 7 行、status 第 29／51／92 行、00-warmup 第 17 行、curriculum 第 114 行，清單完整；README 不在導覽，沒有審查紀錄。所以「改了這幾頁正文的 tag，除了 README.md 都要重審」屬實，也解決了上一輪的建議；網站建置通過。全文讀過紀錄。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | reviews/preparation-publish.md〈獨立查核〉第 1 次查核，以及〈上述處理與審查紀錄的檢查〉第 2 項 | 第 1 次查核的 4 項建議（#2 第 4 步「同一個分支」、#3 第 10 步驗證公開網站、#4 第 5 節 LFS 第 5 步漏了 id／bytes、#5 第 6 步審查後要回到哪一步），在整份紀錄都找不到處理：其後的「修正必要問題時處理的項目」20 項是先前審查意見的處理，第 2 次複查也沒提到這 4 項；工作流程紀錄（edit／repair／resolve:publish）裡同樣沒有它們的處理。〈上述處理〉第 2 項卻寫「每項發現都有對應的處理」，和〈驗證範圍〉〈全套實驗與審查〉、README 第 7 行「每份紀錄列出…每一項的處理」都不符。我核對過現行頁面，這 4 項都已解決：第 3 步把修改推到分支，說明另一台電腦也要取得，第 5 步「checkout 第 3 步的分支」；第 11 步改成 Verify published lessons 加 verify_release.py save，並寫明不經過 Colab；§5 第 5 步列出 id、path、bytes、sha256、license；第 7 步末段說明審查後改了程式要從第 1 步重來。 | 已修正：第 1 次查核的表格加上處理欄，逐項寫明這 4 項在現行頁面的位置（依這一輪檢查的核對）；上一輪那項處理說明的範圍以這裡為準。 |
| 2 | 建議 | reviews/preparation-publish.md 第 54、91、106、133、148、163–168 行 | 〈上述處理〉第 3 項寫「已修正：內部檔名…換成白話」，但它點名的「派工單說上一版的驗證…」只變成第 4 次查核摘要的「派工指示說它「留在上一版 tag 裡」不正確」，「遺留 1…遺留 5」也仍在修正清單；另有英文內部說明「…; this round adds step 3 and the GPU-failure handling」「extra, not from the checker, §6 step 4: …(untracked; another agent changed it at 00:05…)」，以及殘句「（log 在同層的暫存副本與暫存副本）」「- 暫存副本與紀錄：暫存副本、暫存副本」「讀過查核範圍清單裡…相關的單元」。 | 修正者留下的英文處理說明保留原文，只有路徑換成「暫存副本」。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |
| 3 | 建議 | docs/preparation/publish.md 第 155 行（第 1 步末句） | 「…所以改了這幾頁正文裡的 tag，除了 `README.md`，那幾頁都要重新審查。」同一句裡「這幾頁」「那幾頁」來回指涉，而「這幾頁」又包含不是網站頁面的 README.md，讀起來不順。 | 已修正：改成「改了這些地方的 tag 之後，`README.md` 以外的三頁都要重新審查」。 |

### 第 4 輪：上一輪的處理：有必要問題

第 1 步末句「審查涵蓋只略過 Colab 連結裡的 tag，所以改了這些地方的 tag 之後，README.md 以外的三頁都要重新審查」，對照 scripts/review_coverage.py（digest 只用 COLAB_TAG 正規化 Colab 連結，並切掉執行紀錄區塊；README 不在導覽，沒有審查紀錄）與 build_lesson_notebooks.py（只改 Colab 連結與 source_ref）。用腳本在 docs 與 README 搜尋 Colab 連結以外的 lessons-v，只出現在 README 第 7 行、status 第 29／51／92 行、curriculum 第 114 行、00-warmup 第 17 行，清單完整。句子屬實，回應第 3 輪第 3 項（不再「這幾頁／那幾頁」來回指涉），讀來通順。B：第 1 次查核表格的處理欄涵蓋 5 項，逐項對照現行頁面都屬實：第 4 步的 GPU 失敗處理；第 3、5 步的分支；第 11 步的 Verify published lessons 與「兩項都不經過 Colab」；§5 第 5 步的 id／path／bytes／sha256／license；第 7 步末段。必要問題由第 2 次複查確認（紀錄第 71 行）。第 3 輪第 1、3 項屬實；第 2 項點名的「查核指示」「遺留 N」與路徑殘句確實換掉了，但說明中有一句不實。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | reviews/preparation-publish.md 第 264 行（第 3 輪第 2 項處理），對應第 54、87–92 行 | 處理寫「查核者原文的英文說明保留」，但保留的英文不是查核者的原文。第 54 行「…; this round adds step 3 and the GPU-failure handling」與第 87–92 行〈建議事項的處理〉都是修正者寫的處理說明，第 91 行自己就寫「extra, not from the 查核者」。這些行也已被替換工具改過（「not from the 查核者」「the 維護者's own」「in a 暫存副本」），不是原文。下一位編輯照這句說明，會把這些中英混雜的殘句當成原文證據而留著不處理。 | 已修正：處理說明改成照實寫：英文段落裡的內部用語不再替換，路徑仍統一換成「暫存副本」。 |
| 2 | 建議 | reviews/preparation-publish.md 第 19、23、40、56、121、142 行 | 還有未點名的殘句：「我在暫存副本，repo 裡的檔案沒有動」「在暫存副本 validate_site.py 後」「（已在暫存副本）」「publish.md 本身和暫存副本。」；另有中英混雜的「the 維護者's own」。 | 已處理：第 3 輪的處理說明改成統一的說明。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 5 輪：上一輪的處理：有必要問題

全文對照了第 1 次查核表格、第 3 輪與第 4 輪的處理欄。
第 1 次查核表格的處理欄逐項屬實：
- #1：GPU 失敗的處理寫在 publish.md 第 4 步（換掉結果、不要 commit、用 git restore 還原或刪除），第 2 次複查確認過（第 71 行）。
- #2：第 3 步推到分支，第 5 步 checkout 第 3 步的分支。
- #3：第 11 步改用 Verify published lessons 加 verify_release.py save，並寫明「兩項都不經過 Colab」。
- #4：§5 第 5 步列出 id、path、bytes、sha256、license。
- #5：第 7 步末段有說明。
- 處理欄的「最後一輪檢查核對過」對應第 4 輪 B 段。
第 3 輪第 1、3 項與第 4 輪第 2 項屬實。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | 第 4 輪第 1 項處理欄（第 274 行）；第 3 輪第 2 項處理欄開頭（第 265 行） | 第 4 輪第 1 項寫「英文段落裡的字不再被替換」，第 3 輪第 2 項開頭寫「修正者留下的英文處理說明保留原文」。checker、maintainer 已經恢復，但發現點名的第三個例子「in a 暫存副本」仍在第 92 行（- Tests: in a 暫存副本, `zensical build --clean --strict` exited 0…），英文段落裡還有被替換的字。 | 已修正：處理說明改成照實寫：英文段落裡的內部用語不再替換，路徑仍統一換成「暫存副本」。 |
| 2 | 建議 | 第 3 輪第 2 項處理欄（第 265 行） | 「9 處中 6 處已改寫或刪除」把發現引用的第 2 輪處理說法「已修正：內部檔名…換成白話」也算成已改寫，但它仍在第 2 輪第 3 項處理欄（第 256 行）。實際改寫的紀錄文字是 5 處。 | 已處理：用詞類的處理說明改成統一的說明（紀錄保留查核者的原文，只統一替換路徑與內部名稱），不再逐句計數。 |

### 第 6 輪：上一輪的處理：通過

用同樣的方法核對第 3–5 輪處理欄。第 3 輪 #1：第 1 次查核表格第 2–5 列逐項寫了在現行頁面的位置，和第 3 輪發現核對的內容一致。第 3 輪 #3：publish.md 確實有「改了這些地方的 tag 之後，`README.md` 以外的三頁都要重新審查」。第 4 輪 #2、第 5 輪 #2：第 3 輪 #2 已經含統一說明，也沒有計數。原文到紀錄的替換只有路徑、內部名稱、「遺留 N」改成「上一輪第 N 項建議」，以及 A：標籤。修正者的英文處理項目（第 37–56、87–92 行）只有第 92 行和原文不同；查核者、維護者、總評都保留原字。沒有必要問題。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | 第 92 行「- Tests: in a 暫存副本, …」；第 3 輪 #2（第 265 行）、第 4 輪 #1（第 274 行）、第 5 輪 #1（第 291 行）的處理欄 | journal 裡修正者的原文是「Tests: in a scratch copy, …」，並沒有路徑，是產生器的暫存副本規則把英文詞暫存副本換成了「暫存副本」。三則處理寫「只有路徑換成「暫存副本」」「路徑仍統一換成「暫存副本」」，等於把第 5 輪點名的這個例子說成路徑。只看紀錄看不出矛盾，「暫存副本」也確實指同一個暫存位置，讀者不會因此做錯事，所以列為建議。但下一位拿 journal 核對的人還會再報一次。 | 已修正：英文段落裡的「scratch copy」不再替換，第 92 行恢復原文。 |


## 2026-10-05 clear-tutorial 三輪重審

以上是原審查歷史；不追溯改成首次盲讀。這次由固定基線 `16f6910` 分段開放並保存當時理解，再修改、核技術及檢查銜接，詳見 [本輪方法與限制](clear-tutorial/16f6910/README.md)。

- 第一輪：[reference當場閱讀原始紀錄](clear-tutorial/16f6910/first-read/reference.jsonl)，基線來源與圖指紋保留；共享檔案系統不是技術隔離。
- 第二輪：[非作者技術／證據核對](clear-tutorial/16f6910/technical/reference.md)，實際來源、數字及必要執行範圍見該報告。
- 第三輪：[另一位讀者前文→本節→後文複查](clear-tutorial/16f6910/transitions/reference.md)，此輪完整頁閱讀非盲讀；受影響段落及圖另有delta核回。
- [原始卡點與具體處理](clear-tutorial/16f6910/decisions.json)保留未新增的選讀建議。原先前提包漏發及08提前brief的限制另列，沒有算成教材錯或冒稱08全程盲讀。

本輪修正後沒有未解的必要問題；這是AI閱讀／技術查核的實際範圍，不是學生學習成效驗收。全站實際Zensical桌面／手機、公式換頁與執行檢查見 [verification.json](clear-tutorial/16f6910/verification.json)，不以SVG檔存在或strict build取代視覺查核。

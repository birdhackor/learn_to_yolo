# 審查紀錄：公開課程研究

審查範圍：`docs/planning/course-research.md`。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面改寫後，由另一位 AI 獨立查核：對照 repo 的程式、指令、紀錄與頁面引用的來源，實際執行頁面上的部分指令與步驟，並檢查與其他頁的說法是否一致。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：通過

通過，沒有必要問題。

1. 六條先前審查意見都處理了：補了 H1；第 3 行的 subagent、主代理、環境恢復敘述改成一句現在式的來源說明加 AI 揭露（寫法和 feedback.md 的「由 AI 進行」一致）；「這次」「本輪」「修訂建議」都改掉了；連「保留為待查來源」這句待辦口吻也改成現況；/tmp 那段刪了。我掃過全文，沒有剩下的製作或修訂經過、待辦口吻，也沒有寫作平台的細節。「前版模型」「目前提供」「目前證據」都不是在講頁面改版，留著是對的。

2. 事實查核：
   - 我自己用 curl 重查，三個 CS50 AI 的 /2024/ 網址都是 302 轉到現行頁面，現行頁面回 200。
   - 「Michigan EECS 498（WI2022）」和 feedback.md 列的 umich 網址一致，也和那頁的寫法一致。現在那些頁面回 404，但這頁寫的是 2026-10-02 查核當時回 503，有日期限定，不算錯。
   - git ls-files 找不到 Food-11、HW03、GTSRB 等課程資料，「不重新散布課程資料」成立。
   - scripts、tests、.github 都不讀這頁的文字；artifacts/checks/zensical-publication.json 只記這頁的 URL 和 200，沒有內容雜湊。
   - 簡轉繁的字修正都是正確的臺灣用字。

3. 刪掉的內容沒有遺失重要資訊：/tmp 目錄已不存在，「未修改儲存庫」是代理給 owner 的回報，兩者都沒有任何讀者或程式會用到；Michigan 讀不到內容這條限制還在。

4. 只有一個 should：八項的編號清單因為縮排 3 格，渲染成八個各自從 1 起算的清單，畫面上全是「1.」。這是 HEAD 就有的問題，建議改成 4 格縮排，我在暫存副本驗證過可以修好。另外，第 94 行的「各分支」原本可能和第 3 行講代理分工的「各分支」混淆，現在第 3 行刪了，只剩 YOLO 分支一種意思。

5. 渲染：我在暫存副本用指定的 rsync 建副本，這頁和 repo 逐位元組相同。zensical build --clean --strict 以 0 結束（No issues found）；python3 scripts/validate_site.py 以 0 結束，連結與錨點、markdown_rendering、artifact 邊界都 passed。頁面只有一個 h1（公開教材與課程研究），title 是「公開課程研究 - Learn to YOLO」。在編輯的工作時段裡，docs 底下只有 docs/planning/course-research.md 被改過；同時段其他被改的檔案都是 lesson_cases 和 scripts，看起來是其他部分並行的工作。日誌在暫存副本。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/planning/course-research.md 第 9–81 行（八項來源分析的編號清單） | 每項標題下的說明段落和「來源：」行只縮排 3 格。zensical 用的 Python-Markdown（3.11）要縮排 4 格，續行才算在清單項目裡。結果八項各自變成一個只有一個 <li> 的 <ol>：rendered 頁面有 8 個 <ol>，都沒有 start 屬性，主題 CSS 也沒有跨清單的 counter。所以瀏覽器裡八項的編號全是「1.」，說明段落也不在項目裡。這個問題 HEAD 就有，不是這次改出來的；validate_site 的 markdown_rendering 不檢查清單編號，所以照樣通過。讀者看到八個「1.」會覺得頁面格式壞了，和 line 1 那條「看起來像貼上的報告」是同一類觀感問題，但內容不會被讀錯。 |

### 第 2 次查核：通過

（這一輪同時查核：`docs/research/pages-colab.md`、`docs/research/lfs.md`、`docs/research/detection-data.md`、`docs/research/foundation-data.md`、`docs/planning/course-research.md`；下表只列和本頁有關的發現。）

結論：通過。沒有必要等級的問題，只有兩條建議，都在 pages-colab.md。

**這次查核的頁面改了哪些檔**
- 只改了 pages-colab、lfs、detection-data、foundation-data 四頁。
- course-research.md 與 HEAD 相同，HEAD 已經是 4 格縮排。
- 受程式改動影響的段落沒有這五個檔的條目。

**遺留意見的判斷：9 條都正確**
- pages-colab #1–#3、lfs #1–#2、foundation-data：已改，改法和程式一致。
- pages-colab #4：判為「事實已改變」是對的。依據有三：
  - verify_release.py 的 scope 明寫 "Colab itself was not opened"。
  - 審查用的事實與寫作規範清單寫的是 Verify published lessons workflow。
  - publish.md 第 6 節第 11 步現在就是這個 workflow，所以編輯者擔心的 publish.md 落差已經不存在。

**detection-data 的 must（COCO 的 HTTPS 網址）**
- 官方 download.htm 列的是 http:// 網址，頁面連結也是這組。
- 我在 2026-10-04 19:21 UTC 用 HEAD／range 重查：三個檔的官方 HTTP 網址與 S3 HTTPS 路徑都回 200／206，Content-Length、ETag、Last-Modified 兩邊相同；https://images.cocodataset.org 則是 curl exit 60。
- 其他說法也都對得上：
  - DNS 的 CNAME 指向 images.cocodataset.org.s3.amazonaws.com。
  - 憑證是 CN=s3.amazonaws.com，SAN 不含 images.cocodataset.org；urllib 回報 Hostname mismatch。
  - 單張圖片用 S3 HTTPS 也回 200。
- 暫存副本的 coco-probe-final.log（16:43:27Z）支持頁面寫的 16:43 UTC 結果。
- data/manifest.json 的 coco2017 是三個 S3 HTTPS 網址，status 是 candidate，與頁面一致。

**逐條核對程式**
- 對照的程式：validate_preparation、validate_lessons（code_lines、SOURCES）、review_coverage、validate_curriculum_evidence（finite_json、check_machine，以及 custom 1600 步、video、fashion 的內容檢查）、evidence_records、validate_site（含 markdown_list_problems）、build_lesson_notebooks 的 bootstrap()、pages.yml、verify-release.yml、verify_release.py、tests/test_notebook_bootstrap.py、zensical.toml、overrides/、docs/assets/（mathjax.js 訂閱 document$；MathJax 3.2.2）。頁面的敘述都成立。
- 外部來源：
  - PyPI 0.0.67 的 Requires-Python 是 >=3.10，wheel 為 cp310-abi3。
  - Zensical 數學公式文件確有 document$ 與 instant navigation 的整合說明。
  - git-lfs 在 0043a645 的 FAQ 有 `git lfs migrate import --everything`；GitHub 的 LFS 單檔上限單位是 GB。
- git 歷史裡最大的 blob 是 1.17 MB。
- 五頁的 82 個外部連結都回 200 或 206。

**實際照頁面做的步驟**
- 用 Python 3.12 建 .venv-docs、裝 requirements-docs.txt。
- 依序跑 5 個指令：
  - validate_preparation 0。
  - validate_lessons 只因所有導覽頁都「no review」而失敗。
  - validate_curriculum_evidence 只因紀錄還沒重產而失敗（00-warmup 的 case_sha256 不符）。
  - zensical build --clean --strict 0。
  - validate_site 0（含 numbered lists 檢查）。
- validate_lessons 與 validate_curriculum_evidence 的失敗都屬於審查用的事實與寫作規範清單列的「發布時才完成」狀態。
- zensical serve 預設在 localhost:8000，根路徑會轉到 /learn_to_yolo/。
- 用 Python 3.9 實跑：validate_lessons 與 validate_site 確實停在 `ModuleNotFoundError: No module named 'tomllib'`；用 3.10 語法解析這些腳本都通過。
- tests/test_notebook_bootstrap.py 6 項通過。

**數字**
- 沒有新加入 Mac 上量的數字，也沒有依紀錄而變的數字。
- 頁面的 byte 數、換算與加總都正確。
- 用 SHA-256 為 52425fb6… 的 annotations.tar.gz 重驗 Pet 的盤點：3,686／7,390／3,680／3,669／9／41，以及 Abyssinian_1 的 [107,81,444,328)、[92,66,460,343)、CRC，全部相符。

**敘述與一致性**
- 沒有敘述修訂經過，範圍限制都寫成現況。
- 與 status.md、validation/curriculum.md、architecture.md、publish.md、data.md、README 一致。
- 「像素」改「畫素」與術語表一致；連結文字〈驗證範圍〉與 status.md 標題相同。

**渲染**
- README 的 9 個相對連結都存在；43 本 notebook 都是有效 JSON。
- course-research 建置後只有一個 8 項的 &lt;ol&gt;，8 行「來源：」都在項目內；headless Chromium 截圖顯示編號 1–8。

**檔案位置**
- 紀錄與截圖：同路徑的暫存副本（prep.log、lessons.log、evidence.log、build.log、site.log、links.log、course-research-top.png）

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 2 輪：上一輪的處理與審查紀錄：通過

讀了紀錄：第 2 次查核寫「course-research.md 與 HEAD 相同，HEAD 已經是 4 格縮排」，唯一的建議有交代。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/planning-course-research.md | 「建議事項在下方〈定稿修正〉逐項處理」指向不存在的段落；修正後的檢查是別批頁面的內容；「六條 trace 都處理了」「看起來是其他單元並行的工作」。 | 已修正：沒有〈定稿修正〉時不再寫指向句；修正後的檢查只掛在這一批真的有改動的頁面。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 3 輪：上一輪的處理與審查紀錄：通過

以腳本核對紀錄：第 1 次查核的建議，由第 2 次確認已修好（「course-research.md 與 HEAD 相同，HEAD 已經是 4 格縮排」）；不再指向不存在的段落，也不再掛別批的檢查；頁面沒有指令。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/planning-course-research.md 第 26、97 行 | 殘句：「日誌在暫存副本和暫存副本.validate.log。」「- 暫存副本- 紀錄與截圖：同路徑的暫存副本（prep.log、…」。 | 未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 4 輪：上一輪的處理：有必要問題

第 3 輪第 1 項：第 26 行已改成「日誌在暫存副本。」；第 97 行只改了一個字元，處理說明不實。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | reviews/planning-course-research.md 第 97 行 | 點名的「- 暫存副本- 紀錄與截圖：同路徑的暫存副本（prep.log、…」只少了一個「- 」，變成「- 暫存副本紀錄與截圖：同路徑的暫存副本（prep.log、…）」，仍是殘句。處理卻寫已清理。 | 已處理：第 3 輪的處理說明改成統一的說明。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |


## 2026-10-05 clear-tutorial 三輪重審

以上是原審查歷史；不追溯改成首次盲讀。這次由固定基線 `16f6910` 分段開放並保存當時理解，再修改、核技術及檢查銜接，詳見 [本輪方法與限制](clear-tutorial/16f6910/README.md)。

- 第一輪：[reference當場閱讀原始紀錄](clear-tutorial/16f6910/first-read/reference.jsonl)，基線來源與圖指紋保留；共享檔案系統不是技術隔離。
- 第二輪：[非作者技術／證據核對](clear-tutorial/16f6910/technical/reference.md)，實際來源、數字及必要執行範圍見該報告。
- 第三輪：[另一位讀者前文→本節→後文複查](clear-tutorial/16f6910/transitions/reference.md)，此輪完整頁閱讀非盲讀；受影響段落及圖另有delta核回。
- [原始卡點與具體處理](clear-tutorial/16f6910/decisions.json)保留未新增的選讀建議。原先前提包漏發及08提前brief的限制另列，沒有算成教材錯或冒稱08全程盲讀。

本輪修正後沒有未解的必要問題；這是AI閱讀／技術查核的實際範圍，不是學生學習成效驗收。全站實際Zensical桌面／手機、公式換頁與執行檢查見 [verification.json](clear-tutorial/16f6910/verification.json)，不以SVG檔存在或strict build取代視覺查核。

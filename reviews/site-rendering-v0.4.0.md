# v0.4.0 網站修正獨立複查

審查日期：2026-10-05 UTC。審查者：`release_ui_review`。範圍：`zensical.toml` 的 `extra_css`、公式／表格捲動 CSS、MathJax 啟動與瞬間導覽排版。以獨立 rsync 副本 `/tmp/lessons-v0.4.0-reviews/review-ui/` 建置；未修改原始儲存庫、既有 reviews 或 coverage，未使用 Git、憑證、GPU。

結論：原始修正版有一項必修的瞬間導覽字形遺失問題，已由主代理加入 MathJax 官方支援的 `chtml: {adaptiveCSS: false}` 解決。正式行為版本完成全站 118 次靜態檢查、59 頁實際點擊導覽及競態複查後，沒有未解決的必要問題。保留修正前失敗證據，避免只以公式容器數量判定成功。

## 原始 P1：瞬間導覽後公式字元消失，已修正

從直接載入的 `03-identity` 實際點側邊導覽到 `04-coordinates`，原始 JavaScript 出現 `MathJax: can't insert css rule … Cannot read properties of null (reading 'insertRule')`。沿 `04-coordinates → 07-loss → 11-iou-loss → 16-inference-head → 03-identity` 點擊，共記錄 83 個 MathJax 警告；前四頁各取前 20 個 CHTML 字形，分別有 4、5、7、14 個 `::before` 的 `content: none`。頁面上的公式容器、輔助 MathML 與 MathItem 數量仍正確，MathItem 也到 state 200，不能因此宣告畫面正常。

Zensical 瞬間導覽先移除動態 style；MathJax 在 `CHTML.styleSheet()` 嘗試對已拆下的 style 插入新增字形 CSS，當時 `.sheet` 為 null。之後 style 會重新接回 head，但失敗的新規則沒有恢復，原始規則數一直停在第一頁的 177。這是插入規則與重新接回節點的時序問題；完成排版後再查看 `style.sheet`，會看到非 null，不能據此排除失敗。

修法預先產生完整字形與 wrapper CSS，避免導覽後新增規則。正式檔的未包裝 MathJax 真實點擊測試確認：每頁都有 2,788 條規則、取樣字形全部有定義、MathJax 警告為 0。被動 `document$` 觀察也確認導覽事件當下 cached style 曾 `isConnected: false / sheet: false / parent: null`，完成排版後完整 stylesheet 接回。沒有使用內部 API 重綁 stylesheet。

原始失敗：`review-ui-original-exact-style-nav.json`、`review-ui-uninstrumented-nav.json`，以及 `review-ui-original-exact-style-*.png`。正式正常畫面與時序：`review-ui-final-uninstrumented-nav.json`、`review-ui-final-uninstrumented-*.png`。

## 正式版驗證

| 檢查 | 實際結果 |
| --- | --- |
| 建置與站內驗證 | `zensical build --clean --strict` 通過；`validate_site.py` 通過本機連結／anchors、Colab 配對、Markdown、搜尋索引與 artifact boundaries。59 個內容頁加 404 頁，共 60 HTML；42 課；137 artifacts、8,441,244 bytes。 |
| 真實瀏覽器 | Chromium `151.0.7922.173`，以 HTTP 提供獨立建置網站；viewport 390×844 與 1280×844。 |
| 全站靜態檢查 | 59 頁 × 2 寬度＝118 次，1,436 個公式實例；wrapper＝CHTML＝assistive MathML＝MathItem，全部 state 200，位於目前 article 且 connected；巢狀容器、遺失字形、`mjx-merror` 皆 0。 |
| 文件寬度 | 390 viewport 的文件寬 375；1280 viewport 的文件寬 1265，差值為垂直捲軸；118 次皆沒有文件水平溢出。捲動容器內的離屏子元素不列為文件溢出。 |
| 真實瞬間導覽 | 59 頁全部透過現有側邊連結點擊；加上 5 次正常換頁、3 輪快速換頁、4 次返回、2 次前進、延遲排版與 390px footer 前後頁，共 76 個完成快照。`performance.timeOrigin` 持續相同，確認使用 Zensical 瞬間導覽。字形、容器、MathItem 與文件寬度皆正常，沒有 MathJax 警告。 |
| 同頁 anchors | 目錄兩個 anchor 與重複點擊：排版呼叫數 26→26，沒有再次清除／排版目前公式。 |
| 排版序列與舊頁清理 | 300ms 延遲的實際點擊序列最大同時排版數 1；2 個排版中的舊 article 斷開後，目前頁項目正常。額外 1500ms 排隊測試以實際 pointer click 產生 5 次導覽，只執行 2 次排版，明確跳過 3 個已斷開的排隊 article；最後 identity 保有 31 wrappers／31 MathItems、0 missing glyph、0 nested。沒有 Promise cycle。 |
| 窄螢幕捲動 | `03-identity`、`03-projection`、`04-localization`、`04-coordinates`、`07-loss`、`11-iou-loss`、`16-inference-head`、`research/lfs`，共 24 個超寬公式／表格，打開 details 後實際鼠輪與 Tab／ArrowRight 均到最右端，右方式子或欄位可達。 |
| 程式碼 | 原有 code 的 `overflow-x: auto` 與 tabindex 保留。projection 的 775px 程式碼在 375px 可視寬內，實際鍵盤到 `scrollLeft=400`，文件寬仍 375，右端可讀。 |
| 輔助閱讀 | 所有公式保留一份 assistive MathML，視覺 CHTML 的重複內容 aria-hidden。identity 的 Chromium AX tree 有 29 個未忽略的 `MathMLMath`，並有 Identifier／Operator／Number／Fraction 等節點；另 2 式在收合的 details。驗證範圍為 DOM 與瀏覽器 accessibility tree，未執行外部螢幕閱讀器的語音讀出。 |
| 行內分式可讀性 | 額外檢視 `07-loss` 的 sigmoid 分式。雖 wrapper 的 scrollHeight 大於 clientHeight，分子、分母與實際字形 bounds 都在可視 wrapper 內，截圖完整；沒有把匿名 line box 的捲動尺寸誤判為文字裁切。 |

主控台仍觀察到主題 repository widget 自動請求 `https://api.github.com/repos/birdhackor/learn_to_yolo/releases/latest` 的既有 404。所有本機資產正常；正式複查沒有 MathJax 或程式例外。這個外部資料回應與本次公式／捲動修正分開記錄，沒有將「MathJax 無警告」寫成「整體主控台完全無訊息」。

## 檔案綁定與證據

118 次全站掃描與 76 個導覽快照執行的正式行為版 MathJax JS SHA-256：`c08808fa8216306e7b91e217857c5532458a164e1ac3fce187b2e1d040ed09ed`。

最後只精確化一行註解，最終 root、副本與重新建置的 JS SHA-256 都是 `8bfec822da079692e1dcf9ffe5369dad43fbfedcc77eacf983f94beac3cc5f05`。移除整行註解後，測試版與最終版逐字相同，SHA-256 同為 `17452629da86b5fb7de85673df340009726e1462b024f6e67e91348d04181649`；最後重新 strict build 與站內驗證通過。未因註解修改假稱重跑 118 次。

主要 JSON：`review-ui-summary.json`（彙整）、`review-ui-evidence.json`（118 次完整 DOM／幾何）、`review-ui-final-navigation.json`（76 快照／排版時間線）、`review-ui-queued-navigation.json`（跳過舊 article）、`review-ui-final-uninstrumented-nav.json`（無排版 wrapper 的真實點擊）、`review-ui-interactions.json`（24 次真實捲動）、`review-ui-accessibility.json`（AX tree／code）、`review-ui-fraction.json`（分式 bounds）、`review-ui-js-binding.json`（雜湊）。命令證據為 `review-ui-final-build.log`、`review-ui-final-validation.json`。所有檔案位於 `/tmp/lessons-v0.4.0-reviews/`。

## 判斷依據

- 已讀副本 `AGENTS.md`，遵守網站 strict build／驗證流程與繁體中文審查。
- MathJax v3.2 官方 typeset 文件快取 `/tmp/lessons-v0.4.0-mathjax-web-typeset.rst`：非同步排版應串接 Promise；清除已移除內容用 `typesetClear()`；它本身不移除既有 DOM；可用 `getMathItemsWithin()` 檢查項目。
- MathJax v3.2 官方 startup 文件快取 `/tmp/lessons-v0.4.0-mathjax-startup.rst`：`typeset: false` 與 `input: ["tex"]` 是有效設定；`pageReady()` 可回傳排版 Promise。檢視 bundle 的 `typesetPromise()` 實作不讀取 `startup.promise`，因此目前 pending 寫回不造成自我等待。
- MathJax-src 3.2.2 官方 `ts/output/chtml.ts` 快取 `/tmp/lessons-v0.4.0-mathjax-chtml.ts`，170–201 行：只有 adaptiveCSS 開啟時才對 cached stylesheet `insertRules`；false 使用完整 wrapper styles。另與網站 vendor bundle 的 `CHTML.styleSheet()`、`HTMLDocument.addStyleSheet()`、`HTMLAdaptor.insertRules()` 實作交叉核對。
- Zensical 實際產物 `assets/javascripts/bundle.67a597eb.min.js` 的 head 更新路徑與真實 DOM 觀察支持上述時序；未僅依文件推論導覽成功。

測試涵蓋 Chromium 及本次網站變更；沒有擴張到模型、課程執行、GPU 或遠端發布。

## 發布保存

彙整與逐頁結果保存在 [`artifacts/checks/site-rendering-v0.4.0.json`](../artifacts/checks/site-rendering-v0.4.0.json)：118 次直接載入與 76 個導覽快照的完整計數、字形／項目／文件寬檢查均保留，省略重複的 MathItem 文字及 DOM 座標。原始暫存 JSON 的 SHA-256 也記在裡面；暫存截圖與全文 DOM 未放入 Git。這份紀錄限於網站呈現修正，不替代逐頁教材審查或發布後的公開 HTTP 檢查。

### 修正後的獨立複查

2026-10-05，由與原 UI 審查者不同的 AI，在 `/tmp/lessons-v0.4.0-reviews/fix-20-ui/` 的獨立副本複查正式 MathJax JS 與溢位 CSS。以指定 rsync 複製 base；root 只讀，沒有修改 coverage、Git、遠端或 GPU。只用 read-only `git show HEAD:docs/assets/javascripts/mathjax.js` 取得舊版對照。自有 HTTP port 8913 已停止，沒有操作 8794。

**首次 double typeset、導航後字形 CSS 失效、舊頁 MathItems 清除與指定頁面的手機溢位均通過，沒有新必要問題。** 嚴格建置與 `validate_site.py` 均 exit 0（Zensical 0.0.67，60 頁／42 課）。Playwright 用既有 docs Python、`/usr/bin/chromium` 與指定啟動參數。

正式未 instrument 的新 browser context 首次只開 03 identity，再用原生 sidebar `locator.click()` 點 04 coordinates→07 loss→11 IoU loss→16 inference head→20 deployment。整路只有一次 document request，`window.documentMarker` 持續保留，證明是瞬時導航。每頁等待排版 promise、展開 details，查 DOM、MathItems、glyph 的 `::before content` 與 stylesheet：

| 頁面 | 公式／容器／MathItems | nested | 舊頁或不屬現頁的 MathItems | 缺字 |
|---|---|---:|---:|---:|
| 03 | 31／31／31 | 0 | 0 | 0 |
| 04 | 25／25／25 | 0 | 0 | 0 |
| 07 loss | 74／74／74 | 0 | 0 | 0 |
| 11 IoU loss | 100／100／100 | 0 | 0 | 0 |
| 16 inference head | 6／6／6 | 0 | 0 | 0 |
| 20 | 13／13／13 | 0 | 0 | 0 |

每頁一個 connected 的 `#MJX-CHTML-styles`，2,788 條規則，adaptiveCSS 實際為 false。逐一查所有 `mjx-c`，排除官方設為空字串的 U+2061 無形函式套用符後，沒有缺字。再快速連點 07→11、不等中間的 MathJax，browser back 返回 07、點同頁標題錨點；最終公式數 100、74、74，nested、缺字與舊 MathItems 皆 0。正式導航與手機檢查無 MathJax warning／error、無 pageerror；唯一 console error 是既有 GitHub releases/latest API 404，實際 URL 已記錄，不是本地公式資產失敗。

獨立重現舊版：另開新 context，只攔截該 JS 請求為 root HEAD 原版，其餘同站資產不變；原生點 03→04→07。03 首次 31 公式卻有 62 容器、nested 31；04 缺 75 glyph、07 缺 351 glyph，共 59 個 MathJax CSS 插入警告（insertRule null）。原版 JS SHA-256 `3e8998260905693dee9d5e901bbf0ac9c0ac51395c061c6ec79d6e7f4e3dcb5c`。這不是只繼承原 UI JSON 的結論。

正式主路徑結束後另做 controlled stress：觀測排版／clear，延遲每個排版呼叫 400 ms，在 04 工作開始後點 11→16。最大活動工作 1，只有 04 與最終 16 進入排版，已斷開的中間 11 被略過；最終 6 個 MathItems，無 disconnected／unrelated 項。這個有包裝的壓力測試與上述未 instrument 導航分開記錄。

重新取得並讀取官方 [MathJax-src 3.2.2 chtml.ts](https://github.com/mathjax/MathJax-src/blob/3.2.2/ts/output/chtml.ts) 第 155、170～201 行及 [FontData.ts](https://github.com/mathjax/MathJax-src/blob/3.2.2/ts/output/chtml/FontData.ts) 第 180～197、223 行起：adaptiveCSS=false 使用全部 wrapper、delimiter、variant 字形規則，避免導航後再對失效 stylesheet 插入規則。[官方 v3.2 startup](https://github.com/mathjax/MathJax-docs/blob/v3.2/options/startup/startup.rst) 與 [web/typeset](https://github.com/mathjax/MathJax-docs/blob/v3.2/web/typeset.rst) 核對禁用預設首次排版、串連非同步呼叫及移除 MathItems。正式 JS 的 typeset:false、同 article 略過、pending 串連、isConnected 檢查及 typesetClear 均與來源和實測一致。來源均 HTTP 200；首次嘗試不存在的 v3.2-latest／短 startup 路徑回 404，改用正確官方路徑後成功。

390×844 手機 context 的六頁均展開 details，document scrollWidth／clientWidth 全為 390／390，nested、缺字皆 0。超寬公式及表格的內部 scrollLeft 均可移動。再用原生水平 wheel 操作 11 頁：公式 client358／scroll438、scrollLeft 0→80；表格 client358／scroll672、scrollLeft 0→220。前後截圖已目視確認右側內容可讀。首次表格 wheel 落到 sticky header，調整臨時 probe 落點至表格正文後重跑通過，沒有把錯誤落點當成 CSS 問題。

臨時 probe 首次也曾因 runtime 將 href 轉為絕對路徑而 selector 逾時；修正 selector 後從頭完成。以上只記成功完成的正式輪次。證據與實際腳本在副本 `artifacts/fix20ui-browser.py`、`fix20ui-old-browser.py` 和 `artifacts/runs/fix20ui/` 的 `browser.json`、`old-browser.json`、前後截圖、來源快照與雜湊清單。

| 快照檔案 | SHA-256 |
|---|---|
| 正式 `mathjax.js` | `8bfec822da079692e1dcf9ffe5369dad43fbfedcc77eacf983f94beac3cc5f05` |
| 正式 `extra.css` | `bd285e60fca6236ecca730a05c9ebe0320645bd2d6261a973a97dcf8f123aca6` |
| `zensical.toml` | `43607d3f13b9e47f3accb3b6e7f3d16c7a8c55e849e0080ea93361f178fb7d68` |
| vendor `tex-mml-chtml.js` | `300480069078b5892d2363a2b65e2dfbbf30fe5c80f83edbfecf4610fd093862` |
| 官方 3.2.2 `chtml.ts` | `754eb7513f55d4bfedd12092a2d5d8d7c8b729201cea89a6b855c291860e76ee` |
| 官方 3.2.2 `FontData.ts` | `7948bbe4339ccf1f1fe5b917fa2178f5515c26b31985c8307101859f0d3d4ce6` |
| 官方 v3.2 `startup.rst` | `ab8b338782366e69fb3b04e04944dbcc5f3bfff1434044b8fb141a14eb7747ba` |
| 官方 v3.2 `web/typeset.rst` | `59639f3dbbb9d854fe59f9d4ac964af28dafc8784be802ba6c71287cef190562` |

結束前正式頁面、JS、CSS、設定、vendor 都與 root 逐 byte 相同。限制：只測指定六頁與上述操作，不代替全站掃描；未測其他瀏覽器、實體手機手指滑動、離線或發布後網站，也未觸發遠端／GPU。沒有覆蓋原作者證據。

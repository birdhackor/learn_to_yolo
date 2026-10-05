# 協調者實際頁面視覺複查

Chromium 開啟本機 Zensical build 的全部59頁，1280×900與390×844，共118組；112次頁內圖片均載入，沒有整頁水平溢出、未渲染公式、巢狀MathJax或pageerror。`whole-site.json`保留每頁實際尺寸；這些自動檢查不代表已逐字目視全部圖。

協調者另外實際看過頁內截图：390寬的07兩圖、08兩圖、CNN素材、09anchors、12assignment、11CSP、11augmentation、發布流程；1280寬的07曲線、08預測圖。07四條曲線、16×16框的FP/FN及08三種定位失敗可在正文寬度讀到；圖片不需點原圖才能讀主要例子。舊generated複合圖仍可選讀，沒有冒稱改大全部舊圖。六張較早機制圖另有獨立讀者的實际頁面檢查。

頁內 `img.screenshot()` 為長圖捲動後擷取整個元素，可能把浮動頁首或右下的導覽按鈕一併截入；保存截圖中的浮動元件不是SVG原圖內容。長圖也需要正常捲動閱讀，並未聲稱手機一個畫面顯示整圖。這一輪59頁snapshot之後僅07-heldout圖說有兩個字詞delta（類別紅／藍），final build後另實際查該頁，不冒稱舊HTML就是最後版。

`math-navigation.json`實際包含桌面9次／手機6次換頁、返回與錨點觀察，以及快速點擊和延遲typeset佇列；沒有缺字、巢狀或脱離當頁的公式，stress maxParallel=1。窄螢幕長公式及表格有原生水平捲動，並非把溢出截掉。測試用的documentMarker仍沿用舊腳本的字面名稱，實際BASE與Colab pin是本機v0.5預览，不拿marker名稱當發布證據。

這些是AI與自動化檢查，沒有實際學生的學習成效資料。

## 公開新版追加

`public-pages/results.json`實開首頁、00、01、07-heldout、08-own-data、13-dual，共6頁×桌面/手機12組；圖載入、頁面溢出、公式、42節模式的Colab tag及每張出現的SVG bytes逐一核。首頁是獨立環境檢查notebook，刻意跟main，不是42節之一；最初的測試腳本把它錯套42節的pin斷言，修正測試範圍後重跑，沒有因測試假設改已發布tag。Playwright的APIRequest曾選到不可路由IPv6；改用頁面實際Chromium fetch核bytes，後續12組完成。

協調者實際看公開首頁、01素材入口、08主例入口的390頁面截圖。公開MathJax完成9桌面、6手機加queue stress與原生長公式/表格捲動；沒有JS pageerror或缺字/巢狀/脱離當頁公式。console有桌面/手機各一次GitHub `releases/latest` 404（repo只有tags，沒有GitHub Release metadata）；不是教材頁、圖檔或MathJax404。原console保留，不寫成零console錯誤。

公開全文、全部圖檔和42notebook的內容仍以發布後Actions逐檔比對為證；6頁browser只支持這個實際呈現範圍，不冒稱公開全部59頁又逐字目視一遍。

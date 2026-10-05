# Foundations 作者修訂交接

首次閱讀工作先完成：110/110 單位，原始記錄保留於 `reviews/clear-tutorial/16f6910/first-read/foundations.jsonl`，摘要 `/tmp/clear-tutorial-foundations-summary.md`。以下是作者修訂與作者自查，**不是獨立技術審查或新一輪首次閱讀通過**。

## 修改範圍

只修改 `docs/lessons/11-csp.md`、`11-fusion.md`、`11-augmentation.md`、`11-iou-loss.md`，改寫專用 `docs/assets/diagrams/11-fusion.svg`、`11-iou-loss.svg`，新增專用 `11-bilinear-coordinates.svg`、`11-crop-resize-target.svg`。未改 `lesson_cases`、`miniyolo`、`scripts`、`notebooks`、`reviews`；沒有 commit、push 或 GPU 工作。

修改前核對 `scripts/verify_curriculum.py`：`SITE_FIGURES` 只複製 17-capstone、18-video、19-tracking；在 `scripts`、`lesson_cases`、`miniyolo`、`.agents` 用 rg 搜尋本組既有圖名，沒有找到會重產這四張第 11 章圖的程式。新／改 SVG 均有 title、desc、viewBox，且加上自然 width/height，手機用正文既有 max-width 縮放。

## 逐 issue 處理

### evolution-0085：CSP 開場的歷史負擔

把完整 CSPNet、DenseNet 梯度說法、YOLOv4/v5、C3、BatchNorm/bias 比較移到兩路 shape、concat/add、成本與實際程式核對之後的「選讀：和原始 CSP、YOLO 的差別」。保留所有原歷史內容與來源連結。主例成本段改為只解釋單一 block 參數比例不能推成全偵測器成本或速度；論文 20% 結果仍在後面的歷史比較，不再在主例前引出長歷史。

來源：原 `11-csp.md`、`lesson_cases/11-csp.py`，以及原文已固定的 CSPNet 論文、YOLOv5 v6.0 common.py/yolov5s.yaml 連結。這次只重排既有歷史主張，沒有重新宣稱外部架構已查證。

### evolution-0095：特徵金字塔名稱與箭頭方向相反

重畫 `11-fusion.svg`：Deep 深粗 4×4 在上，Shallow 淺細 8×8 在下；實線只表示本節四步 reduce→nearest→concat→mix，藍色 top-down 向下、綠色 lateral 橫向進入 concat。圖內及正文明示隨機輸入，沒有 backbone/head/PAN 回程。先看實作；其他節的 backbone/head 接法、原始 FPN add 與 lateral 1×1、PAN bottom-up、v4/v5 完整 neck 比較移到後面的選讀文字，避免把未實作路徑放在主例圖裡。保留第 10 章 channel 16/32 與本例 8/16 的不同、C3/BN/SiLU/第三尺度等簡化限定。

來源：原 `11-fusion.md` 與 `11-fusion.svg`、`lesson_cases/11-fusion.py` 的唯一 forward、`artifacts/checks/curriculum/11-fusion.json` 的四步 shape 與參數 1296。沒有宣稱小物件 AP 或速度改善。

### evolution-0098：nearest 主例後混入多套 bilinear 推導

nearest 的四數矩陣與 sum 梯度 4 留在主文；主文明說本程式只用 nearest。bilinear 的來源座標、align_corners、平方平均梯度改成選讀 details，分設定與 loss 小段，配 `11-bilinear-coordinates.svg`。新圖三行共用來源座標，清楚分開 False 的 −0.25/0.25/0.75/1.25 與 True 的 0、1/3、2/3、1。原插值數字、梯度、AP 插值的區別全部保留。

來源：原段落、Fusion 程式的 mode='nearest' 與 saved stdout；另用現有 CPU PyTorch 對 2×2→4×4 做針對性自查，False 第一列 [1,1.25,1.75,2]、sum 梯度全 4、平方平均梯度 [[0.78125,1.09375],[1.40625,1.71875]] 對上原文。True 第一列符合 1、4/3、5/3、2。這不是新的訓練證據，沒有寫入 artifacts/checks。

### evolution-0107：crop 同一長段混 32 與 64 座標

拆為「在 32×32 座標裁框、判斷 keep」「用同一個 keep 篩 labels，檢查剩下的監督」「從 32×32 接回 64×64，重新分配格子」。必要的 keep 邊界、裁切輸入範圍、漏篩 labels、刪框後未標前景與 objectness=0 限制仍在主文。

新增 `11-crop-resize-target.svg`：32 圖 [0,4,8,20]→同時乘 2→64 圖 [0,8,16,40]，新中心 (8,24)，4×4 每格 16 pixel，藍虛線標 (gx=0,gy=1)。兩站等寬呈現，明示各自座標；圖是依座標畫的示意，resize 是手算，沒有說本節程式已接上 resize 或 targets。

來源：原正文的 resize 手算、`lesson_cases/11-augmentation.py` 的 crop 與整張 tensor 斷言、`artifacts/checks/curriculum/11-augmentation.json`。未新增訓練數據。

### evolution-0115：DIoU 標題讓人預期每方向都拉近

標題改「看中心距離相對於包圍框的尺度」。定義 ρ、c 後立即提醒分母 c² 會變，loss 下降不保證每一步縮短 ρ。保留原梯度推導與實測單步更新；用三列小表比較起點 576/1856、左移 529/1777、往下 577/1889。解釋往下時 C 高 16→17，分母相對增幅抵銷距離平方增加；表是改位置後重算，並非三次 SGD。保留對齊折角 y 梯度實際為 0。

來源：原正文既有三個位置與數字、`lesson_cases/11-iou-loss.py` 的 losses/box_from_center、`artifacts/checks/curriculum/11-iou-loss.json`。現有 CPU 用原函式核對三列均對上 1.310345、1.297693、1.305453；沒有改 loss 實作或原式子。

### modern-0027（由 modern-0024 圖底誤讀延伸）

原 SVG 原始碼與正文皆是正確的 1.179487，沒有把數值改成另一個答案。把 720 寬圖改成 400 寬直排，保留 xyxy、C、ρ、c 與面積，把 footer 長句拆成獨立大字卡：P 左移 1 pixel／C 面積 624、空白 112／GIoU loss／**1.2→1.179487**。圖面幾何仍等比例，只把繪圖倍率從 10 改成 8 並同步 desc/comment；框座標與 loss 數字不變。

來源：原 `11-iou-loss.svg`、原正文 112/624 手算、saved stdout 1.179487。手機正文 inline screenshot 已直接看清新 footer。

### modern-0033：練習第 2 題「呢」範圍不明（optional）

明說 GIoU、DIoU 這兩項只求 loss，不求中心梯度，和既有參考答案一致。沒有添加導數或改答案數字。

## 已完成的作者自查

- `python3 scripts/validate_preparation.py` 通過。
- `python3 scripts/validate_curriculum_evidence.py` 通過（42 lesson + 11 supplementary 與現有程式相符）。四頁 curriculum-evidence footer 逐字與 HEAD 比對相同。
- `validate_lessons.py` 因教材／圖與既有 review 指紋不一致失敗，包含本組四頁和其他正在修訂的頁面；沒有為了讓它通過而改 reviews。
- `.venv-docs/bin/zensical build --clean --strict` 以 `/tmp/clear-foundations-build` 教材副本、相同配置建置；修訂後增量 strict build 亦通過。
- 用未改動的 `validate_site.py` module，只把 ROOT 指向該副本，完整網站檢查通過：60 HTML、42 lessons、連結/anchor、Colab 配對、search、Markdown 與發布檔案邊界。結果 `/tmp/clear-foundations-site-validation.json`。
- Playwright 重用 `/usr/bin/chromium`，實際 Zensical desktop 1280×1000／mobile 390×844。四頁成功載入，已看四張改／新圖的 inline 截圖、DIoU 表與標題、crop 過渡。圖片載入正常，頁面沒有水平溢出。必要 SVG 文字符合 viewBox，圈號字型空方框已改普通數字並重看。桌面與手機各實際點頁尾「下一頁」，CSP→fusion→augmentation→IoU 的頁名與圖片均成功切換；手機網址含 navigation tracking 的 fragment，檢查以 pathname 判斷，沒有把 fragment 誤當跳頁失敗。
- SVG XML/title/desc/viewBox 和 `git diff --check` 通過。

視覺證據在 `/tmp/clear-foundations-browser/`，重要截圖：desktop/mobile-11-fusion-inline.png、11-bilinear-coordinates-inline.png、11-crop-resize-target-inline.png、11-iou-loss-inline.png；mobile-diou-table.png、mobile-diou-heading.png、mobile-crop-transition.png、mobile-csp-history-transition.png。

## 仍需另一位審查者處理

本報告只是作者自查。四頁需要非作者技術／證據與前後銜接審查，受影響單位需複查並更新 review 指紋。原始 first-read 問題不覆寫，不能僅憑這份作者交接就標為獨立驗收通過。沒有重新執行完整 CSP/Fusion 訓練例、Colab、GPU 或 AP 評測；既有實測原封保留，這次只做表與插值疑點的相稱 CPU 核對。

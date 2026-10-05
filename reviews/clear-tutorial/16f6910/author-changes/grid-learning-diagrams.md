# 07-training 160步圖解：作者重排

完成：2026-10-05T10:43:57.655282+00:00。作者 applications。這是root明確追加的作者工作，因此本項不算我的独立第三輪通過。

只新增兩張靜態SVG與此報告。沒有改正文、舊generated SVG、保存JSON、程式、notebook、coverage或既有transitions/foundations；沒有重訓160步或使用GPU。

## 檔案與重排

- `docs/assets/diagrams/07-grid-loss-readable.svg`：viewBox `0 0 520 746`，字級22–24。原有total／未乘5的box／objectness／classification四條線保留全部160點；legend與total公式分行，原32張／batch8／seed7／learning rate .01及更新前量測意義保留。
- `docs/assets/diagrams/07-grid-predictions-readable.svg`：viewBox `0 0 520 1122`，字級22–24，四張220×220圖片按原index0、1／2、3排列。直接複製舊SVG的4個PNG data URI，保留底圖逐位，不重建、不重選。原GT與pred完整xyxy用220/64比例畫，框旁只顯示#k，數字／TP／FP／IoU移到每張下方；圖0紅GT的FN另列。

兩圖都有viewBox、title、desc、role／aria-labelledby與報告SHA標記，明示既存單次實測、固定合成材料與非照片效果。曲線不拿validation／test作training loss，四張圖不替代全16張validation的AP50。

## Root需要處理的正文連結與方向詞

`docs/lessons/07-training.md` 的正文未由我修改：

1. 原第151行曲線連結 `../assets/diagrams/grid-learning-curve.svg` 換成 `../assets/diagrams/07-grid-loss-readable.svg`。
2. 原第164行圖板連結 `../assets/diagrams/grid-learning-predictions.svg` 換成 `../assets/diagrams/07-grid-predictions-readable.svg`。
3. 第166行「標籤第一行是…；第二行是…」建議改為「橙框旁的 #k 對應各圖下方的資料：第一行列預測編號、類別與 score，第二行列 TP／FP 與配對時可用的同類真值最大 IoU；沒配到的真值標 FN。」框號仍各圖內依score排序。
4. 圖片0的.980/.47 FP與FN段可保留；新圖仍是0–3材料，圖0在左上、圖1右上、圖2左下、圖3右下。曲線仍單張圖，沒有需要改成左右子圖的指示。

## 忠實來源與完整SHA256

已確認保存報告實際名稱是 `artifacts/checks/grid-learning.json`，不在 `artifacts/checks/curriculum/`。創建時間2026-10-05T07:19:34.408039+00:00，steps_completed160、validation seed700。

|來源／成品|SHA256|
|---|---|
|artifacts/checks/grid-learning.json|`c66887b113691433f4f8787cb55fbbb6f4dacf26d635565dd259b5988417a298`|
|grid-learning-predictions.svg（原圖未改）|`8eb7ba62adfa0df235a3530982ed6c73cdec7ced2ae47fcceabb1eac8b2b0bd4`|
|grid-learning-curve.svg（原圖未改）|`818baea73d025d34b16eb67e16947d46b7615f49b6d4618f3c3f0567c120682d`|
|07-grid-loss-readable.svg（新静態圖）|`8975637e5d4912b68f9bbd9a02dd092c31e9d0f029cc5f6268f730d3fe14adec`|
|07-grid-predictions-readable.svg（新静態圖）|`ac0aef6ee5429918f840f78711b29e6abeebc48235456d0e6fb8b151f0c0ecc3`|
|本次生成文字來源TXT，`/tmp/07-grid-readable-preview/source-labels.txt`|`d037365439bf9f50677e623ca2fcadffab669f9c0c393947e2be76191e5fef36`|

|原picture index|PNG字節SHA256|預測與配對（顯示精度沿原圖）|
|---|---|---|
|0|`bb939cbd21c1aeff699ebfb2833de2e8df0fa781d73e21f8a2b54707e4841e15`|#0 class1 score1.000 TP IoU0.62；#1 class0 score0.980 FP IoU0.47；紅GT FN|
|1|`264215121d58729bb9ff5a1566e7479c1f45ecc5e94155d819a5aabf1178ad21`|#0 class1 score1.000 TP IoU0.73；#1 class0 score0.998 TP IoU0.75|
|2|`e418c134de2a25ad18155364048ff835358b730891f1d4381469098fef580bd2`|#0 class0 score1.000 TP IoU0.77|
|3|`d413861d61858022cc809df2d315e7c027653c1268d7658a498813ab9c8c87a1`|#0 class0 score1.000 TP IoU0.90|

曲線重排是同一份loss_history的坐標映射：x=64+(step−1)/159×426，y=158+280−280×loss/1.25；四條線各160點，沒有抽樣或平滑。完整點輸出到小數第3位只為SVG座標排版，數值來源未改。

框完整xyxy另外以data-box-xyxy存入每個rect；新圖中每邊坐標由原完整浮點數映到圖內位置。score顯示3位、IoU顯示2位，與舊SVG文字一致。純CPU Python幾何核對同圖／同類／未配GT的最大IoU，得到同6個TP／FP標籤與1個FN；不是重新推論模型。

排版參考08-custom-loss-readable.svg／08-custom-predictions-readable.svg，沒有複製它們的材料或結果。暫存生成／自查程式位於`/tmp/07-grid-readable-preview/make.py`、`render.py`、`check.py`；來源文字及JSON provenance也只在/tmp，不新增repo腳本。

## 實際作者視覺自查

用已安裝Python Playwright、`/usr/bin/chromium`，args `--no-sandbox --disable-gpu --disable-dev-shm-usage`，把SVG原文放入無其他版型的HTML main：desktop1100×900、mobile390×844，CSS內距18；舊圖desktop max-width720，手機354px；新圖desktop max-width520，手機354px。作者實際查看原2圖與新2圖各desktop/mobile screenshot（共8張最終圖），修後又查看新4張截圖。

舊图在手機缩到约5.9px的15號原字，score/IoU難讀；新圖source22/24在354px顯示約15.0/16.3px。實看新圖每張下方score／IoU／TPFPFN可以直接讀，框ID在圖內黑底可辨。曲線legend及公式不重疊，底部說明完整。

Chromium实际getBBox檢查所有text均在viewBox內，桌面／手機均無水平溢出；新兩圖所有text的source font最低22。另確認4個PNG href字串一字相同、GT與pred全座標相同且新幾何誤差<1e-6 SVG單位、4×160曲線點映射符合原數據（SVG座標小數捨入≤.0005）。

截圖與browser-check.json存於`/tmp/07-grid-readable-preview/`，檔名desktop/mobile-07-grid-loss-readable.png、desktop/mobile-07-grid-predictions-readable.png；原圖亦同名grid-learning-curve／grid-learning-predictions。

**這是作者standalone SVG檢查，不是Zensical頁面與正文整合的獨立browser驗收。** Root換連結後，由modern／foundations獨立複查目前頁面，不能把此作者自查算作自己的第三輪通過。

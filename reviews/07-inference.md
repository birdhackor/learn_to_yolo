# 第 07 課推論審閱

審閱範圍：只讀 `docs/lessons/07-inference.md`、`lesson_cases/07-inference.py`，以及頁面所指的 `docs/assets/diagrams/07-data.svg`。未讀其他教材、大綱或 `miniyolo` 實作；忽略 Colab 佔位。

結論：decode 的座標範例、score 公式、分類別 NMS、64×64 輸入回原圖的說明互相一致；人工 fixture 與未訓練模型也有明確標示。不過，完全陌生且只具基本 PyTorch/CNN 知識的讀者，仍會卡在 head 通道／軸的定義與「單物件」樣本的圖像內容。建議先修正下列兩點。

## 1. 「單物件」樣本實際仍含兩個可見物件

位置：`lesson_cases/07-inference.py:11–18`；`docs/lessons/07-inference.md:19`。

`image` 同時畫了紅、藍矩形；第二筆資料仍傳入同一張 `image`，只是把 annotation 截成紅框。因此它是「兩個物件、只標一個」的圖片，而非文中聲稱的單物件圖片。`fixture counts [2,1,0]` 是依 annotation 人工寫入 logits 的結果，不能修正圖片與標註不一致的事實。新手容易因此誤以為物件數是由標註數任意決定，或以為 decoder 漏了藍框。

具體建議：建立 `one_image = image.clone()`，清除藍矩形所在的 `one_image[2,36:52,40:56]`，再以 `(one_image, one)` 組成第二筆樣本。保留第一張紅／藍圖及第三張空圖，便能確實展示二／一／零物件的 batch。

## 2. 本頁未完整定義七個 head 通道與格子軸

位置：`docs/lessons/07-inference.md:5、9–13、19–26`；`lesson_cases/07-inference.py:29–33`。

頁面給了 `[B,4,4,7]`、前四維 sigmoid 與 score 公式，但沒有集中說明兩個 `4` 各是哪個軸，或最後七維各自代表什麼。讀者要由 `fixture[b,y,x,...]` 和數值範例反推；也沒有明說寬高除以整張輸入尺寸，而中心 offset 的單位是一個 cell。僅靠基本 CNN 知識，無法穩定把 head 接到公式。

具體建議：在座標範例前補一段完整契約：`raw[b,y,x,:]` 的最後一軸依序為 `[tx,ty,tw,th,obj_logit,red_logit,blue_logit]`；`B` 是圖片數，格子軸依序是 y、x，索引從 0 開始；類別 id `0=紅、1=藍`。`sigmoid(tx,ty)` 是格內中心 offset，`sigmoid(tw,th)` 是相對整張 64×64 輸入的寬高；class softmax 只沿最後兩個類別通道計算。再列出 `cx=(x+sigmoid(tx))/4×64`、`cy=(y+sigmoid(ty))/4×64`，便能直接對照程式。

## 可選的小補強

- `docs/lessons/07-inference.md:29` 的 NMS 文字正確，但目前 fixture 每張圖每個類別至多一框，實際不會壓掉任何候選。若要讓案例也驗證整條流程，可加兩個同類且重疊的人工候選，列出 NMS 前後數量；目前 `[2,1,0]` 主要驗證 decode、score filtering、空結果與 batch 契約。
- `docs/lessons/07-inference.md:35` 只連到另一教材頁，沒有直接顯示引用圖。若本頁要能獨立閱讀，可直接嵌入 `07-data.svg`，並補一句 x 向右、y 向下、xyxy 為輸入 pixel 邊界。原圖恰為 64×64 的理由已說清楚，不需要在此展開 letterbox 實作。
- `docs/lessons/07-inference.md:5` 的「本次不改訓練設計」較像編修過程備註，可改成「本節沿用既有 head 定義，聚焦推論」。沒有發現其他應刪的怪內容。

## 數值與證據界線

- `docs/lessons/07-inference.md:11–15` 的 `[8,12,24,28]`、`.8×.9=.72` 及約 `.0016 pixel` 邊界誤差皆正確。`.72` 是以「若」引入的教學例子；程式 fixture 使用 objectness logit `10` 與類別 logits `±10`，分數接近 1，並非 `.72`。兩者沒有矛盾。
- `docs/lessons/07-inference.md:5、31` 與 `lesson_cases/07-inference.py:19、25、42` 明確區分人工 fixture 和未訓練模型。這份案例沒有訓練、載入 checkpoint 或展示 trained inference；不應把通過 fixture 稱為模型學會偵測。現有文字沒有這種誤導。
- pixel／normalized 寬高、xyxy／xywh、score／precision、score threshold／NMS IoU／matching IoU 的區別已明確寫出，沒有實質錯誤。

## 執行結果

在專案根目錄執行 `PYTHONPATH=. .venv-model/bin/python lesson_cases/07-inference.py`，exit code 為 0，所有內建 assertions 通過：

```text
untrained model candidate counts (no quality claim) [0, 0, 0]
artificial known-logit fixture counts [2, 1, 0]
fixture red box [8.00160026550293, 12.0, 24.00160026550293, 28.0]
```

因此文件宣稱的 fixture 誤差與實際輸出一致，空結果和 batch=1 契約均通過檢查。未修改教材或案例。


## 作者修訂與驗證（2026-10-02）

第二張改one_image並實際清除藍色像素；補7通道、軸、offset／wh比例單位，直接嵌資料圖。另加相鄰cell同類重疊候選，真NMS從2框減為1。

已實跑 `PYTHONPATH=. .venv-model/bin/python lesson_cases/07-inference.py`，exit code 0，相關assertions通過。此段是作者修改與執行紀錄，並非獨立reviewer重審通過的宣告。

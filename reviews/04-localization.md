# 04-localization 陌生讀者審查

範圍：只讀 `docs/lessons/04-localization.md`、`lesson_cases/04-localization.py`，並查看本課 SVG 與程式生成的 PNG；未讀其他課。讀者假設只懂基本 Python、PyTorch 與 CNN。Colab 佔位不列問題。

結論：可以理解「同一個 backbone 接分類與四座標 head」、GAP 為何不直接保留位置、Flatten 的固定位置對應，以及 xyxy、cxcywh、正規化、IoU 和加權 loss。沒有發現數值錯誤、程式執行失敗或圖文矛盾。以下是最值得補的三項，沒有必須刪除的段落。

## 1. 練習沒有說明要改哪裡，無法直接用案例核對答案

- 位置：文章第 60 行；程式第 66–70 行。
- 問題：「把人工預測改成真值向右移 6px」看似程式修改任務，但案例的 `predicted_boxes` 是模型 forward 的結果，沒有圖中 `[6,8,18,20]` 的人工預測變數，也沒有計算 IoU 的程式。陌生讀者不知道要改 SVG、改標籤，還是覆寫模型輸出；只執行原案例不會印出練習答案。
- 具體修法：將第 60 行開頭改為「**手算練習，不需改模型或標籤：**真值 xyxy 是 `[4,6,16,18]`，另設人工預測 xyxy 為 `[10,6,22,18]`，也就是整框向右移 6px。先換成正規化 cxcywh，再算 MSE 與 IoU。」將後面的答案另起一段。若作者希望是程式練習，則明確要求在獨立片段建立 `manual_pred_xyxy = pixel_boxes[0] + torch.tensor([6., 0., 6., 0.])`，並提供計算 IoU 的入口與預期輸出，不能要求修改目前不存在的人工預測。

## 2. MSE 手算例子省略了座標格式切換

- 位置：文章第 28、34–36、42 行。
- 問題：框 head 剛被定義成正規化 cxcywh，程式區塊的 `boxes` 也是此格式；緊接著第 42 行卻只寫「人工預測 `[6,8,18,20]`」。這其實是 xyxy pixels。數學答案正確，但讀者需要自行猜出這裡換了格式，容易將這四個數直接和正規化目標做 MSE。
- 具體修法：第 42 行明寫完整一條轉換：「人工預測 **xyxy pixels** `[6,8,18,20]` → **cxcywh pixels** `[12,14,12,12]` → **正規化 cxcywh** `[0.375,0.4375,0.375,0.375]`。與真值 `[0.3125,0.375,0.375,0.375]` 相減，誤差是 `[0.0625,0.0625,0,0]`，所以 MSE 為 `0.001953125`。」保留既有公式即可。這也能直接呼應第 60 行的「pixels 與正規化混用」警告。

## 3. 首次出現的 logits、GT、pred 少了最小定義

- 位置：文章第 15、28、54 行；SVG 圖例；程式第 74、79 行生成的圖例。
- 問題：backbone、head、B、MSE、IoU 都已有說明，但 `logits` 沒有；圖中的 `GT` 和 `pred` 也只用英文縮寫。基本 CNN 讀者能靠上下文猜出大意，但不一定知道分類輸出不是機率，或為何分類 loss 沒先 softmax。框 head 的四個 logits 又在下一節經 sigmoid 變成座標，值得明確區別。
- 具體修法：第 15 行補「logits 是未轉成機率的原始分數；分類的兩個分數直接交給 cross entropy。」第 28 行補「這裡 sigmoid 後的四個值代表座標比例，不是四種類別的機率。」在第 54 行首次介紹疊圖時補「GT（ground truth）是真值框，pred（prediction）是預測框」，或把 SVG 與 PNG 圖例改成「真值 GT／預測 pred」。不需要另增長篇術語章。

## 實際驗證

執行 `PYTHONPATH=. .venv-model/bin/python lesson_cases/04-localization.py`，退出碼 0，梯度與參數更新 assertions 通過：

```text
two different positions -> same global mean=0.0625
step=0, classification=0.7173, box=0.0185, total=0.8099
step=1, classification=0.7146, box=0.0127, total=0.7779
step=2, classification=0.7123, box=0.0091, total=0.7576
first_target_cxcywh=[0.3125, 0.375, 0.375, 0.375]
class_shape=(2, 2), box_shape=(2, 4)
```

SVG 的兩框各 `12×12=144`，交集 `10×10=100`，聯集 `188`，IoU 約 `0.532`，與圖及文章一致。右移 6px 練習的交集 `72`、聯集 `216`、IoU `1/3` 與 MSE `0.0087890625` 均正確。參數比較 `4100` 對 `20` 也正確。PNG 可清楚區別綠色真值與黃色模型框；文章已明確區分固定人工框圖與三步模型輸出，沒有把疊圖當作學會定位的證據。


## 作者修訂

已列出人工xyxy pixels→cxcywh pixels→normalized cxcywh完整MSE轉換，定義logits／GT／pred及四座標比例；練習明示手算人工[10,6,22,18]，不用改模型預測。原CPU案例與疊圖已實跑核對。

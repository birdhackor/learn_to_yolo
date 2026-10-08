# 7.5 Grid MiniYOLO：把輸出接回圖片

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.6.1/notebooks/07-inference.ipynb){ .md-button }

訓練時，框被換成格內 xy 與全圖 wh；使用模型時，需要反過來，把每格的七個數變成圖上畫得出來的 pixel 框、類別與 score。這就是解碼（decode）。接著用第 6 章的分數過濾與同類 NMS，留下要交給使用者或評估器的框。

先替紅框填一份答案已知的輸出，逐步驗算；再讓真正模型走同一個 decoder。讀完能手算一格的位置與分數，也能處理同批圖片各有零、一、多個框的結果。

## 將 target 的換算倒過來

`raw` 為 `[B,4,4,7]`，前三軸依序是圖片 b、列 gy、欄 gx；`raw[b,gy,gx]` 的七項是 `tx,ty,tw,th,obj_logit,red_logit,blue_logit`。紅 id=0、藍 id=1，softmax 只沿最後兩個類別通道計算。

框四項 sigmoid 後是比例：xy 以一格為尺，wh 以整張圖為尺。格寬 64/4=16 pixel，所以：

\[
c_x=(g_x+\sigma(t_x))\times16,\qquad c_y=(g_y+\sigma(t_y))\times16
\]

\[
w=\sigma(t_w)\times64,\qquad h=\sigma(t_h)\times64
\]

再以中心減／加半個寬高，得到輸入圖 pixel 的 xyxy，x 向右、y 向下。

![資料頁固定紅藍矩形的真值圖](../assets/diagrams/07-data.svg)

這張是原始真值圖，沒有模型結果。下面只畫紅框，把負責格標清楚：

![紅框中心與負責格](../assets/diagrams/object-journey.svg){ width="400" }

淺藍虛線是負責格，不是藍色物件；圖例由 1 數的第 2 列、第 2 欄，在程式是 gy=1、gx=1。假設這格的四個 sigmoid 比例為 `[0,.25,.25,.25]`，就能將同一紅框還原：

1. 中心 `((1+0)×16,(1+.25)×16)=(16,20)`。
2. 寬高 `(.25×64,.25×64)=(16,16)`。
3. 左上 `(16−8,20−8)=(8,12)`，右下 `(16+8,20+8)=(24,28)`。

結果 `[8,12,24,28]` 與資料頁紅框相同。這是把已知比例還原的手算，尚不是模型的偵測成果。

## 一格的分數與類別從哪裡來

decoder 為每格選出 softmax 機率最高的類別，score 則是 `sigmoid(obj)×最大類別機率`。假設 obj sigmoid=.8，紅類 softmax=.9，score=.72、label=0。score 用來排序和過濾，不能稱為 precision，也沒有經過正確機率的校準。

這個 head 的紅、藍是互斥類別，因此用 softmax，兩個機率總和為 1。若各自用 sigmoid，紅藍 logits 都是 2 時各得 .88，總和 1.76，不能當成這裡的二選一類別機率。

過濾 score 之後，在同一張圖、同一預測類別內做 NMS。`score_threshold=.25` 決定哪些候選進來，`nms_iou=.5` 決定同類候選彼此重疊到什麼程度時刪較低分框；紅、藍兩類不互相刪。這裡比較預測和預測，不使用 GT。評估 matching 的 .5 則比較預測與真值，是另一個門檻，即使數字相同也不能混用。

## 如何給 decoder 一份答案已知的輸出

有限 logit 的 sigmoid 永遠不恰為 0，所以剛才紅框的格內 x=0 是理想值。要建立人工 fixture（固定測試輸入），用 `logit(clamp(target,1e-4,1-1e-4))` 反推四個 logits。

`logit(p)=ln(p/(1−p))` 是 sigmoid 的反函數，例如 logit(.8)=ln4≈1.386。clamp 把比例限制到指定上下界，避免 logit(0)=−∞：紅框的 x=0 改為 .0001，解碼中心右移 `.0001×16=.0016 pixel`，得到約 `[8.0016,12,24.0016,28]`。檢查用 .02 pixel 容差，並不要求端點完全相等。

fixture 對每個正格填這四個框 logits，obj=10，正確類別=10、另一類=−10。正格類別機率幾乎為 1，score 約 sigmoid(10)=.99995。其餘格的 obj=−20、兩類 logits=0，score 約 `sigmoid(−20)×.5≈10⁻⁹`，都過不了 .25。

這份輸出不是由圖片算出，而是由標註填好，因此能檢查解碼數字，不能證明模型學會。

## 三張圖片，結果如何保留各自的框數

本例同批放三張 `[3,64,64]` 圖：第 0 張有紅藍，第 1 張只有紅，第 2 張全黑。第 1 張複製第 0 張，再把藍色畫素也清掉，不是只刪藍標註；程式檢查紅通道總和仍為 256、藍通道為 0，確保圖片與答案一致。

模型永遠為每張圖產生 16 個候選，過濾與 NMS 後留下的數量 N 才各不相同。`decode_grid` 回傳長度 B 的 list，每張圖一個 dict：

| 鍵 | shape | 內容 |
| --- | --- | --- |
| boxes | `[N,4]` | 輸入圖 pixel xyxy |
| scores | `[N]` | 由高到低的 score |
| labels | `[N]` | 與框同順序的類別 id |

人工 fixture 的結果應為 `[2,1,0]`；空圖 boxes 仍為 `[0,4]`。`result[0]` 取第 0 張圖，`result[b]['boxes'][0]` 才取某圖第一個框；後者要先確定 N>0，否則會出現 IndexError。

只送一張圖時，回傳 list 長度仍為 1。decoder 不用無指定軸的 `squeeze()`；那會刪掉 `[1,4,4,7]` 的 batch 軸，讓單圖與多圖走不同格式。

## 真正模型先跑一次，再逐項核對

完整 lesson case 實際先用 `collate` 將三張圖與標註組成 `images`、`anns`，建立未訓練模型，讓它的輸出走一次 decoder：

``` { .python data-excerpt="lesson_cases/07-inference.py" }
images, anns = collate([(image,two), (one_image,one), (torch.zeros_like(image),empty)])
# 剛建立、未訓練的模型；.eval() 切到評估（eval）模式，並回傳模型本身
model = GridDetector(num_classes=2, grid_size=4, width=8).eval()
with torch.inference_mode():
    raw = model(images)  # [3,4,4,7]
    smoke = decode_grid(raw, image_size=64, score_threshold=.25, nms_iou=.5)
assert len(smoke) == 3
assert all(p['boxes'].shape == (len(p['scores']),4) for p in smoke)
print('untrained model candidate counts (no quality claim)', [len(p['boxes']) for p in smoke])
```



這是冒煙測試（smoke test）：只確認流程接得通與結果 shape。起始 obj 約 −2，兩類機率約 .5，每格 score 約 .06，低於 .25，所以印出 `[0,0,0]` 是預期行為。它不能判斷模型好壞，也不代表 decoder 壞了。

`eval()` 切換會影響 Dropout、BatchNorm 等層的模式；本模型沒有這些層，因此數字不變。`inference_mode()` 不記計算圖，與 `no_grad()` 相似但對新 tensor 有更多限制，適合這裡純推論；要接回需要梯度的流程時用 `no_grad()`。

??? note "完整程式裡建立人工 fixture 的幾行"

    `anns` 和 `raw` 都來自上面那段程式；`raw` 是未訓練模型的輸出，這裡只借用它的形狀。

    ``` { .python data-excerpt="lesson_cases/07-inference.py" }
    target = build_targets(anns,4,64,2)  # 4×4 格、64×64 圖、2 類
    fixture = torch.zeros_like(raw)      # 形狀和模型輸出一樣：[3,4,4,7]
    fixture[...,4] = -20                 # 先把每一格的 obj 都填 -20（背景）
    # 逐一取出正格；這裡的 y、x 就是正文的 gy、gx
    for b,y,x in target['positive'].nonzero().tolist():
        fixture[b,y,x,:4] = torch.logit(target['box'][b,y,x].clamp(1e-4,1-1e-4))
        fixture[b,y,x,4] = 10            # obj
        fixture[b,y,x,5:] = -10          # 兩個類別先都填 -10
        fixture[b,y,x,5+target['class_ids'][b,y,x].item()] = 10  # 正確類別改成 10
    ```



執行頁首 Colab，或在 repo 根目錄執行 `PYTHONPATH=. python lesson_cases/07-inference.py`，依序核對：

- 未訓練模型的 `[0,0,0]` 只查回傳格式。
- 人工 fixture 的 `[2,1,0]` 查框數與每框 .02 pixel 容差，`fixture red box` 印第 1 張紅框約 `[8.0016,12,24.0016,28]`。紅藍 score 相同時順序不固定，所以按類別比對，不靠排列猜身份。
- batch=1 用斷言核對相同格式，不另印出。
- 同類 NMS 另加一個紅框在鄰格 `(gx=0,gy=1)`，sigmoid(tx)=.999，框約 `[7.98,12,23.98,28]`，與原紅框 IoU≈.998；obj=9，分數稍低。NMS 門檻為 1 時兩框都留，因為只刪 IoU**大於**門檻的框；改 .5 才刪較低分框，留下 1 個。斷言通過後才印 `same-class overlapping fixture NMS counts 2 -> 1`，2、1 是 print 中固定數。

## 預測框超出畫布時

套件 `decode_grid` 在算出 pixel xyxy 後還會裁切到 0～64，並丟掉寬／高≤0 的框。例如左上格的四個比例都為 .5，中心 (8,8)、寬高 32，原框 `[-8,-8,24,24]` 裁成 `[0,0,24,24]`。

寬高數學上應大於 0，但極小值在 float32 下可能捨入成零面積，所以也要丟掉。這些處理是針對模型預測；GT 越界或零面積則是資料錯誤，必須修標註，不能用裁切掩蓋。

??? note "sigmoid 仍非零，框為什麼會成為零面積"

    中心 x=24 時，float32 附近可分辨的差約 1.9×10⁻⁶。tw=−18 時 sigmoid 約 1.5×10⁻⁸，寬約 9.7×10⁻⁷ pixel；24−w/2 與 24+w/2 都會捨入成 24，因此 x2−x1=0。再更負（float32 約 −89 以下）時，sigmoid 本身也會算成 0。這是有限精度計算，與數學上的開區間要分開看。

本節輸入與原圖皆為 64×64，可直接疊框。非正方形照片經 letterbox 後，decoder 的框仍在補邊後的輸入座標，要先減 padding、除實際比例，才能疊回原圖。〈[自己的圖片推論](08-own-images.md)〉會沿用同一 decoder 做這一步。

畫圖也要分清 xyxy 與 `(x,y,w,h)`：後者是左上角加寬高，要由 `w=x2−x1,h=y2−y1` 轉換，不是中心 cxcywh。也不能把 normalized wh 比例直接當 pixel。

同一個 `decode_grid` 供評估與畫圖使用，算 loss 不經它。訓練頁的橙框圖板正是模型經這個函式的結果，不過用低候選門檻 .05，因此可能包含本節 .25 不畫的框。每格一框的容量仍在；NMS 只刪框，補不回漏掉的物件。提高 score 會減少畫面框數，也可能降低 recall，必須交給獨立評估判斷。

## 自主練習

自主練習（先自己想，再展開答案）：

人工 fixture 的 score 都接近 1（約 0.99995），門檻改成 0.75 也不會被刪，所以這題不改人工 fixture，另用一份練習用 fixture。它只在 (gx=1,gy=1) 放一個紅框：sigmoid(obj)=0.8、紅類 softmax=0.9，score=0.72，就是前面手算的假設值；其他格都是背景。

題目：用下面的練習用 fixture（score=0.72），把 score 門檻從 0.25 改成 0.75（NMS 的 IoU 門檻仍是 0.5），這個框會怎樣？提高 NMS 的 IoU 門檻能救回它嗎？

想用程式核對時，先依序執行 notebook 原本的程式格，再在最後新增一個程式碼儲存格（code cell），貼上下面建立練習用 fixture 的程式。核對用的解碼和斷言放在參考答案裡；想好答案再展開，把那幾行貼在這段程式後面執行。

```python
score_raw = torch.zeros(1,4,4,7)
# 先把 16 格的 obj logit 都設成 -20：sigmoid≈2×10⁻⁹，等於背景
score_raw[...,4] = -20
# [0,1,1] 是 b=0、gy=1、gx=1。x 不能填 0：logit(0)=-∞ 會被 decode_grid 拒絕（ValueError）。
# 本題只看 score，框的位置不重要；x 用 0.5，
# 所以框是 [16,12,32,28]，不是正文的 [8,12,24,28]
score_raw[0,1,1,:4] = torch.logit(torch.tensor([.5,.25,.25,.25]))
score_raw[0,1,1,4] = torch.logit(torch.tensor(.8))  # sigmoid 之後是 0.8
# 類別用 log：softmax(ln 0.9, ln 0.1)=(0.9,0.1)，因為 e^(ln p)=p，且 0.9+0.1=1
score_raw[0,1,1,5:] = torch.log(torch.tensor([.9,.1]))
```

??? note "參考答案"

    框在 NMS 之前就被刪掉了。`decode_grid` 先用 score 門檻過濾：0.72 小於 0.75，這個框過不了門檻，根本進不到 NMS。NMS 只處理留下來的候選，所以提高 NMS 的 IoU 門檻也救不回它。

    核對程式如下，貼在建立練習用 fixture 的程式後面執行。這段程式用 0.70 和 0.75 兩個 score 門檻各解碼一次；兩次呼叫只改 `score_threshold`，`nms_iou` 都是 0.5。

    ```python
    low = decode_grid(score_raw, image_size=64, score_threshold=.70, nms_iou=.5)[0]
    high = decode_grid(score_raw, image_size=64, score_threshold=.75, nms_iou=.5)[0]
    assert len(low['boxes']) == 1 and len(high['boxes']) == 0
    assert abs(low['scores'][0].item()-.72) < 1e-6
    ```

    門檻 0.70 和原本的 0.25 一樣低於 0.72，所以留下 1 個框（score 0.72）；門檻 0.75 時剩 0 個。兩個斷言都通過，表示結果和這份答案一致。


## 本章與原版的分數定義

本章沿用 [YOLOv1](https://arxiv.org/abs/1506.02640) 的 grid 思路，使用自己的七維 head 與解碼規則：

??? note "和 YOLOv1 原版差在哪"

    第 7 章採用自己的 7 維 head（每格輸出 7 個數）、sigmoid wh（寬高也先經過 sigmoid）和 `sigmoid(obj)×softmax(class)` 分數，不使用原論文的 confidence 定義。

    YOLOv1 讓每個框的 confidence 去學 Pr(Object)（有物件的機率）×預測框與真值框的 IoU：有物件時，目標就是這個 IoU；沒有物件時是 0。推論時再乘上類別條件機率（在有物件的前提下，屬於各類別的機率）。第 7 章的 objectness 只學 0 或 1：物件中心落在這格就是 1，否則是 0。



接下來把解碼框與原始 GT 交給評估器，核對配對與獨立資料的結果。

<!-- curriculum-evidence:start -->

## 實際執行紀錄

本節的完整程式於 2026-10-08 在 AMD EPYC 9V74 80-Core Processor（2 個執行緒）上用 PyTorch 2.9.1+cpu 執行，程式裡的 assert 全部通過。下面是那次印出的原始輸出；輸出裡若有計時或訓練得到的數字，換一台電腦會略有不同。每個數字的意思，以本頁正文的說明為準。[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/07-inference.json)

??? example "展開本次實際輸出"

    ```text
    untrained model candidate counts (no quality claim) [0, 0, 0]
    artificial known-logit fixture counts [2, 1, 0]
    fixture red box [8.00160026550293, 12.0, 24.00160026550293, 28.0]
    same-class overlapping fixture NMS counts 2 -> 1
    ```

<!-- curriculum-evidence:end -->

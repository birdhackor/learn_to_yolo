# 7.4 Grid MiniYOLO：三步訓練與診斷

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.4.1/notebooks/07-training.ipynb){ .md-button }

前置：[資料](07-data.md)、[targets](07-targets.md)、[loss](07-loss.md)。這三節已經確認：畫素和框對得上（資料）、每個物件由哪一格負責（targets）、loss 與梯度能手算（loss）。本節才把真正的 CNN 和 optimizer 接到這條偵測管線上，先確認 forward、backward、`optimizer.step()` 三段都真的執行、參數真的會改變；之後才規劃少量 overfit 和獨立資料評估。讀完本節，你能說出這個 CNN 怎麼輸出 4×4 格的預測、看懂三步訓練印出的數字，也知道「一個框都畫不出來」時該依序查什麼。

本節的出發點模型（起始分支）是本書的 4×4 grid MiniYOLO，後面許多章節都從它改起。歷史上的 [YOLOv1](https://arxiv.org/abs/1506.02640) 是單階段（single-stage）設計：一次 forward 就從整張圖直接預測所有框，不先挑出候選區域再逐一分類。本節的小 CNN、64×64 的圖、每格只預測一個框，以及〈[Grid MiniYOLO loss](07-loss.md)〉定義的 loss，都是教學簡化。本節只在 CPU 上做 3 次參數更新（optimizer step），用來確認程式每一段都真的執行。這不是原版 YOLO 的訓練，也還沒做第 7 章要求的泛化檢查：在沒參與訓練的圖片上偵測正確（見〈[獨立資料評估](07-heldout.md)〉）。

## 先看這個 CNN 如何輸出格子

`width=8` 是第一層卷積的 channel 數，後面兩層卷積依序加倍。一張 64×64 的圖依序經過下表各層，最後變成 4×4 格、每格 7 個數（B 是一個 batch 的圖片數）：

| 層 | 輸出 shape | 作用 |
| --- | --- | --- |
| 輸入 | `[B,3,64,64]` | RGB 圖片 |
| 3×3 卷積（stride 2）＋ReLU | `[B,8,32,32]` | 邊長減半 |
| 3×3 卷積（stride 2）＋ReLU | `[B,16,16,16]` | 邊長再減半，channel 加倍 |
| 3×3 卷積（stride 2）＋ReLU | `[B,32,8,8]` | 邊長再減半，channel 加倍 |
| adaptive pool | `[B,32,4,4]` | 把 8×8 每 2×2 一區取平均，變成 4×4 |
| 3×3 卷積（stride 1）＋ReLU | `[B,32,4,4]` | shape 不變，再混合相鄰格的特徵 |
| 1×1 卷積（head） | `[B,7,4,4]` | 每一格各自把 32 個特徵組合成 7 個數；各格共用同一組權重 |
| permute | `[B,4,4,7]` | 只把 7 移到最後一軸，數值不變 |

adaptive pool 是 `AdaptiveAvgPool2d`：它自動分區取平均，輸出指定的大小。〈[VGG 風格小 CNN](01-small-cnn.md)〉用它縮成 1×1，這裡縮成 4×4。輸出第 y 列、第 x 欄位置的 7 個數，就是圖上第 y 列、第 x 欄那一格的預測。target 也用同樣的 `[b,y,x]` 位置存放每格的答案，所以兩者能逐格對齊。GridDetector 的完整定義在 [miniyolo/models.py](https://github.com/birdhackor/learn_to_yolo/blob/main/miniyolo/models.py)。

head 有兩組 bias 是手動設定的。框寬高（tw、th）的 bias 設 −1.8，讓起始的框寬、高都約 9 畫素，接近資料裡的矩形。objectness 的 bias 設 −2：sigmoid(−2)≈0.1192，所以起始 objectness 約 0.12，而不是〈[Grid MiniYOLO loss](07-loss.md)〉零 logits 的 0.5；這比較符合「有物件的格子很少」（物件稀疏）的實況。這只是手動設定的起點，模型沒有載入預訓練權重。

??? note "為什麼是 −1.8 和 −2？"

    head 是 1×1 卷積，每個輸出通道有一個 bias，所有格子共用。剛初始化時，卷積那一項（權重乘特徵再加總）很小，所以每格輸出的 logit 大約就等於 bias。本頁最下方執行紀錄裡，step 0 的正負格平均都是 0.1202，和 sigmoid(−2)≈0.1192 很接近，正好看得到這件事。

    - tw、th 的 bias −1.8：sigmoid(−1.8)≈0.14。框寬高是相對整張圖的比例，乘上 64 約 9 畫素；資料裡矩形的邊長是 8～15 畫素，起點和它們接近。
    - obj 的 bias −2：本節的 batch 共 64 格，只有 4 格有物件（約 6%），這就是「稀疏」。假設每格都輸出同一個機率 p，平均 BCE 是 [4×(−ln p)+60×(−ln(1−p))]/64。p=0.5 時是 0.693；p≈0.12 時約 0.25；最小值出現在 p=4/64 時，約 0.234。從 −2 起步，起點的 objectness loss 已經很接近這個最小值。
    - 其他通道（tx、ty 與兩個類別）沒有手動設定，bias 是接近 0 的小隨機值。

## 一個固定 batch 的完整更新

本節的快速實驗只用一個固定的 batch：

- **資料**：生成 4 張圖，每張恰好一個矩形，影像 shape `[4,3,64,64]`。4 張圖共 4×16=64 格，其中 4 個正格（負責物件的格子）、60 個負格（背景）。每一步都重複用這個 batch，所以步與步之間只有模型參數在變。
- **輸出**：prediction 的 shape 是 `[4,4,4,7]`。前三軸依序是圖片、格子列 y、格子欄 x；最後 7 項依序是 `tx,ty,tw,th,obj,class0,class1`。它們都是 logits，也就是還沒經過 sigmoid 或 softmax 的實數（框的四項經 sigmoid 後是位置和尺寸的比例，不是機率）。
- **mask**：`target['positive']` 是 `[4,4,4]` 的布林 mask，True 標出正格。下方程式在 `torch.no_grad()` 底下用 `prediction[..., 4]` 取出 objectness 通道，sigmoid 後再用這個 mask 分出正格和負格，各自求平均，用來監看訓練。
- **重現設定**：資料生成用 seed=7，模型也固定 `torch.manual_seed(7)`，CPU threads=2。seed（亂數種子）決定亂數從哪裡開始：同一個 seed 每次都產生同樣的圖片和初始權重，結果才能重現；換一個 seed（例如本頁補充實驗用的 700、7000）就會產生另一批圖。threads=2 表示只用 2 個 CPU 執行緒計算；執行緒數不同時，加總的順序可能不同，loss 的末幾位小數就可能不同。

下面摘錄完整程式（Colab 最後一格）裡的訓練迴圈。單獨占一行的 `...` 表示那裡省略了幾行，旁邊的註解寫著省略了什麼；它和 `prediction[..., 4]` 裡表示「前面的軸全部照拿」的 `...` 不同。

``` { .python data-excerpt="lesson_cases/07-training.py" }
model = GridDetector(num_classes=2, grid_size=4, width=8)
optimizer = torch.optim.Adam(model.parameters(), lr=.01)
...                                     # 省略：複製第一個參數供最後比對、切到訓練模式
for step in range(3):
    optimizer.zero_grad(set_to_none=True)
    prediction = model(images)          # [4,4,4,7]
    ...                                 # 省略：shape 的斷言
    parts = grid_loss(prediction, target)
    ...                                 # 省略：total 是有限值的斷言
    parts['total'].backward()
    ...                                 # 省略：檢查梯度的斷言（見下方說明）
    optimizer.step()
    with torch.no_grad():               # 以下只是監看，不需要梯度
        obj = prediction[..., 4].sigmoid()                  # prediction 是這次更新「前」算出的
        pos_mean = obj[target['positive']].mean().item()    # 4 個正格的平均
        neg_mean = obj[~target['positive']].mean().item()   # 60 個負格的平均；~ 把 True／False 對調
    ...                                 # 省略：印出這一步的分項 loss 與兩個平均
```

迴圈每一圈（例如 step 0）依序做四件事：

1. forward 產生 logits；
2. loss 依 target mask 決定哪些格子要算哪一項 loss；
3. backward 把梯度填進每個參數的 `.grad`；
4. `optimizer.step()` 才真正修改參數。

每圈開頭的 `zero_grad` 清除上一圈的梯度，不會重設參數。若重新建立 model，必須重新建立使用它參數的 optimizer；舊 optimizer 仍指向舊 tensor。

本節的 optimizer 是 Adam。它和〈[一次學習的超短暖身](00-warmup.md)〉的 SGD 一樣，要到 `optimizer.step()` 才修改參數。不同的是，Adam 替每個參數記住過去梯度的平均與梯度平方的平均，自動調整各自的步長，所以改變量不等於「學習率×梯度」：例如第一步，每個有梯度的參數大約都移動 0.01（剛好是 lr 的值），方向與梯度相反。本節只需記得：`step()` 才會更新參數（細節見 [PyTorch 的 Adam 說明](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html)，英文）。

完整程式的斷言（assert）檢查這幾件事：

- 迴圈開始前，先複製第一個參數 tensor（第一層卷積的權重，shape `[8,3,3,3]`）；三步跑完後，斷言它和更新後的值不同。
- 每一步檢查 prediction 的 shape 是 `[4,4,4,7]`、total 是有限值、每個梯度 tensor 的元素都有限。再把所有梯度取絕對值相加，得到「梯度總量」（和〈[VGG 風格小 CNN](01-small-cnn.md)〉的 L2 長度不同），要求它有限而且大於 0。只檢查「大於 0」的話，無窮大也會通過。
- 三步跑完後，再斷言正格數是 4。

完整程式每一步最後印出 total、box、objectness、classification，以及上面算出的正格／負格 sigmoid(objectness) 平均。這兩個平均用的是這一步更新前算出的 prediction，所以每一行顯示的是該次更新前的狀態，不是更新後重新 forward 的結果。

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.4.1/notebooks/07-training.ipynb)，或在 repo 根目錄執行 `PYTHONPATH=. python lesson_cases/07-training.py`。應看到 step 0、1、2 三行分項 loss，都是有限值；最後一行是 `3 real CPU optimizer steps; parameters changed; no generalization claim`（意思是：真的在 CPU 做了 3 次更新、參數已改變、不宣稱泛化）。通過的條件是梯度有限、參數有更新、正格數與 shape 正確；三個點的 loss 不要求一路下降（單調）。PyTorch 版本或底層運算的實作不同時，loss 的最後一兩位小數可能不同。

以本頁最下方的執行紀錄為例，step 0 的 total 與 objectness 能用前幾節的公式手算核對，classification 能估出接近 ln 2 的值；三步之間 box 微升、total 仍下降，這是正常的。

??? note "核對 step 0 的數字"

    以下數字取自本頁最下方的執行紀錄；自己重跑時，末位可能略有不同。

    - **total**：5×0.009+0.2525+0.6842=0.9817，和印出的 total 相同。
    - **objectness**：step 0 時各格的 sigmoid(obj) 幾乎相同，正負格平均都是 0.1202。代入 BCE（負格用 1−0.1202=0.8798）：[4×(−ln 0.1202)+60×(−ln 0.8798)]/64≈0.2525。它遠小於〈[Grid MiniYOLO loss](07-loss.md)〉零 logits 的 0.693，這就是 obj bias 設成 −2 的效果。
    - **classification**：0.6842 接近 ln 2≈0.693，因為兩類的機率都還接近 0.5。
    - **三步的變化**：box 從 0.009 微升到 0.0094，total 仍從 0.9817 降到 0.8323。短短幾步內，各項 loss 不一定每步都下降，所以通過條件不要求單調。

自主練習：在 Colab 裡改「本節可修改的完整實驗」下面那一格程式；本機則改 `lesson_cases/07-training.py`，兩者是同一份程式。刪掉迴圈裡 `optimizer.step()` 那一行再執行：哪些檢查仍可能通過？哪一個會失敗？三行輸出會有什麼變化？

??? note "參考答案"

    shape、有限 loss、有限梯度、梯度總量大於 0 這幾個斷言都會通過，因為 backward 照樣算出梯度。step 0、1、2 三行印出的數字會完全相同：參數沒變、輸入也沒變，三次 forward 算的是同一件事。最後檢查「參數和更新前不同」的斷言會失敗，程式在那裡報 AssertionError 停下；排在它後面的正格數斷言不會執行，最後一行也不會印出。

    這正是要把 backward 與更新分開驗證的理由：算出梯度，不代表參數已經更新。

## 「全部背景」該如何查

**症狀**：訓練了許多步，objectness loss 一直下降，但把預測解碼後，每一格的 score 都低於顯示門檻（〈[人工框解碼與 NMS](06-decode-nms.md)〉用 0.25），所以一個框也畫不出來，看起來像模型把每一格都當成背景。

本節的三步模型也畫不出框，但原因不同：3 步更新後，正格的 sigmoid(obj) 仍約 0.12，乘上約 0.7 的類別機率，score 不到 0.1，低於 0.25 的顯示門檻。這是因為只更新了 3 步，不是程式錯誤。訓練很久之後仍然如此，才依序檢查：

1. **資料與正格**：畫出同一 batch 的疊圖，數 target 的正格數。本批 4 個物件都是紅色（class 0）；若程式把 label 0 當成背景，正格數會變成 0。
2. **梯度方向**：用〈[Grid MiniYOLO loss](07-loss.md)〉的人工 logits，確認正格 objectness 的梯度是負的（−0.03125）。optimizer 減去負梯度，正格的 logit 才會提高。
3. **參數有沒有更新**：確認本節的參數在更新前後確實不同。

三項都通過，才去調學習率、loss 各項的權重（例如 box 前面的 5），或處理正格太少的問題。

訓練初期，正格和負格的平均可能一起下降：60 個背景格先被學會，而所有格子共用同一組 head 權重和 bias，正格也會被一起往下拉（原因見下方摺疊說明）。這不一定是錯誤；但如果一直下降，而且正格平均始終沒高過負格平均，就要檢查分項 loss 與解碼後的框，只看 total 變小不夠。印出的正／負格平均只是模型對這 4 張訓練圖的分數，不是 precision；precision 要把預測解碼、再和真值配對後才算得出來。

??? note "只學會背景，正格的平均為什麼也會下降？"

    所有格子共用同一組 head 權重和 obj bias：1×1 卷積在每一格用的是同一組數字。平均 BCE 對 obj bias 的梯度，等於各格「sigmoid(obj)−target」的平均。以 step 0 為例，正負格的 sigmoid(obj) 都是 0.1202：

    - 60 個負格（target 0）各貢獻 0.1202；
    - 4 個正格（target 1）各貢獻 0.1202−1=−0.8798；
    - 平均：(60×0.1202−4×0.8798)/64≈+0.058。

    負格往下推的力量大於正格往上拉的力量，梯度是正的，所以共用的 obj bias 會先被調低，所有格子（包括正格）的 objectness 都跟著降。共用的卷積權重也在改變；要等特徵能分出「有矩形的格子」和背景，正格才會回升。

    本頁的三步輸出只看得到開頭：負格平均 0.1202→0.1181→0.1169 一路下降；正格 0.1202→0.119→0.12 大致持平，step 2 只多 0.001，還不能算是上面說的回升。三步裡看得到的分開跡象，是從 step 1 起正格平均就略高於負格（0.119 對 0.1181、0.12 對 0.1169）；正格真正回升，要訓練更多步才看得到。

## 少量 overfit 與泛化是後續兩個關卡

三步的收益是成本小、出錯時容易找到是哪一段；代價是完全不足以判斷學習效果。後續可以固定這 4 張，增加訓練步數，記錄分項 loss、正負格分數與框，直到模型對這幾張圖的定位與分類明顯正確，也就是先在少量資料上 overfit（把看過的圖學到幾乎全對）。若做不到，先修管線，不急著加更多資料。達成少量 overfit 後，再凍結設定：之後不再改步數、學習率、門檻等設定（凍結的是設定，不是模型權重），然後拿另一個 seed 生成、或其他來源的圖片來評估。

下方的補充是一次 160 步的合成資料實驗，提供三步檢查沒有的效果數字。它保存了 checkpoint，也就是把訓練後的模型參數、optimizer 狀態與設定存成檔案，之後可以直接載入來推論，或接著訓練。它也使用事先固定的評估規則（固定的 validation／test seed，以及 score、NMS、配對的門檻）與圖板（疊上真值框與預測框的圖）。上面的三步快速檢查只有 batch=4、64×64、3 steps、CPU 這些實際設定，沒有「訓練後 AP 提升」或「沒看過的圖也偵測成功」的數字。真正訓練要幾步、多久、多少記憶體要另外量，不能從三步推估。

## 補充：160 步合成資料短訓練

前面的三步案例只檢查程式。這裡的 160 步是另一次小實驗：用 32 張圖訓練，再用從未參與更新的 validation／test 圖量 mAP50，提供「這個受控任務學得動」的初步證據（受控任務：資料由程式生成，條件都能控制）。它不是上面說的「固定 4 張」那一關；怎麼用獨立資料做完整的泛化評估，見〈[獨立資料評估](07-heldout.md)〉。

這次使用同一套 miniyolo 程式（同樣的模型、target、loss），另外做了 **160 次 CPU 參數更新**：train 32 張（seed 7）、validation 16 張（seed 700）、test 16 張（seed 7000），每張 64×64、batch 8、Adam、learning rate 0.01、模型 width 8。模型從頭訓練：權重隨機初始化（只有 head 的 wh／obj bias 照前面設為 −1.8／−2），沒有載入預先訓練好的權重。

這些圖片都是紅／藍矩形。和三步案例不同，這裡每張圖有 0～2 個矩形，也有空圖。物件刻意放在不同格，每個矩形都完整落在自己那一格內，尚未包含同格衝突或真實照片的複雜背景。每步依固定順序取 8 張，4 步輪完 32 張，160 步等於每張圖看過 40 次。

```bash
python -m miniyolo.train --steps 160 --samples 32 --device cpu
```

此命令在 repository 根目錄執行，會把 checkpoint、每一步的 loss、loss 曲線圖、validation 圖與完整報告 `report.json` 都存到 `artifacts/runs/grid-learning/`。

用網頁的實測圖先核對，不需要先重跑：

![實測 160 次 CPU 更新的 loss 曲線](../assets/diagrams/grid-learning-curve.svg)

這張圖這樣讀：

- 藍線 total=5×box+objectness+classification。紅線 box 是**還沒乘 5** 的原始值，所以紅、綠、紫三條分項線直接相加，會比藍線小。
- 紫線 classification 約第 16 步就降到 0.01 以下；綠線 objectness 約第 50 步才接近 0。之後 total 剩下的主要是 5×box。
- 前段的鋸齒，一個原因是每步換一批不同的 8 張圖。
- 橫軸從 1 數起：第 1 點是第 1 次更新前的 loss，位置相當於三步案例的 step 0（但兩者用的圖不同，數值不能直接比）。
- 曲線只反映訓練圖；在沒參與訓練的新圖上表現如何，要看下表的 validation／test mAP50。

| 固定評估規則下的結果 | 數值 |
| --- | --- |
| 更新前 validation mAP50 | 0.0018 |
| 更新後 validation mAP50 | 0.8036 |
| 訓練結束後的 test mAP50（只評一次） | 0.7749 |
| test precision／recall | 0.8824／0.7895 |

表中的數字都照下面這套事先固定的評估規則計算，各步驟在〈[人工框解碼與 NMS](06-decode-nms.md)〉與〈[人工框評估與 AP50](06-evaluation.md)〉學過：

1. **解碼**：每格輸出一個框，score=sigmoid(obj)×最大類別機率。
2. **分數篩選**：只留 score≥0.05 的框。這個門檻刻意設低，讓分數較低的候選也能排進 PR 曲線；這和〈人工框解碼與 NMS〉決定畫出哪些框的顯示門檻 0.25 不同。
3. **同類 NMS**：同一張圖、同一類的框之中，每一輪保留剩下分數最高的框，刪掉和它 IoU 大於 0.5 的其他框，再對剩下的框重複；已被刪掉的框不會再拿來刪別的框。
4. **配對**：依 score 由高到低，每個預測框和同一張圖、同一類、還沒被配對的真值比 IoU；最大的 IoU 達到 0.5 就算 TP，那個真值不能再配給別的框。沒配上的預測算 FP，沒被配到的真值算 FN。
5. **AP 與 mAP50**：每一類用 all-points 插值算 AP，再對有真值的類別取平均，就是 mAP50。

這不是 COCO 常報的 AP@[.50:.95]；那是在 IoU 門檻 0.50、0.55、…、0.95 共十個門檻各算一次、再取平均的分數。每個門檻的分數也對有真值的類別取平均（照本書用語，其實是 mAP），但讀 precision 時只取 101 個 recall 位置，不是 all-points；COCO 另有其他計分規則，見〈[人工框評估與 AP50](06-evaluation.md)〉。

表中 test 的 precision／recall，是用評估規則第 2、3 條（分數篩選、同類 NMS）之後留下的全部預測算的：test 有 19 個真值、17 個預測，其中 15 個配對成功，所以 precision=15/17≈0.8824、recall=15/19≈0.7895。validation 則是 18 個真值、16 個預測、15 個配對成功。

test 只在訓練結束後評一次，沒有拿來修改模型或設定。不過只有一個 seed，測試集也只有 16 張圖，所以 validation 0.80 與 test 0.77 這種小差距，或和其他設定比出的小差距，都不代表穩定的好壞。

更新前的 validation mAP50 是 0.0018 而不是 0：未訓練模型的每個框，分數都剛好略高於 0.05 門檻而全部進入評估，其中有幾個碰巧和同類真值重疊夠多。

??? note "沒訓練的模型，mAP 為什麼不是 0？"

    未訓練時，每格都輸出約 9×9 畫素的框（wh bias −1.8 的效果），score 約 sigmoid(−2)×0.5≈0.06，略高於 0.05 門檻。所以 16 張×16 格=256 個框全部進入評估；各格中心相隔 16 畫素，這些框互不重疊，NMS 一個也沒刪。其中只有 3 個碰巧和同類真值的 IoU≥0.5，因此 precision=3/256≈0.0117、recall=3/18≈0.17，mAP 才是 0.0018 而不是 0。

![四張獨立 validation 圖的真值與實測預測框](../assets/diagrams/grid-learning-predictions.svg)

綠色虛線是真值，橙色實線是模型預測的框：經過分數篩選與同類 NMS（評估規則第 2、3 條）後留下的全部框，也就是實際拿去配對的那些框。矩形的填色就是類別：紅=class 0、藍=class 1。每個預測框旁有一個兩行的標籤。第一行「#k class c score s」依序是預測編號（同一張圖裡依 score 由高到低，從 0 編號）、預測類別和 score（sigmoid(obj)×最大類別機率，取三位小數）。第二行是照評估規則第 4 條的判定：TP 或 FP，後面接著這個框和同類、還沒被配對的真值之間最大的 IoU（同一張圖若已沒有這樣的真值，就改寫「沒有可配對的同類真值」）。沒被任何預測配對到的真值，旁邊標著 FN。

圖片 0 值得細看。左上藍色矩形的預測框（#0）明顯向上偏，但標籤是 TP、IoU 0.62，重疊仍達到 0.5 的配對門檻。右下紅色矩形的預測框（#1）score 高達 0.980，標籤卻是 FP、IoU 0.47，沒達到 0.5；所以那個紅色真值沒被配對到，旁邊標著 FN（漏檢）。validation 唯一的 FP 就是它。分數高，不代表位置準到能通過配對門檻。

這四張圖只是圖板，mAP 使用全部 16 張。多數框接近真值、mAP 從接近 0 升到約 0.8，支持「這條管線在受控任務學得動」；但這只是紅／藍矩形上的結果，還不能回答模型是否認得照片裡的行人。

原始設定與數值保留在 [實測 JSON](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/grid-learning.json)，其中的 `loss_history` 存著每一步的各項 loss，上面的曲線就是用它畫的。這次實測只計訓練迴圈本身，在 Intel Xeon Platinum 8573C、PyTorch 2.9.1+cpu、2 threads 上約 1.18 秒；程式啟動、資料建立、畫圖和評估都不算在內。不同機器要自己量，也不能用這個時間預估真實資料的訓練。

想在本節 Colab 重跑這個 160 步實驗：先執行 notebook 最上面的環境格（下載本書程式、安裝套件的那一格），再另開一個 code cell，貼上下面的程式。它等同 notebook 說明裡的 `!python -m miniyolo.train --steps 160 --samples 32 --device cpu`，只是改用 Python 寫，並用 `--output` 明寫輸出資料夾（寫的就是預設值）。

```python
import subprocess, sys
# subprocess.run：在 Python 裡執行一行命令列指令
# sys.executable：目前這個 Python；check=True：指令失敗就報錯停下
subprocess.run([sys.executable, '-m', 'miniyolo.train', '--steps', '160', '--samples', '32',
                '--device', 'cpu', '--output', 'artifacts/runs/grid-learning'], check=True)
# 顯示訓練程式存下的 loss 曲線圖
from IPython.display import display
from PIL import Image
display(Image.open('artifacts/runs/grid-learning/loss.png'))
```

換一台機器重跑，loss 等數字的小數可能和本頁略有不同，這是正常的。loss.png 和網頁上的曲線圖用同一組顏色：藍 total、紅 box、綠 objectness、紫 classification。圖上的字是英文（PNG 用執行環境的字型畫字，Colab 多半沒有中文字型）：橫軸的 optimizer step 是第幾次參數更新，括號裡說每一點是那次更新前的 loss；圖例的 box (raw, before ×5) 就是還沒乘 5 的 box。

同一個資料夾還有 `checkpoint.pt`、`history.json`（每一步的 loss）、前 4 張 validation 圖的 PNG，以及完整報告 `report.json`：設定、電腦與訓練時間、訓練前後的 validation 指標與訓練後的 test 指標、每一步的 loss，以及這 4 張圖的真值與預測框。它和本頁連結的實測 JSON 是同一種報告，欄位相同，可以逐項對照。

網頁上的兩張圖不會因為重跑而改變：它們由 `scripts/render_learning_evidence.py` 只讀 repo 保存的實測 JSON 畫出。曲線用 JSON 裡逐步保存的 loss；圖板用 JSON 記下的 validation seed 700 重新產生同樣的前 4 張圖，疊上 JSON 裡的預測框，再照前面的評估規則標出 TP／FP／FN。

## 本節證明了什麼、還沒證明什麼

**已經證明**

- 三步案例：forward、backward、`optimizer.step()` 都真的執行；loss 與梯度都是有限值，參數確實改變。
- 160 步補充：在紅／藍矩形的合成資料上，validation mAP50 從 0.0018 升到 0.8036，test 是 0.7749；這是「這個受控任務學得動」的初步證據。

**還沒證明**

- 真實照片上的效果：這些圖都是合成矩形，不能回答模型是否認得照片裡的行人。
- 和其他架構或較新設計的比較：本節只訓練了一種模型，沒有證明哪個較新的機制比較好。
- 穩定的好壞：只有一個 seed，測試集也只有 16 張，小差距可能只是波動。
- 速度：1.18 秒只描述那一台機器上、那一次的訓練迴圈，不能拿來預估別的機器或真實資料的訓練。

<!-- curriculum-evidence:start -->

## 實際執行紀錄

本節的完整程式於 2026-10-05 在 INTEL(R) XEON(R) PLATINUM 8573C（2 個執行緒）上用 PyTorch 2.9.1+cpu 執行，程式裡的 assert 全部通過。下面是那次印出的原始輸出；輸出裡若有計時或訓練得到的數字，換一台電腦會略有不同。每個數字的意思，以本頁正文的說明為準。[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/07-training.json)

??? example "展開本次實際輸出"

    ```text
    step 0 {'total': 0.9817, 'box': 0.009, 'objectness': 0.2525, 'classification': 0.6842} positive 0.1202 negative 0.1202
    step 1 {'total': 0.9248, 'box': 0.0091, 'objectness': 0.2508, 'classification': 0.6283} positive 0.119 negative 0.1181
    step 2 {'total': 0.8323, 'box': 0.0094, 'objectness': 0.2491, 'classification': 0.5362} positive 0.12 negative 0.1169
    3 real CPU optimizer steps; parameters changed; no generalization claim
    ```

<!-- curriculum-evidence:end -->

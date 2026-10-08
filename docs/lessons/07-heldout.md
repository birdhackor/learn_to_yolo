# 7.6 Grid MiniYOLO：獨立資料與評估證據

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.6.1/notebooks/07-heldout.ipynb){ .md-button }

解碼得到框，還沒回答「它找對多少物件」。要把預測與同圖同類的真值（GT）配對，才能算第 6 章的 precision、recall 與 AP50；要回答模型是否能處理新圖，還得讓這些圖不參與權重更新。

先用一個故意誤報、重複與漏檢的人工例子核對評估器，再讓三步模型走一次獨立圖片管線。最後讀同一次 160 步實驗的結果。讀完能分開判斷評估計算是否正確、管線是否接通，以及獨立資料支持何種效果結論。

## 有誤報又漏檢，AP 會怎麼變

兩張圖 A、B 各有一個紅類 GT，分別為 `[8,12,24,28]`、`[40,36,56,52]` pixel xyxy。B 只是借用先前藍框的位置，這份人工例子兩個 GT 的 class id 都為 0。圖 A 有三個人工預測，圖 B 沒有預測。

這是 fixture（答案已知的固定測試輸入），直接交給評估器，不經 NMS；重複框故意保留，才能核對每個 GT 只配一次的規則。預測按 score 排序，在同圖、同類、尚未配過的 GT 中找最大 IoU，達 .5 才記 TP；配不成的預測為 FP，沒被配到的 GT 為 FN。

手機上可左右滑動表格，查看完整欄位。

| score 排序 | 框 | 位置與判定 | 累計 precision | 累計 recall |
| --- | --- | --- | --- | --- |
| 0.9 | `[8,12,24,28]` | A 的正確框，IoU=1，TP | 1 | 0.5 |
| 0.8 | `[0,0,4,4]` | A 的背景框，和 GT 沒有交集（IoU=0），FP | 0.5 | 0.5 |
| 0.7 | `[8,12,24,28]` | A 同一 GT 的重複框，IoU=1，但 GT 已被 0.9 的框用掉，FP | 1/3 | 0.5 |



B 的 GT 是 FN，A 的預測不能拿去配 B。三個預測只有一個 TP，因此最後 precision=`1/(1+2)=1/3`，recall=`1/(1+1)=.5`。前者分母是預測框，後者是 GT；不是圖片數。

AP50 看的是排序後的 precision–recall（PR）曲線，不只看最後一點。沿用 all-points 插值，對每個 recall 取往後的最高 precision，形成包絡。本例 recall 0～.5 的高度為 1；因為沒找到 B，.5～1 沒有新增 TP，補上 precision=0 的終點後高度為 0。因此面積 `1×.5+0×.5=.5`。

低分 FP 在 TP 後面，雖把最後 precision 降到 1/3，卻不改前半包絡。若背景框 score 改成 .95，順序成為 FP、TP、FP，前半高度只有 .5，AP50=.25。AP=.5 因而不是最後 precision，也不是 precision×recall=1/6；排序與漏檢都會影響它。

評估設定雖為 `num_classes=2`，藍類沒有 GT，AP 回傳 None，不平均進 mAP。因此這份 mAP 仍為 .5，不能把沒有樣本的類別當 0 分或滿分。

## 獨立圖的管線，先確認能跑通

完整程式生成 seed=7 的四張 train，以及 seed=901 的四張 held-out，每張 64×64、一個物件。seed 決定生成位置、大小與顏色的亂數序列；兩組圖不同。held-out 指保留、不拿來更新參數的資料。

只用 train 以 Adam 更新三次，再到 held-out 推論，並沿用上一節回傳的 boxes／scores／labels 格式交給評估器：

``` { .python data-excerpt="lesson_cases/07-heldout.py" }
model.eval()  # 切換成評估模式
with torch.inference_mode():  # 推論時不記錄計算圖
    # predicted：每張圖一個 dict，含 boxes、scores、labels
    predicted = decode_grid(model(heldout_images), image_size=64, score_threshold=.01, nms_iou=.5)
# heldout_anns：每張圖的 GT boxes／labels，不是 build_targets 的格子 target
result = evaluate_ap(predicted,heldout_anns,num_classes=2,iou_threshold=.5)
```

`heldout_anns` 是逐圖 GT，不是 `build_targets` 的格子答案。評估要和原始物件配對，不能用格子 target 的四個比例算 IoU。此處圖本來就是 64×64，因此 GT 與預測都在同一輸入座標。

這條路徑有三個不同門檻：

手機上可左右滑動表格，查看完整欄位。

| 門檻 | 本例 | 比較與用途 |
| --- | --- | --- |
| 候選截斷 score | .01 | 每框自己的 score，先刪極低分候選 |
| NMS IoU | .5 | 同圖同類的預測互相比，刪重複框 |
| 配對 IoU | .5 | 預測與可用同類 GT 比，達到才記 TP |

評估保留低分候選，是讓它們也參與排序，避免先刪掉低分 TP 而失去高 recall。若先提高 score 截斷，AP 只能持平或下降。顯示給使用者的 .25 門檻則是畫多少框的選擇，不是這次評估規則。

執行頁首 Colab，或在 repo 根目錄執行 `PYTHONPATH=. python lesson_cases/07-heldout.py`，第一行 `artificial evaluation fixture` 應為 AP=.5、precision=1/3、recall=.5；第二行為 `3-step held-out PIPELINE SMOKE, not trained detector evidence`。

第二行是冒煙測試：確認模型→解碼→評估能跑通。格式錯誤，如類別越界、x2<x1，評估器會拒絕；完整程式另斷言 mAP 在 0～1。這次 AP、precision、recall 都為 0，沒有任何成功配對，符合三步模型尚未學好的情況。評估器算對不對由人工 fixture 核對，不能用零分結果反推它已正確；三步也不當最後效能成績。

## 什麼資料才足以回答「沒看過的圖也會」

先固定資料與評估規則，再保存模型與結果，才知道一個分數由什麼條件產生：

| 紀錄 | 需要保存的內容 |
| --- | --- |
| 資料 | 來源、train／validation／test 切分與 seed |
| 模型 | 架構、所用 checkpoint |
| 評估規則 | 輸入尺寸、候選截斷、NMS、配對門檻與 AP 算法 |
| 結果 | 每類 AP、時間，以及成功、誤報、漏檢的疊圖 |

資料與規則構成評估協議。模型參數可在訓練中改變，評估條件要事先固定，否則無法分辨分數變化來自模型還是規則。剛初始化模型也能算出合法 AP；有 checkpoint、有分數，都不單獨證明有效。

held-out 是統稱，validation、test 都屬於它。反覆用某批圖挑學習率、步數或門檻，它就是 validation；test 在設定定案後只評一次。本節 seed=901 的四張只核對管線，不挑設定、不當最後成績。

對生成矩形，獨立 seed 只代表同一規則生成的新樣本，不能外推照片。真實影片還要按來源分組：相鄰影格幾乎相同，隨機切圖片可能讓 train 與 test 共享近似畫面。應將同影片或同次拍攝整組放一邊。這樣排除記住相似圖的錯覺，代價是訓練資料少一部分，小 test 的分數也容易波動。

## 用同一套規則讀 160 步的實測

這是〈[三步訓練與診斷](07-training.md)〉後半同一次實驗，不必重跑：32 張 train（seed 7）從零更新 160 次，validation／test 分別為 seed 700／7000 的 16 張；每張有 0～2 個紅藍矩形，含空圖。訓練前後各評一次 validation，test 結束後只評一次。程式沒有自動用 validation 挑設定，也未看 test 後回改。

| 固定評估協議下的結果 | 數值 |
| --- | --- |
| 更新前 validation mAP50 | 0.0018 |
| 更新後 validation mAP50 | 0.8036 |
| 訓練結束後 test mAP50 | 0.7749 |
| test precision／recall | 0.8824／0.7895 |

這次候選 score≥.05，同類 NMS IoU=.5，配對 IoU=.5，平均有 GT 類別的 all-points AP50。前面的三步用 .01，是不同實驗；各自的訓練前後、validation／test 都維持同規則，但兩實驗 AP 不直接比較。

test 共 19 個 GT，留下 17 個預測，TP=15、FP=2、FN=4，所以 precision=15/17、recall=15/19；紅 AP=.857、藍 AP=.693，平均約 .775。只多漏一個物件，recall 就差 `1/19≈.053`，約五個百分點。資料小、只跑一個 seed，validation .80 與 test .77 不能當穩定優劣。

![四張獨立 validation 圖的真值與實測預測框](../assets/diagrams/07-grid-predictions-readable.svg)

綠虛線是 GT，橙實線是預測，紅藍填色才是物件類別。各圖上方有圖號與 GT／預測數；框旁 #k 按 score 從 0 編號，對應下方同編號的類別、score、TP／FP 與可用同類 GT 最大 IoU。若沒有可配同類 GT，會明寫；未配到的 GT 另列 FN。score 1.000 是取三位小數的結果。

圖片 0 的藍框 #0 上緣偏高，IoU≈.62，仍為 TP。紅框 #1 的 score=.980，IoU≈.47，未達 .5，因此是 FP，紅 GT 是 FN。這是 validation 唯一 FP：16 張共 18 個 GT、16 個預測、15 個 TP，precision=.9375=15/16。此處分母 16 是預測數，剛好等於圖片數。

圖板固定看前四張，mAP 用全部 16 張。分數與框共同支持這條管線在受控矩形任務學得動；沒有照片、複雜背景或同格衝突，也沒有現代機制比較。設定與原始數值在 [實測 JSON](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/grid-learning.json)，重跑路徑見訓練頁選讀；三步 notebook 不會產生這個 160 步 checkpoint。

## 計算時別改掉評估的對象

AP 要將全部圖片的同類預測一起排序，不能每圖算完再平均，那會漏掉跨圖的高分 FP 排序；本頁人工例子恰好兩種算法都 .5，不能用它證明可互換。同一 GT 最多一個 TP，沒有 GT 類別記 None，顯示高門檻與評估低截斷也不能混成同條件。

本書 AP50 是教學評估器的結果，不是公開 benchmark（固定官方資料與工具的基準測試）成績，也不是 COCO AP@[.50:.95]。需要官方比較時，必須用相應資料與規則：

??? note "和官方評估的差異"

    Pascal VOC 和 COCO 是兩個著名的公開物件偵測資料集，各有官方的評估規則與程式。AP 的定義可參照 [Pascal VOC evaluation](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/htmldoc/index.html#SECTION00044100000000000000)（VOC2012 開發套件文件的 3.4.1 節）。本節和官方做法的差別：

    - **AP 面積的算法**：本節用 all-points 插值，每個 recall 改變的位置都算進來。VOC2007 用 11 點近似：只在 recall=0、0.1、…、1 這 11 個位置讀包絡高度再平均。VOC2010 起改用 all-points。COCO 則在每個 IoU 門檻下，只在 recall=0、0.01、…、1 這 101 個位置讀包絡高度再平均，也不是 all-points。
    - **IoU 門檻**：本節只用 0.5 一個門檻。COCO 的主要指標寫成 AP@[.50:.95]，是 IoU 門檻 0.50、0.55、…、0.95 共 10 個門檻的 AP 平均，也對各類別平均（照本書用語，其實是 mAP）。
    - **配對順序**：本書先排除已配對的真值（GT），再找 IoU 最高的；官方 VOC 則先在全部 GT 中找 IoU 最高的，再判斷它是否已配對；COCO 官方程式（pycocotools）的配對順序則和本書相同。同一張圖有互相重疊的 GT 時，兩種做法可能得到不同結果。[人工 AP 那一節](06-evaluation.md)列出了差異，以及它們在官方程式裡的出處。

    以上只列主要差異。COCO 另有 crowd 區域、每張圖每個類別最多計 100 個預測等規則，見[第 6 章的摺疊區](06-evaluation.md)；所以把本節評估器在這 10 個門檻各跑一次再平均，也不等於 COCO 的 AP@[.50:.95]。

## 自主練習

若把人工例子裡 0.8 與 0.7 這兩個 FP 刪掉，AP50 會是多少？precision、recall 又會變成多少？先手算，再展開答案；答案裡也附了貼進完整程式驗證的方法。

??? note "參考答案"

    AP50 仍是 0.5；precision 升為 1，recall 仍是 0.5。

    刪掉的兩個 FP 分數比 TP 低，本來就排在 TP 後面，不影響 recall 0 到 0.5 的包絡高度 1。B 仍漏檢，AP 仍停在前面說的上限 0.5。第 6 章的練習刪掉的是排在最前面的 FP，包絡才被抬高、AP 才上升。

    畫面更乾淨，未必代表找到物件的能力提升：recall 沒有變。

    用程式驗證：在完整程式（Colab 最後一格，或 `lesson_cases/07-heldout.py`）的 `main()` 裡，找到 `print('artificial evaluation fixture', metrics)` 這一行，在它下一行貼上下面這幾行。它們要放在 `main()` 裡面，所以每行前面要空 4 格，和 `print` 對齊：

    ```python
        # 圖 A 的 boxes、scores、labels 都只留第 1 筆（score 0.9 的正確框），等於刪掉兩個 FP；圖 B 不變
        # clone()：複製一份新的 tensor，避免動到原本 predictions 裡的資料
        cleaned = [{k: v[:1].clone() for k, v in predictions[0].items()}, predictions[1]]
        clean_metrics = evaluate_ap(cleaned, targets, num_classes=2, iou_threshold=.5)
        assert abs(clean_metrics['map']-.5)<1e-6
        # 上一行的 map 經過多步浮點運算，可能有捨入誤差，所以用容差比較
        # precision=1/1、recall=1/2 都是整數相除；商 1 和 0.5 能被浮點數精確表示，除出來剛好就是這兩個值，所以可以直接用 == 比較
        assert clean_metrics['precision']==1 and clean_metrics['recall']==.5
    ```

    執行後的輸出和原本一樣；沒有出現 `AssertionError`，就表示答案對了。

    原本的 predictions 與斷言都要保留，不要改掉：以後若改壞了程式，原例子的答案一變，斷言就會報錯。這種用舊例子的固定答案防止程式被改壞的檢查，叫回歸測試（regression test），和框的「回歸」無關。

<!-- curriculum-evidence:start -->

## 實際執行紀錄

本節的完整程式於 2026-10-08 在 AMD EPYC 9V74 80-Core Processor（2 個執行緒）上用 PyTorch 2.9.1+cpu 執行，程式裡的 assert 全部通過。下面是那次印出的原始輸出；輸出裡若有計時或訓練得到的數字，換一台電腦會略有不同。每個數字的意思，以本頁正文的說明為準。[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/07-heldout.json)

??? example "展開本次實際輸出"

    ```text
    artificial evaluation fixture {'ap_per_class': {0: 0.5, 1: None}, 'map': 0.5, 'precision': 0.3333333333333333, 'recall': 0.5}
    3-step held-out PIPELINE SMOKE, not trained detector evidence {'ap_per_class': {0: 0.0, 1: 0.0}, 'map': 0.0, 'precision': 0.0, 'recall': 0.0}
    ```

<!-- curriculum-evidence:end -->

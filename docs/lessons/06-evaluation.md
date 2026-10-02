# 人工框評估與 AP50：把預測逐筆算成證據

一張疊圖看起來不錯，不表示漏檢少或背景誤報少。評估要把同類預測與同圖真值配對，按score排序，再計算TP、FP、FN與PR曲線。本節用人工框手算，建立可核對的計分規則。本頁先補上xyxy與IoU的意思；不需要已訓練模型。

[在 Colab 執行](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.2.0/notebooks/06-evaluation.ipynb)，或 `PYTHONPATH=. python lesson_cases/06-evaluation.py`。CPU純評估，所有框與score人工指定，不做backward。AP採**all-points interpolated**定義；這不是完整COCO evaluator重現。原始評估來源可讀 [Pascal VOC官方評估說明](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) 與 [COCO detection evaluation](https://cocodataset.org/#detection-eval)。

## Matching是評估規則，不是訓練assignment

真值（GT）是標註框；pred是模型或本節人工給定的預測。每張圖GT boxes為[N,4]、labels為[N]，預測boxes為[M,4]、scores與labels為[M]；N、M各圖可以不同，也可0。所有框先放到同一座標系，類別id從0開始。

xyxy的[x1,y1,x2,y2]表示左上角與右下角，x向右、y向下；本課使用半開區間，框寬x2−x1、框高y2−y1，不加1。IoU是兩框交集面積除聯集面積，介於0與1；相同框為1、無交集為0。例如主例[0,0,10,10]與[1,0,11,10]各有100pixels²面積、交集90，IoU為 \(90/(100+100-90)=9/11\approx0.818\)。第三筆變FP是因GT已被認領，並非位置重疊不足。

此處IoU門檻0.5。每個類別把**所有圖片的預測收集起來**，按score由高到低看。某預測只找同圖、同類、尚未使用的GT；若最高IoU至少0.5，記true positive（TP），並鎖定該GT。否則是false positive（FP）。最後沒有被配到的GT是false negative（FN、漏檢）。

一個GT只可被認領一次。本節主例中，後來的重複框找不到另一個達標GT，因此算FP，即使它與已認領GT的IoU很高。不同圖片框數值相同也不能互相匹配。NMS是推論時候選去重；評估matching則有GT參與。這個案例故意保留重複候選，檢查評估器是否會把它錯算成第二個TP。

本書評估器先排除已用GT，再找最高IoU。若有兩個高度重疊的GT，相同預測框可能各配到一個GT，兩筆都算TP。官方VOC則先在全部GT中找最高IoU，再檢查它是否已使用；同一情況可能把第二筆判FP。這是本書的簡化配對契約，不能稱為完整VOC評估器。COCO也有crowd／ignore、候選數上限等額外規則；正式benchmark應用官方工具，不能只替換AP積分公式。

三種門檻比較的對象不同：**score門檻**比較候選自己的分數，決定哪些候選交給評估；**NMS IoU門檻**比較候選框彼此，決定是否抑制重複候選；**matching IoU門檻**比較候選與GT，決定是否匹配成功。三者各自設定。本例不執行NMS；評估函式的 `match_iou_threshold=.5` 只控制預測與GT的匹配，後面的score≥0.85實驗則先截斷候選。

## 四個預測、三個真值

圖A有GT1 [0,0,10,10]及GT2 [20,0,30,10]；圖B有GT3 [0,20,10,30]，皆類別0。按score排序的預測如下：

| 排名 | 圖／預測框 | Score | 判定理由 | TP累積 | FP累積 |
| --- | --- | --- | --- | --- | --- |
| 1 | A，[40,40,50,50] | 0.95 | 背景誤報 | 0 | 1 |
| 2 | A，[0,0,10,10] | 0.90 | 命中GT1 | 1 | 1 |
| 3 | A，[1,0,11,10] | 0.80 | GT1已被用過，重複 | 1 | 2 |
| 4 | B，[0,20,10,30] | 0.70 | 命中GT3 | 2 | 2 |

GT2沒有預測，FN=1。最後precision表示「所有保留預測中幾個是TP」，recall表示「全部GT中找回幾個」：

\[
P=\frac{TP}{TP+FP}=\frac{2}{4}=0.5,\qquad
R=\frac{TP}{TP+FN}=\frac{2}{3}\approx0.6667.
\]

分母不同，所以模型少畫框可能precision提高、recall降低；若刪掉正確低分框，也可能兩者一起變差。Score0.95的錯框證明confidence本身不是precision。

## PR曲線：逐步降低score門檻

從只保留最高分開始，依序納入更多候選，得到(P,R)：(0,0)、(0.5,1/3)、(1/3,1/3)、(0.5,2/3)。PR圖的橫軸是recall、縱軸是precision；分數不是圖的橫軸，分數控制「目前納入到哪一筆」。

![四筆人工預測的PR點、右側最大值包絡與AP面積](../assets/diagrams/06-evaluation.svg)

All-points interpolated AP先對每個recall，取它右側能達到的最高precision，也叫precision envelope。第三筆的1/3可以由第四筆的0.5抬到0.5；這是計分曲線的定義，不是更改某筆預測真假。

最後包絡在recall0到2/3是0.5，其餘到1是0，面積 \((2/3)\times0.5=1/3\)。程式更一般地在recall增加處累加「recall增加量×包絡precision」：

```python
ap = sum((recall_next - recall_previous) * precision_envelope_next)  # 公式偽碼，名稱表示各段宽度與高度
```

完整案例先補起終點，再從右往左取最大值。這不是直接對原始鋸齒曲線做梯形積分，也不是舊VOC的11點平均；若比較不同工具，先確認AP定義。

## AP50、mAP與COCO多門檻

AP50表示上述matching使用IoU≥0.5。多類別先分別算AP，再平均成mAP。例如兩個有GT的類別AP為1/3及1，mAP就是2/3。沒有GT的類別本節AP記None、不放進平均；有GT但沒預測則AP=0。整個資料沒有GT時mAP也None，避免把「沒東西可測」當作滿分。

COCO常報AP在0.50、0.55、…、0.95共10個IoU門檻平均，並有101個recall取樣及其他規則，因此不能把本節單門檻all-points數字直接當COCO AP。較高IoU要求更準的框；同一模型AP50高、嚴格門檻低，可能是定位問題。

## 截斷、核對與代價

主例應印TP=2、FP=2、FN=1、AP50=0.333333；第二類沒GT所以AP=None。本節precision／recall在分母0時按0回報，實務工具可能另標未定義。若評估前只保留score≥0.85，圖B正確低分框消失，recall變1/3、AP50降1/6。這是**評估候選截斷**影響可達recall，不只是顯示少幾個框。

收益是同時看排序、誤報與漏檢；代價是排序、IoU配對、規則與資料切分都要固定。空圖片仍可貢獻FP，漏標也會把真實物件的預測錯評為FP，所以評估前先看標註。小資料的AP變動很大，不能靠一次小差異推論架構普遍更好。

## 自主練習與答案

手算移除排名1的已知背景誤報，GT不變。答案順序TP、FP、TP；最終P=2/3、R=2/3，包絡在第一段1/3寬度為1、第二段1/3為2/3，AP為 \(1/3+2/9=5/9\)。案例已有此assertion與輸出0.555556。這個改動用GT認識錯框，是人工檢查，不是可直接在部署時執行的「理想濾除器」。

<!-- curriculum-evidence:start -->

## 本輪實際執行紀錄

本節範例已於 2026-10-02 使用 PyTorch 2.9.1+cpu 在 CPU 執行，程式中的斷言全部通過。以下是該次輸出；人工輸入、短步更新與模型效果的意義仍依本頁說明區分。[完整紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/06-evaluation.json)

??? example "展開本次實際輸出"

    ```text
    rank=1, score=0.95, FP, precision=0.0000, recall=0.0000
    rank=2, score=0.90, TP, precision=0.5000, recall=0.3333
    rank=3, score=0.80, FP, precision=0.3333, recall=0.3333
    rank=4, score=0.70, TP, precision=0.5000, recall=0.6667
    TP=2, FP=2, FN=1, AP50=0.333333, ap_per_class=[0.3333333432674408, None]
    candidate threshold .85: AP50=0.166667, recall=0.3333
    remove known high-score FP: AP50=0.555556
    no predictions => AP=0 when GT exists; no-GT class AP=None; background FP counted
    Artificial scoring exercise; not measured detector performance.
    ```

<!-- curriculum-evidence:end -->

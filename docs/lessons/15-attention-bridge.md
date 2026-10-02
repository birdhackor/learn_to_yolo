# 15.1 Feature map 到 attention：四個位置怎麼互相讀取

[開啟 Colab](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.1.0/notebooks/15-attention-bridge.ipynb) · 原始碼：`lesson_cases/15-attention-bridge.py`

前置是feature map、矩陣乘法、softmax與backward。卷積在固定區域性鄰域使用共享濾波器；attention則根據當前特徵，計算某個位置要向其他位置讀取多少資訊。本節不先背Q、K、V的名稱，而是用2×2特徵圖，一列一列算出相似度、權重和輸出。

歷史機制是YOLOv12的attention設計；本節起點是CNN輸出的feature map。本章簡化為單head、兩個channel、identity初始化的可學QKV投影、一般full attention，不含多head、位置卷積或完整YOLOv12。實驗只驗證資訊加權與梯度，下一節再改互動範圍。

![從空間特徵到加權輸出的shape路徑](../assets/diagrams/15-attention-bridge.svg)

## 空間位置變成列，channel變成特徵

輸入`[B,C,H,W]=[1,2,2,2]`。channel0為`[[1,0],[1,0]]`，channel1為`[[0,1],[1,0]]`。把H、W攤平成N=4，再交換C和N，得到tokens`[1,4,2]`：依序是左上`[1,0]`、右上`[0,1]`、左下`[1,1]`、右下`[0,0]`。

這一步沒有學到新特徵，只重新排列。`flatten(2)`使用row-major空間順序，還原也要同樣順序。`permute`不是任意reshape的同義詞，錯誤轉換可能shape合法、位置卻全部錯置。

QKV線性投影將每個2維token轉為6維，再切成Q、K、V各2維，shape都為`[1,4,2]`。初值三份投影都設為identity，讓Q、K、V恰好等於tokens，方便手算；它們是可學參數，不是永遠相同。

## 真正算第一個位置的權重

第一個query是`[1,0]`，與四個key的內積為`[1,0,1,0]`。除以`√d=√2`，softmax得到約`[.3349,.1651,.3349,.1651]`，四個權重加起來為1。

```python
similarity = q @ k.transpose(-2, -1) / math.sqrt(2)
attention = similarity.softmax(-1)  # along the source/key axis
output_tokens = attention @ v
```

Q與K的乘積shape是`[1,4,4]`，第i列代表接收位置i，欄j代表來源位置j。softmax沿最後key軸；若沿query軸，便不再是「每個接收者自己選一份來源分佈」。除以√d讓維度增加時內積尺度不致迅速把softmax推向極端。

第一個output是四份V的加權和：第一channel`.3349×1+.1651×0+.3349×1+.1651×0=.6698`，第二channel`.3349×0+.1651×1+.3349×1+.1651×0=.5`。Q、K決定讀取比例，V供應實際讀取內容。第一位置讀到了遠處的第三token，不只是複製自己。

## 還原圖片佈局，並讓投影學一次

輸出tokens仍為`[1,4,2]`；轉回`[1,2,4]`再reshape為`[1,2,2,2]`。本例以原feature作reconstruction target，計算MSE、反傳並做一次SGD；assert QKV權重有非零梯度且確實改變。這不表示attention學到了好的偵測語意，只證明weighting、還原和loss都在同一條可微路徑。

執行`PYTHONPATH=. python lesson_cases/15-attention-bridge.py`，核對tokens順序、first attention row約`[.3349,.1651,.3349,.1651]`、first output約`[.6698,.5]`、affinityshape`(1,4,4)`及step成功。CPU可執行，沒有FlashAttention依賴。

## 互動更遠，成本也更快成長

full attention的affinity每head有N²個數。8×8特徵圖N=64，需要4,096；16×16特徵圖N=256，需要65,536，空間邊長變兩倍，affinity變16倍。N²是傳統顯式權重矩陣的儲存規模；FlashAttention能改變實際記憶體存取，卻不應直接當成減少所有pair互動。

收益是內容依賴的長距讀取；代價是pair計算、投影、記憶體與位置資訊設計。卷積有區域性位置先驗，這份純attention在沒有位置訊號時，主要由token內容決定互動。官方YOLOv12另含位置卷積、多head、區域化等結構，不能用這份identity小例子的行為概括整個網路。

常見錯誤是把channel當token、漏掉K的transpose、softmax軸錯、用argmax替代加權和，以及誤認attention權重就是可直接解釋的因果重要性。本例Q、K皆為`[1,4,2]`；`q @ k`會讓4×2乘4×2，直接維度不匹配，不會得到C×C。若將channel當token，則可能另算成channel之間的C×C關係。權重說明當次計算的讀取比例，不足以證明某位置對最終decision的唯一貢獻。

自主練習：將第四token改成`[2,0]`，保持identity投影。案例已在optimizer step之前增加獨立練習段：複製`tokens.detach().clone()`，修改副本的第四列，再用尚未更新的qkv重算；原例assert保留，不直接改最上方feature而撞到舊答案。第一query分數為`[1,0,1,2]/√2`，第一列權重約`[.2212,.1091,.2212,.4486]`，輸出約`[1.3395,.3302]`；程式逐值核對。第四位置權重最大，第一channel增加。答案必須重新做softmax，不能只把舊輸出的第四項乘2，因為K改變了所有權重的分母。

來源查覈：2026-10-02。[Attention Is All You Need](https://arxiv.org/abs/1706.03762)、[YOLOv12作者AAttn實作](https://github.com/sunsmarterjie/yolov12/blob/2abab7153a065fb2925e8088e9ca2b19016ab7d6/ultralytics/nn/modules/block.py)。

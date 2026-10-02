# 特徵融合：把深層資訊送回細網格

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.1.0/notebooks/11-fusion.ipynb){ .md-button }

前置：[多尺度head](10-multiscale.md)、[卷積shape](01-small-cnn.md)。增加8×8head讓細格子可輸出框，但淺層feature的語義可能不足。這次只加入深層到淺層的top-down融合，不同時改anchor、augmentation或loss。

歷史機制：[FPN](https://arxiv.org/abs/1612.03144) 以top-down與lateral connections組建特徵金字塔；[PANet](https://arxiv.org/abs/1803.01534) 增加bottom-up路徑；[YOLOv4](https://arxiv.org/abs/2004.10934) 與固定版本[YOLOv5 v6.0模型配置](https://github.com/ultralytics/yolov5/blob/v6.0/models/yolov5s.yaml)有各自的neck設計。這些不是同一篇論文提出的一個模組。本章簡化為一條deep→shallow的nearest upsample／concat／卷積，不是完整PANet，也不聲稱完整v4/v5重現。起始為第10章兩尺度介面，先驗證融合，是否保留由正式品質／成本對照決定。

neck在backbone（特徵提取網路）與head（輸出框與類別的預測器）之間做融合。FPN的top-down把深層特徵送往淺層，lateral connection接入backbone同尺度特徵，原始FPN使用相加；PAN再增加淺層往深層的bottom-up融合。本例只有一次deep→shallow concat，輸出融合feature供細head連接，沒有整套雙向路徑。

![同一輸入的兩種stride與本節融合路徑](../assets/diagrams/11-fusion.svg)

## 為什麼不能直接 concat

同一張64×64輸入，總stride8的shallow有8×8格，總stride16的deep有4×4格。總stride表示相對輸入的採樣間距：兩層本來就不同，不是要求stride彼此相同。淺層 `[B,8,8,8]` 保有較細空間位置；深層 `[B,16,4,4]` 通道更多、網格較粗。NCHW最後兩軸不同，不能直接沿channel接起來。先用1×1把深層16channel減至8，shape成 `[B,8,4,4]`；1×1只改channel，不改4×4格；再nearest上取樣到 `[B,8,8,8]`，把deep搬到shallow的網格大小；concat後 `[B,16,8,8]`，最後3×3混合到8channel。

```python
reduced = reduce_1x1(deep)
up = F.interpolate(reduced,size=shallow.shape[-2:],mode='nearest')
joined = torch.cat([shallow,up],dim=1)
fused = mix_3x3(joined)
```

`size=shallow.shape[-2:]` 比假定scale_factor=2更明確，非整齊輸入尺寸時也能對齊shape。但shape一致不代表pixel對齊一定正確；各stage相對原圖的格點位置、stride與padding需要被追蹤；不同裁切造成的位移不會因shape一致而消失。

## 用四個數看上取樣

單channel 2×2 feature為 `[[1,2],[3,4]]`，nearest放大成4×4：

```text
1 1 2 2
1 1 2 2
3 3 4 4
3 3 4 4
```

每個來源值被複製四次；若loss是所有輸出總和，每個來源梯度恰為4。這裡「複製」不是生成四份新細節：深層feature仍只有原本四個值。細位置的資訊來自shallow，3×3融合學習如何使用兩路。若改bilinear，數值與梯度會不同，必須記錄插值方法，而不能把兩者當完全一樣。

本例參數是reduce `8×16+8=136`，mix `8×16×9+8=1160`，總1296。與沒有neck的兩head基線比較，這些是新增成本。concat的中間feature有`B×16×8×8`元素，也比單路8channel多；以float32、B=1，僅這張concat是4096bytes，實際訓練還包括梯度與其他activation。

執行 https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.1.0/notebooks/11-fusion.ipynb 或在 repo 根目錄執行 `PYTHONPATH=. python lesson_cases/11-fusion.py`。應核對手算4×4矩陣、來源gradient全為4、融合輸出`[1,8,8,8]`、參數1296；兩來源均有非零梯度並完成一次真SGD更新。這個feature實驗沒有訓練detector，因而沒有小物件AP或速度結論。

## 收益與代價

收益是細head可以同時利用淺層空間細節與較深層表達；代價是額外卷積、activation及對齊管理。深層低解析度資訊不一定能救回已經丟失的小物件，錯誤融合也可能把背景訊號帶進細head。正式對照固定輸入、資料、預算、decode/NMS協議，單獨替換neck，分別觀察小物件和背景誤報。

常見錯誤：cat dim=3把兩張圖在寬度拼接；改動reduction後，未同步修改mix的輸入channel；把upsampling當作物件框座標轉換；對所有feature做global pooling後期待空間定位。自主練習：若shallow為`[2,12,10,10]`，deep減至12channel再上取樣，concat shape是多少？答案：`[2,24,10,10]`；mix的輸入channel必須24。若不reduction，8ch與16ch仍能合法concat成`[B,24,8,8]`，只是mix要改成24ch輸入，成本也改變；channel reduction本身不是concat的必要條件。若用add，兩支整個shape必須一致，對應channel逐值相加後仍是8ch，mix要改8ch輸入；concat則保留兩組成16ch。

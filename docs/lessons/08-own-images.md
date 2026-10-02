# 用自己的圖片：先保持座標與類別契約

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.3.0/notebooks/08-own-images.ipynb){ .md-button }

前置：[完整推論](07-inference.md)、[座標轉換](04-coordinates.md)。本節處理單張非正方形圖片：讀取、RGB／CHW 轉換、letterbox、推論、框還原。它與「新增自己的類別」是兩件事；把照片放進模型不會讓紅／藍矩形模型自動認識汽車。

本次起始分支仍是兩類 grid MiniYOLO，沒有架構改動，也沒有下載 pretrained 權重。為讓沒有圖片的讀者也能獨立執行，case先在持久輸出目錄生成一張PNG，再用Pillow讀回；另外完成三步真實CPU更新，儲存並重載checkpoint，再作圖片推論。三步只驗證權重與圖片管線，人工已知框另用來驗證幾何，皆不代表照片偵測效果。

![非正方形圖片的縮放與padding](../assets/diagrams/08-own-images.svg)

## 120×80 圖片怎麼進 64×64 網路

原圖寬 120、高 80，紅框 `[20,10,60,30]`，寬高 40×20。影像 `[80,120,3]` uint8 經 `convert('RGB')`、`permute(2,0,1)`、除以 255，得到 `[3,80,120]` float32。透明圖或灰階圖同樣先轉RGB，避免channel數悄悄變動。模型實際接收`[1,3,64,64]`的NCHW，單張N=1、RGB、float32、值域`[0,1]`。

等比例 letterbox 的理想 scale=`64/120=.533333`。寬縮至 64，高理想為 42.6667；實際整數影像取 43，所以實作儲存 `sx=64/120`、`sy=43/80=.5375`。上 padding=10，下 padding=11。對框使用實際 scale 後，輸入框約為 `[10.6667,15.375,32,26.125]`。

還原則 x 減 left padding 再除 sx，y 減 top 再除 sy，得到原框 `[20,10,60,30]`。若只記理想 scale，把 y 也除 .533333，會出現取整誤差。metadata中的`original_size`、`resized_size`是`(H,W)`，`padding`是`(left,top)`，`scale_xy`是`(sx,sy)`；它們的順序並不相同。必須讓每張圖帶自己的 metadata，不能拿 batch 第一張的去還原全部圖片。

```python
image = torch.from_numpy(rgb).permute(2,0,1).float() / 255
input_image, _, meta = letterbox(image, torch.empty(0,4), size=64)
with torch.inference_mode():
    pred = decode_grid(model(input_image.unsqueeze(0)),64,.25,.5)[0]
original_boxes = undo_letterbox(pred['boxes'], meta)
```

傳空框表示推論時沒有 GT，不表示原圖裡沒有物件。模型看到的是 padded 64×64；decoder 也輸出該坐標系的 pixel xyxy。最後才還原到原圖、裁切合法範圍並畫框。若要量測速度，讀檔、縮放、模型、後處理要一起列出，而不是隻報模型 forward。

## 可核對的快速實驗

執行 https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.3.0/notebooks/08-own-images.ipynb 或在 repo 根目錄執行 `PYTHONPATH=. python lesson_cases/08-own-images.py`。應看到原圖 shape `(3,80,120)`、上述 input box 與 metadata，roundtrip 回 `[20,10,60,30]`。人工高分logits經decoder和undo再核對同一個框。另一段先做三次參數更新，再儲存、用`weights_only=True`載入，確認重載前後logits逐值相等；最後真正讀取PNG，輸出原圖座標JSON及疊框PNG。class數與配置不一致的checkpoint也必須被拒絕。三步產生的候選只證明權重已套用，不稱為偵測成功。

## 實際載入模型，推論指定圖片

第7章訓練CLI儲存的是包裝字典：`model_state_dict`是權重，`config`儲存image_size／grid_size／width，`class_names`儲存有順序的類別名稱。不是把整個包裝字典直接送進`load_state_dict`。重建與載入的關鍵步驟為：

```python
checkpoint = torch.load(checkpoint_path,map_location='cpu',weights_only=True)
cfg, names = checkpoint['config'], checkpoint['class_names']
model = GridDetector(num_classes=len(names),grid_size=cfg['grid_size'],width=cfg['width'])
model.load_state_dict(checkpoint['model_state_dict'],strict=True)
model.eval()
```

`strict=True`要求每個權重key與shape吻合；loader還檢查配置為正整數、類別名稱非空且不重複，若有num_classes則須等於名稱數。class id仍按names的索引，不可交換名稱順序冒充新類別。本入口只接受本課GridDetector儲存格式；原始裸state_dict需另外提供同樣配置與有序名稱，不能把任意YOLO權重交給它。

在repo根目錄可先建立第7章的160步模型，再使用完整圖片CLI：

```bash
python -m miniyolo.train --steps 160 --samples 32 --device cpu
PYTHONPATH=. python scripts/detect_image.py --image my.png --checkpoint artifacts/runs/grid-learning/checkpoint.pt --output artifacts/predictions/my-image.png
```

也可以把`--image`指定為訓練CLI輸出的`artifacts/runs/grid-learning/validation-00.png`，先核對紅／藍合成圖。CLI實際依序讀Pillow RGB、轉CHW float32、按checkpoint的image_size做letterbox、增加batch軸、載入後模型forward、decode／分類別NMS、undo_letterbox，最後儲存疊框PNG及同名JSON。無框時仍儲存原圖及空陣列；完全落在padding的零面積還原框會被刪除。

預設score與NMS值使用checkpoint配置，可用`--score-threshold .25 --nms-iou .5`明確覆蓋，並記在JSON。這個訓練模型只學過紅／藍矩形；一般照片可檢查讀取和座標，但照片裡的框不構成汽車、行人等類別的辨識證據。要學自己的新類別，接下一節重新建立資料與訓練。


收益是非正方形圖片保持長寬比，原圖框能正確展示；代價是 padding 佔用輸入範圍，小物件被縮小，極端長寬比圖的有效畫素更少。直接拉成正方形省略 padding，卻改變物件比例；兩者需要固定訓練與推論策略再比較。

常見錯誤：只 resize 圖不改框；把傳入空 GT 當成空圖片；先在 padded 圖畫框再把框當原圖座標；錯用另一張 metadata。自主練習：原圖 `[0,0,120,80]` 的整張框在輸入中是什麼？答案：`[0,10,64,53]`，不包括上下 padding。下一節才增加類別、標註與重新訓練。

本節實跑後，輸入圖、三步checkpoint、疊框`prediction.png`與座標`prediction.json`保留在`artifacts/lesson-08-own-images/`，終端會印出路徑。它們是本機／Colab輸出，受.gitignore排除。Colab可另開cell查看：

```python
from IPython.display import display
from PIL import Image
import json
from pathlib import Path
display(Image.open('artifacts/lesson-08-own-images/prediction.png'))
print(json.loads(Path('artifacts/lesson-08-own-images/prediction.json').read_text()))
```

框來自三步模型，仍是管線檢查；幾何示意圖不冒充這份模型預測。

<!-- curriculum-evidence:start -->

## 本輪實際執行紀錄

本節範例已於 2026-10-02 使用 PyTorch 2.9.1+cpu 在 CPU 執行，程式中的斷言全部通過。以下是該次輸出；人工輸入、短步更新與模型效果的意義仍依本頁說明區分。[完整紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/08-own-images.json)

??? example "展開本次實際輸出"

    ```text
    original CHW (3, 80, 120) input box [[10.666666984558105, 15.375, 32.0, 26.125]]
    metadata {'original_size': (80, 120), 'scale': 0.5333333333333333, 'scale_xy': (0.5333333333333333, 0.5375), 'padding': (0, 10), 'resized_size': (43, 64)} roundtrip [[20.0, 10.0, 59.999996185302734, 29.999998092651367]]
    3-step checkpoint safely reloaded: exact logits; original-space box count 16
    annotated PNG and JSON saved: artifacts/lesson-08-own-images/prediction.png artifacts/lesson-08-own-images/prediction.json
    pipeline evidence only, not photo detection quality
    class/config mismatch rejected
    artificial known box restored [[20.000001907348633, 10.0, 59.999996185302734, 29.999998092651367]]
    ```

<!-- curriculum-evidence:end -->

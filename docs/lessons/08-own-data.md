# 用自己的資料：類別、標註與來源切分

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.2.0/notebooks/08-own-data.ipynb){ .md-button }

前置：[自己的圖片推論](08-own-images.md)、[資料與訓練](07-data.md)。本節要解決兩個獨立問題：讓資料能表達新類別，以及讓評估真的使用沒見過的來源。模型類別數改變後需要重新訓練；改顯示名稱不能把「紅矩形」權重變成「汽車」。

本次使用一種 JSON 契約，保留原有 pixel xyxy，新增 class 2 黃矩形。這是本書資料格式，不是宣稱符合所有 YOLO 標註格式。案例無下載，在暫存目錄真正儲存六張PNG與JSON，再用Dataset讀檔，僅對train做一步backward／step；沒有真實資料效能主張。

## 固定一種可以查覈的格式

```json
{
  "classes": ["red_rectangle", "blue_rectangle", "yellow_rectangle"],
  "images": [{
    "path": "video_A/0.png", "width": 120, "height": 80,
    "source_id": "video_A", "split": "train",
    "boxes": [[20, 10, 60, 30]], "labels": [2]
  }]
}
```

classes 陣列的索引就是 class id，從 0 起算。每個框對應一個 labels 元素；不把「未標註」表示成空 labels 配非空 boxes。空圖兩者都為空陣列，確認合法後明確建立`torch.empty(0,4)`。非空boxes必須逐列是四個數字，不能先reshape把兩個二座標列拼成一個框。width／height 指未前處理的原圖，框也使用原圖 pixel；若 letterbox，影像和框必須一起變換。

依序檢查：所有紀錄的width／height都是正整數，空圖也相同；boxes每列嚴格是`[x_min,y_min,x_max,y_max]`四個有限數字，原始列數等於labels數；邊界須`0≤x_min<x_max≤width`、`0≤y_min<y_max≤height`；class id需`type(label) is int`且在範圍內，JSON的true／false不合格。接著確認所有split的圖片能讀、實際尺寸等於紀錄，轉RGB後再畫疊圖找漏標與錯位。程式只能查結構與範圍，不能替你知道一張圖有沒有漏掉第三個物件。漏標會把真正物件所在位置當背景監督，並非單純少一點訓練資料。

## 按來源分組，先切分再調參

假設 video_A 的連拍 A0、A1 幾乎相同。若 A0=train、A1=test，模型可能利用相同背景與姿態，看起來未見圖片也很好。案例把 video_A 全放 train，video_B 全放 validation，video_C 全放 test；每組兩張。source_id 可是影片、攝影機工作日、病患或同一個拍攝物，依任務選能阻止相近畫面洩漏的單位。

固定切分後只用 validation 選參數，test 最後一次評估。資料太少時可按來源交叉驗證，但每個 fold 仍不能拆同組。案例 validator 記住 source_id 首次出現的 split，若同來源出現在另一個 split 就失敗；也拒絕重複 path。這不能抓出改名複製圖，真實資料還需要來源盤點或內容重複檢查。

## 從path真的走到batch

`JsonDetectionDataset`位於`miniyolo/custom_data.py`，入口是標註檔與資料根目錄。`path`是相對root的圖片路徑；未指定root時用JSON所在目錄。初始化先驗證全部紀錄的類別、框列、尺寸、來源與檔案，再挑選指定split，因此train的讀取不會掩蓋validation裡的非法資料。

讀每張圖時，用Pillow`convert('RGB')`，HWC uint8轉CHW float32並除255，再將影像和pixel xyxy框一起letterbox到64。120×80圖上的`[20,10,60,30]`變成約`[10.6667,15.375,32,26.125]`；類別2不變。Dataset回傳`[3,64,64]`影像與boxes／labels字典，空圖仍有`[0,4]`框。

```python
from miniyolo.custom_data import JsonDetectionDataset
from miniyolo.data import collate
from torch.utils.data import DataLoader

dataset = JsonDetectionDataset('my-data/annotations.json',root='my-data',split='train',image_size=64)
loader = DataLoader(dataset,batch_size=2,shuffle=False,collate_fn=collate)
images, annotations = next(iter(loader))  # 同一順序的影像與target
classes = dataset.classes  # 保留JSON中有順序的類別對照
```

自己的替換入口就是JSON與root，毋須改硬編碼畫素。`collate`疊影像成`[B,3,64,64]`，各張變動物件數保留list，不可以先重新排序其中一邊。

## 新類別如何改到 head

兩類 head 的最後軸是 `5+2=7`，三類變 `5+3=8`，本例輸出 `[2,4,4,8]`。box 的四維和 obj 那一維不變；正格 class id=2 要交給三類 CE。需重建 model 和 optimizer，舊 checkpoint 的最後層 shape 不相容，不可只把 labels 改成 2 後繼續套兩類 head。

```python
model = GridDetector(num_classes=len(classes),grid_size=4,width=8)
target = build_targets(annotations,grid_size=4,image_size=64,num_classes=len(classes))
optimizer = torch.optim.Adam(model.parameters(),lr=.01)
optimizer.zero_grad(set_to_none=True)
loss = grid_loss(model(images),target)['total']
loss.backward()
optimizer.step()
```

這裡grid_size=4表示每邊4格，image_size=64是前處理後輸入pixel邊長，num_classes來自固定有序classes；target使用變換後框，而非原圖框。

執行 https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.2.0/notebooks/08-own-data.ipynb 或在 repo 根目錄執行 `PYTHONPATH=. python lesson_cases/08-own-data.py`。應看到 `rejected source leakage`、每個 split 2 筆、head `(2,4,4,8)`，再完成一步有限 loss 更新。案例實際寫出六張黃矩形／空圖PNG，查驗各檔尺寸並讀入train兩張，再完成參數更新；另外拒絕錯列長、bool類別、空圖非正尺寸、非有限座標及圖片實際尺寸不符。source_id與內容是人工fixture，重複顏色場景不提供來源獨立性的效果證據；它不是六張真實照片的評估。

收益是新增類別與資料切分都有可重現的規則；代價是標註、檢查、來源整理佔時間，新增類別也可能需要更多樣本及重新訓練。先選幾張標註疊圖、少量 overfit，再看獨立圖的漏檢與誤報，沿第 7 章順序排查。

自主練習：A0 是 train，A1 改成 validation，能否透過？答案：不行，同 source_id 洩漏。新增「綠矩形」成 class 3，最後軸應是多少？答案：9；還要驗證對映、重建head、target及optimizer，並安排新增類別的獨立 GT，不能只改陣列名稱。

圖片可放`my-data/images/`，JSON放`my-data/annotations.json`；JSON內`path`寫`images/example.png`。`root="my-data"`解析圖片路徑，JSON本身的路徑仍須完整寫明，不會自動加root。

<!-- curriculum-evidence:start -->

## 本輪實際執行紀錄

本節範例已於 2026-10-02 使用 PyTorch 2.9.1+cpu 在 CPU 執行，程式中的斷言全部通過。以下是該次輸出；人工輸入、短步更新與模型效果的意義仍依本頁說明區分。[完整紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/08-own-data.json)

??? example "展開本次實際輸出"

    ```text
    rejected source leakage : source leakage
    rejected malformed box rows : box/label row counts differ
    rejected box row length with matching label count : each box row must have exactly four coordinates
    rejected boolean class id : class ids must be integer indices; bool is invalid
    rejected empty-image dimensions : width/height must be positive integers, including empty images
    rejected nonfinite coordinate : coordinates must be finite numbers
    rejected actual image/annotation size mismatch
    6 real PNG files loaded/verified; counts by split {'train': 2, 'validation': 2, 'test': 2}
    synchronized input box [[10.666666984558105, 15.375, 32.0, 26.125]]
    new class 2; head (2, 4, 4, 8) one training step 1.4407
    ```

<!-- curriculum-evidence:end -->

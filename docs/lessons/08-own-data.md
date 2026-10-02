# 用自己的資料：類別、標註與來源切分

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.3.0/notebooks/08-own-data.ipynb){ .md-button }

前置：[自己的圖片推論](08-own-images.md)、[資料與訓練](07-data.md)。本節要解決兩個獨立問題：讓資料能表達新類別，以及讓評估真的使用沒見過的來源。模型類別數改變後需要重新訓練；改顯示名稱不能把「紅矩形」權重變成「汽車」。

本次使用一種 JSON 契約，保留原有 pixel xyxy，新增 class 2 黃矩形。這是本書資料格式，不是宣稱符合所有 YOLO 標註格式。主例無下載，在暫存目錄真正儲存六張PNG與JSON，再用Dataset讀檔，僅對train做一步backward／step；後半的完整學習閉環另用48張PNG訓練1600步，兩者都沒有真實資料效能主張。

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

執行 https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.3.0/notebooks/08-own-data.ipynb 或在 repo 根目錄執行 `PYTHONPATH=. python lesson_cases/08-own-data.py`。應看到 `rejected source leakage`、每個 split 2 筆、head `(2,4,4,8)`，再完成一步有限 loss 更新。案例實際寫出六張黃矩形／空圖PNG，查驗各檔尺寸並讀入train兩張，再完成參數更新；另外拒絕錯列長、bool類別、空圖非正尺寸、非有限座標及圖片實際尺寸不符。source_id與內容是人工fixture，重複顏色場景不提供來源獨立性的效果證據；它不是六張真實照片的評估。

收益是新增類別與資料切分都有可重現的規則；代價是標註、檢查、來源整理佔時間，新增類別也可能需要更多樣本及重新訓練。先選幾張標註疊圖、少量 overfit，再看獨立圖的漏檢與誤報，沿第 7 章順序排查。

自主練習：A0 是 train，A1 改成 validation，能否透過？答案：不行，同 source_id 洩漏。新增「綠矩形」成 class 3，最後軸應是多少？答案：9；還要驗證對映、重建head、target及optimizer，並安排新增類別的獨立 GT，不能只改陣列名稱。

## 把JSON接成完整的學習閉環

上面的一步檢查確認介面連得上，還不能說新類別已學會。`scripts/run_custom_data_learning.py`沿用同一個Dataset、GridDetector、target、loss與checkpoint，補上少量overfit、獨立圖片推論和重讀。先用不用下載的三類資料驗證：

```bash
python scripts/run_custom_data_learning.py --fixture --steps 1600 --fixture-test-seed 7001
```

它真的寫出48張非正方形PNG與JSON：24張train，12張validation，12張test；每個split各3張空圖，三類都有GT。source_id不跨split，PNG及解碼RGB沒有跨split的完全重複。這些source是獨立seed的合成資料，不是三支真實影片，也沒有證明真實連拍之間不存在近似重複。

先把所有圖與框letterbox到64×64，再以width8、4×4grid的三類模型從零訓練；共有15,544參數，batch8、Adam .01、模型seed7、CPU 2threads。訓練與validation資料seed為7／700，最後的保留test為7001；設定和1600步預算在這次訓練前固定，每次固定實驗只在結束後評估test一次。獨立重現或修圖後的相同設定重跑，不再用test挑配置。

### 先保留160步沒學好的結果

最初160步的完整train loss已從1.55017降至.16921，但train AP50只有.00680、validation為0。檢查實際PNG、框座標、positive mask與梯度後，21個GT都對應21個正格，寬高target非零；背景格的box梯度為0，符合mask設計。問題是位置仍偏，部分預測高度只剩約.05畫素；objectness和class loss下降，掩蓋了框還沒學好。這說明為什麼不能只看total loss。

我們保留[160步失敗紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/custom-data-160-step.json)，再只增加到預先固定的1600步；train／validation每張PNG、模型、optimizer、learning rate與門檻都相同。先前的test已看過，因此最後改用事前固定的新test seed7001；沒有用test挑seed、參數或最佳checkpoint。

### 1600步後，記住資料與泛化是兩件事

|檢查|本次結果|
|---|---|
|同一套完整train loss，更新前／全部更新後|1.55017 → .000354692|
|Train AP50／precision／recall|均為1.0，這個小資料集已overfit|
|Validation AP50|.388889|
|新的Test AP50|.666667|
|1600步梯度L2|每步有限且非零；.020489–32.557129|
|參數變化L2|16.519738|
|checkpoint重讀|raw、decoded、Adam state及RNG完全一致|

所有split都用score≥.1、同類NMS IoU .5，以及matching IoU .5的本書AP定義。AP採all-points interpolation，三類都有GT；不是COCO的AP .50:.95。每個保留split只有9個物件，validation和test的差異不能當成穩定的泛化排名。

![實測loss曲線與四張獨立validation圖；綠虛線為GT、橙線為模型預測](../assets/diagrams/08-custom-learning.svg)

圖中四張圖依序包含紅、藍、黃矩形及空背景，來自未參與更新的validation；框與分數是真實預測，綠GT用來看錯位。圖及AP使用64×64 letterbox座標。獨立PNG推論入口則以checkpoint的有序class_names重建三類head，再把預測還原至80×120或120×80原圖；兩套座標不能直接比數值。

四張圖固定取各類的第一張與第一張空圖，沒有依偵測效果挑選；可[開啟原尺寸圖](../assets/diagrams/08-custom-learning.svg)查看小字。其餘validation也有漏檢與框錯位，不能只看這四张就忽略整體AP。

本次CPU訓練約7.517秒，包含batch索引、target建立、前向／反向、有限梯度校驗、optimizer與純量記錄；完整流程約8.273秒，另含資料生成／查驗、評估、存檔、重讀和SVG輸出，不含套件安裝、import／程序啟動與最後JSON寫入。這是一次小實驗的耗時，不是正式效能。

### 換成自己的JSON與圖片

```bash
python scripts/run_custom_data_learning.py --annotations my-data/annotations.json --root my-data --steps 1600 --output artifacts/runs/my-data
PYTHONPATH=. python scripts/detect_image.py --image my-data/images/example.png --checkpoint artifacts/runs/my-data/checkpoint.pt --output artifacts/runs/my-data/prediction.png
```

第一行會讀JSON的有序classes、查驗三種split、訓練並保存`checkpoint.pt`、`report.json`、`learning.svg`和預測；第二行用存下來的模型偵測你指定的原图。使用小批已標註資料，每個split至少一張圖；此入口為方便檢查，把全部圖片載入CPU記憶體，不適合直接塞大型資料集。它會拒絕跨split的完全重複PNG／RGB；建立train target時拒絕同格兩物件，因為每格只負責一框。validation／test保留所有GT評估，不為配合模型而刪除同格物件；模型的容量限制仍可能造成漏檢。

只使用`--fixture`時，所有檔案預設保存在`artifacts/runs/custom-data-learning/`，結束後仍可查看；notebook可選格會顯示這次生成的`learning.svg`。checkpoint使用format v2，保存model、optimizer、step、設定、類別與RNG；本例固定learning rate，scheduler欄為None。重讀一致驗證的是此CPU模型與狀態，精確續訓的另外一條對照見[L4驗證](../validation/gpu-smoke.md)。[完整1600步結果](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/custom-data-learning.json)保留全部loss、SHA-256、圖與推論檢查；權重和資料不放普通Git。

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

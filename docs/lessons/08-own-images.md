# B.8.1 用自己的圖片：先把座標換算和類別順序弄對

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.6.1/notebooks/08-own-images.ipynb){ .md-button }

第 7 章的圖片原本就是 64×64，解碼框可以直接疊回去。自己的圖片可能是 120×80：先縮放補邊才進模型，預測還要換回原圖才能畫對。模型權重也要從存檔載入，而非每次重新用隨機權重推論。

這一頁沿用紅藍兩類的 GridDetector，處理讀圖、letterbox、載入 checkpoint 與框還原。把照片送進去，不會讓這個模型自動認識汽車；新增類別需要下一頁的標註與重新訓練。讀完能沿著同一張非正方形圖片追蹤 shape、座標和類別順序。

## 同一個紅框，先縮小再補邊

![非正方形圖片的縮放與 padding](../assets/diagrams/08-own-images.svg)

本例原圖寬 120、高 80，紅框 `[20,10,60,30]`，寬高 40×20 pixel。圖右整個正方形是 64×64 輸入，中間深色內容寬 64、高 43；灰色標示 padding，程式實際補黑色 0。這是座標示意圖，還沒有模型預測。

沿用第 4 章的 letterbox：保持比例，把原圖放進輸入，再補剩餘邊。理想比例為 `min(64/120,64/80)=64/120≈.533333`，寬縮到 64，高理想為 42.6667，但實際影像要取整成 43。

所以實作記兩個實際比例 `sx=64/120`、`sy=43/80=.5375`。高還差 21 列，上補 `21//2=10`、下補 11；左右不補。框先乘比例，再加左／上 padding：

| 原圖邊界 | 換算 | 輸入邊界 |
| --- | --- | --- |
| x1=20、x2=60 | ×64/120+0 | 10.6667、32 |
| y1=10、y2=30 | ×43/80+10 | 15.375、26.125 |

輸入框約 `[10.6667,15.375,32,26.125]`。還原則先減 padding，再分別除 sx、sy，回到 `[20,10,60,30]`。只用理想 .533333 還原 y，會算成約 10.08、30.23；內容下緣 y=53 甚至還原為約 80.6。差別來自高被取整成 43，不能省略實際比例。

letterbox 回傳的 metadata 就是這張圖的變換紀錄：

手機上可左右滑動表格，查看完整欄位。

| 欄位 | 本例 | 順序／用途 |
| --- | --- | --- |
| original_size | `(80,120)` | H、W |
| resized_size | `(43,64)` | H、W |
| padding | `(0,10)` | left、top |
| scale_xy | `(64/120,43/80)` | sx、sy；還原使用它 |
| scale | `64/120` | 理想比例 |

這是 `miniyolo.geometry` 的欄位名，第 4 章手寫版本的 `original_hw` 在這裡叫 `original_size`。意義相同，每張圖仍必須帶自己的 metadata，不能拿批次第一張的紀錄還原其他圖。

## 從 PNG 到模型的四個軸

Pillow 的 `Image.open(...).convert('RGB')` 先統一三個顏色通道；轉成 NumPy 陣列後是 `[80,120,3]` uint8，0～255。用 `permute(2,0,1)` 換成 CHW、轉 float32、除 255，才符合第 7 章資料契約。

形狀一路是：`[80,120,3]` → `[3,80,120]` → letterbox `[3,64,64]` → 加 batch 軸 `[1,3,64,64]`。NCHW 的 N=1 指圖片數，與框數無關。

以下是 `scripts/detect_image.py` 推論核心的簡化示意，使用與完整程式相同的 imports：

```python
# 'my.png' 換成你的圖片；.copy() 複製成可寫入的陣列，避免 torch.from_numpy 發出警告
rgb = np.asarray(Image.open('my.png').convert('RGB')).copy()  # 本例 [80,120,3] uint8（HWC）
image = torch.from_numpy(rgb).permute(2,0,1).float() / 255  # [3,80,120] float32，值域 [0,1]
# input_image 是 [3,64,64]；_ 是換算後的框，這裡傳入空框，所以它也是空的
input_image, _, meta = letterbox(image, torch.empty(0,4), size=64)
with torch.inference_mode():
    # model 是載入好的模型，載入方法見下方「實際載入模型，推論指定圖片」
    # unsqueeze(0) 在最前面加 batch 軸：[3,64,64] → [1,3,64,64]
    # 64、.25、.5 依序是 image_size、score_threshold、nms_iou
    # decode_grid 回傳「每張圖一個 dict」的 list；[0] 取第一張（也是唯一一張）
    pred = decode_grid(model(input_image.unsqueeze(0)),64,.25,.5)[0]
original_boxes = undo_letterbox(pred['boxes'], meta)  # 用這張圖自己的 meta 還原到原圖座標
```



傳空框表示推論時沒有 GT，不代表圖中沒有物件。模型和 decoder 都在 64×64 補邊後座標運作，最後用這張圖的 meta 還原。

灰階與 RGBA 也要先 convert RGB，否則軸數或卷積通道不符。若檔案有透明度，直接轉 RGB 只是丟掉 alpha，不會自動合成指定底色；處理透明圖時要先決定背景。

??? note "透明 PNG 的背景如何固定"

    透明處仍有檔案儲存的 RGB，直接 convert 會露出它，常為黑色但不保證。要合成白底，可先建 `background = Image.new('RGB',image.size,(255,255,255))`，再 `background.paste(image,mask=image.getchannel('A'))`。alpha 遮罩 0 保留底色、255 使用原色，中間值按比例混合。

## 載入的是哪個模型、哪套類別

checkpoint 是保存模型權重與設定的檔案。這裡讀回 Python dict：`model_state_dict` 裝權重，`config` 記 image_size／grid_size／width，`class_names` 是有順序的類別名稱。第 7 章存檔還可含 optimizer 狀態等推論用不到的欄位。

state_dict 是參數名稱到 tensor 的對照，例如兩類 width 8 的 `head.weight` 為 `[7,32,1,1]`。它不保存模型的層與接法，因此先按 config 重建同結構 GridDetector，再載入權重：

``` { .python data-excerpt="scripts/detect_image.py" }
def load_grid_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    ...
    config, classes = checkpoint['config'], checkpoint['class_names']
    ...
    model = GridDetector(num_classes=len(classes),grid_size=config['grid_size'],width=config['width'])
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    ...
    model.eval()
    return model, config, classes
```



傳給 `load_state_dict` 的是 `checkpoint['model_state_dict']`，不是整個 checkpoint。image_size 用於 letterbox、decode，不是模型建構參數。`strict=True` 要求名稱一一對上；參數／buffer 缺少或多出會報錯，shape 不同無論 strict 都會報錯。三類 head 為 `[8,32,1,1]`，不能直接接兩類的 `[7,32,1,1]`。

`load_grid_checkpoint` 還檢查尺寸設定為正整數、類別名稱非空且不重複，若有 config.num_classes，必須等於名稱數。這些只能抓格式，抓不到名稱被換錯：id 0、1 仍是訓練時紅、藍矩形，只將 class_names 改為 `['car','person']`，會把紅矩形顯示為 car，沒有學到汽車。類別順序必須與訓練相同。

這個 loader 只接受本書 GridDetector 格式。只有 state_dict 的檔案要另外提供架構設定與有序類別；其他 YOLO 架構的權重不能直接交給它。

??? note "torch.load 與 state_dict 的細節"

    `map_location='cpu'` 將 tensor 載到 CPU，GPU 存檔也能在沒有 GPU 的機器讀取。`weights_only=True` 並非只讀權重，它允許 tensor、數字、字串、list、dict 等單純資料，所以 config、class_names 也讀得到。torch.load 使用 pickle，這個設定會拒絕允許清單之外的函式／類別；PyTorch 稱其「就目前所知是安全的」，不明來源檔案仍宜在隔離環境載入。

    有 BatchNorm 的模型還有 running_mean 等 buffer，state_dict 也會存；本模型沒有 buffer。

## 快速實驗如何分開檢查座標與權重

頁首 notebook 不需自備圖片，程式先畫上面的黑底紅矩形，寫成 PNG 再讀回。在 CPU 用第 7 章生成圖訓練三步，保存 checkpoint、重建模型並載入。三步讓存檔參數確實更新，又能很快完成流程；沒有下載預訓練權重，也不以三步宣稱偵測品質。

在 repo 根目錄可執行 `PYTHONPATH=. python lesson_cases/08-own-images.py`，七行輸出分別檢查：

1. 前兩行是框座標往返：CHW `(3,80,120)`，輸入框與上面手算一致，還原約 `[20,10,60,30]`。float32 可能把 60 算為 59.999996…，因此 print 取四位小數，assert 用 allclose 容差。
2. 第 3～5 行是存檔、載入、PNG 推論。重新載入前後同圖 logits **逐值相等**，才是權重套用成功的證據；框數不是。`detect_image()` 真正讀圖，只畫 score≥.25 的框，保存 PNG 與原圖座標 JSON，並核對記錄門檻與每框分數。三步模型的框數多少都不代表照片效果。
3. 第 6 行故意保存三個類別名稱、config 卻寫兩類，要求載入時拒絕，不等 shape 錯才發現。
4. 第 7 行以人工已知輸出檢查整段座標：紅框正格填能解碼出輸入框的四項，obj=10、紅藍類別 logits=10／−10，其他 obj=−20；decode 再 undo_letterbox 應回到原框。這份輸出不是模型學出的。

## 用存好的模型，對指定圖片推論

CLI（command-line interface／命令列介面）用指令與參數執行。先建立第 7 章 160 步模型，再換圖片路徑：

```bash
python -m miniyolo.train --steps 160 --samples 32 --device cpu
python scripts/detect_image.py --image my.png --checkpoint artifacts/runs/grid-learning/checkpoint.pt
```

jpg 也可；先用訓練腳本的 `artifacts/runs/grid-learning/validation-00.png` 能對照紅藍合成圖。兩行在 repo 根目錄執行，不需 PYTHONPATH：`python -m` 由目前目錄找模組，detect_image 自行加入 repo 路徑。

輸出預設為 `artifacts/runs/predictions/my-image.png` 和旁邊同名 JSON，可加 `--output` 改位置。橙框上寫類別與 score；JSON 保存原圖 boxes、scores、labels、class_names 與實際門檻。沒有框仍保存圖片與空陣列。CLI 摘要也列框數、score_threshold、nms_iou、路徑與類別。

手機照片可能用 EXIF（Exchangeable Image File Format／可交換影像檔案格式）記旋轉方向。Pillow 不自動按這個標籤轉正，輸出與框以實際儲存畫素為準。若方向與相簿不同，先用 `PIL.ImageOps.exif_transpose()` 轉正另存，再推論。

### 在 Colab 用自己的照片

本節 notebook 裡沒有上面這兩行指令，可以照下面的步驟自己加：

1. 先執行 notebook 最上方的環境格。它會把目前資料夾切換到下載好的 repo，之後的指令都在 repo 根目錄執行。
2. 點左側的「檔案」面板（資料夾圖示），上傳你的照片。上傳的檔案通常放在 `/content/` 底下，例如 `/content/my.jpg`；不確定時，可以在檔案面板裡從該檔案的選單複製完整路徑。
3. 另開一個 code cell，訓練第 7 章的 160 步模型；同一個執行階段只要跑一次。開頭的 `!` 表示把這一行交給命令列執行：

    ```python
    !python -m miniyolo.train --steps 160 --samples 32 --device cpu
    ```

4. 再開一個 cell，對照片推論並顯示結果（`--image` 換成你的照片路徑）：

    ```python
    !python scripts/detect_image.py --image /content/my.jpg --checkpoint artifacts/runs/grid-learning/checkpoint.pt
    from IPython.display import display
    from PIL import Image
    display(Image.open('artifacts/runs/predictions/my-image.png'))
    ```

    框的座標、分數與類別存在同資料夾的 `my-image.json`。


## 畫框門檻與評估門檻各自留下什麼

CLI 預設 score≥.25，為本書顯示門檻，不使用 checkpoint 裡評估的 .05。未訓練時每格約 `.12×.5=.06`，.05 會留下 16 個低分框；一堆框看不出學會什麼，也不能判斷權重是否載入。權重應由上面的同圖 exact logits 比對確認。

`--score-threshold` 可指定顯示門檻，例如 .05 看低分候選；`--nms-iou` 可指定同類去重門檻。NMS 預設讀 config.nms_iou，本節與第 7 章皆為 .5，沒有該欄也採 .5。兩值須在 0～1，並印在摘要、保存到 JSON，才能知道實際畫了哪一組候選。

完整 CLI 在還原時裁到原圖範圍，再丟零面積框。完全在上 padding 的輸入框 `[20,2,30,8]`，還原 y 約 −14.9、−3.7，裁到原圖後兩者都為 0，高度為零，會被刪除。前面的短示意程式只到 undo_letterbox，完整 CLI 才有刪框這一步。

讀圖→RGB／CHW→letterbox→加 batch→模型→decode／NMS→還原／零面積過濾→PNG／JSON，都是推論的一部分。若量速度，讀檔、前處理、模型與後處理需各自交代，不能只報 forward。

## 選 letterbox，需要承擔什麼

letterbox 保留物件比例，本例 40×20 變約 21.3×10.75，仍近 2:1；代價是只有 64×43 是圖像內容，小物件也縮小，極端長寬比讓內容占比更低。直接拉正方形沒有 padding，但框變約 21.3×16，物件變形。

兩者只要保存 sx／sy（letterbox 另存 padding）都能還原。訓練與推論必須用同一種前處理；比較兩者效果，也要各自訓練並用相同方式推論。本節補 0，第 7 章合成圖沒有 letterbox，背景為 0～.04 的近黑雜訊。原圖縮放時框也要一起換算；傳空 GT 不是空圖；padding 後座標不能直接當原圖，也不能借另一圖 metadata。

## 自主練習

練習 1：原圖 `[0,0,120,80]` 的整張框，換算到輸入中是什麼？

??? note "參考答案"

    `[0,10,64,53]`，不包括上下 padding。換算一樣是先乘實際比例、再加 padding：x1=0×64/120+0=0、x2=120×64/120+0=64；y1=0×0.5375+10=10、y2=80×0.5375+10=53。上方 0～10、下方 53～64 是 padding。

練習 2：反過來，輸入框 `[32,32,48,48]` 還原到原圖是多少？

??? note "參考答案"

    約 `[60,40.93,90,70.70]`。還原是先減 padding、再除以實際比例：x1=(32−0)/(64/120)=60、x2=(48−0)/(64/120)=90；y1=(32−10)/0.5375≈40.93、y2=(48−10)/0.5375≈70.70。四個值都在原圖範圍內，不必裁切。

## 查看本節輸出檔

跑完本節程式後，輸入圖 `my_image.png`、3 步 checkpoint `checkpoint.pt`、`detect_image()` 存下的 `prediction.png` 與 `prediction.json`，以及快速實驗第 6 行故意做壞的 `wrong-class-count.pt`，都會留在 `artifacts/lesson-08-own-images/`；PNG 與 JSON 的路徑印在輸出第 4 行。`prediction.png` 是原圖疊上 score 至少 0.25 的框，沒有框時就和原圖一樣。`prediction.json` 記錄原圖座標的框 `boxes`、分數 `scores`、類別 `labels`，以及所用的門檻 `score_threshold`、`nms_iou`；沒有框時，這三個陣列都是空的。這些檔案只存在你自己執行的電腦，或這次的 Colab 執行階段裡；這個資料夾列在 .gitignore（git 不追蹤的檔案清單）中，不會被提交進 repo。Colab 執行階段結束後，檔案就會消失，要保留請先下載。在 Colab 可另開 cell 查看：

```python
from IPython.display import display
from PIL import Image
import json
from pathlib import Path
display(Image.open('artifacts/lesson-08-own-images/prediction.png'))
print(json.loads(Path('artifacts/lesson-08-own-images/prediction.json').read_text()))
```

`prediction.png` 出自只訓練 3 步、還學不會偵測的模型，只用來確認流程接得通；頁首那張圖是依座標畫的示意圖，不是模型的預測結果。

<!-- curriculum-evidence:start -->

## 實際執行紀錄

本節的完整程式於 2026-10-08 在 AMD EPYC 9V74 80-Core Processor（2 個執行緒）上用 PyTorch 2.9.1+cpu 執行，程式裡的 assert 全部通過。下面是那次印出的原始輸出；輸出裡若有計時或訓練得到的數字，換一台電腦會略有不同。每個數字的意思，以本頁正文的說明為準。[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/08-own-images.json)

??? example "展開本次實際輸出"

    ```text
    original CHW (3, 80, 120) input box [[10.6667, 15.375, 32.0, 26.125]]
    metadata {'original_size': (80, 120), 'scale': 0.5333333333333333, 'scale_xy': (0.5333333333333333, 0.5375), 'padding': (0, 10), 'resized_size': (43, 64)} roundtrip [[20.0, 10.0, 60.0, 30.0]]
    3-step checkpoint safely reloaded: exact logits; original-space box count 0 at score threshold 0.25
    prediction PNG and JSON saved: artifacts/lesson-08-own-images/prediction.png artifacts/lesson-08-own-images/prediction.json
    pipeline evidence only, not photo detection quality
    class/config mismatch rejected
    artificial known box restored [[20.0, 10.0, 60.0, 30.0]]
    ```

<!-- curriculum-evidence:end -->

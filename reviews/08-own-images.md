只讀本頁、對應 case 及引用 SVG；已執行 `PYTHONPATH=. .venv-model/bin/python lesson_cases/08-own-images.py`，exit 0。原圖 `(3,80,120)`、輸入框 `[10.666667,15.375,32,26.125]`、metadata 與兩種還原結果均符合文字；roundtrip 的尾數差異是 float32 誤差。SVG 中框的位置、43 高有效圖、上 10／下 11 padding 也一致。未訓練模型輸出 0 框，程式和頁面都清楚標示只驗證介面；整張框練習答案正確，易錯處有實際用途，未見應刪的奇怪內容。

1. **自己的圖片與權重還沒有完整可照做的入口。** `docs/lessons/08-own-images.md:31` 僅說替換 `Image.open` 與用 `torch.load` 讀取；`lesson_cases/08-own-images.py:30` 實際仍每次建立隨機模型，沒有套用權重。只讀本節的新手容易改完照片路徑、甚至讀了 checkpoint，卻繼續用隨機參數推論。建議在第 31 行加入一段完整替換範例，明列 `image_path`、`checkpoint_path`、相同設定的 `GridDetector(...)`、`model.load_state_dict(state_dict, strict=True)`、`model.eval()`，再接本頁推論。註明檔案須為本課 GridDetector 的原始 state_dict、class ID 順序與訓練前處理須一致；若保存格式是包裝字典則須先取對應欄位，不能將任意 YOLO 權重交給它。

2. **張量與 metadata 的軸順序尚未寫成明確契約。** `docs/lessons/08-own-images.md:15–22` 列出 metadata 名稱與 `unsqueeze(0)`，但未說模型收到 `[1,3,64,64]` 的 NCHW，也未定義 metadata tuple 順序。case 的輸出同時使用 `original_size=(80,120)`、`scale_xy=(sx,sy)`、`padding=(0,10)`，新手容易把 H/W 與 x/y 混為同一種排列。建議補上：影像為 RGB、float32、值域 `[0,1]`；模型輸入為 NCHW，單張 N=1；`original_size`／`resized_size` 是 `(H,W)`、`padding` 是 `(left,top)`、`scale_xy` 是 `(sx,sy)`。這會讓換圖、批次還原與自行畫框時有可查的契約。


## 作者修訂與驗證（2026-10-02）

新增scripts/detect_image.py：weights_only載入、config與有序class契約、strict state_dict、PIL RGB、letterbox、decode／NMS、框還原及PNG／JSON輸出。case自建三步真更新後checkpoint，安全重載後logits逐值相等；也拒絕class/config不符。另用root160步模型及validation-00完成CLI，輸出2框；頁面明說只有紅／藍合成圖能力，沒有照片品質主張。

已實跑 `PYTHONPATH=. .venv-model/bin/python lesson_cases/08-own-images.py`，exit code 0，相關assertions通過。此段是作者修改與執行紀錄，並非獨立reviewer重審通過的宣告。

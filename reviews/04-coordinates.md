# 第 04 課陌生讀者審閱

只讀本課 Markdown、範例程式與對應圖。`PYTHONPATH=. .venv-model/bin/python lesson_cases/04-coordinates.py` 成功；原圖、stretch、letterbox、奇數 resize 的輸出與文字及圖一致，直向練習也得到 `[20,8,36,40]`。x／y 方向、畫布 pixels／正規化座標、先扣 padding 再除 scale、半開區間及奇數補邊的說明足以理解，未見需刪除的異常內容。

1. **浮點座標契約未明說，合理的整數輸入會破壞往返。** `lesson_cases/04-coordinates.py:22–23` 用 `boxes.new_tensor` 建 scale，因此 `torch.tensor([[10,5,50,25]])` 的整數 dtype 會把 0.8 變成 0；實跑得到框 `[[0,16,0,16]]`，`undo` 得到四個 NaN。`docs/lessons/04-coordinates.md:44` 只說 `[N,4]`，第 60 行練習也直接列整數座標，讀者無從知道需要浮點 dtype。**具體修法：** helper 在第 14 行前把非浮點 `boxes` 轉為 `torch.float32`（保留原 device），或明確拒絕整數 dtype；第 44 行補上「座標 tensor 必須為浮點數」，並示範 `torch.tensor([[10,5,50,25]], dtype=torch.float32)`。補一個整數座標輸入檢查，確認能正常還原或收到清楚的錯誤。


## 作者修訂

helper現在將非浮點boxes轉為float32，文件明說圖片與座標dtype；新增整數標註輸入的有限值與精確往返assertion，CPU重跑通過，奇數尺寸與空框檢查仍通過。

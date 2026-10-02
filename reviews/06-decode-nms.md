# 第 06 課陌生讀者審查

範圍僅為 `docs/lessons/06-decode-nms.md`、`lesson_cases/06-decode-nms.py` 與 `docs/assets/diagrams/06-decode-nms.svg`；已閱讀渲染圖並實際執行案例。Colab 佔位不列問題。

## 關鍵問題（1 項）

1. **練習的操作指示會讓可執行案例先斷言失敗，讀者看不到預期答案。**
   - 位置：教材第 58 行；程式第 14、28、55–61 行。
   - 教材叫讀者「只把 NMS 門檻改 0.6」，但修改 `class_nms` 的預設值或第 60 行呼叫後，第 61 行仍要求 `[2, 1]`，實際正確結果 `[2, 1, 0]` 會觸發 `AssertionError`。再將 `decode` 的預設 score 門檻改成 0.70，則第 55 行仍要求三框，會在輸出前失敗；後面的 `boxes[1]`、scores 與索引斷言也對應原先三框，不能只改一個數字便完成練習。
   - 具體修法：保留原案例所有門檻與斷言，在第 73 行後新增獨立練習呼叫；教材第 58 行改成「保留主案例，在其後分別用下列參數呼叫」。例如：

     ```python
     keep06 = class_nms(boxes, scores, labels, threshold=.6)
     assert keep06.tolist() == [2, 1, 0]
     print("NMS .6 keeps", keep06.tolist())

     boxes70, scores70, labels70 = decode(logits, score_threshold=.70)
     assert torch.allclose(scores70, torch.tensor([.72, .855]), atol=1e-6)
     print("score .70 keeps", [round(s.item(), 3) for s in scores70])
     ```

     同時交代兩次實驗都以原始 logits 獨立進行，`keep06` 指向原三框，score 篩選後的索引則指向新回傳的候選陣列，避免把兩組索引混用。

## 已核對、無需修正

- 教材第 9–26 行充分交代 `[row, col, channel]`、`cell_xy=(col,row)`、格內 xy 與整圖 wh 的不同尺度。數值確實得到中心 `(36,28)`、尺寸 `(16,8)`、框 `[28,24,44,32]`。
- 教材第 30–42 行的 score 乘法、先篩分再依類別 NMS、嚴格大於 IoU 門檻才抑制，以及 NMS 無法刪除不重疊高分錯框，均與程式一致。
- 圖將 64×64 畫布等比例放大三倍；藍、黃、紅框的位置和尺寸、真值填色、被移除框及保留框均與人工數值相符，文字沒有截斷。
- 執行 `PYTHONPATH=. .venv-model/bin/python lesson_cases/06-decode-nms.py` 通過全部原有斷言，輸出 scores `[0.64,0.72,0.855]`、IoU `0.5385`、NMS 索引 `[2,1]`；score 0.75 只留下背景錯框，不同類相同框都保留，空輸入回空索引。
- 另外以不改動原檔的獨立呼叫確認：NMS 0.6 保留 `[2,1,0]`，score 0.70 留下 `[0.72,0.855]`。練習答案數值正確，問題只在操作方式和固定斷言的衝突。

沒有需刪除的概念或圖；除上述練習操作問題外，未發現影響本課理解的關鍵錯誤。


## 作者修訂

保留基準門檻與所有基準assertion；新增獨立NMS threshold=.6及decode score_threshold=.70練習呼叫與assertion，文件明說原三候選與新兩候選索引不能混用、兩實驗不累積。2026-10-02 CPU重跑退出碼0；新增輸出分別[2,1,0]與[0.72,0.855]，原檢查仍通過。

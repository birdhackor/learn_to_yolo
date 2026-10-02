從只有基礎 PyTorch／CNN、未讀其他課文的讀者角度，這課的 baseline → 單項改動 → validation 選擇 → chosen test 評估一次可以照程式完成。資料 seed、模型 seed、batch、optimizer、320 steps 總預算、AP 門檻、顯示門檻、保留規則與時間量測範圍都有具體設定。以下只有兩項需要補齊的交付／練習缺口，未發現數值造假或需要刪除的效能承諾。

實跑核對：在獨立 `/tmp` 目錄，以現有 `.venv-model/bin/python`、`PYTHONPATH=/workspace/learn_to_yolo` 執行原程式，exit 0；PyTorch 為 2.9.1+cpu。validation mAP50 為 0.444444／0.701389，precision／recall 與覆蓋計數全部符合課文第 35–43 行；chosen test mAP50 0.445238、precision 0.636364、recall 0.538462，參數 15,511，符合第 47、49 行。生成 SVG 與原圖逐位元相同。這次訓練耗時約 0.397／0.405 秒、端到端約 1.220 ms，與原文不同，但第 33、49 行已正確限定為單次硬體相關數值，不能據此宣稱 loss weight 加大會變快。原課文、程式與圖皆未修改。

1. **P1：weight2 練習會交付被標成 weight10 的報告與圖。** 位置：`docs/lessons/17-capstone.md:57`；`lesson_cases/17-capstone.py:130`、`:89`、`:93`、`:140`。照練習把 changed 的訓練權重由 10 改為 2，訓練本身正確，但 report 的 `box_weights` 仍是 `[5, 10]`，SVG 的描述與第二列標題仍寫 weight10。實際只替換該 fit 參數後，validation mAP50 得到 0.471528，依既定規則 `keep_change=true`；交付檔卻宣稱保留的改動是 weight10。這會使讀者無法由報告辨識實驗條件，直接破壞結業交付的可查核性。具體修法：在 main 中只設一次 `baseline_weight=5` 與 `changed_weight=10`，讓 fit、report、`save_panel` 的 title／desc／標題都使用這兩個值；課文練習明確寫「將 changed_weight 改為 2，其他設定不變」。

2. **P2：最後要求用分類錯誤與背景 FP 決定下一個假設，但沒有可操作的診斷輸出。** 位置：`docs/lessons/17-capstone.md:19`、`:21`、`:57`；`lesson_cases/17-capstone.py:52`、`:57`、`:65`、`:142`。目前 diagnostics 只看同類候選，`below_iou10` 把「框到物件但分類錯誤」「沒有候選」「框完全偏離」合在一起，而且完全不數沒有對應 GT 的預測。覆蓋診斷的限制在第 19 行寫得正確；然而只有基礎知識的讀者仍無法照第 57 行完成後半段練習，也無法用現有 report 排除分類／背景問題。具體修法：保留現有覆蓋統計，另加一段小診斷範例及 report 欄位：逐 GT 比較不限類別與同類別的最佳 IoU，列出 IoU≥.5 但類別錯的案例；對固定 score .05、matching IoU .5 的一對一匹配後剩餘預測列出 FP，並標明哪些與所有 GT 的 IoU 都很低、哪些是重複或類別錯誤。附一張帶分類錯誤或背景 FP 的 validation 例子，並要求練習交付這些計數與下一個假設。若本課不打算提供這一步，應把第 57 行練習縮成現有程式能完成的「交付保留／不保留與 GT 覆蓋診斷」，不要宣稱讀者已能完成分類和背景診斷。

## 作者修訂（2026-10-02）

- P1：在`main`集中設定`baseline_weight`與`changed_weight`；fit、report與SVG的title、desc、兩列標題都使用這兩個參數。練習明确指定只改changed_weight為2，交付當次報告而不沿用本頁weight10表格。
- P2：保留原GT覆蓋欄位，新增兩份`errors`診斷：逐GT列出同類不足但錯類IoU≥.5的案例；score .05、matching IoU .5的一對一匹配後，把FP分成背景、定位、錯類與重複。逐例保留圖片／預測編號、class、score與最佳IoU。正文解釋每類規則及零錯類計數的限制，練習要求交付計數和一例證據。SVG新增validation #10的實際背景FP，score約.068、最佳IoU為0，以紅框標示。
- 驗證：修改後以`.venv-model/bin/python`及`PYTHONPATH=.`跑預設案例，exit 0，原validation／chosen test數值保持一致；baseline為4定位FP＋1背景FP，weight10為2定位FP＋1背景FP。另於隔離暫存目錄實跑weight2，report為`[5,2]`、mAP50 .471528、keep_change=true，圖的title／desc／標題均無殘留weight10。以人工小案例檢查四種FP、錯類GT覆蓋與空GT，全部符合預期；預設SVG已重新生成。

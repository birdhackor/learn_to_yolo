# 第 07 課 held-out 審閱

以只具基本 PyTorch/CNN 知識、對專案陌生的讀者角度，僅閱讀 `docs/lessons/07-heldout.md` 與 `lesson_cases/07-heldout.py`。本次頁面尚未引用可供檢查的圖；依任務要求不把待 root 補入的實測記錄與 Colab 占位列為問題。未修改教材。

結論：人工 fixture、三步模型與訓練成果的界線清楚，沒有發現數值或程式邏輯不一致。仍有一個執行入口問題，以及幾處會讓指定讀者依賴外部教材才能理解的缺口。

1. **P2：本頁列出的直接執行方式缺少本地套件路徑。** `docs/lessons/07-heldout.md:29` 寫 `python lesson_cases/07-heldout.py`。在目前 repo 根目錄用 `.venv-model/bin/python lesson_cases/07-heldout.py`，會在 `lesson_cases/07-heldout.py:2` 報 `ModuleNotFoundError: No module named 'miniyolo'`；加上 `PYTHONPATH=.` 後通過。建議本頁明說從 repo 根目錄執行 `PYTHONPATH=. python lesson_cases/07-heldout.py`，或列出已安裝本地套件的必要前提。

2. **P2：三個閾值的比較對象與作用還不夠具體。** `docs/lessons/07-heldout.md:21` 已把候選保留、去重、判定分開，但只有 PyTorch/CNN 背景的讀者仍不知道 NMS IoU 和 matching IoU 分別在比較什麼；`.5` 相同尤其容易讓人誤認為同一步驟。建議各補一句：score 門檻先篩掉低分候選；NMS 在同圖、同類的預測框之間比較重疊，刪去低分重複框；matching 則在同圖、同類的預測框與 GT 之間比較重疊，判定 TP 並標記該 GT 已被配對。對應 `lesson_cases/07-heldout.py:41` 的位置參數，也可寫成註解或具名參數，提高對照性。

3. **P2：人工例子的 AP=.5 正確，但計算理由缺少一步。** `docs/lessons/07-heldout.md:17` 從「曲線只達 recall=.5」直接推出「插值 AP50=.5」。最大 recall=.5 本身不足以推出 AP=.5；還需要這個 fixture 在 recall 0 到 .5 的插值 precision 都是 1。建議補一句：all-points 插值取往更高 recall 看去的最高 precision，因此本例 0 到 .5 的高度為 1，.5 到 1 的高度為 0，面積是 `1×.5=.5`。這也能説明為何後來的兩個 FP 降低最後 precision，卻沒有降低本例 AP。保留 `:37` 的刪 FP 練習，與此説明相互印證。

4. **P2：關鍵縮寫首次出現時未展開。** `docs/lessons/07-heldout.md:3-17` 直接使用 AP、GT、TP、FP、FN、mAP；`:21`、`:33` 使用 NMS、PR、checkpoint。前置連結指出應先讀哪裡，但這次限定只讀本頁的讀者無法由 PyTorch/CNN 知識推知這些詞。建議在現有敘述中簡短展開 GT（標註真值）、TP（正確配對）、FP（誤報）、FN（漏檢）、PR（precision–recall），並説明 AP 是插值 PR 曲線面積、mAP 是有 GT 類別 AP 的平均。NMS 可由上一項補句定義；checkpoint 可改成「保存的模型權重」。人工框座標 `:9` 也宜標註順序為 `[x1,y1,x2,y2]`。

5. **P3：held-out 與最終 test 的敘述清楚，但程式命名會削弱這個區別。** `docs/lessons/07-heldout.md:35` 清楚説明反覆拿 held-out 選參數會成為 validation，最終 test 要另留；但 `lesson_cases/07-heldout.py:29`、`:41-42` 將同一份 heldout 寫成 `test_images`、`test_anns`。建議改名為 `heldout_images`、`heldout_anns`，與 `:27` 的資料來源及 md `:24-25` 一致。這只是教學命名問題，沒有發現資料混用。

验证命令：`PYTHONPATH=. .venv-model/bin/python lesson_cases/07-heldout.py`，exit code 0。人工 fixture 得到 `ap_per_class={0:0.5,1:None}`、`map=0.5`、`precision=0.3333333333333333`、`recall=0.5`；三步模型得到兩類 AP、mAP、precision、recall 全為 0，與頁面只把它當流程檢查的定位一致。

不需刪除的內容：`docs/lessons/07-heldout.md:5` 區分 all-points、VOC2007 與 COCO 的句子有助固定 AP 協議；`:33` 的合成分布限制、`:35` 的 validation/test 角色、`:37` 的常見錯誤與練習都與本課主題直接相關。沒有發現需刪除的大段旁支內容。


## 作者修訂與驗證（2026-10-02）

補GT／TP／FP／FN／PR／AP／mAP、precision／recall公式與AP區間面積；明分NMS是pred-pred、matching是pred-GT。案例改heldout命名與具名門檻；命令加入PYTHONPATH=.。root的160步独立結果與圖保持完整。

已實跑 `PYTHONPATH=. .venv-model/bin/python lesson_cases/07-heldout.py`，exit code 0，相關assertions通過。此段是作者修改與執行紀錄，並非獨立reviewer重審通過的宣告。

# 第 15.2 課 Area Attention 審查

審查範圍：只讀 `docs/lessons/15-area-attention.md`、`lesson_cases/15-area-attention.py`，並確認本文沒有引用區域示意圖；未讀其他教材。以對專案陌生、具基本 PyTorch／CNN 知識的讀者為準。Colab 佔位不列為問題。

結論：CNN／global／area 的單層互動範圍、水平帶分區、pair 計數及跨區訊號限制都講對了。主程式可以執行，數值與文字一致。需要補兩處，才能讓讀者直接完成練習，並清楚區分本例與原版 YOLOv12。

## 1. 「改 areas」練習會撞上固定參數與斷言

位置：教材第 44 行；程式第 26、30–33、37 行。

教材請讀者改 `areas=2`，但程式沒有集中設定的 `areas` 變數，兩次 area 計算各寫死 `4`。只改第一次呼叫，未干預與干預的分區方式就不同，第 31 行會失敗；兩次都改成 2，第 33 行的 `64` 仍會失敗。改成 1 時，area 就是 full，第 31 行的「第一區完全不變」斷言與第 37 行寫死的 `area=False` 也不再成立。這些失敗會讓初學者誤以為自己的分區推論錯了。

具體修法：在 `main()` 設一個 `areas = 4`，第 26、30 行都使用它；pair 斷言改成 `tokens.shape[1] ** 2 // areas`。干預檢查改為比較實際的 token 0 輸出，根據 token 15 是否與 token 0 同區決定預期結果，輸出文字也由比較結果產生。教材第 44 行明確說只改這個變數。若暫時不改程式，至少列出 areas=2 必須修改兩次呼叫與 pair 斷言，areas=1 必須調整跨區斷言。

## 2. 沒有明確交代 toy 單 head 與原版 multihead 的 shape 差異

位置：教材第 7、11–20、42 行；程式第 11–14、22 行。

第 7 行說原版有 multihead、位置卷積及輸出投影，卻只明確指出位置卷積被省略。範例後續用 `sqrt(C)` 與三維 weights，未交代這其實是 **一個 head、head dimension 就是 C=2**，而且輸出投影也省略了。第 42 行雖提醒不要把 area 當 head，讀者仍看不到兩者如何同時出現在 tensor 中，也可能把這個公式直接套到原版的總 channel 數。

具體修法：在第 7 行後加上「本例 num_heads=1、head_dim=C=2，省略位置卷積與輸出投影；areas 是空間分組數，不是 head 數」。再加一個 shape 對照：若原版有 M 個 heads、D=C/M，每個 head 的 Q／K／V 可表示為 `[B×A,M,N/A,D]`，weights 為 `[B×A,M,N/A,N/A]`，縮放因子是 `sqrt(D)`；本例 M=1，所以省略那個大小為 1 的維度。如此也能說明目前 256／64 是單 head 的 weights 元素數，M-head 情況會再乘上 M，而 `N²/A` 是每個 head 的 pair 數。無需搬入完整 YOLOv12 程式。

## 已驗證、無問題的部分

- `PYTHONPATH=. .venv-model/bin/python lesson_cases/15-area-attention.py` 成功，印出 weights `(1,16,16)`／256 與 `(4,4,4)`／64、首 token `0.4688`／`0.0938`，loss 約 `0.0048`。
- 使用相同 QKV 權重另外驗證 areas=1／2／4：pair 數為 256／128／64，首 token 為 0.46875／0.21875／0.09375。token 15 各 channel 加 5，首 token 的變化為 0.31250／0／0；token 3 各 channel 加 5，變化為 0.31250／0.62500／1.25000。教材的干預與練習答案正確。
- MSE backward 後 Q、K、V 三段權重各有非零梯度；SGD 可執行。此處作為可訓練性的簡短實驗足夠，沒有精度收益的錯誤推論。
- 第 26–28 行有區分 pair 成本與總延遲；第 34、40、42 行有區分單層隔離與完整網路跨區傳遞，CNN 對照沒有編造實測結果。
- 本文沒有引用圖，沒有圖文矛盾可指出。若新增本節圖，應畫出 row-major 的 0–15 編號及四條水平帶，不應畫四個 2×2 象限。沒有需要刪除的內容。


## 作者修訂紀錄（2026-10-02）

已集中areas與changed_index設定，兩次attention、pair數與跨區干預assert/輸出均動態計算。補單head省略M軸、原版multihead QKV/weights shape與sqrt(D)尺度；pair數明示每head。CPU主例及areas=2、areas=1、changed_index=3三種練習均通過，pair為64/128/256且干預是否同區判斷正確。

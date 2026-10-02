# 第 19 課審查

只讀 `docs/lessons/19-tracking.md`、`lesson_cases/19-tracking.py` 與所連的 SVG。以懂基本 Python／PyTorch、不了解專案的讀者檢查。

原程式在獨立暫存工作目錄執行成功，未覆寫教材圖。last-box 的 IDs 為 `[[1,2],[1,2],[2,1],[2,1],[2],[2,3]]`、switches=3；motion 的 IDs 為 `[[1,2],[1,2],[1,2],[1,2],[1],[1,2]]`、switches=0。非貪心反例、空 detections 的 assert 通過。已實際看過原 SVG，圖中的 IDs 與上述輸出一致。

整體可以理解：一對一與 unmatched、真值 A／B 與 tracker ID 的區別、交叉造成的兩次 switch、速度乘 frame gap、未匹配 track 的壽命，均有正文與可執行程式支撐。第 7、24、44、50 行明確限定這是人工等速案例與小型 tracker，沒有冒充完整 SORT 或完整 MOT 指標。沒有需要刪除整節的理由。

1. **第 5 幀為何另外增加一次 switch，少了關鍵的 unmatched 推導。**位置：`docs/lessons/19-tracking.md:20`、`:38`、`:42`、`:44`；對應 `lesson_cases/19-tracking.py:49`、`:52`、`:59`。正文說 track 可保留到第 5 幀，也列出 B 得到 ID3，但沒有串起「保留仍不代表能配對」。初學者可能以為第三次 switch 是舊 track 過期，或誤解 `max_age=2` 已保證恢復原 ID。實際上 last-box 的 ID1 在第 5 幀仍在 `tracks`，框是 `[16,20,28,32]`；B 的新偵測為 `[0,20,12,32]`，IoU=0，低於 .1，故兩者 unmatched，再新建 ID3。建議在第 38 行後補這個兩框數值及一句「max_age 只保留候選，重接還必須通過 IoU 門檻；舊 ID1 此時仍保留」。如此讀者能完整說明 3 次 switch 的來源。

2. **`max_age=1` 練習會讓圖的 switch 數字與實際結果矛盾。**位置：`docs/lessons/19-tracking.md:52`；`lesson_cases/19-tracking.py:88`、`:91`、`:95`、`:129`。實測 `Tracker(motion=True, max_age=1)` 得到最後一幀 `[1,3]`、motion switches=1，符合文字答案；但圖標題硬寫 `Velocity + IoU: 0 switches`，SVG 的 desc 也固定宣稱保留身份。依練習只改 assert，輸出圖仍會標錯。建議讓 `save_panel` 接收本次 `raw_switches`、`motion_switches`，用它們產生標題與 desc，或明確要求練習同步修改這些文字。另請重新生成並提交 SVG：現存圖的 viewBox 高度是 390，少了現有程式第 110 行的「A/B 為展示分列、matching 使用相同 y」註記；程式目前生成的是 420 高，重跑可同步這個差異。

## 作者修訂（2026-10-02）

- 問題1：補上第5幀last-box舊ID1仍存活、舊框`[16,20,28,32]`與B新偵測`[0,20,12,32]`的IoU=0推導；兩者unmatched所以建立ID3。明說max_age只保留候選，沒有保證重接，第三次switch並非舊ID過期。
- 問題2：`main(max_age=2)`把參數傳給兩條run；max_age1／2分別檢查motion的switch數及最後IDs。save_panel接收兩份實際switch計數與max_age，產生title、desc與列標題，移除固定0／3宣稱。正文練習對照max_age1時motion的`[1,3]`與1次switch。重新生成預設420高SVG，包含展示分列、matching同y的註記。
- 驗證：修改後預設案例exit 0，IDs維持原序列、switches 3／0，非貪心與空detections斷言通过。隔離實跑max_age1，motion最後`[1,3]`、switches=1，程式斷言及圖title／desc／列標題一致；預設SVG通過XML解析。

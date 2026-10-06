# 審查紀錄：簡易 tracking

審查範圍：`docs/lessons/19-tracking.md`、頁面上的圖（`docs/assets/diagrams/19-tracking.svg`），以及 `lesson_cases/19-tracking.py` 與它 import 的 repo 模組；頁尾自動產生的執行紀錄區塊不在範圍內，由 `scripts/validate_curriculum_evidence.py` 對照紀錄檢查。審查者都是 AI，沒有真人學生測試。這份紀錄涵蓋的內容以 SHA-256 記在 `reviews/coverage.json`；頁面、圖或程式之後再改，`scripts/validate_lessons.py` 就會要求重新審查。

## 獨立查核

頁面依目前的程式改寫後，由另一位 AI 獨立查核：在獨立的副本執行該節程式、照頁面做練習，逐句對照程式、執行紀錄與手算，檢查程式摘錄與網頁轉換，並從初學讀者（高中程度、數學好、程式新手）的角度看用詞與說明順序。有必要問題時，修正後再由另一位 AI 複查；建議事項另外處理，處理後同樣再查一次。

### 第 1 次查核：通過

結論：通過。沒有必要問題，只有兩項 should：一項是用詞統一，一項是第 124 行兩句的順序。頁面現在描述的是定稿程式（SHA e1fc77ce…），所有在暫存副本實際執行的結果，都和頁面說的一致。

1. 程式敘述
- 原程式 exit 0，印出 7 行，和受程式改動影響的段落的新輸出相同。
- 逐句核對了 `truth_boxes`／`frames`／`evaluate_detections` 的行為、回傳順序 (TP, 真實框數, FP)、斷言 `(11, 12, 0)` 在畫圖和所有 print 之前、印出的字串，以及圖的存檔路徑。
- 練習 1 照頁面做（只把 `def main(max_age=2):` 改成 1）：速度預測法第 5 幀是 [1, 3]，switch 是 1；上一框法 3；recall 那一行不變；圖標題變成 3 次／1 次，<title> 也寫 max_age=1。和參考答案相同。
- 改成 max_age=3，程式停在第一個斷言。
- Colab 看圖那一行用 IPython 8.18 測過，會產生 image/svg+xml。notebook 環境格有 os.chdir(repository)，所以相對路徑在 Colab 裡成立。
- runpy 接線範例和頁面上的區塊逐字相同，執行 exit 0。

2. 先前審查意見與受程式改動影響的段落
- 每一條都處理了：三段 git checkout／覆寫的注意事項已刪；「recall 是寫死字串」那句已改成程式實際計算的說明；圖例翻譯那句已刪；圖的路徑改成 artifacts/lesson-19/ids.svg（這個路徑和 artifacts/runs/ 都在 .gitignore 裡，所以刪掉那些警告是對的）。
- 全頁用 grep 掃過，沒有修訂或審查的經過敘述。
- decisions 的 I11 和程式衝突，編輯照程式改寫，這是對的。

3. 數字
- 以下都由我自己算過：11/12、FP=0；不同物件之間的 IoU 最多 0.2；6.2 的逐筆配法不論處理順序都是 11 個 TP；2 對 2 有 7 種配法；10 對 10 有 234,662,231 種（頁面寫約 2.3 億）。
- 影片實測的「第 6、7 幀回傳 [2]」取自現有紀錄（torch 2.9.1+cpu，Linux 機器），不是取自本機；本機第 7 幀是 []。編輯已把它列入待重錄數值清單。
- 附帶一點：18-video.py 和 verify_video_file.py 在紀錄之後也改過。18-video 的改動只在 main()、圖文和註解，不碰 fit_detector、run_stream、synthetic_frames，所以在錄製機器上重產時，這些值應該不變。重產後仍要核對一次。

4. 摘錄
- 摘錄比對工具印出 `docs/lessons/19-tracking.md []`。
- 兩個 python 區塊都不是逐字摘錄；示意區塊有寫明是示意。
- 正文沒有引用程式行號。

5. 可讀性：新段落清楚，新名詞（TP、FP、真實框、評估配對）都有說明。兩項建議見下表。

6. 建置與圖
- zensical build --clean --strict、validate_site、validate_preparation 都是 exit 0。
- validate_lessons 失敗，原因只有 review 涵蓋不到（19-tracking 等頁是 no review）。validate_curriculum_evidence 失敗，原因是紀錄過期。這兩項都是錄製和審查之前的預期狀態。
- 新圖 artifacts/lesson-19/ids.svg 用 qlmanage 轉成 PNG 看過：中文清楚、沒有重疊，viewBox、<title>、<desc> 都在。頁面的讀圖說明逐項對得上：列標題的切換次數、第 N 幀、A：ID1、B：漏檢、紅／藍／紫、上列第 2 幀互換、第 5 幀 B 變紫色。
- docs/assets/diagrams/19-tracking.svg 現在仍是舊的英文圖。它在各頁圖檔清單標為 generated；錄製時 verify_curriculum.py 的 SITE_FIGURES 會把新圖複製過去，record_evidence.py 的 OUTPUTS 也包含 docs/assets/diagrams。所以發布時，「上面那張圖就是頁尾執行紀錄那次畫出的檔案」這句會成立。
- 和 19 相關的檔案裡只有這一頁被改；程式、notebook、SVG、紀錄、review 都沒動。

日誌和 PNG 在暫存副本。

| # | 嚴重度 | 位置 | 發現 |
|---|---|---|---|
| 1 | 建議 | docs/lessons/19-tracking.md 第 26、122、124 行（新加的「真實框」） | 新加的句子把 `truth_boxes(f)` 的框叫做「真實框」。但同一頁第 12 行，以及術語表的 GT 條目和 6.2，用的都是「真值框（GT）」。讀者讀到第 122 行「和 `truth_boxes(f)` 的兩個真實框做評估配對（matching）」時，得自己推斷「真實框」就是第 12 行說的、6.2 評估用的真值框（GT）。寫作規則也要求術語和術語表一致。 |
| 2 | 建議 | docs/lessons/19-tracking.md 第 124 行前兩句 | 這段第一句先說「不論用 6.2 那種一次處理一個框的配法，還是本節的完整列舉，結果都是 TP=11、FP=0」，第二句才說「這組人工框沒有 score；程式用的是本節的完整列舉」。讀過 6.2 的讀者知道 6.2 是依 score 由高到低逐筆配，看到第一句會先卡在「沒有 score，要照什麼順序配」。答案在下一句，而且「先處理哪個框都一樣」要讀者自己推出來。內容沒錯（我窮舉過：每一幀的任何處理順序加起來都是 11 個 TP），只是兩句的順序倒過來。 |

最後一次查核的建議事項，在下方〈定稿修正〉逐項處理。

## 定稿修正

上面各項意見與先前查核留下的建議，由 AI 逐項核實後處理：必要問題全部修正，建議事項只在修正明確、範圍小時採用。

| # | 來源 | 意見 | 處理 |
|---|---|---|---|
| 1 | 先前查核 | 「真實框」和同頁及術語表的「真值框（GT）」不一致 | 已修正：把正文（執行紀錄區塊以上）所有的「真實框」統一改成「真值框」，包括第 126 行的「真值框數」，數字不動。 |
| 2 | 先前查核 | 「沒有 score」放在「6.2 配法結果相同」之後，讀者會先卡在要照什麼順序配 | 已修正：兩句對調：先說人工框沒有 score，沒辦法照 6.2 依 score 由高到低逐筆配，程式用的是完整列舉（門檻 0.5）；再說就算一次處理一個框，不論先處理哪個框，結果都是 TP=11、FP=0。另手算確認跨物件 IoU 最多 0.2；實跑後輸出 recall=11/12、FP=0。 |

修正後由另一位 AI 檢查這一批頁面（`docs/lessons/17-capstone.md`、`docs/lessons/18-video.md`、`docs/lessons/19-tracking.md`、`docs/lessons/20-deployment.md`）的改動，第 1 次：通過。檢查內容：每項改動是否符合程式、紀錄與引用的來源（需要時重算或重跑），回報已修正的必要問題是否真的修好、沒改的理由是否成立，改動是否符合寫作規範，網站嚴格建置與程式摘錄比對是否通過。

## 後續編輯的檢查

上面各輪之後的編輯（各頁的小修正、審查方式的說明），由另一位 AI 對照程式、紀錄與來源再檢查；檢查找到的問題處理後，再交給另一位 AI 檢查，直到沒有必要問題。

### 第 2 輪：上一輪的處理與審查紀錄：通過

讀了 reviews/19-tracking.md 並在頁面搜尋論文與函式庫說法。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | 摺疊區〈外觀特徵、Kalman filter 與常見追蹤器〉與匈牙利演算法段 | 頁面對 SORT、DeepSORT、ByteTrack、Kalman filter 與匈牙利演算法有說法，紀錄沒有記錄這些說法和原論文核對過，也沒有來源清單；〈驗證範圍〉說論文說法另由 AI 核對、查閱的來源記在該頁紀錄。 | 已處理：另做一次來源對照（SORT、DeepSORT、ByteTrack 等原論文），見〈來源對照〉。 |

### 補做的來源對照

頁面上關於原始論文、官方程式與函式庫行為的說法，由 AI 打開原始來源逐句核對。已逐條核對 docs/lessons/19-tracking.md 中關於外部方法、論文與函式庫的敘述（以工作目錄目前版本為準；它相對 HEAD 3152741 有未提交的修改），範圍包含摺疊區，跳過本課程自身程式的敘述與 curriculum-evidence 區塊。結果：must 0 條，should 4 條，集中在第 93 行與第 139–141 行。 (1) 第 93 行把「替不配訂代價」寫成正式系統的共同做法。但 SORT 的論文 §3.3 和官方 sort.py 都是先配完、再剔除低 IoU 的配對。我照抄 SORT 的函式跑了一個 2×2 例子，確認兩種做法的配對結果不同。 (2) DeepSORT 不只是「SORT 加外觀特徵」：它把配對依據換成外觀距離加 Mahalanobis 門檻，另加 matching cascade，IoU 配對只留在最後一輪。 (3) ByteTrack 那句少了兩輪配對的限定：低分框只配給第一輪剩下的 track，沒配上的低分框丟掉，不拿來開新 track。 (4) IDF1 的原論文和 HOTA 論文都把它定位為偏重身份的指標，不宜和 HOTA 並列為平衡偵測與身份的綜合分數。 核對後正確的部分： - SORT 是 Kalman 預測加 IoU 加匈牙利演算法。 - Kalman filter 的描述（不確定性、對偵測框的信任程度、需先設定雜訊）。 - 匈牙利演算法與 assignment problem 的說法（Kuhn 1955）。SORT 和 ByteTrack 的官方程式實際呼叫 lap.lapjv 解同一個指派問題，不影響頁面說法。 - 列舉數：2×2 是 7 種，10×10 是 234,662,231 種，約 2.3 億。 - 第 13 行的 ID switch 定義：和最近一次配到的幀比、漏檢的幀跳過、逐物件計次，與 TrackEval CLEAR 的 IDSW 計法一致。 - runpy.run_path、zip、IPython SVG 的行為描述。 - 第 184 行「每呼叫一次 update 才加 age」的假想寫法，正是 SORT、DeepSORT、ByteTrack 官方程式的做法；SORT 的 docstring 明文要求空幀也要呼叫 update，頁面的警告與此相符。 附帶觀察（不是頁面錯誤，頁面也沒說兩者相同）：SORT 官方程式在配對之後才用 time_since_update > max_age 刪除 track，所以 max_age=2 時間隔 3 仍可接回；本節在配對前就刪，兩者差一格。 重現腳本。我沒有修改任何 repo 檔案。

查閱的來源：

- SORT：Bewley 等，〈Simple Online and Realtime Tracking〉，arXiv 1602.00763v2（TeX 原始檔），摘要、§3.2 Estimation Model、§3.3 Data Association、§3.4 Creation and Deletion of Track Identities
- SORT 官方實作：https://github.com/abewley/sort ，commit 2236dff5019565958b84df7d871d41cc1db58ac7，sort.py（linear_assignment L36–44、associate_detections_to_trackers L154–196、Sort.update L210–253）
- DeepSORT：Wojke、Bewley、Paulus，〈Simple Online and Realtime Tracking with a Deep Association Metric〉，arXiv 1703.07402v1（TeX 原始檔），摘要、§1 Introduction、§2.1 Track Handling and State Estimation、§2.2 Assignment Problem、§2.3 Matching Cascade、§2.4 Deep Appearance Descriptor
- DeepSORT 官方實作：https://github.com/nwojke/deep_sort ，commit f08cf1dc470eeb1cd2add1cbf077d95ac6c48aab，deep_sort/tracker.py（_match L93–131）、deep_sort/linear_assignment.py（min_cost_matching L11–75、matching_cascade L78–141）、deep_sort/kalman_filter.py（運動與量測雜訊設定）
- ByteTrack：Zhang 等，〈ByteTrack: Multi-Object Tracking by Associating Every Detection Box〉，arXiv 2110.06864v3（TeX 原始檔），摘要、§1 Introduction、§2.2 Data Association、§3 BYTE 與 Algorithm 1
- ByteTrack 官方實作：https://github.com/FoundationVision/ByteTrack （原 ifzhang/ByteTrack），commit d1bf0191adff59bc8fcfeaa0b33d3d1642552a99，yolox/tracker/byte_tracker.py（BYTETracker.update L159–289）、yolox/tracker/matching.py（linear_assignment L39–50）、tools/track.py（預設門檻 L105–107）
- IDF1：Ristani 等，〈Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking〉，arXiv 1609.01775v2（TeX 原始檔 ristani2016MTMC.tex），摘要、§3.4 Identification Precision, Identification Recall, and F1 Score
- HOTA：Luiten 等，〈HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking〉，arXiv 2009.07736v2（TeX 原始檔），摘要、§1 Introduction、§4 Overview of Previous Metrics（IDF1 一節）、§9 開頭與 §9.2 Problems with IDF1
- 匈牙利演算法：Kuhn，〈The Hungarian method for the assignment problem〉，Naval Research Logistics Quarterly 2(1–2):83–97，1955，DOI 10.1002/nav.3800020109（摘要，經 Crossref 取得）
- ID switch 計數：https://github.com/JonathonLuiten/TrackEval ，commit 12c8791b303e0a0b50f753af204249e622d0281a，trackeval/metrics/clear.py（IDSW 計算 L61–105）
- Python 標準函式庫 runpy.run_path 的 docstring 與原始碼（.venv-model 的 Python 3.12.15）：沒給 run_name 時，__name__ 設為 '<run_path>'；並對照 lesson_cases/18-video.py:154、lesson_cases/19-tracking.py:164 的 if __name__ == '__main__': 守衛

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | 「配對要一對一，也要允許不配」小節最後一段（第 93 行）：「正式系統也會替「不配」訂一個代價，讓演算法在 IoU 太低時寧可不配。」 | 這句把「把不配的代價放進配對問題」寫成正式系統的共同做法，但同頁第 139 行拿來當代表的 SORT 不是這樣做。SORT 論文（arXiv 1602.00763v2 §3.3）原文："The assignment is solved optimally using the Hungarian algorithm. Additionally, a minimum IOU is imposed to reject assignments where the detection to target overlap is less than IOU_min."，也就是先在完整 IoU 矩陣上解指派，配完才剔除低於門檻的配對。官方 sort.py（commit 2236dff）的 associate_detections_to_trackers 也是先呼叫 linear_assignment(-iou_matrix)（L170），再在 L183–190「filter out matched with low IOU」。真正把「不配」放進最佳化的是 ByteTrack 官方 matching.py L43 的 lap.lapjv(..., extend_cost=True, cost_limit=thresh)；DeepSORT 官方 linear_assignment.py 在 L57 把超過門檻的代價一律截成 max_distance+1e-5，再於 L70–72 剔除，效果相近。兩種做法結果可能不同：IoU（列是偵測框、欄是 track）為 [[0.50, 0.29], [0.45, 0.00]]、門檻 0.3 時，照抄 SORT 的函式來跑（求解器以窮舉代替 lap/scipy），track 0 接到 IoU 0.45 的框 1，和它 IoU 0.50 的框 0 反而沒配到，會被開成新 track；改用「不配代價」的寫法或 DeepSORT 的截斷寫法，結果都是 track 0 接框 0。 | 已修正：改成兩種做法並列：SORT 先用匈牙利演算法配完，再剔除 IoU 低於門檻的配對；ByteTrack 的官方程式把「不配」的代價放進配對問題本身；並說明兩種做法的結果不一定相同。 |
| 2 | 建議 | 摺疊區「外觀特徵、Kalman filter 與常見追蹤器」（第 139 行）：「DeepSORT 是 SORT 再加上外觀特徵」 | DeepSORT 摘要確實寫 "we integrate appearance information to improve the performance of SORT"，但它並不是在 SORT 原本的 IoU＋匈牙利配對上多加一項外觀分數。§1 原文："We overcome this issue by replacing the association metric with a more informed metric that combines motion and appearance information"。§2.2 的配對代價是 λ×（相對 Kalman 預測的 Mahalanobis 距離）＋(1−λ)×（外觀 cosine 距離），並說相機移動明顯時取 λ=0 是合理選擇，這時 "only appearance information are used in the association cost term. However, the Mahalanobis gate is still used"。§2.3 另外加了 matching cascade：依幾幀沒配到分批配對，最近剛配到的 track 優先。SORT 式的 IoU 配對只留在最後一輪，處理 unconfirmed track 和 age 1（上一幀還配到、這一幀外觀沒配上）的 track。官方 deep_sort/tracker.py 的 _match（commit f08cf1d，L93–131）也是這樣。照現在的寫法，讀者會以為 DeepSORT 就是原封不動的 SORT 再加一個外觀分數。 | 已修正：改成 DeepSORT 沿用 Kalman filter 與匈牙利演算法，配對主要看外觀特徵的距離（再用 Kalman 預測的位置排除不可能的配對），最近才配到的 track 先配，IoU 配對只留在最後一輪。 |
| 3 | 建議 | 同一摺疊區同一句（第 139 行）：「ByteTrack 連低分的偵測框也拿來配對」 | 這句少了 ByteTrack 的關鍵限定：低分框只在第二輪才配，而且只配給第一輪沒配到的 track。少了這個限定，讀起來就像「把分數門檻調低」，而論文 §1 正好拿這種做法當反例："if we take every detection box into consideration, more false positives will be introduced immediately"。§3 與 Algorithm 1 的做法：第一輪用高分框配所有 track（含 lost tracks）；第二輪只用 IoU，把低分框配給剩下的 track；"just delete all the unmatched low score detection boxes, since we view them as background"；新 track 只從沒配到的高分框建立。官方 yolox/tracker/byte_tracker.py（commit d1bf019）也一致：低分框的範圍是 0.1 < score < track_thresh（L177–181；分數 ≤0.1 的照樣丟掉，所以摘要寫 "almost every detection box"），第二輪只取 Tracked 狀態的 track（L230），開新 track 要求 score ≥ track_thresh+0.1（L154、L266）。 | 已修正：改成 ByteTrack 分兩輪配對：先拿高分框配所有 track，第一輪沒配到的 track 再只用 IoU 去配低分框；沒配上的低分框當成背景丟掉，也不拿來開新 track。 |
| 4 | 建議 | 同一摺疊區最後一段（第 141 行）：「多物件追蹤的正式評估，常用 IDF1、HOTA 這類綜合分數，同時看偵測準不準、身份有沒有維持住。」 | 這句對 HOTA 成立（HOTA 摘要："explicitly balances the effect of performing accurate detection, association and localization into a single unified metric"），對 IDF1 不準確。IDF1 原論文把它定位為身份指標：摘要寫 "emphasizes correct identification over sources of error"；§3.4 寫 "IDF1 is the ratio of correctly identified detections over the average number of ground-truth and computed detections"，而且計分前先把整段序列的軌跡做一對一的身份對應。HOTA 論文 §1 直接寫 "MOTA and IDF1 overemphasize detection and association respectively"；§9 開頭說 IDF1 "ends up being biased towards association"；§9.2 指出它 "produces counter-intuitive and non-monotonic results for measuring detection"。把兩者並列為「同時看偵測與身份的綜合分數」，和這兩篇原文的定位不符。 | 已修正：改成 IDF1 先把軌跡和真實身份一對一對應再計分，著重身份有沒有維持住；HOTA 把偵測準不準和關聯對不對分開量，再合成一個分數。 |

### 第 3 輪：上一輪的處理與審查紀錄：通過

全文讀過紀錄。第 93 行與摺疊區逐句對照：SORT（arXiv 1602.00763 §3.3「solved optimally using the Hungarian algorithm…a minimum IOU is imposed to reject assignments」；abewley/sort 2236dff sort.py 先 linear_assignment(-iou_matrix)，再剔除低 IoU）；DeepSORT（arXiv 1703.07402 §2.2 λ=0 加 Mahalanobis gate、§2.3 年齡小的先配、最後一輪對 unconfirmed 與 age 1 的 track 做 IoU；nwojke/deep_sort f08cf1d tracker.py _match 一致）；ByteTrack（arXiv 2110.06864 §3 與 Algorithm 1：高分框配全部 track，剩下的 track 只用 IoU 配低分框，沒配上的低分框當背景刪除，新 track 只從剩下的高分框建立；FoundationVision/ByteTrack d1bf019 matching.py 用 lap.lapjv(extend_cost=True, cost_limit=thresh)）；IDF1（Ristani 等 arXiv 1609.01775 §3.3–3.4 全序列一對一 truth-to-result match）；HOTA（Luiten 等 arXiv 2009.07736：DetA 與 AssA 的幾何平均）。內容都屬實，解決了來源對照的 4 項發現，合寫作規則；網站建置通過。紀錄確有〈來源對照〉並列出版本。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | reviews/19-tracking.md〈來源對照〉的位置，以及〈定稿修正〉表後的批次檢查段 | 依〈上述處理〉第 1 項的處理，加上工作流程紀錄的時間（來源對照 09:34，晚於 09:14 的上一輪檢查），來源對照是在上一輪檢查之後才做的，卻排在〈定稿修正〉之前。緊接表後的「修正後由另一位 AI 檢查這一批頁面（17–20）的改動，第 1 次：通過」，讀起來像也檢查了來源對照那 4 項修改（第 93、139、141 行），其實沒有。目前紀錄裡沒有任何一輪核對這 4 項，只有這次最終檢查。 | 已修正：〈補做的來源對照〉移到〈後續編輯的檢查〉，排在上一輪檢查之後；這 4 項修改由這一輪檢查核對。 |
| 2 | 建議 | docs/lessons/19-tracking.md 第 139 行（摺疊區〈外觀特徵、Kalman filter 與常見追蹤器〉第一段） | 一句話介紹 SORT、DeepSORT、ByteTrack 三種追蹤器，約 337 字、2 個分號、11 個逗號。DeepSORT 那段還有括號與多層修飾（「IoU 配對只留在最後一輪，給剛建立、還在試用期的 track，以及上一幀還配到、這一幀外觀沒配上的 track」）。內容正確，但對初學讀者負擔很大。 | 已修正：拆成三個子項目，每種追蹤器一項。 |
| 3 | 建議 | reviews/19-tracking.md 第 45 行 | 路徑殘句：「日誌和 PNG 在暫存副本（暫存副本-*.log／.out、暫存副本）。」 | 未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 4 輪：上一輪的處理：通過

摺疊區的 SORT／DeepSORT／ByteTrack 三個子項目，逐項對照三份固定 commit 的官方程式。SORT（abewley/sort 2236dff sort.py）：Kalman 預測，再用 linear_assignment 依 IoU 配對。DeepSORT（nwojke/deep_sort f08cf1d tracker.py _match、linear_assignment.matching_cascade、track.py）：依 time_since_update==1+level 分層配對；confirmed track 用外觀距離加 Kalman gate；IoU 只用在 unconfirmed 與 time_since_update==1 的 track。ByteTrack（FoundationVision/ByteTrack d1bf019 byte_tracker.py update）：高分框配 tracked 加 lost 的 track；第二輪只用 IoU 配低分框；未配上的低分框丟掉；新 track 只從高分框建立。內容屬實，拆成三項回應了第 3 輪第 2 項，合寫作規則。B：第 3 輪三項處理都屬實：〈補做的來源對照〉排在第 2 輪之後；第 3 輪逐句核對了那 4 項修改；第 45 行的殘句已刪。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | docs/lessons/19-tracking.md 摺疊區〈外觀特徵、Kalman filter 與常見追蹤器〉的 DeepSORT 子項 | 原發現點名的括號與多層修飾句（「IoU 配對只留在最後一輪，給剛建立、還在試用期的 track，以及上一幀還配到、這一幀外觀沒配上的 track」）原樣保留。「讓最近才配到的 track 先配」的「才」可能被讀成剛開始配到的 track；實際做法是距離上次配到越近的越先配（time_since_update 小的先配）。 | 已修正：改成「上次配到的時間越近的 track 越先配」，最後一句拆成「處理兩種 track：……」。 |
| 2 | 建議 | reviews/19-tracking.md 第 11、73、77 行 | 第 11 行「所有在暫存副本，都和頁面說的一致」是殘句；第 77 行末尾只剩「重現腳本。」；第 73 行處理寫「見〈來源對照〉」，但段名其實是〈補做的來源對照〉。 | 已處理：第 3 輪的處理說明改成統一的說明。未逐句改寫：紀錄保留查核者的原文，只由產生器統一替換路徑與內部名稱；點名的文字可能因此改變，也可能仍在。 |

### 第 5 輪：上一輪的處理：通過

查證摺疊區 DeepSORT 子項（第 142 行），對照 nwojke/deep_sort commit f08cf1d：
- tracker.py 的 _match（第 93–131 行）：confirmed track 走 matching_cascade，代價是 gated_metric（外觀距離，再由 gate_cost_matrix 用 Kalman 的 Mahalanobis gate 排除）；IoU 候選是 unconfirmed track，加上 time_since_update==1 的未配 track，由 min_cost_matching 在最後一輪處理。
- linear_assignment.py：用 scipy linear_sum_assignment；matching_cascade 第 124–131 行依 time_since_update==1+level，由小到大分層配對。
- track.py：新 track 是 Tentative，命中 n_init 次才轉 Confirmed，Tentative 漏一次就刪；update() 會把 time_since_update 歸零。
所以「上次配到的時間越近的 track 越先配」和「處理兩種 track：剛建立、還在試用期的，以及上一幀還配到、這一幀外觀沒配上的」都屬實。這樣改消除了第 4 輪第 1 項指出的「才」歧義，也拆開了多層修飾句，讀來通順，符合寫作規則。副本建置、validate_site 與摘錄比對（[]）都通過。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 建議 | 摺疊區 DeepSORT 子項（第 142 行）；reviews/19-tracking.md 第 4 輪第 1 項處理欄 | 第 4 輪第 1 項同時點名「括號與多層修飾句」。多層修飾句已拆開，但括號「（再用 Kalman 預測的位置排除不可能的配對）」原樣保留，處理欄只寫了兩處改動，沒交代括號為什麼保留。 | 未改：括號是補充說明，留在句中比拆成獨立子句短，讀起來沒有歧義；第 4 輪的兩處改動照原樣保留。 |

### 第 5 輪：上一輪的處理：有必要問題

第 3 輪第 1、3 項屬實：〈補做的來源對照〉排在第 2 輪之後，第 45 行的殘句已經不在。第 4 輪第 2 項不實。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | 第 4 輪第 2 項處理欄（第 117 行） | 處理欄寫「已修正：點名的文字已不在紀錄裡」，其實只有第 11 行那句已改寫。另外兩處還在：第 73 行（第 2 輪處理欄）仍寫「見〈來源對照〉」，段名其實是〈補做的來源對照〉；第 77 行段末仍有孤立的「重現腳本。」。這兩處都在〈後續編輯的檢查〉裡，產生器沒有比對到。 | 已處理：用詞類的處理說明改成統一的說明（紀錄保留查核者的原文，只統一替換路徑與內部名稱），不再逐句計數。 |

### 第 6 輪：上一輪的處理：有必要問題

第二個〈第 5 輪：上一輪的處理：有必要問題〉的必要項，處理欄顯示的是別一項的處理，成因是產生器的處理欄錯位。

| # | 嚴重度 | 位置 | 發現 | 處理 |
|---|---|---|---|---|
| 1 | 必要 | 第二個〈第 5 輪：上一輪的處理：有必要問題〉表格 #1（第 139 行）的處理欄 | 這項必要發現講的是第 4 輪第 2 項處理欄「點名的文字已不在紀錄裡」不實（第 73、77 行的文字仍在），處理欄顯示的卻是同一輪頁面那一節的處理「未改：括號是補充說明…；第 4 輪的兩處改動照原樣保留。」，和這項無關。成因和 08-own-images 相同：delta-handling.json 的 delta:fifth 對這一頁有 2 筆處理，紀錄那一節又從第 0 筆開始取。讀者會以為這項必要問題被以「未改」駁回了。 | 已修正：產生器依各節的發現順序對應處理，不再錯開。 |

## 紀錄重產後的檢查

2026-10-05 由另一位 AI 在獨立副本 `/tmp/lessons-v0.4.0-reviews/review-19/` 完整查核。結論：通過，**必要問題 0 項、建議問題 0 項**。全文（含摺疊區、練習、真實模型接線與頁尾輸出）、既有 `reviews/19-tracking.md`、主程式、顯示的 SVG、notebook 和兩份現行 JSON 都實際讀過；舊審查只用來定位待核事項，沒有沿用其結論。沒有修改作者的頁面、程式、圖、notebook、紀錄或 coverage。

### 執行與資料核對

模型命令皆以副本為 cwd、`PYTHONPATH=.`，使用 `/workspace/learn_to_yolo/.venv-model/bin/python`（Python 3.12.14、PyTorch 2.9.1+cpu、NumPy 2.3.5、Pillow 12.0.0、opencv-python-headless 4.13.0.92），2 個 PyTorch 執行緒。第 19 課沒有 import repo 模組；真實預測接線的第 18 課與影片驗證則逐一讀了 `miniyolo/__init__.py`、`data.py`、`geometry.py`、`inference.py`、`losses.py`、`metrics.py`、`models.py`、`targets.py`、`provenance.py`，確認訓練、letterbox 還原、class 0 篩選與 score 排序的資料流。

| 實際檢查 | 結果 |
| --- | --- |
| `lesson_cases/19-tracking.py` 預設執行 | exit 0；兩串 IDs、switch 3／0、TP=11、GT=12、FP=0 皆與正文、現行 JSON、notebook 輸出相同，所有內建 assert 通過。 |
| 正文練習，僅將 `def main(max_age=2):` 改成 `def main(max_age=1):` | 用該一行改動後的完整程式實際執行。速度法最後 IDs `[1,3]`、switch 1；上一框法仍 switch 3；recall 11/12 不變。練習圖的兩列標題、`title`／`desc` 和最後 B 的紫色 ID3 都跟著更新。 |
| 額外邊界 | `main(max_age=3)` 確實在第一個 assert 拒絕。空偵測框 `[0,4]` 回傳 `[]`，track 在間隔 1、2 保留，間隔 3 刪除；`[0,2]`、`[2,0]` IoU 矩陣的 matching 均回傳空清單。 |
| 正文「接上真正的逐幀預測」Python 區塊 | 直接從 Markdown 擷取第二個 Python 區塊原樣執行，模型只訓練一次；12 幀 IDs 為 `[[1],[1],[],[],[],[],[2],[2],[],[],[],[]]`。 |
| `scripts/verify_video_file.py` 真實 FFV1 AVI 管線 | 實際編碼、OpenCV 讀檔、訓練、推論及 tracker，exit 0。12 幀框數 `[1,1,0,0,0,0,1,1,0,0,0,0]` 與 IDs 均重現現行 `video-file.json`；RGB、預測 boxes／scores／labels、overlay 的逐項相等檢查，以及 EOF、提早 close、開檔失敗時的 capture release 全通過。產生的 AVI 9,232 bytes、SHA-256 `1c124d4aae1e30c4eed7f6c2e5cb98613ad1e059bce7f657c78be00c83fd64dc`，與原紀錄相同。 |
| 動態排除 GT 灌入 | 另實跑一次影片驗證，將 tracking 模組的 `truth_boxes`、`frames`、`evaluate_detections`、`run` 全換成一被呼叫就拋錯的函式，仍通過。觀察到 12 次 `Tracker.update`，輸入逐項等於 class 0 的真實 `prediction['boxes']`；沒有 GT helper 被呼叫。 |
| 現行影片預測重播 | 使用原 JSON 的 `tracking.original_pixel_boxes` 逐幀送入未改的 tracker，所得 IDs 與原 JSON 完全相同。第 1 幀最後配到的 ID1 在第 2、3 幀仍保留，第 4 幀刪除；第 6 幀建立 ID2，符合正文。 |
| notebook／摘錄 | 最後一格 source 逐字等於 `lesson_cases/19-tracking.py`，保存的 stdout 逐字等於現行課程 JSON。`source_ref` 與 Colab 按鈕都是 `lessons-v0.4.0`；環境格有 `os.chdir(repository)`，可選影片格與實測腳本相符。使用 repo 的 `excerpt_problems()` 查本頁，結果 `[]`；示意區塊有明說變數名不同，接線區塊已實跑。 |
| 依賴測試與建置 | `pytest tests/test_core.py -q`：23 passed；固定文件環境的 `zensical build --clean --strict`：exit 0，無警告。 |

影片腳本原本會用 `git rev-parse` 取得額外 metadata；為遵守此次「不跑 git」的限制，兩次影片驗證都只攔截這一個 subprocess 呼叫，讓 `code_commit=None`，其他程式、I/O、模型與斷言照常執行。原始腳本沒有改，依賴 hash 也沒有改。重新執行的計時、日期留在副本，未覆蓋作者紀錄，也未將計時相等當成重現要求。

本機顯示練習另用既有系統 Python 的 IPython 9.16.1 執行頁面的 `SVG(filename=...)` 顯示命令，實際 formatter 回傳 `image/svg+xml`；模型環境沒有 IPython，沒有為此改動共享環境。

### 逐幀手算與 tracker 狀態

用獨立的純量交集／聯集公式及 `Fraction` 算每一對框，再與程式的 tensor IoU 和逐幀 state 比對。以下列依目前 track 清單的順序，欄依 A、B 的偵測順序；第 4 幀只有 A，`∅` 表示尚無 track。

| f | 上一框法 IoU | 上一框法 IDs | 速度預測法 IoU | 速度預測法 IDs |
| --- | --- | --- | --- | --- |
| 0 | ∅（0×2） | `[1,2]` | ∅（0×2） | `[1,2]` |
| 1 | `[[1/5,0],[0,1/5]]` | `[1,2]` | `[[1/5,0],[0,1/5]]` | `[1,2]` |
| 2 | `[[1/5,1],[1,1/5]]` | `[2,1]` | `[[1,1/5],[1/5,1]]` | `[1,2]` |
| 3 | `[[0,1/5],[1/5,0]]` | `[2,1]` | `[[1,0],[0,1]]` | `[1,2]` |
| 4 | `[[0],[1/5]]` | `[2]` | `[[1],[0]]` | `[1]` |
| 5 | `[[0,0],[1/5,0]]` | `[2,3]` | `[[1,0],[0,1]]` | `[1,2]` |

每次 update 後也核對所有 track 的 `id`、四個 box 座標、四個 velocity 座標和 `last_frame`，不是只看回傳 IDs。例如上一框法在第 2 幀交叉後，ID1／ID2 的 x1 仍分別是 24／32，估得速度都變 0；第 3 幀再估為 −8／+8。第 5 幀舊 ID1 仍在，狀態為 x1=16、velocity=−8、last_frame=3，卻和 B 的 x1=0 沒重疊，因此新建 ID3；這次換號確實不是舊 track 過期。速度法第 4 幀後 ID2 保留 x1=16、velocity=−8、last_frame=3，第 5 幀用間隔 2 預測到 x1=0，接回原 ID2。

重算相鄰框的交集 4×12=48、聯集 144+144−48=240，IoU=1/5。第 2 幀上一框法交叉總和 2，大於正確身份總和 2/5。逐個真實身份計數且跳過漏檢幀，上一框法 A 換 1 次、B 換 2 次，合計 3；速度法 0。11 個人工偵測各自與真值框 IoU=1、跨身份 IoU 最多 1/5，所以不論逐筆順序，IoU≥0.5 評估都得 TP=11、FP=0、recall=11/12；人工案例的 GT 只用來生成可控框與評分，tracker 的 API 只收 boxes、frame。

貪心反例由真框獨立重算為 `[[9/11,2/3],[7/13,1/4]]`，直配總和 47/44≈1.0682，交叉總和 47/39≈1.2051。配法數以 `Σ C(n,k)^2·k!` 重算，2 對 2 是 7，10 對 10 是 234,662,231，符合約 2.3 億。正文假想連漏兩幀的例子也核對：max_age=3 才能留到第 6 幀；速度×3 得 x1=−8，只加一次得 8，差 16 畫素、IoU=0。沒有把這個假想例子冒稱成主程式原本有的情境。

### 圖與網頁實際渲染

預設程式重畫 `artifacts/lesson-19/ids.svg`，與展示用 `docs/assets/diagrams/19-tracking.svg` **逐 byte 相同**。另逐項解析兩張圖的 22 個物件矩形：水平座標均等於該幀 panel 起點加 `6+1.9*x1`，顏色均依 ID1 紅、ID2 藍、ID3 紫，A／B 分排的 y 位置與正文說明一致。沒有拿圖上的分排位置當成真正配對用的 y=20–32。

在副本建置網站後自行開 HTTP server 8819，使用 `/usr/bin/chromium` 與指定 `--no-sandbox --disable-gpu --disable-dev-shm-usage`，實際查看本頁、主 SVG 和練習 SVG。頁面及圖皆 HTTP 200、圖成功載入，兩張表、四個摺疊區、程式區塊轉換正常，沒有 pageerror。已實際看 `default-svg.png`、`exercise-svg.png`、`page-figure-context.png`：六格框內標籤無重疊或截字；第 2 幀上列互換顏色、第 4 幀漏檢、第 5 幀 B 紫色，以及練習速度法末幀 ID3，都與結果相同；`viewBox`、`title`、`desc` 也核對。伺服器已停止，未碰 root 的 8794。

### 重新開啟的原始來源

來源皆重新從下列 URL 取得並讀相關段落，不使用舊審查摘錄代替原文。此次來源請求全成功，沒有未交代的 HTML 讀取失敗；檔案與 SHA-256 保存在副本 `artifacts/runs/review19/sources/fetch-log.json`。Python 官方 `/3.12/` HTML 現在標示 3.12.15，因此涉及此次 3.12.14 執行環境的細節另以固定 `v3.12.14` 官方原始碼／文件來源核對。

- SORT：[arXiv 1602.00763v2](https://arxiv.org/pdf/1602.00763v2)，§3.2–3.4；[官方 sort.py，2236dff5019565958b84df7d871d41cc1db58ac7](https://raw.githubusercontent.com/abewley/sort/2236dff5019565958b84df7d871d41cc1db58ac7/sort.py)。Kalman 等速預測、IoU 一對一指派、一般情況先解指派再剔除低 IoU；官方另有唯一可配情形的捷徑。頁面沒有把本節列舉叫成 Hungarian，也沒有把本節 max_age 的配對前刪除規則叫成 SORT 的規則。
- DeepSORT：[arXiv 1703.07402v1](https://arxiv.org/pdf/1703.07402v1)，§2.1–2.4；[官方 tracker.py，f08cf1dc470eeb1cd2add1cbf077d95ac6c48aab](https://raw.githubusercontent.com/nwojke/deep_sort/f08cf1dc470eeb1cd2add1cbf077d95ac6c48aab/deep_sort/tracker.py)，同 commit 的 `linear_assignment.py`、`kalman_filter.py`、`track.py`。外觀距離加 Mahalanobis gate、time_since_update 小的先配，以及最後 IoU 輪的 unconfirmed 和未配到 age=1 track，均符合正文；Kalman 不確定性及運動／量測雜訊的說明也成立。
- ByteTrack：[arXiv 2110.06864v3](https://arxiv.org/pdf/2110.06864v3)，§3、Algorithm 1；[官方 byte_tracker.py，d1bf0191adff59bc8fcfeaa0b33d3d1642552a99](https://raw.githubusercontent.com/FoundationVision/ByteTrack/d1bf0191adff59bc8fcfeaa0b33d3d1642552a99/yolox/tracker/byte_tracker.py)，同 commit 的 `matching.py`。正文的兩輪核心對應原論文；官方首輪 tracked＋lost，次輪只取未配的 Tracked，另處理 unconfirmed；低分框不建新 track，且 `lapjv(..., extend_cost=True, cost_limit=thresh)` 把不配放入指派問題。正文沒有宣稱此簡化 tracker 完整實作了這些處理。
- IDF1：[arXiv 1609.01775v2](https://arxiv.org/pdf/1609.01775v2)，§3.3–3.4。全序列一對一 truth-to-result matching 後計 ID precision／recall／F1，著重身份，符合正文。
- HOTA：[arXiv 2009.07736v2](https://arxiv.org/pdf/2009.07736v2)，摘要、§1、§5 的式 18–21 與 §9.2。偵測 DetA、關聯 AssA 的幾何平均及跨定位門檻平均，支持正文的簡短介紹；本節實際沒有計 HOTA。
- ID switch：[TrackEval CLEAR，12c8791b303e0a0b50f753af204249e622d0281a](https://raw.githubusercontent.com/JonathonLuiten/TrackEval/12c8791b303e0a0b50f753af204249e622d0281a/trackeval/metrics/clear.py)，保留最近一次配到的 tracker ID 跨漏檢比較，符合本例簡化計數定義；沒有以本例直接相連的 A／B 當成完整 CLEAR 評估實作。
- 匈牙利演算法：[Kuhn 1955，DOI 10.1002/nav.3800020109 的 Crossref 原摘要](https://api.crossref.org/works/10.1002/nav.3800020109)，確認 assignment 是一對一配置、最大化總分；只查摘要，沒有宣稱讀完整付費論文。
- Python：[固定 CPython v3.12.14 runpy.py](https://raw.githubusercontent.com/python/cpython/v3.12.14/Lib/runpy.py)、[固定 functions.rst 的 zip](https://raw.githubusercontent.com/python/cpython/v3.12.14/Doc/library/functions.rst)。`run_path` 未指定 run_name 時用 `<run_path>`，兩課 main 守衛不會執行；`zip` 預設在最短 iterable 結束，與正文篩選後 boxes 的警告相符。
- IPython：[8.18.1 官方 display.py](https://raw.githubusercontent.com/ipython/ipython/8.18.1/IPython/core/display.py) 用來重新核對舊審查所用版本；[此次實測 9.16.1 官方 display.py](https://raw.githubusercontent.com/ipython/ipython/9.16.1/IPython/core/display.py) 的 SVG 與 `_repr_svg_` 行為相同，且 MIME 已實際測到。

### 發現與處理

此次新增必要／建議發現皆為 0，沒有待修項目。既有 review 的待重錄提醒與修正說明重新核對如下。

| 核對項目 | 此次處理／結果 |
| --- | --- |
| 舊紀錄提到展示圖仍是舊英文圖、待重產複製 | 現在展示圖已是中文重畫圖，與預設程式重畫逐 byte 相同；該歷史提醒已由實際重產解決。 |
| 舊紀錄要求重錄後確認影片第 6、7 幀 `[2]` | 現行 JSON 有這兩幀，正文與 JSON 相同；此次兩次實際影片驗證及原 boxes 重播也都得到 `[2]`。 |
| 舊來源修正：SORT 不配處理、DeepSORT cascade、ByteTrack 低分限制、IDF1／HOTA 定位 | 固定版本原文／官方程式逐一重查，目前正文均已處理，無新增矛盾。 |
| 舊 review 最後一輪指出必要項處理欄錯位 | 目前第二個「第 5 輪」必要項的處理欄已改成用詞處理說明，第 6 輪處理寫明產生器按發現順序對應，沒有再顯示無關的「括號補充」理由；歷史殘句沒有被冒稱逐句清除。 |
| 初學讀者理解 | frame、track／ID、association、GT、IoU、索引與 ID 的差別、max_age 的「間隔」語意、圖的分排 y 與真實 y、人工框與真實模型輸出都有先交代；練習只要求一行改動且參考答案解釋原因。未發現會阻礙無專案背景讀者照做的必要問題。 |

### 覆核快照與限制

覆核時的原始檔案 SHA-256：

| 檔案 | SHA-256 |
| --- | --- |
| `docs/lessons/19-tracking.md`（含頁尾） | `f19a6ed8a16b9217fb2eafa2b7e22d067a2523c4aaccaca110dabc8b1dc7f1d7` |
| 本頁正文（依 repo coverage 規則排除頁尾、正規化 Colab tag） | `6ef6eee6fbce7df290afb1b688a3caa627848489022a32ac240ddf6874fe054a` |
| `docs/assets/diagrams/19-tracking.svg` | `b49961cef2c248e1b163e652d77c89ff9e7670c5ce0a5f164ffeae36740bafe3` |
| `lesson_cases/19-tracking.py` | `e1fc77ce9e6e8e09a34f4dedf8543155154ab58709eaf1ab61756793f7a14efa` |
| `notebooks/19-tracking.ipynb` | `93ae820348223376140b3523d8ee9c4ff47aa314943de41b2ad7e55299ad3d91` |
| `reviews/19-tracking.md`（附加本節前） | `89272d833e7af3b1407a0f0e79b97b52bb74750195e55f7397573e2b498a2e2c` |
| `artifacts/checks/curriculum/19-tracking.json` | `9c741b58ab6511e8a09fe69ddaa33d2d588d4bc164595065645f585f0831e6fc` |
| `artifacts/checks/curriculum/video-file.json` | `e03980d47e7e4f87c0c4243f63754ad56a905cfacc4cf2ebc30bfd07b1394d58` |
| `lesson_cases/18-video.py` | `0aa6c1713098e244dd3651c6f319844ad0e17b10ee28a187e8a224ee6e9024a0` |
| `scripts/verify_video_file.py` | `ec77505526972a57b74cf757e303ccfea7459dfba2ffd9857b7634129070dcec` |

兩份作者紀錄的全部 `dependencies_sha256` 已逐檔比對現行副本；課程紀錄 1 檔、影片紀錄 12 檔均相符。查核輸出在副本 `artifacts/runs/review19/`：`probe.stdout`、`manual-trace.json`、`exercise.stdout`、`exercise-ids.svg`、`video-file/result.json`、`no-gt-proof.json`、`no-gt-video-file/result.json`、`browser.json`、PNG、來源下載清單、hash 清單與建置／測試 log。

限制：這是 AI 查核，沒有真人初學者測試；沒啟動 GPU、遠端 workflow、資料下載或 Colab 託管 runtime，沒有實體相機、有損 MP4、可變幀率 PTS 或完整 MOT 指標驗證。notebook 的 clone／套件安裝環境格只讀碼核對，沒有在本次執行；最後一格以相同完整程式在固定 CPU 環境實跑。全站 MathJax／CSS 另有專人檢查，本次只檢查本頁轉換與實際圖像。影片的 tracking IDs 是真實偵測輸入的接線結果，沒有 GT 身份評分，不能報成新增的 ID switch、IDF1 或 HOTA 成績。


## 2026-10-05 clear-tutorial 三輪重審

以上是原審查歷史；不追溯改成首次盲讀。這次由固定基線 `16f6910` 分段開放並保存當時理解，再修改、核技術及檢查銜接，詳見 [本輪方法與限制](clear-tutorial/16f6910/README.md)。

- 第一輪：[applications當場閱讀原始紀錄](clear-tutorial/16f6910/first-read/applications.jsonl)，基線來源與圖指紋保留；共享檔案系統不是技術隔離。
- 第二輪：[非作者技術／證據核對](clear-tutorial/16f6910/technical/modern-applications.md)，實際來源、數字及必要執行範圍見該報告。
- 第三輪：[另一位讀者前文→本節→後文複查](clear-tutorial/16f6910/transitions/modern-applications.md)，此輪完整頁閱讀非盲讀；受影響段落及圖另有delta核回。
- [原始卡點與具體處理](clear-tutorial/16f6910/decisions.json)保留未新增的選讀建議。原先前提包漏發及08提前brief的限制另列，沒有算成教材錯或冒稱08全程盲讀。

本輪修正後沒有未解的必要問題；這是AI閱讀／技術查核的實際範圍，不是學生學習成效驗收。全站實際Zensical桌面／手機、公式換頁與執行檢查見 [verification.json](clear-tutorial/16f6910/verification.json)，不以SVG檔存在或strict build取代視覺查核。

## 2026-10-06：最新版 clear-tutorial 全套重審

本次以 `64a25d4fbcff5577965c29efbbcb5d9898ba95d9` 凍結來源從頭閱讀，不把以前的審閱當作此次首次閱讀。方法、完整範圍與限制見[本輪報告](clear-tutorial/full-review-2026-10-06/README.md)。

- 首次閱讀：主要讀者 `applications` 實讀本頁 7 個凍結單元；首次使用／前文方法範圍四題位置為 19-tracking/00:first_use, 19-tracking/03:first_use, 19-tracking/05:first_use，頁末為 19-tracking/06。[當時理解與問題](clear-tutorial/full-review-2026-10-06/first-read/applications.jsonl)與[分段披露](clear-tutorial/full-review-2026-10-06/first-read/applications-disclosures.jsonl)按原樣保留；實際前置閱讀見[該組報告](clear-tutorial/full-review-2026-10-06/reports/applications.json)。
- 處置：[決策表](clear-tutorial/full-review-2026-10-06/decisions.json)。本頁處置：R016、R020、R022；各項原位置、分級、實際改寫／保留理由見決策表。
- 非作者技術／證據：[本頁所屬報告](clear-tutorial/full-review-2026-10-06/rechecks/technical-applications.json)，只以報告列出的正文、實作、數值、圖與實際執行範圍作結論。
- 另一位讀者的前文→本節→後文與網站：[第三輪紀錄](clear-tutorial/full-review-2026-10-06/rechecks/transitions-visual.json)。52節正文有閱讀紀錄；實看圖／公式的頁面與截圖另列，不將捕捉或DOM載入當成每張圖可讀。

本輪未留下已裁定的必要問題。所有讀者均為 AI，沒有真人學生學習效果驗收。原首讀中仍有漏報、引用未支持全部主張及明說／推論混分，見[獨立裁定](clear-tutorial/full-review-2026-10-06/rechecks/record-adjudication.md)；不能宣稱四題保證抓到所有缺漏或原始紀錄嚴格規則全合格。程式與依賴、正式CPU紀錄、Notebook、建置和全站掃描的實際檢查見[驗證結果](clear-tutorial/full-review-2026-10-06/verification.json)。本頁最新文字、所用SVG／raster圖片與實驗依賴綁定在[coverage.json](coverage.json)。

# ViT／DINO v0.6.0：獨立非盲讀導讀與實際網站視覺審閱

審閱類型：獨立 AI 非盲讀全文對照＋本機 Chromium 實際視覺。此報告不是首次閱讀盲讀、第二輪技術審查或第三輪前後銜接驗收；不取代這三輪的原始紀錄。審閱者未撰寫教材，先讀 repo `clear-tutorial/SKILL.md` 與 `references/review-protocol.md`，未讀作者筆記、其他作者或首次閱讀者的報告。

本輪發現兩項必要圖問題與兩項閱讀負擔，修正後均已核回；指定導讀與本機視覺範圍沒有待修的主要概念阻礙。

基底 commit：`823232bf9eba4074b4db4c99b6b076feaa1a9992`，含工作區未提交內容。審閱日期：2026-10-05 UTC。教材與圖以來源 SHA-256 凍結；來源變動的局部修正會另保存修前、修後範圍，不追認其他頁面通過。

## 範圍與未執行項目

主要導讀範圍：`README.md`、`docs/index.md`、`docs/learning-path.md`、`docs/planning/outline.md`、`docs/status.md`、`docs/preparation/architecture.md`、`docs/research/pages-colab.md`、`docs/validation/curriculum.md` 的本次變更，以及 `15-attention-bridge.md` 新增的選讀入口。為核對導讀承諾，完整非盲讀新增 10 頁的正文及必要圖。

協調者後續指定的局部複查：`docs/preparation/publish.md` 四處 42→52、`docs/preparation/data.md` 一處 42→52、`docs/research/foundation-data.md` 一處 42→52；`00-warmup.md` 環境成功輸出版本；`01-small-cnn.md` 40 步 JSON 欄位對應段及相應四列表格。沒有重新審閱整篇舊 00／01 或這三篇資料／發布頁的其餘內容。

實際本機頁面：10 頁各以 **1280×800 桌面**與 **390×844 手機**開啟；正文圖片共 19 個位置，每種尺寸都截圖並使用 `view_image` 實際檢視。Playwright 使用系統 `python3` 所在的 Python 3.12，`chromium.launch(executable_path='/usr/bin/chromium', args=['--no-sandbox','--disable-dev-shm-usage','--disable-gpu'])`。網站由協調者預先建置與提供本機 server；本輪沒有把建置成功當成視覺驗證。

沒有登入或執行 Google Colab，沒有跑公開 GitHub Pages 驗證，也沒有重跑課堂訓練／官方預訓練實驗。Colab 的已驗範圍只是本機頁面按鈕網址固定到 `lessons-v0.6.0` 及正確 notebook 檔名。CPU 數字核對來自已保存的 JSON，不宣稱本輪重新執行。

## 導讀承諾核對

| 承諾／位置 | 本輪對照與理解 | 結果 |
| --- | --- | --- |
| 52 節＝42 主線＋10 選讀 | `section-map.json` 52 個 lesson ID 與 `lesson_cases/*.py` 的 ID 集合一致；原 0–20 有 42，新 21–23 有 4＋4＋2 | 數量與章節配對一致 |
| CNN／ResNet／15.1 可分岔，主線續讀 15.2 | learning-path E、首頁、README 與 15.1 的選讀框都把 21.1 設為入口；21.1 本身連到 1、3.1、15.1；23.2 的框／IoU前置另列在閱讀路線 | 入口與主線關系清楚 |
| 純閱讀可學，notebook 獨立 | 10 頁正文均顯示材料、答案規則、必要圖、實作範圍及結果；22.4、23.2明說自行重做短 SSL，不依賴上一節 checkpoint；23.1手工主例與官方選讀分開 | 導讀沒有以後續 notebook 補必要素材 |
| ViT 23,970 參數，CPU 60步、固定 train loss 0.710059→0.007155、val/test各64/64、續訓一致 | 核 21-training.json 的 parameters、fixed_train_loss_before/after、heldout、resume；表述仍限定同規則紅／藍矩形 | 數字與限制相符 |
| DINO 核心 CPU160步；random/SSL都64/64，不宣稱分類優勢 | 核22-features.json兩個models的1-NN與linear_probe、正文的訓練／下游標籤區分 | 相符，沒有把高分單獨說成SSL改善 |
| 定位平均IoU0.605699、聯合正確54/64，不是AP | 核23-detection-bridge.json metrics.test；頁面直接按圖配對一個真值／預測，沒有PR排序 | 相符 |
| DINO（2021）與DETR同名DINO detector分開；DINOv3沒有實作 | 導讀及22.3／23.1／23.2均明說，23.1的Gram例只有手工機制 | 沒有混用名稱或實測範圍 |
| 預設不下載資料／權重、官方DINOv2另選 | 23.1區分無下載Gram主例與約84.2MiB固定官方權重操作；architecture/research亦明說另選、沿用torch/NumPy/Pillow | 相符；本輪未重跑官方下載 |
| 新版v0.6與舊v0.5發布證據分開 | README、首頁、路線、status、architecture、research用『本版固定到v0.6；公開驗證以紀錄tag結果為準』；現有兩份公開驗證JSON的source_ref都是v0.5 | 相符；curriculum的G1局部範圍問題已核回 |
| 01-small-cnn局部JSON對應 | initial_loss=0.6941587329、last_pre_update_loss=0、train_accuracy=1.0、validation_accuracy=null；新段對應第1／40點loss、8張訓練圖、未評heldout | 局部已核回，沒有宣稱重審整頁 |
| 00-warmup局部版本 | 環境成功訊息寫v0.6，notebook第二格REF與metadata source_ref同為v0.6 | 局部相符 |

## 原始問題與局部複查

### V1：22.3 更新次序圈號在實際瀏覽器顯成方框

優先序：P1，阻礙圖與正文指定步驟的對應。

位置：`docs/lessons/22-distillation.md` 圖前後說明『按①、②、③、④追蹤更新次序』；`docs/assets/diagrams/22-distillation.svg` 的 backward／teacher EMA／center EMA 標籤。

當時可見：桌面圖中的①正常，但②③④變成空方框。讀者仍能從文字知道各部件，但不能按正文指定的號碼找到圖中②–④，需要自行補猜順序。`view_image`核對的是實際Chromium截圖，不是只看SVG原始碼。

修前證據：`artifacts/runs/vision-browser-review/22-distillation-desktop-clean-test.png`（完整圖，520×1120實際正文寬）；原始 `22-distillation-desktop-figure01.png`／`22-distillation-mobile-figure01.png`亦保留。原始元素截圖可能把sticky header疊入圖頂；clean截圖以scroll top下的document坐標區域截取，沒有改DOM或CSS，排除了截圖方式造成的頂部遮擋。

建議修正：用字體可靠的ASCII步驟號並同步正文指定的對應。

修後實際複查：2026-10-05 12:06 UTC 在重建的本機網站另截桌面／手機，使用 `view_image` 逐張核回。圖中的 ASCII 1、2、3、4 全部可讀，正文同步說明按 1、2、3、4 追蹤；teacher／student、停止梯度與兩種 EMA 路徑仍可沿箭頭閱讀。V1 已處理。修後證據：`22-distillation-desktop-recheck-figure01.png`、`22-distillation-mobile-recheck-figure01.png`；來源與圖片 SHA 見 `guide-visual-raw-recheck.json`。

### V2：22.4 所稱160步完整曲線在下界被裁去一段

優先序：P1，必要的訓練曲線可見不完整。

位置：`docs/lessons/22-features.md`〈曲線和特徵統計，各補充一件事〉；`docs/assets/diagrams/22-features.svg`上半圖。

當時可見：CE曲線約step129–143落到畫圖區下界外而中斷，然後重新出現。正文說上半部是同設定160步曲線，讀者會缺少谷底形狀；不能把缺線解讀成沒有紀錄。SVG中實際path使用clip-path，線段座標越過plot下界，支持此為曲線裁切問題。

修前證據：`artifacts/runs/vision-browser-review/22-features-desktop-figure02.png`／`22-features-mobile-figure02.png`。scope為實際圖內上半曲線；其餘統計與64/64比較仍可見。

建議修正：將y軸範圍包含全部實際loss，並保留可讀刻度。

修後實際複查：同一輪重建後另截桌面／手機，使用 `view_image` 逐張核回。y 軸完整包含實際 loss 範圍，約 step 129–143 的谷底與後續上升都可見；1–160 的整條曲線連續，刻度與下方統計仍可讀。V2 已處理。修後證據：`22-features-desktop-recheck-figure02.png`、`22-features-mobile-recheck-figure02.png`；圖 SHA 與本機 site 一致。

### G1：curriculum公開驗證段的新51/52描述與現存v0.5紀錄範圍容易混讀

優先序：P2，增加版本證據理解負擔，需明確歷史範圍。

位置：`docs/validation/curriculum.md`〈發布後的公開驗證〉。此段稱本版v0.6，敘述51本同環境格與52節runtime，連結的兩份JSON目前source_ref都是v0.5。讀者若只看本頁，會把新版流程數量讀成這兩份舊紀錄的實際範圍。

建議：明標這是新版驗證流程；已保存結果以各自source_ref與passed為準，v0.5不能驗證新增十節。已向協調者回報。修後核回：第143行明標下文是本版驗證流程，連結紀錄看各自source_ref；並明說v0.5只涵蓋原有42節、不驗證新增十節。與目前兩份JSON相符，G1已處理。

### V3：21.3 兩步公式並排造成手機需橫滑

優先序：P2，增加閱讀負擔；正文與圖已提供完整兩步，未阻礙主要概念。

位置：`docs/lessons/21-transformer.md`〈MLP〉的並排U／Y公式。手機的MathJax寬458px，正文寬343px，arithmatex父容器使用overflow:auto。初始畫面只能讀到第二式的`Y=U+ML...`；使用者可以橫滑，沒有造成整頁溢出。

修前證據：`artifacts/runs/vision-browser-review/21-transformer-mobile-math01.png`及`raw-math-inspection.json`。建議以aligned拆成兩行，使兩次修正都能直接閱讀。此項已報協調者。修後兩行 aligned 公式在桌面／手機實際截圖中完整可見；手機公式寬約 227px、正文容器 343px，不需橫滑讀第二式，MathJax 無錯誤。V3 已處理。修後證據：`21-transformer-desktop-recheck-math.png`、`21-transformer-mobile-recheck-math.png`及 `guide-visual-raw-recheck.json`。

## 實際視覺與互動觀察

| 頁面 | figure位置數／桌面與手機觀察 |
| --- | --- |
| 21.1 patches | 3；實際紅藍素材、0/1答案、patch1–16逐列順序、CLS與位置相加都可見，手機不需放大才辨認編號 |
| 21.2 attention | 2；來源patch1/6/7/10送往patch6的新內容、交換內容而位置留槽的交叉線清楚；箭頭含義與正文一致 |
| 21.3 transformer | 2；Pre-LN兩次shortcut及X/U/Y、多head→MLP、完整CLS分類流程有足夠字級，手機仍可沿上下流程讀 |
| 21.4 training | 3；共用實際材料、60步曲線與step30虛線、兩路resume的optimizer/RNG標籤可讀 |
| 22.1 views | 1；原圖crop邊界、兩份global view與不含物件反例實際可見，圖說區分原圖與view座標 |
| 22.2 collapse | 1；保留差異／常數向量、同圖loss都0的兩個區塊與箭頭可讀，圖明標手工例 |
| 22.3 distillation | 1；teacher／student兩路、停止梯度、EMA顏色及raw logits路徑可辨；V1號碼已於修後桌面／手機核回 |
| 22.4 features | 2；test0→SSL train59／random64藍圖、score分母及統計可讀；V2完整曲線已於修後桌面／手機核回 |
| 23.1 versions | 2；Gram A/B/C行列、旋轉／collapse說明，官方兩圖的query與top1綠色／top2橘色patch框均可辨 |
| 23.2 bridge | 2；綠真值／橘虛線預測、test0/1失敗及test2/3達標保留；16patch→4×4grid→凍結backbone／只更新head的流程可讀 |

20次直接開頁均HTTP200，19×2個圖位置均載入完成且naturalWidth>0；頁面document沒有水平溢出。手機長程式與寬表使用各自的水平滾動區域，未把它們誤報為整頁溢出；公式另外用實際截圖確認。詳情框每頁實際點開／關第一個，均成功。

實際點擊21.1正文『下一節：21.2』在兩種viewport都進入21.2，window哨兵仍保留，確認發生instant navigation而非全頁重載。新頁面出現1個MathJax container，0個mjx-merror。實際切換深色再回淺色，公式與正文仍可讀。

兩種viewport均輸入英文`DINO`（UI顯示46results）與中文『矩形』（76results），然後實際點首個結果：DINO進入23.1官方特徵錨點，矩形進入21.2矩形跨格錨點；點擊後URL與結果href相符。已保存搜尋介面和換頁後公式截圖。Google Colab runtime仍未執行。15.1選讀框也在兩種viewport實際展開，文字明說主線接15.2與10節支線，點21.1連結確實到patch頁；curriculum新增表格實際有四欄表頭與10列，沒有沿用先前缺表頭的建置結果。

最初的互動腳本曾因用中文文字尋找隱藏的 accessible 按鈕、只匹配不含anchor的URL、或把含query的skip連結誤選成搜尋結果而超時。調整為實際可見搜尋按鈕、pathname匹配及搜尋結果a.i後重跑；成功結果以raw-interactions.json為據，不把腳本超時紀錄當網站缺陷。

## 證據文件

實際截圖與運行腳本：ignored `artifacts/runs/vision-browser-review/`。Raw browser、互動、數學排版與修後結果已保存到本報告同目錄；完整文章截圖保留檢查範圍，判讀採用正文實際寬度的每張圖、頁首及公式 viewport 截圖。

本輪結束時，V1、V2、V3 的實際視覺與 G1 導讀局部均已核回，本範圍沒有待修的主要概念阻礙。這只支持本文列出的導讀與本機視覺範圍，不擴張到全書重審、三輪閱讀驗收或公開發布通過。上述修前發現與截圖保留。


最終局部文字複查另外核回 22.4〈第一種讀法〉提前定義 SSL 為 160 步後 teacher、random 為同起點未更新 backbone，避免把 random 誤讀成隨機選參考圖；23.2 的原論文引用分清 §3.1 backbone 與 §3.2 評測。這兩處是非盲讀文字複查，沒有另稱技術輪或首次閱讀。

同目錄證據：

- [最終來源指紋與檢查範圍](guide-visual-source-fingerprints.json)
- [初次頁面與圖片原始結果](guide-visual-raw-browser-results.json)
- [搜尋、深淺色與實際點擊換頁](guide-visual-raw-interactions.json)
- [修前公式排版](guide-visual-raw-math-inspection.json)
- [15.1 支線入口及 curriculum 表格](guide-visual-raw-guide-interactions.json)
- [三項視覺修正的桌面／手機複查](guide-visual-raw-recheck.json)

初次 browser JSON 保存當時的來源 SHA；最終來源指紋另外保存最新正文。後加的局部定義與引用文字不會追寫成初次截圖已包含它們。


## 最終導讀兩處局部文句複查

2026-10-05T12:19:44.508446+00:00：僅重新讀取 `docs/status.md`〈誰檢查過內容〉第 96 行附近，以及 `docs/validation/curriculum.md`〈審查〉第 103 行附近。status 移除會隨發布時間過期的「審查尚在進行」，改為本次逐段閱讀、技術與銜接紀錄的目錄連結，並明說下列是既有主線上一輪的範圍；curriculum 明標下列 `16f6910`／59 頁是上一輪歷史範圍，新支線另記下方。兩處均保持舊版與新支線的範圍分開，沒有把舊版紀錄改稱 v0.6 已通過，也沒有額外承諾公開發布通過。局部文句已核回。

來源 SHA 更新如下；其餘 55 份來源與前次結束時相同。未重截圖或重跑瀏覽器，原始 actual visual 結果仍只代表原先記錄的範圍。

| 來源 | 複查後 SHA-256 |
| --- | --- |
| `docs/status.md` | `7215ab00b9cd33cef61fc31455dbb53951af915e94fa8b6a95c92ed4451f946b` |
| `docs/validation/curriculum.md` | `777fb49673f68d6b3d9adc3033de19f84f954f821edd7a4ac802329df1086507` |

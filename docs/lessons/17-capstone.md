# 17 靜態偵測結業：用一次有理由的改動交付結果

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.5.0/notebooks/17-capstone.ipynb){ .md-button }

本節是靜態偵測（單張圖片，不是影片）的結業任務。這次不從「要用哪一版 YOLO」出發，而是從一個具體需求出發：先訓練模型、看它錯在哪裡，再根據失敗只改一個設定。讀完後，你能用事先寫好的規則判斷一次改動該不該保留，並交付真實跑出的數字、失敗的圖，以及這次還不能下結論的地方。

前置：[資料切分與 held-out](07-heldout.md)（validation 用來做決定，test 只在最後評一次）、[grid loss](07-loss.md)（box、objectness、class 三項 loss）、[AP50](06-evaluation.md)（TP、FP 與 AP 怎麼算）、[完整圖片推論](07-inference.md)（模型輸出怎麼變成框）。

需求：辨識 64×64 圖片中的紅、藍矩形，每張圖有零到兩個物件；還要在 CPU 上量出單張圖從輸入到畫好框的時間（本例只要求量得出來、寫清楚量了哪幾步，不設時間上限）。

本節路線：訓練 baseline（基線：不做改動、拿來對照的版本）→ 看它錯在哪裡 → 提出假設、先定保留規則 → 只改一項重新訓練 → 用 validation 決定保不保留 → 只對選定的模型測一次 test。

本例紅矩形是 class 0，藍矩形是 class 1。圖上的 `0:1.00` 表示這個預測是 class 0、score 1.00。

本次的 baseline 是最小的 grid MiniYOLO，也就是第 7 章的 GridDetector：4×4 格，每格只預測一個框，共 16 個候選；box loss 的權重是 5。它從零開始訓練，不用預訓練權重，也不下載資料。選它的理由是：這份資料刻意讓每個物件的中心落在不同格（cell），每格一框的簡單 head（模型最後輸出預測的那一層）就夠用。若真實資料常有兩個物件落在同一格，這個選擇就要重做。

唯一的改動是把 box loss 的權重從 5 改成 10，下文稱為改動版。模型架構、assignment（責任分配：訓練時決定哪一格負責哪個物件）、optimizer、資料和訓練步數都不變，也不混入第 12 章的 DFL 或第 15 章的 attention。

## 先寫下實驗規則（協議），再看結果

協議（protocol）是事先寫好、看到結果後不再更改的實驗規則：用哪些資料、怎麼訓練、怎麼評估、怎樣才保留改動。先寫下來，才不會看了結果再回頭調規則。

| 資料 | 張數 | 資料 seed（亂數種子） | 用途 |
| --- | --- | --- | --- |
| train | 32 | 1100 | 更新模型參數 |
| validation | 16 | 2200 | 診斷失敗、決定保不保留改動 |
| test | 16 | 3300 | 選定模型後只評一次 |

三份資料各用不同的 seed 產生，是不同的圖片。但它們出自同一套隨機畫矩形的規則，所以結果不能代表模型在真實相機照片上的表現。

| 訓練設定 | 值 |
| --- | --- |
| 模型 seed | 7，兩次訓練前都重設 |
| batch | 8 |
| optimizer | Adam，學習率（learning rate）0.01 |
| 訓練步數 | 每次 160 步 |
| 輸入 shape | `[8,3,64,64]` |
| head 輸出 shape | `[8,4,4,7]`：每格 4 個框參數、1 個 objectness、2 個類別 |

兩次訓練前都重設 seed 7，所以兩個模型的初始權重完全相同；每一步取哪 8 張訓練圖，也照固定順序、不洗牌。因此兩次訓練唯一的差別，就是 box 權重。

在這兩次訓練之前，程式還用 baseline 的設定先訓練 10 步當作暖機（warmup，3.3 節〈[Plain／residual 對照](03-comparison.md)〉說明過：正式計時前先不計時地跑幾次），訓練出的模型直接丟掉。暖機只是為了讓訓練時間量得公平（見〈本次 CPU 實跑的結果〉）；負責訓練的函式 `fit()` 每次開始都重設 seed 7，所以暖機不會改變後面兩次訓練的結果：baseline 與改動版的模型，和不暖機時訓練出來的完全相同。

loss 是 `wbox×box_MSE+objectness_BCE+class_CE`，和[第 7 章的 loss](07-loss.md) 相同，只是把固定的 5 換成可調的 wbox（程式裡的 `box_weight`）。box 項只對正格（負責某個物件的格子）的四個座標取平均，objectness 項（這格有沒有物件）對全部格子取平均，class 項只對正格取平均。

wbox 到底改了什麼？總 loss 是三項的加權和，而三項都由同一個網路算出；所以總 loss 對網路參數 θ 的梯度，就是三項梯度的加權和：

\[
\frac{\partial L}{\partial \theta}
=w_{\text{box}}\frac{\partial L_{\text{box}}}{\partial \theta}
+\frac{\partial L_{\text{obj}}}{\partial \theta}
+\frac{\partial L_{\text{cls}}}{\partial \theta}
\]

在同一組參數下，wbox 從 5 改成 10，梯度裡 box 這一項變成兩倍，另外兩項不變。所以改的是三項在梯度裡的相對份量：模型會比較在意把框對準。這不等於每一步的更新量都變成兩倍：本節用的 Adam 會依每個參數梯度的大小調整步伐，而且訓練過程中三項梯度本身也會改變。

第 7 章提過，若把框座標改成 pixel，box 項的大小會差很多，權重就得重新決定；本節沒有改單位，框座標仍是正規化的比例（沒有單位的 0～1 數值），改的只是乘在 box 項前面的係數。

也因為權重不同，兩次訓練的總 loss 不能直接比高低。舉例：同一個模型若 box_MSE=0.01，權重 5 時這一項貢獻 0.05，權重 10 時貢獻 0.10；模型完全一樣，總 loss 卻差了 0.05。要比較兩個模型，得用同一套評估規則算出的 mAP50。

評估規則也事先固定：

- 候選截斷門檻 0.05：拿每個候選自己的 score 來比，低於 0.05 的候選先刪掉，不交給評估器。
- NMS 的 IoU 門檻 0.5：比兩個預測框之間的重疊。每個類別各自做 NMS：每一輪保留剩下分數最高的框，刪掉和它 IoU 超過 0.5 的同類候選，再對剩下的框重複；已被刪掉的框不會再拿來刪別的框（做法見〈[人工框解碼與 NMS](06-decode-nms.md)〉）。
- 配對 IoU 門檻 0.5：比預測框和 GT 的重疊。預測框和同類 GT 的 IoU 達到 0.5，才算配對成功（TP）。
- AP 用 all-points 插值（第 6 章的算法）。兩類都有 GT，所以 mAP50 是兩類 AP 的平均；在程式最後輸出的 report（`report.json`）裡，它是 `map` 欄位。

畫圖時另用顯示門檻 0.25（也是拿候選的 score 來比）：只畫 score≥0.25 的框，免得大量低分框遮住畫面。圖上看不到的低分候選，仍可能參與評估，所以這兩個 score 門檻不能混在一起。

## 由失敗提出可測的假設

以下只看 baseline 的診斷；改動版的結果放在後面的〈本次 CPU 實跑的結果〉。

validation 共有 17 個 GT。先做 GT 覆蓋診斷（coverage）：對每個 GT，找和它 IoU 最高的同類候選（通過候選截斷門檻 0.05 與 NMS 後的預測），看這個最佳 IoU 落在哪一段。baseline 的結果（report 的 `baseline_coverage`）如下：

- 9 個 GT 的同類最佳 IoU≥0.5：有夠準的同類框覆蓋到。
- 4 個在 0.1 到 0.5 之間：有同類框碰到，但不夠準。本節把這種 GT 叫做近失敗（near miss）。
- 4 個低於 0.1，或根本沒有同類候選：多半是沒有同類框碰到它（IoU 低於 0.1 不保證完全沒碰到，見下方背景 FP 那一條）。

覆蓋數不能直接當 recall。覆蓋診斷替每個 GT 找最好的同類框時，不管那個框是不是已經配給別的 GT；正式評估卻規定一個預測只能配對一個 GT。所以覆蓋數一定大於或等於 TP 數，覆蓋數除以 17 只是 recall 的上限。如果兩個 GT 都只靠同一個框覆蓋（物件互相重疊的資料，例如擁擠的人群，就可能發生），覆蓋數就會比 TP 多。本次實際配對結果裡，這 9 個 GT 各有一個成功配上的同類預測，baseline 的 TP 也是 9；因此這次的覆蓋數等於 TP 數，9/17 正好等於 recall 0.5294。

覆蓋只從 GT 端看。為了分清失敗是框不準、類別錯，還是框落在空地上，程式另外輸出錯誤診斷：baseline 的放在 report 的 `baseline_errors`，改動版的放在 `changed_errors`。每份都從兩端檢查。

**GT 端：錯類覆蓋。** 某個 GT 的同類候選 IoU 都低於 0.5，卻有別類候選和它的 IoU≥0.5，就記為一次錯類覆蓋：位置找對了，類別卻錯了。這些案例列在 `wrong_class_gt_cases`，個數是 `wrong_class_gt_count`。

**預測端：FP 分四種。** 先照正式評估的方式配對：每張圖把預測依 score 由高到低排好，逐一去配同類、還沒被配走的 GT，IoU≥0.5 就算 TP。剩下沒配對成功的預測都是 FP。每個 FP 再用它和所有 GT（不分類別）的最佳 IoU，依序判斷：

1. 低於 0.1 → 背景 FP（`background`）：框和每個物件的 IoU 都很低，多半是落在沒有物件的地方。但 IoU 低不保證沒碰到物件：框比物件小很多（例如整個落在物件裡）或大很多（把物件包住）時，IoU 也可能低於 0.1；只有 IoU 是 0，才表示框完全沒和物件重疊。
2. 0.1 到 0.5 之間 → 定位 FP（`localization`）：碰到了物件，但不夠準。
3. 達到 0.5，而且和同類 GT 的 IoU 也達到 0.5 → 重複 FP（`duplicate`）：那個 GT 已經被更高分的框配走。
4. 其餘（IoU 達到 0.5 的只有別類 GT）→ 錯類 FP（`wrong_class`）：位置對、類別錯。

例如 baseline 在 validation #14（圖片編號從 0 起算）有一個 class 0 的 FP，它和所有 GT 的最佳 IoU 是 0.23，符合第 2 條，所以是定位 FP。各種 FP 的個數記在 `false_positive_counts`；程式也用斷言（assert）確認 TP 數加上 FP 數等於候選總數。這是能逐筆回查的小診斷規則，不是完整的錯誤分析工具。

baseline 的 5 個 FP 裡，4 個是定位 FP，1 個是背景 FP；錯類覆蓋和錯類 FP 都是 0。不過，附近完全沒有候選的 GT 不會算進這兩項，所以不能因此說「分類都正確」。

每個 FP 的明細列在 `false_positive_cases`：validation 圖片編號、預測編號、class、score 與最佳 IoU，可以照編號回查。後面圖的最下方就是其中一例：validation #10 的背景 FP，class 1、score 約 0.068，和所有 GT 的 IoU 都是 0。它低於顯示門檻 0.25，所以一般的圖上看不到；但評估用的候選截斷門檻是 0.05，它仍會被算成 FP。

這個 FP 影響的是 precision：baseline 的 precision 是 9/(9+5)=9/14≈0.6429，少了它會是 9/13≈0.6923。但它這次不影響 AP：它的分數排在 class 1 所有正確框之後，所以沒有改變 class 1 的 AP。真正會拉低 AP 的，是排在正確框前面的高分 FP，例如 **validation 圖片 #4** 裡兩個 class 0 的定位 FP（score 都約 1.00）。#4 是圖片編號，不是預測編號。

**由診斷推出假設。** baseline 沒覆蓋到的 8 個 GT 裡，4 個是近失敗，另 4 個幾乎沒有同類框碰到；5 個 FP 裡有 4 個是定位 FP，錯類為 0。所以先懷疑主要問題是框不準，而不是類別認錯。

假設：box 項在 loss 裡的份量相對太弱；把 box 權重從 5 加倍到 10，可能把近失敗推過 IoU 0.5。這個假設可以檢查：若它成立，近失敗和定位 FP 應該變少。若問題其實是類別錯誤，或根本沒有候選（像那 4 個幾乎沒有同類框碰到的 GT），增加 box 權重可能無效。

**先宣告保留規則，再看結果。** 改動版的 validation mAP50 至少要比 baseline 高 0.01，才保留改動。0.01 是絕對差：baseline 是 0.4444，改動版至少要到 0.4544。這是事先約定的範例門檻，不是統計方法算出來的界線（用統計判斷「差距不太可能只是運氣」的方法叫顯著性檢定，本節沒有做）。validation 只有 17 個 GT，多配對或漏掉一個物件，recall 就差 1/17≈0.059；和這種波動相比，0.01 很小。要確認改善是否穩定，得換 seed 多跑幾次，或用更多資料。

test 要等選擇完成後，只評估選定的模型（程式變數 `chosen`）一次；不能拿 test 的結果反覆調整權重。

完整程式裡對應的幾行如下（摘錄；中文註解是本頁加的，`...` 表示中間省略的程式）。開頭幾行是 `fit()` 裡每一步訓練做的事。在這幾行之前，程式已選好這一步的 8 張訓練圖：`ids` 是它們的編號，`target` 是它們每一格的訓練目標（第 7 章的 [grid targets](07-targets.md)）。

``` { .python data-excerpt="lesson_cases/17-capstone.py" }
# fit() 裡的每一步訓練
optimizer.zero_grad()
# losses 是 grid_loss 回傳的 dict；這裡只取未乘權重的三項，不用 losses['total']（它固定把 box 乘 5）
losses = grid_loss(model(images[ids]), target)
loss = box_weight * losses['box'] + losses['objectness'] + losses['classification']
assert torch.isfinite(loss)  # 每一步都檢查 loss 是有限值（不是無限大或 NaN）
loss.backward()
optimizer.step()
...
# main() 裡：兩次 160 步訓練，各自在 validation 上評估
baseline, base_seconds = fit(train_x, train_y, baseline_weight)
base_metrics, base_preds = evaluate(baseline, val_x, val_y)
changed, change_seconds = fit(train_x, train_y, changed_weight)
change_metrics, change_preds = evaluate(changed, val_x, val_y)
...
# 兩次都評估完，才照事先的規則做一次決定
keep = change_metrics['map'] >= base_metrics['map'] + .01
chosen = changed if keep else baseline  # keep 為 True 就選改動版，否則選 baseline
test_metrics, _ = evaluate(chosen, test_x, test_y)  # test 只評估選定的模型這一次
```

## 本次 CPU 實跑的結果

以下是一次實際執行的結果：PyTorch 2.9.1、CPU、用 2 個 CPU 執行緒（threads）、模型 seed 7。時間會受硬體與同一台電腦上其他程式的負載影響，重新執行可能不同。品質數字也以當次的 report 為準：不同 CPU 或 PyTorch 版本的浮點數加總順序可能不同，數字可能有小差異；剛好在門檻附近的框（例如 IoU 0.49）可能因此改變計數。

| validation 指標 | baseline：box 權重 5 | 唯一改動：box 權重 10 |
| --- | --- | --- |
| class 0 AP50 | 0.3333 | 0.6250 |
| class 1 AP50 | 0.5556 | 0.7778 |
| mAP50 | 0.4444 | 0.7014 |
| precision | 0.6429 | 0.8000 |
| recall | 0.5294 | 0.7059 |
| TP（配對成功的預測） | 9 | 12 |
| 定位 FP | 4 | 2 |
| 背景 FP | 1 | 1 |
| 同類最佳 IoU≥0.5（覆蓋） | 9（共 17） | 12（共 17） |
| 同類最佳 IoU 在 0.1 到 0.5 之間（近失敗） | 4（共 17） | 2（共 17） |
| 同類最佳 IoU 低於 0.1（含沒有同類候選） | 4（共 17） | 3（共 17） |

表中的個數可以驗算 precision 與 recall。兩次訓練的錯類 FP 和重複 FP 都是 0，所以 FP 只有定位與背景兩種：改動版的 precision=12/(12+2+1)=0.8，recall=12/17≈0.7059。兩次的錯類覆蓋也都是 0。本次改動版的這 12 個覆蓋 GT，也各有一個成功配上的同類預測；因此這次的覆蓋數 12 等於 TP 數。

這張表只能比較同一組資料上的兩次訓練。對照[第 7 章 held-out 那一節](07-heldout.md)末段的 160 步實驗：它的訓練設定和本節 baseline 相同，只是資料 seed 不同（7／700／7000），validation mAP50 是 0.80；換成本節的 1100／2200／3300，baseline 只有 0.44。這示範了小資料切分的波動有多大，所以本節只比較同一組資料上的 baseline 與改動版，不拿別頁的數字比高低。

![相同 validation 圖片的真實兩次訓練結果：上 baseline，中改動版（box 權重 10），下方另外檢查 baseline 的一個背景 FP](../assets/diagrams/17-capstone.svg)

圖的讀法：上面兩列是同樣四張 validation 圖片，第一列是 baseline（圖上寫「基準」），第二列是改動版。綠框是 GT（真值），橘框是預測；每個橘框左上角的深色小標籤寫「class:score」。兩列都只畫 score≥0.25 的框；score 在 0.05 到 0.25 之間的候選沒畫出來，但評估時仍算在內。

這四張不是挑最好看的，而是依 baseline「沒被 IoU≥0.5 覆蓋的 GT 數」由多到少挑出（validation 圖片編號從 0 起算；同數時取編號較大者）。判斷框是否配對成功，要看 IoU，不能只看框很接近或 score 很高。圖中的兩例都是 class 0、score 約 0.99–1.00 的高分框，可直接對照：

| validation 圖片／模型 | 最佳 IoU | 判定 |
| --- | --- | --- |
| #4／baseline（兩框） | 0.4918、0.3034 | 兩個定位 FP |
| #14／baseline | 0.2275 | 定位 FP |
| #14／改動版 | 0.4917 | 仍是定位 FP |

這幾個框的同類最佳 IoU 都低於配對門檻 0.5。尤其 #14 的改動版看起來已很接近，仍差一點；原始值是 0.4917，四捨五入寫成 0.49。表中每個 # 都指圖片，不是框的排序編號；圖與表的數值來自本次紀錄的 `false_positive_cases`。

最下方是前面提過的 #10，改用候選截斷門檻 0.05 來畫，橘框是 score≥0.05 的全部候選。紫色虛線框標出那個背景 FP（圖上寫「背景誤報」；誤報就是 FP），那裡只有背景雜訊，沒有物件。它的標籤 1:0.07 表示 class 1、score 0.068（四捨五入成 0.07）。

**回頭對照假設。** 事先的預測是：近失敗和定位 FP 會變少。表中近失敗從 4 個降到 2 個，定位 FP 也從 4 個降到 2 個，和預測一致。但「低於 0.1」那組也從 4 個降到 3 個，表示改動也影響了原本幾乎沒被同類框碰到的 GT，不只是把近失敗的框推準。原因是三項 loss 共用同一個網路，box 權重一改，整個網路的訓練過程都跟著變：objectness 與類別分數會變，預測的排序也會變，而 AP 取決於排序。所以 mAP50 的提升不能全部算在定位改善上。改動版也仍有未解決的失敗：17 個 GT 裡還有 5 個沒被覆蓋，#14 的框也還差一點才過 0.5。

**保不保留，只看事先的規則。** validation mAP50 從 0.4444 升到 0.7014，高於 0.4544，所以本次保留改動版（report 裡「是否保留」的欄位 `keep_change` 是 `true`）。

選定模型接著在獨立的 test 上評估一次：mAP50 0.4452、precision 0.6364、recall 0.5385，都比它在 validation 上低。0.4452 是另外 16 張圖的分數，不能拿來和 baseline 在 validation 的 0.4444 比。test 只用來回報選定模型的成績一次，所以刻意不測 baseline，免得看了 test 又想回頭改選擇。test 也只有 13 個物件：recall 0.5385=7/13，多找到或漏掉一個物件，recall 就差 1/13≈0.077。這麼小的切分波動很大，不能把 0.7014 宣稱為穩定的效果，也不能由這一次斷定改動版在新圖片上一定比較好。

兩個模型都是 15,511 個參數。兩次 160 步訓練各花了約 1.03 秒（baseline）與 1.12 秒（改動版）。這個時間只計 `fit()` 裡的訓練迴圈，不含建立模型、準備 target 的時間（report 的 `train_timing_scope`）。同一個程式裡，第一次訓練常會多花一些只需要做一次的準備時間；沒有前面那次 10 步的暖機，這筆時間會算進先跑的 baseline。兩次訓練每一步的運算完全相同（box 權重只是乘在 loss 前面的一個數），而且各只量一次，本來就會有波動；所以兩個時間若有差距，不能說是改 loss 權重讓訓練變快或變慢。

選定模型的端到端時間（從輸入到畫好框的整段）中位數約 2.96 毫秒。計時用的是選定模型在 validation 上留下最多候選的那張圖（編號記在 report 的 `timed_validation_image`），所以 NMS 有實際的候選要處理。計時範圍是：uint8 RGB 圖（每個顏色值是 0～255 的整數）→ 轉成 tensor → 模型 → 用候選截斷門檻 0.05 解碼，再做 NMS → 畫框；batch 大小 1，先暖機 3 次，再計時 12 次取中位數（12 是偶數，取排序後第 6、7 小兩次的平均），不含讀檔或影片解碼。其中的畫框還包含把圖放大到 192×192、畫上 GT 框與 score≥0.25 的預測框；這是為了診斷才畫的圖，所以這個時間只代表這套診斷視覺化流程，不等於只做推論的時間。這是小型合成資料模型在當次 CPU 上的數值，不是一般 YOLO 的速度承諾。

計時路徑留下的候選數記在 `timed_image_candidates`，程式用斷言要求它大於 0。它不一定等於挑圖時數到的候選數：挑圖用的是前面 16 張 validation 圖一起評估的結果，像素值是 0～1 之間的小數；計時的路徑則一次只算一張，而且圖先轉成 0～255 的整數再轉回來，像素值會有極小的差異。score 剛好在 0.05 附近的候選，可能因此在一條路徑上留下、在另一條路徑上被刪掉。

## 交付、停止條件與下一個檢查

在 repo 根目錄執行 `PYTHONPATH=. python lesson_cases/17-capstone.py`，會得到 `artifacts/lesson-17/report.json` 與 validation 圖 `artifacts/lesson-17/validation.svg`（輸出的最後一行也會印出這張圖的路徑）；上方的圖就是執行紀錄那一次產生的同一張圖。本機命令要先依 [README 環境步驟](https://github.com/birdhackor/learn_to_yolo#readme)安裝固定版本的套件；Colab 則先執行本節的環境格（notebook 的第一個程式格）。

完整程式用斷言（assert）檢查訓練與計時：每次訓練（10 步的暖機與兩次 160 步）的每一步 loss 都是有限值，而且最後一步的 loss 比第一步低；計時的那張圖至少留下一個候選；baseline 的 validation mAP50 要大於 0.05。這個 0.05 是「至少學到一點東西」的下限，和候選截斷門檻 0.05 無關。暖機的步數也受這條「比第一步低」的檢查限制：只跑 1 步時，最後一步就是第一步，檢查一定失敗；只跑 2 步時，拿來比的是更新 1 次後的另一批圖，訓練正常也可能不成立。暖機跑 10 步，比的是更新 9 次之後的 loss，這種誤判少很多，但不能完全排除：最後一步用的仍是和第一步不同的 8 張圖，只更新 9 次時，loss 降低的幅度不一定大過這兩批圖本身難易不同造成的差距。

若這些斷言沒有通過，先查資料、target 與訓練更新，不要繼續宣稱完成。但若錯誤訊息標出的出錯行是暖機那一行 `fit(train_x, train_y, baseline_weight, steps=10)`，先排除上一段說的誤判。錯誤訊息會一層層列出出錯時每個函式執行到的那一行，要看的是標著 `in main` 的那一層；notebook 還會把那一行的前後幾行一起印出，只在出錯行的行號前加上箭頭（例如 `-->`），所以暖機那一行要被箭頭指著才算，只是出現在訊息裡不算。排除的方法是在這一行的開頭加上 `#`，讓它變成不執行的註解，再跑一次。`fit()` 每次開始都重設 seed 7，所以拿掉暖機不會改變兩次 160 步訓練的結果，只是 baseline 先跑，它的訓練時間會多算第一次訓練的準備時間。若這樣所有斷言都通過，就是暖機那次的誤判：兩次正式訓練都通過了同樣的檢查。若仍在斷言停下，就是真的問題，照前面說的去查資料、target 與訓練更新。這條快速路徑共訓練 330 步（10 步不計時的暖機，加上兩次各 160 步），通過後即可停止；正式的比較應該增加資料、換不同的模型 seed，並加入和需求相關的真實場景。

收益：改動有明確的理由，兩個版本只差一項，test 也不參與選擇。代價：每個候選方案都要重新訓練、重新看失敗，而且只跑一次的小資料切分可能誤導。若本次沒有改善，照樣可以交付「不保留」和診斷結果，不必為了完成作業去改評估門檻或捏造提升。

常見錯誤：

- **把 mAP50 變高全部歸因於定位。** box 權重會改變整個網路，分數與排序也會跟著變；理由見〈本次 CPU 實跑的結果〉裡「回頭對照假設」那一段。
- **用 train 的結果當 held-out 的成績。** 模型看過 train 圖，這個分數不能代表它在沒看過的圖上的表現。
- **挑 test 分數最好的權重。** 這樣 test 就參與了選擇，不再是獨立的成績。
- **比較不同候選截斷門檻下的 precision。** 門檻不同，交給評估器的候選就不同，兩個 precision 不能直接比。

## 自主練習

把改動版的 box 權重從 10 改成 2（baseline 仍是 5，其他設定都不變），重跑一次，交付你的結論。

1. 在 notebook〈本節可修改的完整實驗〉標題下方的程式格裡，找到 `def main(baseline_weight=5, changed_weight=10):`，把 10 改成 2 再執行；其他程式（包括斷言）都不用改。
2. 先確認改動有生效：report 的 `box_weights` 是 `[5, 2]`，圖第二列的標題寫「只改一項：box weight 2」。第 1 步的程式格只會印出 report 和圖的檔案路徑，不會直接顯示圖；要看圖，就在 notebook 裡新增一個程式格，執行 `from IPython.display import SVG; SVG(filename='artifacts/lesson-17/validation.svg')`（環境格已把目前目錄切到 repo 根目錄，所以這個相對路徑找得到檔案）。之後只看這次的 report，不要沿用本節 box 權重 10 的結果表。
3. 照當次 report 的 `keep_change`（是否保留改動），交付「保留」或「不保留」。若 validation mAP50 沒有比 baseline 高至少 0.01，答案就是保留 baseline。
4. 交付兩次訓練的診斷（清單見下方）。
5. 從 FP 明細（`false_positive_cases`）挑一例，用它的圖片編號、score 與 IoU，寫出下一個假設。先不要把第二個改動混進同一個實驗。

第 4 步要交付的診斷（baseline 與改動版各一份，都在 report 裡）：

- 覆蓋診斷（`baseline_coverage`、`changed_coverage`）：同類最佳 IoU≥0.5、0.1 到 0.5、低於 0.1 的 GT 各有幾個，欄位依序是 `covered_iou50`、`iou10_to_50`、`below_iou10`。
- 錯類覆蓋數（`baseline_errors`、`changed_errors` 裡的 `wrong_class_gt_count`）：位置找對、類別卻錯的 GT 數。
- 各種 FP 的個數（同樣在這兩份錯誤診斷裡的 `false_positive_counts`）：背景、定位、錯類、重複 FP 各幾個。

??? note "參考答案"

    這題沒有可以照抄的標準數字：答案以你當次的 report 為準，也不要預設「改小權重一定變差」。照下面的順序判讀：

    1. **確認改對了參數。** `box_weights` 要是 `[5, 2]`。baseline 的設定沒變，所以 `baseline_validation`、`baseline_coverage`、`baseline_errors` 應該和本節相同；換了機器可能有小差異。
    2. **保不保留只看規則。** 比較 `changed_validation` 的 `map` 是否至少比 `baseline_validation` 的 `map` 高 0.01。你的結論要和 `keep_change` 一致；若不一致，先重新檢查自己的計算。
    3. **回頭對照假設。** 本節的假設是「box 項的份量太弱」。把改動版的近失敗數（`changed_coverage` 的 `iou10_to_50`）和定位 FP 數（`changed_errors` 裡 `false_positive_counts` 的 `localization`）拿來和 baseline 比，記下是變多、變少還是不變。不論哪一種，都只是這組資料、這一次訓練的觀察，不能當成穩定的規律。
    4. **下一個假設要從一筆具體的 FP 出發。** 寫出圖片編號、score、和所有 GT 的最佳 IoU，以及它屬於哪一種 FP；再說明打算改哪一個設定、預期哪個數字會變。寫法示範（用的是本節 box 權重 10 的資料，不是本題答案）：「改動版在 #14 有一個定位 FP，score 約 1.00，和所有 GT 的最佳 IoU 是 0.49，只差一點就過 0.5。下一個假設：box loss 用的是座標的 MSE，不直接以 IoU 為目標；改用第 11 章的 IoU loss，可能把這類框推過 0.5。下一次實驗仍只改這一項。」

<!-- curriculum-evidence:start -->

## 實際執行紀錄

本節的完整程式於 2026-10-05 在 INTEL(R) XEON(R) PLATINUM 8573C（2 個執行緒）上用 PyTorch 2.9.1+cpu 執行，程式裡的 assert 全部通過。下面是那次印出的原始輸出；輸出裡若有計時或訓練得到的數字，換一台電腦會略有不同。每個數字的意思，以本頁正文的說明為準。[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/17-capstone.json)

??? example "展開本次實際輸出"

    ```text
    {
      "data_seeds": [
        1100,
        2200,
        3300
      ],
      "train_samples": 32,
      "validation_samples": 16,
      "test_samples": 16,
      "steps_per_run": 160,
      "box_weights": [
        5,
        10
      ],
      "score_threshold": 0.05,
      "display_threshold": 0.25,
      "eval_iou": 0.5,
      "baseline_validation": {
        "ap_per_class": {
          "0": 0.3333333333333333,
          "1": 0.5555555555555556
        },
        "map": 0.4444444444444444,
        "precision": 0.6428571428571429,
        "recall": 0.5294117647058824
      },
      "changed_validation": {
        "ap_per_class": {
          "0": 0.625,
          "1": 0.7777777777777778
        },
        "map": 0.7013888888888888,
        "precision": 0.8,
        "recall": 0.7058823529411765
      },
      "baseline_coverage": {
        "covered_iou50": 9,
        "iou10_to_50": 4,
        "below_iou10": 4
      },
      "changed_coverage": {
        "covered_iou50": 12,
        "iou10_to_50": 2,
        "below_iou10": 3
      },
      "baseline_errors": {
        "score_threshold": 0.05,
        "matching_iou": 0.5,
        "background_iou_below": 0.1,
        "wrong_class_gt_count": 0,
        "wrong_class_gt_cases": [],
        "true_positives": 9,
        "false_positive_counts": {
          "background": 1,
          "localization": 4,
          "wrong_class": 0,
          "duplicate": 0
        },
        "false_positive_cases": [
          {
            "image": 4,
            "prediction": 0,
            "kind": "localization",
            "class": 0,
            "score": 1.0,
            "best_any_iou": 0.4917936623096466,
            "best_same_class_iou": 0.4917936623096466
          },
          {
            "image": 4,
            "prediction": 1,
            "kind": "localization",
            "class": 0,
            "score": 0.9999768733978271,
            "best_any_iou": 0.30342045426368713,
            "best_same_class_iou": 0.30342045426368713
          },
          {
            "image": 5,
            "prediction": 0,
            "kind": "localization",
            "class": 1,
            "score": 0.2380521446466446,
            "best_any_iou": 0.2817327380180359,
            "best_same_class_iou": 0.2817327380180359
          },
          {
            "image": 10,
            "prediction": 1,
            "kind": "background",
            "class": 1,
            "score": 0.0679774135351181,
            "best_any_iou": 0.0,
            "best_same_class_iou": 0.0
          },
          {
            "image": 14,
            "prediction": 0,
            "kind": "localization",
            "class": 0,
            "score": 0.986950695514679,
            "best_any_iou": 0.22750937938690186,
            "best_same_class_iou": 0.22750937938690186
          }
        ]
      },
      "changed_errors": {
        "score_threshold": 0.05,
        "matching_iou": 0.5,
        "background_iou_below": 0.1,
        "wrong_class_gt_count": 0,
        "wrong_class_gt_cases": [],
        "true_positives": 12,
        "false_positive_counts": {
          "background": 1,
          "localization": 2,
          "wrong_class": 0,
          "duplicate": 0
        },
        "false_positive_cases": [
          {
            "image": 4,
            "prediction": 1,
            "kind": "localization",
            "class": 0,
            "score": 0.9965797066688538,
            "best_any_iou": 0.2567991614341736,
            "best_same_class_iou": 0.2567991614341736
          },
          {
            "image": 10,
            "prediction": 1,
            "kind": "background",
            "class": 1,
            "score": 0.0718860924243927,
            "best_any_iou": 0.0,
            "best_same_class_iou": 0.0
          },
          {
            "image": 14,
            "prediction": 0,
            "kind": "localization",
            "class": 0,
            "score": 0.9985275268554688,
            "best_any_iou": 0.4916878342628479,
            "best_same_class_iou": 0.4916878342628479
          }
        ]
      },
      "keep_change": true,
      "chosen_test": {
        "ap_per_class": {
          "0": 0.2571428571428571,
          "1": 0.6333333333333333
        },
        "map": 0.4452380952380952,
        "precision": 0.6363636363636364,
        "recall": 0.5384615384615384
      },
      "parameters": 15511,
      "train_seconds": [
        1.0319139630009886,
        1.1217075799941085
      ],
      "train_timing_scope": "only the training loop of each run, after one untimed 10-step warm-up fit; CPU, 2 threads",
      "chosen_end_to_end_median_ms": 2.9623954760609195,
      "timed_validation_image": 2,
      "timed_image_candidates": 2,
      "timing_scope": "uint8 RGB -> tensor -> model -> decode/NMS at score .05 -> drawing, on the validation image with the most candidates; CPU, batch 1, 2 threads; median of 12 runs after 3 warm-up runs; no file I/O",
      "limits": "single initialization, tiny synthetic rectangles; no real-image or multi-seed evidence"
    }
    actual validation panel: artifacts/lesson-17/validation.svg
    ```

<!-- curriculum-evidence:end -->

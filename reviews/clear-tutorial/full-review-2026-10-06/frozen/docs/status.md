# 驗證範圍

教材裡的「通過」和各種數字，能證明的事不一樣。本頁把「程式跑得動、算得對」、「模型學到合成圖形」、「真實照片上的效果」分開，讀完你會知道每種結果能支持哪個結論，以及哪些事沒有驗證。

每節頁面最下方的「實際執行紀錄」，就是該節程式跑出的輸出，並寫明那次執行的日期、CPU 型號與 PyTorch 版本；這些紀錄怎麼產生、什麼時候重跑，見下方〈執行紀錄〉。

目前安排 52 節：第 0–20 章保留 42 節主線，第 21–23 章新增 10 節 ViT／DINO 選讀支線。本版教材與 notebook 固定到 `lessons-v0.6.0`。新支線的執行、內容審查與發布驗證分開保存；公開入口驗證以紀錄內的 tag 與結果為準。既有 `lessons-v0.5.0` 的發布驗證及主線審查保留原本範圍，不能當成新支線已通過的證據。

**第一次讀：**先看開頭、表格每列的最後一欄，以及〈執行方式〉第一段的 Colab 說明；其餘部分會用到第 7、8 章（以及選修的第 20 章）的內容，讀完那幾章再回來看。

## 各種結果各代表什麼

人工答案與少數步更新檢查「程式跑得動、算得對」；合成圖形短訓練檢查受控任務的學習。GPU 的存檔續訓及匯出模型屬工程檢查。Fashion-MNIST 是唯一用到真實資料的一項，只核對分類管線；「真實照片上的效果」沒有測。

| 結果類型 | 做了什麼 | 能支持的結論 |
| --- | --- | --- |
| 人工已知答案 | 用答案事先算好的小例子，檢查座標、target、NMS、AP、assignment、DFL 等計算 | 程式在這些例子上符合定義；不是模型學到的效果 |
| CPU 少數幾步更新 | 只更新幾步（例如第 0 章 1 步、第 7 章 3 步），檢查 loss 是有限值（不是 NaN 或無限大）、梯度能從 loss 傳回要學的參數（都是有限值、總量大於 0）、參數確實改變 | 訓練程式跑得動；少數幾步不代表已收斂（loss 降下來並趨於穩定） |
| 合成圖形短訓練 | 在程式畫的幾何圖形上練比較多步（例如第 1、3、4、10 章的 40 步補充實驗、第 7 章 160 步、第 8 章 1600 步）；第 7、8 章另用不同 seed（亂數種子）畫的新圖評估 | 小模型有能力學會這種受控的簡化任務（各章結果不一，例如第 3 章沒有捷徑的 plain 網路就沒學會）；不能外推到真實照片。其中第 1、4、10 章的 40 步實驗只看訓練圖，只能說明模型對訓練圖記得多少 |
| 雲端 GPU（NVIDIA L4）小規模實測 | 第 7 章的 GridDetector 在 GPU 上短訓練、中途存檔再續訓；另一次把練了 40 步的同款模型匯出，交給 TensorRT 在 L4 上執行，並和同一台機器 CPU 上的 ONNX Runtime 一起與 PyTorch 比對輸出 | 已完成小規模實測：能用 CUDA（NVIDIA 的 GPU 運算平台）在 GPU 上更新參數；中途存檔、讀回後續訓，結果和不中斷相同；PyTorch、ONNX Runtime（執行 ONNX 格式模型檔的程式）、TensorRT（NVIDIA 的推論加速工具）的輸出在容許誤差內一致。不是正式速度比較 |
| 真實資料：只有分類管線核對 | 只用 Fashion-MNIST（28×28 灰階、10 類的分類資料集，圖片是衣服、鞋與包等服飾商品，沒有框）跑 40 步，確認分類管線接得起來；不算偵測結果，也不屬於 52 節（見[資料規劃頁](preparation/data.md#fashion-mnist)） | 不支持任何偵測結論：真實照片的偵測 AP、泛化（對沒看過的照片也做得好）、各機制的效果差異與正式 GPU 效率都沒有測 |

原 42 節主線的逐項紀錄和補充訓練，見[全套實驗與審查頁的〈逐節執行清單〉與〈補充實驗清單〉](validation/curriculum.md)。各節 notebook 的結果取自對應程式的實際 CPU 輸出。有些節為了示範，直接用人手指定的框與分數（例如第 6 章的人工框），不是模型的預測，這些地方都會寫明。沒有訓練的模型畫出框，只代表管線能接起來。

## ViT／DINO 支線的 CPU 實測

[21.4 的小 ViT](lessons/21-training.md)有 23,970 個參數，在 32×32 RGB 紅／藍矩形上更新全部參數 60 步。固定 train 的分類 loss 從 0.710059 降到 0.007155；用不同 seed 產生的 validation 與 test，各答對 64／64 張。梯度有限且非零，主要權重確實更新；第 30 步存模型、optimizer、步數與 RNG，恢復後再走 30 步，模型和 optimizer 與連續 60 步的最大差異都是 0，loss 序列與 RNG 狀態也相同。這項續訓是在本機 CPU 完成。

[22.4 的特徵評估](lessons/22-features.md)先做 160 步簡化 DINO 自監督訓練，再凍結特徵，用 train 的類別答案訓練線性分類器（linear probe）。相同設定下，未經自監督訓練的隨機特徵與自監督特徵，在 test 都答對 64／64 張。因此這次分類題沒有顯示自監督的優勢；loss 有變、特徵沒有完全相同，也不能代替下游能力改善的證據。

[23.2 的定位橋接](lessons/23-detection-bridge.md)凍結 patch 特徵，只訓練一個框與類別的 head。64 張 test 圖的平均 IoU 為 0.605699；類別正確且 IoU≥0.5 的有 54／64 張。每張圖只含一個物件，一個預測框直接配對一個真值框；這是單物件定位，沒有算多物件偵測的 AP／mAP。

三項都只支持固定生成規則的紅／藍矩形任務。DINO（2021）是自監督影像特徵方法，與同名的 DETR 系列偵測器不同；這次沒有新增 GPU 訓練、自然照片評測或 DINOv3 實作。官方預訓練特徵操作是另選，預設 10 節 CPU 實驗不下載資料或權重。

## 沒有驗證的事

- 真實照片上的偵測 AP 與泛化。
- 用真實偵測資料長時間訓練：例如 Penn-Fudan 的行人照片，教材沒有它的格式轉換程式，也沒有用它訓練（見[資料規劃頁](preparation/data.md)）。
- 在真實資料上，用相同資料與訓練預算（例如訓練步數）比較各機制的效果。合成資料上只有少數小型對照，例如 3.3 節的 plain 與 residual。
- 正式的 GPU 速度與效能測試。
- 新 ViT／DINO 支線的 GPU 訓練、原版規模訓練及 DINOv3 實作。
- 在 Google Colab 上執行 notebook，包括 Colab 可能分配給你的 GPU：逐節執行紀錄都在 Colab 以外的電腦上用 CPU 跑出。代替的檢查有兩項。第一，環境格（每節 notebook 最上面的程式格，見下方〈執行方式〉）處理各種情況的方式，例如 PyTorch 版本不同、已經 import 過 torch，由自動測試模擬檢查（`tests/test_notebook_bootstrap.py`，不會真的安裝套件）。第二，網站發布後，在 GitHub 提供的 Linux 電腦上，從指定的公開 tag 重新下載教材（驗證版本以紀錄內的 tag 為準），在全新的 Python 環境裡照原樣執行 notebook 的環境格和實驗格（notebook 最後一格，內容和 `lesson_cases/` 裡該節的程式相同）：第 0 章試沒裝 PyTorch、裝了別的版本、已是 2.9.1、已經 import 過別的版本四種情況，以及第 20 章（它的環境格還要另裝 ONNX 套件）；也依序執行 README 列出的指令（預覽網站用的 `zensical serve` 除外）。結果記在[發布驗證紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum-release-bootstrap.json)。這台電腦不是 Colab，也沒有 GPU。
- 用 TensorRT 跑更大的模型，或改用其他數值精度時的結果，例如強制每一層都用 FP16（16 位元浮點數），或用 INT8（8 位元整數，要先經校準或量化感知訓練等流程決定縮放）。
- 接上實體攝影機當影片來源。
- 真人學生的學習效果。

所以教材裡的 AP 與 loss，只能說明程式算得對、機制照定義運作，以及小模型學得會合成圖形；要用在真實照片上，需要照本頁最後的〈用真實資料做正式實驗的順序〉重做實驗。

## 執行紀錄

每節程式實際執行後，印出的文字會存成一份紀錄（JSON 檔，放在 `artifacts/checks/curriculum/<節的代號>.json`，例如 `07-training.json`），再放進該節的 notebook 和頁面最下方。紀錄寫明執行日期、產生它的電腦（作業系統、CPU 型號、執行緒數、Python 與 PyTorch 版本等）和全部輸出，並用 SHA-256 綁定產生它的程式：該節程式，加上它直接或間接 import 的每個本專案檔案（`miniyolo/`、`scripts/` 裡的模組）。SHA-256 是由檔案內容算出的「指紋」，內容改一點就會不同。第 1、3、4、10 章的 40 步實驗、第 7 章的 160 步訓練、第 8 章的 160 步與 1600 步訓練、第 18、19 章的影片檔實驗，以及[資料規劃頁](preparation/data.md#fashion-mnist)的 Fashion-MNIST 40 步分類管線核對，也各有一份紀錄，同樣記下電腦並綁定程式；Fashion-MNIST 的紀錄另外記下四個資料檔的 SHA-256。

綁定的檔案都沒變，紀錄就一直有效；只要其中一個改了，那份紀錄就過期。發布新版時只重跑過期的紀錄（`scripts/record_evidence.py` 會列出並重新產生），其餘沿用，紀錄裡的日期與電腦也不變，所以各節紀錄的日期不一定相同。

- **CPU 紀錄的電腦：**Linux x86_64（kernel 6.18、glibc 2.41）、Intel Xeon Platinum 8573C，Python 3.12.14、PyTorch 2.9.1+cpu，用 2 個執行緒。
- **GPU 紀錄：**GPU 上的實測只有兩項：第 20 章的 ONNX／TensorRT 核對，以及 GPU 上的存檔續訓（見下方〈已完成的檢查〉）。兩項都由手動啟動的 GitHub Actions（GitHub 提供的自動執行程式服務）工作流程，在 Modal 雲端的一張 NVIDIA L4 上執行（PyTorch 2.9.1+cu128），紀錄同樣綁定所執行的程式。

網站建置前，`scripts/validate_curriculum_evidence.py` 會確認以上每份 CPU 與 GPU 紀錄綁定的程式都沒有改過；配對表中 52 節的紀錄另外要和該節 notebook 存的輸出一字不差，該節頁面也要寫著這份紀錄的日期、CPU 型號與 PyTorch 版本。任何一項不符，網站就無法發布。

計時和訓練後得到的數字會因電腦而不同，訓練得愈久，差異可能愈大（例如第 8 章 1600 步訓練後的評估分數，可能和紀錄明顯不同）；程式的邏輯、tensor 的 shape 和 assert 檢查不會因電腦而變。你在 Colab 或自己的電腦上看到的計時與訓練結果和紀錄不同，是正常的。

## 執行方式

各節的 Colab 按鈕與環境格固定到同一個教材 tag，本版為 `lessons-v0.6.0`。執行紀錄用的是 Python 3.12、PyTorch 2.9.1（CPU 版）。Colab 預先裝好的 PyTorch 版本可能不同；環境格發現版本不是 2.9.1 時，會改裝成 2.9.1 的 CPU 版（預先裝好的若已是 2.9.1，不論 CPU 版或 CUDA 版都保留不動）；改裝後就用不到 GPU。各節實驗都只用 CPU，不必選 GPU 執行階段。若需要改裝，而這個工作階段已經 import 過 torch，環境格會在改裝後停下，提示重新啟動工作階段；照做後從第一格重跑。每個實驗都可單獨執行。

想在自己的電腦上執行：先從 [GitHub](https://github.com/birdhackor/learn_to_yolo) 下載本專案（在頁面上按 Code → Download ZIP 後解壓，或用 `git clone`），完整步驟與套件版本見 [repository README](https://github.com/birdhackor/learn_to_yolo#readme)。照 README 設定好環境後，在 repository 根目錄（教材專案資料夾的最上層）執行下面的指令；Windows 的寫法見 README：

```bash
# 跑第 0 章暖身；可換成其他節的檔名
PYTHONPATH=. .venv-model/bin/python lesson_cases/00-warmup.py
# 用 pytest（Python 的自動測試工具）跑核心與 checkpoint（訓練中途的存檔）測試；
# 全部通過時，最後一行會顯示幾項 passed，沒有 failed 或 error
.venv-model/bin/python -m pytest tests/test_core.py tests/test_checkpoint.py
# 依序執行配對表中 52 節 notebook 的實驗格，每節印一行 PASS 或 FAIL；
# 報告寫在 artifacts/runs/lesson-runtime.json，repo 裡 git 追蹤的檔案都不會改變
.venv-model/bin/python scripts/check_lesson_runtime.py
```

`.venv-model` 是照 README 建好的虛擬環境，也就是專給本教材用、裝好固定版本套件的 Python 環境；`.venv-model/bin/python` 就是這個環境裡的 Python。

核心測試檢查容易出錯的地方，例如空圖、錯誤標註、兩個物件落在同一格、座標縮放取整後的還原，以及重複預測框的配對。checkpoint 測試在 CPU 上跑一次中斷、存檔、讀回、續訓，要求結果和不中斷的訓練相同；也確認舊格式 checkpoint 若沒有保存 RNG 狀態（亂數產生器當下的進度），仍能用來推論，但不能用來續訓。

`check_lesson_runtime.py` 在你自己的電腦上用 CPU 逐節執行，記錄每節印出的文字（stdout）、錯誤訊息與是否通過，並和 `artifacts/checks/curriculum/` 裡的執行紀錄比對。報告寫在 `artifacts/runs/lesson-runtime.json`；各節程式畫的圖和寫出的其他檔案，也都存在 git 不追蹤的位置（例如第 17 章的 `artifacts/lesson-17/`），所以跑完後 repo 裡 git 追蹤的檔案都不會改變。網站上第 17、18、19 章的結果圖，只在產生執行紀錄時，由 `scripts/verify_curriculum.py` 從這些位置複製到 `docs/assets/diagrams/`。換一台電腦，計時和部分數字可能和紀錄不同，報告會標出輸出和紀錄不同的節；輸出不同本身不算失敗，只要程式在 120 秒內完整跑完、沒有報錯，該節仍算 PASS。每節印一行英文結果，例如 `07-training: PASS; output identical to the recorded run`；輸出和紀錄不同時，PASS 後面改印 `output differs from the recorded run`，後面括號裡的說明是固定文字，不是比對後的判斷。報告只標出哪一節不同，不標出哪幾行不同：想知道差在哪裡，把 `lesson-runtime.json` 裡該節的 stdout 和該節頁尾的「實際執行紀錄」逐行對照；計時與訓練後的數字不同是正常的，其他行不同就要回頭查程式。執行 `.venv-model/bin/python scripts/check_lesson_runtime.py --section 07-training` 就只檢查這一節。`check_lesson_runtime.py` 不經過 Colab，所以測不到 Colab 上的實際執行，包括 Colab 可能分配給你的 GPU。

## 已完成的檢查

- **既有主線逐節執行紀錄：**原 42 節實驗已有 **42／42** 的通過紀錄（通過＝程式完整跑完、沒有報錯，任何 assert 檢查不成立都會報錯停下）。這只代表程式跑得動、算得對，不代表模型在真實照片上準。[逐節執行總表（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/index.json)列出每節是否通過與執行日期；各節印出的輸出，在每節頁面最下方的「實際執行紀錄」，完整紀錄在 `artifacts/checks/curriculum/<節的代號>.json`（例如 `00-warmup.json`）。新支線的 CPU 結果見上方，不把舊主線紀錄當成新版全部完成的證據。你自己跑 `check_lesson_runtime.py` 的結果，則寫在 `artifacts/runs/lesson-runtime.json`。
- **核心與 checkpoint 測試：**`tests/test_core.py` 與 `tests/test_checkpoint.py` 的測試全部通過；測試內容見上方〈執行方式〉。
- **匯出與部署：**第 20 章把模型匯出成 ONNX（一種通用的模型檔格式），在 batch 大小 B=1、2、3 時，比對 ONNX Runtime（執行 ONNX 檔的程式）與 PyTorch 的輸出。另在 NVIDIA L4（資料中心 GPU，透過 Modal 雲端租用）上，把另一份練了 40 步的模型，用 TensorRT 的 Python 介面（Python API）建成兩個 engine（為這張 GPU 最佳化後的模型檔，要由 TensorRT 載入才能執行）：一個用 FP32（32 位元浮點數），一個允許 FP16（16 位元浮點數；由 TensorRT 決定哪些層改用 FP16，不保證每一層都是）。兩個 engine 都在 B=1～4 時和同一張 L4 上的 PyTorch 比對輸出（在同一台雲端機器的 CPU 上，ONNX Runtime 也先和這份 PyTorch 輸出比對過）。以上比對的差異都在容許範圍內，詳見[第 20 章〈L4 GPU 上的 TensorRT 實測〉](lessons/20-deployment.md#l4-results)。
- **雲端 GPU 存檔續訓：**也在 L4 上，用第 7 章的 GridDetector 和 8 張合成圖練兩次：一次連續練 40 步；另一次練 20 步後存檔，在另一個全新開啟的雲端 GPU 環境讀回存檔，再練 20 步（兩次合計 80 次更新）。最後兩者的模型與 optimizer 逐一相減，最大差異為 0；學習率排程（StepLR，每隔固定步數把學習率乘上固定比例）與亂數產生器（RNG）的狀態也相同。存檔時也存了這兩種狀態，所以續訓後它們也和不中斷時相同；訓練若用到亂數（例如隨機打亂資料或隨機增強），少存 RNG 狀態，續訓的結果就可能不同。紀錄見 [GPU／checkpoint 實測](validation/gpu-smoke.md)。
- **第 8 章自己的資料：**第 8 章「JSON 標註＋PNG 圖片」的[完整訓練流程](lessons/08-own-data.md)，用的是 48 張程式畫的三類矩形合成圖，不是真實照片。這個實驗有兩份紀錄。第一份是 160 步的診斷：用全部 24 張訓練圖算的 loss 從 1.55017 降到 0.16977，但 train mAP50（在訓練圖上算的偵測評分：紅、藍、黃三類各自 AP50 的平均，最高 1.0）只有 0.00680，validation 是 0；每一步的 loss 紀錄顯示，第 154–158 步出現 loss 尖峰，160 步的評估正好落在這次不穩之後。第二份沿用同一批 train／validation 與全部設定，只把步數改成開跑前就定好的 1600 步（在 CPU 上）；160 步那次已經評過 test，所以這次改用新 seed 畫的 test，也沒有用 test 挑設定。1600 步後，train mAP50 達到 1.0（已背熟訓練圖）；沒參與訓練的 validation 是 0.296296（等於 8/27），換新 seed 畫的 test 是 0.777778（等於 7/9）。這兩組各只有 9 個物件，換一台電腦重跑，分數也可能不同，所以兩者的差距不代表穩定的泛化高低。存檔重新載入，以及在原尺寸圖片上推論，也都通過檢查。
- **第 18、19 章影片檔與追蹤：**把 12 幀畫面（影片中連續的 12 張圖）寫成無損 AVI 影片檔再讀回：讀回的 RGB 畫素和模型預測，都和直接用記憶體裡的畫面時相同；讀到檔尾、提前停止或開檔失敗時，程式都會關閉影片（`capture.release()`）。最後把模型的實際預測接上[第 19 章的追蹤器（tracker）](lessons/19-tracking.md)。沒有測實體攝影機。

## 誰檢查過內容

既有主線上一輪依 repo 的 `clear-tutorial` skill 重審，審查者都是 AI。以下是那一輪的範圍，不涵蓋本次新支線。新支線的逐段閱讀、技術與銜接審閱另保存在[本次紀錄](https://github.com/birdhackor/learn_to_yolo/tree/main/reviews/clear-tutorial/vision-v0.6.0)。先前的完整頁面審查仍保留，不改標成逐段盲讀。

1. **先記首次閱讀的理解。** 凍結 `16f6910` 的 59 個導覽頁，由六位獨立讀者按段落讀。每開放下一段前，先保存當下的理解、原文卡點、猜測與缺圖；沒有先交全文再要求扮演初學者。各組按指定路線實際補讀前提，並非每組都已讀過整本書。
2. **修改後查技術與證據。** 由未撰寫該頁的人對照程式、執行紀錄與圖；疑點回查論文、固定版本的官方程式或文件。必要實驗用 CPU 重跑；這次文字與圖解修訂沒有新增 GPU 訓練。
3. **再看銜接與修正。** 另一位讀者檢查受影響的前文、本節和下一節，並在實際 Zensical 頁面看桌面、手機的圖與公式。修改處有對應複查，不以一份全文摘要代替。

共用檔案系統沒有技術隔離，首次閱讀依揭露規則執行。本輪也記下協調者的問題：部分前提頁漏列、08後段太早收到作者任務提示；這些紀錄不能全部算嚴格盲讀證據，原始問題保留，修正後另作複查。這些限制和每項處理見[本輪原始閱讀與修正紀錄](https://github.com/birdhackor/learn_to_yolo/tree/main/reviews/clear-tutorial/16f6910)。

AI 審查能幫忙找卡點，不等於真人學生已看懂；本專案沒有做真人學生的學習效果測試。

教材介紹的版本到 YOLO26 為止，不包含之後的版本，也不涵蓋 YOLOv6、v7、v9 的機制（範圍見[課程大綱](planning/outline.md)）；只寫找得到原文或公開程式碼可以查證的設計，不為無法確認的版本編造架構。每個已介紹的 YOLO 版本，都以原論文或官方程式碼的某個固定版本為準（commit：程式碼某一次提交的版本編號，對應的內容固定不變），而不是會隨時變動的最新版。

審查紀錄每頁一份，存在 repository 的 `reviews/` 資料夾，記下查核的方法、每個發現的問題與處理方式；審查的總覽見[全套實驗與審查頁](validation/curriculum.md)。每份審查看過的內容（頁面文字、頁面上的 SVG 圖，課程頁還包括該節程式和它 import 的模組）都用 SHA-256 記在 `reviews/coverage.json`。審查之後，只要其中任何一項改了，網站建置前的檢查就會失敗，直到那一頁有一份對應新內容的審查。頁尾自動產生的「實際執行紀錄」和 Colab 連結裡的教材版本（例如 `lessons-v0.6.0`）不算在內。

這些都是編輯審查，沒有做過真人學生的學習實驗：沒有請真人學生實際用本教材學習，再測量學習效果。

## 用真實資料做正式實驗的順序

L4 上只做過小規模的功能測試（8 張合成圖的短訓練、存檔續訓與 TensorRT 數值核對），沒有用真實資料長時間訓練，也沒有公平比較各機制。這類正式實驗可以照下面的順序做；你自己做偵測專案時，也適用。

1. 先用少量資料跑，確認圖片與框對齊、參數真的更新、loss 是有限值（不是 NaN 或無限大），再讓模型在這幾張圖上練到幾乎全對（少量資料的 overfit）。這一步失敗就先修管線。
2. 固定 train／validation／test 切分、類別、圖片大小、兩種 score 門檻（展示時只看高分框；計 AP 時通常保留較低分候選）、NMS 與 AP 定義。兩個門檻用途不同，不把顯示用的 0.25 直接拿來截斷 AP 候選；每次評估明記實際門檻。只用 validation 選設定，test 留到設定確定後。
3. 跑真實資料的 baseline（之後每個改動都拿來比較的基準），儲存設定、seed、checkpoint、成功與失敗圖，以及端到端時間（從讀入圖片到輸出框的總時間）。
4. 每次只改一項機制，在相同資料與訓練預算下比較；記錄候選數、參數量、記憶體與額外設定。不因某個 seed 的小差異就宣布勝負。
5. 對有希望的設定，換幾個不同的 seed（亂數種子，決定初始權重等隨機結果）各跑一次，確認優勢不是運氣；再試影片與部署。TensorRT 只核對過本教材小模型在 L4 上的輸出；換成更大的模型、真實場景或其他數值精度（例如強制每一層都用 FP16，或用 INT8），都要分別驗證。

# 第08課JSON學習閉環補項：獨立技術準確性審查

審查日期：2026-10-02。初審發現1項required文字修正，修改後已獨立重查通過；作者也採用O1可選guard強化，另經七個反例及合法fixture獨立驗證關閉。最終沒有未關閉的required問題；三類訓練、實際框座標、loss／梯度、AP、checkpoint及固定實驗數值均通過獨立核查。修改後的獨立重查另記於文末，保留本節初審發現，不以作者修訂自述代替重查。

範圍：`docs/lessons/08-own-data.md`新增學習閉環、`scripts/run_custom_data_learning.py`、`miniyolo/{custom_data,data,geometry,targets,models,losses,train,inference,metrics,checkpoint}.py`、`scripts/detect_image.py`、`custom-data-160-step.json`與`custom-data-learning.json`。只新增本審查檔；獨立執行及檢查程式均放在`/tmp`。沒有覆寫既有教材／程式／產物，沒有Git命令、Actions、GPU工作、下載或seed search。

## 初審發現

**R1 — required：限定「同格兩物件拒絕」的split範圍。** 初審時[自訂JSON入口說明](/workspace/learn_to_yolo/docs/lessons/08-own-data.md:120)稱「它還會拒絕……同格兩物件」。實際runner只為train呼叫`build_targets`；validation／test只把原始GT交給評估。因此同格兩物件在train會拋`ValueError`，在validation／test不會被拒絕，也沒有被悄悄丟棄。

獨立反例位於`/tmp/custom-accuracy-validation-collision/`：三個split各一張64×64、不同像素及source_id的合法PNG；validation的`[2,2,6,6]`與`[8,8,12,12]`中心都落在cell(0,0)。執行runner的`--annotations ... --steps 1`成功exit 0，`final_validation.ground_truth_objects=2`；另直接呼叫`build_targets`編碼該validation target才拋`same-cell collision`。最小修正是明說「train建立target時拒絕碰撞；validation／test保留所有GT做評估」。固定48張fixture每張至多一框，本項不影響其數值或學習結論。

**O1 — optional：加強follow-up的manifest約束。** `--prior-diagnostic`會比較固定設定與train／validation的path、split、PNG SHA，但不逐列比較舊新框、labels、source_id。此次兩份manifest的完整train／validation記錄已經由本審查獨立確認完全相同，所以沒有違反此次協定；若將這個guard推廣成一般實驗協定，可再比較這些欄位。這不列成此次required修正。

## 獨立執行與provenance

使用現有`.venv-model`：Python 3.12.14、PyTorch 2.9.1+cpu、NumPy 2.3.5、Pillow 12.0.0，CPU兩個threads。獨立執行：

```bash
.venv-model/bin/python scripts/run_custom_data_learning.py \
  --fixture --steps 1600 --fixture-test-seed 7001 \
  --prior-diagnostic artifacts/checks/curriculum/custom-data-160-step.json \
  --output /tmp/custom-accuracy-closure-1600 \
  --report /tmp/custom-accuracy-closure-1600/report.json \
  --diagram /tmp/custom-accuracy-closure-1600/learning.svg
```

該次exit 0。初審runner SHA-256為`f5f7996fb22ae0f2c3bf0486953a30a0c1ca8c5b8b6f0c1ad54f82f889094140`，與當時主1600步報告相符；主報告SHA為`a58e4b73c5a4308d2a2df841a2782dc225bdc9bfd08ce37b7a7f2123b59a4f51`。舊160步報告SHA為`59e0b9fb8a1af599d28a882f83e4a80a155a2ca76425d42372e9e531039dc488`，其中保留的`executed-160-step-script.py` SHA確實為該報告記載的`5c3d9d89c3d59d8d9ed47d3fdc049476955dd41f494ff592da225a8ccc5c13a4`。

兩份報告的manifest、PNG inventory、checkpoint SHA及全部八個core source SHA皆逐檔相符。`repository_commit=c0463720b24c4d8a86c739f8c2784ad25164a6ee`明確標成checkout基底commit；本輪working-tree程式由script SHA辨識，沒有把基底commit誤稱成已包含新增程式的commit。

手算與資料檢查程式為`/tmp/custom-accuracy-closure-check.py`，結果為`manual-check.json`；兩種原圖方向的獨立CLI與手算逆映射結果為`cli-manual-check.json`，都位於上述`/tmp`目錄。

## 三類、尺寸、target與loss

|核查|獨立結果|
|---|---|
|實際JSON／PNG|48張均能讀取，24 train／12 validation／12 test；全為120×80或80×120，每split3張空圖；train各類7個GT，保留split各類3個GT|
|原圖框|直接從實際PNG找前景邊界；24張train含空圖全部與原始JSON一致，48張變換後框均符合手算|
|letterbox|120×80→64×43、top=10；80×120→43×64、left=10。框用取整後實際比例64/120及43/80，最大手算誤差4.58e-6 pixel|
|三類head與target|head最後維度8、15,544參數；class id 2進入三類CE；21個train GT→21個positive cell，手算offset與wh最大誤差1.20e-7|
|mask／gradient|box MSE與class CE只取positive；背景box及class logit梯度精確為0；box梯度與解析式最大差1.17e-10|
|1600步|全1600條loss history、梯度L2範圍及參數delta與主紀錄完全相同；每步整體梯度有限且非零|

原圖pixel xyxy→letterbox pixel→center cell offset加整圖normalized wh→sigmoid logits loss→decode的尺度契約相符。正格box MSE平均於21×4座標；全格objectness BCE平均於24×4×4；class CE平均於21正格。背景參與objectness監督，不表示每個已飽和float32 logit都必有非零數值梯度：舊160步有341/363個背景obj梯度非零，22個極負logit的sigmoid已下溢。教材「每步梯度L2非零」指整個模型的norm，與此不矛盾。

160步checkpoint獨立forward的完整train loss為`.16920891404151917`，不是最後一個minibatch更新前的`.161305770277977`。最小正格預測高度為`.0548070073` pixel，而最小GT高度為`9.67499924` pixel；背景box mask正常，wh target非零，框仍未對準。教材保留train AP50 `.0068027211`、validation AP50 `0`及這個未收斂診斷，沒有拿total loss下降宣稱已學會定位。

1600步的`final_train_loss=.000354691524989903`在全部1600次optimizer更新後，對同一套完整24張train重新forward計算；曲線則明說是每次minibatch更新前loss，最後一點為`.0007454031729139388`。兩者沒有混用。重建seed7初始化模型與checkpoint逐參數相減，獨立算得delta L2 `16.519737946734185`，與紀錄一致。

## AP定義與獨立重算

沒有再次對test做模型forward或設定選擇；使用那次獨立run保存的predictions，另以純NumPy／Python的IoU、matching、precision envelope及recall跳點積分重算。半開pixel框面積不加1；按類別跨圖排序分數，同圖同類尚未使用的GT最多配對一次，IoU≥.5才是TP。

|split|TP／預測／GT|各類AP50（紅／藍／黃）|mean AP50|micro precision／recall|
|---|---|---|---|---|
|train|21／21／21|1／1／1|1|1／1|
|validation|5／10／9|4/9／1/6／5/9|.388888888889|.5／5/9|
|新test7001|6／9／9|1／2/3／1/3|.666666666667|2/3／2/3|

全部與主紀錄及本次獨立runner一致。score為sigmoid(objectness)×最大softmax class probability；score≥.1、同類NMS IoU .5、matching IoU .5在三split一致。這是教材明示的單一IoU all-points AP，沒有冒充COCO AP .50:.95，也沒有把micro precision／recall當成Ultralytics自身最佳F1門檻的指標。

查閱既有官方source cache `/tmp/yolo_sources/yolov10-metrics.py`的`compute_ap`（第500行）：precision envelope後的continuous分支就是recall跳點加權積分，與此教材算法相符；該官方預設實際用101點插值，教材已明說自己的AP定義。官方map屬性（第723行）則平均IoU .5–.95，與本頁所排除的宣稱一致。

## checkpoint與原圖推論

format v2包含model、Adam state、steps=1600、設定、有序class_names及Python／NumPy／torch CPU／CUDA RNG欄位；CPU run的CUDA list為空、scheduler為None。獨立run的raw輸出`torch.equal`、decoded、全部Adam state和恢復後RNG均完全相同；raw最大差0，Adam10個state entry的step都為1600。這是狀態重讀與CPU推論一致檢查，沒有把它外推成跨裝置bitwise相同或本次已驗證續訓軌跡。

另真正執行`detect_image.py` CLI，使用validation index1（80×120）和index2（120×80）。從64×64預測手算扣padding、除各軸實際比例並clip原圖範圍，與CLI boxes最大差分別8.39e-6及5.77e-6 pixel；labels與scores亦相符，輸出PNG保持各自原圖尺寸。模型由checkpoint有序三類名稱重建三類head。主AP／SVG仍用letterbox座標，本页原圖推論框没有与它們直接比數值。

PyTorch語義以本機已安裝2.9.1官方source核對：`torch/nn/functional.py`的MSE（3815）、BCEWithLogits（3529）、CE（3375）預設reduction都是mean；`interpolate`（4530）以明示`size`及`align_corners=False`處理影像；`torch/optim/optimizer.py`的`load_state_dict`（867）還原state及param groups；`torch/random.py`（10）說明CPU RNG state恢復。解析式梯度亦以實際autograd數值獨立核對，沒有只憑API印象。

## 固定協定、來源保留與test使用

舊新train／validation的完整manifest記錄（包含框、類別、尺寸、source_id）及逐PNG inventory完全相同；新1600步history前160項與舊160步逐值完全相同，模型／optimizer／seed／lr／門檻及core source未變。step預算由160改為事前指定的1600，final-step直接保存，沒有best-validation checkpoint邏輯或AP接受門檻。

舊test7000已看過，教材與舊報告沒有隱瞞；follow-up以預先宣告的7001產生新test PNG，與舊test無完全重複，train／validation未改。protocol檔在任何train step及保留集評估之前寫入；主初次1600步宣告時間為18:08:45.111280 UTC。訓練前只evaluate validation初始結果；訓練迴圈只消費train。完成1600步後的split迴圈對test呼叫evaluate一次，之後reload／原圖檢查只使用validation。

`test_evaluations=1`是每次固定run的呼叫次數，不是聲稱這套可重現資料在整個審查歷史中從未再次運行。獨立重現與之後僅展示修正的重跑需要誠實區分；都不能再用test成績挑設定。本次source、既有診斷與固定設定一致，沒有test驅動的seed／模型／門檻搜尋證據；單靠self-reported布林欄位也無法證明任意未記錄的歷史行為。

source_id是合成來源標籤；PNG與decoded RGB跨split完全重複皆為0。這僅支持此次fixture的資料盤點，不能證明真實影片近似幀獨立性。本頁已明示這個限制與每個保留split只有9個物件，沒有正式泛化或YOLO版本排名宣稱。

初審計時核對：主紀錄training約6.807秒、完整流程7.573秒，計時起訖與正文相符；獨立run training約8.062秒，屬環境負載差異。完整流程起點在資料生成前、終點在SVG輸出後，排除imports／程序啟動、套件安裝及最後report JSON write。沒有把這個單次小模型CPUwall time當正式效能。

## 第一次修改後獨立重查（R1／SVG）

作者完成修改後，本審查重新讀取最新source、正文、兩份JSON及其依賴檔案，再執行手算／數值parity檢查；沒有修改作者程式或拿自己的修訂當作通過證據。

**R1已關閉。** [最新自訂資料入口](/workspace/learn_to_yolo/docs/lessons/08-own-data.md:120)明說「建立train target時拒絕同格兩物件」以及「validation／test保留所有GT評估，不為配合模型而刪除同格物件」。這與本審查保留的實跑反例和實際call graph一致。算法及GT保留行為沒有被改掉。

主例的一步檢查與後半48圖／1600步閉環已在第一段分開說明。新增四張圖的選擇說明也符合runner：從validation各類第一張及第一張空圖取index `[1,2,3,0]`，沒有依AP或圖像效果挑選。

最新runner SHA為`e970cb91ba6e7e350bf476452727ed73a6f371083996d6f5681e508546adf98f`；主1600步報告SHA為`4e836fcb5fbe2ec9bd5ef4ef5e94c099186faa9002c27d4e3013d307823aa48b`；公開SVG SHA為`bdf04fb565023f8d783f6a38f99a3a0bc2fbef069db9422e682654f75d796cfe`。主報告記錄的script SHA與最新bytes相符。舊160步報告仍為初審的`59e0b9fb…`，沒有被覆寫成成功結果。

最新八個core source SHA均與獨立run及舊160步相同；最新完整1600步loss history、initial／final loss、三split AP／precision／recall、梯度範圍、參數delta與reload checks全部與獨立run逐值相同。最新train／validation完整manifest及逐PNG inventory仍與舊160步完全相同，各檔manifest／inventory／checkpoint SHA亦逐檔重新核對。

另一路閱讀審查提出的GT／prediction標籤間距問題只修改SVG renderer。使用本審查原先獨立run的validation圖片／GT／真實預測，再呼叫最新renderer，生成`/tmp/custom-accuracy-closure-1600/latest-render.svg`；它與公開SVG逐byte完全相同。三個正圖panel的GT caption在y=460、prediction caption在y=475，分數及框仍為原始實測內容。此重查沒有重新對test做模型forward。

作者為保存新renderer SHA所做的固定1600步重跑，protocol宣告時間為18:15:07.214790 UTC，仍在訓練前。設定與學習core沒有變更，main中test評估仍只有完成更新後的一次呼叫；這是相同設定的展示／provenance重現，沒有用test成績選擇新配置。最新training `6.693091570s`、完整流程`7.467487915s`與正文`6.693／7.467秒`相符，初審時間仍在本報告保留，不混作同一次量測。

主紀錄的standalone CLI與callable JSON也重新核查：`image_path`及`checkpoint_path`一份為相對字串、一份為絕對字串，`resolve()`後為同一檔案；其餘prediction／class／門檻／尺寸欄位完全相同。各自JSON SHA都與主報告相符，沒有宣稱兩個原始JSON bytes相同。`detect_image.py` SHA仍為`1f3379f5fcccc0b7c17b38e7434b4f816d46b0ff9e834e8c5966d05c4875515f`。

重查結果另存於`/tmp/custom-accuracy-closure-1600/latest-recheck.json`。本範圍沒有未關閉required問題；O1保留為可選guard強化。未驗證真實照片泛化、正式YOLO完整模型、跨裝置bitwise一致或新remote tag的Colab安裝；本頁並未據此宣稱這些能力，故不列成required。

## 最後guard補強獨立重查（O1）

作者隨後採用O1：讀取舊新manifest，以原有split過濾函數逐列比較完整train／validation記錄；除了既有PNG SHA盤點，現在也檢查boxes、labels、source_id、尺寸、path、split與記錄順序。protocol增加`same_train_validation_annotations=True`。本審查保留上述初審optional建議及第一次重查紀錄，另驗證新增行為。

獨立程式`/tmp/custom-accuracy-followup-guard-check.py`把舊manifest、report、inventory複製至`/tmp/custom-accuracy-followup-guard/`，不修改原始診斷資料。保持train／validation PNG inventory逐列相同，在train與validation分別只改boxes、labels、source_id，共六案；另交換兩筆train記錄的順序，共七案。七案均以新的annotations／sources guard拋`ValueError`，而監測的`GridDetector`與`optimizer_step`呼叫次數皆為0，證明在模型建立與學習前拒絕，不是靠稍後訓練碰巧失敗。

第八案為合法、未改標註的fixture；它通過全部guard，protocol的兩個`same_train_validation_*`欄位都為True，並到達第一次`optimizer_step`。本檢查在該呼叫用sentinel停止，沒有實際參數更新或test評估，也沒有挑新設定。該檢查exit 0，結果保存為`/tmp/custom-accuracy-followup-guard/guard-check.json`；原160步報告SHA仍為`59e0b9fb8a1af599d28a882f83e4a80a155a2ca76425d42372e9e531039dc488`。

**O1已獨立驗證關閉。** 最終runner SHA為`53799d56b5be6bbe31aa52f18f7bef9c9b2fce48987a5ea55810c0513c44c789`，主1600步JSON SHA為`fb4bf08300a24615d755c90a44f5585e7e1023836c750b1795d328d5450d1cb8`。最終報告的script SHA與實際bytes相符，新增annotation flag為True。八個learning core source仍未變；全部1600步loss history、更新前／後完整train loss、三split AP／precision／recall、gradient norm範圍、weight delta、reload checks及各validation例子的真實框／分數均與原先獨立1600步run完全相同。

再次逐檔核對manifest、inventory、checkpoint與CLI JSON SHA，舊新train／validation完整manifest及PNG inventory均相同；最終history前160項仍等於舊失敗run。standalone與callable的兩個path欄位resolve等價，其餘欄位精確相同。公開SVG仍為`bdf04fb565023f8d783f6a38f99a3a0bc2fbef069db9422e682654f75d796cfe`，與先前獨立使用最新renderer生成的SVG逐byte相同。

這次保存guard provenance的固定重跑，在18:22:46.881585 UTC宣告相同1600步／test seed7001 protocol；新增guard不改模型、學習core、門檻或選擇規則。training `7.517184449s`、完整流程`8.272836981s`与最終正文`7.517／8.273秒`相符。最終逐項重查結果為`/tmp/custom-accuracy-followup-guard/final-recheck.json`。R1及O1均已關閉，最終未關閉required數為0。

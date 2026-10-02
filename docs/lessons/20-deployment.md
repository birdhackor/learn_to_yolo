# 20 ONNX／TensorRT：匯出後先證明同一個輸入得到同一個結果

[開啟 Colab](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.3.0/notebooks/20-deployment.ipynb) · 原始碼：`lesson_cases/20-deployment.py`

前置是model.eval、圖片契約、letterbox與decode。模型檔成功產生，不代表部署成功；運算圖、輸入layout、分數公式、NMS和框還原任一處不同，都可能造成框偏移。本節真的匯出ONNX、執行checker與ONNX Runtime CPU，再比較raw輸出和還原後的框。TensorRT另用Actions／Modal在單張L4實際建置與執行，紀錄見下方；Colab本身仍執行CPU範例。

本次使用GridDetector(width8)，從零做一次實際forward、grid loss、backward與SGD step，再轉eval。只有一次更新，不足以證明偵測品質；匯出一致性可以用這份固定權重驗證，準確率則要用第17章那類獨立評估。案例不下載pretrained、不需torchvision。

ONNX是儲存模型運算圖的格式；ONNX Runtime（ORT）讀取它並執行。TensorRT則在NVIDIA GPU建置並執行最佳化engine。backend是執行模型的這些後端；opset是ONNX運算子的版本集合；engine是經建置產生的可執行檔，optimization profile指定允許的輸入shape範圍。

數值`4.77e−7`等於.000000477。atol是固定允許差值，rtol再按參考數值大小放寬；allclose逐值要求`abs(actual−reference)≤atol+rtol×abs(reference)`。接近不等於逐位元完全相同。

## 固定部署契約纔有可比性

來源是uint8 HWC RGB，三張尺寸為64×96、48×80與80×48。各自轉float32 CHW、除255、letterbox64×64，metadata分開保留。模型輸入`[B,3,64,64]`，raw輸出`[B,4,4,7]`；前四項為grid框logits，第五為objectness，最後兩項為class logits。score仍是`sigmoid(obj)×softmax(class)`，不能改成第16章的純sigmoid類別分數。

ONNX本節只匯出model raw，前處理與後處理在外部Python執行，因此兩個backend可共享同一套程式。decode使用score .05、class-wise NMS IoU .5，再undo_letterbox到每張原圖。這不是把整條pipeline都包進ONNX，檔案介面要明確說清。

```python
torch.onnx.export(model, example, 'grid.onnx',
    input_names=['images'], output_names=['raw_grid'],
    dynamic_axes={'images': {0: 'batch'}, 'raw_grid': {0: 'batch'}},
    opset_version=17, dynamo=False)
onnx.checker.check_model(onnx.load('grid.onnx'))
session = ort.InferenceSession('grid.onnx', providers=['CPUExecutionProvider'])
```

PyTorch2.9預設新exporter；本例明確選`dynamo=False`的legacy路徑以使用已測過的opset17與dynamic_axes，會印出棄用warning。沒有關掉warning或宣稱它永遠受支援；升級exporter時要重新比對輸出，並使用該版本支援的dynamic_shapes與依賴。

## 三層驗證都有可能失敗

第一層checker檢查ONNX圖的型別與結構，不證明數值等價。第二層用ORT執行B=1、2、3，逐值比較raw；本輪三種batch的最大絕對差均約`4.77e−7`，低於assert的`atol=1e−5,rtol=1e−5`，數值也寫入report。第三層兩份raw分別經相同decode與還原，確認框座標`atol=1e−4`、分數`atol=1e−6`、類別完全相同。

raw接近仍可能在score或NMS閾值附近跨過分支，讓候選集合不同；所以第三層不能省略。實際部署還要放入空圖、小物件、極端長寬比和擁擠圖，記錄差異是哪個階段產生。這個案例已比較三張非正方形來源，但沒有驗證所有圖片。

## 動態batch不是動態空間

ORT輸入metadata為`['batch',3,64,64]`，只有B可變；B=1／2／3都實際透過。每次執行前assert輸入確實為`[B,3,64,64]`，再assert兩個backend的raw都是`[B,4,4,7]`，避免只有兩張卻把`batch[:3]`當成三張測試。輸入80×80會被ORT拒絕，程式assert這個行為，避免誤把Python模型可接受別尺寸當成ONNX也已支援。H、W若要動態，需重新匯出並確認pooling、candidate生成與decode的shape契約，逐尺寸測試；不只把64改成字串就算完成。

部署前也要固定batch服務策略。B=2的batch latency是一次完成兩張所需的時間；raw throughput是`2 ÷ batch latency（秒）`，單位為images/s。它不包含湊batch的等待，也不能當成單張請求的服務延遲。若服務等湊滿batch，低流量下可能增加單張latency。產品要的是即時回應，還是大量離線吞吐，會影響是否採batch。

## 同裝置量raw與整條管線

程式CPU固定2threads，3次warmup後量20次median。它分別量PyTorch raw、ORT raw，以及兩者的RGB前處理→raw→decode/NMS→原圖框還原；另列ORT raw batch2和images/s。沒有磁碟、螢幕或cameraqueue時間。

本輪實跑PyTorch raw約0.153毫秒、ORT raw約0.036毫秒，同一套端到端分別約2.329與2.144毫秒；raw比約4.2倍，端到端減少約7.9%，前後處理不能忽略。ORT raw batch2約0.043毫秒，約46,966張／秒是模型raw吞吐，不含前後處理或湊batch等待。重新跑會在`artifacts/lesson-20/report.json`得到當次兩backend端到端數字，不能把單一raw speedup直接宣傳成產品總加速。ONNX1.19.1、ORT1.23.2、PyTorch2.9.1 CPU的版本也寫入report。

執行`PYTHONPATH=. python lesson_cases/20-deployment.py`，需安裝`onnx==1.19.1`與`onnxruntime==1.23.2`。成功條件是產生真實`artifacts/lesson-20/grid.onnx`、checker透過、ORT實際執行三種batch、raw與原圖輸出比對都透過。僅能import套件或印出provider名稱不算完成。

## 有NVIDIA GPU時，TensorRT怎麼接

下面的`trtexec`命令是CLI對照說明，尚未執行；本輪實際使用的是固定TensorRT 10.13.3.9的Python API（下節）。先記GPU、driver、CUDA與TensorRT版本，用相容版本讀取已透過ORT比對的ONNX。以本節動態batch、固定64空間的模型為例：

```bash
trtexec --onnx=artifacts/lesson-20/grid.onnx --saveEngine=grid-fp32.engine --noTF32 \
  --minShapes=images:1x3x64x64 --optShapes=images:2x3x64x64 --maxShapes=images:4x3x64x64
trtexec --onnx=artifacts/lesson-20/grid.onnx --saveEngine=grid-fp16.engine --fp16 --noTF32 \
  --minShapes=images:1x3x64x64 --optShapes=images:2x3x64x64 --maxShapes=images:4x3x64x64
trtexec --loadEngine=grid-fp16.engine --shapes=images:1x3x64x64
```

依安裝的TensorRT版本檢查flags與支援的ONNX運算元；engine通常受GPU架構與runtime版本限制，不當成跨裝置通用檔案。FP32基線加`--noTF32`，目的是停用可能預設允許的TF32乘法；FP32輸入或engine檔名本身不能保證完整FP32計算。這個flag及兩條建置命令都未在本次CPU環境測試，使用時須記錄版本與實際精度設定。先對FP32 engine用相同RGB輸入比raw與decoded結果，再測FP16，重新評估AP和閾值敏感樣本。INT8還需要具代表性的校準或量化流程，不能只加flag期待品質不變。`trtexec`量的是engine相關執行，不會自動含本教材的letterbox和NMS；產品端到端另量。GPU計時須同步或使用CUDA events，避免只量到排入工作。

收益是跨runtime執行與可能的效能最佳化，代價是版本、shape、運算元和精度契約。常見錯誤是BGR未轉RGB、float32未除255、輸出軸讀錯、忽略框還原、把FP16誤差視為必然可忽略，以及在無GPU環境寫「TensorRT已驗證」。自主練習：沿著已新增的第三張RGB source，檢查processed、batch與metadata都有三份；在B=3迴圈核對輸入`[3,3,64,64]`、ORT輸出`[3,4,4,7]`，確認report的`batches_checked`為`[1,2,3]`，raw與三張原圖還原框都透過。再暫時移除第三張source，預期B=3的輸入shape斷言失敗；不能用兩張的`batch[:3]`冒充B3。若改H=W=80，應像本例被拒絕，需另行匯出與decode設計，不降低assert以掩蓋契約錯誤。

參考：[PyTorch ONNX官方檔案](https://pytorch.org/docs/stable/onnx.html)、[ONNX Runtime Python入門](https://onnxruntime.ai/docs/get-started/with-python.html)、[TensorRT 10.13官方trtexec範例](https://github.com/NVIDIA/TensorRT/blob/b8db91e15be2cae4465ac17fab19e0f969e45407/samples/trtexec/README.md#example-3-running-an-onnx-model-with-full-dimensions-and-dynamic-shapes)與[flags定義](https://github.com/NVIDIA/TensorRT/blob/b8db91e15be2cae4465ac17fab19e0f969e45407/samples/common/sampleOptions.cpp)。CLI命令未實測；Python API的GPU結果見下節。

## 本輪L4的實際TensorRT結果

[Actions實測37035865497](https://github.com/birdhackor/learn_to_yolo/actions/runs/37035865497)使用原GridDetector、15,511參數，固定8張合成圖、40次Adam更新後匯出。PyTorch 2.9.1+cu128／CUDA build 12.8、TensorRT 10.13.3.9、NVIDIA L4；loss為0.9848→0.0710。這與上方CPU的一步模型是兩份不同權重，不能互相比速度或AP。

Python API真的建置、deserialize engine，使用`set_input_shape`／`set_tensor_address`／`execute_async_v3`執行。optimization profile為B=1～4、固定64×64，實際逐一核對B=1、2、3、4；PyTorch與ORT CPU先比對，再與TensorRT比對，並確認非空decode的框、類別、候選順序與分數。

|建置設定|raw最大絕對差|框最大差（pixel）|raw B1 median|
|---|---|---|---|
|FP32、TF32停用|4.77e−6|1.15e−5|0.160ms|
|允許FP16、TF32停用|4.77e−6|7.63e−6|0.162ms|

**允許FP16不等於每層都以FP16執行。** Builder可混用精度與選擇tactic；本次沒有逐層精度稽核，誤差與時間接近也不能說FP16一定提速。以上是建置flag的結果，不是強制全FP16的品質或效能對照。INT8未測。

計時先warmup3次、再取20次median，包含shape/address設定、輸出buffer配置、enqueue與stream／device同步；不含前處理、decode、CPU拷貝、engine建置與container啟動。Actions client總時間223.41秒則包含建置、啟動與傳輸。這個小模型的單次測試不能當正式訓練或服務效能。

ONNX與兩份engine存於專案專用Modal Volume的`projects/learn-to-yolo/deployment/github-37035865497-1-e64ea79207ed/`，明確commit後由另一CPU container重新讀取，三個檔案的SHA-256均相符。Modal已停止、tasks=0。模型與engine沒有進普通Git。[完整JSON](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/deployment-gpu.json)；重跑入口是手動workflow `deployment-gpu.yml`，一般push／PR不啟動GPU。

L4表的raw／框差是B1～4所有結果中的最大值，時間只量B1。CPU JSON中的限制只描述CPU case；GPU另有獨立Actions證據，不能把兩份report的範圍混為一談。

本機命令需先依[README環境步驟](https://github.com/birdhackor/learn_to_yolo#readme)安裝固定依賴，並在repository根目錄執行；Colab則先跑本節環境格。

<!-- curriculum-evidence:start -->

## 本輪實際執行紀錄

本節範例已於 2026-10-02 使用 PyTorch 2.9.1+cpu 在 CPU 執行，程式中的斷言全部通過。以下是該次輸出；人工輸入、短步更新與模型效果的意義仍依本頁說明區分。[完整紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/20-deployment.json)

??? example "展開本次實際輸出"

    ```text
    {
      "onnx": "artifacts/lesson-20/grid.onnx",
      "opset": 17,
      "providers": [
        "CPUExecutionProvider"
      ],
      "input_shape": [
        "batch",
        3,
        64,
        64
      ],
      "batches_checked": [
        1,
        2,
        3
      ],
      "max_abs_raw_errors": [
        4.76837158203125e-07,
        4.76837158203125e-07,
        4.76837158203125e-07
      ],
      "decoded_boxes_scores_labels_match": true,
      "dynamic_spatial": false,
      "spatial80_rejected": true,
      "median_ms": {
        "torch_raw_batch1": 0.15258449980137812,
        "ort_raw_batch1": 0.03642449973995099,
        "torch_preprocess_to_restored_boxes": 2.3286364998966746,
        "ort_preprocess_to_restored_boxes": 2.144289499938168,
        "ort_raw_batch2": 0.04258399985701544
      },
      "ort_raw_batch2_images_per_second": 46965.99677614626,
      "versions": {
        "torch": "2.9.1+cpu",
        "onnx": "1.19.1",
        "onnxruntime": "1.23.2"
      },
      "limits": "CPU float32 parity; one training step is not detection-quality evidence; this CPU case does not run TensorRT/GPU; separate L4 evidence is documented"
    }
    real ONNX export + checker + ORT CPU + restored-box parity completed
    ```

<!-- curriculum-evidence:end -->

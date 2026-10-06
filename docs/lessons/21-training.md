# 21.4 ViT training：更新參數，也能從中途接回來

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.6.1/notebooks/21-training.ipynb){ .md-button }

[上一節](21-transformer.md)接好完整 TinyViT。**這個小模型真的能學會紅藍分類，並在存檔後接著同一次訓練嗎？** 本節把前面的模型一起更新，再測試未參與更新的圖，最後檢查 checkpoint 還原。

## 先分開三批材料

答案仍是紅矩形 0、藍矩形 1。每張圖是 32×32 RGB，矩形寬高、位置、亮度和背景雜訊的抽樣不依賴類別。下圖仍是 seed 101 的前四張訓練圖，沒有換成另一個任務。

![實際訓練資料前四張32乘32RGB圖；答案是紅0藍1，矩形的位置與尺寸不是答案](../assets/diagrams/21-materials.svg)

現在把生成器的亂數種子分成三個，建立三批不同的圖片：

|用途|張數|生成 seed|能否進 optimizer 更新|
|---|---:|---:|---|
|train，訓練|128|101|可以|
|val，驗證|64|202|不可以|
|test，測試|64|303|不可以|

訓練資料用來算梯度；驗證資料可用來觀察模型；測試資料用來在設定固定後評估。本次固定執行 60 次更新，沒有按驗證得分挑選 checkpoint。三批都由同一套顏色規則生成，因此測的是「同樣規則的新矩形」，不是自然照片、形狀辨認或物件定位。

每次從訓練資料抽 32 張，可重複抽到同一張。整批圖片是 `[32,3,32,32]`，答案是 `[32]` 的整數 0／1；模型輸出 `[32,2]` logits，順序紅、藍。交叉熵直接讀 logits 和類別答案，`argmax` 預測是否等於答案才用來計算分類正確率。

## 60 次更新，哪些部件一起學

模型仍是圖 32／patch 8／寬度 32／深度 2／4 heads／MLP 中間 64，共 **23,970 個可學參數**。本次啟用 dropout=0.1，也就是訓練時每個 dropout 部件隨機刪掉約一成輸入值並縮放保留值。圖片、模型和計算都在 CPU，以 float32 執行；初始權重與訓練抽樣的 seed 是 7。

optimizer 使用 **AdamW**，W 指 weight decay（權重衰減）：這是將權重衰減與梯度更新分開處理的 Adam 變體。本次的學習率為 0.003、weight decay 為 0.01。AdamW 保留過去梯度的移動平均，用來調整下一次更新；weight decay 在更新時讓權重稍微往 0 縮。這裡沒有學習率 scheduler，60 步都使用同一個學習率。它與第 1 章的 SGD／Adam 示範設定不同，所以不能把兩節分數當成 ViT 和 CNN 的公平比較。

這次不是只更新 logits：patch 投影、CLS、位置向量、兩個 blocks 的 LN／QKV／輸出投影／MLP，以及分類 head 都交給 optimizer。batch 抽樣與 dropout 會用到 RNG（Random Number Generator，亂數產生器），決定這次的隨機選擇。一次更新先用 `train()` 啟用 dropout，再算 loss、清梯度、反傳、檢查梯度，最後 `optimizer.step()` 更新參數。完整程式的前半段如下：

``` { .python data-excerpt="lesson_cases/21-training.py" }
def train_step(model, optimizer, dataset, batch_size):
    model.train()
    # batch抽樣與dropout都消耗torch RNG；只恢復權重不能重現下一步。
    indices = torch.randint(len(dataset), (batch_size,))
    loss = F.cross_entropy(model(dataset.images[indices]), dataset.labels[indices])
    if not torch.isfinite(loss):
        raise RuntimeError("training loss 非有限值")
    optimizer.zero_grad()
    loss.backward()
    parameters = dict(model.named_parameters())
    if not all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in parameters.values()):
        raise RuntimeError("有缺失或非有限的gradient")
    grad_norm = float(sum(parameter.grad.square().sum() for parameter in parameters.values()).sqrt())
    if not grad_norm > 0:
        raise RuntimeError("gradient全部為0")
```

這裡的 `grad_norm` 把所有參數梯度的平方相加再開根號，量整體梯度有多大。程式接著記錄幾組部件的梯度，才呼叫 `optimizer.step()`；不是算完 `backward()` 就宣稱完成學習。每次記錄的 loss 都在該次更新**之前**，並且有當次抽樣和 dropout 的影響。

## Loss 曲線與新圖，分開看

![實際60次更新的訓練batch交叉熵曲線；第30次更新後存checkpoint，曲線每點在當次更新前測量](../assets/diagrams/21-training-loss.svg)

曲線前半段有起伏，因為每次抽到的 batch 和 dropout 不同；後半段下降到接近 0。第 1 點是 0.730835，第 60 點是 0.008318。這兩點描述各自那批訓練圖的 loss，不能直接拿它們評分新圖。

為了比較同一批材料，程式另用 `eval()` 關閉 dropout，在固定完整訓練集上量更新前後的交叉熵，再以相同方式評估 val 和 test。正確率的分母是該批總張數；判準是兩個 logits 中較大的類別與 0／1 答案相同，不涉及 IoU、AP 或框。

|觀察時間／資料|交叉熵|分類答對數|
|---|---:|---:|
|更新前，固定 train|0.710059|64/128 = 50%|
|60 次更新後，固定 train|0.007155|128/128 = 100%|
|60 次更新後，val|0.007153|64/64 = 100%|
|60 次更新後，test|0.007154|64/64 = 100%|

更新時所有參數梯度都是有限值，每一步的整體梯度長度介於 **0.071033～15.675092**，沒有全為 0。程式也比對了 patch 投影、第一個 block 的 QKV 和 MLP、CLS、位置向量及分類 head 的更新前後副本，六組都確實改變。這些證據加上分類結果，支持「這個小模型在本次執行學會了這批顏色規則，並答對這兩批同規則的新圖」。

它還不能支持真實照片也有 100% 正確率、位置向量或 attention 是成功的唯一原因，或 ViT 比 CNN 好。這是一個 seed 的受控顏色任務，沒有架構對照；兩類共享矩形生成規則也不代表模型學會了辨認多種形狀。

## 第 30 步存檔，需要保留什麼

**Checkpoint** 是能還原訓練狀態的存檔。只有模型權重，可以重新推論；要接著原本那次訓練，還需要 optimizer 的梯度歷史、已完成步數，以及隨機數產生器的狀態。後者簡稱 **RNG state**，保存的是「亂數抽到哪裡」，不只是最初的 seed。

本例的 batch 抽樣與 dropout 都會使用 RNG。若只重新設 seed=7，亂數會回到開頭，不能接到第 30 步後的下一個 batch。模型設定和類別順序也要存下來，才知道該建立哪個模型，以及分數 0、1 各表示什麼。

![手工續訓流程：同一起點一路60步，或在30步保存模型optimizer與RNG，再建新物件還原並完成31至60步；最後比較狀態](../assets/diagrams/21-checkpoint-resume.svg)

圖中上路一路做 60 次更新；下路在第 30 次更新**後**存檔，重新建立模型和 optimizer，再載入，從第 31 次接著做。為了看出 optimizer 真的有還原，新 optimizer 故意先設學習率 0.9、weight decay 0.5；載入後應回到存檔的 0.003／0.01，而不是繼續用這組不同設定。

``` { .python data-excerpt="lesson_cases/21-training.py" }
    resumed_model = TinyViT(**MODEL_CONFIG)
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=.9, weight_decay=.5)
    restored = load_checkpoint(checkpoint_path, resumed_model, resumed_optimizer)
    restored_steps = optimizer_steps(resumed_optimizer)
    restored_learning_rate = resumed_optimizer.param_groups[0]["lr"]
    resumed_history = []
    for step in range(restored["steps_completed"] + 1, steps + 1):
        loss, _, _ = train_step(resumed_model, resumed_optimizer, splits["train"], batch_size)
        resumed_history.append({"step": step, "loss": loss})
```

`MODEL_CONFIG` 是前面的 TinyViT 設定，`load_checkpoint` 同時載入模型、optimizer 和 RNG。重新建立模型本身會消耗亂數，所以這個函式在完成還原後才恢復 RNG。兩路使用同一批固定訓練圖，沒有把驗證或測試圖放進更新。另一次重跑也可用原本三個 seed 重新生成這三批材料。

實測中，載入後的 optimizer 步數都是 30、學習率是 0.003。再更新 30 次到第 60 步，結果是：

|比較上路與下路|實際結果|
|---|---|
|模型參數最大絕對差|0|
|optimizer 狀態最大絕對差|0|
|第 31～60 步每一點 loss|完全相同|
|最後 RNG 狀態|完全相同|
|optimizer 已完成步數|兩路皆 60|

這驗證了**本次 CPU 程式和設定**能接回同一次訓練，包含抽樣與 dropout。它沒有測試跨硬體或不同 PyTorch 版本的逐位元重現，也不需要用秒數來判定成功。

## 小變化：只還原模型，保留新的 optimizer

假設仍載入第 30 步模型，卻不載入 optimizer 狀態，只用剛建立的 AdamW，其他設定與 RNG 都還原。可以用模型推論嗎？下一次更新保證和原來第 31 步相同嗎？

??? note "參考答案"

    可以用還原的權重推論，但下一次更新沒有相同保證。新的 optimizer 缺少前 30 步的梯度移動平均與步數；若還保留示範的學習率 0.9，也連學習率都不同。checkpoint 能讀取、shape 合法、推論能完成，和完整續訓一致是不同的檢查。

    若反過來還原 optimizer，卻不還原 RNG，下一個 batch 和 dropout 也可能不同。要核對續訓，應比完整模型、optimizer、接續 loss 與 RNG，不能只確認檔案存在。

??? note "重跑與查證"

    本機從專案根目錄執行 `PYTHONPATH=. .venv-model/bin/python lesson_cases/21-training.py`；在 Colab 執行本節完整程式。程式會建立本機 `artifacts/runs/vit-training/result.json` 和 `checkpoint.pt`。JSON 有全部 60 個 loss、梯度、資料 seed 和還原比較。課文的版本化實際輸出見下方紀錄；checkpoint 是執行產物，不是下載到的預訓練模型。

架構依據：[ViT 原論文 §3.1](https://arxiv.org/html/2010.11929#S3.SS1)；原論文的資料規模、訓練和微調安排在 [§4.1](https://arxiv.org/html/2010.11929#S4.SS1) 與 Appendix B.1。這裡直接在小型合成資料上訓練，沒有重現原論文的 ImageNet／JFT 預訓練或自然照片成績。

現在已經有一個能輸出整圖 CLS 與逐 patch 表示的小型 ViT。下一節保留它的讀圖路徑，換掉「靠紅／藍標籤學習」的訊號：同一張圖做成兩種視角，讓兩邊的表示對齊。

[上一節：21.3 完整 Transformer](21-transformer.md) · [下一節：22.1 DINO 的兩種視角](22-views.md) · [在 Colab 重做訓練](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.6.1/notebooks/21-training.ipynb)

<!-- curriculum-evidence:start -->

## 實際執行紀錄

本節的完整程式於 2026-10-06 在 AMD EPYC 9V74 80-Core Processor（2 個執行緒）上用 PyTorch 2.9.1+cpu 執行，程式裡的 assert 全部通過。下面是那次印出的原始輸出；輸出裡若有計時或訓練得到的數字，換一台電腦會略有不同。每個數字的意思，以本頁正文的說明為準。[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/21-training.json)

??? example "展開本次實際輸出"

    ```text
    {"event": "vit_training", "device": "cpu", "seed": 7, "model_config": {"image_size": 32, "patch_size": 8, "embed_dim": 32, "depth": 2, "num_heads": 4, "mlp_ratio": 2, "num_classes": 2, "dropout": 0.1}, "parameters": 23970, "optimizer": "AdamW", "learning_rate": 0.003, "weight_decay": 0.01, "steps": 60, "batch_size": 32, "split": {"train": {"samples": 128, "seed": 101}, "val": {"samples": 64, "seed": 202}, "test": {"samples": 64, "seed": 303}}, "class_names": ["red", "blue"], "answer_rule": "0=red rectangle; 1=blue rectangle; both classes share shape/position rules", "fixed_train_loss_before": 0.7100589871406555, "fixed_train_loss_after": 0.007154775783419609, "train_before": {"correct": 64, "count": 128, "accuracy": 0.5, "loss": 0.7100589871406555}, "train_after": {"correct": 128, "count": 128, "accuracy": 1.0, "loss": 0.007154775783419609}, "validation_before": {"correct": 32, "count": 64, "accuracy": 0.5, "loss": 0.7110880613327026}, "heldout": {"val": {"correct": 64, "count": 64, "accuracy": 1.0, "loss": 0.007152687758207321}, "test": {"correct": 64, "count": 64, "accuracy": 1.0, "loss": 0.007154297083616257}}, "loss_history": [{"step": 1, "loss": 0.7308351397514343}, {"step": 2, "loss": 0.855794370174408}, {"step": 3, "loss": 0.6613631844520569}, {"step": 4, "loss": 0.6734545826911926}, {"step": 5, "loss": 0.6836580634117126}, {"step": 6, "loss": 0.6668286919593811}, {"step": 7, "loss": 0.6884805560112}, {"step": 8, "loss": 0.7003495097160339}, {"step": 9, "loss": 0.6685369610786438}, {"step": 10, "loss": 0.6673541069030762}, {"step": 11, "loss": 0.6837713718414307}, {"step": 12, "loss": 0.642971932888031}, {"step": 13, "loss": 0.6408815383911133}, {"step": 14, "loss": 0.5884948372840881}, {"step": 15, "loss": 0.5772535800933838}, {"step": 16, "loss": 0.48671919107437134}, {"step": 17, "loss": 0.3998703360557556}, {"step": 18, "loss": 0.41314026713371277}, {"step": 19, "loss": 0.30822816491127014}, {"step": 20, "loss": 0.5322189331054688}, {"step": 21, "loss": 0.5027516484260559}, {"step": 22, "loss": 0.21528875827789307}, {"step": 23, "loss": 0.4430631995201111}, {"step": 24, "loss": 0.6368469595909119}, {"step": 25, "loss": 0.16455000638961792}, {"step": 26, "loss": 0.15104880928993225}, {"step": 27, "loss": 0.14399372041225433}, {"step": 28, "loss": 0.2785598635673523}, {"step": 29, "loss": 0.12201113998889923}, {"step": 30, "loss": 0.09960782527923584}, {"step": 31, "loss": 0.08667132258415222}, {"step": 32, "loss": 0.07601938396692276}, {"step": 3
    …（完整輸出見 notebook 與 JSON 紀錄）…
    4 80-Core Processor", "logical_cpus": 5, "threads": 2, "python": "3.12.14", "torch": "2.9.1+cpu", "torch_cpu_capability": "AVX512"}, "source_hashes": {"lesson_cases/21-training.py": "6c9593ba8ef5fb0e4a1f445f0ccd22ce237a5766d87682dea78af8f12ef44322", "miniyolo/__init__.py": "785b058b2b011124243b06ee59df8e59dfa75b77f050b897479e5be636763fc9", "miniyolo/checkpoint.py": "2144e2afa4d89382a7b3cd253755bb851e35fdc0847ee7179dee02dbbf4ca49b", "miniyolo/data.py": "cccad00e2c4f96eb6567eafc9e12248379c6b715fc1790d75518a253baa6181d", "miniyolo/geometry.py": "6a6b57d3493888e99dae4a012dab78127b8b543a0e01d8107963a3d6e63d8483", "miniyolo/inference.py": "995ac8f942d0c1e43d94f2efb3b7adcb91a9feb6b9bab923615209f0c190c313", "miniyolo/losses.py": "81fa9331c9a2aebe2e9c6c453a5313e64566bb20c3fe77ca45ec9df53424d4e4", "miniyolo/metrics.py": "53ce982e37cbd96c784f75c7d30faf99d52f79ab83ca7b8114eb21b4327330e0", "miniyolo/models.py": "49d029ea4ba2650ce8933cf97e3d25dc7aff2ca4e972eec19cdda17b0f4900e6", "miniyolo/provenance.py": "21769813c36b45f513c9f0d6125194f3baa5ae265278ffd44a796d298d124ed3", "miniyolo/targets.py": "2c8e32f2845b3bf970c77304c5cca0f999083d36a4d5af2f75f14aa89b823f16", "miniyolo/vision_data.py": "92b1cc3885a0b897eba2a24ac1cd604c55b5ea903a1886904e64b9364c67dde1", "miniyolo/vision_transformer.py": "212c8fcd59df0abd2ff9d0e545dbee051bbbe306089f5e0973d47783090fa394"}, "limitation": "Only this controlled color classification distribution; no natural-image, localization, or ViT-vs-CNN performance claim."}
    ```

<!-- curriculum-evidence:end -->

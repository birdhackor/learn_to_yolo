# 22.3 Teacher 怎麼提供答案，又不跟著 student 一起塌縮？

[在 Colab 執行本節](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.6.0/notebooks/22-distillation.ipynb){ .md-button }

[上一節](22-collapse.md)看到，一致的常數回答也能把簡單 loss 降到 0。現在仍用同一張矩形的兩個 view，讓一個模型讀 view 1，提供另一個模型讀 view 2 時要接近的分佈。本節要回答：**DINO 的答案怎麼產生、誰會被梯度更新，以及這些更新怎麼分開？**

## 兩份 ViT，先從相同權重開始

材料仍是 [22.1 節](22-views.md)的 32×32 RGB 矩形及兩個 view。準備結構相同、初始權重相同的兩份 ViT：student（學生）由 optimizer 更新；teacher（教師）提供學習目標，之後只追蹤 student 權重的移動平均。teacher 不是另外訓練好、已經知道紅藍答案的模型。

每份 ViT 都先得到 32 維 CLS 特徵；這是前文用來彙整圖片資訊的向量。現在用 **projection head（投影頭）**取代原紅藍分類頭的用途，把特徵變成 K 個原始分數，記作 logits `z`。本例是 `Linear(32,64)` → GELU → `Linear(64,16)`，把得到的 16 維向量除以自己的長度，再用無 bias 的 `Linear(16,16)` 產生 K=16 個分數；原分類頭仍保存在 backbone，但凍結且不使用。

softmax 把 K 個分數轉成加總為 1 的分佈。K 是投影頭輸出長度，**這 K 個槽沒有事先命名的紅、藍類別**；不能對第 0 槽直接說「這是紅色」。紅藍答案在此不進 loss。

![一個 DINO 更新步驟：舊中心產生固定 teacher 目標，student 走梯度，center 和 teacher 各走自己的 EMA](../assets/diagrams/22-distillation.svg)

先看圖中實線的前向路徑，再按 1、2、3、4追蹤更新次序。「停止梯度」表示目標只供比較，loss 不會沿這條路改 teacher。橘色箭頭更新 teacher 的參數，綠色箭頭更新輸出中心；兩條箭頭不改同一個量。

## 先用舊 center 和兩個溫度，算出這一步的答案

teacher 對一個 view 輸出 logits `z_t`。我們為 K 個槽各保留一個 center 值 `c`，先減掉**這一步開始前的 center**，再除以 teacher 溫度 `τ_t`，最後 softmax：

\[
p_t=\operatorname{softmax}\left((z_t-c)/\tau_t\right).
\]

center 是過去 teacher 原始輸出在各槽的移動平均，不是紅藍資料的平均顏色，也不是特徵向量的平均。某個槽若經常有很高的原始分數，減去它累積的 center 後，這個槽就比較難只靠固定偏高的分數一直占優勢。

以下只是一個 K=3 的手工例子，不是本節訓練輸出：若 `z_t=[0.20,0,0]`、舊 `c=[0.10,0.05,-0.05]`，先相減得到 `[0.10,-0.05,0.05]`。取 `τ_t=0.10`，softmax 的輸入是 `[1,-0.5,0.5]`，分佈約 `[0.5465,0.1220,0.3315]`。先減 center 再 softmax，不能拿 softmax 後的機率直接減 center。

student 的 logits `z_s` 不減這份 teacher center，使用自己的溫度：

\[
p_s=\operatorname{softmax}(z_s/\tau_s).
\]

溫度是正數，作用是縮放 logits 之間的差距。相同的兩個 logits `[0.2,0]`，溫度 0.2 得到 softmax(`[1,0]`)≈`[0.7311,0.2689]`；溫度 0.1 則得 softmax(`[2,0]`)≈`[0.8808,0.1192]`。溫度較小，分佈就更偏向高分槽，叫 sharpening（銳化）。DINO 通常讓 teacher 比 student 更尖銳，避免 teacher 的所有答案都接近平均分佈。

中心化壓制某個槽長期獨占；銳化避免全部槽都平均。它們的作用方向不同，需要配合。這個機制分析不保證任意資料、任意溫度和步數都能學出有用特徵；我們的小實驗仍要在[下一節](22-features.md)檢查凍結特徵。

## 只讓 student 朝跨 view 的目標移動

這一張圖有 view 1、view 2，各送入 student 和 teacher，得到四個分佈。用 teacher 的 view 1 來教 student 的 view 2，再用 teacher 的 view 2 來教 student 的 view 1；不拿同一個 view 的 teacher／student 配對。兩個方向的 loss 取平均，batch 中各張圖也取平均。

對其中一個方向，用交叉熵：

\[
H(p_t,p_s)=-\sum_{k=1}^{K}p_{t,k}\log p_{s,k}.
\]

`k` 是第幾個輸出槽，`p_t` 是固定的 teacher 目標，`p_s` 是 student 回答。teacher 覺得某個槽比例高時，student 把那個槽的比例提高，這項 loss 通常會變小。這裡是軟答案，不用 `argmax` 先選一個槽，再當成紅藍標籤。

計算 teacher 輸出時停止梯度，程式使用不記錄 teacher 計算圖的區塊；teacher 參數也不交給 optimizer。`loss.backward()` 只為 student 的 backbone 與投影頭計算梯度，`optimizer.step()` 只更新這些參數。student 的梯度有限且非零、更新前後權重有差，才能證明這一步真的有學習更新。

## 再更新 center 和 teacher，供下一步使用

本步的目標已經用舊 center 算完，現在才把本 batch 的 **raw teacher logits** 在圖片與兩個 view 間取平均，記作 `mean(z_t)`，更新 center：

\[
c_{\mathrm{new}}=m_c c_{\mathrm{old}}+(1-m_c)\operatorname{mean}(z_t).
\]

`m_c` 是中心的動量。這種「舊值留大部分，新值加少部分」叫 EMA（exponential moving average，指數移動平均）。手工例子：某個槽舊 center 為 0.1、本 batch 的 raw logits 平均為 0.3，取 `m_c=0.9`，新 center 為 `0.9×0.1+0.1×0.3=0.12`。這個 0.12 到下一步才拿來產生目標，不能回頭替換本步的 0.1。

student 做完 optimizer 更新後，再把 teacher 的每個參數 `θ_t` 朝新 student 參數 `θ_s` 移動一點：

\[
\theta_{t,\mathrm{new}}=m_t\theta_{t,\mathrm{old}}+(1-m_t)\theta_{s,\mathrm{new}}.
\]

`m_t` 是 teacher 權重的動量，和 `m_c` 是不同的設定。例如一個 teacher 權重為 2，新 student 權重為 4，取 `m_t=0.9`，teacher 變為 2.2。這一步沒有 backward。teacher 因而追蹤 student 的歷史，而不是每一步立即複製 student；但這種追蹤本身仍不是防塌縮保證。

現在可以把一整步按順序說完：取兩個 view → teacher 用舊 center 產生並固定目標 → 算跨 view loss → 更新 student → 更新 center 與 EMA teacher，供下一步使用。center 更新資料是 teacher 的原始輸出，teacher 更新資料是 student 的參數；這是圖中兩條更新箭頭分開畫的原因。

## 跑一次真更新，並檢查能否接續

本節完整程式在 CPU 上跑小型自監督訓練，使用 seed 101 生成的 128 張 train 圖，不下載權重，也不要求上一節的 checkpoint。每步抽 24 張圖，只以 AdamW 更新 student 的 backbone 與投影頭；學習率固定 0.0005，weight decay=0.01，梯度範數最大截為 3：把所有參數梯度的平方相加再開根號，若長度超過 3，就把全部梯度乘上約「3／原長度」的共同比例，縮小長度而保留方向；長度不超過 3 就不縮放。紀錄的 `student_grad_norm` 是裁剪前的長度，所以第一步的 24.1192 可以大於 3；optimizer 使用的是縮放後的梯度。兩個溫度固定為 `τ_s=0.15`、`τ_t=0.08`，兩個動量固定為 `m_t=0.95`、`m_c=0.9`。這些是小模型的設定，前面的 K=3、溫度 0.1 與 EMA 0.9 是另外標明的手工例子。

下面是實作一個更新步驟的核心：

``` { .python data-excerpt="miniyolo/self_distillation.py" }
        self.optimizer.step()
        update_teacher(self.student, self.teacher, self.config.teacher_momentum)
        self.loss.update_center(teacher_outputs)  # loss 已算完，才更新給下一步
        self.step += 1
```

實作在本步 loss 和 student 更新完成後，先做 teacher EMA，再更新 center。center 使用的仍是本步前向時已算好的 raw teacher outputs，不會重新用 EMA 後的 teacher 產生本步 logits。center 和 teacher 都只影響下一步目標。

執行 `PYTHONPATH=. python lesson_cases/22-distillation.py` 或頁首 notebook。本次 160 步的第一步／最後一步 loss 為 1.943／1.874；student 的 backbone 權重有變，teacher 沒有梯度，並保存了逐步梯度範數。這些檢查證明更新路徑成立，並不等於特徵可用。下一節會畫出完整曲線並檢查下游用途。

checkpoint 要保存的不只有 teacher backbone：student、teacher、optimizer 狀態、center、已完成 step、整份 schedule 設定，以及產生 batch／view 的 RNG（亂數產生器）狀態都會影響下一步。schedule 是「第幾步用哪個溫度、動量與學習率」的安排；若接續時把步數重設為 0，便改變了訓練。

程式另比較同種子下的「連續訓練到 160 步」與「第 80 步儲存、重建、讀回再接續 80 步」。本次實測兩路的第 81～160 步 loss 完全相同，最後 student、teacher、center 與 optimizer 狀態完全相同，下一組隨機 view 也完全相同。這是本 CPU 小模型的接續測試，不表示跨硬體或改變 PyTorch 版本也能逐位相等。若只想取特徵，載入 teacher backbone 足夠；若想精確接續，則需要完整狀態。

## 改一件事，先預測

在前面的 teacher 權重手算中，把 `m_t` 從 0.9 改成 1，teacher 權重會到多少？若整段訓練始終取 1，teacher 還會追蹤 student 嗎？

??? note "參考答案"

    權重仍為 2：`1×2+0×4=2`。一直取 1，就一直保留初始 teacher，不吸收 student 的更新。動量越大，更新越慢；它不是「越大一定越好」。

??? note "本節和原始 DINO 的範圍"

    這裡教的是 2021 年自監督 **DINO（self-distillation with no labels）**，來源為 [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294)，不是 2022 年同名的 [DINO 物件偵測器](https://arxiv.org/abs/2203.03605)。原始 DINO 的 §3.1、公式 1–4 與 Algorithm 1 支持 student／teacher、交叉熵、停止梯度、center 和 EMA；§5.3 分析中心化與銳化。

    原始 DINO 使用兩個 global view 加上較小的 local views（multi-crop），teacher 只讀 global views，student 讀全部 views；本節只有兩個 global view。也縮小了 ViT、簡化投影頭（包括沒有原版最後層的 weight normalization）、資料量，並以固定溫度／動量／學習率取代原版 warmup 與 cosine schedule。因此本節是完整小模型的教學訓練，並非重現原論文的 ImageNet 實驗。

    固定官方版本：[main_dino.py](https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/main_dino.py)。`DINOLoss.forward` 先用舊 center 算 teacher 分佈並 `detach`，跳過同 view 的配對，最後呼叫 `update_center`；`update_center` 平均的是 raw logits。`train_one_epoch` 在 student optimizer step 後以 EMA 更新 teacher。這些順序支持本文的機制說明；本節參數值以我們的程式與執行紀錄為準，不混用論文設定和官方不同版本的預設值。

[上一節：22.2 常數回答的漏洞](22-collapse.md) · [下一節：22.4 凍結特徵後評分](22-features.md)

<!-- curriculum-evidence:start -->

## 實際執行紀錄

本節的完整程式於 2026-10-06 在 AMD EPYC 9V74 80-Core Processor（2 個執行緒）上用 PyTorch 2.9.1+cpu 執行，程式裡的 assert 全部通過。下面是那次印出的原始輸出；輸出裡若有計時或訓練得到的數字，換一台電腦會略有不同。每個數字的意思，以本頁正文的說明為準。[完整紀錄（JSON）](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/22-distillation.json)

??? example "展開本次實際輸出"

    ```text
    {
      "seed": 7,
      "device": "cpu",
      "steps": 160,
      "checkpoint_step": 80,
      "train_samples": 128,
      "batch_size": 24,
      "backbone": {
        "image_size": 32,
        "patch_size": 8,
        "embed_dim": 32,
        "depth": 2,
        "num_heads": 4,
        "mlp_ratio": 2,
        "num_classes": 2,
        "dropout": 0.0
      },
      "training_config": {
        "output_dim": 16,
        "hidden_dim": 64,
        "bottleneck_dim": 16,
        "batch_size": 24,
        "learning_rate": 0.0005,
        "student_temperature": 0.15,
        "teacher_temperature": 0.08,
        "teacher_momentum": 0.95,
        "center_momentum": 0.9,
        "global_scale": [
          0.65,
          1.0
        ]
      },
      "first_step": {
        "step": 1,
        "loss": 1.9427120685577393,
        "teacher_entropy": 1.7583107948303223,
        "student_grad_norm": 24.119152069091797
      },
      "last_step": {
        "step": 160,
        "loss": 1.8742694854736328,
        "teacher_entropy": 1.8415940999984741,
        "student_grad_norm": 0.9462160468101501
      },
      "backbone_weights_changed": true,
      "teacher_has_no_grad": true,
      "resume": {
        "next_random_views_identical": true,
        "student_teacher_optimizer_center_identical": true,
        "loss_history_identical": true,
        "final_step": 160
      },
      "diagnostics_train": {
        "feature_std": 0.32309943437576294,
        "normalized_feature_std": 0.05657871067523956,
        "mean_pair_cosine": 0.8623802661895752,
        "mean_output_entropy": 1.7878104448318481,
        "marginal_output_entropy": 2.2156848907470703
      },
      "elapsed_training_and_resume_seconds": 4.2405333849983435,
      "checkpoint_path": "artifacts/runs/dino/22-distillation-midpoint.pt",
      "simplifications": "2 global views; ordinary MLP and no weight normalization; fixed temperatures/LR/EMA; no multi-crop or official schedules",
      "limitation": "DINO2021 mechanism demonstration from scratch; loss/diagnostics are not downstream or natural-image capability proof"
    }
    actual history: artifacts/runs/dino/22-distillation.json
    ```

<!-- curriculum-evidence:end -->

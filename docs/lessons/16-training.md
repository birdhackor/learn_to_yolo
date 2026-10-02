# 16.3 YOLO26 訓練補強：先把已查證機制分開

[開啟 Colab](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.3.0/notebooks/16-training.ipynb) · 原始碼：`lesson_cases/16-training.py`

前置是dual head、loss權重和SGD。推論用one-to-one head，不代表整個訓練期間應以相同權重分配兩條路；小物件也可能因候選格太疏而沒有正訊號。官方YOLO26介紹Progressive Loss、STAL和MuSGD，本節逐項界定已公開可核對的行為，再選Progressive權重作主要對照。不要看到三個名稱就一次全加入，否則不知道結果由誰造成。

歷史機制以查覈日的官方source為準；發布checkpoint還有預訓練、增強、超參數與內部設定，不能由一段小實驗聲稱完整復現。本次起點是兩個線性head，主要改動僅為loss權重排程；沒有執行MuSGD，STAL只做一個資格mask數值檢查，兩者都不混入權重對照的結果。

## Progressive Loss：總loss往推論分支移動

one-to-many是每個真值物件可監督多個候選；one-to-one則每個真值最多選一個正候選。早期many分支提供較密集的學習訊號，後期加重NMS-free推論模式所用的one分支，是這項排程的動機。代價是後期留給many的監督份量減少，而且需要選擇轉移速度與終點；不能假設一種排程適合所有資料。

查覈版本`E2ELoss`初始化many權重.8、one權重.2；many最後降到.1，one升到.9，使用線性衰減。本例將epoch索引定為0至29，計算：

`a(e)=max(1−e/(E−1),0)×(.8−.1)+.1`，`b(e)=1−a(e)`。

所以第0回合`.8/.2`，中間約`.438/.562`，第29回合`.1/.9`。這裡的權重改變loss和梯度中的相對份量。對one head參數θ，有`∂L/∂θ=b×∂Lone/∂θ`；同一個未加權梯度乘上較大的b，更新係數就較大，但訓練中的未加權梯度也會變，不能說後期每一步實際更新必然更大。

```python
optimizer.zero_grad()  # 清除上一輪累積的梯度
many_prediction, one_prediction = many(features), one(features)
many_mse = F.mse_loss(many_prediction, target)
one_mse = F.mse_loss(one_prediction, target)
many_weight, one_weight = weights(epoch, epochs, final_many=.1)
loss = many_weight * many_mse + one_weight * one_mse
loss.backward()
optimizer.step()
```

輸入features`[4,1]=[-1,0,1,2]`，target為`2x+1=[-1,1,3,5]`，兩個`Linear(1,1)`輸出也為`[4,1]`。一條對照固定`.8/.2`，另一條採排程；兩者每次重設seed7，資料、初始權重、30次SGD及lr.05完全相同。兩個線性分支互不影響，使用完全相同target，沒有不同assignment或shared backbone。對這個純SGD小例，one的有效learning rate相當於`.05×b`，由.01增至.045；實驗只展示這個權重作用，不能把差距當成YOLO26的AP增益。

seed7第一步可逐值核對：

| one分支計算 | 數值 |
| --- | --- |
| 初始w／bias | .318423／.313781 |
| 四筆forward `wx+bias` | −.004643、.313781、.632204、.950627 |
| mean squared error | `mean((prediction−[-1,1,3,5])²)=5.866377` |
| 已乘b=.2的dw／dbias | −1.146190／−.610803 |
| SGD更新後w／bias | .375733／.344321 |

b是給one loss調整份量的係數。MSE由平方來，所以導數有2；四筆平均除4；再乘one權重.2：`dw=.2×(2/4)×sum((prediction−target)×x)=−1.146190`，`db=.2×(2/4)×sum(prediction−target)=−.610803`。例如`w_new=.318423−.05×(−1.146190)=.375733`。案例印出並assert這份首步紀錄；最後報告的MSE則是30次更新完成後，重新forward得到，不是沿用更新前loss。

執行`PYTHONPATH=. python lesson_cases/16-training.py`，會核對首末權重，並印出one-head MSE。此固定小例中progressive較低，assert檢查此觀察；換資料、步數或learning rate不保證仍勝出。它支援「分支監督預算有可觀察作用」，不支援「所有場景都該使用這個排程」。

## STAL：擴張候選資格，不放大真值框

GT是標註真值，ranking是按預測品質排序。下面四點只是進入候選池，後續top-k選擇與衝突解決未在本例執行。

![原真值、候選資格框與四個stride8格心](../assets/diagrams/16-stal-candidates.svg)

在查覈版本中，`TaskAlignedAssigner.select_candidates_in_gts`把小於某stride門檻的GT邊長，僅在選候選時以中心為軸擴張。預設尺度`[8,16,32]`的門檻取第二項16；之後仍有品質ranking、top-k、衝突及one-to-one二次選擇，不是所有進入擴張框的點都成正樣本。

[YOLO26原論文](https://arxiv.org/abs/2606.03748)的STAL公式(5)寫的是邊長小於8才替換成16；本節所固定的官方程式碼使用小於16的門檻。兩者範圍不同，這裡以固定程式碼為準。本例邊長2同時滿足兩者，0→4的數字不受這個差異影響。

數字例子：真值`[7,7,9,9]`僅2×2畫素，中心`(8,8)`，stride8格心`(4,4),(12,4),(4,12),(12,12)`全部不在真值內。候選資格框邊長擴到16，變成`[0,0,16,16]`，四點都有資格。程式assert0→4，同時確認原GT仍`[7,7,9,9]`。回歸真值不能被換成16×16，否則會教模型畫大框。

收益是小物件較容易取得候選池，代價是框外點參與、品質排序和衝突更重要，也可能增加錯誤正訊號。只有資格增加，不代表四個點全被分配，也不保證每個小物件最後都有正樣本。本節沒有用STAL訓練detector，因此沒有小物件AP結論。

## 延伸查證（可跳讀）：MuSGD

主例不需要實作這個優化器。它除了沿梯度更新，也整理矩陣權重的更新方向；此處只解釋與SGD的差別。

momentum是保留前幾步梯度的加權記憶，讓更新不只取決於當下的一次梯度。MuSGD的Muon部分對矩陣參數的更新方向與尺度做整理，再和SGD動量更新組合，目的在改變多維權重的更新幾何。2D線性權重與4D卷積可reshape為矩陣；其他參數組可使用純SGD。官方用矩陣近似運算（Newton–Schulz）與另一份動量狀態實作，分組、混合比例和權重衰減也影響結果。

額外的矩陣計算、動量狀態、分組與混合設定會增加運算、記憶體和調參成本。是否換得更好收斂，要另做配對實驗。本節沒有執行MuSGD，不能從兩個線性head的MSE宣稱這個優化器較好。

這些細節已可讀到，但本章不自行拼出未驗證的MuSGD縮版，也不聲稱重現checkpoint recipe。官方訓練指南還指出發布模型有Objects365預訓練再COCO微調，部分內部超參數需實驗分支。正式比較最佳化器時，要固定資料、初始化、步數、增強和排程，再記錄收斂、參數更新尺度與成本；把預訓練模型與從零MiniYOLO比最終AP不能歸因於最佳化器。

常見錯誤是把真實共享模型的progressive權重一概當成單一lr、同時改三項卻只報總loss、將STAL擴張框當annotation，以及只做一次SGD便宣稱驗證MuSGD。自主練習：只改`main()`的`final_many=.1`為.3。答案末期one為.7，首期仍.2，首步數值也不變；排程、終點assert與印出結果會使用同一變數。這保留更多many監督預算，效果要重跑配對比較。再問STAL四點進池後one-head最多取幾個：本例未做ranking；若一對一top-1，每GT至多一個，不能直接回答四個。

來源查覈：2026-10-02。[E2ELoss排程](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/utils/loss.py)、[小物件資格與topk2](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/utils/tal.py)、[MuSGD公開實作](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/optim/muon.py)、[官方訓練recipe與重現範圍](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/docs/en/guides/yolo26-training-recipe.md)。

<!-- curriculum-evidence:start -->

## 本輪實際執行紀錄

本節範例已於 2026-10-02 使用 PyTorch 2.9.1+cpu 在 CPU 執行，程式中的斷言全部通過。以下是該次輸出；人工輸入、短步更新與模型效果的意義仍依本頁說明區分。[完整紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/16-training.json)

??? example "展開本次實際輸出"

    ```text
    many/one weight first: (0.8, 0.2)
    many/one weight last: (0.1, 0.9)
    one-head MSE fixed=0.663073, progressive=0.016285
    one-head first forward/backward/step: {"weight": 0.31842339038848877, "bias": 0.3137805461883545, "prediction": [-0.004642844200134277, 0.3137805461883545, 0.6322039365768433, 0.950627326965332], "one_mse": 5.866377353668213, "one_gain": 0.19999999999999996, "weighted_dw": -1.1461899280548096, "weighted_dbias": -0.6108031272888184, "updated_weight": 0.3757328987121582, "updated_bias": 0.34432071447372437}
    small-object eligible points original / expanded: 0 4
    GT regression box remains: [7.0, 7.0, 9.0, 9.0]
    not a full YOLO26/STAL/MuSGD training reproduction
    ```

<!-- curriculum-evidence:end -->

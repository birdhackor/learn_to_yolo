# 13.1 YOLOv10 dual assignment：訓練時多教，推論時少重複

[開啟 Colab](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.1.0/notebooks/13-dual-assignment.ipynb) · 原始碼：`lesson_cases/13-dual-assignment.py`

前置是 sample assignment、分類 loss 和 decoupled head。第 12 章讓一個 GT 監督多個候選，提供較密集的學習訊號；這些候選卻可能都在推論時報出同一物件。如果每個 GT 只教一個候選，重複較容易被壓低，但可用正訊號也變少。YOLOv10 的 dual assignment 在訓練保留兩種分支，推論使用一對一分支。

歷史機制是 consistent dual assignments：一對多和一對一使用相容的品質排序，減少兩套監督方向不一致。本次起點是同一組共享特徵；簡化成兩個分類 head、三個候選、兩個 GT。為清楚看「全域一對一」約束，實驗另外用列舉求最優配對；**這不是宣稱 YOLOv10 使用 Hungarian algorithm**。官方 YOLOv10 的一對一 assignment 採 task-aligned top-1 型選擇。本例不重現官方完整 loss 或框回歸。

## 一對多不等於一個候選有多份 target

品質表 shape `[G,P]=[2,3]`，每列一個真值，每欄一個候選，數字越大越適合負責：

| GT／candidate | p0 | p1 | p2 |
| --- | --- | --- | --- |
| A | .90 | .85 | .10 |
| B | .88 | .10 | .20 |

一對多先讓 A 選 p0、p1，B 選 p0、p2；p0 衝突後交給品質較高的 A，結果 owner `[0,0,1]`。一個 GT 有多個正候選，一個候選最後仍只擁有一個 GT。GT A 的兩個候選都被鼓勵報出 A，這就是推論重複的來源之一。

本次一對一列舉所有不同欄配對。把 A 給 p1、B 給 p0，總品質為 `.85+.88=1.73`，owner `[1,0,-1]`。未使用的 p2 學背景。若貪心先把最大 .90 的 p0 給 A，B 只能選 p2 的 .20，總和為 1.10。這是可驗證的反例：逐次拿當前最大值不是全域最佳。列舉是極小案例的精確解，不是把貪心更名為 Hungarian。

```python
best = max(permutations(range(P), G),
           key=lambda cols: sum(quality[g, c] for g, c in enumerate(cols)))
```

這種搜尋有 `P!/(P−G)!` 種選擇，G=2、P=3 只有6種；G、P稍大便不能這樣做。一般需要真正的最優指派 solver。官方 top-1 與本例全域最優的計算、正樣本覆蓋不同，讀者不能把本例 owner 當成官方模型必然輸出。

## 兩個 head 與一條共享路徑

輸入 `[3,4]` 每列一個候選的四維特徵，backbone 線性層後仍為 `[3,4]`；兩個 head 都輸出 `[3,2]`，兩個 GT 恰好是兩類。owner 轉成 two-class target：一對多是 `[[1,0],[1,0],[0,1]]`，一對一是 `[[0,1],[1,0],[0,0]]`。

```python
features = backbone(inputs)
many_logits = many_head(features)
one_logits = one_head(features.detach())
loss = many_bce + one_bce
```

官方 YOLOv10 路徑對一對一分支的特徵做 detach：該分支仍訓練自己的 head，卻不直接把梯度送到共享 backbone。程式先只 backward 一對一 loss，assert backbone `.grad is None`、one head 梯度非零；再反傳總 loss，backbone 由一對多分支得到梯度，一次 step 更新兩個 head。detach 不能放在 head 輸出後，否則連 head 都學不到。

執行 `PYTHONPATH=. python lesson_cases/13-dual-assignment.py`，應得到 many owner `[0,0,1]`、global one owner `[1,0,-1]`、品質 1.73 對 1.10，以及梯度路徑確認。這是可獨立執行的 CPU 監督實驗，沒有宣稱 NMS-free AP 或推論速度。

## 收益、代價與留下的問題

一對多提供較多正樣本供共享特徵學習，一對一分支學會讓較少候選取得高分；推論若只保留後者，可以減少需要消除的重複。代價是訓練 head、assignment 與 loss 增加，特徵 detach 的位置也成為契約。兩套品質規則若差很大，可能選出互相矛盾的負責人；所以歷史設計強調 consistent 的度量。

常見錯誤是「只用一對一 target」卻叫 dual、兩分支共享同一個最後 head 卻給矛盾 labels、把一對一誤認成整張圖只准一個物件，以及把訓練 assignment 和 inference top-k 混用。是否已成功 NMS-free，必須進一步檢查held-out重複與漏檢，下一節先用人工已知框驗證原因。

自主練習：把B的p2品質改為.95。答案為全域配對A→p0、B→p2，總約1.85；one_owner要同步改為`[0,-1,1]`，many_owner仍是`[0,0,1]`。原先「optimal嚴格大於greedy」assert也要改為`abs(optimum-greedy_value)<1e-6`；兩者用同樣Python float求和，不拿float32／float64末位差異當提升。原先非最佳的貪心在這張表變得最佳，說明一個成功案例不能證明貪心總是正確。再把GT數改為4、候選仍3，應停止並重新定義允許unmatched GT的問題，不能用permutations悄悄丟掉一個物件。

來源查覈：2026-10-02。[YOLOv10 論文](https://arxiv.org/abs/2405.14458)、[作者官方 repository，固定 commit](https://github.com/THU-MIG/yolov10/tree/453c6e38a51e9d1d5a2aa5fb7f1014a711913397)、[雙分支與 detach 實作](https://github.com/THU-MIG/yolov10/blob/453c6e38a51e9d1d5a2aa5fb7f1014a711913397/ultralytics/nn/modules/head.py)。

# 16.2 YOLO26 推論 head：訓練用分支怎麼離開部署圖

[開啟 Colab](https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/lessons-v0.2.0/notebooks/16-inference-head.ipynb) · 原始碼：`lesson_cases/16-inference-head.py`

前置是dual assignment、DFL-free與模型的train/eval模式。訓練時有兩個head，不代表部署也要執行兩個。要省掉輔助分支，必須確認推論取的是正確權重、解碼單位正確，而且模型真的不再攜帶那條路。本節建立一份只有推論head的module，逐值比對保留前後的raw輸出。

歷史機制已核對YOLO26官方`end2end:True`、訓練雙分支，以及`fuse()`移除未使用偵測分支。固定版本的官方預測入口預設使用many分支加NMS；選擇`nms=False`的NMS-free模式時才取one-to-one分支，見[官方訓練說明](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/docs/en/guides/yolo26-training-recipe.md)。本節選擇後者，起點是雙head小CNN，簡化為單尺度、兩類、16候選和單標籤top-3。本例的softplus距離、簡單卷積與人工平方loss不是官方YOLO26，也沒有訓練出可用物件偵測器。實驗只驗證這條推論路徑契約。

one-to-many指每個真值物件被分配多個正候選，提供較密集的訓練訊號；one-to-one指每物件分配一個正候選，讓推論分支學習減少重複。名稱描述監督關係，不是整張圖只准輸出一框。官方的NMS-free部署模式使用one分支，再以top-k選輸出，省略NMS的兩框IoU抑制；前提是該分支已學好適合的分數。案例只是兩個獨立卷積head，平方loss沒有實作任何assignment，也沒有學到唯一性。`detach()`只控制梯度路徑，不會把隨機框自動變成一物件一框。

[原論文](https://arxiv.org/abs/2606.03748)§3.2.1把one-to-one標為default；上段則描述本書固定commit的高階預測API。論文設計與這個API的預設選項要分開看，實際部署應明確設定要走哪條路。

softplus(raw)=ln(1+exp(raw))把任意raw轉成正距離。人工fixture的`log(expm1(distance))`只是倒算，確保softplus後是指定距離，並非模型學得的結果；不熟反函數時先看指定距離與decode即可。top-k的ids為[B,k]，展成[B,k,4]後用gather沿候選軸取每框四座標。top5練習同時改topk=5与shape斷言[B,5,4]／[B,5]，不是沿座標軸挑五項。

## 兩種介面不能混讀

輸入`[B,3,64,64]`，小backbone產生`[B,8,4,4]`。train module的many和one各輸出`[B,6,4,4]`，六channel依序為`[l,t,r,b,class0,class1]`；前四項raw經softplus後纔是特徵格單位的四邊距離。train回傳dict`{'many':..., 'one':...}`，deploy真正回傳的是`(top_boxes,top_scores,top_labels)`三個tensor：

| 部位 | shape，B=2 | 單位 | 是否為deploy回傳值 |
| --- | --- | --- | --- |
| raw候選 | `[2,16,6]` | 四邊raw＋兩類logits | 否，forward內部中間值 |
| decoded boxes | `[2,16,4]` | 64×64輸入的pixel xyxy | 否，forward內部中間值 |
| top-3 boxes | `[2,3,4]` | 依分數取前三個候選 | 是，第一項 |
| top-3 scores／labels | `[2,3]`／`[2,3]` | sigmoid分類分數／整數類別 | 是，第二／三項 |

raw一致性檢查是在解碼前，另外呼叫`deploy.head(deploy.backbone(image))`，取得尚未flatten的`[B,6,4,4]`；它不是從三個top-k回傳值中取raw。

train先以many與one raw平方mean做一次backward和step。這是通路驗證，target不是物件標註，不能據此聲稱NMS-free偵測成功。one讀`features.detach()`，訓練其自身head；many訓練backbone。部署時不需要detach，它不改forward數值。

![訓練保留雙head，部署只帶backbone與one](../assets/diagrams/16-head-paths.svg)

## 真正移除而不是隻是忽略

`DeployToy`複製已更新的backbone與one head，沒有many欄位。`deepcopy`避免兩份module共享同一個可被後續改動的參數物件。eval/no_grad下將train的one raw輸出與deploy的raw輸出比較，assert逐值相等至`1e-7`。

```python
self.backbone = copy.deepcopy(trained.backbone)
self.head = copy.deepcopy(trained.one)
reference = model(image)['one']
fused = deploy.head(deploy.backbone(image))
assert torch.allclose(reference, fused, atol=1e-7)
```

本例backbone224參數，每個`8→6`的1×1head54參數；train總332、deploy278。移除的54是這個小模型的輔助head，不能直接拿這個百分比預測YOLO26全模型收益。程式也檢查deploy參數名稱沒有many，防止「輸出不用它，但檔案仍帶著它」。

## 從16個位置到前三個

4×4候選以stride16產生中心`8,24,40,56`畫素。DFL通常每邊輸出多個bins，softmax後取期待距離；DFL-free每邊只直接輸出一個值。本例再用softplus限制為正，是教學加工，官方reg_max1可直接回歸帶符號值。距離乘16再由點加減，得到pixel xyxy。分類使用獨立sigmoid，沒有另外objectness，所以不能沿用早期grid detector的`obj×softmax(class)`分數公式。

案例先驗證一組人工raw，再看隨機head：點`(24,40)`、stride16、經softplus後距離`[1,.5,2,1.5]`，四邊解碼為`[24−16,40−8,24+32,40+24]=[8,32,56,64]`。類別logits`[0,ln3]`經sigmoid為`[.5,.75]`，應取label1、score.75。程式用softplus反函數建立對應raw，逐值assert框、分數與label，驗證單位和channel順序，不只驗證shape。

本例先每候選取最大類別分數，再top-3，保證同一候選只輸出一個類別。官方head的普通多類路徑可在候選與類別兩階段top-k，與本例不完全相同。兩者都不依兩框IoU刪除框；這裡沒有NMS，卻仍有分數選擇與輸出數上限。

執行`PYTHONPATH=. python lesson_cases/16-inference-head.py`，核對train keys、top-3 shapes、332／278、raw相等和沒有many參數。top-3一定產生三個候選，即使它們都是背景；實際產品還需score filtering與失敗樣本檢查。這段隨機小模型不應拿來做物件數量判斷。

## 收益、代價與部署檢查

部署module只保留必要分支，減少權重與forward操作，也讓匯出介面明確。代價是多一個轉換步驟、輸入尺寸和candidate生成的契約要固定；若複製錯head，程式仍可執行卻行為錯誤。原版fuse與卷積／BN融合也是不同事項，不能只看到方法名就認定每個運算元都已融合。

常見錯誤是將`.eval()`當成自動刪除many、複製未更新的初始one權重、部署仍使用訓練dict解碼、把64×64的固定stride搬到任意尺寸，以及對sigmoid分數再乘一次不存在的objectness。第20章會用真正ONNX Runtime比對，這節先把模型輸出介面穩定下來。

自主練習：把top-3改成top-5。答案為boxes`[2,5,4]`、scores與labels`[2,5]`，權重數不變；需要同步修改assert。若輸入改128×128而仍adaptivepool4，stride應是32，不能仍以16解碼。本例固定64，修改尺寸時先重算points和stride，而不只是讓Conv接受輸入。

來源查覈：2026-10-02。[YOLO26配置的end2end與reg_max](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/cfg/models/26/yolo26.yaml)、[Detect forward、postprocess、get_topk_index及fuse](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/nn/modules/head.py)。



<!-- curriculum-evidence:start -->

## 本輪實際執行紀錄

本節範例已於 2026-10-02 使用 PyTorch 2.9.1+cpu 在 CPU 執行，程式中的斷言全部通過。以下是該次輸出；人工輸入、短步更新與模型效果的意義仍依本頁說明區分。[完整紀錄](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/16-inference-head.json)

??? example "展開本次實際輸出"

    ```text
    manual ltrb decode box: [[8.0, 32.0, 56.0, 64.0]]
    manual class probabilities: [[0.5, 0.7500000596046448]] ; winner label=1, score=.75
    training output keys: ['many', 'one']
    deploy boxes / scores / labels: (2, 3, 4) (2, 3) (2, 3)
    parameters training / deploy: 332 278
    retained one-head raw output equals reference: verified
    top-k has no pairwise IoU/NMS; random toy predictions are not accuracy evidence
    ```

<!-- curriculum-evidence:end -->

# MiniYOLO 大綱的讀者審查與公開學習回饋

查核與修訂日期：2026-10-02。對應修訂：[大綱第二版](./outline.md)。前一輪官方教材與課程範圍見 [研究紀錄](./course-research.md)。

## 本輪做了什麼

六個 subagent 分工：兩位只讀獨立大綱，以不了解專案的讀者視角審查；四位搜尋 D2L／Stanford、DeepLearning.AI／YOLO 教學 repo、fast.ai／Harvard，以及中文自學紀錄。研究代理直接讀可取得的論壇貼文、GitHub issue／回覆、README 與 notebook 個人心得段落。

兩份讀者審查是**模擬意見**。網路來源是**公開帳號的學習自述與提問**，未驗證真人身份或正式修課資格；它們是具體個案，不能推論多少比例的學生會卡住。沒有把搜尋摘要、官方課綱或未讀到的討論當作學生心得，也沒有用作者的未受控數字當模型效果結論。

Michigan WI2022 主頁與 assignment5 重查後均為 HTTP **503**，依要求停止重試，未採用未取得的作業內容：

- <https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/>
- <https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/assignment5.html>

## 模擬新讀者提出的缺口

| 讀者會問的問題 | 原大綱待補之處 | 第二版的處理 |
| --- | --- | --- |
| 一張圖有兩個物件，究竟輸出多少數字，誰監督誰？ | assignment、target、head、loss 各有提到，但交接隱含 | 第 5–7 章反覆追蹤同一張兩物件圖 |
| resize 後框還對嗎？怎麼畫回原圖？ | bbox／dataset 有列出，缺完整座標契約 | 第 4 章開始記單位、轉換與還原；第 7 章加入空圖及變動物件數 |
| 能畫框就算會了嗎？ | 「精度」未明訂評估方法 | 第 6 章手算 TP／FP／FN、PR、AP50；第 7 章固定 held-out 評估 |
| loss 降卻沒框，先改什麼？ | 分類診斷未延伸到偵測訊號 | 看資料、target／decode、分項 loss、正樣本數，再做小量 overfit |
| 我現在做到哪裡算完成？ | 前半與長篇版本演化之間缺出口 | 第 7／8 章設最小成果；版本實驗分段完成，實測後填資源預算 |
| v1 沒 anchor，為何 v8 叫 anchor-free？ | 版本排序容易遮住不同設計軸 | 每版對照候選位置、框表示、assignment、head／分數與推論篩選 |
| v8／26 一章怎麼改了那麼多東西？ | 版本章過密，與單一主要改動矛盾 | 拆成短單元，各自沿用基線比較 |
| 換一個任務，我能自己選方案嗎？ | 應用篇偏流程，缺獨立決策 | 第 17 章選分支、做基線、診斷、改一項並回報品質與成本 |

沒有採用「將所有新版整批移成選修」的建議：使用者要求理解整個演化，故保留 v12 與 YOLO26 的章節；將重複長訓練、完整配置及硬體部署列作延伸。也沒有新增完整數學先修、半監督、分割或姿態系列。

第二版再請兩位代理複查讀者理解與引文。依複查補上每版的起始分支，避免讀者誤以為所有改動要依序累積；也在 v12 章明列從 feature map 到 Q／K／V 的最短 attention 橋接。來源複查未發現重大錯引，並將評估個案的標題改為「需要釐清」，避免把求證協議說成作者不理解。

## 最值得吸收的公開學習者個案

以下日期用月／年表示貼文或公開紀錄時間；Git commit 僅證明該版本紀錄，不代表實驗發生日。引文保持短，完整上下文可由來源核對。

### 1. 跑完 YOLO 作業，仍接不起訓練與解碼

[Pushkar_Kadam，DeepLearning.AI，2024/01](https://community.deeplearning.ai/t/c4w3-yolo-training-anchor-boxes-and-networks-output-tensor/538828/1) 明說有「things I did not understand after completing the assignment」，追問 loss 如何實作、anchor 尺寸、輸出 head 與框。此為自述完成作業的學習者；他對模型版本的猜測不當成官方版本事實。

[big-bbox，2022/01–02](https://community.deeplearning.ai/t/defining-a-custom-class-for-yolo-loss/89436/10) 比較 YOLOv2 程式時卡在 target、mask、body／head 與 loss 介面，最後說「I gave up and implemented a different architecture」，也追問 AP 計算。

**採用：**第 5–7 章完整追蹤 annotation → target → loss → decode → evaluation；訓練介面明確，mask 隨當前 batch 產生／取得。第 7 章拆里程碑，評估提前教。

### 2. 反覆看 anchor，仍把候選框與類別混在一起

[michael_yoon，DeepLearning.AI，2022/06](https://community.deeplearning.ai/t/yolo-anchor-boxes/134600/1) 說重看影片仍無法掌握，問為什麼不每個 class 配一個 box。

[markchangliu，D2L SSD，2022/03](https://discuss.d2l.ai/t/single-shot-multibox-detection-ssd/1604/2) 預期 anchor 座標應直接傳入 predictor，顯示 feature、候選框順序與幾何解碼仍沒接起來。[Aaron_L，D2L，2021/07](https://discuss.d2l.ai/t/anchor-boxes/1603/10) 則被 target 與 decode 的背景索引差異困住。SSD 的 C+1 類別約定不直接搬成 YOLO 規則。

**採用：**同格兩個同類物件、不同類但相似形狀兩個反例；標明 cell／anchor slot／class 各軸。列 dataset、target、ignore、decode 的索引及分數語義，分別畫 forward、assignment／loss、decode 資料流。

### 3. 看得懂 tensor 運算，仍不知道 loss mask 為何這樣設計

[luoh226，中文 YOLOv3 教學 issue #160，2022/11](https://github.com/bubbliiiing/yolo3-pytorch/issues/160) 問「这么取的意义是什么？」並區分 ignore 判斷使用 anchor 初始框或網路修正後的框。[維護者回覆](https://github.com/bubbliiiing/yolo3-pytorch/issues/160#issuecomment-1328073285) 只解釋該 repo 的具體實作，不能泛化所有版本。

[a-g-moore，AladdinPersson issue #137，2022/12](https://github.com/aladdinpersson/Machine-Learning-Collection/issues/137) 自述首個嚴肅專案，提出 confidence、責任 mask、座標量綱與 loss reduction 疑點，亦承認自身改寫可能有錯。此處是檢查建議，並非已重現的 repo 缺陷。

**採用：**手算少量候選的正／負／ignore 與分項 loss；明列原始版本、本章教學簡化、encode／decode 與 reduction 的契約。

### 4. 作業得滿分，完整推論還是接錯 batch 維度

[FluffyFirefly，DeepLearning.AI，2025/02](https://community.deeplearning.ai/t/yolo-car-detection-section-3-5-error-messages/764817/1) 自述「received 100% from the auto grader」；[後續自行定位](https://community.deeplearning.ai/t/yolo-car-detection-section-3-5-error-messages/764817/5) 為 class 軸處理不適用多了 batch 的 tensor。雖用舊 TensorFlow，這次已定位為作者程式問題。

**採用：**完整推論里程碑覆蓋 batch=1／>1、零／一／多物件；局部檢查與完整管線驗證各有位置。

### 5. 多尺度 anchor 的公式看到了，單位仍不清楚

[rjtshrm，Ayoosh issue #153，2021/10](https://github.com/ayooshkathuria/pytorch-yolo-v3/issues/153) 自述 anchor-based detection 新手，不解已有不同尺度 anchor，為何解碼還要按尺度縮放。

**採用：**第 9–10 章沿同一個實際框，逐一列 pixel／normalized／grid 單位與 stride，逆向核對 decode，不只給一行換算公式。

### 6. mAP 對不上，評估門檻與協議需要釐清

[ksmdnl，AladdinPersson issue #123，2022/09](https://github.com/aladdinpersson/Machine-Learning-Collection/issues/123) 報告 mAP 達不到教學數字，追問較低評估 cutoff 與較高畫框 cutoff。這只能指出協議需釐清，不能證明 cutoff 是落差的唯一原因。

**採用：**第 6 章就區分 score 門檻、NMS IoU、評估 matching IoU、AP50／COCO AP；比較結果記錄候選截斷與 AP 規則。

### 7. ResNet 講「保留原值」，projection 卻改 channel

[Vedant_Kaushik，D2L，2023/09](https://discuss.d2l.ai/t/residual-networks-resnet-and-resnext/86/15) 問改了輸入 channel 後如何仍產生原輸入。這是 identity 的動機與跨 stage 實作之間的概念落差。

**採用：**第 3 章先同 shape 的 identity，再獨立講 projection；兩支 shape 與各自成立的條件一起看。

### 8. 「小模型」仍跑不動，或 optimizer 接到另一個模型

[fanbyprinciple，D2L VGG，2021/09](https://discuss.d2l.ai/t/networks-using-blocks-vgg/78/12) 自述縮小版仍卡 GPU／Kaggle，縮 FC head 後才能訓練；他對最低輸入尺寸的猜測不是通用 VGG 規則。[Nish，2020/08](https://discuss.d2l.ai/t/networks-using-blocks-vgg/78/4) 手敲訓練後自行找到 optimizer 未接到真正訓練的網路，後來完成章節。

**採用：**小輸入、下採樣與 head 一起縮；資源規格要實測。暖身檢查一次參數更新，診斷中交代 notebook 的 model／optimizer 生命週期。

### 9. 以為是 CUDA 問題，換 driver 一天，最後找到 label=-1

[xTaiwanPingLord，李宏毅 ML2022 HW3 自學紀錄，2022/11](https://github.com/xTaiwanPingLord/ML2022-Spring-HW/blob/1a688de4901430fd36be7d1bc7d1b153d6f44f7c/HW3/hw3.md) 記錄換 CUDA driver 仍無效；CPU 單步揭露標籤越界，根因是路徑解析。花一天是除錯自述，非訓練時間。

**採用：**在長訓練前看一個 batch 的值、shape、範圍；CPU 單步列作排查手段。具體症狀對應下一個檢查點。

### 10. 學會了，不想繼續為 baseline 刷分

[1am9trash，李宏毅 ML2021 HW3 自學 notebook，2021/07 公開版本](https://github.com/1am9trash/Hung_Yi_Lee_ML_2021/blob/48db17948900af3d3f5d33ea235cfe1c9f9dd09e/hw/hw3/hw3_code.ipynb) 在長訓練後寫不想把時間持續投入調參。作者談 validation 增強的想法不作推薦，也沒有可引用的 wall-clock 預算。

**採用：**快速機制驗證與完整效果比較分列，設通過／停止條件；學習成果不用以追到最高 baseline 分數判定。

### 11. 只有 accuracy，做完沒有可見成果

[languephone，Harvard CS50 AI Traffic 個人 README，2022/02 固定版本](https://github.com/languephone/AI50-Traffic/blob/d78c7edeec26581dbf077c10b64fa2cd53c60ae6/README.md) 說完成作業仍「felt unsatisfied」，因只輸出 accuracy；自己加入真實路牌圖片推論與拼圖，並記錄一張 30kph 被誤認成 80kph。這是分類作業的個人反思，圖片示範不能替代泛化評估。

**採用：**從 CNN 第一章就有推論及錯誤圖板，逐章沿用；第 8 章加入自己的圖片，demo 與固定切分評估清楚標明。

### 12. 想同時會用框架，也會自己改模型；小數字例子有幫助

[JonathanSum，fast.ai，2020/08](https://forums.fast.ai/t/fastai2-and-new-course-now-released/75684/47) 希望學會自己建模型，後續澄清同時重視框架使用。這不是「fast.ai 沒教從零實作」的證據。

[przem8k，fast.ai，2023/11](https://forums.fast.ai/t/lesson-3-official-topic/96254/385) 肯定簡單 spreadsheet 的 gradient descent 直覺，並自述重新實作；本輪未驗證其 notebook 的正確性。[guptamols，2024/04](https://forums.fast.ai/t/lesson-7-collaborative-filtering/111822/3) 問當下可否視 Learner 為黑盒、何時該深挖 `nn.Parameter`。後者任務為 collaborative filtering，借用的是學習導航問題。

**採用：**每單元能跑、能解釋、能自主改一處並驗證；公式接最小計算例與呼叫流程，標明當下必懂與後續拆解深度。

## 本輪證據的實際邊界

可靠學習者證據集中在 D2L、DeepLearning.AI、fast.ai、Harvard 作業自述、李宏毅課程自學紀錄與三類 YOLO 教學 repo。沒有找到足以引用的 MIT 6.S191 第一手心得；Stanford 只見自學 repo 的資源組合，不能稱為已核實修課學生。因此仍可參考 MIT／Stanford 官方教學設計，但沒有替它們編造學生評價。

部分一般搜尋引擎遇到 JS 頁、驗證或限流；Reddit 回 403，未讀到正文，未採用其心得。本輪以可直接讀取的公開論壇與 GitHub 來源為準。中文另有明確承認 AI 協助的自學 Review，未放進主要引文證據。

本輪未重現來源程式或訓練結果；issue 的疑似缺陷標為作者主張。沒有把「大模型未過 baseline」推成架構結論，也沒有將自學者的每項調參建議照單全收。

原始回應、完整 subagent 報告與重查紀錄保存在 `/tmp/yolo-review-2026-10-02/`。本輪只新增研究及課綱文件，未修改 `learn_to_yolo` 程式或雲端環境設定。

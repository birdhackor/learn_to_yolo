# 讀者問題與公開學習者個案

查核日期：2026-10-02。對應課綱：[課程大綱](./outline.md)；官方教材與課程範圍見[公開教材與課程研究](./course-research.md)。

課綱的章節安排參考兩類意見：模擬讀者讀大綱時會問的問題，以及公開論壇與 GitHub 上學習者自述的卡關經驗。本頁列出這些問題與個案，以及課程對應的安排。

## 來源與方法

兩份讀者審查都由 AI 模擬讀者完成：審查者只讀大綱，以不了解本專案的讀者視角提問。公開學習紀錄分四組搜尋：D2L／Stanford、DeepLearning.AI／YOLO 教學 repo、fast.ai／Harvard，以及中文自學紀錄；只引用直接讀到的論壇貼文、GitHub issue 與回覆，以及 README、筆記與 notebook 裡的個人心得段落。搜尋與摘錄同樣由 AI 進行，每則引文都附原始連結。

所以兩份讀者審查是**模擬意見**，不是真人讀者的回饋。網路來源是**公開帳號的學習自述與提問**，未驗證真人身份或正式修課資格；它們是具體個案，不能推論多少比例的學生會卡住。搜尋摘要、官方課綱與沒讀到的討論都不當作學生心得，作者未受控的數字也不當作模型效果的結論。

## 模擬讀者的提問

| 讀者會問的問題 | 容易卡住的地方 | 課程的安排 |
| --- | --- | --- |
| 一張圖有兩個物件，究竟輸出多少數字，誰監督誰？ | assignment、target、head、loss 各自看得懂，交接處卻接不起來 | 第 5 章用一張固定的兩物件圖手算 target；第 7 章另用一張有紅、藍兩框的固定場景：7.1 核對兩框的畫素與標註，7.2 把兩框都換成 target，紅框再接著走過 7.3 的 loss 與 7.5 的推論 |
| resize 後框還對嗎？怎麼畫回原圖？ | 只記得把圖片縮成固定大小，沒記下框的單位、縮放比例與補邊 | 4.2 節先寫明三種座標系的單位，把框轉到輸入畫布再還原；第 7 章的資料含空圖與每張不同的物件數 |
| 能畫框就算會了嗎？ | 框看起來不錯，卻沒有固定的評估方法 | 第 6 章用人工框手算 TP／FP／FN、PR 與 AP50；第 7 章用固定的 held-out 資料評估 |
| loss 降卻沒框，先改什麼？ | 分類學過的診斷順序，不知道怎麼套到偵測 | 7.4 節先依序查資料與正格數、梯度方向、參數有沒有更新，並提醒要看分項 loss 與解碼後的框，不能只看 total；少量 overfit 是下一關 |
| 我現在做到哪裡算完成？ | 前半的基礎與很長的版本演化之間，看不出哪裡可以停 | 第 0–7 章先完成單張圖片的偵測流程，第 8 章接自己的圖片與資料；要做第 17 章的結業任務，也不必先讀第 9–16 章 |
| v1 沒 anchor，為何 v8 叫 anchor-free？ | 照版本號讀，容易把框表示、候選與 assignment 這些不同的設計選擇混成一條線 | 12.1 節對照第 7 章的 grid 模型：兩者都沒有預設框尺寸，框表示、候選點與 assignment 卻不同；anchor-free 也不等於 NMS-free |
| v8／26 一章怎麼改了那麼多東西？ | 一章同時改好幾項機制，看不出每一項各自的作用 | 第 12、16 章拆成數個短單元，每節寫明起點，只改一項機制 |
| 換一個任務，我能自己選方案嗎？ | 應用練習只剩照著流程做，沒有練習自己做選擇 | 第 17 章從一個需求出發：選定 baseline、看失敗、先寫保留規則，只改一項，再交付評估結果、失敗圖與時間 |

版本章多，是因為課程的目標是看懂整段演化：YOLOv12 與 YOLO26（第 15、16 章）也屬於第 9–16 章的完整演化路線，不列為選修。成本靠實驗規模控制：各節實驗都是 CPU 上的小規模實驗，沒有在真實照片資料集上長時間訓練，也不重現官方的完整配置；影片、追蹤與部署（第 18–20 章）是選修。數學隨問題補充，不另設完整的數學先修；半監督、分割與姿態估計不在範圍內。

## 最值得吸收的公開學習者個案

以下日期是貼文或公開紀錄的年／月；固定在某個 Git commit 的來源，commit 只證明那個版本存在的時間，不代表實驗發生的日期。引文保持簡短，完整上下文可由來源核對。

### 1. 跑完 YOLO 作業，仍接不起訓練與解碼

[Pushkar_Kadam，DeepLearning.AI，2024/01](https://community.deeplearning.ai/t/c4w3-yolo-training-anchor-boxes-and-networks-output-tensor/538828/1) 明說有「things I did not understand after completing the assignment」，追問 loss 如何實作、anchor 尺寸、輸出 head 與框。此為自述完成作業的學習者；他對模型版本的猜測不當成官方版本事實。

[big-bbox，2022/01–02](https://community.deeplearning.ai/t/defining-a-custom-class-for-yolo-loss/89436/10) 比較 YOLOv2 程式時卡在 target、mask、body／head 與 loss 介面，最後說「I gave up and implemented a different architecture」，也追問 AP 計算。

**採用：**第 5–7 章依序接起 annotation → target → loss → decode → evaluation。7.2 節列出 target 每個欄位的 shape，以及哪些格子進 loss；完整訓練程式在每一步替當步那批圖重新產生 target 與 mask。第 7 章拆成六個可逐段驗證的小節，評估則在第 6 章就用人工框先教。

### 2. 反覆看 anchor，仍把候選框與類別混在一起

[michael_yoon，DeepLearning.AI，2022/06](https://community.deeplearning.ai/t/yolo-anchor-boxes/134600/1) 說重看影片仍無法掌握，問為什麼不每個 class 配一個 box。

[markchangliu，D2L SSD，2022/03](https://discuss.d2l.ai/t/single-shot-multibox-detection-ssd/1604/2) 預期 anchor 座標應直接傳入 predictor，顯示 feature、候選框順序與幾何解碼仍沒接起來。[Aaron_L，D2L，2021/07](https://discuss.d2l.ai/t/anchor-boxes/1603/10) 則被 target 與 decode 的背景索引差異困住。SSD 的 C+1 類別約定不直接搬成 YOLO 規則。

**採用：**第 5 章用同格兩個同類物件示範容量限制。第 9 章標明格、anchor 槽與類別是各自獨立的軸：每個槽都輸出全部類別的分數；anchor 是不參與訓練的固定寬高，用來挑負責的槽，也用在 encode 與 decode。grid 模型的背景由另一個 objectness 分數表示，不占類別索引；判斷正負格一律用 `positive` 遮罩，不看類別值。

### 3. 看得懂 tensor 運算，仍不知道 loss mask 為何這樣設計

[luoh226，中文 YOLOv3 教學 issue #160，2022/11](https://github.com/bubbliiiing/yolo3-pytorch/issues/160) 問「这么取的意义是什么？」並區分 ignore 判斷使用 anchor 初始框或網路修正後的框。[維護者回覆](https://github.com/bubbliiiing/yolo3-pytorch/issues/160#issuecomment-1328073285) 只解釋該 repo 的具體實作，不能泛化所有版本。

[a-g-moore，AladdinPersson issue #137，2022/12](https://github.com/aladdinpersson/Machine-Learning-Collection/issues/137) 自述首個嚴肅專案，提出 confidence、責任 mask、座標量綱與 loss reduction 疑點，亦承認自身改寫可能有錯。此處是檢查建議，並非已重現的 repo 缺陷。

**採用：**第 7 章手算正格與負格的分項 loss，第 9 章再加上 ignore 槽；相關各節寫明原版的做法與課程的教學簡化，以及 encode／decode 與 reduction（各項 loss 取平均或加總）的約定。

### 4. 作業得滿分，完整推論還是接錯 batch 維度

[FluffyFirefly，DeepLearning.AI，2025/02](https://community.deeplearning.ai/t/yolo-car-detection-section-3-5-error-messages/764817/1) 自述「received 100% from the auto grader」；[後續自行定位](https://community.deeplearning.ai/t/yolo-car-detection-section-3-5-error-messages/764817/5) 為 class 軸處理不適用多了 batch 的 tensor。雖然用的是舊版 TensorFlow，問題已定位在作者自己的程式。

**採用：**7.5 節的完整推論涵蓋 batch=1／>1 與零／一／多物件；局部檢查與完整管線驗證各有位置。

### 5. 多尺度 anchor 的公式看到了，單位仍不清楚

[rjtshrm，Ayoosh issue #153，2021/10](https://github.com/ayooshkathuria/pytorch-yolo-v3/issues/153) 自述 anchor-based detection 新手，不解已有不同尺度 anchor，為何解碼還要按尺度縮放。

**採用：**第 9、10 章各沿一個實際框，逐一列出 pixel／normalized／grid 單位與 stride，再 decode 回去核對，不只給一行換算公式。

### 6. mAP 對不上，評估門檻與協議需要釐清

[ksmdnl，AladdinPersson issue #123，2022/09](https://github.com/aladdinpersson/Machine-Learning-Collection/issues/123) 報告 mAP 達不到教學數字，追問較低評估 cutoff 與較高畫框 cutoff。這只能指出協議需釐清，不能證明 cutoff 是落差的唯一原因。

**採用：**第 6 章就區分 score 門檻、NMS IoU、評估 matching IoU、AP50／COCO AP；比較結果記錄候選截斷與 AP 規則。

### 7. ResNet 講「保留原值」，projection 卻改 channel

[Vedant_Kaushik，D2L，2023/09](https://discuss.d2l.ai/t/residual-networks-resnet-and-resnext/86/15) 問改了輸入 channel 後如何仍產生原輸入。這是 identity 的動機與跨 stage 實作之間的概念落差。

**採用：**第 3 章先同 shape 的 identity，再獨立講 projection；兩支 shape 與各自成立的條件一起看。

### 8. 「小模型」仍跑不動，或 optimizer 接到另一個模型

[fanbyprinciple，D2L VGG，2021/09](https://discuss.d2l.ai/t/networks-using-blocks-vgg/78/12) 自述縮小版仍卡 GPU／Kaggle，縮 FC head 後才能訓練；他對最低輸入尺寸的猜測不是通用 VGG 規則。[Nish，2020/08](https://discuss.d2l.ai/t/networks-using-blocks-vgg/78/4) 手敲訓練後自行找到 optimizer 未接到真正訓練的網路，後來完成章節。

**採用：**第 1 章用 32×32 的小輸入，逐層算參數量與計算量，也估記憶體，看下取樣與 head 大小怎麼影響成本。暖身章確認一次更新前後參數真的改變，並說明 notebook 裡重建模型時，optimizer 也要重建。

### 9. 以為是 CUDA 問題，換 driver 一天，最後找到 label=-1

[xTaiwanPingLord，李宏毅 ML2022 HW3 自學紀錄，2022/11](https://github.com/xTaiwanPingLord/ML2022-Spring-HW/blob/1a688de4901430fd36be7d1bc7d1b153d6f44f7c/HW3/hw3.md) 記錄換 CUDA driver 仍無效；CPU 單步揭露標籤越界，根因是路徑解析。花一天是除錯自述，非訓練時間。

**採用：**第 2 章在長訓練前看一個 batch 的值、shape、範圍，並把 CPU 單步列為 GPU 報錯時的排查手段；具體症狀對應下一個檢查點。

### 10. 學會了，不想繼續為 baseline 刷分

[1am9trash，李宏毅 ML2021 HW3 自學 notebook，2021/07 公開版本](https://github.com/1am9trash/Hung_Yi_Lee_ML_2021/blob/48db17948900af3d3f5d33ea235cfe1c9f9dd09e/hw/hw3/hw3_code.ipynb) 在長訓練後寫不想把時間持續投入調參。作者談 validation 增強的想法不作推薦，也沒有可引用的 wall-clock 預算。

**採用：**快速機制驗證與完整效果比較分開寫，並寫明何時可以停（例如第 2、17 章）；結業任務沒有改善也能交付「不保留」與診斷結果，學習成果不以追到最高 baseline 分數判定。

### 11. 只有 accuracy，做完沒有可見成果

[languephone，Harvard CS50 AI Traffic 個人 README，2022/02 固定版本](https://github.com/languephone/AI50-Traffic/blob/d78c7edeec26581dbf077c10b64fa2cd53c60ae6/README.md) 說完成作業仍「felt unsatisfied」，因只輸出 accuracy；自己加入真實路牌圖片推論與拼圖，並記錄一張 30kph 被誤認成 80kph。這是分類作業的個人反思，圖片示範不能替代泛化評估。

**採用：**第 1 章就有推論與錯誤圖板，第 4、7、8 章也畫真值與預測框的疊圖。第 8 章接自己的圖片，並標明哪些圖只是流程示範、哪些是固定切分上的評估。

### 12. 想同時會用框架，也會自己改模型；小數字例子有幫助

[JonathanSum，fast.ai，2020/08](https://forums.fast.ai/t/fastai2-and-new-course-now-released/75684/47) 希望學會自己建模型，後續澄清同時重視框架使用。這不是「fast.ai 沒教從零實作」的證據。

[przem8k，fast.ai，2023/11](https://forums.fast.ai/t/lesson-3-official-topic/96254/385) 肯定簡單 spreadsheet 的 gradient descent 直覺，並自述重新實作；本頁沒有驗證其 notebook 是否正確。[guptamols，2024/04](https://forums.fast.ai/t/lesson-7-collaborative-filtering/111822/3) 問當下可否視 Learner 為黑盒、何時該深挖 `nn.Parameter`。後者任務為 collaborative filtering，借用的是學習導航問題。

**採用：**每節都附可執行的完整程式與自主練習：改一處再驗證。公式接最小計算例與程式呼叫流程；較深的推導多放在摺疊區，部分標明「可先跳過」或「可跳讀」。

## 證據的邊界

可靠的學習者證據集中在 D2L、DeepLearning.AI、fast.ai、Harvard 作業自述、李宏毅課程自學紀錄與三個 YOLO 教學 repo。沒有找到足以引用的 MIT 6.S191 第一手心得；Stanford 只見自學 repo 的資源組合，不能稱為已核實的修課學生。MIT 與 Stanford 的官方教學設計仍可參考（見[公開教材與課程研究](./course-research.md)），但本頁沒有它們的學習者個案。

部分一般搜尋引擎回傳需要 JavaScript 的頁面，或要求驗證、遭到限流；Reddit 回 HTTP 403，讀不到正文，所以沒有採用其心得。引文只取可直接讀取的公開論壇與 GitHub 來源。另有一篇明確承認由 AI 協助的中文自學 review，沒有列入主要引文。

Michigan EECS 498 WI2022 的主頁與 assignment5 查核時都回 HTTP **503**，所以沒有採用其作業內容：

- <https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/>
- <https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/assignment5.html>

本頁沒有重現來源的程式或訓練結果；issue 裡的疑似缺陷都標為作者的主張，自學者的調參建議也不照單全收。

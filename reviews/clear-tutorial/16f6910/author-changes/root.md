# 主協調者的改寫（非獨立通過證據）

基線為 `16f69103d1f216d1024a40610623083557324701`。首次閱讀原始紀錄不改寫；下面只是作者的處理說明，獨立核回見 `../technical/` 和 `../transitions/`。

- 首頁先以同張圖片的「分類答案／偵測答案」展示任務差別，刪除尚未建立背景前的完整 NMS、IoU、loss 步驟。新圖沒有假稱紅框中心為 (16,20)；該例子在 7.1/7.2 的圖中獨立介紹，所有舊首頁紅框指涉已修。
- 暖身先手算 x=2、y=4、w=1 的 prediction、loss、微小變化、梯度與一次 SGD；再解讀可執行五行程式及輸出。長推導、None/zero、detach、參數 rebinding 等移至需要時回讀的選讀。
- CNN 先展示真正 `make_batch()` 的八張輸入和類別，再說共享卷積視窗、ReLU、pooling、GAP、shape 與完整程式。新增素材、視窗、資料流、感受野圖；三步結果和40步曲線改大字圖。歷史細節、感受野邊界、MACs、float32捨入放選讀，仍保留颜色与形状绑在一起、没有held-out等限制。
- 03 projection 不再引用暖身已移掉的矩陣／內積段落；在需要處直接解釋。06 evaluation 先完成 PR/AP 主例，官方 VOC/COCO 差異移至其後；AP 與最後 P×R 關係獨立選讀。
- 07 data 明說生成矩形中心嚴格在格內；07 targets 以原始標註→格子答案的格式轉換取代 targets/target 的拼字提醒；07 loss 分清輸出 logits 梯度量與 CNN 權重梯度。07 training 存的 validation PNG 只是 RGB 輸入，框在 report/圖板。
- 08 的合併實測圖在桌面縮排後字太小。沿用原始 JSON 全1600個 loss、原 PNG、GT、預測與配對結果，另排窄版曲線和四張圖；不改原生成圖、不重新挑漂亮案例、不新訓練1600步。重排來源腳本保存在 `readable-custom-source.txt`，各新SVG記錄JSON SHA。
- 12 assignment 補 owner→GT類別→分類target→本例前景target 的橋接表，區分官方品質軟target、本書硬target和本toy四個獨立logits。
- 13 的官方 top-1 描述經官方固定源碼CPU反例查證後修正：初選額度並非衝突後每GT至多一個。反例與原方法執行保存；單正样本的軟target寫精確 m×CIoU/(m+eps)，僅m遠大於eps才近似CIoU。26衝突後另topk2，不能外推10的反例。13.2區分人工target、設計目標與實際推論不保證的框數。
- 17 重跑後只有計時變動，正文改成1.03/1.12秒與2.96毫秒；所有AP、FP類型與框保持相同。19 ByteTrack 依原作者程式分清confirmed/lost/unconfirmed及低分輪僅仍Tracked者。
- 其他頁補 tag、COCO xywh/crowd、RNG 的必要短定義，發布／LFS區分乾淨runtime與既有版控目錄；發布流程加入是否過期的分支，不要求全GPU/CPU重跑，render-only與帶回圖、人工數字核對分開。

CPU程式、模型、資料生成與訓練設定沒有修改。頁面重產紀錄只刷新真實輸出；新 release 使用不同tag，不移動舊tag。桌面／手機與MathJax檢查在實際Zensical頁面上執行，原SVG预览與頁面呈現證據分開記錄。

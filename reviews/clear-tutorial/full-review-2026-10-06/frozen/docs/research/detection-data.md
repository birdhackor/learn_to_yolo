# 物件偵測資料來源核對

> 本頁是偵測資料來源的查核紀錄：比較四個真實照片資料集的下載條件、標註格式、切分與授權，並記下查核時建議的用法。教材的偵測實驗都用合成圖或小型自製資料，沒有用這四個資料集訓練；實際下載與採用狀態見〈[資料規劃](../preparation/data.md)〉。

查核日期：2026-10-02。文中的章節編號依〈[課程大綱](../planning/outline.md)〉。本查核只看來源、下載條件與小型標註樣本，沒有下載完整資料集。以下大小是壓縮／封裝檔的 HTTP `Content-Length`（伺服器在回應標頭回報的檔案大小），不是解壓後空間；MB 採十進位。HEAD 請求只取回應標頭、不下載內容，成功時回 200；range GET 只要求檔案的一段，伺服器支援時回 206。COCO2017 下載網址的 HEAD／range 結果量測於 2026-10-04 16:43 UTC。大型檔只讀 1 KiB 的 range 或檔案前段就關閉連線，不能據此宣稱全檔完整性已驗證。

## 建議主線與備援

1. **第 4–7 章的小型真實資料先用 Penn-Fudan**：170 張、單一 pedestrian 類別，從「恰好一個完整標註實例」子集到多人圖，沿用同一來源。單物件子集須在固定切分之後挑選，不把多人圖的其餘行人刪成背景。幾何圖形仍適合空圖片、同格兩物件等人工已知答案。
2. **多類別主線用 VOC2007 固定子集**：保留官方 train／val／test 的邊界，可先選數百張，用小規模實驗量過訓練耗時與結果，再決定要不要擴充。Penn-Fudan 若不適合教學目標，VOC 亦可直接作第 4 章單物件子集備援。
3. **第 10–17 章的小物件／擁擠／重疊評估用 COCO2017 固定子集**：從 train2017 選訓練、val2017 選獨立評估；不需先下載 19 GB 訓練圖。先取 annotations、建立固定 image ID 清單，再按張取得官方圖片；本查核不訂張數與下載預算，要採用時先用小規模實驗量測再決定。COCO 官網也介紹 FiftyOne 的 selective download，但那是額外工具而非資料授權。
4. **Oxford-IIIT Pet 為單物件定位備選**：head ROI 適合類別＋框的教學，但框指動物頭部，不能稱全身物件框；下載量較大，且現行網頁與檔內舊 README 授權不同，不列為預設。

| 候選 | 能支持的情境 | 封裝大小與可用性 | 授權／可再散布 |
| --- | --- | --- | --- |
| Penn-Fudan | 單人→多人、長寬比、少量遮擋；沒有真正無人的圖片 | ZIP **53,723,336 bytes（53.72 MB）**；HEAD 200、range GET 206，已讀實際 README／一份 annotation | 沒有開放照片授權；README 明示通常不能未經權利人許可重新張貼。不能直接打包進公開 repo |
| VOC2007 | 單物件子集、多類別、多物件、truncated／difficult、重複框 | trainval TAR **460,032,000 bytes（460.03 MB）**；test TAR **451,020,800 bytes（451.02 MB）**；官方新主機均 HEAD 200／range 206，抽讀一份 XML | 官方要求遵守 Flickr 條款；照片逐張權利及 annotation 再散布授權未明。不能把 devkit／GitHub 程式授權套到照片 |
| COCO2017 | 多類別、小物件、多尺度、crowd、重疊、NMS／assignment | train ZIP **19,336,861,798 bytes（19.34 GB）**；val ZIP **815,585,330 bytes（815.59 MB）**；annotations ZIP **252,907,541 bytes（252.91 MB）**；官方 HTTP 網址與 S3 HTTPS 路徑均 HEAD 200／range 206；`images.cocodataset.org` 走 HTTPS 會因憑證不符而失敗（見下方 COCO2017 的核對） | annotation **CC BY 4.0**、可按條件再散布；照片權利不屬 COCO，依 Flickr／每張 license。不得把整包照片稱為 CC BY 4.0 |
| Oxford-IIIT Pet | 單物件頭部定位、類別與框、非正方形圖 | images TAR.GZ **791,918,971 bytes（791.92 MB）**；annotations TAR.GZ **19,173,078 bytes（19.17 MB）**；HEAD／GET 200、gzip magic 正確。服務忽略 Range，已限制只讀前段並關閉 | 現行官網寫 **CC BY-SA 4.0**、可商業／研究使用，版權仍屬原作者；檔內 README 卻寫 research only／原網站條款。再散布前應釐清衝突，不聲稱照片無條件可再散布 |

## 逐一核對

### 1. Penn-Fudan：最省下載的真實資料

- [作者／UPenn 官方資料頁](https://www.cis.upenn.edu/~jshi/ped_html/)；[實際 ZIP](https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip)。官方描述 170 張圖片、345 名有標註行人、UPenn 96 張／Fudan 74 張，每張至少一名行人。發布版 README 另說新增非常小或高度遮擋行人；網站的站立中大型行人描述與舊論文用途不足以證明它是完整小物件基準。
- 原生是 `PNGImages/*.png`、`PedMasks/*_mask.png`、`Annotation/*.txt`。TXT 為 **PASCAL Annotation Version 1.00 文字格式，非 VOC XML**；有 `Xmin,Ymin,Xmax,Ymax`，樣本明寫左上畫素 `(1,1)`。mask 的 0 是背景、正數為 instance ID；用 mask 轉框時須明定座標契約，不能與 TXT 的 1-based 框混用。
- 小樣本 `FudanPed00001` 為 559×536、兩名行人；框 `(160,182)-(302,431)` 及 `(420,171)-(535,486)`，可用作非正方形與多物件的標註管線範例。只抽取 README／TXT，ZIP 目錄與小檔合計傳輸 46,575 bytes，未下載圖片包。
- **授權證據**：檔內 README：「Copyright and all rights therein are retained by authors or by other copyright holders」及「In most cases, these works may not be reposted without the explicit permission of the copyright holder.」公開可下載不等於可再散布；PyTorch tutorial 的程式或文件 license 亦不授權照片。
- **切分**：壓縮檔目錄未提供現成 train／val／test manifest；[PyTorch 官方教學](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html) 的隨機留出只是示範。應固定 scene／近重複圖片分組，保存不可變 manifest；同一場景、augmentation 與衍生 crop 都隨原圖留在同一 split。若按校園整批留出，需說明是跨場景分布測試。沒有空圖，需另用可控資料或自攝並確認標註完整的負例，不把遮擋漏標當負例。

### 2. Pascal VOC2007：多類別基線

- [Oxford 官方頁](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/index.html)、[開發套件文件](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/htmldoc/index.html)、[官方統計](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/dbstats.html)。20 類、共 9,963 張；train 2,501、val 2,510、trainval 5,011，test 為其餘 4,952 張。
- 可用下載：[trainval](https://thor.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)、[test](https://thor.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar)。舊 `https://host.robots.ox.ac.uk/pascal/...` 查核時回 503；Oxford 官網的 `www.robots.ox.ac.uk/~vgg/projects/...` 連結會轉址至以上 `thor` 主機，官方 torchvision [VOC reader](https://raw.githubusercontent.com/pytorch/vision/main/torchvision/datasets/voc.py) 也使用它。
- `JPEGImages/*.jpg`、`Annotations/*.xml`、`ImageSets/Main/{train,val,trainval,test}.txt`；XML 有 class name、`bndbox`、`difficult`、`truncated`、來源／owner。VOC 為 1-based 座標；轉到 0-based 半開 `xyxy` 要一致處理，不只調 xmin/ymin 卻漏掉端點含義。已抽讀 `000005.xml`，包括同圖多個 chair、difficult／truncated 的實例。
- **授權**：官方「Database Rights」要求對 Flickr 圖片遵守其使用條款，公開頁未提供可涵蓋所有照片的 CC 授權，也未查到清楚的 annotation 再散布 grant。部分 XML 有 Flickr source ID／owner，但不能以此推定全部圖片同一權利。
- **切分與評估**：保留官方邊界，`trainval` 是 train＋val，不能訓練 trainval 又在官方 val 宣稱獨立評估。小型練習先 train 子集訓練、val 子集調參，test 留作最後檢查；跨 VOC 年份拼接先檢查圖片 ID／內容重複。不要把 `difficult=1` 刪除後當背景，應採 ignore 或明示簡化協議；AP50 也須區分 VOC2007 的 11-point AP 與其他積分版本。

### 3. COCO2017：困難場景與後段演化實驗

- [官方下載頁](https://cocodataset.org/#download)的實際本文：[download.htm](https://cocodataset.org/dataset/download.htm)；[標註格式本文](https://cocodataset.org/dataset/format-data.htm)；[授權本文](https://cocodataset.org/dataset/termsofuse.htm)。有 bbox、instance segmentation、crowd，適合單／多尺度與密集候選比較。
- 可用下載：[train2017.zip](http://images.cocodataset.org/zips/train2017.zip)、[val2017.zip](http://images.cocodataset.org/zips/val2017.zip)、[annotations_trainval2017.zip](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)，即官方下載頁列出的 HTTP 網址。
- **HTTPS 要換網址**：`images.cocodataset.org` 在 DNS 上是 Amazon S3 bucket（存放檔案的儲存空間）的別名，連線時 S3 出示的是發給 `s3.amazonaws.com` 等 Amazon 網域的憑證，不含 `images.cocodataset.org`；所以把上面的網址改成 `https://` 會因憑證主機名不符而連線失敗，curl 與 Python 的 urllib 實測都如此。需要 HTTPS 時，整包下載與按張取得圖片都改用同一個 bucket 的路徑式網址 `https://s3.amazonaws.com/images.cocodataset.org/…`，例如 `https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip`；不要為了連上而關掉憑證驗證。`data/manifest.json` 的 `coco2017` 記的就是這三個檔的 S3 HTTPS 網址。2026-10-04 16:43 UTC 量測時，三個檔的官方 HTTP 網址與 S3 HTTPS 路徑都是 HEAD 回 200、range GET 回 206，兩邊回報的 `Content-Length`、`ETag`、`Last-Modified` 相同。
- 偵測使用 `instances_train2017.json`／`instances_val2017.json`：`images`、`annotations`、`categories`、`licenses`；bbox 是 **0-based 畫素 `[左上x,左上y,width,height]`**，包含 `area` 與 `iscrowd`。category ID 不應直接當連續 class index，必須存 mapping。每張 image 的 `license` 指向 license 表，另有 Flickr URL。
- **授權**：官方 annotation／website 明示 **Creative Commons Attribution 4.0**。官方也明示 COCO 不擁有圖片版權，照片須遵守 Flickr 條款；再散布需按 image license 逐張核對 attribution、NC／SA／ND 等限制與用途。開源 API 的程式 license 與照片授權是不同來源。
- **切分與小物件**：118K train／5K val。2014 與 2017 使用相同圖片重新切分，混用年份會造成洩漏。真實小物件選樣應保存原 annotation `area` 與輸入後大小；[官方 evaluator](https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/cocoeval.py#L502-L511) 的 small area range 為 0–32²，但原圖 `area` 不等同 resize 後 bbox 面積，不能混成同一指標。`iscrowd=1` 要保留群體區域的特殊規則，見下方說明。
- 建子集先固定選樣條件與 image ID，再對每張入選圖保留所選任務全部相關實例，不能只留下觸發「小物件」選樣的那個框。重疊框的 IoU 不直接代表遮擋程度；還需看圖／mask。單獨拿 val2017 再分 train／val 只能稱自訂 COCO-val 子集，不能同時當官方 held-out 評估。

### COCO 的 crowd 區域不是一個普通物件 { #coco-crowd }

`iscrowd=1` 標記同類物件密集、難以逐一區分的群體區域。例如一群人擠在一起，標註的是群體區域，不是要模型輸出一個「大人物」。保存標記，也不要把區域刪掉當背景。

COCO 官方評估器把 crowd GT 當作忽略區域：匹配到它的預測不計一般 TP／FP，而且同一個 crowd 區域允許接住多個預測。對 crowd 的重疊判準也不是普通 IoU，而是「預測與 crowd 的交集面積 ÷ 預測面積」；這讓落在群體內的小框能匹配該區域。一般非 crowd GT 仍使用普通 IoU 和一個 GT 配一個預測的規則。

這裡說的是**官方評估規則**，不表示訓練時直接把 crowd 當成一個正樣本。本專案的合成資料 loader 與 AP50 評估器沒有 crowd 支援；接 COCO 時，需要另訂訓練時忽略區域的策略，評估則使用官方 API，不能直接沿用本書簡化計分報 COCO 成績。

依據：[官方 `mask.py` 的 crowd 重疊公式](https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/mask.py#L58-L67)、[官方 `cocoeval.py` 的忽略與配對](https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/cocoeval.py#L261-L299)，另見 [crowd GT 設為 ignore 的位置](https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/cocoeval.py#L106-L109)。以上固定 commit `8c9bcc3`。

### 4. Oxford-IIIT Pet：頭部定位備選，不是小型多人資料

- [Oxford／作者官方頁](https://www.robots.ox.ac.uk/~vgg/data/pets/)；[images](https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz)、[annotations](https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz)，查核時轉址到 `/pets/...`。37 品種、每類約 200 張。`annotations/xmls/` 是 PASCAL VOC **head** 框，`trimaps/` 是 foreground／background／unknown，不是多個物件的 instance mask。
- 本查核只讀 annotations 壓縮檔前段 262,144 bytes，取得 README，沒有下載完整圖片或標註包。官方頁稱圖片都有 head ROI；完整標註包的盤點見〈[基礎資料](foundation-data.md)〉：trainval 有 9 筆沒有 XML，test 全部沒有 XML。因此不能承諾全部約 7K 張圖都能直接做 bbox 訓練，選資料前要先交叉檢查 XML 與 split 清單。
- **授權衝突要保留**：現行頁明示 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)，商業／研究可下載；封裝 README 的原文為「Dataset is made available for research purposes only. Use of these images must respect the corresponding terms of use of original websites from which they are taken.」查核時沒有找到解釋兩者差異的說明，因此只建議提供官方下載連結與 manifest；要隨教材再散布照片，須先向作者確認適用條款。
- 官方 `trainval.txt`／`test.txt` 作來源邊界；再從 trainval 分出 validation，test 不參與選模型。近重複圖片、裁切及 resize 衍生檔都隨原圖分組。以 cat／dog 或品種做類別都須固定映射；不能用單一 foreground trimap 宣稱存在多物件標註。

## 收集時固定保存的資料與再散布決定

- 最少保存：資料集版本、來源 URL、原圖 ID、原始 split、來源／場景 group、類別 mapping、座標格式、annotation 來源、每張圖片的 license／attribution；下載後另算 SHA-256。**本查核只確認下載網址可用，不拿 multipart ETag（分段上傳檔案的版本標記，不是整檔雜湊）當 SHA／MD5，也不宣稱已驗證完整封裝。** Penn-Fudan ZIP 與 Pet annotations 這兩個封裝另有完整下載，由完整檔算出的 SHA-256 記在 `data/manifest.json`，校驗結果分別見〈[資料規劃](../preparation/data.md)〉與〈[基礎資料](foundation-data.md)〉；本查核沒有下載這兩個完整封裝。
- 固定困難集從訓練之外建立：一／多物件、同格同類兩物件、小物件、重疊、非正方形、極端長寬比。真實資料中的空標註通常只代表「沒有所選類別／沒有已標註實例」，不是已證明畫面完全無物件。人工幾何圖與自攝授權清楚負例可補缺口。
- 查核的官方下載都公開可讀，不需要登入或 API key。
- 若要把照片打包進公開 repo 或教材下載包，就要先決定再散布用途與商業需求，並對 Penn-Fudan 取得權利人許可、對 VOC／COCO 逐張核對，或改用自攝、權利清楚的素材；Oxford Pet 的網頁與 README 衝突也要先釐清。不打包照片時，可只散布下載連結與自己的子集 ID 清單。
- 本查核沒有用這些資料集訓練；VOC／COCO 子集需要的張數、下載量、訓練耗時與 GPU 資源都沒有實測，因此不對訓練耗時或小模型品質下結論。要採用時，先用小規模實驗量測。

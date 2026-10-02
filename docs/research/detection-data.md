# 物件偵測資料來源核對

> 這是研究代理在子任務當時的查核報告。後續整合已套用前置配置並下載部分資料；目前狀態與操作步驟請以「前置準備」頁為準。

查核日期：2026-10-02；對應 `/workspace/yolo-curriculum-outline-v2.md`。本輪只查來源、下載條件與小型標註樣本，沒有下載完整資料集、修改 `learn_to_yolo`、編寫教材或模型。以下大小是壓縮／封裝檔的 HTTP `Content-Length`，不是解壓後空間；MB 採十進位。大型檔只讀 1 KiB range／前段後關閉連線，不能據此宣稱全檔完整性已驗證。

## 建議主線與備援

1. **第 4–7 章的小型真實資料先用 Penn-Fudan**：170 張、單一 pedestrian 類別，從「恰好一個完整標註實例」子集到多人圖，沿用同一來源。單物件子集須在固定切分之後挑選，不把多人圖的其餘行人刪成背景。幾何圖形仍適合空圖片、同格兩物件等人工已知答案；本輪不產生它們。
2. **多類別主線用 VOC2007 固定子集**：保留官方 train／val／test 的邊界，可先選數百張，再視原型實測擴充。優先沿用課綱已提到的 VOC 路線。Penn-Fudan 若不適合教學目標，VOC 亦可直接作第 4 章單物件子集備援。
3. **第 10–17 章的小物件／擁擠／重疊評估用 COCO2017 固定子集**：從 train2017 選訓練、val2017 選獨立評估；不需先下載 19 GB 訓練圖。先取 annotations、建立固定 image ID 清單，再按張取得官方圖片；具體張數與下載預算待原型制定。COCO 官網也介紹 FiftyOne 的 selective download，但那是額外工具而非資料授權。
4. **Oxford-IIIT Pet 為單物件定位備選**：head ROI 適合類別＋框的教學，但框指動物頭部，不能稱全身物件框；下載量較大，且現行網頁與檔內舊 README 授權不同，不作目前預設。

| 候選 | 能支持的情境 | 封裝大小與可用性 | 授權／可再散布 |
| --- | --- | --- | --- |
| Penn-Fudan | 單人→多人、長寬比、少量遮擋；沒有真正無人的圖片 | ZIP **53,723,336 bytes（53.72 MB）**；HEAD 200、range GET 206，已讀實際 README／一份 annotation | 沒有開放照片授權；README 明示通常不能未經權利人許可重新張貼。不能直接打包進公開 repo |
| VOC2007 | 單物件子集、多類別、多物件、truncated／difficult、重複框 | trainval TAR **460,032,000 bytes（460.03 MB）**；test TAR **451,020,800 bytes（451.02 MB）**；官方新主機均 HEAD 200／range 206，抽讀一份 XML | 官方要求遵守 Flickr 條款；照片逐張權利及 annotation 再散布授權未明。不能把 devkit／GitHub 程式授權套到照片 |
| COCO2017 | 多類別、小物件、多尺度、crowd、重疊、NMS／assignment | train ZIP **19,336,861,798 bytes（19.34 GB）**；val ZIP **815,585,330 bytes（815.59 MB）**；annotations ZIP **252,907,541 bytes（252.91 MB）**；均 HEAD 200／range 206 | annotation **CC BY 4.0**、可按條件再散布；照片權利不屬 COCO，依 Flickr／每張 license。不得把整包照片稱為 CC BY 4.0 |
| Oxford-IIIT Pet | 單物件頭部定位、類別與框、非正方形圖 | images TAR.GZ **791,918,971 bytes（791.92 MB）**；annotations TAR.GZ **19,173,078 bytes（19.17 MB）**；HEAD／GET 200、gzip magic 正確。服務忽略 Range，已限制只讀前段並關閉 | 現行官網寫 **CC BY-SA 4.0**、可商業／研究使用，版權仍屬原作者；檔內 README 卻寫 research only／原網站條款。再散布前應釐清衝突，暫不聲稱照片無條件可再散布 |

## 逐一核對

### 1. Penn-Fudan：目前最省下載的真實資料

- [作者／UPenn 官方資料頁](https://www.cis.upenn.edu/~jshi/ped_html/)；[實際 ZIP](https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip)。官方描述 170 張圖片、345 名有標註行人、UPenn 96 張／Fudan 74 張，每張至少一名行人。發布版 README 另說新增非常小或高度遮擋行人；網站的站立中大型行人描述與舊論文用途不足以證明它是完整小物件基準。
- 原生是 `PNGImages/*.png`、`PedMasks/*_mask.png`、`Annotation/*.txt`。TXT 為 **PASCAL Annotation Version 1.00 文字格式，非 VOC XML**；有 `Xmin,Ymin,Xmax,Ymax`，樣本明寫左上像素 `(1,1)`。mask 的 0 是背景、正數為 instance ID；用 mask 轉框時須明定座標契約，不能與 TXT 的 1-based 框混用。
- 小樣本 `FudanPed00001` 為 559×536、兩名行人；框 `(160,182)-(302,431)` 及 `(420,171)-(535,486)`，可用作非正方形與多物件的標註管線範例。只抽取 README／TXT，ZIP 目錄與小檔合計傳輸 46,575 bytes，未下載圖片包。
- **授權證據**：檔內 README：「Copyright and all rights therein are retained by authors or by other copyright holders」及「In most cases, these works may not be reposted without the explicit permission of the copyright holder.」公開可下載不等於可再散布；PyTorch tutorial 的程式或文件 license 亦不授權照片。
- **切分**：壓縮檔目錄未提供現成 train／val／test manifest；[PyTorch 官方教學](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html) 的隨機留出只是示範。應固定 scene／近重複圖片分組，保存不可變 manifest；同一場景、augmentation 與衍生 crop 都隨原圖留在同一 split。若按校園整批留出，需說明是跨場景分布測試。没有空圖，需另用可控資料或自攝並確認標註完整的負例，不把遮擋漏標當負例。

### 2. Pascal VOC2007：課綱一致的多類別基線

- [Oxford 官方頁](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/index.html)、[開發套件文件](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/htmldoc/index.html)、[官方統計](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/dbstats.html)。20 類、共 9,963 張；train 2,501、val 2,510、trainval 5,011，test 為其餘 4,952 張。
- 可用下載：[trainval](https://thor.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)、[test](https://thor.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar)。舊 `https://host.robots.ox.ac.uk/pascal/...` 此次 503；Oxford 官網的 `www.robots.ox.ac.uk/~vgg/projects/...` 連結會轉址至以上 `thor` 主機，官方 torchvision [VOC reader](https://raw.githubusercontent.com/pytorch/vision/main/torchvision/datasets/voc.py) 也使用它。
- `JPEGImages/*.jpg`、`Annotations/*.xml`、`ImageSets/Main/{train,val,trainval,test}.txt`；XML 有 class name、`bndbox`、`difficult`、`truncated`、來源／owner。VOC 為 1-based 座標；轉到 0-based 半開 `xyxy` 要一致處理，不只調 xmin/ymin 卻漏掉端點含義。已抽讀 `000005.xml`，包括同圖多個 chair、difficult／truncated 的實例。
- **授權**：官方「Database Rights」要求對 Flickr 圖片遵守其使用條款，公開頁未提供可涵蓋所有照片的 CC 授權，也未查到清楚的 annotation 再散布 grant。部分 XML 有 Flickr source ID／owner，但不能以此推定全部圖片同一權利。
- **切分與評估**：保留官方邊界，`trainval` 是 train＋val，不能訓練 trainval 又在官方 val 宣稱獨立評估。小型練習先 train 子集訓練、val 子集調參，test 留作最後檢查；跨 VOC 年份拼接先檢查圖片 ID／內容重複。不要把 `difficult=1` 刪除後當背景，應採 ignore 或明示簡化協議；AP50 也須區分 VOC2007 的 11-point AP 與其他積分版本。

### 3. COCO2017：困難場景與後段演化實驗

- [官方下載頁](https://cocodataset.org/#download)的實際本文：[download.htm](https://cocodataset.org/dataset/download.htm)；[標註格式本文](https://cocodataset.org/dataset/format-data.htm)；[授權本文](https://cocodataset.org/dataset/termsofuse.htm)。有 bbox、instance segmentation、crowd，適合單／多尺度與密集候選比較。
- 可用下載：[train2017.zip](https://images.cocodataset.org/zips/train2017.zip)、[val2017.zip](https://images.cocodataset.org/zips/val2017.zip)、[annotations_trainval2017.zip](https://images.cocodataset.org/annotations/annotations_trainval2017.zip)。相同官方 bucket 的 `https://s3.amazonaws.com/images.cocodataset.org/...` 此次也 HEAD／range 成功，可作下載故障備援；優先保留官方 canonical URL。
- 偵測使用 `instances_train2017.json`／`instances_val2017.json`：`images`、`annotations`、`categories`、`licenses`；bbox 是 **0-based 像素 `[x,y,width,height]`**，包含 `area` 與 `iscrowd`。category ID 不應直接當連續 class index，必須存 mapping。每張 image 的 `license` 指向 license 表，另有 Flickr URL。
- **授權**：官方 annotation／website 明示 **Creative Commons Attribution 4.0**。官方也明示 COCO 不擁有圖片版權，照片須遵守 Flickr 條款；再散布需按 image license 逐張核對 attribution、NC／SA／ND 等限制與用途。開源 API 的程式 license 與照片授權是不同來源。
- **切分與小物件**：118K train／5K val。2014 與 2017 使用相同圖片重新切分，混用年份會造成洩漏。真實小物件選樣應保存原 annotation `area` 與輸入後大小；[官方 evaluator](https://raw.githubusercontent.com/cocodataset/cocoapi/master/PythonAPI/pycocotools/cocoeval.py) 的 small area range 為 0–32²，但原圖 `area` 不等同 resize 後 bbox 面積，不能混成同一指標。`iscrowd=1` 不可改成一般單物件，也不應簡單刪掉當背景；正確保留 ignore／crowd 評估規則。
- 建子集先固定選樣條件與 image ID，再對每張入選圖保留所選任務全部相關實例，不能只留下觸發「小物件」選樣的那個框。重疊框的 IoU 不直接代表遮擋程度；還需看圖／mask。單獨拿 val2017 再分 train／val 只能稱自訂 COCO-val 子集，不能同時當官方 held-out 評估。

### 4. Oxford-IIIT Pet：頭部定位備选，不是小型多人資料

- [Oxford／作者官方頁](https://www.robots.ox.ac.uk/~vgg/data/pets/)；[images](https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz)、[annotations](https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz)，目前轉址到 `/pets/...`。37 品種、每類約 200 張。`annotations/xmls/` 是 PASCAL VOC **head** 框，`trimaps/` 是 foreground／background／unknown，不是多個物件的 instance mask。
- 只讀 annotations 壓縮前段 262,144 bytes，取得 README；未下載完整圖或標註包。官方頁稱圖片都有 head ROI，但此次未驗證 XML 是否完整覆蓋所有 split；實際選資料前必須交叉檢查 XML 與 split 清單，不能承諾全 7K 圖都能直接做 bbox 訓練。
- **授權衝突要保留**：現行頁明示 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)，商業／研究可下載；封裝 README 的原文為「Dataset is made available for research purposes only. Use of these images must respect the corresponding terms of use of original websites from which they are taken.」目前無證據解釋兩者差異，先提供官方下載與 manifest 建議；要隨教材再散布照片時請作者確認適用條款。
- 官方 `trainval.txt`／`test.txt` 作來源邊界；再從 trainval 分出 validation，test 不參與選模型。近重複圖片、裁切及 resize 衍生檔都隨原圖分組。以 cat／dog 或品種做類別都須固定映射；不能用單一 foreground trimap 宣稱存在多物件標註。

## 收集時固定保存的資料與待使用者事項

- 最少保存：資料集版本、來源 URL、原圖 ID、原始 split、來源／場景 group、類別 mapping、座標格式、annotation 來源、每張圖片的 license／attribution；下載後另算 SHA-256。**目前只查 HTTP 可用性，不拿 multipart ETag 當 SHA／MD5，也不宣稱已驗證完整封裝。**
- 固定困難集從訓練之外建立：一／多物件、同格同類兩物件、小物件、重疊、非正方形、極端長寬比。真實資料中的空標註通常只代表「沒有所選類別／沒有已標註實例」，不是已證明畫面完全無物件。人工幾何圖與自攝授權清楚負例可補缺口。
- **不需要使用者提供登入或 API key**，本次核對的官方下載均公開可讀。選取主線、建立來源 manifest 可直接往下做。
- 若目標包含「把照片打包進公開 repo／教材下載包」，才需要使用者決定再散布用途／商業需求，並對 Penn-Fudan 取得權利人許可、對 VOC／COCO 逐張核對，或改用自攝／權利清楚素材；Oxford Pet 的網頁／README 衝突亦需釐清。現階段可只散布下載連結及自己的子集 ID 清單。
- 完整 VOC／COCO 下載量、訓練張數與 GPU 預算仍待最小原型測量；這不是目前查來源工作的 blocker，不應先承諾訓練耗時或小模型品質。

下載驗證記錄保存在 `detection-sources/archive-probes.json`／`archive-probes-2.json`。引用授權原文與 annotation 小樣本分別在 `detection-sources/penn-sample-readme.txt`、`penn-sample-FudanPed00001.txt`、`voc-sample-000005.xml`、`pets-annotation-README.txt`；這些是研究核對檔，不是完整訓練資料。

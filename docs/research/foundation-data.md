# 分類 → ResNet → 單物件定位：資料準備研究

> 這是研究代理在子任務當時的查核報告。後續整合已套用前置配置並下載部分資料；目前狀態與操作步驟請以「前置準備」頁為準。

查核日期：2026-10-02。依 `/workspace/yolo-curriculum-outline-v2.md` 第 1–4 章與第 7 章資料約定整理；本文件是資料選型、來源與小型檢查結果，尚未製作教材、模型或訓練實驗。網路讀取使用 Python 標準函式庫，未安裝套件。資料與來源快照均位於 `/workspace/prep-research/`，不在 repo。

**推薦低負擔主線：同一份自製 RGB 幾何資料做 VGG 風格分類 → plain／ResNet 比較 → class＋bbox 單物件定位。** 先保留完全相同的圖片、class id 與切分，定位只多使用已存在的框；這比中途從灰階分類換成真實偵測資料更容易定位管線錯誤。Fashion-MNIST 可作真實資料的低負擔分類檢查；CIFAR-10 為可選的真實 RGB 分類對照；Oxford-IIIT Pet 為較後期的真實 RGB＋單物件資料候選。後兩者都不列入第一個機制驗證必下載清單。

## 1. 自製幾何資料規格（尚未實作）

| 項目 | 建議契約／規模 |
| --- | --- |
| 來源與權利 | 自己畫圓形、矩形、三角形，不使用外部照片、圖示或字型；日後發布時分別明列生成器及生成資料的授權。可選程式 MIT、自己產生的圖片／標註 CC0，但本次尚未產生或宣告這些資料。 |
| 圖片 | PNG、RGB、`uint8`、HWC；基本版 128×128，檔案像素 0–255。輸入模型時另明列 CHW／NCHW 和縮放規則；不要把檔案契約與 tensor 契約混為一談。 |
| 類別 | `0=circle, 1=rectangle, 2=triangle`；形狀決定 class，RGB 顏色、位置、大小獨立抽樣，避免只看顏色就能分類。 |
| 第 1–4 章共用圖 | 每張恰有 1 個物件，基本版不裁切、不遮擋。尺寸先取 16–64 像素，位置隨機；矩形長寬比約 1:3–3:1。以實際 raster mask 的非零像素求框，避免解析幾何邊界與繪圖取整不一致。 |
| 標註 | 一張一筆圖片紀錄：`image_id, file_name, width, height, objects`；每個 object 為 `class_id, bbox_xyxy`。框使用 **0-based、像素、half-open** `[x1,y1,x2,y2)`；保留原始像素框，正規化是衍生值。 |
| 最小檢查集 | 24 張固定、每類 8 張，供標籤／少量 overfit 檢查；不得當作泛化證據。另保留非正方形已知答案圖（160×96、96×160），檢查 resize／padding／框還原。 |
| 起始獨立切分 | 建議 train 1,500／validation 300／test 300，各 split 各類平衡；不同 split 使用分離的 seed 序列並記錄 generator version 與 seed。這是待測的起始規模，不是足夠訓練或時間的實測保證。 |
| 延伸到第 5–7 章 | 保留同一 class／bbox 契約，`objects=[]` 表示空圖；再加入 0–4 個物件、同格同類、小物件、長寬比、重疊與裁切等固定案例。遮擋時要事先固定用可見框或完整物件框，不能混用。 |

優點是下載量為零、標註可核對、RGB 與 bbox 可全程共用；限制是背景與物件過於規則，合成圖上的改善不能證明真實照片效果。生成時間、磁碟量、訓練時間與峰值記憶體均 **待測**。

## 2. 三個真實資料候選

以下 MB 採十進位；另保留伺服器原始 byte 數。壓縮量是官方頁面或 HEAD 所得，不等於已下載、解壓或訓練量測。

| 候選 | 已核對的資料與格式 | 壓縮量／下載負擔 | 適用性與限制 |
| --- | --- | --- | --- |
| Fashion-MNIST | 官方 README：60,000 train、10,000 test；10 類、28×28 灰階。四個 gzip 包含 IDX 影像／label；像素是 byte，並非 RGB 照片。 | 四檔 HEAD 合計 **30,878,645 bytes（30.88 MB）**；本研究只下載 5,148-byte test labels。 | 真實商品影像、低解析度、分類資料完整；適合分類和 plain／residual 對照。可將灰階複製成三通道以符合介面，但仍無色彩資訊；原資料無 bbox。若貼到大畫布並求前景框，須明列為二次合成資料，物件邊界也需規則，不能稱官方定位標註。 |
| CIFAR-10 | 官方頁：50,000 train、10,000 test，10 類，32×32 colour。Python tar.gz 含五個 train batch、一個 test batch及 metadata；每列 3,072 個 uint8，依 R／G／B 各 1,024，非 HWC 交錯順序。 | Python 檔 HEAD **170,498,071 bytes（170.50 MB，162.60 MiB）**；頁面顯示「163 MB」。完整資料未下載。 | 原生 RGB，適合小 CNN／ResNet 真實分類對照；沒有原生 bbox，無法直接共用到定位。把整張 32×32 都當物件框會讓定位退化成常數答案。官方頁未提供明確再散布授權文字，故不建議把圖片封裝進教材 repo。 |
| Oxford-IIIT Pet | 官方頁與包內 README：37 品種、species（cat/dog）、JPEG 圖片、PASCAL VOC XML **頭部 ROI**、PNG trimap。原圖尺寸可變；讀取時需明確轉 RGB。 | images HEAD **791,918,971 bytes**；annotations **19,173,078 bytes**；合計 **811,092,049 bytes（811.09 MB）**。本研究已下載 annotations，未下載 images。 | 真實 RGB、類別＋頭部框／整體 mask 可共用圖源。但 head 框不等於整隻寵物框；官方 test split 無 XML。若用 trimap 推導 bbox，可保留官方 trainval/test，但必須標示為「前景衍生框」，不能稱官方整體框。全37類不宜作第一個低負擔驗證；可先以 species 2 類、固定平衡子集試資料管線。 |

### Fashion-MNIST：官方 URL、checksum、再散布

官方資料 repo：[README](https://github.com/zalandoresearch/fashion-mnist/blob/master/README.md)、[LICENSE](https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE)。README 原始下載表使用 S3 website 的 HTTP URL；本次核對了同一官方 repo 中的 HTTPS 原始檔，四檔 HEAD 均回 200：

| 檔案 URL | HEAD bytes | README 官方 MD5 |
| --- | ---: | --- |
| [train-images-idx3-ubyte.gz](https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-images-idx3-ubyte.gz) | 26,421,880 | `8d4fb7e6c68d591d4c3dfef9ec88bf0d` |
| [train-labels-idx1-ubyte.gz](https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-labels-idx1-ubyte.gz) | 29,515 | `25c81989df183df01b3e8a0aad5dffbe` |
| [t10k-images-idx3-ubyte.gz](https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-images-idx3-ubyte.gz) | 4,422,102 | `bef4ecab320f06d8554ea6380940ec79` |
| [t10k-labels-idx1-ubyte.gz](https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-labels-idx1-ubyte.gz) | 5,148 | `bb300cfdad3c16e7a12a480ee83cd310` |

README 明列 `The MIT License (MIT) Copyright © [2017] Zalando SE`，LICENSE 亦有完整 MIT 條款，允許使用、複製、修改、發布、散布及銷售等。再散布原資料或子集時保留 copyright 與 permission notice；可將官方 LICENSE 與來源紀錄一起提供。不要把 loader 所屬套件的授權當作資料授權。

### CIFAR-10：官方 URL與授權證據範圍

[官方資料頁](https://www.cs.toronto.edu/~kriz/cifar.html) 指向 [cifar-10-python.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)，本次 HEAD 最終 redirect 為 `https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz`，回 200，頁列 MD5 `c58f30108f718f92721af3b95e74349a`。未下載完整檔，未驗證該檔 checksum。

官方頁請使用者引用技術報告，也說明 CIFAR 是 80 million tiny images 的已標註子集；**本次實讀頁面沒有明列 dataset license 或重包／再散布授權**。因此可保存官方來源與取得指引，把再散布照片的權利狀態列「未確認」，不要由公開可下載推論為 CC0／MIT，也不要採用第三方平台標籤補成官方授權。

### Oxford-IIIT Pet：官方 URL、標註覆蓋、授權不一致

[官方頁](https://www.robots.ox.ac.uk/~vgg/data/pets/) 的檔案連結：

- [images.tar.gz](https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz) → 本次 HEAD redirect 為 `https://thor.robots.ox.ac.uk/pets/images.tar.gz`。
- [annotations.tar.gz](https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz) → 本次下載 redirect 為 `https://thor.robots.ox.ac.uk/pets/annotations.tar.gz`。

現行網頁明列：`available to download for commercial/research purposes under a Creative Commons Attribution-ShareAlike 4.0 International License`，並說圖片 copyright 留在原所有者。按這份網頁授權，分享資料要署名、附 CC BY-SA 4.0 連結、註明修改；散布修改後資料需遵循同授權的 ShareAlike 條件，且不施加額外限制。這不直接要求所有訓練程式也採 CC BY-SA。

然而，**實際 annotations.tar.gz 內舊 README 寫**：`Dataset is made available for research purposes only. Use of these images must respect the corresponding terms of use of original websites from which they are taken.` 兩份官方材料對商業用途有不一致。本次不自行宣告哪一份消除了另一份限制；推薦研究下載留來源、將照片再散布範圍列待釐清，公開 repo 先收下載指引與資料清單。若日後要把照片或加工照片直接打包發布，再查核作者對舊文字的更新說明。

**實查 annotation 包，不能沿用網頁「每張都有所有標註」的概括句作 loader 假設：**

- 排除 Apple `._` metadata 後有 3,686 XML、7,390 trimap；官方 trainval 3,680 筆、test 3,669 筆，兩 split 沒有相同 image id。
- trainval 中 9 筆沒有 XML；test 中 3,669 筆全無 XML。若選頭部定位，從有 XML 的 trainval 另切 train／validation／test並明列篩選規則；不能把官方 test 當有框評估集。
- trimap 完整覆蓋兩 split；另有 41 個 trimap 不在兩 split 清單。以官方清單為準，不由 glob 自動把這些圖混回切分。
- README 定義 mask 值 `1=Foreground, 2=Background, 3=Not classified`。若選整體寵物定位，建議先固定用值 1 的非空前景求 tight bbox，框標「前景衍生」；值 3 的處理會改變 extent，必須固定。不要把 3 當另一個物件類別。
- class id 是 1–37、species id 是 1=cat／2=dog；轉成模型 0-based index 時保存明確映射。XML object name 是 cat/dog，不能單靠 XML name 還原37品種。

單一 `Abyssinian_1` 的 XML 圖尺寸 600×400、depth 3、head 框原值 `(333,72,425,158)`。同圖 trimap 為 600×400、8-bit grayscale，像素只含 1／2／3。本次標準庫解碼與 PNG CRC 檢查通過；值 1 的 0-based half-open bbox 是 `[107,81,444,328)`，值 1 或 3 的 bbox 則是 `[92,66,460,343)`。這直接顯示 head ROI 與由 mask 求的寵物區域是兩個任務。XML 的 VOC 座標慣例及端點轉換還需在實際讀取原圖疊框時確認，本次未下載原圖，未聲稱完成疊圖驗證。

## 3. 本次已取得的小型檢查資產

路徑均在 `/workspace/prep-research/data-probe/`。壓縮包沒有 `extractall`；只依固定 member 名稱讀出 README、split 清單、一份 XML 與 trimap 供檢查。

| 實際下載檔案 | bytes | SHA256 | 已驗證 |
| --- | ---: | --- | --- |
| `oxford-iiit-pet-annotations.tar.gz` | 19,173,078 | `52425fb6de5c424942b7626b428656fcbd798db970a937df61750c0f1d358e91` | 完整 tar.gz 可讀；MD5 `95a8c909bbe2e81eed6a22bccdf3f68f`，符合 [torchvision 官方 loader](https://github.com/pytorch/vision/blob/main/torchvision/datasets/oxford_iiit_pet.py) 的預期值；split／檔案覆蓋已盤點。 |
| `fashion-t10k-labels-idx1-ubyte.gz` | 5,148 | `67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5` | MD5 符合官方 README；解壓 10,008 bytes，IDX magic 2,049、10,000 labels、0–9 每類 1,000。這份檔不含圖片，不能用它宣稱 Fashion 影像管線已驗證。 |

下載／完整性證據：`foundation-download-log.json`、`foundation-download-budget.json`、`pet-annotation-summary.json`、`pet-trimap-summary.json`、`fashion-label-summary.json`。官方來源原文與 HEAD 記錄位於 `/workspace/prep-research/foundation-sources/`。

本研究總 network body 下載量為 **19,257,361 bytes（19.26 MB）**：來源文字 79,135 bytes、資料檔 19,178,226 bytes；HEAD 沒有下載 body。本地解壓輸出與紀錄檔不算網路下載量。未下載 Fashion train-images、CIFAR 或 Pet images；另行資料準備若下載更多檔案，需獨立登錄，不能混入這個探測額度。

## 4. 後續資料決策

1. 首先備妥自製幾何資料規格與固定檢查清單；同一圖片用於分類及定位，先驗證 RGB／座標契約、少量 overfit 與 held-out pipeline。正式規模與停止條件由原型實測決定。
2. 若要一份現在容易取得、可明確再散布的真實分類資料，選 Fashion-MNIST；保留 MIT notice。這是分類診斷分支，不是 RGB bbox 共用主線。
3. 若後續已有較多下載與訓練預算，Pet 的 trimap-derived bbox 可延續真實 RGB 圖片的分類＋定位，但要明列任務、衍生規則、split 與授權衝突。CIFAR-10 只在需要真實 RGB 分類對照時加上，無需為第一個 detector 額外下載。

尚未實測任何資料的完整影像讀取速度、訓練時間、準確率、AP 或記憶體；本文件的 byte／split／checksum／單一 trimap 結果是實測，其他資源與效果均待測。

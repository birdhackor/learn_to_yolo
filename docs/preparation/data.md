# 訓練資料前置規劃

查核日期：2026-10-02。本頁整理教材用到與可以接上的資料：來源、大小、授權、下載方式與切分規則。各節實驗都只用程式產生的合成圖或小型自製資料，不需要下載。Fashion-MNIST 是選用的真實分類資料，42 節教材沒有用到；四個真實照片偵測資料集的來源、格式、授權與下載方式也列在這裡，但教材沒有用它們訓練。

## 各階段的資料

| 教學階段 | 教材用的資料 | 適合接上的真實資料（教材沒有用到） |
| --- | --- | --- |
| 小 CNN／ResNet、單物件定位 | 各節程式直接寫出的小張量，或畫出的紅／藍矩形 | Fashion-MNIST 分類，見本頁下方〈[Fashion-MNIST：選用的真實分類資料](#fashion-mnist)〉。Oxford-IIIT Pet 是真實單物件定位的選讀備選：頭框（只框頭部）沒有涵蓋全部官方 split，全身框要從 trimap（把每個畫素標成前景、背景或未分類的遮罩）推導，授權說法也互相衝突 |
| 第一個多物件 detector | 手寫的小例子、`ShapeDataset` 產生的紅／藍矩形；8.2 節另把程式畫的紅、藍、黃三類矩形存成 PNG 與 JSON 標註 | Penn-Fudan：170 張真實行人照片、單一類別，完整 ZIP 校驗過 |
| v2／v3 與多類別比較 | 合成圖與手寫的小例子 | VOC2007 的固定小型子集，保留官方 train／val／test 邊界。查核過下載端點與封裝大小，沒有下載整包 |
| 現代版本、小物件／重疊 | 合成圖與手寫的小例子 | COCO2017 的固定子集：先取 annotation，再按固定 image ID 取圖，不必下載完整 19 GB 訓練包。查核過下載端點與封裝大小，沒有下載整包 |

比較不同版本時，要用同一份資料與同一組固定的困難案例；每換一版就換資料，就分不出差異來自改動還是資料。各節合成實驗的設定與執行結果，寫在各節頁面與〈[全套實驗與審查](../validation/curriculum.md)〉；教材沒有在真實照片上做各版本的完整比較。

`python scripts/download_data.py list` 列出 `data/manifest.json`（資料清單：每份資料的來源、大小、校驗值與狀態）的每一項。狀態是 `download-ready` 的 Fashion-MNIST 與 Penn-Fudan 能用 `fetch` 下載；合成資料是 `generated`，由程式產生；VOC2007、COCO2017 與 Oxford-IIIT Pet 是 `candidate`，下載器不會下載。對 `generated` 或 `candidate` 項目執行 `fetch`，下載器會回「Candidate only」並指回本頁：合成資料不必下載，候選資料請從本頁下方〈[偵測資料：大小、格式與下載](#detection-downloads)〉表格的官方連結取得。

## Fashion-MNIST：選用的真實分類資料

[官方 repository 與 README](https://github.com/zalandoresearch/fashion-mnist) 明列 MIT 授權；[原授權文字](https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE)另存一份在 `data/licenses/FASHION_MNIST_LICENSE.txt`。資料有 60,000 張 train、10,000 張 test，都是 28×28 灰階、10 類，以 IDX（存放多維陣列的簡單二進位格式）加 gzip 壓縮。它是分類資料，沒有 bbox。

四個原始檔共 **30,878,645 bytes（30.88 MB）**。`data/manifest.json` 記著每個檔的 bytes、官方 README 列出的 MD5，以及 SHA-256（由檔案內容算出的校驗值，內容只要改一點，值幾乎一定不同）。下載器預設存到 `data/downloads/fashion-mnist/`，這個目錄不進 Git。解壓後約 55 MB，但 loader 直接讀 gzip，不必先解壓：

```bash
python scripts/download_data.py fetch fashion-mnist
```

只取一個檔案，或指定存放位置：

```bash
python scripts/download_data.py fetch fashion-mnist --asset train-labels-idx1-ubyte.gz
python scripts/download_data.py fetch fashion-mnist --output /content/data
```

下載器先檢查既有檔案的大小與 SHA-256：相符的直接沿用，不符的要先移走再重試。它只下載與校驗，不解壓、不轉換，也不訓練。用 `--output /content/data` 時，檔案存在 `/content/data/fashion-mnist/`，之後執行 `scripts/run_fashion_cnn.py` 要加 `--data-root /content/data/fashion-mnist`。

### 讀取與 40 步訓練核對

`scripts/run_fashion_cnn.py` 用這份資料跑一條完整的分類管線：

- 讀檔前，四個 gzip 的大小與 SHA-256 必須和 manifest 相符；loader 再檢查 IDX 檔頭：magic number（檔頭最前面、用來辨認檔案種類的固定數字，圖片檔是 2051、標籤檔是 2049）、張數與 28×28，接著核對內容長度和檔頭相符，最後確認每個類別編號都在 0–9。
- 官方 train 用 seed 7 打亂，前 6,000 張當 validation，其餘 54,000 張當 train；官方 test 原樣保留。
- 灰階圖複製成三個相同的通道，好接上 RGB 輸入的 TinyCNN；這只為了共用介面，沒有增加顏色資訊。

紀錄 [`fashion-mnist-learning.json`](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/fashion-mnist-learning.json) 是其中一次執行：取 64 張 train、128 張 validation 與官方 test 的前 128 張，在 CPU 上做 40 次 Adam 更新（batch 8、lr 0.001）。紀錄存著設定、每步的 loss、正確率、權重是否改變、四個資料檔的 SHA-256，以及執行的日期、電腦與程式（`run_fashion_cnn.py` 和它直接或間接 import 的每個 repo 模組，都以 SHA-256 記下）。

這次執行證明管線接得起來：資料讀得進來，40 步的 loss 與梯度都是有限值，分類層的權重確實改變，validation 與 test 的評估也跑完。

它沒有證明模型學會分類。40 步的 loss 一直在 2.11–2.39 之間，和 ln 10≈2.30 差不多；10 類都給 1/10 機率時，交叉熵（CE）正好是 −ln(1/10)＝ln 10。validation 正確率 0.1406（128 張答對 18 張），test 正確率 0.0938（答對 12 張），都和亂猜的 1/10 相當。所以這不是 Fashion-MNIST 的成績，也和偵測無關。

在自己的電腦跑同樣的設定：

```bash
python scripts/run_fashion_cnn.py --train-steps 40 --subset 64 --eval-samples 128
```

結果寫到 `artifacts/runs/fashion-cnn/result.json`（不進 Git），不會覆寫上面的紀錄。換一台電腦，loss 的最後幾位小數可能不同。

## Git LFS 的可再散布封裝

Git LFS（Git Large File Storage）是讓 Git 另外存放大型檔案的擴充功能：repo 裡只放一個記著 SHA-256 與大小的小指標檔，檔案內容存在 GitHub 的 LFS 空間。`data/curated/fashion-mnist-v1.tar` 收著完整的四個原始 gzip 與 MIT LICENSE，共 **30,894,080 bytes**，SHA-256 為 `3f90f28291da3cb42cb75c4aca18014a6af397e307ee14a60516bc87e882977f`；封裝的 metadata 記在 `data/manifest.json` 的 `lfs_assets`。

這份封裝由 [Publish and verify LFS dataset](https://github.com/birdhackor/learn_to_yolo/actions/runs/36969537551) 工作流程（`.github/workflows/prepare-lfs.yml`）發布：從官方來源下載並校驗四個原始檔、打包、上傳到 LFS，再從空的 LFS 快取下載回來，核對封裝與其中四個原始檔的大小、SHA-256 以及 LICENSE 內容，全部相符才把指標檔推上 main。各步驟的結果記在 `data/hosted-lfs-run.json`。

只取這份封裝（沒有安裝 `git-lfs` 的電腦要先安裝它）：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/birdhackor/learn_to_yolo.git
cd learn_to_yolo
git lfs install --local --skip-smudge
git lfs pull --include="data/curated/fashion-mnist-v1.tar" --exclude=""
```

封裝裡的路徑和下載器的存放位置一致，照下面解開，loader 就能直接讀，不必指定 `--data-root`：

```bash
mkdir -p data/downloads
tar -xf data/curated/fashion-mnist-v1.tar -C data/downloads
```

網站本身不發布資料集；官方來源的下載器可作備援。Penn-Fudan 的 README 寫明照片通常不得未經權利人許可重新張貼，所以只用下載器取到自己的電腦，不打包進 repo。

## 偵測資料：大小、格式與下載 { #detection-downloads }

下表的大小是封裝檔的 bytes，不是解壓後的空間。Penn-Fudan 的完整 ZIP 與 Pet 的 annotations 下載後校驗過；其他大型檔只發過 HEAD（只問檔案資訊、不下載內容的請求）或小段 GET，端點讀得到不代表整個檔案完整。

| 資料 | 封裝大小 | 原生標註 | 官方下載 |
| --- | --- | --- | --- |
| Penn-Fudan | 53,723,336 bytes，53.72 MB | PNG instance mask＋PASCAL 1.00 TXT；不是 VOC XML | [ZIP](https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip)／[說明](https://www.cis.upenn.edu/~jshi/ped_html/) |
| VOC2007 | trainval 460.03 MB；test 451.02 MB | VOC XML、類別名、difficult／truncated | [trainval](https://thor.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)／[test](https://thor.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar) |
| COCO2017 | train 19.34 GB；val 815.59 MB；annotations 252.91 MB | JSON；`xywh=(左上x,左上y,寬,高)`，單位畫素；category ID 需重編為連續類別；iscrowd 是群體區域標記 | [官方下載頁](https://cocodataset.org/#download) |
| Oxford-IIIT Pet | 圖片 791.92 MB；annotations 19.17 MB | 頭部 XML、foreground／background／unknown trimap | [官方頁](https://www.robots.ox.ac.uk/~vgg/data/pets/) |

VOC 的連結指向 Oxford 現用的主機 `thor.robots.ox.ac.uk`；舊主機的 `https://host.robots.ox.ac.uk/…` 查核時無法下載。COCO 官方下載頁列的檔案網址以 `http://images.cocodataset.org/` 開頭；要用 HTTPS（加密連線）下載，不能只把開頭改成 `https://`，2026-10-04 查核時這樣會連線失敗。請改用指向同一批檔案的 `https://s3.amazonaws.com/images.cocodataset.org/…`，原因見〈[偵測資料](../research/detection-data.md)〉的 COCO2017 一節。

Penn-Fudan 完整 ZIP 的 SHA-256 為 `9095a9613c95586f1c7f2a327d454833d16e0f5e17e5f83d35027ffd315b48e2`（記在 `data/manifest.json`）。ZIP 內每個檔的 CRC 校驗碼都相符，圖片、mask 與標註 TXT 各 170 份。用下載器取得：

```bash
python scripts/download_data.py fetch penn-fudan
```

這個指令從作者的網站取得原始資料，不代表有權把照片重新上傳到公開 repo。能用下載器取得的兩份完整資料（Fashion-MNIST 與 Penn-Fudan）合計 84.60 MB。本書沒有 Penn-Fudan 的格式轉換程式，也沒有用它訓練。

Pet annotations 完整包的 SHA-256 是 `52425fb6de5c424942b7626b428656fcbd798db970a937df61750c0f1d358e91`。官方 split 是 trainval 3,680 張、test 3,669 張；XML 只有 3,686 份：3,671 份屬於 trainval（trainval 缺 9 份），15 份屬於不在官方 split 清單裡的圖，test 全部沒有 XML。要做整隻寵物的框，必須先按 trimap 訂好規則，不能直接拿頭框當全身框。

CIFAR-10（32×32 RGB、封裝約 170.50 MB）也可作分類對照；它沒有 bbox，教材沒有用到，manifest 也沒有收錄。

## 再散布與 LFS 的分工

- **自製資料：**由本專案的程式依固定 seed 產生；保存程式、seed 與切分就能重現，不必把圖片存進 Git。
- **Fashion-MNIST：**MIT，可依條件再散布，須保留 copyright 與 permission notice；精選封裝見上方〈Git LFS 的可再散布封裝〉，官方來源的下載器作備援。
- **Penn-Fudan：**檔內 README 保留原權利人的版權，並寫明通常不得未經許可重新張貼；本書只提供來源連結與下載器。
- **VOC：**官方要求遵守 Flickr 條款；照片的逐張權利與 annotation 的再散布授權都不明確。要用 VOC 子集時，只公開來源連結與自己選的 image ID 清單。
- **COCO：**annotation 是 CC BY 4.0，照片另依每張的 license 與來源條款；不能把所有照片都稱為 CC BY 4.0。
- **Pet：**現行官方網頁寫 CC BY-SA 4.0，包內的舊 README 卻寫 research only 與原網站條款。兩者的衝突沒有釐清，本 repo 不打包它的照片。

Git LFS 只用來放確認可再散布的精選封裝、自有權重與匯出模型，也就是 `.gitattributes` 指定的 `data/curated/`、`artifacts/checkpoints/` 與 `artifacts/exports/`；來源的下載快取不進 Git。這些來源都不需要登入或 API key。要再散布權利受限的照片，就得先取得權利人許可，或改用權利清楚的素材。

## 切分與資料契約

從真實資料取子集時，以下規則都要在訓練前固定：

1. 每個子集保存來源版本、image ID、split、class mapping、選樣條件與 checksum。大小記封裝 bytes，解壓後的空間另外量。
2. Penn-Fudan 先按來源、場景與近重複圖片分組再切；單人子集只挑恰好一個實例的圖片，不能把多人圖裡其他行人刪掉當背景。它沒有真正無人的圖片。
3. VOC 小型實驗以官方 train 訓練、val 調參、test 最後檢查；trainval 包含 val，訓練 trainval 之後就不能再把 val 當獨立評估。標成 difficult 的物件照 ignore 規則處理，不刪掉後當背景。
4. COCO 從 train 選訓練、從 val 選評估，不混用 2014 與 2017 的 split。`iscrowd=1` 表示難以逐一區分的同類群體區域，不當成普通的單物件框，也不刪掉後當背景。保存這個旗標；COCO 評估會特殊處理落入該區域的預測，詳見〈[COCO crowd 規則](../research/detection-data.md#coco-crowd)〉。本專案的合成資料評估器沒有實作這套規則，不能直接拿它報 COCO 指標。
5. 圖片與框同步 resize、padding、flip；bbox 端點、畫素或正規化座標、0-based 或 1-based，以及 class index，各有明確的轉換。
6. 小物件與重疊子集保留相關的完整標註；官方的 small 面積與 resize 後的大小分開記。來源 ID 清單在看過資料之後才固定，不事先憑空訂張數。

## 自製 RGB 幾何資料

`miniyolo.data.ShapeDataset` 是第 7 章起共用的合成資料。每張圖是暗色雜訊背景上的紅／藍實心矩形，類別就是顏色（紅＝0、藍＝1），這是刻意的簡化。預設 64×64、每張 0–2 個物件，可以是空圖。圖切成 4×4 格，每個物件整個落在一格內，不同物件用不同格，所以不會有兩個物件的中心落在同一格。每個索引用自己的 seed 產生，圖與框一起生成；train、validation、test 用不同的 seed（例如 `miniyolo.train` 預設 7、700、7000）。

第 1–4 章需要圖片的實驗不用 `ShapeDataset`，而是在各節程式裡直接畫紅／藍矩形；要分類時，類別同樣由顏色決定。同格兩物件的碰撞（第 5 章、7.2 節）與裁切（11.3 節）等情境，由各節手寫的小例子檢查。

延伸規格（`ShapeDataset` 不產生這些情境）：更多種形狀、不只靠顏色決定的類別、同格同類的兩個物件、遠小於一格的小物件、互相重疊的物件、極端長寬比與非正方形圖。擴充時沿用同樣的做法：held-out 的 seed 與組合事先固定，標註和圖一起產生。

合成資料上的成功，只說明對應機制在受控任務上可行，不能取代真實圖片的評估。

## 完整查核來源

- [基礎資料研究](../research/foundation-data.md)
- [偵測資料研究](../research/detection-data.md)
- [Git LFS 研究](../research/lfs.md)

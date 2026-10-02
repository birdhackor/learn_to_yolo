# 訓練資料前置規劃

查核日期：2026-10-02。本頁保留資料來源、下載與切分規格，並同步目前可執行範圍：ShapeDataset、Fashion-MNIST loader、簡化模型與42節教材已實作；尚未使用的真實偵測資料另列。

## 推薦路線

| 教學階段 | 資料 | 用途與目前狀態 |
| --- | --- | --- |
| 小 CNN／ResNet、單物件定位 | 自製 RGB 幾何圖形；Fashion-MNIST 分類支線 | 各節已有自產幾何資料與訓練程式；Fashion完整四檔已下載、校驗及40步實測 |
| 第一個多物件 detector | 幾何圖形＋Penn-Fudan | 人工控制場景搭配 170 張真實行人圖。完整包已下載、SHA-256／ZIP CRC 與各 170 份圖片／mask／標註通過 |
| v2／v3 與多類別比較 | VOC2007 固定小型子集 | 官方 train／val／test 保留邊界。下載端點與封裝大小已查核，未下載整包 |
| 現代版本、小物件／重疊 | COCO2017 固定子集 | 先取得 annotation，再按固定 image ID 取圖；不要求每章下載完整 19 GB 訓練包 |
| 真實單物件定位選讀 | Oxford-IIIT Pet | annotations 已下載並抽查；頭框不完整覆蓋官方 split，全身框需從 trimap 定義。授權材料有衝突，列備選 |

同一份資料與固定困難集跨版本沿用，避免每換一版就換資料，無法判斷改動的效果。各節合成實驗的張數、尺寸、batch、步數與本次量測已列於教材及[實驗清單](../validation/curriculum.md)；未跑的真實資料完整比較仍待量測。

## 已備妥：Fashion-MNIST

[官方 repository 與 README](https://github.com/zalandoresearch/fashion-mnist) 明列 MIT；已保留 [原授權文字](https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE)。60,000 張 train、10,000 張 test，28×28 grayscale、10 類、IDX gzip；它是分類資料，沒有 bbox。

四個原始檔共 **30,878,645 bytes（30.88 MB）**，已對照官方 MD5，並保存 SHA-256 於 `data/manifest.json`。本工作區快取位於 `data/downloads/fashion-mnist/`，完整解壓後約 55 MB。快取不進 Git，讀者可重下載：

```bash
python scripts/download_data.py fetch fashion-mnist
```

只取一個檔案或指定快取位置：

```bash
python scripts/download_data.py fetch fashion-mnist --asset train-labels-idx1-ubyte.gz
python scripts/download_data.py fetch fashion-mnist --output /content/data
```

下載器先檢查既有檔案的大小與 SHA-256；正確快取可重用，異常快取會要求移走後重試。它只下載封裝，不解壓、不轉換或訓練。

## Git LFS 的可再散布封裝

已建立 `data/curated/fashion-mnist-v1.tar`，包含完整四份原始 gzip 與 MIT LICENSE，共 **30,894,080 bytes**。SHA-256 為 `3f90f28291da3cb42cb75c4aca18014a6af397e307ee14a60516bc87e882977f`，封裝 metadata 記錄在 `data/manifest.json` 的 `lfs_assets`。

已透過 [Publish and verify LFS dataset](https://github.com/birdhackor/learn_to_yolo/actions/runs/36969537551) 完成遠端上傳、空 LFS 快取下載校驗及 Git pointer 提交；本雲端另從空快取下載，封裝與四份原始檔的 SHA-256 均相符。讀者可只取這份封裝：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/birdhackor/learn_to_yolo.git
cd learn_to_yolo
git lfs install --local --skip-smudge
git lfs pull --include="data/curated/fashion-mnist-v1.tar" --exclude=""
```

在沒有安裝 `git-lfs` 的機器先安裝它。網站本身不發布 dataset；官方來源下載器仍可作備援。Penn-Fudan 只留在本地快取，不打包重新公開。

## 偵測資料：大小、格式與下載

大小為已查到的封裝 bytes，不是解壓後空間。Penn-Fudan 完整包與 Pet annotations 已下載校驗；其他大型檔只做 HEAD／小段 GET，端點可讀不表示全檔完整性已驗證。

| 資料 | 封裝大小 | 原生標註 | 官方下載 |
| --- | --- | --- | --- |
| Penn-Fudan | 53,723,336 bytes，53.72 MB | PNG instance mask＋PASCAL 1.00 TXT；不是 VOC XML | [ZIP](https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip)／[說明](https://www.cis.upenn.edu/~jshi/ped_html/) |
| VOC2007 | trainval 460.03 MB；test 451.02 MB | VOC XML、類別名、difficult／truncated | [trainval](https://thor.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)／[test](https://thor.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar) |
| COCO2017 | train 19.34 GB；val 815.59 MB；annotations 252.91 MB | JSON；像素 xywh、非連續 category ID、iscrowd | [官方下載頁](https://cocodataset.org/#download) |
| Oxford-IIIT Pet | 圖片 791.92 MB；annotations 19.17 MB | 頭部 XML、foreground／background／unknown trimap | [官方頁](https://www.robots.ox.ac.uk/~vgg/data/pets/) |

VOC 舊 host 本輪回 503，已找到官方 `thor.robots.ox.ac.uk` 可用端點。COCO 原始圖片包及 annotations 的 HTTPS 端點亦已核對。

Penn-Fudan 完整 ZIP 已存入忽略的下載快取，SHA-256 為 `9095a9613c95586f1c7f2a327d454833d16e0f5e17e5f83d35027ffd315b48e2`，可重取得：

```bash
python scripts/download_data.py fetch penn-fudan
```

此指令取作者提供的來源資料，不代表有權將照片重新上傳公開 repo。分類與偵測兩份已備妥的完整來源合計 84.60 MB；Penn-Fudan真實偵測的轉換adapter與訓練仍待後續；Fashion-MNIST分類管線已實測。

Pet annotation 的完整包 SHA-256 是 `52425fb6de5c424942b7626b428656fcbd798db970a937df61750c0f1d358e91`。已查 split：trainval 3,680、test 3,669；XML 只 3,686 份，test 無 XML，trainval 缺 9 份。若做完整寵物框，必須按 trimap 制定規則，不能直接拿頭框當全身框。

CIFAR-10 亦已列入研究備選：32×32 RGB、約 170.50 MB，適合分類對照，沒有 bbox；不是目前必要下載。

## 再散布與 LFS 的分工

- **自製資料：**目前幾何資料已由本專案生成，保存規格、seed與split，不必將所有可再生圖片存進 Git。
- **Fashion-MNIST：**MIT，可依條件再散布，保留 copyright 與 permission notice。目前先採官方下載快取。
- **Penn-Fudan：**檔內 README 保留原權利人版權，明示通常不得未經許可重新張貼；先提供來源連結。
- **VOC：**官方要求遵守 Flickr 條款；照片逐張權利及 annotation 再散布 grant 未充分明確，先提供來源與自己的 ID manifest。
- **COCO：**annotation 為 CC BY 4.0，照片另按每張 license 與來源條款；不能把所有照片稱為 CC BY 4.0。
- **Pet：**現行官方網頁寫 CC BY-SA 4.0，包內舊 README 寫 research only 與原網站條款。未釐清前不打包照片到公開 repository。

Git LFS 適用於已確認可再散布的精選封裝、自有權重與匯出模型；來源下載快取不進 Git。這些資料源目前均不需 API key。只有決定再散布受限照片時才需另取得權利或換素材。

## 切分與資料契約先固定

1. 每個 subset 保存來源版本、image ID、split、class mapping、選樣條件與 checksum。尺寸先用封裝 bytes，解壓空間另測。
2. Penn-Fudan 按來源／場景／近重複分組再切；單人子集只挑恰好一個實例的圖片，不能刪掉多人圖其他行人當背景。它沒有真正無人的圖片。
3. VOC 小型實驗以官方 train 訓練、val 調參、test 最後檢查；trainval 包含 val，不可訓練 trainval 後仍把 val 當獨立評估。
4. COCO 從 train 選訓練、val 選評估；不混用 2014／2017 split。crowd／difficult 保留處理規則，不刪掉後當背景。
5. 圖片與框同步 resize／padding／flip；bbox 端點、像素／正規化、0／1-based 以及 class index 各有明確轉換。
6. 小物件與重疊子集保留相關完整標註；官方 small 面積與 resize 後大小分開。來源 ID 清單在資料探索後固定，不先憑空編張數。

## 自製 RGB 幾何資料規格

分類、單物件定位、多物件共用背景、顏色、形狀與 seed：先一個物件，再加入不同數量、空圖、同格同類兩物件、小物件、重疊、極端長寬比與非正方形圖。類別不單靠顏色決定，held-out seed／組合固定，annotation 與圖同步產生。

目前`ShapeDataset`已實作固定seed的64×64紅／藍矩形、空圖與最多2物件，中心刻意避開同格衝突；各節另以人工fixture檢查碰撞、裁切等情境。上段較豐富形狀、非顏色類別及困難集是延伸規格，尚未全部做成共用generator。合成資料成功只支持對應機制與受控任務，不能替代真實圖片評估。

## 完整查核來源

- [基礎資料研究](../research/foundation-data.md)
- [偵測資料研究](../research/detection-data.md)
- [Git LFS 研究](../research/lfs.md)

## 本輪真實資料讀取與短訓練核對

Fashion-MNIST四份gzip的大小與SHA-256通過manifest驗證，IDX的magic、張數、28×28與標籤範圍也由loader檢查。沿用官方train內seed7切分，再取64張train、128張validation、128張官方test，CPU跑40次Adam更新（batch8、lr=.001）。

本次validation accuracy為0.1406、test accuracy為0.0938，權重確實改變；步數很少，這是資料與分類訓練管線證據，不能稱為完整Fashion-MNIST成績。資料本身沒有bounding boxes。

重跑：`python scripts/run_fashion_cnn.py --train-steps 40 --subset 64 --eval-samples 128`。灰階重複成三個相同channel只為共用介面，沒有新增顏色資訊。[完整結果](https://github.com/birdhackor/learn_to_yolo/blob/main/artifacts/checks/curriculum/fashion-mnist-learning.json)。

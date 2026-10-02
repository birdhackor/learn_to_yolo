# Learn to YOLO

從小型 CNN、ResNet 到 YOLO 演化的中文教學專案。目前在前置準備階段，正式教材與模型尚未撰寫。

- [教學大綱](docs/planning/outline.md)
- [資料來源與下載規劃](docs/preparation/data.md)
- [GitHub Pages／Colab／Git LFS 操作步驟](docs/preparation/publish.md)
- [網站與 notebook 的編排規格](docs/preparation/architecture.md)

規劃網站已發布：<https://birdhackor.github.io/learn_to_yolo/>。目前提供教學大綱、資料規劃與前置操作步驟。

已完成 Fashion-MNIST 封裝的 Git LFS 上傳、空快取下載與 SHA-256 驗證，並實跑 GitHub Pages 部署。測試紀錄見 [LFS](data/remote-lfs-verification.json) 與 [Pages](data/pages-verification.json)。

## 本地預覽前置規劃網站

```bash
python -m venv .venv-docs
source .venv-docs/bin/activate
python -m pip install -r requirements-docs.txt
python scripts/validate_preparation.py
mkdocs serve
```

Windows 的啟用指令為 `.venv-docs\Scripts\activate`。正式建置使用 `mkdocs build --strict`。

## 取得已查核的小型分類資料

```bash
python scripts/download_data.py list
python scripts/download_data.py fetch fashion-mnist
```

原始資料存入 `data/downloads/`，已被 Git 忽略。Manifest 記錄來源、大小、校驗碼與授權；其他候選資料的限制見資料規劃。Git LFS 預留給明確可再散布的精選資料與模型產物；完整公開資料集以來源下載為主。

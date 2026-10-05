# 固定上游來源與授權

此目錄的四份 Python 檔案是官方 repository 固定 commit 的原始檔案，保留原文與上游版權／AGPL header；它們不是本課程自己撰寫的 toy implementation。審查 probe 以 AST 取出未修改的節點，在 CPU 執行；不匯入完整上游套件，不使用 GPU。

- `v10-tal.py`、`v10-metrics.py`：THU-MIG/yolov10，commit `453c6e38a51e9d1d5a2aa5fb7f1014a711913397`；同一 commit 的官方 LICENSE 完整原文見 `LICENSE-yolov10.txt`。來源 URL 與程式 SHA 見上一層 `top1-conflict-provenance.json`。
- `u-tal.py`、`u-muon.py`：ultralytics/ultralytics，commit `441632cdfd19e22e60a4b1b1999d46326ca51ec4`；同一 commit 的官方 LICENSE 完整原文見 `LICENSE-ultralytics.txt`。來源 URL 與程式 SHA 見上一層 `external-sources.json`。

LICENSE 的固定 URL、完整 SHA256 與下載時間保存在 `licenses-provenance.json`。探針作者的周邊測試程式與上游原始碼來源分開列明，授權文本不作改寫。

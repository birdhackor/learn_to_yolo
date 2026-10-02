# 網頁與 Colab 的前置規格

## 發布架構

採用 **Zensical** 的 modern 主題：Markdown 與小型圖片建成純靜態 GitHub Pages。`zensical.toml` 管理中文導覽、內建搜尋、深淺色模式與快速換頁；`requirements-docs.txt` 固定建置版本。獨立 `.ipynb` 放在同一個 repository，網頁以 Colab URL 連過去。網站依賴與模型依賴分開，建置網站不需要 GPU、PyTorch 或訓練資料。

```text
docs/                       網頁文字、圖與已保存的閱讀結果
zensical.toml               網站設定、主題與小節導覽
docs/planning/              大綱與研究，目前已有
docs/preparation/           前置規劃，目前已有
notebooks/                  可獨立開啟的 Colab notebook
data/manifest.json          資料來源與校驗碼
data/downloads/             本地或 Colab 快取，不進 Git
data/curated/               授權清楚的精選資料，使用 LFS
artifacts/checkpoints/      精選模型權重，使用 LFS
artifacts/exports/          精選匯出模型，使用 LFS
scripts/                    下載與基礎配置檢查
section-map.json            小節 ID、閱讀頁、notebook 與資料版本配對
```

## 每個小節的出版約定

1. 網頁先寫本節問題、前置知識與完成條件。
2. 說明、公式的符號／shape、示意圖與關鍵程式片段放在網頁。
3. 保存固定實驗的設定、成功／失敗圖與觀察，說明結果能支持什麼。網頁能直接閱讀這些成果。
4. 同一節附「在 Colab 執行」按鈕，以及先下載哪些資料、使用哪個模型分支、預期資源。
5. notebook 可由全新 runtime 從頭執行；先固定程式版本，再按需下載本節資料，不依賴上一節的隱藏狀態或自己的 Google Drive 路徑。
6. 正式教材的說明與 notebook 實驗對應同一個固定 release tag／commit。網站主題與建置工具可以獨立更新，不覆寫教材 tag；修改實驗或教學結果時才配對新版本。原始資料固定 checksum 與 split；不讓讀者看到新版說明卻執行另一版程式。

Colab URL 格式：

```text
https://colab.research.google.com/github/birdhackor/learn_to_yolo/blob/<ref>/notebooks/<section>.ipynb
```

`section-map.json` 記錄 42 個教學小節 ID，各自配對網頁與獨立 notebook。`ready` 表示兩份檔案已建立；CPU 執行與讀者審查的證據另行保存，不把此標記當成 GPU 訓練或效果驗證。正式教材的 notebook 與初始化程式固定到同一個 release tag，避免改標題讓外部連結失效。

## 純閱讀模式如何保留成果

網站不在讀者瀏覽時執行 Python。訓練曲線、框圖、shape 表與結果說明，要在作者的固定實驗後存成文字、表格、SVG／PNG 等網頁資源，再建置 Pages。notebook 保留必要的小型輸出即可，避免巨型 base64 讓 Git 與 Colab 變慢。

大權重、原始資料與影片不放進 Pages artifact。LFS 檔案的 Git pointer 也不能直接當圖片或下載內容；需要閱讀的圖使用普通 Git 的小型資源。發布流程只上傳 `site/`。

## Colab 取得資料與 LFS 的約定

主要資料從已查核來源下載到 runtime 快取，驗證 checksum，原始資料與衍生 split 分開記錄。runtime 中斷後可能須重新下載，因此各小節只取自己需要的部分。

clone repository 時可用 `GIT_LFS_SKIP_SMUDGE=1`，避免同時抓完所有版本的權重與資料。必要時才裝好 `git-lfs`，再對指定 LFS 路徑執行 pull。沒有 LFS 需求的段落，不要求讀者設定 LFS 或付費帳號。

預訓練權重用於免訓練推論展示時，網頁清楚標示。從零訓練的對照實驗使用同一資料及預算，不能把預訓練收益歸因給某個架構改動。

## 官方參考

- [GitHub Pages 自訂 workflow](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)
- [Colab 官方 GitHub notebook 範例](https://colab.research.google.com/github/googlecolab/colabtools/blob/main/notebooks/colab-github-demo.ipynb)
- [Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [Zensical](https://zensical.org/docs/get-started/)
- [Zensical 導覽設定](https://zensical.org/docs/setup/navigation/)

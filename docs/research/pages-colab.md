# Zensical、GitHub Pages 與 Colab

本站使用 **Zensical + Markdown 閱讀頁 + 獨立 `.ipynb`**。這頁記錄目前的配置；從原本 MkDocs Material 遷移後，維護網站請使用以下 Zensical 指令。

## 閱讀與執行分開

42 節教材以 Markdown 保存直覺、公式、SVG、關鍵程式與實驗結果。GitHub Pages 提供純靜態閱讀；每節另有固定版本的 Colab 連結。讀者不執行 Python，也能看到例子與答案。

建置網站不執行 notebook、不安裝 PyTorch、不下載 dataset。原始資料、模型權重與完整訓練產物放在 `docs/` 外；網頁只包含選出的文字與小型結果圖。

## 版本與原生設定

文件依賴固定為 `zensical==0.0.67`，Python 下限由套件 metadata 宣告為 3.10；本站本機與 GitHub Actions 使用 Python 3.12。這些版本只決定網站建置，不代表 CUDA 或訓練程式的相容性。

```text
zensical.toml                   網站設定與完整小節導覽
requirements-docs.txt            固定文件建置版本
.github/workflows/pages.yml      GitHub Pages artifact 發布流程
docs/lessons/                    42 節閱讀頁
docs/assets/                     SVG、結果圖、公式與字型資產
notebooks/                       固定版本的獨立 Colab notebook
section-map.json                 閱讀頁與 notebook 配對
site/                           產生的靜態網站，不進 Git
.cache/                         本地建置快取，不進 Git
```

`zensical.toml` 使用 `[project]`、`[project.theme]` 與 `[project.markdown_extensions]`。本案選 modern 主題、`zh-TW`、深淺色模式、內建搜尋與快速換頁；沒有安裝 MkDocs 或 Material 外掛。

來源：[Zensical 安裝](https://zensical.org/docs/get-started/)、[原生設定](https://zensical.org/docs/setup/basics/)、[PyPI metadata](https://pypi.org/pypi/zensical/json)、[MkDocs 相容性](https://zensical.org/docs/compatibility/mkdocs/)。

## 搜尋與公式的遷移重點

Zensical 使用自己的搜尋引擎，搜尋語言由 `theme.language` 決定，不沿用舊版 search plugin 的 `lang` 或 Lunr pipeline 選項。檢查時須實際搜尋中文與英文詞，確認結果連到正確的教材頁；建置成功不代表搜尋一定正常。

官方文件指出新搜尋介面的部分文字目前是英文；多語搜尋仍可使用。主網站導覽設定為繁體中文，英文術語也保留在教材中。

公式使用 Arithmatex 標記與本地 MathJax 3.2.2 資產。快速換頁不會重新載入整個網站，所以 `mathjax.js` 訂閱換頁事件，對新頁內容重新排版。驗證須包含直接開頁、點擊內部連結換頁與深淺色切換。

來源：[Zensical 搜尋](https://zensical.org/docs/setup/search/)、[快速換頁](https://zensical.org/docs/setup/navigation/)。

## 建置與預覽

在 repository 根目錄：

```bash
python3 -m venv .venv-docs
.venv-docs/bin/python -m pip install -r requirements-docs.txt
python3 scripts/validate_preparation.py
python3 scripts/validate_lessons.py
.venv-docs/bin/zensical build --clean --strict
.venv-docs/bin/zensical serve
```

`--clean` 清除建置快取，CI 每次從目前來源建站；`--strict` 在警告時中止。`site_url` 固定為 `https://birdhackor.github.io/learn_to_yolo/`，並保留原本各節的 `/lessons/<id>/` 網址。圖片與本地連結使用相對路徑，適用 Pages 的 `/learn_to_yolo/` 子路徑。

## Pages 發布

目前 `.github/workflows/pages.yml` 手動觸發。build job 安裝 `requirements-docs.txt`、執行兩個配對／資產檢查、執行 `zensical build --clean --strict`，再以 `actions/upload-pages-artifact` 上傳 `site/`；deploy job 使用 `actions/deploy-pages`。

build 只需 `contents: read`；deploy 只另加 `pages: write` 與 `id-token: write`。網站使用已設定的 GitHub Actions Pages source 與 `github-pages` environment，不需要新增 PAT 或改用 gh-pages 分支。Git push 與 Pages 部署分開，推送後仍要觸發 workflow，確認 build、deploy 和公開網址。

公開 Pages 與 repository 的存取條件見 GitHub 官方文件；將 repository 改為 private 前需另核對方案與網站 visibility。大型資料與 LFS object 不經 Pages 發布，網站 artifact 不包含 credentials、weights、資料封裝、LFS pointer 或符號連結。

來源：[Zensical 發布指南](https://zensical.org/docs/publish-your-site/)、[GitHub Pages 自訂 workflow](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)、[Pages 限制](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits)、[Git LFS 與 Pages](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)。

## Colab 與教材版本

每節 Colab 固定為 `lessons-v0.2.0`，實驗程式已在 CPU 執行，公開 notebook 原始檔也已核對。網站主題可獨立更新，不移動已發布 tag；修改教材實驗時再建立新 tag，重新執行相應檢查。網站建置不會清除 notebook 中已儲存的 CPU 輸出。

讀者執行 Colab 仍需 Google 帳號，GPU 分配依當時配額與可用性，不保證免費 GPU。從 GitHub 開啟 notebook 不會覆寫 repository；保留修改請選 **File → Save a copy in Drive**。notebook 副本不包含 runtime 已下載的資料與權重，重要產物需另外保存。

初始化格取得固定程式版本，跳過無關 LFS 下載；每節只準備自身所需的資料與依賴，不依賴上一節 runtime 或作者的 Drive。Google 登入、實際 Colab GPU 與完整 GPU 訓練不在目前 CPU 驗證範圍內。

來源：[Colab 官方 GitHub notebook 示範](https://github.com/googlecolab/colabtools/blob/main/notebooks/colab-github-demo.ipynb)、[Colab FAQ](https://research.google.com/colaboratory/faq.html)。操作細節見 [發布步驟](../preparation/publish.md)，教學效果與未完成項目見 [驗證範圍](../status.md)。

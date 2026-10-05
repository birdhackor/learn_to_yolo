# ViT／DINO v0.6.0：寫作、實跑與獨立審閱

本輪新增21.1–23.2十頁（ViT4／DINO4／延伸2），遵循 repo [clear-tutorial](../../../.agents/skills/clear-tutorial/SKILL.md)。主線0–20仍可獨立閱讀。這份檔案是協調者索引，原始審閱者的紀錄與限制不被此摘要取代。

## 第一輪：實際前文與逐段揭露

兩位非作者以沒有作者對話的上下文開始，按gate每次讀一個unit、保存當時理解與小變化預測，才取得下一單位。[ViT46份](first-read/vit.jsonl)＋[DINO93份](first-read/dino.jsonl)＝139份當場紀錄，其中新十頁63個units，其餘76份是實際前置閱讀事件，不能宣稱139個不同新單位。兩人都讀實際CNN、identity residual、15.1；DINO讀者再讀四頁ViT及4.1。共同檔案系統只是用指示控制揭露，不是技術隔離。

[ViT首讀](vit-first-reader.md)／[DINO首讀](dino-first-reader.md)、[凍結packets](frozen-packets.json)／[圖](frozen-figures/)、[來源manifest](materials-manifest.json)保留修前版本。baseline823232b不包含尚未提交的新稿，精確版本看SHA，不用HEAD冒充全文版本。

11個原始問題實例（包含兩人同一CNN措辭卡點）保存在[issues-original.json](issues-original.json)，處理另見[decisions.json](decisions.json)。修後[ViT局部核回](vit-recheck.md)／[DINO局部核回](dino-recheck.md)不倒改原note。DINO首讀未讀6.2，直到非盲讀複查才完整補讀並回到23.2；不假造第一次已具備AP前置。共同CNN問題由ViT讀者及導讀者核回，不追認DINO讀者也複查。

## 第二／三輪與實際頁面

[非作者技術審閱](technical-review.md)直接查ViT、DINO、DINOv2、registers、DINOv3原論文與固定官方code，另算中心／EMA、非預設checkpoint、全部bridge IoU、probe與160點曲線。真實notebook cell暴露__file__／kernel argv入口錯誤；修後cell與CLI分別再驗，原版本保存在[frozen-code](frozen-code/)。技術者沒有重跑官方DINOv2 forward，其核查使用實跑的SHA一致NPZ，沒有冒稱全部訓練均由他重跑。

[第三輪另一位讀者](transitions-review.md)先讀實際前置、再讀十頁，23.2前補讀4.1／6.2，沒有必改。收尾誤搜尋造成的其他repo內容接觸發生在判讀寫入後，原report如實記錄。這輪是全文非盲讀銜接，不是首次盲讀或技術執行。

[獨立導讀與視覺](guide-visual-review.md)實際看Zensical十頁桌機1280×800／手機390×844的19個必要圖片位置；缺字編號、裁掉低谷的曲線及手機公式修後再看。測search、instant navigation、MathJax、深淺主題與入口，並有界核導讀及舊前置修改。原始瀏覽器JSON／來源fingerprints留在本資料夾。截圖在ignored artifacts/runs，不把本機畫面當成公開網站驗證。

## 實跑與範圍

[52本notebook本機cell實跑](local-notebook-runtime.json)全部通過；96個pytest通過。本輪ViT60步、TinyDINO160步、frozen head200步與恢復均在CPU完成，正式每節JSON見artifacts/checks/curriculum。選讀固定官方DINOv2 ViT-S/14真的下載並CPU forward，22056576參數，224px兩張合成材料，權重／程式與特徵SHA有紀錄；checkpoint／weights／NPZ都不進普通Git。

沒有新增GPU費用、沒有長時間官方DINO或DINOv3訓練、沒有自然照片泛化或CNN／ViT／SSL架構排名，也沒有登入Google Colab測試。AI閱讀通過是找問題的輔助，不是真人學生學習成效。發布後Actions另驗公開頁面和從tag新建環境；結果以artifacts/checks/curriculum-publication.json與curriculum-release-bootstrap.json的source_ref/passed為準。

逐頁摘要位於reviews/21-*.md、22-*.md、23-*.md；其正文／SVG／code依賴以reviews/coverage.json的hash覆蓋。舊14頁僅追加本次範圍，不把舊0–20全書冒稱再完成全新盲讀。

發布前差異檢查：除Matplotlib產生的22-features.svg及其修前凍結副本含path屬性行末空白外，staged diff無格式問題。兩份SVG保留精確bytes以維持凍結與獨立核查SHA；沒有把這些行末空白當成圖面問題或更改原始審阅紀錄。

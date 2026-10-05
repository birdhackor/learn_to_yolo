> 第三輪 current 全頁銜接閱讀的當場筆記，非盲讀；在開 reference.jsonl 之前保存。下文原樣保留，沒有依後續 raw IDs 或修正文案改寫。審查者有其他章節既有作者／技術背景，不能將它算作本組第一輪揭露式閱讀。

# 第三輪當場閱讀筆記（原始 reference IDs 尚未開啟）

先讀 current 首頁、learning-path 全文與 outline 的 42 課概要。我的理解：先 0–7 接完整偵測，8 接自訂輸入，9–16 是獨立機制實驗而非逐代疊成同一網路，17 可在基础路線後做，18–20另有座標與影片前置。首頁只讀不需要 runtime；開 Colab 要帳號，維護頁屬另一條操作路線。

## planning/outline

我能分清第 7 章六個里程碑、原始真值／target／loss／decode／評估，及學會合成圖與學會真實照片的界線。完成條件要求成功／失敗與成本，不是新版機制一定勝出。小改變預測：若第 17 章只改顯示門檻，模型和固定 AP 候選不變，不能算新訓練；若變成模型宽度對照，须保留同資料／預算和validation決策後test。課程規劃頁可導到 course-research/feedback 看設計來源，或回learning-path開始讀；目前無卡點，長細節是維護／深入用途。

## glossary

這是卡住時查詞後點回課文的七組表，不需預讀背誦。※提醒同名詞在不同章有不同用途；C、N、P等字母要回本節定義。小改變預測：讀到attention的head，應用同名不同義定位到第15章，而不是借偵測head解釋；將DFL bin数改K，距離最大只到K−1，不是類別數。四種門檻與score/precision不同可從表重述。正文密度高，但分組、搜尋與詳見已給方法；沒有本輪browser，不能說手機表格可讀性已驗。

## status

當場理解：人工答案只證定義，少數CPU步確認梯度／更新，合成長些訓練仍不等於照片泛化；Fashion-MNIST只是額外分類管線。L4两次工程檢查與真人Colab／相機／真實照片AP分開。程式hash變→record過期；review hash變→重審，不是多跑GPU。小改變預測：若改7章decoder，绑定该模块的紀錄要重做，原舊日期不能假裝新跑；若只是自己Colab數字略變，不代表教材管線一定壞。当前 v0.5.0 尚待發布，頁内「網站發布後」驗證屬後續操作，我不会因为新tag暂时404认成帳號問題；待讀publish再核前後文字是否把計畫／既有結果混淆。

## preparation/architecture

我理解普通静态Pages只上傳site，notebook独立开Colab，docs模型環境分離；42節都有獨立case／配對／紀錄，ready只代表三者存在。從source摘code要Markdown data-excerpt標記，非改Python。小改變預測：改lesson標題不改section ID可維持URL；若改實驗要新release配對而不能只更新網站；下載改--output=/content/data，就要把分類--data-root指到/content/data/fashion-mnist。LFS pointer不会當Pages图，42節不需拉LFS；只有選真實資料才download/校验或pull。这条维护路線与首页只读／Colab学习并无权限混淆。

## preparation/data

我能照功能選資料：課內直接生成，Fashion是額外灰階分類、不是框；只有Fashion/Penn-Fudan能fetch，candidate用官方連結，generated無須下載。checksum過才沿用快取，下載不會替我做解壓／標註轉換／訓練。小改變預測：用--output /content/data後分類需--data-root /content/data/fashion-mnist；若加入Penn-Fudan多人圖，不能只留一人把其他人當背景。LFS包含license可以選拉，不等於可重新公開別的資料集照片。資料、封裝、權利與原split關係讀得清楚。

## research/foundation-data

首段把候選規格與現行兩色矩形／64²實作區分，我沒有把128²三形狀或1500/300/300當實測配置。Fashion無bbox；Pet頭框、trimap衍生框與官方test覆蓋是不同任務。小改變預測：把trimap值3也算前景會擴大框，需固定規則且重新核標註，不能稱同一官方頭框任務。HEAD bytes僅傳輸資訊，只有下載小label與完整annotations有checksum，無原圖片疊圖。遇到這些限制可回preparation/data選現行可用步驟。

## research/detection-data

我讀懂各資料是候選來源研究，完整下載另頁做，來源可讀／range成功不保證完整archive或再散布權利。Penn mask要單獨定0-based半開契約；VOC官方val不能在trainval訓練後當獨立；COCO IDs與crowd需特殊處理。小改變預測：將普通GT標為iscrowd=1後，不能用repo一般IoU一對一TP/FP規則，需要官方忽略區域規則；只改HTTP成https會主機名憑證不合，改成S3 bucket路徑而非關TLS驗證。這些建議不等於課程已訓練此dataset。

## research/lfs

我能區分一般Git小圖／manifest、選擇性LFS pack、外部完整資料。skip-smudge只保留pointer，include pull才取真正檔；本地fsck、遠端寫權限、從空快取重取是三件事。小改變預測：500MiB檔換一版又占整份storage；200讀者各抓一次約97.66GiB，成本歸owner，--depth1不能控制LFS大小。範本mini-voc是不存在的示意，已跑課程bootstrap時沿用tag目錄，不再clone固定/content/learn_to_yolo。不能把LFS路徑當Pages下載URL。

## preparation/publish

當場理解發布主線：新tag配對→列過期→需要遠端紀錄先推分支→GPU有過期才手動做→CPU重產→帶回紀錄和圖→人工核數字及非作者審查→本地檢查→不可變tag/Pages→公開驗證。權限分清Git contents/workflows、啟動Actions、Pages工作token、Modal/HF，首頁讀者不需這些寫權限。小改變預測：只有文案改就重審，不觸發GPU；bundle若把舊docs/lessons直接解覆蓋會丟新正文，故只取紀錄／圖後render-only，且render-only不補圖。若GPU結果失敗但hash匹配，不能留給dryrun當現行結果跳過；按指名檔恢復再修原因。本頁很長，但圖、三類檔表、編號與失敗處理能讓我列出下一步；本輪僅讀圖SVG來源，無browser。v0.5.0未發布的404可由步9解釋，沒有推成認證缺少。

## research/pages-colab

我能從純閱讀到獨立Colab，或以Python3.12+docsvenv預覽，不必模型環境。搜尋/MathJax靜態驗證只查索引與標記，仍需browser實際查詢/快速換頁。小改變預測：只改網站主題無須新tag，修改bootstrap／套件／實驗則要重建notebook與新tag；Colab先import錯版torch，改裝後要restart；存Drive只存notebook，runtime checkpoint要另下載。真人Google/Colab與GitHub runner bootstrap檢查分清。

## research/version-sources

我能把現代機制定位到固定commit，表的右欄是教材採用範圍，不能把教學列舉匹配或省BN當官方實作。小改變預測：16²以下STAL說法若改採論文，邊長10不再符合<8門檻；本頁分別寫論文<8与代码<16，不能靠名字把兩套混合。預設one与实际predict many+NMS也有區分。此輪讀來源對照頁，不聲稱重新閱讀外部repo所有程式。

## planning/course-research

事實段和對課綱的建議分開；來源有只讀公開作業名稱、沒執行repo／影片的限制。借CS231n診斷、D2L小實驗或Harvard紀錄格式，不代表它們教了全部現代YOLO或從零框偵測。小改變預測：MIT Lab2的輸出仍只有face/non-face分類，即便名稱detection也不能當框定位證據；把研究建議移為現行規格時須回outline/current課文，不可按舊候選連續模型當已實作。沒有必執行命令，數值變化預測不適用，採來源范围辨識。

## planning/feedback

AI模擬大綱讀者與公開帳號個案是兩類資料，都不等於本教材真人學習效果。每則卡點→採用章節能連回target、batch、座標、門檻與optimizer接線；沒有把issue作者猜測當已重現bug。小改變預測：一個人稱作業100%後還接錯batch，不能推論全班比例，應增batch=1/>1完整管線檢查；一個accuracy拼圖案例也不能替代泛化。無本頁數值實驗，採證據類型預測。

## validation/curriculum

我能查每節case原始JSON、單頁review，分辨逐節和補充長實驗；通過條件是本節定義/shape/梯度/流程，不是各版全模型品質排名。新證據hash沒變保留日期；validator不是重新訓練也不逐字核正文數字／圖。小改變預測：重產補充JSON但其他页旧数字未改，coverage不会必然提醒，维护者须搜旧值/檔名并人工核回；新tag先部署再从runner验证，current tag内是上一版publication。读得出已做CPU/L4与真人Colab／相機未做的界線。

## validation/gpu-smoke

我能说清40步对照、20步保存、CPU container核Volume/HF、全新L4续20，共80更新，Volume commit与git commit不同。没有HF_TOKEN仍可用Volume通過；有token却公开repo/上传失败是partial，不能发布checkpoint到公有repo。小改變預測：只是Modal wrapper改、沒動hash绑定实验模组，dryrun不會標過期，要手動重新workflow；有wrong config就step暂停不是自动开GPU；保存缺RNG可推論不能exact resume。warm_only实为warn_only warning允许有限容差、实际差0不保証跨设备；訓練／wall time不能拿暖機不相等比速度。没有在本輪取得HF私有权重或启任何GPU。

## README

我能从Python3.12 venv+cpu依賴到暖身／tests／42節runtime，輸出artifacts/runs不改原checks；python命令需activate或絕對venv路徑，Windows用Scripts和PYTHONPATH。小改變預測：steps160/samples32改epochs20/samples1024，batch8下每轮128更新、总2560，不能同时steps+epochs；--output新目錄留舊成果。Fashion下載與兩步選用，docs建站另venv；新tag發布維護流程導到publish詳細帳號與失败处理。本頁仍舊寫技術審查總述，新clear三階段在前首段與publish明确；后段需要核对原始问题再判是否矛盾。

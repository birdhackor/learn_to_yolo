# 六節獨立技術準確性審查 E

審查日期：2026-10-02。範圍：`14-feature-module`、`15-attention-bridge`、`15-area-attention`、`16-dfl-free`、`16-inference-head`、`16-training`。完整閱讀六份 docs、六份 case、六份 notebook 的全部 24 個 cells、六份 curriculum JSON，以及六張對應 SVG；未使用既有 reader reports 作結論。僅撰寫本報告，未修改教材、程式或既有證據，未使用 GPU、長訓練、push 或對外發訊。

初審結論：一處 P2 表述需要限定模式（16.2 的官方預設推論分支），其餘五節未發現確實技術錯誤。六節 CPU 重跑均成功，case SHA-256、stdout 與各節 JSON、notebook 儲存輸出均完全一致。toy 的 shape、數值與梯度檢查支持教材所界定的機制，不支持整版 YOLO 效能；教材已合理限制這些推論。

## 實際來源與版本契約

- Transformer：實讀 [Attention Is All You Need v7](https://arxiv.org/pdf/1706.03762v7)，§3.2.1–3.2.2、Eq. (1)，p.4；本地 `/tmp/yolo_sources/1706.03762.txt:164–220`。短引："apply a softmax function to obtain the weights on the values"、"extremely small gradients"。不是用摘要代替公式。
- YOLOv12：實讀 [作者原論文 v1](https://arxiv.org/pdf/2502.12524v1)，§3.1–3.4、§4.5；本地 `/tmp/yolo_sources/2502.12524.txt:167–241,292–319,504–529`。作者程式固定 `2abab7153a065fb2925e8088e9ca2b19016ab7d6`，並讀取相同 commit 的配置、parser；以下所有 v12 GitHub 連結均固定此版本。
- YOLO26：新增讀取 [作者原論文 v1](https://arxiv.org/pdf/2606.03748v1)，§3.2.1–3.3.3、§4.1、S3；本地 `/tmp/yolo_sources/accuracy-e-2606.03748v1.txt:311–470,666–715,1367–1376`。PDF SHA-256：`8dec3ca2adbbc78cbd4f338eb33235e4b9c13704e355e19d029fd0a4bc67fcff`。該論文 2026-06-02 已發布，不能視為尚無原論文。
- YOLO11 / YOLO26 官方程式：固定 `441632cdfd19e22e60a4b1b1999d46326ca51ec4`。實讀 block、head、loss、TAL、11/26 YAML、MuSGD、recipe；另取相同 commit 的 predictor、default.yaml、PyTorch backend、trainer，以追蹤預設模式與每 epoch 更新。以下所有 Ultralytics GitHub 連結均固定此版本。現行 TAL 僅用來核對本節明指的 YOLO26 查核版本，沒有用它反推原始 YOLOv8。

## 14-feature-module

### 實際核對來源

[YOLO11 YAML L20–30](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/cfg/models/11/yolo11.yaml#L20) 確實包含 C3k2、SPPF、C2PSA。[C2f L306–315](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/nn/modules/block.py#L306) 實際使用 `chunk(2, 1)`、`y.extend(m(y[-1]) for m in self.m)`、`torch.cat(y, 1)`；[C3k2 L1069–1105](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/nn/modules/block.py#L1069) 繼承 C2f 並選用 Bottleneck / C3k / attention 內部路徑；[Bottleneck L475–482](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/nn/modules/block.py#L475) 的 residual 受 shortcut 與相同 channel 條件控制。

### 核實結果

- `[2,8,8,8]` 經 8→8 投影拆成兩份 4ch，依序保留 a、b、b1、b2，concat 16ch、fuse 16→8，與 case、SVG、notebook 一致。
- 含 bias 的參數帳：plain `2×(8×8×9+8)=1168`；toy `72+4×148+136=800`。三 blocks 練習 `72+6×148+168=1128` 正確。
- `b1` 既可從 concat 直接收到梯度，也可經 b2 間接收到梯度；教材 docs:32–36 正確區分總梯度與 concat 槽梯度，case:50–65 沒有以第一個 block 非零梯度冒充中間路徑證明。
- MSE `1.0722→1.0049`、四個直接槽 gradient L1 `[.2967,.3349,.3481,.2779]` 重跑與證據一致。

確實錯誤：無。`SplitAggregate` 的內部寬度、ReLU / bias / 無 BN、固定 residual、無 C3k 開關都是 docs:7 已說明的合理簡化，不能要求它等同官方 C3k2 的參數數量或 activation。參數比較沒有被外推成 YOLO11 AP / latency 收益。

## 15-attention-bridge

### 實際核對來源

Transformer §3.2.1 Eq. (1) 是 `softmax(QKᵀ/√d_k)V`，縮放原因在本地 txt:184–201。v12 作者 [AAttn L1210–1261](https://github.com/sunsmarterjie/yolov12/blob/2abab7153a065fb2925e8088e9ca2b19016ab7d6/ultralytics/nn/modules/block.py#L1210) 使用 qk / v 卷積、多 head、`head_dim**-0.5`、key 軸歸一化和輸出投影。

### 核實結果

- feature 兩個 channels 的 flatten / transpose 得到 row-major tokens `[[1,0],[0,1],[1,1],[0,0]]`，Q/K/V 各 `[1,4,2]`，權重 `[1,4,4]`，還原 `[1,2,2,2]` 正確。
- 第一 query 的四分數 `[1,0,1,0]/√2`：分母 `2exp(1/√2)+2=6.05622996`，權重約 `[.33488077,.16511923,.33488077,.16511923]`，輸出 `[.66976155,.5]`；教材四位小數正確。
- 改第四 token `[2,0]` 後分數 `[1,0,1,2]/√2`，重新計算 softmax 得約 `[.22118,.10906,.22118,.44858]`、輸出 `[1.3395,.3302]`。練習在 SGD 前以副本計算，沒有被更新後 projection 污染。
- case:46 按 Q/K/V 三份權重分別核對非零梯度，權重更新與 reconstruction MSE `.1795` 一致。
- N=64→256 的顯式 affinity 元素 4096→65536，為 16 倍。這是 N² 儲存/pair 規模，不是宣称 FlashAttention 做近似或減少所有 pair。

確實錯誤：無。單 head、identity 初始 QKV、沒有位置卷積或輸出投影均明指為教學簡化。SVG 的列 / 欄 / softmax 軸與數學一致。

## 15-area-attention

### 實際核對來源

v12 論文 §3.2（p.3，本地 txt:225–239）短引："requiring only a simple reshape operation"，區分 `(H/l,W)` / `(H,W/l)`，l=4 將 attention 項由 `2n²hd` 降為 `.5n²hd`。作者 [AAttn L1222–1261](https://github.com/sunsmarterjie/yolov12/blob/2abab7153a065fb2925e8088e9ca2b19016ab7d6/ultralytics/nn/modules/block.py#L1222) 的實際順序是 row-major flatten，`reshape(B*area,N//area,C)`，區內多 head attention，恢復空間，最後 `proj(x+pp)`。

另核對實際位置與 residual，而沒有把後來 Ultralytics block 代替作者版本：作者 [配置 L21–45](https://github.com/sunsmarterjie/yolov12/blob/2abab7153a065fb2925e8088e9ca2b19016ab7d6/ultralytics/cfg/models/v12/yolov12.yaml#L21) 的 attention A2C2f 在 backbone P4 (`a2=True,area=4`) / P5 (`a2=True,area=1`)；該 turbo 配置 neck A2C2f 是 `a2=False`。作者 [ABlock L1295–1312](https://github.com/sunsmarterjie/yolov12/blob/2abab7153a065fb2925e8088e9ca2b19016ab7d6/ultralytics/nn/modules/block.py#L1295) 是 `x+self.attn(x)` 再 `x+self.mlp(x)`；[A2C2f L1353–1369](https://github.com/sunsmarterjie/yolov12/blob/2abab7153a065fb2925e8088e9ca2b19016ab7d6/ultralytics/nn/modules/block.py#L1353) 有可選 `x+gamma*cv2(cat(y))`，gamma 初始 .01。作者 [parser L1034–1038](https://github.com/sunsmarterjie/yolov12/blob/2abab7153a065fb2925e8088e9ca2b19016ab7d6/ultralytics/nn/tasks.py#L1034) 只為 L/X 增加 residual 選項。

### 核實結果

- 4×4、row-major、A=4 的連續 token reshape 對應四條水平帶，而不是 2×2 windows；A=2 是每組兩列。SVG 的 token 0/3/15、顏色、邊界與 case 完全一致。
- `[1,16,2]→[4,4,2]`、weights `[4,4,4]`，256→64；一般元素數 `B M N²/A` 與運算項 `O(B N² C/A)` 正確，投影成本仍在，教材沒有將四倍 pair 減少冒充整模型四倍 latency。
- query0 為 `[0,0]`，full / area 均勻平均 `.46875/.09375`；token15 加5使 full 首值 `.78125`、area 不變。A=2 首值 `.21875`，A=1 恢復 full。area 僅本層沒有跨區 pair，教材正確保留其他層間接跨區的可能。
- 原版位置卷積是在分區前作用於完整 v feature：`pp=self.pe(v)`（L1223–1225），最後加至 attention 結果。它本身可以跨 area 邊界，本例刻意移除，因此干預隔離只代表 toy 路徑。
- 論文 §3.4 / §4.5 Position Perceiver 描述 7×7；所固定作者程式 L1214 實際是 5×5 depthwise Conv。教材沒有指定 kernel，沒有混稱兩版本。這個差異值得保留，不能擅自把現行作者 cache 當成論文完全相同設定。

確實錯誤：無。沒有 multihead、pp、projection、ABlock / R-ELAN residual 的 toy 是隔離 area 機制所需的合理簡化。沒有宣稱是完整 attention block 或整版精度重現。

## 16-dfl-free

### 實際核對來源

YOLO26 論文 §3.2.2 Eq. (1)（p.6，本地 txt:347–386）明確給 `d∈[0,K−1]`、4K raw logits 與有限 support；S3（txt:1367–1376）短引："direct regression with an L1 loss is used instead"。官方 [YAML L9–10](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/cfg/models/26/yolo26.yaml#L9) 的 `reg_max:1`；[Detect L124–139](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/nn/modules/head.py#L124) 最終 box Conv 沒有 softplus，DFL / Identity 分支條件是 `reg_max>1`。[BboxLoss L131–154](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/utils/loss.py#L131) 保留 CIoU，加上 pixel / image-size 正規化的 L1。[bbox2dist L479–494](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/utils/tal.py#L479) 在未指定 reg_max 時不 clamp。

### 核實結果

- K=16 的 bin 0–15、均勻機率1/16、期待值7.5，以及 target1.25 的相鄰 bin 權重 `.75/.25` 正確。有限期待範圍不能表達18，但有較大 stride 的多尺度模型不因此必然不能偵測大物件。
- 直接 `[1.25,2.5,18,3]`、點(80,80)、stride8 得 `[70,60,224,104]`；signed `[-1,2,3,4]` 得 `[88,64,104,112]`，點可以在有效框外。另以固定源碼的 `bbox2dist` 驗證後者正是這組帶符號 target。
- Smooth L1 beta1 初始分項 `[.75,2,17.5,2.5]`、mean5.6875、四份梯度−.25、lr.5 首步+.125；300步收斂至 assert 範圍，六份输出一致。這是四個自由參數的擬合，沒有 feature/head 訓練證據，教材已說明。
- B1/P100/C2 時 raw tensor6600/600、float32 bytes26400/2400。沒有將該比值外推成完整模型 RAM / latency。

確實錯誤：無。官方 signed direct regression、toy Smooth L1、前節 / 下節的 softplus 被正確區分。docs:45 所稱超範圍 target 不能直接作 CE 索引，是對裸 CE 契約的說明；官方 [DFLoss L99–108](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/utils/loss.py#L99) 會先 clamp target，不能由這句推論官方 DFL 處理大框時必然報索引錯誤。這是可補充的細節，尚不足列為確實錯誤。

## 16-inference-head

### 實際核對來源

YOLO26 論文 §3.2.1（p.6，txt:359–386）同時介紹 one / many，稱 "One-to-One Head (default)"，並明指 many 使用 NMS。這是論文設計語境，和下列固定軟體的 API 預設需要分開。

官方 [Detect L194–204](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/nn/modules/head.py#L194) 訓練時 one 使用 detached features，推論按 `self.end2end` 選頭；[L217–247](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/nn/modules/head.py#L217) 直接距離解碼與獨立 sigmoid；[L279–295](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/nn/modules/head.py#L279) 兩階段 class-aware top-k、agnostic 選項與按當前模式 fuse 分支。

預設另實際追至官方 [recipe L26](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/docs/en/guides/yolo26-training-recipe.md#L26)："inference uses NMS by default, with `nms=False` selecting the NMS-free head"；[default.yaml L61](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/cfg/default.yaml#L61) 預設 None；[predictor L459–469](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/engine/predictor.py#L459) 傳 `end2end=self.args.nms is False`；[PyTorch backend L56–62](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/nn/backends/pytorch.py#L56) 在 fuse 前設定 head 模式。

### 核實結果

- toy raw `[2,6,4,4]→[2,16,6]`、中心8/24/40/56、stride16、top3 `[2,3,4]` / scores/labels `[2,3]`，圖和程式一致。
- 人工 `[1,.5,2,1.5]` 在(24,40)得到 `[8,32,56,64]`，logits `[0,ln3]` 得 `[.5,.75]`、label1。softplus fixture 反函數是有效手造輸入，沒有當成學得結果。
- backbone224、每 head54、train332 / deploy278 正確；只保留更新後的 one 权重，raw 前後相等。另做 one-only loss backward：backbone 全部 grad None、one head 有梯度，與 detach 描述一致。
- 官方普通 class-aware top-k 可重複輸出同候選的不同類；用固定 source 方法、scores `[[.99,.98],[.90,.10],[.80,.20]]` 實際得到 ids `[0,0,1]`。toy max-class→topk 得每候選最多一類，docs:50 已正確說明差異。

### 確實錯誤

**P2：官方推論分支被描述為無条件使用 one。** 初審 docs:7 的「推論取one-to-one」與 docs:9 的「官方部署使用one分支，再以top-k選輸出，省略NMS」缺少模式限定。讀者依所引用的固定版本直接使用官方 predict API，預設會經 many 分支與 NMS，而非教材所暗示的 one + top-k。YAML 的 `end2end:True` 不能單獨證明高階 predict 的預設，因為 predictor 明確在 fuse 前覆寫選頭。

建議改為：「論文描述原生 one-to-one NMS-free 路徑；在所查核軟體版本，設定 `nms=False` 才選 one 分支與 top-k，預設預測仍採 many 與 NMS。本 toy 只驗證選定 one 分支後的部署契約。」不用改 toy 或證據。兩套預設不能相互覆寫。

其餘差異是合理簡化：square loss 沒有 assignment、one 不會因 detach 自動唯一化、softplus 不是官方 signed regression、deepcopy 不等於 BN fusion；教材都已明指。

## 16-training

### 實際核對來源

YOLO26 論文 §3.3.2 Eq. (2)/(3)（p.7，txt:395–410）與 §4.1（txt:666–675）給 `.8/.2→.1/.9`、`max(E−1,1)` 線性排程，每 epoch 更新。官方 [E2ELoss L1327–1354](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/utils/loss.py#L1327) 給 many TAL10、one TAL7/topk2=1、相同 schedule；[trainer L638–639](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/engine/trainer.py#L638) 在每 epoch 結束呼叫 criterion.update。

STAL 原論文 §3.3.3 Eq. (4)–(6)（p.7–8，txt:434–470）短引："modifies only the candidate-selection mask"；§4.1（txt:676–683）明指 `s_min=8`、`s_ref=16`。固定官方 [TAL L58–59](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/utils/tal.py#L58) / [L334–348](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/utils/tal.py#L334) 的 floor 採第二 stride，原 GT targets 在 [L299–300](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/utils/tal.py#L299) 保留；[L373–379](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/utils/tal.py#L373) 在衝突處理後再做 topk2。

MuSGD 原論文 §3.3.1（p.7，txt:409–428）短引："weighted mixture of the Muon update and the SGD update"、"pure SGD for 1D parameters"。固定官方 [Muon L10–50](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/optim/muon.py#L10) 做5步近似 Newton–Schulz，未聲稱精確 UVᵀ；[L89–127](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/optim/muon.py#L89) 做 reshape、方向尺度處理；[L240–264](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/optim/muon.py#L240) 保留兩份動量 buffer，hybrid 中 weight decay 只經 SGD 項。真正分組在 [trainer L1183–1231](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/ultralytics/engine/trainer.py#L1183)，2D/4D 走 hybrid，其餘走 SGD，該版本 trainer 預設混合係數是 `.2/1.0`，不能只引用 MuSGD constructor 的 `.5/.5` 當完整 recipe。

官方 [recipe L21](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/docs/en/guides/yolo26-training-recipe.md#L21) 與論文 §4.1 都確認 Objects365-v1 150 epochs 再 COCO 微調；[recipe L158–172](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/docs/en/guides/yolo26-training-recipe.md#L158) / [L237–245](https://github.com/ultralytics/ultralytics/blob/441632cdfd19e22e60a4b1b1999d46326ca51ec4/docs/en/guides/yolo26-training-recipe.md#L237) 明指 checkpoint 實驗分支參數，不能直接當 main 的可配置參數。

### 核實結果

- E30、e0–29 的 schedule、e15 `.437931/.562069`、首/末 `.8/.2→.1/.9` 正確。獨立 heads / 同 target / 無 momentum SGD 的 one 有效 LR `.05b` 解釋成立，教材也正確限制為這個 toy。
- seed7 one 首值 `.318423/.313781`、MSE5.866377、weighted gradients `−1.146190/−.610803`、step後 `.375733/.344321`，mean / 導數2 / gain.2 / lr.05 全部正確。30更新後重新計算的 MSE fixed`.663073` / progressive`.016285` 與 JSON / notebook / 重跑一致。
- STAL 2×2例從 `[7,7,9,9]` 擴候選資格為 `[0,0,16,16]`，四個 stride8中心0→4，原 regression GT 不變。SVG按16px=256畫布pixel繪圖，每點/GT一致。資格池不等於最終正樣本、one至多一個、仍可能失去某 GT 都被正確區分。
- **原論文與固定程式有實際差異：** 論文 Eq. (5) 是邊長 `<8` 才替換16；固定 TAL L335 是邊長 `<16` 即替換16。教材 docs:52 明確說「查覈版本」，並引用該 code，其敘述正確，不列缺陷。建議補一句原論文與固定程式的門檻差異，避免讀者把 `<16` 誤寫成原論文 Eq. (5)。本題2×2在兩版本都得到相同0→4。
- MuSGD 的近似矩陣運算、額外動量、2D/4D與其他參數組、混合/decay成本概述與 source 一致。case 沒有執行 MuSGD，教材沒有作 MuSGD AP / convergence 驗證宣稱。

確實錯誤：無。progressive toy、STAL eligibility-only、未跑 MuSGD 都是明確的合理簡化。recipe 和版本差異應保留，不能用固定 source 的 default schedule 或 API 反推已發布 checkpoint 的完整訓練。

## 可重現的審查驗證

在現有 `.venv-model/bin/python`（Python3.12 / torch2.9.1+cpu）下重跑六個 case，各 subprocess exit0。沒有重新寫入既有 JSON；在記憶體比較每份 case 的 SHA-256 與 JSON 的 `case_sha256`，比較 stdout 與 JSON.stdout、notebook stream output，18項對照均為 True。

全部 notebook 各四個 cells：cell0教材/邊界說明；cell1固定 tag bootstrap與依賴；cell2修改實驗說明；cell3逐字等於相應 case，execution_count1，單個stdout output。讀取 cell1 不代表重新執行 clone/pip；未將待發布 `lessons-v0.2.0` 的狀態列為錯誤。

補充小驗算只在記憶體進行：one-only loss 的 detach 梯度隔離；抽取固定 Detect 的原始 top-k 方法驗證 `[0,0,1]`；抽取固定 bbox2dist 原始函式驗證負距離 target。這些不是完整 Ultralytics 模型/GPU效能測試。讀取 / 下載的來源在 `/tmp/yolo_sources`，repository 唯一新增文件為本報告。

初審 P2 已通知主代理；待教材修改後再作獨立複查，不以這份初審結論自動認定修正完成。

## 修後獨立定點複查與關閉狀態

2026-10-02 修後重新閱讀 `16-inference-head.md:7–11`、`16-training.md:5–17,46–58` 和完整 `docs/research/version-sources.md`，並重新對照實際原始缓存；保留以上初審紀錄。結論：**唯一 P2 已關閉；本組六節沒有剩餘已確認缺陷。**

- **P2 推論模式，已關閉。** `16-inference-head.md:7` 現在明述「固定版本的官方預測入口預設使用many分支加NMS」與 `nms=False` 才選 one；:9 將 top-k / 無 IoU 抑制限定為 NMS-free 部署；:11 單列原論文 §3.2.1 的 default one。所引固定 recipe 與來源頁:13–15 都一致。再次對照 `/tmp/yolo_sources/ultralytics-docs_en_guides_yolo26-training-recipe.md:26`、`accuracy-e-ultralytics_cfg_default.yaml:61`、`accuracy-e-ultralytics_engine_predictor.py:469`、`accuracy-e-ultralytics_nn_backends_pytorch.py:57–62`，確認預設 None 經 `is False` 得到 end2end=False，且選頭發生在 fuse 前；對照原論文 v1 txt:359–386，default one 和 many 的 NMS 路徑仍被準確保留，沒有以 API 現狀改寫论文。
- **Progressive Loss 用途，已確認。** `16-training.md:11` 現在明指後期加重「NMS-free推論模式」的 one。原論文 v1 txt:390–418 / Eq. (2)/(3) 就是對齊 NMS-free inference；固定 E2ELoss:1332–1354 的 .8/.2→.1/.9 公式沒有變，教材沒有由新的模式限定引入參數或梯度錯誤。
- **STAL 论文 / code 門檻對照，已確認。** `16-training.md:54` 与來源頁:15 正確記錄原論文 `<8→16`、固定 code `<16→16`。重新讀論文 v1 Eq. (5) / txt:434–447 和 §4.1 / txt:676–683，以及固定 TAL:58–59,334–348，兩個門檻和採用範圍吻合。2px 例同時符合兩條件，仍是候選資格0→4且原 GT 不变；沒有把資格擴張等同四個正樣本。
- **來源連結，已確認。** 新增的 YOLO26 arXiv ID `2606.03748` 與實讀 v1 PDF 標題、§3.2.1、Eq. (5) 對應；來源頁:11 的官方配置與:13 的 recipe 都保持固定 `441632cdfd19e22e60a4b1b1999d46326ca51ec4`，未混用 main 連結來支撐新限定。

本輪未重跑未變動的模型實驗；重新核對兩節 case SHA-256 均等於原 JSON.case_sha256，兩份 notebook cell3 仍逐字等於 case，初審實跑證據仍適用。本輪唯一寫入是本報告的追加紀錄。

複查正文 SHA-256：`16-inference-head.md` = `929125362c29f0ea59f8639b84d3e0be0413553847500c7f33b79771fa3161c7`；`16-training.md` = `5379cae6d1400f49532220a029f525cb926fc589d660df07ee33633d7be0073b`；`version-sources.md` = `d281f4d5aeefdaae2d1f23e0032ad90a8cfc04c13ab9e3406201f8130355485b`。

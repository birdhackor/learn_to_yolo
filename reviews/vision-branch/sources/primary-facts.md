# ViT → self-supervised DINO 權威來源核對

查證日期：2026-10-05。這份紀錄只查 primary papers、官方固定 commit 與獨立預訓練 probe，未讀其他作者教材草稿；不能當成分段首次閱讀或教材驗收。遵循 `.agents/skills/clear-tutorial/SKILL.md` 與 `AGENTS.md` 的版本／簡化範圍約定。

## 原始論文與可支持的位置

| 論文（讀到的固定版本） | 主要位置 | 可支持的事實 |
| --- | --- | --- |
| [An Image Is Worth 16×16 Words, 2010.11929v2](https://arxiv.org/abs/2010.11929v2), ICLR 2021 | §3.1、Fig.1、Eq.1–4；§3.2 | 非重疊 P×P RGB patches，展平後共用線性投影到 D，額外可學 CLS 與可學位置向量；pre-LN attention／MLP 與殘差；最終 CLS 供分類。改解析度需處理位置向量插值。 |
| [Emerging Properties in Self-Supervised Vision Transformers, 2104.14294v2](https://arxiv.org/abs/2104.14294v2), ICCV 2021 | §3.1、Fig.2、Alg.1、Eq.1–4；§3.2；§4.2.2；§5.3 | DINO teacher／student、cross-view soft target、multi-crop、EMA、center／sharpen；backbone feature 與 projection head 的用途分開；frozen linear／kNN；特定評測中的 patch／attention 性質與抗 collapse 實驗。 |
| [DINOv2: Learning Robust Visual Features without Supervision, 2304.07193v2](https://arxiv.org/abs/2304.07193v2), TMLR 2024（初版 2023） | §3；§4；§5；§6.5；App.A | 資料整理 LVD-142M；CLS DINO + masked-patch iBOT；分開 heads；teacher SK 3 iterations；KoLeo；高解析訓練；大模型蒸餾成較小模型。 |
| [Vision Transformers Need Registers, 2309.16588v2](https://arxiv.org/abs/2309.16588v2), ICLR 2024 | §2（high-norm token 分析與 register 解法）、§3（實驗）；官方 README 2023-10-26 更新與 hub 變體 | 額外 tokens 供內部運算可改善特定 dense-feature artifacts。此為後續工作，DINOv2 baseline 與 register 變體並存。 |
| [DINOv3, 2508.10104v1](https://arxiv.org/abs/2508.10104v1), 2025 technical report | §3.1–3.2；§4.1–4.3、Eq.2–3；§5；App.A、App.C | 規模／資料／架構與訓練 recipe 變化；long-run patch consistency 退化；Gram anchoring 及 snapshot refresh；後續解析度訓練與蒸餾。 |
| [DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection, 2203.03605v4](https://arxiv.org/abs/2203.03605v4), ICLR 2023 | 標題、Abstract、§3 | IDEA 系列 DETR detector；與 Caron 等人的 self-supervised DINO 為不同團隊、任務與算法。 |

論文 PDF、官方 source 的 URL、內容 SHA-256、檔案大小保存在 [`source-downloads.json`](source-downloads.json)。PDF 實體在 ignored `artifacts/runs/vision-sources/`；不把下載權重或原始論文放普通 Git。

## ViT：不要把架構與訓練方法混在一起

- ViT 原論文 §3.1 的 CLS 是可學 token，其最終狀態聚合全圖資訊供分類。patch 輸出仍各有對應的原始圖塊位置，但 attention 後已可包含全圖內容。
- §3.1「Inductive bias」指出 CNN 的局部性與 2D 鄰域結構內建較強，ViT self-attention 可全域互動。這不支持「ViT 完全沒有局部結構」：patch extraction／projection 本身使用局部圖塊。
- 官方 [models_vit.py](https://github.com/google-research/vision_transformer/blob/64801f1b3b367b3611cc27a3d45cc22870a36fb3/vit_jax/models_vit.py)：`Encoder1DBlock` L136–156 是 pre-LN MSA／MLP residual；`Encoder` 加 position embeddings；`VisionTransformer` 用 patch-stride Conv 實現共用 projection，再選 CLS／其他 pooling。
- 本教材若用小型 torch ViT，應標明教學簡化，不能因相同 patch／attention 名稱就叫官方 ViT 權重或重現原論文性能。

## DINO 2021：精確訓練次序

固定官方 source：[main_dino.py](https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/main_dino.py)。

1. `DataAugmentationDINO` L419–464 生成兩個 224 global crops 以及多個 96 local crops；L318 teacher 只處理 `images[:2]`，L319 student 處理 all crops。小教材若只有兩個 global views、沒有 local crops，需稱簡化的 cross-view 核心機制。
2. `DINOLoss.forward` L384–390 用 student temperature，teacher 先減「本次 forward 前的」center，再除 teacher temperature、softmax、detach。L394–402 比較不同 views，跳過同一 view；teacher soft target 不是人工類名答案。
3. L403 呼叫 `update_center`；L411–416 平均的是 raw teacher logits（含分散式 all-reduce），再 EMA 更新 center。不能用本批新 center 提前重算同批 target，或把概率平均當成原公式。
4. L326–344 student 由 optimizer 更新；L346–350 teacher 在 student optimizer 後由 EMA 更新；L210–211 teacher params 不需梯度。teacher 的 EMA 與 center 的 EMA 是兩個不同的量。
5. paper §3.1「Avoiding collapse」與 §5.3：centering 限制某一維主導但促向 uniform；低溫 sharpening 作用相反。兩者在 momentum-teacher 設定下互補，不能把「有 EMA」單獨說成任何設定皆保證不 collapse。
6. paper §3.1「Network architecture」：`g = h ∘ f`；downstream features 用 backbone `f`，不是 K 維 projection distribution。ViT 的 `h` 接 CLS。`main_dino.py` L55 預設 K=65536，這些 slot 沒有貓／狗類名；DINO paper §3.2 特別說 CLS 不接 labels／supervision。
7. paper §3.2 Evaluation protocols：linear probe 在 frozen features 上訓練分類器；kNN 使用特徵、鄰居的資料集標籤投票。表示預訓練不用 labels，不等於下游所有評測都不需要 labels。
8. 參數要分清 paper 與 code default：paper §3.2 描述 teacher temperature .04→.07 的 30-epoch warmup；固定 source L68–75 default 最後仍 .04、warmup epochs 0。teacher momentum .996→1 cosine（L61–63）。簡化固定温度／momentum不等於完整 recipe。

官方 [vision_transformer.py](https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py) 的 `get_last_selfattention` 讓可視化 attention 和 backbone patch features 區分；attention heatmap 本身不是 detector 的 box／class confidence。

## DINOv2：版本差异与設定差异

教材實際 preload 固定官方 commit：[`e1277af2ba9496fbadf7aec6eba56e8d882d1e35`](https://github.com/facebookresearch/dinov2/tree/e1277af2ba9496fbadf7aec6eba56e8d882d1e35)（2024-02-22）。本次也抓取了 2026 HEAD 供查資料，實跑沒有用 HEAD。

- paper §4 的 image-level DINO loss 比較不同 crop 的 CLS；patch-level iBOT 對 student 被 mask 的位置，比較 teacher 未被 mask 的相同位置表示。這不是 pixel RGB reconstruction，也不是只有 CLS 的 DINO2021 recipe。
- paper §4「Untying head weights」採分開 DINO／iBOT heads。官方 [`train/vitg14.yaml`](https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/configs/train/vitg14.yaml) 明設 `ibot.separate_head: true` 與 `train.centering: sinkhorn_knopp`。
- SK 是 teacher 目標正規化的一個替代分支，不是「先原 DINO subtract-center softmax，再多做 SK」。[`ssl_meta_arch.py` L199–225](https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/train/ssl_meta_arch.py#L199) 是 `centering`／`sinkhorn_knopp` 互斥 branches；[`dino_clstoken_loss.py` L35–61](https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/loss/dino_clstoken_loss.py#L35) 在 temperature 後對 K×B assignment 交替按 prototype／sample 正規化 3 次。student 仍用 softmax。
- repo 的 [`ssl_default_config.yaml`](https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/configs/ssl_default_config.yaml) default 仍是 `centering`、`separate_head:false`。所以版本頁應說「paper／完整訓練 recipe」，不可說所有官方 configs 一律只 SK／一律分頭。
- paper §4 的 KoLeo 是 normalized sample features 的 batch 近鄰距離之 `-mean(log distance)`，鼓勵特徵分散；與 cross-view teacher matching 是不同目的。[官方 loss](https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/loss/koleo_loss.py) 排除每個 sample 自己作近鄰。
- [README](https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/README.md) 第一行記 2023-10-26 新增 register backbones；[`backbones.py` L26、L64–68、L98–109](https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/hub/backbones.py#L26) 清楚分開 baseline 0 registers 與 `dinov2_vits14_reg` 的 4 registers。本次實載 `dinov2_vits14` 的 shape 是 `[2,0,384]`。
- [`vision_transformer.py` L254–269](https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L254) 的 `x_norm_*` 是 LayerNorm 後的 CLS／register／patch features，不是已 L2 unit normalized。計 cosine 要另 L2 normalize。
- [README License](https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/README.md#license) 明文 code 與 model weights 都 Apache-2.0；[LICENSE](https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/LICENSE)。

## DINOv3：Gram anchoring 的適用範圍

固定官方 commit：[`6876159a11b4df116f30f667f8c9888617df0751`](https://github.com/facebookresearch/dinov3/tree/6876159a11b4df116f30f667f8c9888617df0751)。

- paper §3.1–3.2 整合資料規模與混合、6.7B ViT teacher、patch16、axial RoPE、4 registers、長訓練與訓練穩定化；版本差異不只有一個 Gram loss。不能用教材微型模型說已重現整套 DINOv3。
- §4.1 退化的是 long-run patch 間一致性；registers 可改善 high-norm artifact，但仍會發生 patch consistency 下降（§4.1 與 App.A）。
- §4.2 Eq.2：每張圖的 normalized patch matrices `XS, XG` 為 P×D，各自形成 P×P `X Xᵀ`；比較兩者 Gram 的差，而非逐維要求 feature vectors 相等。保留 patch 間相似關係仍允許 feature 空間整體旋轉等變化。
- 初始 Gram teacher 是早期 EMA teacher 的 snapshot，其 dense features 比晚期更佳；作為額外 refinement phase，paper 為效率於 1M 主訓 iterations 後才加入。不是把初始化 random model 固定一生，也不是每一步用即時 student 自己當 target。
- §4.2 每 10k iterations 把 Gram teacher snapshot 更新成當時 main EMA teacher；App.C 限最多 3 次 refresh。[官方 Gram config](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/configs/train/dinov3_vit7b16_gram_anchor.yaml#L46) 寫 `ema_teacher:false`、`rep_update:true`、`update_frequency:10000`、`it_first_update:1010000`、`max_updates:3`；freeze 是兩次 refresh 間的狀態。
- §4.2 只在 global crops 計 LGram；§4.3 用更高解析 Gram teacher feature 降採樣到 student grid。Eq.3 refinement 仍含 DINO、iBOT、KoLeo，不是用 Gram 完全替代它們。
- [`gram_loss.py` L49–84](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/loss/gram_loss.py#L49) normalize features、形成 patch Gram、MSE。類別 constructor 的 defaults 有負相似 clipping 選項，但完整官方 Gram config 明設 `remove_neg:false`、`remove_only_teacher_neg:false`；不可只看 constructor default 就描述論文 recipe。
- [README](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/README.md#license)／[LICENSE.md](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/LICENSE.md) 是 DINOv3 License，不能沿用 DINOv2 Apache-2.0 結論。官方權重入口要求 access request；本次未下載或執行 DINOv3 checkpoint，不需以此要求使用者 HF token。

## 23.1 可重跑的公開官方 feature 實測

唯一選讀下載命令（主 lesson case 不執行下载）：

```bash
.venv-model/bin/python scripts/run_dino_pretrained.py \
  --report artifacts/runs/dino-pretrained/report.json
```

腳本只需既有 torch、numpy、Pillow；不用 torchvision、GPU、憑證。官方 source ZIP 固定 commit+SHA，下載後使用原封不動的 local torch.hub entrypoint，核權重 SHA 再 `weights_only=True`、`strict=True` 載入。cache 每次核 SHA，code 每次從已核 archive 重新展開。本次測了 first download、cache repeat、獨立 NumPy 數值檢查和拒絕錯誤 cache SHA。hash 是本次由官方 HTTPS artifact 量得的 SHA，非廠商獨立數位簽章。

| 觀察 | 實際結果 | 可支持的結論 |
| --- | --- | --- |
| checkpoint | 88,283,115 bytes；SHA `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9` | 公開官方 baseline DINOv2 ViT-S/14 可用，沒有 gated 403。 |
| parameters／input | 22,056,576；兩張 RGB 224×224 控制合成圖；CPU1 thread | 有載入並執行官方模型，非 random／近似小模型。 |
| shapes | CLS `[2,384]`；patch `[2,256,384]`；register `[2,0,384]` | 16×16 patch grid 可接回空間特徵；CLS 另供全圖表示。 |
| query | 圖A row4,col4，resized pixel box `[56,56,70,70]`，紅 square 內 | patch序列 row-major，圖塊与向量位置可核對。 |
| 圖B top1／top2 | row11,col11 紅 square 內 .833138；row4,col4 綠 disk 內 .768546 | 本例最近的確為移動的 square，同位置 disk 也有正相似度。不能斷言模型理解形狀／顏色語意或有自然影像泛化。 |
| CLS cosine | .985041 | 這對圖全局表示相近；非分類／偵測正確率。 |
| Gram代數 | channel reverse 後 patch Gram max difference `1.61e-6` | 向量改變而 patch相似關係保留（float誤差）。只是在DINOv2 feature上示範Gram性質，未跑DINOv3或Gram訓練。 |
| cacherepeat | inputs RGB SHA、CLS cosines、完整 neighbor map 逐項相等 | 同一CPU設定可重做；不是跨硬體bitwise保證。 |

已保存 [`pretrained-report.json`](pretrained-report.json) 和 [`pretrained-probe.json`](pretrained-probe.json)；原始 PNG／features NPZ 在 report 記錄的 ignored paths。前處理為 RGB→bicubic direct square224→`uint8/255`→NCHW→ImageNet mean/std；本次原圖本來就是224，沒有 crop。使用者自行圖片要以兩个 `--image-a`／`--image-b` 同時指定；其不同 aspect ratio 會直接 resize 成 square，report會記錄。

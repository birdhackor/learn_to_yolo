# Curriculum technical accuracy review C

日期：2026-10-02。独立审查范围：08-own-images、08-own-data、09-anchors、09-anchor-clustering、10-multiscale、11-csp、11-fusion。未读取旧 reviews 作为结论依据。

结论：这七节没有发现 P0/P1 技术错误。核心数学、数据／checkpoint 契约、原始机制来源以及实际产物的解释均可核实。10-multiscale 的补充曲线有一项 P2 横轴标号问题，不改变训练结果或 AP 结论。

## 审查与验证范围

- 完整读取七份 docs 正文、七份 case、七份 notebook 的全部四个 cells（共 28 cells，包含环境格、markdown、case 和保存输出），以及七份 `artifacts/checks/curriculum/<id>.json`。
- 七节的 case SHA256 全部等于对应 JSON 中的 `case_sha256`；notebook 实验格源码全部等于 case，保存 stdout 全部等于 JSON stdout。
- 在 `/tmp` 临时目录以 `.venv-model/bin/python`、CPU 实际复跑七个短 case；全部断言通过，stdout 与保存 JSON 逐字一致。未重训 40 步实验、运行 GPU、启动长训或修改正式教材。
- 完整读取 `scripts/run_multiscale_learning.py`、`10-multiscale-learning.json`，重新计算保存预测的 IoU 与 AP；核对学习 SVG 的全部 40 个曲线点和五个 GT／预测框路径。读取并实际渲染检查 `08-own-images.svg`、`11-csp.svg`、`11-fusion.svg`、`10-multiscale-learning.svg`。
- 同时读取实际依赖：`miniyolo/geometry.py`、`models.py`、`targets.py`、`losses.py`、`inference.py`、`metrics.py`、`custom_data.py`、`scripts/detect_image.py`，及训练 CLI／checkpoint 保存相关部分。读取持久的 own-images prediction JSON，并用其既有 checkpoint 再做推论，输出框／分数／类别均完全相同。
- 原论文直接读取 YOLO9000、YOLOv3、CSPNet、YOLOv4 相关章节。CSPNet 的文字提取漏掉公式，因此另外查看 PDF 第 4 页的公式 (1)–(4)。另外取得并读取 FPN 第 3 节、PANet 第 3.1 节，以及 YOLOv5 固定 v6.0 的 C3 和模型 YAML。下载原文和验算／渲染均只留在 `/tmp`。
- Notebook 环境格做静态核查，不把尚待发布的 `lessons-v0.2.0` 当作缺陷，也未把静态检查写成全新 Colab clone／安装已执行。

严重度：P0 为阻断运行或核心结论根本错误，P1 为会误教关键机制／结果的技术错误，P2 为局部精确性问题。合理 toy 缩简按其公开的范围评价，不要求完整官方复现。

| 节次 | 实际核查结论 | 错误严重度 |
| --- | --- | --- |
| 08-own-images | 几何、图片读取、checkpoint 格式／类数与输出路径一致 | 无 |
| 08-own-data | 自定义 JSON、来源切分、文件验证与三类 head 一致 | 无 |
| 09-anchors | sigmoid／logit、log／exp、尺寸 IoU 与 slot／class 区分正确 | 无 |
| 09-anchor-clustering | 1−IoU、mean／median 统计及原版／toy 边界正确 | 无 |
| 10-multiscale | 预测尺度、assignment、真实小框失败与缺少对照均如实说明 | P2：补充曲线横轴 |
| 11-csp | 分流、concat、梯度连通、参数与来源正确 | 无 |
| 11-fusion | FPN／PAN／v4／v5 来源、shape 与成本正确 | 无 |

## 08-own-images

**核实结果：通过，无需技术修订。** `Pillow.convert('RGB')` → CHW float32／255 → `[1,3,64,64]` 与代码一致。120×80 原图实际缩为 64×43，`sx=64/120`、`sy=43/80`，padding 左／上 `(0,10)`，下边 11；框 `[20,10,60,30]` 变成 `[10.666667,15.375,32,26.125]`，roundtrip 误差小于 1e-4。整图框变成 `[0,10,64,53]`；SVG 的有效区与框坐标也是这套取整后的几何。

`miniyolo.train` 保存包裹字典，包含 `model_state_dict`、`config`、有序 `class_names`，与正文入口一致；`scripts/detect_image.py` 使用 `weights_only=True`、CPU、`strict=True`，检查尺寸／grid／width 配置、类名与可选 `num_classes` 一致性、有限参数，并调用 `eval()`。类名顺序代表 class id，不能靠改名称获得新类别。三类名字配两类 config 的 fixture 确实被拒绝。

持久目录实际有输入 PNG、三步 checkpoint、prediction PNG／JSON 和错误类数 fixture。保存 PNG 尺寸 120×80，prediction JSON 有 16 个原图范围内的正面积框，score threshold=.05、NMS=.5。我重新载入该既有 checkpoint 推论，16 个框及分数／类别与 JSON 完全相同。正文将其限定为管线验证，人工高分 logits 的 roundtrip 另列，未把它冒充照片识别效果。

出处短引文：入口实现 `"checkpoint needs model_state_dict, config and ordered class_names"`，以及 `model.load_state_dict(..., strict=True)`。来源：[detect_image.py](https://github.com/birdhackor/learn_to_yolo/blob/main/scripts/detect_image.py)、[checkpoint.py](https://github.com/birdhackor/learn_to_yolo/blob/main/miniyolo/checkpoint.py)、[本次记录](../artifacts/checks/curriculum/08-own-images.json)。这些是本课自定义格式的主来源，本节没有伪称某篇 YOLO 论文规定该 checkpoint 格式。

## 08-own-data

**核实结果：通过，无需技术修订。** JSON `classes` 的有序索引是 class id，框为原图 pixel xyxy；空图明确保留 `[0,4]`。validator 在 tensor 转换前验证行长度、框／label 数、正整数 width／height、有限坐标、边界、非 bool 的整数 label、唯一 path、合法 split 和同 source 不跨 split。Dataset 在选择 train 前验证所有 split 的实际图片尺寸与文件，再读 RGB、同步 letterbox 图／框，保留 class 2。

路径说明准确：`annotation_path` 独立按调用目录解析；`root` 仅用于解析记录里的相对图片 path；省略 root 则用 JSON 的父目录。文末 `my-data/annotations.json`、`root='my-data'`、记录 `images/example.png` 会得到 `my-data/images/example.png`，没有重复拼接 JSON 路径。

fixture 确实写六张 PNG，三组各两张；train batch 为 `[2,3,64,64]`，其中一张黄色矩形、一张空图。head 为 `[2,4,4,8]`，class 2 进入三类 CE，一次真实 Adam 更新有限，loss 1.4407。新增第三类后的 head 权重 shape 与旧两类 checkpoint 不兼容；正文重建 model／optimizer 的要求正确。来源分组验证只能证明元数据规则，fixture 在不同 source 复制同样场景并不能证明泛化；正文已明确说明，未声称独立真实照片评估。

出处短引文：Dataset 文档字符串 `"All records are validated before selecting a split"`；validator 错误 `"class ids must be integer indices; bool is invalid"`。来源：[custom_data.py](https://github.com/birdhackor/learn_to_yolo/blob/main/miniyolo/custom_data.py)、[本次记录](../artifacts/checks/curriculum/08-own-data.json)。此 JSON 是课程契约，正文已明确不等同所有 YOLO 标签格式。

## 09-anchors

**核实结果：通过，无需技术修订。** 原论文第 2 节 Direct location prediction 的几何是 `bx=σ(tx)+cx`、`by=σ(ty)+cy`、`bw=pw*exp(tw)`、`bh=ph*exp(th)`；本课将中心换成 pixel 再乘 stride16，anchor 本来就是 pixel，因此尺寸不额外乘 stride，单位正确。

对 `[8,12,24,28]`，中心 `(16,20)`、wh `(16,16)`，cell `(1,1)`、比例 `(0,.25)`；比例不是 raw logits。`logit(clamp(0,1e-4,...))=-9.210240`，`logit(.25)=-1.098612`；解码中心的 x 误差约 .0016 pixel，正文明确有限 sigmoid 不能精确取 0。训练中心 loss 实际比较 sigmoid 值与比例 target，而非把 logit encode 当比例，符合说明。wh 的自然对数／指数反运算，以及 `[ln2,0]`、`[ln4,ln2]` 练习都正确。

尺寸 IoU 实际为 `[1,.25]`，best=0；`[B,4,4,2,7]` 的 anchor slot 与末轴两类独立，候选数32。fixture 的 same-cell 尺寸 IoU>.2 ignore 产生 1 positive、1 ignore、30 negative；ignore 的全槽梯度和非正槽框梯度为0，真实 SGD 更新通过。正文已标成简化 ignore 规则，case 本质是可学习 raw 槽位表，没有图片 CNN／detector AP，限制已清楚。

与原版的边界也合理：原论文采用每格5个框，并写 confidence 为 `Pr(object)*IoU(b,object)=σ(to)`；本 toy 的 binary objectness BCE／互斥 CE 及两槽规则不能当作完整原版损失，但本节并未作完整 YOLOv2 复现主张。

原文短引文：`"We use a logistic activation to constrain the network’s predictions to fall in this range."` 原始几何公式同页。来源：[YOLO9000，第 2 节、PDF 第 3 页](https://arxiv.org/pdf/1612.08242)、[本次记录](../artifacts/checks/curriculum/09-anchors.json)。

## 09-anchor-clustering

**核实结果：通过，无需技术修订。** 原论文 Dimension Clusters 确实用 `d(box,centroid)=1−IoU(box,centroid)`，并以最近先验的平均 IoU 比较尺寸覆盖；不是检测 AP。原文提到 VOC／COCO、k=5 的复杂度／recall 权衡，不给出 median 更新规则。故不能根据该文把本例逐维 median 说成「YOLOv2 官方更新」，当前正文没有这样归因。

本 case 用6个手选的 train 后处理 wh、k=2、固定初始中心、mean 更新、空群保留旧中心；返回 `[25/3,25/3]` 和 `[94/3,50/3]`，groups `[0,0,0,1,1,1]`。1−IoU 的目标不是欧氏 SSE；mean 不保证最小化群内 IoU 距离，median 也不是此目标的一般最优解。正文已经明确这两点，并将其称为 k-means 风格启发式，没有把简化算法升级为原论文证明。

独立验算 mean best size IoU：重复16×16弱基线 .381713；上一节16×16＋8×8 .709259；mean .911872；median `[8,8]`／`[32,16]` .934028。正文特意区分弱基线与上一节配置，数字准确。co-scale 的 IoU 不变、`[4,40]`／`[40,4]` 新来源覆盖 .1975 也通过。仅用 train 选先验、先 letterbox 再收集 wh、cluster id 不等于正／负／ignore，都与尺寸统计用途一致。

原文短引文：`"If we use standard k-means with Euclidean distance larger boxes generate more error than smaller boxes."` 距离公式为 `"d(box, centroid) = 1 − IOU(box, centroid)"`。来源：[YOLO9000，第 2 节 Dimension Clusters，PDF 第 3 页／Table 1](https://arxiv.org/pdf/1612.08242)、[本次记录](../artifacts/checks/curriculum/09-anchor-clustering.json)。

## 10-multiscale

**核实结果：核心机制与效果解释通过；一项 P2 可修。** YOLOv3 第 2.3 节确为三种预测尺度，每尺度三框，COCO 输出 `N×N×[3*(4+1+80)]`；深层2×上采样后 concat 早期 feature，重复得到下一尺度。第 2.2 节为独立 logistic class／BCE，非互斥 softmax。正文对三尺度／anchor／融合／类别定义与 toy 的两尺度、单框、softmax、无融合均有明确区分。它讲的是预测多尺度，没有混成 YOLOv2 的随机输入尺寸 multi-scale training。

实际 `TwoScale` 三次 stride2 得 `[1,16,8,8]`，再下采样得 `[1,32,4,4]`，heads `[1,8,8,7]`／`[1,4,4,7]`，80个候选、560个 logits，数字正确。同一8×8框两套中心 target／格内偏移／全图 normalized wh 均按表解回原框。小／大 GT 分别只监督 fine／coarse，另一尺度对应中心为 objectness 负例、无 box／class loss，正文没有隐藏该 toy 规则，也没有将责任按类别分配。两 head 有梯度、一次更新通过；人工同框跨尺度拼接后按类 NMS 的2→1可复现。

补充 40 步脚本使用相同 TwoScale 和一张两物件训练图，seed7、Adam .01。JSON 有40个 pre-update loss、8702参数、权重 L2差6.503563、初始loss3.659366、最后一次更新前loss .070337。没有同 backbone 的 coarse-only 品质对照，正文第76行明确指出 TwoScale 不是直接向 GridDetector 加 head，并说明正式比较应固定 backbone、初始化和预算，归因边界成立。

保存实际预测重算结果如下；这里小／大只对应该 fixture 的 class0／class1，不是 COCO area 分桶 APs／APl。

| 保存候选 | 与同类 GT 的 IoU | score | AP50 意义 |
| --- | --- | --- | --- |
| 小框预测 `[6.3247,1.3950,11.3533,14.8066]` | .441043 | .762863 | IoU<.5，FP；class0 AP50=0 |
| 大框预测 `[32.8049,32.3105,54.3092,55.4259]` | .862990 | .835802 | TP；class1 AP50=1 |
| 另一个 class1 框 `[34.8686,45.4370,56.0633,64]` | .299121 | .131394 | 背景误报 |

重算 mAP50=.5、precision=1/3、recall=.5；与 JSON 和正文一致。SVG 的绿框2个、橙框3个精确对应 GT／保存预测，曲线40个路径点全部对应保存 history。小目标仍失败，loss下降不能推出小目标AP提升；正文明确说没有 held-out 提升证据，这不是待修错误。

**P2：补充曲线把更新前 loss 画在「已完成更新次数」1–40上。** 位置：[run_multiscale_learning.py，第31–40、69–70行](https://github.com/birdhackor/learn_to_yolo/blob/main/scripts/run_multiscale_learning.py#L69)，产物 `docs/assets/diagrams/10-multiscale-learning.svg`。每轮先 forward／计算 loss，再 `optimizer.step()`，记录的是更新前的 loss；首点应对应0次已完成更新，末点对应39次，而脚本 `plot(range(1,41),history)`、xlabel `Optimizer updates` 会暗示1–40次之后测得。正文与 JSON 的 `last_pre_update_loss` 已正确，没有虚报最终 post-update loss。可把 x 改为0–39并保留更新次数标签，或保留1–40但标 `Training iteration (pre-update loss)`，再按同一 history 重绘 SVG；不用重训。这是局部标注精度问题，不阻断本节核心结论。

原文短引文：`"YOLOv3 predicts boxes at 3 different scales."`；`"We do not use a softmax ... instead we simply use independent logistic classifiers."`；融合原文为 `"merge it with our upsampled features using concatenation"`。来源：[YOLOv3，第2.2–2.3节，PDF 第2页](https://arxiv.org/pdf/1804.02767)、[单步记录](../artifacts/checks/curriculum/10-multiscale.json)、[40步实际记录](../artifacts/checks/curriculum/10-multiscale-learning.json)。

## 11-csp

**核实结果：通过，无需技术修订。** CSPNet 第3.1节的核心是把 stage base feature 沿通道分成两部分，一部分直接到 stage 尾部，另一部分进 dense block，再经 partial transition／融合。PDF公式(3)给出 `xT=wT*[x0'',x1,...,xk]`、`xU=wU*[x0',xT]`；论文还对 transition→concat→transition、fusion first／last 等顺序分别讨论。本例仅做 chunk、两次小卷积、concat、最终1×1，并没有实现原式的全部 transition hierarchy，正文公开称为简化，足以支撑分流机制教学。

YOLOv4 第3.4节确实列 Backbone=CSPDarknet53；YOLOv5 v6.0 C3 实際通过 `cv1(x)`／`cv2(x)` 形成两路，`m` 后 concat 再 `cv3`，并不是逐值相加或 literal chunk 本课输入。正文分别链接 CSPNet、v4、v5工程分支，并明确不重现完整 CSPDarknet／C3，来源没有错误归属。

toy `[B,8,8,8]` 沿 dim1 拆为两个4ch，右路3×3保持空间尺寸，concat回8，fuse8→8。数值例 concat `[1,2,10,20]`／add `[11,22]` 正确；两路梯度非零只支持连通，正文没把它当梯度重复减少的证明。Full／CSP 的带bias参数1240／368，C=10练习570均正确；SVG 的分流、concat与参数标注一致。

进一步手算仅用于核对计算方向：单图8×8下，卷积权重 MAC 为 `H*W*Cout*Cin*k²`；Full `(2*8*8*9+8*8)*64=77824`，CSP `(2*4*4*9+8*8)*64=22528`，不含 bias 加法、ReLU、split／concat流量。参数及卷积算量均较少，但不等于同比例延迟／精度改善；当前正文未作这种推论。两个随机输入的平方均值loss只验证路径与更新，没有伪称 detector 对照。

原文短引文：`"feature maps of the base layer in a stage are split into two parts through channel"`；CSPNet figure3 将完整配置写为 `"transition → concatenation → transition"`。v4 原文 `"Backbone: CSPDarknet53"`。来源：[CSPNet，第3.1节、公式(3)–(4)、Figure3](https://arxiv.org/pdf/1911.11929)、[YOLOv4，第3.4节](https://arxiv.org/pdf/2004.10934)、[YOLOv5 v6.0 C3](https://github.com/ultralytics/yolov5/blob/v6.0/models/common.py#L125)、[本次记录](../artifacts/checks/curriculum/11-csp.json)。

## 11-fusion

**核实结果：通过，无需技术修订。** FPN 第3节确是 backbone bottom-up hierarchy 上的 top-down／lateral 融合，2× nearest upsample、lateral 1×1 和 element-wise add，再用3×3形成最终 map；PANet 第3.1节额外从 P2 向 P5 添加下采样／融合路径，其3×3 stride2后与上一层同尺度 map 相加，再3×3。YOLOv4 第3.1／3.3–3.4节选 CSPDarknet53＋SPP＋PAN neck，并明说把 PAN shortcut 改成 concat；v5 v6.0 YAML 则可直接看见两次 nearest／Concat 和两次 stride2／Concat。正文没有把 FPN、PANet、v4、v5当作同一篇论文的一套新模块，也准确区分 FPN add 与本 toy concat。

toy 输入是独立随机 shallow `[1,8,8,8]` 与 deep `[1,16,4,4]`，并不是已接上第10节 detector 的 feature。正文已标出与 TwoScale 16／32ch 的差异；SVG 中输入／backbone是空间概念示意，fine head 也写明仅接点示意、case停在融合feature，没有把它冒充实际完整 detector。

reduce16→8 的1×1得到 `[1,8,4,4]`，nearest到8×8，concat为 `[1,16,8,8]`，mix16→8输出 `[1,8,8,8]`，与正文／SVG一致。nearest2×2→4×4的逐值复制正确，sum loss 对每个来源梯度为4，短case可复现。通道不同仍可 concat 的24ch对照及整shape相同时 add 的8ch对照均真实执行。

成本公式正确：reduce `8*16+8=136`，mix `8*16*9+8=1160`，共1296。按第10节通道衔接则reduce32→16、concat32、mix32→16，共 `(32*16+16)+(32*16*9+16)=5152`；float32、B1的toy concat `16*8*8*4=4096bytes`。单图卷积权重 MAC 另外可核为toy `16*8*16+16*8*9*64=75776`、实际衔接版 `32*16*16+32*16*9*64=303104`，所以1296参数不能当作加在TwoScale上的成本。正文已做这一区分，并不报AP／速度。shape相同不能保证原图格点相位对齐、upsample不能创造已丢失细节的解释也正确。

原文短引文：FPN `"by element-wise addition"`，并写 `"nearest neighbor upsampling for simplicity"`；PANet `"Each feature map Ni first goes through a 3 × 3 convolutional layer with stride 2"`；YOLOv4 `"replace shortcut connection of PAN to concatenation"`。来源：[FPN，第3节／Figure3](https://arxiv.org/pdf/1612.03144)、[PANet，第3.1节／Figure2](https://arxiv.org/pdf/1803.01534)、[YOLOv4，第3.3节／Figure6](https://arxiv.org/pdf/2004.10934)、[YOLOv5 v6.0模型配置](https://github.com/ultralytics/yolov5/blob/v6.0/models/yolov5s.yaml)、[本次记录](../artifacts/checks/curriculum/11-fusion.json)。

## 复查入口

若只修10-multiscale横轴，可检查生成脚本的history记录时点和新SVG标签／x值，不需再次训练。其余七节的当前 case／notebook／保存 JSON 一致，AP、参数和几何数字不需要更改。后续正式文件若有修改，可据以上逐节原文依据和实际数字独立复查；本轮唯一仓库写入为本 review 文件。

## 修订后独立定点复查：10-multiscale P2 已关闭

日期：2026-10-02。以上初审保留为历史记录；本段记录修订后的实际状态。

**复查结论：初审唯一 P2 已关闭；本 review 覆盖的七节当前没有未关闭技术问题。** 重新读取当前 `scripts/run_multiscale_learning.py`、`docs/lessons/10-multiscale.md`、`10-multiscale-learning.json` 与新 SVG 源码，并将当前 SVG 在 `/tmp` 重新渲染后实际查看。

- 生成脚本第69–70行仍以1–40表示训练步序，但 xlabel 已为 `Training step (pre-update loss)`。它与实际先计算 loss 再更新参数的顺序一致：step1是第一次更新前的量测，step40是第四十次更新前的量测，已不表示完成了该次数更新后的loss。当前 SVG 源码与实际可见标签均为该文字，旧 `Optimizer updates` 标签已消失；完整标签可读，没有被裁切。
- 正文第78行明确说明「曲線橫軸是訓練步序，每點loss在該次更新前量測；框與AP則是在40次更新全部完成後評估」，将曲线时点与最终预测时点分清。第68行继续正确称末次loss为「最後一次更新前」；不需要将x强制改成0–39。
- 当前 JSON 的全部40个 history 数值逐项等于独立初审时读取的数值，首值3.6593663692474365、末值.07033684849739075，与 `initial_loss`／`last_pre_update_loss` 一致。我用当前SVG坐标系重新核对全部40个路径点，确实对应JSON history在训练步序1–40上的坐标，非仅替换文字而忽略数据。
- 新SVG的两个绿GT框、三个橙预测框逐一与当前JSON坐标相符。重算实际预测IoU仍为小框.441043、大框.862990、另一个class1候选.299121；AP50仍为class0=0、class1=1，mAP50=.5、precision=1/3、recall=.5。正文小目标未达IoU .5、没有held-out小目标提升证据、没有同backbone单尺度质量对照的限制均继续准确。

本次未再训练模型、未改正式文件；仅对指定10-multiscale修订重新验算／渲染并追加本段关闭记录。其他补充曲线和grid160不在本次定点复查范围内，未据此声称已验证它们。

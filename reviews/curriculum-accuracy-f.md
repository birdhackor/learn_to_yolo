# 独立技术审查 F：17–20 与共同页

审查日期：2026-10-02。本人从当前工作区原文、源码及实际产物独立检查，未使用旧 reader reports 代替审查。只修改本报告；没有启动 GPU、长训练、发布、push、访问私有 HF 或读取凭证。

本轮结论：17–20 未发现 P0／P1／P2 技术错误。发现两项 P3 来源／证据文字同步问题，见下表。合成任务、手工输入、单次小样本、未测相机和未审计逐层 FP16 等边界写得明确，不以它们为理由要求完整训练或真实相机实验。待发布 `lessons-v0.2.0` tag 不记为错误。

## 确实发现的问题

| 编号 | 严重度 | 首次发现与影响 | 建议及本轮状态 |
| --- | --- | --- | --- |
| F-01 | P3 | `docs/lessons/20-deployment.md` 参考中的 `https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/command-line-programs.html` 本轮实际请求返回 HTTP 404，读者不能沿该链接核实 CLI。命令本身经官方固定源码核对正确。 | 改为下文实际 HTTP 200 读取的 NVIDIA/TensorRT 固定 10.13 commit 的 `samples/trtexec/README.md`，另可附 `sampleOptions.cpp` 的 flags 定义。首次发现保留，修订后须复查链接。 |
| F-02 | P3 | `artifacts/checks/curriculum/fashion-mnist-learning.json` 的 `steps=40` 与完整 40 条 history、共同页一致，但 `limitations` 仍称 “A two-step smoke”。同一固定文本来自 `miniyolo/classification.py`，因此加长训练仍会输出“两步”。 | 限制文字按实际 `steps` 表达，或使用不固定步数的描述。该问题不改变 40 步真实数据管线及低 accuracy 的结论。首次发现保留，修订后须复查源码与结果。 |

## 完整阅读与复现方式

完整读过四份 `docs/lessons/{17-capstone,18-video,19-tracking,20-deployment}.md`、四份同名 `lesson_cases/*.py`。四份 notebook 各有四个 cells，逐格读过标题说明、环境初始化、实验说明及完整实验／保存输出；实验 cell 与对应 case 逐字一致。

逐项核对四份 `artifacts/checks/curriculum/<lesson>.json` 的 `case_sha256` 与当前源码、完整 stdout 与 notebook 输出、页面展开输出；17／18／20 的 stdout 中 JSON 与各自 `17-metrics.json`／`18-metrics.json`／`20-metrics.json` 完全一致。没有只用 exit code 判断正确。

使用既有 `.venv-model/bin/python`，从 `/tmp/accuracy-f-runtime` 通过 `runpy.run_path` 载入原 case，再调用原 `main`，产物只写临时目录。四节有限 CPU 实验的全部非计时指标复现；历史正式计时没有被重写。实际 SVG 均完整解析并在 Chromium 渲染检查：17、19 与临时重跑 SVG 逐字节一致；18 三张嵌入预测 PNG 逐字节一致，临时计时文字按当次测量变化。

相关完整源码包括 `miniyolo/{data,models,losses,inference,metrics,deployment_gpu,gpu_smoke}.py`、`scripts/{modal_deployment,modal_gpu_smoke,verify_curriculum,run_fashion_cnn}.py`，并核对几何及 checkpoint 的相关契约。实际 GPU 结果按保存证据核验，未重新启动 GPU 或重新下载私有权重。

## 17-capstone

核对：完整页面、case、notebook 全 cells、`17-capstone.json`、`17-metrics.json` 和实际 `17-capstone.svg`。

- `ShapeDataset` 每个索引使用独立 generator（seed 加 index×1009），红=0、蓝=1，框在不同 cell 内。train／validation／test 为 32／16／16 张，seed 1100／2200／3300；没有将训练图冒充 held-out。
- 两次 `fit` 均重置模型 seed7、width8、Adam lr .01、batch8、160 次更新，只改变 box weight5→10。模型 15,511 参数，输入 `[8,3,64,64]`、raw `[8,4,4,7]`，损失各项及平均轴与页面一致。
- score .05、class-wise NMS .5、matching IoU .5、all-points AP。页面明确 AP50 与 display .25 分开、不同总 loss 权重不能直接比高低。
- 复现 baseline class AP .333333/.555556、mAP .444444、TP9、FP5、17GT，precision9/14、recall9/17；weight10 class AP .625/.777778、mAP .701389、TP12、FP3，precision12/15、recall12/17。coverage 分别 9/4/4、12/2/3；逐 GT coverage 不被当成一对一 recall。
- baseline FP 为 background1、localization4；changed 为 background1、localization2。wrong-class 计数0并不被解释成所有未检 GT 的分类正确。SVG 的 validation #10 背景 FP score .068008、best IoU0，与真实 report 相符。
- `keep` 使用 validation 至少提升 .01；只在选择后评 chosen test 一次，复现 mAP .445238、precision7/11、recall7/13。小 split、单次初始化、validation 不等于稳定泛化均已说明。
- 正式 0.506／0.453 秒来自 train loop（不含模型／target 初始化），CPU 同步；1.052933ms 包含 uint8→tensor→forward→decode/NMS→PIL 绘制，12次、3次 warmup。绘制还包含缩放与 GT 注记，页面已补明不是纯产品推论。没有从一次时间差得出 weight10 更快。
- SVG 同样四张 validation #14/#5/#4/#7，来自按 baseline 未覆盖 GT 数选难例；不是精选最好结果。整个图逐字节复现。

结论：通过，未发现技术错误。没有要求多 seed、真实照片或更大训练才算本 toy 协议正确。

## 18-video

核对：完整页面、case、notebook 全 cells、`18-video.json`、`18-metrics.json`、实际 `18-video.svg` 与临时重跑 GIF 管线。

- 模型独立从零训练32张、seed4100、160步；`run_stream` 中只做真实 forward，不以 GT 替代预测。`Frame` 是 uint8 HWC RGB，64×96→float CHW→64×64 letterbox→raw→decode .1→undo_letterbox→原图绘制，坐标契约一致。
- 生成器确实按需求逐幀产出；`run_stream` 使用 no_grad，model 在训练函数返回前 eval。只有短 GIF main 收集12幀，页面明确长影片应逐幀写出。
- 12幀20FPS：index0–11、最后 timestamp .55s、总播放时长 .6s，每幀50ms。24幀练习最后 timestamp1.15、总1.2，x=4+4i，20–22裁切、23离场；代码与练习说明一致。
- 独立复现框数 `[1,1,0,0,0,0,1,1,0,0,0,0]`。12幀都被处理，8幀没有超过阈值的框；正文没有把这种漏检说成 stream 掉幀。SVG 展示真实幀0/5/11，0/.25/.55s及1/0/0框，三张嵌入图复现。
- 正式各项 median .186751/.198889/.367443/.048543ms、total .840675ms 与报告同步，排除第一幀。总 median 不被说成分项 median 之和。`started` 在来源取到幀之后，故不含生成、codec、capture、queue、display 等；正文准确排除这些成本，也没有由模型时间宣称相机 FPS。
- `opencv_frames` 为可选 adapter，使用 BGR→RGB、`finally` release；文件时间 index/fps、相机时间 perf_counter，VFR／PTS 限制及需要消费 generator 均已说明。默认 main 不调用它，文本和 JSON 明确“provided, not executed”；不要求本轮真实相机验证。

结论：通过，未发现技术错误。

## 19-tracking

核对：完整页面、case、notebook 全 cells、`19-tracking.json` 和实际 `19-tracking.svg`。

- 输入是手工 detections；当前节没有 detector／backward。A/B 仅用于最后评 ID switches，`Tracker.update` 只收到框和 frame，不读取真值身份。
- 当前代码的 `exact_gated_matching` 枚举每个 track 的 unmatched 与合格未使用 detection，以 `(配对数, IoU总和)` 为比较键；它确实对此有限问题求全局最优，不是 greedy 或 Hungarian。没有按旧实现印象判为“非全局最优”。组合爆炸／正式 solver 边界已说明。
- 官方算法名不是本 toy 宣称对象，因此不用完整 SORT／ByteTrack 指标要求本节。页面自定义 ID switch 为物理身份相邻观测的 assigned ID 变化，跨漏幀保留 last_id。
- 实际反贪心框 IoU 为约 `[[.818182,.666667],[.538462,.25]]`；交叉和1.205128胜过对角1.068182，代码 assert 返回交叉。空 detections 的 `[2,0]` 矩阵返回空匹配。
- x位置 A16+8f、B40−8f，宽高12；第2幀 last-box 反身份 IoU1、正确身份 .2；速度正确预测为IoU1。漏幀采用 `velocity×(frame−last_frame)`，第5幀 B 能按间隔2重接。
- 复现 last-box IDs `[[1,2],[1,2],[2,1],[2,1],[2],[2,3]]`、3 switches；motion `[[1,2],[1,2],[1,2],[1,2],[1],[1,2]]`、0。`max_age=1` 也独立调用 `run` 验证 motion末幀 `[1,3]`、1 switch，last-box仍3。
- 两路 detections 相同，11/12 recall、FP0是人工序列的性质，没有把 association 改善当成 detector 改善。旧ID1在 max_age2时未过期，而是 IoU0未通过门槛，因此B创建ID3，说明正确。
- SVG ID颜色、每幀分配、漏幀和真实水平位置相符；A/B展示分行没有改变 matching 的 y 坐标。逐字节复现。

结论：通过，未发现技术错误。

## 20-deployment：CPU ONNX／ORT

核对：完整页面、case、notebook 全 cells、`20-deployment.json`、`20-metrics.json`、真实临时 ONNX 文件和真实 ORT 执行。

- GridDetector 只做一次 SGD 更新，页面准确限制为 export parity，不是有效 detector 品质。ONNX 导出的是 model raw，前后处理共享 Python；不是整个 RGB/NMS pipeline 已被封装入 ONNX。
- PyTorch2.9.1 明确 `dynamo=False`、opset17、`dynamic_axes` 只命名 B。官方 v2.9.1 docstring证实2.9起 dynamo默认True及 legacy的 dynamic_axes 路径；弃用 warning 保留在正式 stderr，正文已说明。网页本轮 HTTP403，不绕过；固定官方源码作为依据。
- checker结构／类型验证与实际 numerical parity 分开。三张非方形 uint8 RGB 源图提供真实 B1/B2/B3，代码同时 assert输入与raw shape，防止把两张切片称成B3。
- 独立复现三 batch 最大 raw 差均 `4.76837158203125e-7`，实际调用 ORT CPU，raw与还原后 boxes/scores/labels均通过。80×80实际被 ORT `InvalidArgument` 拒绝；动态B不被混称动态H/W。
- score仍为 `sigmoid(obj)×softmax(class)`，class-wise NMS .5、score .05、metadata逐图还原。页面对临界阈值引起候选变化的提醒与第三层比较相符。
- 正式 PyTorch raw .1525845ms、ORT raw .0364245ms，比值约4.19；pipeline2.3286365→2.1442895ms，降低约7.92%。B2 raw .042584ms、`2000/ms=46965.997 images/s`正确。20次median、3次warmup、CPU2threads；不含磁盘／相机队列、湊batch等待。新计时变化，没有被用于替换正式证据。

结论：技术通过；F-01为失效参考链接。

## 20-deployment：保存的 L4／TensorRT 实证

完整检查 `artifacts/checks/curriculum/deployment-gpu.json`、`miniyolo/deployment_gpu.py`、`scripts/modal_deployment.py`、`.github/workflows/deployment-gpu.yml`。这三份当前源码与实际 run commit `e64ea79207ed3485208a91c824da3840962aff56` 的 Git 内容完全相同。

- 不是另写 probe 模型：同 GridDetector、15,511参数、8张固定合成图、40次Adam；JSON的初末 loss .984767→.071048，与正文一致。不能与CPU一步模型跨设备比较品质或速度，正文已分开。
- 真实 parse、build_serialized_network、deserialize、execution context；输入地址及输出 buffer实际送给 `execute_async_v3`。profile min/opt/max分别B1/B2/B4，真实执行并assertB1、B2、B3、B4，不只是profile包含这些shape。
- 两设置均 clear TF32；FP16分支仅设置 `BuilderFlag.FP16`，输入float32，没有 layer precision audit。官方10.13 docstring是“Enable FP16 layer selection”，官方trtexec help是“in addition to fp32”；文案“允许FP16”及不保证全层FP16正确。FP16 flag在10.12起弃用但10.13仍有该接口，固定实跑未因此被误判失效。
- JSON所有batch均有decoded candidates，数量2/4/6/7；raw、框、分数、labels/order实际分别assert。FP32 raw最大 `4.768372e-6`、框 `1.144409e-5` pixel；允许FP16 raw最大 `4.768372e-6`、框 `7.629395e-6` pixel。表给B1–4最大误差，时间仅B1，页面已明确。
- B1 median .1602175/.1616085ms，3warmup、20次。`infer` 先shape/address设置及分配输出，再enqueue、stream synchronize；计时头尾另有 device synchronize。包括Python/配置/分配/同步开销，不含CPU copies、preprocess/decode、engine build、container startup。client223.405937秒包含build/startup/transfer，两者没有混用。
- 明确Volume commit、另一CPU container reload并重新读文件。producer任务 `ta-01M3YR5QQKJ41W0B9RS5ESBHDR` 与 verifier `ta-01M3YR6Y0XDS0P1YC2ASDV81WR` 不同；ONNX及两engine的三个SHA与storage字典逐项一致，`sha256_verified=true`。
- 保存的停止证据是该app `ap-cmJqI2YJQU65SWaiCaPAx7` 的 `state=stopped`、`running_tasks=0`。`confirm_stopped` 检查特定app ID和tasks，必要时只stop本app；不把 `gpu_calls_finished` 或上下文关闭当成停止证据。
- workflow只有workflow_dispatch，与旧smoke共享concurrency group、单GPU、max1、retries0、GPU timeout600，always停止确认与保存JSON；一般push/PR没有GPU触发。

结论：保存实证与源码支撑页面所述有限部署结果，未发现技术错误。没有把精度flag、一次小模型计时或相同训练图parity扩写成真实AP／完整YOLO效能。

## 共同页同步核对

| 完整读过的页面 | 实际核对及结论 |
| --- | --- |
| `docs/index.md` | 42个独立小节、教学习惯、人工fixture／CPU训练／合成任务／真实场景边界，与42项registry及运行index一致。L4 checkpoint及TensorRT已有实证，没有仍写成全部未跑GPU。 |
| `docs/learning-path.md` | 42项顺序及17整合、18–20应用链接对应；07training当前更名为短步更新与160步学习后应同步导航。已通知root的该编辑不作为缺陷。 |
| `README.md` | CPU固定依赖覆盖ONNX/ORT；42notebooks配对、grid CLI数据与held-out .803571/.774892证据一致。Fashion短步与可选长步被区分，`--train-steps` 不设subset时确用完整54k训练split，`--eval-samples 0`评完整split。手动GPU／不因push启动、80次checkpoint更新与证据一致。 |
| `docs/status.md` | 明确人工输入、少步、合成任务、实际L4、真实数据完整对照的差别。42/42运行index及旧检查索引22core+2checkpoint支撑当前24项的历史检查。没有把保存notebook输出称成已登入Google或已验证Colab GPU。 |
| `docs/preparation/data.md` | 四Fashion原始gzip的全部bytes/SHA与manifest及实际文件一致，IDX确为60000/10000、28×28；总30,878,645 bytes及tar30,894,080 bytes、SHA及全部成员核对。Penn-Fudan53,723,336 bytes、SHA、ZIP CRC及图片/mask/TXT各170核对，包内版权限制与正文一致。Pet现有annotation包SHA、trainval3680/test3669、XML3686、trainval缺9、test0XML实际核对。未下载VOC/COCO全包的界线明确。Fashion40步、64/128/128、accuracy .140625/.09375与保存结果一致；仅F-02的结果limits文本漂移。 |
| `docs/validation/curriculum.md` | 逐项清单对应42结果，17/18/20训练／管线与19机制范围正确。L4补充实验、非空decode、FP16允许但未逐层audit、INT8/相机未测等与实证一致。 |
| `docs/validation/gpu-smoke.md` | 完整旧GPU JSON、现有实现与run commit eaa0cdf逐项核对，见下一节。没有把该次checkpoint smoke说成TensorRT或正式throughput，后续部署单列。 |
| `docs/research/version-sources.md` | 固定官方commits、toy界線、CPU ONNX与另一次L4部署范围一致。root通知正在补16章的官方API默认NMS／显式nms=False、论文defaultone、STAL论文<8→16与固定源码<16→16、progressive针对NMS-free模式；该部分最终同步应在修订后重新读取，当前不借旧报告判通过。 |

## 旧 L4／HF checkpoint 证据

完整核对 `artifacts/checks/gpu-smoke-37032967155.json`、`gpu-smoke-evidence.json`、对应源码和工作流程，不访问私有 HF 仓库。当前 `miniyolo/gpu_smoke.py`、`scripts/modal_gpu_smoke.py`、`gpu-smoke.yml` 均与保存 run commit `eaa0cdf073bec49b09402bfbed40db9b0648b425` 完全相同。

- 40不中断、20中断前、20恢复后，合计80真实 optimizer 更新；所有保存history分项loss公式及有限值核对，interrupted20与baseline前20逐字段完全相等、resume20与baseline后20逐字段完全相等，包括loss、gradient、lr、四类RNG probe。
- 固定数据loss1.040038→.0617025，前5均值 .92157216、后5 .06433246，gradient范围 .152425855–1.594808221，均与正文一致。model/optimizer/history最大差0、scheduler/RNG一致被限定为该次实测；AdaptiveAvgPool2d nondeterministic warning没有掩盖。
- producer、Volume CPU verifier、GPU resume的三个task ID不同，六文件SHA字典相同；中途SHA `15550f...80824` 在producer／Volume／固定HF上传commit下载／resume各处相同，resume入口明确 `hf-downloaded.pt`，不是只声称HF可用。
- 私有repo、上传commit `24f89355835dd12fe77451911c813e98d310efdc`、校验及恢复路径与文档一致。源码token只注入CPU transfer/probe，报告只有存在状态；本审查不打印秘密，也不将私有访问限制判为教材错误。
- 训练计时头尾CUDA同步，包含更新、梯度、RNG、标量和console；setup、存档、Volume commit、启动/传输独立。正文1.556944/.113483/1.958308及相关持久化/网络client时间与JSON一致，没有拿warmup前后或新container的数字排速度名次。
- 保存停止证据明确 `ap-IvP1WCRTduPs9ZTJKY8aG1` stopped、tasks0；检查函数确实查询该app状态与tasks，不只看流程退出。

## 本轮实际读到的官方出处

1. [PyTorch v2.9.1 ONNX API文档源码](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/onnx/__init__.py)：HTTP200、15,688 bytes、SHA `22128fd9193ccca5e6146280304a5f236349d52c0881dbf7dc7e1a9cc8a0f673`。L137–140明确dynamic_shapes／dynamic_axes适用路径，L278–279说明2.9默认dynamoTrue。官网 `/docs/2.9/onnx.html`、stable与pytorch.org跳转入口本轮均403，未绕过。
2. [ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary.html)：HTTP200、299,591 bytes，实际读InferenceSession/providers/run/numpy输入契约；[官方Python入门](https://onnxruntime.ai/docs/get-started/with-python.html) HTTP200、71,598 bytes，checker和真实session执行示例与本章层次一致。
3. [NVIDIA TensorRT官方Python说明](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/python-api-docs.html)：实际阅读已缓存 `/tmp/tensorrt-official.html`，说明set_tensor_address、execute_async_v3、GPU buffer及同步。最新API已是11.3，不据此替代固定10.13的精度语义。
4. [NVIDIA/TensorRT 10.13固定官方API docstrings](https://github.com/NVIDIA/TensorRT/blob/b8db91e15be2cae4465ac17fab19e0f969e45407/python/docstrings/infer/pyCoreDoc.h)：HTTP200、123,651 bytes、SHA `b752417c486304b755edda05b5b4e125c6d73ce240a4e1e118543084cce95105`；官方release/10.13由git ls-remote取得该commit，固定文件与分支文件逐字节相同。L175–193定义profile min/opt/max，L476–494定义shape/address，L553–560定义异步执行与buffer生命周期/stream同步，L1109–1126定义FP16 layer selection与TF32默认允许。
5. [NVIDIA/TensorRT 10.13 trtexec README](https://github.com/NVIDIA/TensorRT/blob/b8db91e15be2cae4465ac17fab19e0f969e45407/samples/trtexec/README.md#example-3-running-an-onnx-model-with-full-dimensions-and-dynamic-shapes)：HTTP200、11,495 bytes、SHA `a84aa8467e6f649cd811c9c86bca991dbf5a940a24f6432fd990c58d676f17ec`，完整读取，动态shape命令在Example3。其[官方flags源码](https://github.com/NVIDIA/TensorRT/blob/b8db91e15be2cae4465ac17fab19e0f969e45407/samples/common/sampleOptions.cpp) HTTP200、143,550 bytes、SHA `5e74b2376628c84c803804e0b0647903eae512156bf88d1f9486640348f7353b`；L2526–2528是min/opt/maxShapes，L2604–2605是noTF32及fp16（明确in addition to fp32），L2651–2652是save/loadEngine，L2750是shapes。它们支撑本章未实跑CLI的语法与精度说明，不能被说成CLI运行证据。

官方原文与独立临时CPU产物分别留在 `/tmp/accuracy-f-official/`、`/tmp/accuracy-f-runtime/`、`/tmp/accuracy-f-visual/`，没有替换正式教学图或计时证据。

## 修订后独立复查

以下为 root 修订后再次读取当前文件、检查原文及实际运行所做的独立复查；上面的初审发现保留。

| 项目 | 实际复查 | 状态 |
| --- | --- | --- |
| F-01 | 重读20章参考段：旧404链接已移除，替换为 `b8db91e15be2cae4465ac17fab19e0f969e45407` 的 `samples/trtexec/README.md` 与 `samples/common/sampleOptions.cpp`，正是本reviewer实际HTTP200取得并阅读的两份官方文件。已核对README的动态shape例子及flags定义；文案仍明确CLI未实测、Python API GPU结果另列。 | 已关闭 |
| F-02 | 重读 `miniyolo/classification.py` 并以AST核对 `limitations` 为实际 `steps` 插值；新40步JSON包含40条history和正确限制文字，与作者新保存的 `artifacts/runs/fashion-curriculum/result.json` 每字段相同。本人随后在 `/tmp/accuracy-f-runtime/fashion-corrected` 使用原函数、40步／subset64／eval128独立重跑，整份结果JSON（包括每一步history、accuracy、weights_changed及新limitations）完全复现。未覆写正式产物。 | 已关闭 |
| 共同页版本来源 | 完整重读新版 `docs/research/version-sources.md`。再直接读固定官方recipe L26、predictor L469 `end2end=self.args.nms is False`，确认API默认many加NMS、显式nms=False取one；读YOLO26 v1 §3.2.1的“One-to-One Head (default)”确认论文与API预设已分开。读论文式(5)及§4.1：smin8、sref16；再读固定TAL初始化及候选函数：stride_val取第二项16，`wh<stride_val`扩到16。新版共同页准确标明两种门槛及2×2例子同时成立；progressive及移除辅助head明确用于本节选择的NMS-free模式。 | 已通过 |
| 导航名称 | 重新核对07training页当前标题“Grid MiniYOLO：三步訓練與診斷”，learning-path、curriculum清单、section-map与zensical导航均使用“三步訓練與診斷”；160步学习作为另一路实验证据，没有混称三步已收敛。 | 已通过 |

最终结论：本审查负责的17–20及共同页相关事实通过；初审两项P3均已独立复查关闭，未关闭问题0项。该结论保留本报告所列CPU、合成任务与保存GPU实证的实际范围。

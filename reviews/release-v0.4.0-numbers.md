# lessons-v0.4.0 數值獨立稽核

基準：main HEAD `3f72bb5eeb4ee417898bcee24c93d52c92085e2c`。完成時間：2026-10-05T07:28:28.059194+00:00。
範圍：Appendix A `/tmp/record-changes.txt` 的 42 份逐節 CPU 紀錄與 9 份 CPU 補充紀錄；逐份 NEW JSON 對照 `git show HEAD:<record>`；各教學頁只稽核 `<!-- curriculum-evidence:start -->` 之前的正文，另搜尋整個 docs/ 與 README.md 的舊數字及紀錄檔名。未修改 tracked files、notebook、程式、圖、evidence；未執行 GPU 或外部 workflow。

## 稽核結果

原定範圍之外的必要修正在 `17-capstone.md` 與 `20-deployment.md`：實測時間、B=1 raw parity error，以及依時間衍生的倍數、占比與吞吐量。另補足原定第 1、3 章裡短版實驗數字與跨段引用；第 4、10 章現有 rounded values 正確，無須為最後幾位漂移改字。
本次稽核期間主代理正在改正文。下表與 JSON 保留原始必修清單，status 明確區分已经同步修复與仍待處理；它不是要求重做已完成的修改。
報告寫出時：16 個修正組已見到在並行修改後的正文，0 個仍待處理。

## 必要修正清單

| ID | 正文位置 | 原 claim → 新 claim | record／source key | 狀態 |
|---|---|---|---|---|
| cnn_initial | docs/lessons/01-small-cnn.md:36,162,166 | step=0 0.6941; 40-step initial 0.694144 → step=0 0.6942; 40-step initial 0.694159 | 01-small-cnn-learning.json／models.cnn.initial_loss; short record 01-small-cnn.json stdout step=0 | resolved_in_concurrent_root_edits |
| cnn_history_and_gradient | docs/lessons/01-small-cnn.md:172,174 | first exactly-zero point 30; minimum gradient about 1e-17, maximum .76 → first exactly-zero point 31; minimum gradient about 1.4e-18, maximum .80 | 01-small-cnn-learning.json／models.cnn.loss_history[29:31], models.cnn.min_gradient_l2, models.cnn.max_gradient_l2 | resolved_in_concurrent_root_edits |
| comparison_table | docs/lessons/03-comparison.md:144,145 | plain .693791→.692943; residual .692160→.030739 → plain .693793→.692948; residual .692138→.030958 | 03-comparison-learning.json／models.plain.initial_loss, models.plain.last_pre_update_loss, models.residual.initial_loss, models.residual.last_pre_update_loss | resolved_in_concurrent_root_edits |
| comparison_short_and_crossquote | docs/lessons/03-comparison.md:177 | short residual .6922→.6855; quoted plain 40-step .693791→.692943 → short residual .6921→.6855; quoted plain 40-step .693793→.692948 | 03-comparison.json／stdout: residual step=0 loss; long record models.plain.initial_loss/last_pre_update_loss | resolved_in_concurrent_root_edits |
| grid_machine_timing | docs/lessons/07-training.md:200,234 | AMD EPYC 9V74; .74 sec → Intel Xeon Platinum 8573C; 1.18 sec | grid-learning.json／hardware.cpu, training_seconds | resolved_in_concurrent_root_edits |
| custom_short_loss | docs/lessons/08-own-data.md:191,195 | .16921 final full-train loss; step158 peak1.86 → .16977 final full-train loss; step158 peak1.87 | custom-data-160-step.json／final_train_loss.total, loss_history[157].total | resolved_in_concurrent_root_edits |
| custom_long_table | docs/lessons/08-own-data.md:222,224,225,226,227 | loss .000354692; validation .388889; test .666667; gradient .020489–32.557129; parameter delta16.519738 → loss .000117048; validation .296296; test .777778; gradient .024197–32.544746; parameter delta17.216838 | custom-data-learning.json／final_train_loss.total, final_validation.map, final_test.map, gradient_l2_range.min, gradient_l2_range.max, weight_delta_l2 | resolved_in_concurrent_root_edits |
| custom_example_conclusion | docs/lessons/08-own-data.md:242,250 | one correct-class box in each object image; blue TP atIoU.51; yellow FP atIoU.30 → red: class0 TP(score≈1.00,IoU≈.58) plus class2 FP(score≈.16,IoU≈.24); blue: score≈.46,IoU≈.41, FP+FN; yellow: score≈.99,IoU≈.42, FP+FN | custom-data-learning.json／validation_examples[0..2].prediction, validation_examples[0..2].matches | resolved_in_concurrent_root_edits |
| custom_timing | docs/lessons/08-own-data.md:256,260 | training≈7.5sec/7.517sec; end-to-end8.273sec → training≈10.9sec/10.877sec; end-to-end14.002sec | custom-data-learning.json／runtime.training_seconds, runtime.end_to_end_seconds | resolved_in_concurrent_root_edits |
| status_machine_and_custom | docs/status.md:42,78 | AMD EPYC; shortloss.16921; long validation.388889=7/18, test.666667=2/3 → Intel Xeon Platinum8573C; shortloss.16977; long validation.296296=8/27, test.777778=7/9 | custom-data-learning.json／runtime.machine.cpu, final_validation.map, final_test.map; short record final_train_loss.total | resolved_in_concurrent_root_edits |
| video_timing_and_rate | docs/lessons/18-video.md:106,107,108,109,110,112,125 | medians .199,.215,.337,.045,total.887ms; sum .796ms;1127frames/sec → medians .276,.310,.439,.070,total1.100ms; sum1.095ms;909frames/sec | 18-video.json／stdout JSON median_ms.preprocess/model/postprocess/drawing/total | resolved_in_concurrent_root_edits |
| capstone_training_timing | docs/lessons/17-capstone.md:169 | baseline.51sec; changed.45sec → baseline.78sec; changed.87sec | 17-capstone.json／stdout JSON train_seconds[0], stdout JSON train_seconds[1] | resolved_in_concurrent_root_edits |
| capstone_pipeline_timing | docs/lessons/17-capstone.md:171 | chosen pipeline median1.05ms → chosen pipeline median2.90ms | 17-capstone.json／stdout JSON chosen_end_to_end_median_ms | resolved_in_concurrent_root_edits |
| deployment_raw_error | docs/lessons/20-deployment.md:95 | all B=1,2,3 max errors≈4.77e-7 → B=1≈1.19e-7; B=2/3≈4.77e-7 | 20-deployment.json／stdout JSON max_abs_raw_errors[0..2] | resolved_in_concurrent_root_edits |
| deployment_timing_and_ratios | docs/lessons/20-deployment.md:143,144,145,163,164,165,166 | raw .153/.036; E2E2.329/2.144; B2.043ms;4.2x;6.6%;rest2.18ms;rawdelta.116ms → raw .255/.112; E2E3.279/3.043; B2.099ms;2.3x;7.8%;rest3.02ms;rawdelta.142ms | 20-deployment.json／stdout JSON median_ms.torch_raw_batch1, stdout JSON median_ms.ort_raw_batch1, stdout JSON median_ms.torch_preprocess_to_restored_boxes, stdout JSON median_ms.ort_preprocess_to_restored_boxes, stdout JSON median_ms.ort_raw_batch2 | resolved_in_concurrent_root_edits |
| deployment_throughput | docs/lessons/20-deployment.md:172,173,175 | 27,454/46,966images/sec; batch throughput1.7x; saving≈.03ms → 8,901/20,218images/sec; batch throughput2.3x; saving≈.13ms | 20-deployment.json／stdout JSON median_ms.ort_raw_batch1, stdout JSON median_ms.ort_raw_batch2, stdout JSON ort_raw_batch2_images_per_second | resolved_in_concurrent_root_edits |

## 特別核算與保留結論

- **第 20 章新 timing 及推導**：PyTorch raw B1 `.2546969917602837 ms`；ORT raw B1 `.11234500561840832 ms`；PyTorch end-to-end `3.278577991295606 ms`；ORT end-to-end `3.0430780025199056 ms`；paired差中位數 `.24904299061745405 ms`；ORT B2 `.09892400703392923 ms`。raw比 `2.2670967`（約2.3倍），模型占端到端 `7.7685201%`（約7.8%），其餘 `3.0238810 ms`（約3.02），raw差 `.1423519861 ms`（約.142）。paired差／PyTorch端到端約 `7.5960673%`，可約為7.6%；主代理後續已在正文寫出這項成對比值，核算正確。B1吞吐 `8901.1523`（8,901張/秒）、B2 `20217.5393`（20,218張/秒），吞吐比 `2.2713396`（2.3倍）。兩張分開 vs 一批兩張省 `.1257660 ms`（.13ms）。原有10ms等湊batch情境與前後處理不可忽略的教學結論仍成立。
- **第 17 章 quality 保留**：baseline class AP `.3333/.5556`、mAP`.4444`、precision`.6429`、recall`.5294`；changed AP`.6250/.7778`、mAP`.7014`、P`.8000`、R`.7059`。coverage9/4/4→12/2/3、定位FP4→2、背景FP各1、wrong-class/duplicate0、keep_change=true、chosen test `.4452/.6364/.5385` 均不變。#10 baseline FP score`.0679774`仍約`.068`／圖標`.07`；#14 baselineIoU`.227509`仍`.23`、changed`.491688`仍`.49`。因此 97/99/105/139–167/215 的數字與分類／選擇結論都保留。
- **第 4 章 rounded training evidence**：NEW last_pre_update_loss `.4302132725715637`仍六位`.430213`；initial `.809883177280426`仍`.809883`；兩個 IoU `.6944893598556519/.7752412557601929`仍`.6945/.7752`，兩位箱座標全部一致。更新前／後資料量、accuracy1.00、validation未評估與訓練圖不能當泛化證據的結論不變。
- **第 10 章 rounded training evidence**：NEW initial`3.6593661308288574`→`3.6594`、final`.07033687829971313`→`.0703`、weightdelta`6.503562899854255`→`6.504`。小框[`6.3247118,1.3950167,11.3533049,14.8066168`]仍[`6.3,1.4,11.4,14.8`]；寬約5、高13.4；class0 AP0/class1 AP1/mAP.5、P1/3/R.5保留。額外FPscore`.13139395`仍`.13`；各IoU rounded值仍.44/.86/.30/.28。
- **第 8 章 short/default case**：160步 minheight`.0545400679`仍約`.05px`；final_train mAP`.0068027211`仍`.00680`、validation0。第144–153步的小loss區間、第154–158步分類尖峰與第157/159/160步box≈.02的描述仍成立；只有第158步total四捨五入需1.87。1600步紀錄的 `prior_diagnostic_comparison.loss_history_identical=true` 與 steps_compared160直接支持前160步逐值相同。
- **第 8 章例圖新判定**：red primary IoU`.58455387` TP；extra class2 score`.15817508`、max-any-IoU`.24119306` FP；blue score`.46211660`、IoU`.40580147` FP+FN；yellow score`.99462885`、IoU`.42066457` FP+FN；empty無預測。不可只更新mAP而保留舊blue TP或「三張各一框」結論。
- **第 12、15、16 章具體變更**：12-anchor-free NEW中心(28,28)／ltrb[2,1.5,1.5,1]和正文匹配；15-attention-bridge 新重建目標的loss`.1595`，正文未引用舊`.1795`；16-dfl-free NEW中心(84,84)、direct[74,64,228,108]／signed[92,68,108,116]正文均已同步。這些是程式／教學案例的預先修改，沒有額外必修。
- **08-own-images default**：NEW門檻.25留下0框；正文沒有寫舊16框，且已說明只憑框數不判定checkpoint載入或品質。原始往返座標四捨五入到四位為[20,10,60,30]，metadata不變，不需要新增數字。
- **Fashion-MNIST**：40個loss的min`2.1085872650`、max`2.3857543468`仍2.11–2.39；validation`.140625`＝18/128→.1406、test`.09375`＝12/128→.0938仍正確。它未學會分類／不是偵測證據的結論保留。
- **第 7 章 cross-page**：NEW grid final_validation.map`.8035714286`→.8036（.80）、final_test.map`.7748917749`→.7749（.77）不變；17-capstone:153 引述第7章.80 vs baseline.44仍正確；07-heldout與07-inference的延伸引用沒有額外舊時間或loss末位quote。

## 映射之外搜尋

實際紀錄檔名搜尋發現 `docs/validation/curriculum.md` 引用所有逐節及補充紀錄，也引用保留的兩份GPU紀錄；原PAGES mapping未列它。該頁列的是紀錄連結、審查連結與證據類型，沒有引用這次CPU變動的loss、AP、框座標或時間，因此無額外数值修正。
另掃過 README.md、docs/index.md、docs/learning-path.md、docs/planning/outline.md、docs/glossary.md、docs/preparation/{architecture,data,publish}.md、docs/research/、docs/validation/。README／首頁／路線／outline現有受控資料／步數／泛化限制等結論仍受新record支持。glossary的ltrb例、attention權重與門檻例都吻合；其.3333/.6667/.5等是人工計算或別一實驗，不能因custom-data的指標變了而全域替換。
未將timestamps、hashes、版本或新增machine欄位自動當成正文修正；但正文直接引用的CPU型號與耗時已全部列在必修清單中。GPU兩份existing紀錄的數字未變，原L4數值與範圍說明不需改，未重跑GPU。

## 全部51份CPU紀錄覆蓋

下表用「紀錄」後面的完整filename作唯一對應；source數值来自该JSON或其stdout。

| 紀錄 | 具體核對結果 |
|---|---|
| artifacts/checks/curriculum/00-warmup.json | 手算 2→3.6、loss4→.16、gradient−8、1×1 shape 正確；正文已含新增 new_prediction，沒有旧 timing quote。 |
| artifacts/checks/curriculum/01-small-cnn-learning.json | initial、zero-point與gradient min/max必修；40steps/8samples、Adam.01、train1.0、validationnull、all-correct/classes[0,1,..]結論不變。 |
| artifacts/checks/curriculum/01-small-cnn.json | 短例 step0 loss需.6942；1158參數、479248MAC、3步全猜1、錯誤0/2/4/6仍正確。 |
| artifacts/checks/curriculum/02-diagnostics.json | train/validation loss .5945/.7937→.1236/2.0153、accuracy1/0、權重±.195/±.975皆仍正確；末位權重差不需改字。 |
| artifacts/checks/curriculum/03-comparison-learning.json | 六位initial/final表必修；train/validation .5/.5 plain對1/1 residual、986params、單seed/4heldout限制不變；.031 final近似仍正确。 |
| artifacts/checks/curriculum/03-comparison.json | 短例residual初始loss需.6921；約.15/.001 stem gradient、986params、248840MAC、3072adds、兩者validation.50仍正確；正文不引用两個秒數的數值或聲稱誰較快。 |
| artifacts/checks/curriculum/03-identity.json | 負值保留、四個input gradient1、2×4×8×8 shape與288參數仍吻合，negatives_preserved用詞匹配。 |
| artifacts/checks/curriculum/03-projection.json | 新增position probe[[0,2],[20,22]]與正文匹配；321、504params、两步loss均不變。 |
| artifacts/checks/curriculum/04-coordinates.json | 新增red_pixel_box[8,20,40,36]、odd restored細節已在正文；sx/sy.7711/.7838與其他手算皆正確。 |
| artifacts/checks/curriculum/04-localization-learning.json | 所有已引用的rounded值不需改：.809883→.430213、boxes[4.48,4.77,18.13,18.93]/[16.33,10.58,29.18,23.12]、IoUs.6945/.7752、train1.0/validationnull。 |
| artifacts/checks/curriculum/04-localization.json | 短例loss .8099→.7576、CE .7173→.7123、box .0185→.0091、shape/targets不變；只有預測座標末位漂移，不需改字。 |
| artifacts/checks/curriculum/05-assignment.json | 新增不對稱fixture的cells[[0,0],[1,2]]、ltrb/target/class順序與正文匹配；原兩正格/30負格、碰撞拒絕、loss皆不變。 |
| artifacts/checks/curriculum/06-decode-nms.json | 僅將座標穩定印為23.2/4.0；scores.64/.72/.855、IoU.5385、NMS/threshold練習的候選次序皆不變。 |
| artifacts/checks/curriculum/06-evaluation.json | stdout AP50→mAP50命名已在正文；TP2/FP2/FN1、1/3、1/6、5/9及no-GT排除邏輯一致。 |
| artifacts/checks/curriculum/07-data.json | 輸出未變；兩張圖counts[2,0]、RGB sums256、8-image contract吻合。 |
| artifacts/checks/curriculum/07-heldout.json | 輸出未變；人工mAP.5/P1/3/R.5、3步heldout全部0；正文仍明確區分人工與pipeline smoke。 |
| artifacts/checks/curriculum/07-inference.json | 輸出未變；未訓練[0,0,0]、人工[2,1,0]、red[8.0016,12,24.0016,28]、NMS2→1吻合。 |
| artifacts/checks/curriculum/07-loss.json | 新增class gradients[−.5,.5]已與正文手算一致；total1.933169、box.109375、obj/CE.693147、obj gradients±.03125及空圖規則不變。 |
| artifacts/checks/curriculum/07-targets.json | 輸出未變；red[0,.25,.25,.25]、blue[0,.75,.25,.25]、2正/30負與兩格索引吻合。 |
| artifacts/checks/curriculum/07-training.json | 短3步的所有分項loss與objectness mean未變，沒有額外必修；160步硬體/時間另見grid-learning。 |
| artifacts/checks/curriculum/08-own-data.json | 短case輸出未變；6PNG、2/2/2split、三類head[2,4,4,8]、一步loss1.4407與錯誤拒絕結論吻合；長訓練另見兩份custom-data。 |
| artifacts/checks/curriculum/08-own-images.json | 新顯示門檻.25留0框；正文已講明任意框數不能當品質/載入證據，沒有硬寫舊16框。metadata與四捨五入roundtrip[20,10,60,30]吻合。 |
| artifacts/checks/curriculum/09-anchor-clustering.json | 輸出未變；mean-anchor[8.333,8.333]/[31.333,16.667]、coverage.3817→.9119、median.934/newsource.1975仍正確且不是AP。 |
| artifacts/checks/curriculum/09-anchors.json | 輸出未變；sizeIoU[1,.25]、logits−9.21024/−1.09861、1positive/1ignore/30negative、red還原座標吻合。 |
| artifacts/checks/curriculum/10-multiscale-learning.json | 全部rounded值仍正確：loss3.6594→.0703、weightdelta6.504、smallbox[6.3,1.4,11.4,14.8]、IoU.44、bigIoU.86、extraFPscore.13/IoU.30/pairIoU.28、trainmAP.5。 |
| artifacts/checks/curriculum/10-multiscale.json | 短case輸出未變；4×4/8×8head、80候選、兩尺度duplicate2→1及小框targets皆吻合。 |
| artifacts/checks/curriculum/11-augmentation.json | 輸出未變；flip[40,12,56,28]、crop[0,4,8,20]、visibility128/256=.5、.5/.6label保留/刪除吻合。 |
| artifacts/checks/curriculum/11-csp.json | 輸出未變；concat/add數字、full1240/CSP368參數、gradient sums與沒有accuracy比較結論吻合。 |
| artifacts/checks/curriculum/11-fusion.json | 新增shape完整鏈與正文吻合：deep16ch→reduce8ch→nearest8×8、concat16ch→mix8ch；1296params、sourcegradient全4正確。 |
| artifacts/checks/curriculum/11-iou-loss.json | 只改穩定印法；gradient[.02,0]、GIoU1.2→1.179487、DIoU1.310345→1.294497、center[38.7515,20]仍正確。 |
| artifacts/checks/curriculum/12-anchor-free.json | NEW候選點(28,28)、target[2,1.5,1.5,1]、.376236→.000012、learnedbox[12.01,16.03,39.97,35.94]；正文已使用新座標/target，沒有舊learnedbox quote。 |
| artifacts/checks/curriculum/12-assignment.json | 僅格式精简；品質.225/.311/.147/.567、owner[0,0,1,−1]、gradients±.125、empty/練習owner吻合。 |
| artifacts/checks/curriculum/12-decoupled-head.json | 輸出未變；shapes(2,4,4,4)/(2,2,4,4)、classification-only boxgrad=None、cosine−.0073、1446params吻合。 |
| artifacts/checks/curriculum/12-dfl.json | 僅格式精简；uniform1.5/DFL1.386294、gradient[.25,−.5,0,.25]、概率.0025/.7475/.2474/.0025、1.2500cells/9.9999px吻合。 |
| artifacts/checks/curriculum/13-dual-assignment.json | 輸出未變；many[0,0,1]/one[1,0,−1]、global1.73/greedy1.10、backbone detach與loss.6855/.7065吻合。 |
| artifacts/checks/curriculum/13-nms-free.json | 僅格式精简；scores.959/.041、候選[0,1,2]→[0,2]、top2misses2、duplicateIoU.8182吻合。 |
| artifacts/checks/curriculum/14-feature-module.json | concat-slot→concat-segment字樣與正文一致；4+4+4+4channels、1168/800params、MSE1.0722→1.0049、L1.2967/.3349/.3481/.2779未變。 |
| artifacts/checks/curriculum/15-area-attention.json | 印5位精確值與正文一致；full.46875/area.09375、干預full.78125、256/64pair與區外不影響結論吻合。 |
| artifacts/checks/curriculum/15-attention-bridge.json | 新重建目標對應loss.1595；正文沒有舊.1795quote，已描述channel1×.5、非零且不同的Q/K/V梯度。attention/練習權重與輸出四位數不變。 |
| artifacts/checks/curriculum/16-dfl-free.json | 候選點已(84,84)、directbox[74,64,228,108]、signedbox[92,68,108,116]；正文匹配新decode。gradient−.25/step.125、K16range0..15、18-distance、6600/600values与bytes不變。 |
| artifacts/checks/curriculum/16-inference-head.json | 精简概率[.5,.75]與retainedraw maxdifference0的字樣已匹配正文；training/deploy332/278params、manual[8,32,56,64]與無NMS說明吻合。 |
| artifacts/checks/curriculum/16-training.json | 新增等總量對照與正文一致；fixedMSE.663073、progressive.016285、matched.016882、b總量6/16.5、所有首步六位數、STAL0→4與GT不變都正確。 |
| artifacts/checks/curriculum/17-capstone.json | 額外必修只有train時間與end-to-end中位數；所有quality/table/coverage/FPcounts/keepchange/chosen-test均不變，細節的score/IoU漂移在正文所用精度仍正確。 |
| artifacts/checks/curriculum/18-video.json | 額外計時與FPS需同步；12幀counts[1,1,0,0,0,0,1,1,0,0,0,0]、最后timestamp.55、warmup1與漏檢8幀結論不變。 |
| artifacts/checks/curriculum/19-tracking.json | 只换stdout说明及artifact路径；3switch/0switch、11/12detector recall、FP0、maxage2、人工IDs次序与同detections比較仍正確。 |
| artifacts/checks/curriculum/20-deployment.json | 額外必修rawB1 error与CPUtiming/ratio/throughput；B1/2/3parity、16boxes各source、dynamicbatch/fixedspatial、80rejected、非品質證據結論不變。 |
| artifacts/checks/curriculum/custom-data-160-step.json | full-trainfinal.16977/step158peak1.87需改；trainmAP.00680/val0、minheight.05454≈.05px、21GT/21positive、train/val/test拆分與短訓練不穩結論保留。 |
| artifacts/checks/curriculum/custom-data-learning.json | full-loss/val/test/gradients/weightdelta/timings與具体example判定必修；trainmAP/P/R1、21TP0FP0FN、9GT各heldout、exactreload/checkpoint與prior160 loss逐值相同仍成立。 |
| artifacts/checks/curriculum/fashion-mnist-learning.json | 只兩個history浮點末位變化；min2.108587→2.11/max2.385754→2.39、val.140625→.1406(18/128)、test.09375→.0938(12/128)与40steps/pipeline-only結論不變。 |
| artifacts/checks/curriculum/video-file.json | pixel/codec equality、12frames/20FPS/.55timestamp、frame counts与tracking IDs不變；保存框末位漂移仍给frame0/1/6/7 IoU≈.50/.47/.30/.46。 |
| artifacts/checks/grid-learning.json | CPU/seconds必修；finaltrainloss.003813与此前常用.0038精度一致；initialval.0018/finalval.8036/test.7749/P.9375/R.8333等heldout quality不變。 |

## 可用產物

詳細報告：`/tmp/lessons-v0.4.0-numeric-audit.md`；具體修正組清單：`/tmp/lessons-v0.4.0-numeric-audit.json`。JSON已含exact page/line、record/source keys、old/new claim與並行修改狀態，方便主代理直接核對。
沒有找到需更改教學程式、notebook、生成圖或evidence的額外問題。

# 匿名判分規則

只讀指定 bundle 與本指引。全部17個來源與 gold 已在首次讀者開始前凍結；items 只有匿名 evaluation_id、case_id、共同格式的逐段問題與最終判定，沒有組別。不要讀 private/identities、protocol、其他讀者紀錄、作者預期或repo其他檔；不要猜組別。文字風格仍可能洩露流程，不能聲稱完全盲評。

bundle 是全部17來源中一部分cases及其15份匿名判斷。用Python按case、unit、最多五個items切片讀，保持每次輸出在工具上限內。不要一次cat完整JSON，也不要因顯示截斷漏看items；同一合法bundle可反覆縮小切片。不需要讀其他bundle。

gold 是來源核定者判定，不是作者的設計期待。每 item 逐一判：
- target_hit：最終 issues 具體指出 gold target_gap，且有位置、承認已教部分、仍缺的主張、當下使用影響與可見引文。只有「未知／不完整／待教」或不通過，不算命中。缺項可用讀者自己的話，不需逐字複製 gold。
- severity_match：gold necessary（burden/blocker）容許兩種必要分級，但實際影響需支持，不可把局部缺漏擴成整個方法沒教；gold optional 只能 optional。none 沒有 target，target_hit/severity_match 記 false，是否正常通過另外計數。
- off_target_necessary：最終有任何不成立的 burden/blocker（包括把只缺名稱升級，或要求未宣稱的圖、訓練、比較與內部細節）。成立的目標缺漏不算 off-target。
- timely_target_hit：timeline_issues 在 gold first_required_unit 當場有具體適當命中。頁末才發現不能充當首次命中；核證者的 timeline 是獨立來源判斷，不是第一位四題成績。
- premature_necessary_flag：問題在尚未需要的前段就被升為必要缺漏，且下一段是合理立即解釋／當時只是符號介紹而非需數值操作。區分正當的當時卡點，不能一概視作早報。
- record_concerns：只針對傳入的issues/timeline/verdict之具體證據問題，如引文不支持完整缺漏主張、已教部分被抹掉、借未讀内容、最終判定與自己的問題矛盾。沒有傳入四題原回答，所以不能聲稱已評完所有核心主張的語義。

對 none 源必要誤報必須寫來源理由；對 name-only 不要漏判「全名optional，但又因未教loss/更新法升burden」。真正必要缺項可能在 unit3 才啟用，u1–u2 不一定已有問題。以當下已讀來源推論，不因自己知道AP50/NMS慣例就假設教材已寫。

輸出指定 grades-<N>.json：
```json
{"grades":[{"evaluation_id":"E...","case_id":"M...","verdict":"原item的verdict",
"target_hit":true,"severity_match":true,"off_target_necessary":false,
"timely_target_hit":true,"first_target_unit":"u3或null","premature_necessary_flag":false,
"record_concerns":[],"reason":"具體指出命中或漏掉什麼、source依據及分級界線，30–100字即可"}],
"access_declaration":"只讀允許的指引和bundle","limitations":"只評已傳入判斷，不宣稱全書、全claim或真人學習驗證"}
```
每個 item 一筆，不刪不利結果、不改 gold，不看作者的組別期待。source/gold 如有實質可疑，記 record_concerns 與 reason，按凍結gold作主分，另列 sensitivity，不在評分後改 gold。

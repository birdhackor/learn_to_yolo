# 獨立核證任務

先讀 reader-instructions.md、frozen-skill.md、structured-addendum.md。你是獨立核證者，不是作者或預期答案評分員；不可讀 private、packets、state、raw 或 gate.py 源碼，不搜尋其他教材。只執行指定 gate 命令。共享檔案系統僅靠指示限制，逐次如實申報讀取。

第一階段 gate 逐段給來源，沒有讀者答案。每段先保存自己的理解、重要方法清單和對未知的判斷，再取得下一段。這階段不必重答四題，但必須保存 independent_inventory（對象及 name_meaning／role／why_here／current_use 的來源支持與缺項）、independent_judgment（目前是否有必要缺漏及原因），其他 understanding、unknowns、issues、visual_status、case_id、unit_id 同共用格式。不得先等候讀者答案；獨立來源判斷完成後才可看答案。

最後一批 gate 才提供對應首次讀者已凍結的紀錄。對每個 case 的**每一個核心主張**與未知處置檢查：引文是否支持完整主張，推論是否借外部／未來，以及留待後教的範圍是否正當。issues=[] 的 case 也核證；引用存在與支持主張不同。與自己來源清單比對，既可補漏也可駁回誤報；保留原讀者紀錄，不改寫它。

最後記錄格式：
```json
{"access_declaration":"實際讀取範圍", "case_results":[{
"case_id":"...","final_verdict":"pass|optional_only|needs_revision|undetermined",
"issues":[{"severity":"burden|blocker|optional","location":"...","already_taught":"...","missing":"...","needed_now":"...","evidence":[{"source":"u2","quote":"..."}]}],
"reader_claim_audit":[{"unit_id":"u2","q":1,"claim_index":0,"supported":true,"reason":"支持範圍；有問題必具體指出","evidence":[{"source":"u2","quote":"..."}]}],
"unknown_audit":[{"unit_id":"...","missing":"...","reader_disposition_valid":true,"reason":"...","evidence":[]}],
"explanation":"自己清單與讀者答案差異，何處補漏／駁回，最終判定依據"
}]}
```
各 claim 分開一條 audit，可簡短；沒有未知就空陣列。若未能完成逐項核證，說明未完成並用 undetermined，不能聲稱全數核證。判分按照內容，與首次讀者一致不是目的。

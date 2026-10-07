# clear-tutorial：通用準則與專案參考

使用入口為 [SKILL.md](SKILL.md)。主入口、寫作 prompt 與審閱流程使用跨主題準則，專案背景和技術事實另存；已知問題與逐句修正不放進 active skill。

| 文件 | 用途與使用者 |
| --- | --- |
| [SKILL.md](SKILL.md) | 作者與審閱者的通用寫作、例子、理由、比較及驗收準則 |
| [寫作短 prompt](references/writing-prompt.md) | 寫作時可直接使用的通用指令 |
| [審閱流程](references/review-protocol.md) | 協調者安排逐段閱讀、保存證據及比對歷史問題 |
| [專案背景與技術核對](references/project-context.md) | 本 repo 的作者、協調者與技術審閱者按需要查核 |
| [UPSTREAM.json](UPSTREAM.json) | 固定來源版本與原始檔案雜湊的維護資料 |

首次閱讀者使用通用判準、實際入口背景及當前閱讀包，不提前取得專案技術答案、已知問題或同款修正案例。先保存新判斷，再由協調者比對舊問題；允許舊問題不成立或分級需修正，不以命中答案表作唯一驗收。

本版補強三個相連的判準：方法如何改變問題或工作、例子是否足以呈現其教學目的，以及比較是否公平描述其他方法。操作正確、理解共同規則、理解設計用意與支持實際效果分開核對，不硬性規定每個例子都用多組數字。

更新 skill 不代表教材已改寫或重新審閱。只分析、只審閱或只改 skill 時，依使用者指定的階段完成並如實報告，不自行擴大範圍。

## 來源與維護

本 skill 以 [tiny-perceptron-vlm 的固定版本](https://github.com/birdhackor/tiny-perceptron-vlm/tree/6b8fb865b6f4f5f1c1649420520232f9911c1227/.agents/skills/clear-tutorial) 為基礎，保留易懂寫作與逐段審閱方法，再依目前工作需求調整文件分工。

[UPSTREAM.json](UPSTREAM.json) 的雜湊描述固定 upstream 原始檔案，不是本地適用版的雜湊。逐位元組首次匯入保存在 [commit 439603b](https://github.com/birdhackor/learn_to_yolo/tree/439603b0380ee6e75b85fb8e8cf5d5ef3aa4872f/.agents/skills/clear-tutorial)。同步來源時應比較差異並保留本地通用準則及專案約定。

來源 VLM 案例與先前教材正誤例已移出 active skill，歷史由審閱紀錄與 Git 保存；它們不是本專案驗收證據，也不作下一輪首次閱讀提示。專案參考只保存背景、事實與操作約定，不收納特定課文的改寫答案。

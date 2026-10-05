# clear-tutorial：YOLO 專案版

使用入口是 [SKILL.md](SKILL.md)。YOLO 補充規則已直接整合進主入口，不需另讀這份 README 才能取得：

- 依使用者指定的 Python／PyTorch、NN／CNN 基礎與不熟練的大學數學設定讀者背景，維持教材已公布的較低門檻。
- 對齊圖片、框、張量、座標與單位，核對圖、圖說及實際 Zensical 頁面。
- 以起點、問題、改動、理由、收益與代價講版本演進，區分簡化示範與控制比較。
- 說明偵測指標與訓練證據的含義，不把 loss 下降或小型 GPU 測試當成泛化或正式效能。
- 逐段提供首次閱讀材料並保存當場理解，再分輪查技術與銜接；保留原始問題及修正後複查。

配套文件也已同步：

- [寫作短 prompt](references/writing-prompt.md)
- [逐段閱讀審閱規則](references/review-protocol.md)
- [來源 VLM 專案 10.5 的問題示範](references/review-example-10-5.md)

更新 skill 不代表教材已照新規則改寫或重新驗收；每次使用仍以實際完成的工作範圍報告結果。

## 來源與維護

本 skill 以 [tiny-perceptron-vlm 的 `6b8fb865`](https://github.com/birdhackor/tiny-perceptron-vlm/tree/6b8fb865b6f4f5f1c1649420520232f9911c1227/.agents/skills/clear-tutorial) 為基礎，保留其易懂寫作與逐段審閱方法，再加入 YOLO 約定。

[UPSTREAM.json](UPSTREAM.json) 記錄的是固定來源版本的原始檔案雜湊，不是目前本地適用版的雜湊。逐位元組的首次匯入保存在 [commit `439603b`](https://github.com/birdhackor/learn_to_yolo/tree/439603b0380ee6e75b85fb8e8cf5d5ef3aa4872f/.agents/skills/clear-tutorial)。之後同步來源時應比對差異，保留本專案的規則。

10.5 範例屬於來源 VLM 專案，已改用固定來源版本連結並明確標示；它不能當成 YOLO 教材的審閱證據。

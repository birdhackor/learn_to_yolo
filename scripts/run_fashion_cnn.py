"""Fashion-MNIST 真實分類 smoke：--steps 2；可選 --train-steps 自行加長。

結果寫到 --output（預設 artifacts/runs/fashion-cnn/，不進 Git）。網站引用的那次執行另加 --record，
把結果連同執行的日期、電腦與程式（SHA-256）寫進 artifacts/checks/curriculum/fashion-mnist-learning.json。
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

# 允許直接從 repo 執行 python scripts/run_fashion_cnn.py，不要求 pip 安裝 package。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from miniyolo.classification import train_fashion_cnn

RECORD = ROOT / "artifacts/checks/curriculum/fashion-mnist-learning.json"
RECORDED_RUN = {"steps": 40, "subset": 64, "eval_samples": 128, "batch_size": 8, "device": "cpu"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=2, help="小 subset 管線驗證步數，預設2")
    parser.add_argument("--train-steps", type=int, help="自行加長訓練步數；不設 --subset 時使用完整54k訓練split")
    parser.add_argument("--subset", type=int, help="限制訓練圖數；短 smoke 預設64")
    parser.add_argument("--eval-samples", type=int, default=128, help="各評估split圖數，0表示完整")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-root", type=Path, help="已通過SHA256檢查的Fashion-MNIST gzip目錄")
    parser.add_argument("--output", default="artifacts/runs/fashion-cnn")
    parser.add_argument("--record", action="store_true",
                        help="另寫網站引用的紀錄；只用於 --train-steps 40 --subset 64 --eval-samples 128（batch 8、CPU）")
    args = parser.parse_args()
    steps = args.train_steps if args.train_steps is not None else args.steps
    subset = args.subset if args.subset is not None else (64 if args.train_steps is None else None)
    run = {"steps": steps, "subset": subset, "eval_samples": args.eval_samples, "batch_size": args.batch_size,
           "device": args.device}
    if args.record and run != RECORDED_RUN:
        parser.error("--record 只用於網站引用的設定：--train-steps 40 --subset 64 --eval-samples 128（batch 8、CPU）")
    result = train_fashion_cnn(steps=steps, root=args.data_root, subset=subset, eval_samples=args.eval_samples,
                               device=args.device, batch_size=args.batch_size, output=args.output)
    if args.record:
        import torch
        from miniyolo.provenance import machine_info, repo_dependencies
        manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
        files = next(d for d in manifest["datasets"] if d["id"] == "fashion-mnist")["resources"]
        record = {"executed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "machine": machine_info(), "torch": str(torch.__version__),
                  # 這支 script 與它直接或間接 import 的每個 repo 模組（scripts/evidence_records.py）。
                  "dependencies_sha256": repo_dependencies(__file__),
                  # loader 讀檔前已核對這四個檔的大小與 SHA-256 和 manifest 相符。
                  "data_sha256": {item["filename"]: item["sha256"] for item in files},
                  **result}
        # 結果檔寫好之後才寫紀錄：中途失敗時紀錄維持過期，scripts/record_evidence.py 會再跑一次。
        RECORD.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

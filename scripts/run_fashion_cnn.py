"""Fashion-MNIST 真實分類 smoke：--steps 2；可選 --train-steps 自行加長。"""

import argparse
from pathlib import Path
import sys

# 允許直接從 repo 執行 python scripts/run_fashion_cnn.py，不要求 pip 安裝 package。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from miniyolo.classification import train_fashion_cnn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2, help="小 subset 管線驗證步數，預設2")
    parser.add_argument("--train-steps", type=int, help="自行加長訓練步數；不設 --subset 時使用完整54k訓練split")
    parser.add_argument("--subset", type=int, help="限制訓練圖數；短 smoke 預設64")
    parser.add_argument("--eval-samples", type=int, default=128, help="各評估split圖數，0表示完整")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-root", type=Path, help="已通過SHA256檢查的Fashion-MNIST gzip目錄")
    parser.add_argument("--output", default="artifacts/runs/fashion-cnn")
    args = parser.parse_args()
    steps = args.train_steps if args.train_steps is not None else args.steps
    subset = args.subset if args.subset is not None else (64 if args.train_steps is None else None)
    train_fashion_cnn(steps=steps, root=args.data_root, subset=subset, eval_samples=args.eval_samples,
                     device=args.device, batch_size=args.batch_size, output=args.output)


if __name__ == "__main__":
    main()

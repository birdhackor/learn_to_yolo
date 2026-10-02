"""真實 Fashion-MNIST IDX gzip loader；使用已校驗快取，不自動下載。

Fashion-MNIST 是 28×28 灰階分類資料，沒有 bbox。本例把 1 個灰階通道
重複成 3 個相同通道，配合 TinyCNN 的 RGB 輸入介面；這不會增加色彩資訊。
"""

import gzip
import json
import struct
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, Subset

from .models import TinyCNN
from scripts.download_data import verify


ROOT = Path(__file__).resolve().parents[1]
FASHION_CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                       "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


class FashionMNISTDataset(Dataset):
    """實際解析 magic/count/payload；每張輸出 Float[3,28,28] 和 Long scalar。"""

    def __init__(self, split="train", root=None):
        if split not in ("train", "test"):
            raise ValueError("split 必須是 train 或 test")
        root = Path(root) if root is not None else ROOT / "data/downloads/fashion-mnist"
        manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
        resource_map = {item["filename"]: item for dataset in manifest["datasets"]
                        if dataset["id"] == "fashion-mnist" for item in dataset["resources"]}
        prefix = "train" if split == "train" else "t10k"
        payloads = []
        for suffix in ("images-idx3-ubyte.gz", "labels-idx1-ubyte.gz"):
            filename = f"{prefix}-{suffix}"
            path = root / filename
            if not path.exists():
                raise FileNotFoundError(f"缺少 {path}；先跑 python scripts/download_data.py fetch fashion-mnist")
            if not verify(path, resource_map[filename]):
                raise ValueError(f"Fashion-MNIST gzip 大小或 SHA256 錯誤：{path}")
            with gzip.open(path, "rb") as stream:
                payloads.append(stream.read())
        images, labels = payloads
        if len(images) < 16 or len(labels) < 8:
            raise ValueError("IDX header 截斷")
        magic_images, count, height, width = struct.unpack(">IIII", images[:16])
        magic_labels, label_count = struct.unpack(">II", labels[:8])
        expected_count = 60000 if split == "train" else 10000
        if (magic_images != 2051 or magic_labels != 2049 or count != expected_count
                or count != label_count or (height, width) != (28, 28)):
            raise ValueError("Fashion-MNIST IDX magic/count/shape 不符")
        if len(images) != 16 + count * height * width or len(labels) != 8 + count:
            raise ValueError("IDX payload 大小不符")
        # bytearray 提供可寫 buffer，clone 讓 dataset 擁有獨立的 tensor 儲存。
        self.images = torch.frombuffer(bytearray(images[16:]), dtype=torch.uint8).reshape(count, height, width).clone()
        self.labels = torch.frombuffer(bytearray(labels[8:]), dtype=torch.uint8).long()
        if ((self.labels < 0) | (self.labels > 9)).any():
            raise ValueError("Fashion-MNIST class id 超出 0..9")
        self.split, self.class_names = split, FASHION_CLASS_NAMES

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        gray = self.images[index].unsqueeze(0).float() / 255
        return gray.repeat(3, 1, 1), self.labels[index]


@torch.no_grad()
def classification_accuracy(model, dataset, device="cpu", batch_size=64):
    model.eval()
    correct, count = 0, 0
    for images, labels in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        labels = labels.to(device)
        predicted = model(images.to(device)).argmax(-1)
        correct += int((predicted == labels).sum())
        count += len(labels)
    return correct / count if count else 0.


def train_fashion_cnn(steps=2, root=None, subset=64, eval_samples=128,
                      device="cpu", batch_size=8, output="artifacts/runs/fashion-cnn"):
    """短 smoke 與自行加長訓練使用相同程式；兩步只驗證管線，不宣稱效果。"""
    if steps < 1 or batch_size < 1 or eval_samples < 0 or (subset is not None and subset < 1):
        raise ValueError("steps/batch_size/subset 必須為正，eval_samples>=0")
    torch.manual_seed(7)
    torch.set_num_threads(2)
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("此環境無可用 CUDA；短驗證請使用 --device cpu")
    dataset, official_test = FashionMNISTDataset("train", root), FashionMNISTDataset("test", root)
    # 從官方 train 以 seed=7 切 54k train/6k validation；test 完全保留官方邊界。
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(7)).tolist()
    validation_indices, training_indices = order[:6000], order[6000:]
    if subset is not None:
        training_indices = training_indices[:subset]
    if eval_samples:
        validation_indices = validation_indices[:eval_samples]
    train_data = Subset(dataset, training_indices)
    validation = Subset(dataset, validation_indices)
    test = Subset(official_test, list(range(min(eval_samples, len(official_test))))) if eval_samples else official_test
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(7))
    model = TinyCNN(num_classes=10, width=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    before = model.classifier.weight.detach().clone()
    iterator, history = iter(loader), []
    print("Fashion-MNIST: grayscale [1,28,28] -> repeat 3 identical channels [3,28,28]; no bbox")
    for step in range(steps):
        try:
            images, labels = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, labels = next(iterator)
        model.train()
        logits = model(images.to(device))
        loss = torch.nn.functional.cross_entropy(logits, labels.to(device))
        if not torch.isfinite(loss):
            raise RuntimeError(f"step {step + 1} loss 非有限值")
        optimizer.zero_grad()
        loss.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError("分類器梯度非有限值")
        optimizer.step()
        history.append({"step": step + 1, "loss": float(loss.detach())})
        if step < 2 or (step + 1) % 100 == 0 or step + 1 == steps:
            print(json.dumps({"event": "train_step", **history[-1]}))
    changed = not torch.equal(before, model.classifier.weight.detach())
    if not changed:
        raise RuntimeError("classifier 權重沒有更新")
    result = {
        "steps": steps, "train_samples": len(train_data), "validation_samples": len(validation),
        "test_samples": len(test), "seed": 7, "device": str(device), "batch_size": batch_size,
        "optimizer": "Adam", "learning_rate": .001, "class_names": FASHION_CLASS_NAMES,
        "validation_accuracy": classification_accuracy(model, validation, device),
        "test_accuracy": classification_accuracy(model, test, device), "weights_changed": changed,
        "loss_history": history, "accuracy_definition": "argmax class correct / evaluated images",
        "split": "official train seed7 permutation: first6000 validation, remaining54000 train; official test preserved",
        "input": "gray [1,28,28] repeated to three identical channels, float32 [0,1]",
        "limitations": "A two-step smoke confirms learning plumbing; it is not evidence of useful classification accuracy.",
    }
    run = Path(output)
    run.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "class_names": FASHION_CLASS_NAMES,
                "config": {"num_classes": 10, "width": 8, "input_channels": 3, "seed": 7},
                "steps_completed": steps}, run / "checkpoint.pt")
    (run / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "heldout_classification", "validation_accuracy": result["validation_accuracy"],
                      "test_accuracy": result["test_accuracy"], "weights_changed": changed}))
    return result

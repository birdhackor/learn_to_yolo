"""固定、無下載的紅/藍矩形分類資料；受控色彩任務，不代表自然照片辨識。"""

import torch
from torch.utils.data import Dataset


class ColorRectangles(Dataset):
    """images=[N,3,H,W] float32 [0,1]；labels=[N] long；boxes=[N,4] xyxy pixel。

    0=red、1=blue，只分類矩形顏色。位置、寬高及亮度由本地 generator
    產生且不依賴 label；兩類都是矩形，不能把結果說成學會形狀分類。
    """

    class_names = ("red", "blue")

    def __init__(self, samples=128, image_size=32, seed=101):
        if samples < 1 or image_size < 8:
            raise ValueError("samples 必須為正，image_size 至少8")
        self.seed, self.image_size = seed, image_size
        generator = torch.Generator().manual_seed(seed)
        self.labels = (torch.arange(samples) % 2)[torch.randperm(samples, generator=generator)].long()
        self.images = .06 + .04 * torch.rand(samples, 3, image_size, image_size, generator=generator)
        self.boxes = torch.empty(samples, 4, dtype=torch.long)
        for index, label in enumerate(self.labels):
            width, height = torch.randint(image_size // 4, image_size // 2 + 1, (2,), generator=generator).tolist()
            left = int(torch.randint(0, image_size - width + 1, (1,), generator=generator))
            top = int(torch.randint(0, image_size - height + 1, (1,), generator=generator))
            bright = float(.75 + .20 * torch.rand((), generator=generator))
            color = torch.tensor([bright, .12, .12] if int(label) == 0 else [.12, .12, bright])
            self.images[index, :, top:top + height, left:left + width] = color[:, None, None]
            self.boxes[index] = torch.tensor([left, top, left + width, top + height])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


def make_color_splits(train_samples=128, val_samples=64, test_samples=64, image_size=32,
                      seeds=(101, 202, 303)):
    """獨立 seed 生成三批資料；validation/test 不進 optimizer，也非真實資料泛化證據。"""
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("train/validation/test 必須使用三個不同 seed")
    return {name: ColorRectangles(samples, image_size, seed)
            for name, samples, seed in zip(("train", "val", "test"),
                                          (train_samples, val_samples, test_samples), seeds)}

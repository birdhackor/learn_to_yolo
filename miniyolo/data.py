"""可控 RGB 矩形資料；每個索引有自己的 seed，不依賴取樣順序。"""

import torch
from torch.utils.data import Dataset


class ShapeDataset(Dataset):
    """回傳 CHW float32 [0,1] 與 pixel xyxy 框；紅=0，藍=1。

    物件中心在不同 4×4 cell；本資料刻意避開第一版 head 無法表示的碰撞。
    真實資料的碰撞仍會由 build_targets 明確報錯。
    """

    def __init__(self, n=8, size=64, seed=0, max_objects=2, allow_empty=True):
        if n < 0 or size < 16 or not 0 <= max_objects <= 16:
            raise ValueError("n>=0、size>=16、0<=max_objects<=16")
        if not allow_empty and max_objects == 0:
            raise ValueError("allow_empty=False 時 max_objects 必須至少為 1")
        self.n, self.size, self.seed = n, size, seed
        self.max_objects, self.allow_empty = max_objects, allow_empty

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        if not 0 <= index < self.n:
            raise IndexError(index)
        rng = torch.Generator().manual_seed(self.seed + index * 1009)
        image = torch.rand((3, self.size, self.size), generator=rng) * .04
        low = 0 if self.allow_empty else 1
        count = int(torch.randint(low, self.max_objects + 1, (), generator=rng))
        cells = torch.randperm(16, generator=rng)[:count]
        boxes, labels = [], []
        for cell in cells.tolist():
            row, col = divmod(cell, 4)
            # 框位於 cell 內，尺寸與位置仍有變化，便於獨立資料驗證。
            left, right = col * self.size // 4, (col + 1) * self.size // 4
            top, bottom = row * self.size // 4, (row + 1) * self.size // 4
            cw, ch = right - left, bottom - top
            width = int(torch.randint(max(2, cw // 2), cw, (), generator=rng))
            height = int(torch.randint(max(2, ch // 2), ch, (), generator=rng))
            x1 = left + int(torch.randint(0, cw - width + 1, (), generator=rng))
            y1 = top + int(torch.randint(0, ch - height + 1, (), generator=rng))
            label = int(torch.randint(0, 2, (), generator=rng))
            color = torch.tensor([.95, .10, .10] if label == 0 else [.10, .10, .95])
            image[:, y1:y1 + height, x1:x1 + width] = color[:, None, None]
            boxes.append([x1, y1, x1 + width, y1 + height])
            labels.append(label)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        return image, target


def collate(batch):
    """圖像堆疊；變動長度標註保留為 list，空框不需要 padding。"""
    if not batch:
        raise ValueError("batch 不可為空")
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)

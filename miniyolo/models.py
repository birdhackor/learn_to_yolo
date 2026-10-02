"""小型 CNN、shortcut 與每格一框的 detector，所有參數由隨機初始化學習。"""

import torch
from torch import nn


def _conv(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
        nn.ReLU(),
    )


class TinyCNN(nn.Module):
    """VGG 風格分類器：局部卷積後用全域平均池化得到 [B,C]。"""

    def __init__(self, num_classes=2, width=8):
        super().__init__()
        if num_classes < 1 or width < 1:
            raise ValueError("num_classes 與 width 必須為正整數")
        self.features = nn.Sequential(
            _conv(3, width), nn.MaxPool2d(2),
            _conv(width, width * 2), nn.MaxPool2d(2),
            _conv(width * 2, width * 4), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(width * 4, num_classes)

    def forward(self, images):
        return self.classifier(self.features(images).flatten(1))


class ResidualBlock(nn.Module):
    """同 shape 使用 identity；跨 stage 使用學得的 1×1 projection。"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        if min(in_channels, out_channels, stride) < 1:
            raise ValueError("channels 與 stride 必須為正整數")
        self.branch = nn.Sequential(
            _conv(in_channels, out_channels, stride),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.shortcut = (
            nn.Identity() if in_channels == out_channels and stride == 1
            else nn.Conv2d(in_channels, out_channels, 1, stride=stride)
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.branch(x) + self.shortcut(x))


class GridDetector(nn.Module):
    """[B,S,S,5+C] logits；每格依序 tx,ty,tw,th,obj,class logits。

    xy 的 sigmoid 是格內 offset；wh 的 sigmoid 是相對整張圖的尺寸。
    此為教學簡化，不宣稱重現任何完整 YOLO 版本。
    """

    def __init__(self, num_classes=2, grid_size=4, width=8):
        super().__init__()
        if min(num_classes, grid_size, width) < 1:
            raise ValueError("num_classes、grid_size、width 必須為正整數")
        self.num_classes, self.grid_size = num_classes, grid_size
        self.backbone = nn.Sequential(
            _conv(3, width, 2), _conv(width, width * 2, 2),
            _conv(width * 2, width * 4, 2),
            nn.AdaptiveAvgPool2d((grid_size, grid_size)),
            _conv(width * 4, width * 4),
        )
        self.head = nn.Conv2d(width * 4, 5 + num_classes, 1)
        # 合理的初始尺寸與稀疏物件先驗；不是已訓練權重。
        with torch.no_grad():
            self.head.bias[2:4].fill_(-1.8)
            self.head.bias[4].fill_(-2.0)

    def forward(self, images):
        return self.head(self.backbone(images)).permute(0, 2, 3, 1).contiguous()

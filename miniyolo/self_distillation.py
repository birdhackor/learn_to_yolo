"""從零訓練的 CPU DINO 機制示範，並非官方 DINO 或 DINOv2 checkpoint。

來源：Caron et al. (2021), https://arxiv.org/abs/2104.14294 ，以及
https://github.com/facebookresearch/dino/blob/main/main_dino.py 的 DINOLoss。
保留跨 view CE、teacher stop-gradient/EMA、舊 center 與不同 temperature。
簡化為兩個 global views、普通 MLP head、固定 LR/temperature/momentum；
省略 multi-crop、weight normalization、官方 warmup/cosine schedules。
"""

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
import random

import torch
from torch import nn
import torch.nn.functional as F

from .vision_transformer import TinyViT


@dataclass
class DINOConfig:
    output_dim: int = 16
    hidden_dim: int = 64
    bottleneck_dim: int = 16
    batch_size: int = 24
    learning_rate: float = .0005
    student_temperature: float = .15
    teacher_temperature: float = .08
    teacher_momentum: float = .95
    center_momentum: float = .9
    global_scale: tuple = (.65, 1.0)

    def __post_init__(self):
        if min(self.output_dim, self.hidden_dim, self.bottleneck_dim, self.batch_size) < 1:
            raise ValueError("dimensions/batch_size 必須為正")
        if min(self.learning_rate, self.student_temperature, self.teacher_temperature) <= 0:
            raise ValueError("learning rate/temperatures 必須為正")
        if not 0 <= self.teacher_momentum < 1 or not 0 <= self.center_momentum < 1:
            raise ValueError("EMA momentum 必須在 [0,1)")
        if len(self.global_scale) != 2 or not 0 < self.global_scale[0] <= self.global_scale[1] <= 1:
            raise ValueError("crop area scale 必須在 (0,1]")


def random_view(images, generator, scale=(.65, 1.), output_size=32,
                flip=True, photometric=True):
    """只讀 images；回傳 [B,3,S,S] 與原圖 crop xyxy pixel [B,4]。

    scale 是 crop 面積佔方形原圖的比例。resize 改變座標；原框不能直接
    當 view 的 bbox。SSL 不接收分類標籤或 bbox，也不保證 crop 留住物件。
    """
    if images.ndim != 4 or images.shape[1] != 3 or images.shape[2] != images.shape[3] or len(images) == 0:
        raise ValueError("images 必須為非空方形 RGB [B,3,H,H]")
    if not 0 < scale[0] <= scale[1] <= 1 or output_size < 1:
        raise ValueError("scale/output_size 不合法")
    size, views, crops = images.shape[-1], [], []
    for image in images:
        area = float(torch.empty(()).uniform_(*scale, generator=generator))
        side = min(size, max(1, round(size * area ** .5)))
        left, top = torch.randint(size - side + 1, (2,), generator=generator).tolist()
        view = F.interpolate(image[:, top:top + side, left:left + side][None],
                             (output_size, output_size), mode="bilinear", align_corners=False)[0]
        if flip and bool(torch.rand((), generator=generator) < .5):
            view = view.flip(-1)
        if photometric:
            # 相同 gain 套用三通道：此受控色彩資料不做 hue/gray augmentation。
            gain = .8 + .4 * float(torch.rand((), generator=generator))
            view = (view * gain).clamp(0, 1)
        views.append(view)
        crops.append([left, top, left + side, top + side])
    return torch.stack(views), torch.tensor(crops, dtype=torch.long)


class DINOModel(nn.Module):
    """CLS [B,D] -> MLP -> L2-normalized bottleneck -> K 個自監督 logits。

    K 個分佈位置並非 red/blue 類別；backbone 原分類 head 不參與 SSL。
    """

    def __init__(self, backbone, config):
        super().__init__()
        self.backbone = backbone
        self.projector = nn.Sequential(nn.Linear(backbone.embed_dim, config.hidden_dim),
                                       nn.GELU(), nn.Linear(config.hidden_dim, config.bottleneck_dim))
        self.last_layer = nn.Linear(config.bottleneck_dim, config.output_dim, bias=False)
        # unused supervised head 不進 optimizer；仍保存於 backbone checkpoint。
        self.backbone.head.requires_grad_(False)

    def forward(self, images):
        features = self.backbone.forward_features(images)["cls"]
        return self.last_layer(F.normalize(self.projector(features), dim=-1))


class DINOLoss(nn.Module):
    """Teacher view i 對 student view j，排除 i==j；teacher 只給停止梯度的目標。

    forward 使用呼叫當下的舊 center，update_center 必須在這次 loss 之後。
    center 是未除 temperature 的 teacher logits EMA，不是 features 均值。
    """

    def __init__(self, output_dim, student_temperature=.15, teacher_temperature=.08,
                 center_momentum=.9):
        super().__init__()
        if output_dim < 1 or student_temperature <= 0 or teacher_temperature <= 0:
            raise ValueError("output_dim/temperatures 必須為正")
        if not 0 <= center_momentum < 1:
            raise ValueError("center_momentum 必須在 [0,1)")
        self.student_temperature = student_temperature
        self.teacher_temperature = teacher_temperature
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, output_dim))

    def teacher_probabilities(self, logits):
        return ((logits.detach() - self.center) / self.teacher_temperature).softmax(-1)

    def forward(self, student_outputs, teacher_outputs):
        if len(student_outputs) != len(teacher_outputs) or len(student_outputs) < 2:
            raise ValueError("這個簡化版必須有相同數量的至少兩個 views")
        expected = student_outputs[0].shape
        if (len(expected) != 2 or expected[0] == 0 or expected[1] != self.center.shape[-1]
                or any(output.shape != expected for output in (*student_outputs, *teacher_outputs))):
            raise ValueError("每個 view 的 logits 都必須是相同非空 [B,K]")
        student_logs = [(output / self.student_temperature).log_softmax(-1)
                        for output in student_outputs]
        teachers = [self.teacher_probabilities(output) for output in teacher_outputs]
        terms = [-(teacher * student_logs[j]).sum(-1).mean()
                 for i, teacher in enumerate(teachers) for j in range(len(student_logs)) if i != j]
        return torch.stack(terms).mean()

    @torch.no_grad()
    def update_center(self, teacher_outputs):
        batch_center = torch.cat(teacher_outputs).mean(0, keepdim=True)
        self.center.mul_(self.center_momentum).add_(batch_center, alpha=1 - self.center_momentum)


@torch.no_grad()
def update_teacher(student, teacher, momentum):
    """梯度只更新 student；teacher <- m*teacher + (1-m)*updated student。"""
    if not 0 <= momentum < 1:
        raise ValueError("momentum 必須在 [0,1)")
    for source, target in zip(student.parameters(), teacher.parameters(), strict=True):
        target.mul_(momentum).add_(source, alpha=1 - momentum)
    # tiny ViT 無 running-stat buffer；若增加 buffer，仍明確複製非參數狀態。
    for source, target in zip(student.buffers(), teacher.buffers(), strict=True):
        target.copy_(source)


class DINOTrainer:
    """固定 CPU 訓練與可精確續跑 checkpoint；資料集仍由呼叫者明確提供。"""

    def __init__(self, seed=7, config=None, backbone_config=None):
        self.seed, self.config = seed, config or DINOConfig()
        self.backbone_config = backbone_config or dict(image_size=32, patch_size=8,
            embed_dim=32, depth=2, num_heads=4, mlp_ratio=2, num_classes=2, dropout=0.)
        # fork_rng 使初始化不偷偷改變呼叫者的隨機狀態。
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            backbone = TinyViT(**self.backbone_config)
            self.initial_backbone_state = copy.deepcopy(backbone.state_dict())
            self.student = DINOModel(backbone, self.config)
        self.teacher = copy.deepcopy(self.student).requires_grad_(False).eval()
        self.loss = DINOLoss(self.config.output_dim, self.config.student_temperature,
                             self.config.teacher_temperature, self.config.center_momentum)
        self.optimizer = torch.optim.AdamW([p for p in self.student.parameters() if p.requires_grad],
                                           lr=self.config.learning_rate, weight_decay=.01)
        self.generator = torch.Generator().manual_seed(seed + 1000)
        self.step = 0

    def next_views(self, images):
        """只取訓練圖片與本地 RNG；不接受 label/bbox。"""
        indices = torch.randint(len(images), (self.config.batch_size,), generator=self.generator)
        batch = images[indices]
        views = [random_view(batch, self.generator, self.config.global_scale,
                             self.backbone_config["image_size"])[0] for _ in range(2)]
        return views

    def peek_next_views(self, images):
        state = self.generator.get_state()
        views = self.next_views(images)
        self.generator.set_state(state)
        return views

    def train_step(self, images):
        self.student.train()
        self.teacher.eval()
        views = self.next_views(images)
        student_outputs = [self.student(view) for view in views]
        with torch.no_grad():
            teacher_outputs = [self.teacher(view) for view in views]
            probabilities = torch.cat([self.loss.teacher_probabilities(output) for output in teacher_outputs])
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1).mean()
        value = self.loss(student_outputs, teacher_outputs)  # 使用 old center
        if not torch.isfinite(value):
            raise RuntimeError("SSL loss 非有限值")
        self.optimizer.zero_grad(set_to_none=True)
        value.backward()
        grads = [p.grad for p in self.student.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(grad).all() for grad in grads):
            raise RuntimeError("student gradients 缺失或非有限值")
        grad_norm = torch.nn.utils.clip_grad_norm_(self.student.parameters(), 3.)
        self.optimizer.step()
        update_teacher(self.student, self.teacher, self.config.teacher_momentum)
        self.loss.update_center(teacher_outputs)  # loss 已算完，才更新給下一步
        self.step += 1
        return {"step": self.step, "loss": float(value.detach()),
                "teacher_entropy": float(entropy), "student_grad_norm": float(grad_norm)}

    def train_steps(self, images, steps=160):
        if steps < 1:
            raise ValueError("steps 必須為正")
        return [self.train_step(images) for _ in range(steps)]

    def save_checkpoint(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format_version": 1, "student": self.student.state_dict(),
                    "teacher": self.teacher.state_dict(), "optimizer": self.optimizer.state_dict(),
                    "loss": self.loss.state_dict(), "step": self.step, "seed": self.seed,
                    "config": asdict(self.config), "backbone_config": self.backbone_config,
                    "initial_backbone": self.initial_backbone_state,
                    "view_rng": self.generator.get_state(), "torch_rng": torch.get_rng_state(),
                    "python_rng": random.getstate()}, path)

    @classmethod
    def load_checkpoint(cls, path):
        # 此函式只讀本專案剛寫出的可信本地 checkpoint，包含 Python RNG tuple。
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("format_version") != 1:
            raise ValueError("不支援的 DINO checkpoint version")
        trainer = cls(state["seed"], DINOConfig(**state["config"]), state["backbone_config"])
        trainer.student.load_state_dict(state["student"])
        trainer.teacher.load_state_dict(state["teacher"])
        trainer.optimizer.load_state_dict(state["optimizer"])
        trainer.loss.load_state_dict(state["loss"])
        trainer.step = state["step"]
        trainer.initial_backbone_state = state["initial_backbone"]
        trainer.generator.set_state(state["view_rng"])
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        return trainer

    def random_baseline(self):
        backbone = TinyViT(**self.backbone_config)
        backbone.load_state_dict(self.initial_backbone_state)
        return backbone


@torch.no_grad()
def extract_features(backbone, images, kind="cls", batch_size=32):
    """凍結評測：CLS [N,D] 或 patches [N,P,D]；不更新 backbone。"""
    if kind not in ("cls", "patches") or batch_size < 1 or len(images) == 0:
        raise ValueError("kind/batch_size/images 不合法")
    backbone.eval()
    return torch.cat([backbone.forward_features(batch)[kind] for batch in images.split(batch_size)])


def patch_feature_map(patches, image_size=32, patch_size=8):
    """row-major [B,N,D] -> [B,D,H/P,W/P]；網格位置仍對應原圖 patch。"""
    if patch_size < 1 or image_size < 1:
        raise ValueError("image_size/patch_size 必須為正")
    side = image_size // patch_size
    if (image_size % patch_size
            or patches.ndim != 3 or patches.shape[1] != side * side):
        raise ValueError("patch 數量必須符合可整除的方形圖尺寸")
    return patches.transpose(1, 2).reshape(len(patches), patches.shape[-1], side, side)


@torch.no_grad()
def feature_diagnostics(features, probabilities=None):
    """std 跨圖片計算、pair cosine 排除同圖；單項指標都不能證明用途。"""
    if features.ndim != 2 or len(features) < 2:
        raise ValueError("features 必須是至少兩張圖的 [N,D]")
    normalized = F.normalize(features, dim=-1)
    similarities = normalized @ normalized.T
    n = len(features)
    result = {"feature_std": float(features.std(0, unbiased=False).mean()),
              "normalized_feature_std": float(normalized.std(0, unbiased=False).mean()),
              "mean_pair_cosine": float((similarities.sum() - similarities.diagonal().sum()) / (n * (n - 1)))}
    if probabilities is not None:
        result["mean_output_entropy"] = float(-(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1).mean())
        marginal = probabilities.mean(0)
        result["marginal_output_entropy"] = float(-(marginal * marginal.clamp_min(1e-12).log()).sum())
    return result


@torch.no_grad()
def nearest_neighbor_accuracy(train_features, train_labels, query_features, query_labels):
    """L2 normalized CLS cosine 1-NN；query 不放進 reference bank。"""
    indices = (F.normalize(query_features, dim=-1) @ F.normalize(train_features, dim=-1).T).argmax(-1)
    predictions = train_labels[indices]
    return float((predictions == query_labels).float().mean()), indices


def linear_probe(train_features, train_labels, val_features, val_labels,
                 test_features, test_labels, steps=120, seed=900):
    """同初始化、固定步數的線性分類頭；labels 首次在這裡參與 optimizer。

    train 統計標準化，validation 僅報分數，不選超參數；test 不進訓練。
    """
    if steps < 1:
        raise ValueError("probe steps 必須為正")
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        head = nn.Linear(train_features.shape[1], 2)
    mean = train_features.mean(0)
    std = train_features.std(0, unbiased=False).clamp_min(.01)
    transformed = [(features.detach() - mean) / std
                   for features in (train_features, val_features, test_features)]
    optimizer = torch.optim.Adam(head.parameters(), lr=.02)
    for _ in range(steps):
        value = F.cross_entropy(head(transformed[0]), train_labels)
        optimizer.zero_grad(set_to_none=True)
        value.backward()
        optimizer.step()
    with torch.no_grad():
        scores = [float((head(features).argmax(-1) == labels).float().mean())
                  for features, labels in zip(transformed, (train_labels, val_labels, test_labels))]
    return {"train_accuracy": scores[0], "val_accuracy": scores[1], "test_accuracy": scores[2],
            "steps": steps, "head_seed": seed, "final_loss": float(value.detach())}

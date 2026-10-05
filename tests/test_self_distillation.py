"""DINO 的目標、EMA、center、隨機 view 與中途續跑之行為契約。"""
import copy
import random

import pytest
import torch
from torch import nn

from miniyolo.self_distillation import (DINOConfig, DINOLoss, DINOTrainer, extract_features,
                                       feature_diagnostics, patch_feature_map, random_view, update_teacher)
from miniyolo.vision_data import ColorRectangles


def test_loss_pairs_different_views_and_stops_teacher_gradient():
    loss = DINOLoss(3, student_temperature=.2, teacher_temperature=.1)
    student = [torch.tensor([[.2, .1, -.3]], requires_grad=True),
               torch.tensor([[-.2, .4, .1]], requires_grad=True)]
    teacher = [torch.tensor([[.5, -.2, .1]], requires_grad=True),
               torch.tensor([[.1, .3, -.1]], requires_grad=True)]
    expected = sum(-(teacher[i].detach().div(.1).softmax(-1) * student[1-i].div(.2).log_softmax(-1)).sum()
                   for i in range(2)) / 2
    actual = loss(student, teacher)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in student)
    assert all(value.grad is None for value in teacher)


def test_center_uses_old_value_for_loss_then_teacher_logit_mean_ema():
    loss = DINOLoss(2, student_temperature=1., teacher_temperature=1., center_momentum=.75)
    loss.center.copy_(torch.tensor([[.3, -.1]]))
    old_center = loss.center.clone()
    student = [torch.tensor([[1., 0.], [0., 1.]]) for _ in range(2)]
    teacher = [torch.tensor([[2., 0.], [1., 3.]]), torch.tensor([[4., 2.], [0., 2.]])]
    expected = sum(-(((teacher[i] - old_center).softmax(-1)) * student[1-i].log_softmax(-1)).sum(-1).mean()
                   for i in range(2)) / 2
    torch.testing.assert_close(loss(student, teacher), expected)
    assert torch.equal(loss.center, old_center)
    loss.update_center(teacher)
    torch.testing.assert_close(loss.center, .75 * old_center + .25 * torch.cat(teacher).mean(0, keepdim=True))


def test_teacher_ema_is_from_updated_student():
    student = nn.Linear(2, 2)
    teacher = copy.deepcopy(student).requires_grad_(False)
    before = [p.clone() for p in teacher.parameters()]
    with torch.no_grad():
        for parameter in student.parameters():
            parameter.add_(2.)
    update_teacher(student, teacher, .8)
    for old, source, target in zip(before, student.parameters(), teacher.parameters()):
        torch.testing.assert_close(target, .8 * old + .2 * source)
        assert target.grad is None and not target.requires_grad


def test_views_reproducible_without_labels_and_crop_inside_source():
    images = ColorRectangles(samples=3).images
    first, crops = random_view(images, torch.Generator().manual_seed(5))
    second, same_crops = random_view(images, torch.Generator().manual_seed(5))
    assert first.shape == images.shape and torch.equal(first, second) and torch.equal(crops, same_crops)
    assert ((crops >= 0) & (crops <= 32)).all()
    assert (crops[:, 2:] > crops[:, :2]).all()
    with pytest.raises(ValueError):
        random_view(torch.empty(0, 3, 32, 32), torch.Generator())


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        for a, b in zip(left, right, strict=True):
            assert_tree_equal(a, b)
    else:
        assert left == right


def test_midpoint_checkpoint_restores_all_state_and_next_random_views(tmp_path):
    torch.set_num_threads(2)
    images = ColorRectangles(samples=12).images
    trainer = DINOTrainer(config=DINOConfig(batch_size=4))
    before = trainer.student.backbone.patch_embed.projection.weight.detach().clone()
    trainer.train_steps(images, 3)
    path = tmp_path / "midpoint.pt"
    torch.manual_seed(123)
    random.seed(123)
    trainer.save_checkpoint(path)
    expected_torch_rng = torch.rand(2)
    expected_python_rng = random.random()
    views = trainer.peek_next_views(images)
    continuation = trainer.train_steps(images, 3)
    restored = DINOTrainer.load_checkpoint(path)
    assert torch.equal(torch.rand(2), expected_torch_rng)
    assert random.random() == expected_python_rng
    assert all(torch.equal(a, b) for a, b in zip(views, restored.peek_next_views(images)))
    assert continuation == restored.train_steps(images, 3)
    for component in ("student", "teacher", "optimizer", "loss"):
        assert_tree_equal(getattr(trainer, component).state_dict(), getattr(restored, component).state_dict())
    assert trainer.step == restored.step == 6
    assert torch.equal(trainer.generator.get_state(), restored.generator.get_state())
    assert not torch.equal(before, trainer.student.backbone.patch_embed.projection.weight)
    assert all(parameter.grad is None for parameter in restored.teacher.parameters())


def test_patch_feature_map_preserves_row_major_positions_and_freeze():
    patches = torch.arange(2 * 16 * 3).reshape(2, 16, 3)
    grid = patch_feature_map(patches)
    assert grid.shape == (2, 3, 4, 4)
    assert torch.equal(grid[:, :, 1, 2], patches[:, 6])
    with pytest.raises(ValueError):
        patch_feature_map(patches[:, :15])
    with pytest.raises(ValueError):
        patch_feature_map(patches, patch_size=0)
    backbone = DINOTrainer().teacher.backbone
    features = extract_features(backbone, ColorRectangles(samples=2).images)
    assert features.shape == (2, 32) and not features.requires_grad


def test_constant_peaked_outputs_can_have_low_loss_and_useless_features():
    logits = torch.tensor([[10., -10., -10.]]).repeat(4, 1)
    loss = DINOLoss(3, student_temperature=1., teacher_temperature=1.)
    assert float(loss([logits, logits], [logits, logits])) < 1e-5
    diagnostics = feature_diagnostics(torch.ones(4, 8), logits.softmax(-1))
    assert diagnostics["feature_std"] == 0
    assert diagnostics["mean_pair_cosine"] == pytest.approx(1.)


@pytest.mark.parametrize("kwargs", [{"batch_size": 0}, {"teacher_temperature": 0},
                                    {"center_momentum": 1}, {"global_scale": (1., .5)}])
def test_invalid_training_config_rejected(kwargs):
    with pytest.raises(ValueError):
        DINOConfig(**kwargs)

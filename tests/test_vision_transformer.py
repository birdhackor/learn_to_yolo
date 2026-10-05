"""ViT真正的空間順序、排列/位置關係、梯度及含RNG續訓。"""

import importlib.util
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from miniyolo.checkpoint import load_checkpoint
from miniyolo.vision_data import ColorRectangles, make_color_splits
from miniyolo.vision_transformer import PatchEmbedding, SelfAttention, TinyViT, patchify


def test_patch_order_and_projection_match_explicit_linear():
    image = torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(1, 3, 8, 8)
    patches = patchify(image, 4)
    assert patches.shape == (1, 4, 48)
    assert torch.equal(patches[0, 0], image[0, :, :4, :4].reshape(-1))
    assert torch.equal(patches[0, 1], image[0, :, :4, 4:].reshape(-1))
    assert torch.equal(patches[0, 2], image[0, :, 4:, :4].reshape(-1))
    assert torch.equal(patches[0, 3], image[0, :, 4:, 4:].reshape(-1))
    projection = PatchEmbedding(image_size=8, patch_size=4, embed_dim=7)
    explicit = F.linear(patches, projection.projection.weight.flatten(1), projection.projection.bias)
    assert torch.allclose(projection(image), explicit, atol=1e-4)


def test_cls_and_position_information_and_permutation_relationship():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    model = TinyViT(dropout=.2).eval()
    # 每個patch都有不同內容，避免交換相同背景無法觀察差異。
    image = torch.empty(1, 3, 32, 32)
    for index in range(16):
        row, col = divmod(index, 4)
        image[:, :, row * 8:(row + 1) * 8, col * 8:(col + 1) * 8] = index / 16
    with torch.no_grad():
        contents = model.embed_tokens(image, use_position=False)
        positioned = model.embed_tokens(image)
        assert torch.equal(contents[:, 0], model.cls_token[:, 0])
        assert torch.allclose(positioned - contents, model.pos_embed, atol=1e-7)
        permutation = torch.cat((torch.tensor([0]), torch.arange(16, 0, -1)))
        encoded = model.encode_tokens(contents)
        swapped = model.encode_tokens(contents[:, permutation])
        assert torch.allclose(swapped, encoded[:, permutation], atol=2e-6)
        assert torch.allclose(swapped[:, 0], encoded[:, 0], atol=2e-6)
        # 內容與位置一起重排依然等變；只換內容、保留位置才改變CLS。
        positioned_swap = model.encode_tokens(positioned[:, permutation])
        assert torch.allclose(positioned_swap, model.encode_tokens(positioned)[:, permutation], atol=2e-6)
        fixed_position_swap = model.encode_tokens(contents[:, permutation] + model.pos_embed)
        assert not torch.allclose(fixed_position_swap[:, 0], model.encode_tokens(positioned)[:, 0], atol=1e-5)
        features = model.forward_features(image)
        assert features["cls"].shape == (1, 32)
        assert features["patches"].shape == (1, 16, 32)
        assert torch.equal(features["tokens"][:, 0], features["cls"])
        assert torch.equal(features["tokens"][:, 1:], features["patches"])


def test_attention_rows_and_full_vit_gradient_flow():
    torch.manual_seed(7)
    attention = SelfAttention(embed_dim=8, num_heads=2).eval()
    tokens = torch.randn(2, 5, 8)
    output, weights = attention(tokens, return_attention=True)
    assert output.shape == tokens.shape
    assert weights.shape == (2, 2, 5, 5)
    assert torch.allclose(weights.sum(-1), torch.ones(2, 2, 5), atol=1e-6)
    assert torch.all(weights >= 0)
    model = TinyViT()
    data = ColorRectangles(samples=4)
    loss = F.cross_entropy(model(data.images), data.labels)
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert model.patch_embed.projection.weight.grad.norm() > 0
    assert model.blocks[0].attention.qkv.weight.grad.norm() > 0
    assert model.cls_token.grad.norm() > 0
    assert model.pos_embed.grad.norm() > 0


def test_empty_batch_preserves_shapes_but_empty_spatial_image_is_invalid():
    image = torch.zeros(0, 3, 32, 32)
    assert patchify(image, 8).shape == (0, 16, 192)
    model = TinyViT().eval()
    assert model(image).shape == (0, 2)
    assert model.forward_features(image)["patches"].shape == (0, 16, 32)
    with pytest.raises(ValueError):
        patchify(torch.zeros(1, 3, 0, 32), 8)


def test_data_is_balanced_seeded_local_and_separate():
    torch.manual_seed(87)
    before = torch.get_rng_state().clone()
    data = ColorRectangles(samples=16, seed=101)
    assert torch.equal(before, torch.get_rng_state())
    repeated = ColorRectangles(samples=16, seed=101)
    assert torch.equal(data.images, repeated.images)
    assert torch.equal(data.labels, repeated.labels)
    assert torch.equal(data.boxes, repeated.boxes)
    assert torch.bincount(data.labels).tolist() == [8, 8]
    assert data.images.shape == (16, 3, 32, 32)
    assert data.images.min() >= 0 and data.images.max() <= 1
    for image, label, box in zip(data.images, data.labels, data.boxes):
        left, top, right, bottom = box.tolist()
        rectangle = image[:, top:bottom, left:right].mean((1, 2))
        assert int(rectangle.argmax()) == (0 if int(label) == 0 else 2)
    splits = make_color_splits(train_samples=16, val_samples=16, test_samples=16)
    assert [splits[key].seed for key in ("train", "val", "test")] == [101, 202, 303]
    assert not torch.equal(splits["train"].images, splits["val"].images)
    assert not torch.equal(splits["val"].images, splits["test"].images)


@pytest.mark.parametrize("function", [lambda: patchify(torch.zeros(1, 3, 31, 32), 8),
                                      lambda: patchify(torch.zeros(1, 1, 32, 32), 8),
                                      lambda: TinyViT(embed_dim=31, num_heads=4),
                                      lambda: TinyViT()(torch.zeros(1, 3, 64, 64)),
                                      lambda: SelfAttention()(torch.zeros(1, 0, 32)),
                                      lambda: make_color_splits(seeds=(1, 1, 2))])
def test_invalid_shapes_and_configuration(function):
    with pytest.raises(ValueError):
        function()


def load_training_case():
    path = Path(__file__).resolve().parents[1] / "lesson_cases/21-training.py"
    spec = importlib.util.spec_from_file_location("vit_training_case", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_optimizer_resume_restores_model_moments_steps_and_rng(tmp_path):
    lesson = load_training_case()
    result = lesson.run_experiment(steps=8, checkpoint_step=3, batch_size=16, output=tmp_path)
    assert result["all_gradients_finite"]
    assert result["gradient_norm_min_max"][0] > 0
    assert all(result["weights_changed"].values())
    assert result["checkpoint"]["restored_optimizer_step_min_max"] == [3, 3]
    assert result["checkpoint"]["restored_learning_rate"] == .003
    assert result["resume"]["optimizer_step_min_max"] == [8, 8]
    assert result["resume"]["model_max_abs_error"] == 0
    assert result["resume"]["optimizer_max_abs_error"] == 0
    assert result["resume"]["loss_history_matches"]
    assert result["resume"]["rng_states_match"]
    model = TinyViT(**lesson.MODEL_CONFIG)
    optimizer = torch.optim.AdamW(model.parameters())
    state = load_checkpoint(tmp_path / "checkpoint.pt", model, optimizer)
    expected_draws = (random.random(), np.random.random(), torch.rand(4))
    # 攪亂三種RNG，再從同一checkpoint恢復；下一個隨機數應完全相同。
    random.seed(993)
    np.random.seed(993)
    torch.manual_seed(993)
    load_checkpoint(tmp_path / "checkpoint.pt", model, optimizer)
    actual_draws = (random.random(), np.random.random(), torch.rand(4))
    assert expected_draws[:2] == actual_draws[:2]
    assert torch.equal(expected_draws[2], actual_draws[2])
    assert state["steps_completed"] == 3

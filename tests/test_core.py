"""用可手算答案驗證核心契約，包含空圖、重複框與一次實際參數更新。"""

import math

import pytest
import torch

from miniyolo import ShapeDataset, collate, build_targets, grid_loss, decode_grid, evaluate_ap
from miniyolo.geometry import (
    xyxy_to_cxcywh, cxcywh_to_xyxy, box_iou, nms, letterbox, undo_letterbox,
)
from miniyolo.models import TinyCNN, ResidualBlock, GridDetector


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(7)
    torch.set_num_threads(2)


def target(boxes=(), labels=()):
    return {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.long)}


def prediction(boxes=(), scores=(), labels=()):
    return {**target(boxes, labels), "scores": torch.tensor(scores, dtype=torch.float32)}


def test_half_open_geometry_and_round_trip():
    boxes = torch.tensor([[0., 0., 2., 2.], [1., 1., 3., 3.]])
    encoded = xyxy_to_cxcywh(boxes)
    torch.testing.assert_close(encoded, torch.tensor([[1., 1., 2., 2.], [2., 2., 2., 2.]]))
    torch.testing.assert_close(cxcywh_to_xyxy(encoded), boxes)
    torch.testing.assert_close(box_iou(boxes, boxes), torch.tensor([[1., 1/7], [1/7, 1.]]))
    assert box_iou(boxes[:1], torch.tensor([[2., 0., 4., 2.]])).item() == 0


def test_degenerate_and_empty_geometry_has_no_nan():
    empty, zero = torch.empty(0, 4), torch.zeros(1, 4)
    assert box_iou(empty, zero).shape == (0, 1)
    assert box_iou(zero, empty).shape == (1, 0)
    assert box_iou(zero, zero).item() == 0
    assert xyxy_to_cxcywh(empty).shape == (0, 4)
    assert cxcywh_to_xyxy(empty).shape == (0, 4)


def test_nms_prefers_score_and_ties_are_stable():
    boxes = torch.tensor([[0., 0., 10., 10.], [0., 0., 10., 10.], [20., 20., 30., 30.]])
    assert nms(boxes, torch.tensor([.8, .9, .7])).tolist() == [1, 2]
    assert nms(boxes, torch.tensor([.9, .9, .7])).tolist() == [0, 2]
    assert nms(boxes, torch.tensor([.8, .9, .7]), 1).tolist() == [1, 0, 2]
    assert nms(torch.empty(0, 4), torch.empty(0)).dtype == torch.long


def test_letterbox_uses_actual_rounded_scales_and_restores_boxes():
    image = torch.rand(3, 37, 83)
    boxes = torch.tensor([[0., 0., 83., 37.], [9., 4., 34., 25.]])
    resized, transformed, meta = letterbox(image, boxes, size=64)
    assert resized.shape == (3, 64, 64)
    assert meta["original_size"] == (37, 83)
    assert meta["resized_size"] == (29, 64)
    assert meta["padding"] == (0, 17)
    assert meta["scale_xy"][0] != meta["scale_xy"][1]
    torch.testing.assert_close(undo_letterbox(transformed, meta), boxes, atol=1e-5, rtol=1e-5)
    assert letterbox(image, torch.empty(0, 4))[1].shape == (0, 4)


def test_dataset_contract_determinism_and_cell_capacity():
    dataset = ShapeDataset(n=24, size=63, seed=7, max_objects=5, allow_empty=True)
    for index in range(len(dataset)):
        image, annotation = dataset[index]
        assert image.dtype == torch.float32 and image.shape == (3, 63, 63)
        assert 0 <= image.min() <= image.max() <= 1
        torch.testing.assert_close(dataset[index][0], image)
        assert annotation["boxes"].ndim == 2 and annotation["boxes"].shape[1] == 4
        assert annotation["labels"].dtype == torch.long
        # 若生成器把两个中心落在同格，build_targets 会报错。
        encoded = build_targets([annotation], image_size=63)
        assert int(encoded["positive"].sum()) == len(annotation["boxes"])
    empty = ShapeDataset(n=1, max_objects=0)[0][1]
    assert empty["boxes"].shape == (0, 4) and empty["labels"].shape == (0,)
    images, annotations = collate([dataset[0], dataset[1]])
    assert images.shape == (2, 3, 63, 63) and isinstance(annotations, list)


def test_targets_cell_offsets_and_full_image_dimensions():
    encoded = build_targets([target([[8, 12, 24, 28], [48, 48, 64, 64]], [1, 0]), target()])
    assert encoded["box"].shape == (2, 4, 4, 4)
    torch.testing.assert_close(encoded["box"][0, 1, 1], torch.tensor([0., .25, .25, .25]))
    torch.testing.assert_close(encoded["box"][0, 3, 3], torch.tensor([.5, .5, .25, .25]))
    assert encoded["class_ids"][0, 1, 1].item() == 1
    assert encoded["positive"].dtype == torch.bool
    assert encoded["objectness"].sum().item() == 2
    assert not encoded["positive"][1].any()


def test_same_cell_collision_is_not_silently_lost():
    with pytest.raises(ValueError, match="same-cell collision"):
        build_targets([target([[1, 1, 9, 9], [2, 2, 12, 12]], [0, 1])])


@pytest.mark.parametrize("annotation", [
    target([[0, 0, 0, 2]], [0]),
    target([[-1, 0, 5, 5]], [0]),
    target([[0, 0, 65, 5]], [0]),
    target([[0, 0, 5, 5]], [2]),
    target([[float("nan"), 0, 5, 5]], [0]),
])
def test_invalid_annotations_are_rejected(annotation):
    with pytest.raises(ValueError):
        build_targets([annotation])


def test_loss_hand_calculation_and_positive_masks():
    encoded = build_targets([target([[8, 12, 24, 28]], [1])])
    logits = torch.zeros(1, 4, 4, 7, requires_grad=True)
    loss = grid_loss(logits, encoded)
    assert loss["box"].item() == pytest.approx(.109375)
    assert loss["objectness"].item() == pytest.approx(math.log(2))
    assert loss["classification"].item() == pytest.approx(math.log(2))
    assert loss["total"].item() == pytest.approx(5 * .109375 + 2 * math.log(2))
    loss["total"].backward()
    background = ~encoded["positive"]
    assert torch.count_nonzero(logits.grad[..., :4][background]) == 0
    assert torch.count_nonzero(logits.grad[..., 5:][background]) == 0
    assert logits.grad[..., 4][background].min() > 0
    assert logits.grad[0, 1, 1, 4] < 0


def test_empty_image_losses_remain_connected_and_background_learns():
    encoded = build_targets([target(), target()])
    logits = torch.zeros(2, 4, 4, 7, requires_grad=True)
    losses = grid_loss(logits, encoded)
    for name in ("box", "classification"):
        assert losses[name].requires_grad and losses[name].item() == 0
        gradient, = torch.autograd.grad(losses[name], logits, retain_graph=True)
        assert torch.isfinite(gradient).all() and gradient.count_nonzero() == 0
    losses["total"].backward()
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad[..., 4] > 0).all()
    assert logits.grad[..., :4].count_nonzero() == 0
    assert logits.grad[..., 5:].count_nonzero() == 0


def test_extreme_finite_logits_do_not_create_nan_loss():
    logits = torch.full((1, 4, 4, 7), 1000., requires_grad=True)
    logits.data[..., 6] = -1000
    losses = grid_loss(logits, build_targets([target([[1, 1, 9, 9]], [1])]))
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    assert torch.isfinite(logits.grad).all()
    with pytest.raises(ValueError, match="非有限"):
        grid_loss(torch.full_like(logits, float("nan")), build_targets([target()]))


def test_decode_score_formula_and_threshold():
    logits = torch.zeros(1, 1, 1, 7)
    result = decode_grid(logits, image_size=64, score_threshold=.25)[0]
    torch.testing.assert_close(result["boxes"], torch.tensor([[16., 16., 48., 48.]]))
    torch.testing.assert_close(result["scores"], torch.tensor([.25]))
    assert result["labels"].tolist() == [0]
    empty = decode_grid(logits, score_threshold=.25001)[0]
    assert empty["boxes"].shape == (0, 4)
    assert empty["scores"].shape == empty["labels"].shape == (0,)


def test_decode_nms_is_class_wise():
    # 相鄰格都預測幾乎全圖，IoU 接近 1；不同類不互相抑制。
    logits = torch.full((1, 2, 2, 7), -20.)
    logits[..., 2:4] = 20
    logits[0, 0, 0, :2] = 10
    logits[0, 0, 1, :2] = torch.tensor([-10., 10.])
    logits[0, 0, :2, 4] = 10
    logits[0, 0, 0, 5] = 10
    logits[0, 0, 1, 6] = 10
    different_class = decode_grid(logits)[0]
    assert len(different_class["boxes"]) == 2
    assert sorted(different_class["labels"].tolist()) == [0, 1]
    logits[0, 0, 1, 5:] = torch.tensor([10., -20.])
    same_class = decode_grid(logits)[0]
    assert len(same_class["boxes"]) == 1


def test_ap_duplicate_prediction_matches_gt_once():
    result = evaluate_ap(
        [prediction([[0, 0, 10, 10], [0, 0, 10, 10]], [.9, .8], [0, 0])],
        [target([[0, 0, 10, 10]], [0])],
    )
    assert result["ap_per_class"] == {0: 1., 1: None}
    assert result["map"] == 1 and result["precision"] == .5 and result["recall"] == 1


def test_ap_sorts_across_images_and_interpolates_all_points():
    # 全域順序 FP(.95),TP(.90),duplicate FP(.80),TP(.70)。
    predictions = [
        prediction([[0, 0, 10, 10], [0, 0, 10, 10]], [.90, .80], [0, 0]),
        prediction([[20, 20, 30, 30], [0, 0, 10, 10]], [.95, .70], [0, 0]),
    ]
    targets = [target([[0, 0, 10, 10]], [0]), target([[0, 0, 10, 10]], [0])]
    result = evaluate_ap(predictions, targets)
    assert result["map"] == pytest.approx(.5)
    assert result["precision"] == .5 and result["recall"] == 1


def test_ap_no_gt_class_is_excluded_but_false_positives_still_count():
    result = evaluate_ap(
        [prediction([[0, 0, 10, 10], [20, 20, 30, 30]], [.9, .99], [0, 1])],
        [target([[0, 0, 10, 10]], [0])],
    )
    assert result["ap_per_class"] == {0: 1., 1: None}
    assert result["map"] == 1 and result["precision"] == .5
    empty_gt = evaluate_ap([prediction([[0, 0, 10, 10]], [.9], [0])], [target()])
    assert empty_gt == {"ap_per_class": {0: None, 1: None}, "map": None, "precision": 0., "recall": 0.}
    missing = evaluate_ap([prediction()], [target([[0, 0, 10, 10]], [0])])
    assert missing["map"] == missing["precision"] == missing["recall"] == 0


def test_full_forward_backward_optimizer_and_inference_step():
    dataset = ShapeDataset(n=4, seed=7, allow_empty=False)
    images, annotations = collate([dataset[index] for index in range(4)])
    model = GridDetector(width=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    before = model.head.weight.detach().clone()
    logits = model(images)
    assert logits.shape == (4, 4, 4, 7)
    loss = grid_loss(logits, build_targets(annotations))
    optimizer.zero_grad()
    loss["total"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.head.weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before, model.head.weight)
    model.eval()
    detections = decode_grid(model(images), score_threshold=0)
    assert len(detections) == 4
    result = evaluate_ap(detections, annotations)
    assert all(math.isfinite(result[key]) for key in ("map", "precision", "recall"))


def test_classifier_and_both_shortcuts_train_from_scratch():
    images = torch.rand(2, 3, 32, 32)
    model = TinyCNN(width=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    logits = model(images)
    assert logits.shape == (2, 2)
    before = model.classifier.weight.detach().clone()
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    optimizer.step()
    assert not torch.equal(before, model.classifier.weight)
    identity = ResidualBlock(4, 4)
    projection = ResidualBlock(4, 8, stride=2)
    x = torch.rand(2, 4, 8, 8, requires_grad=True)
    assert identity(x).shape == x.shape
    assert projection(x).shape == (2, 8, 4, 4)
    (identity(x).square().mean() + projection(x).square().mean()).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

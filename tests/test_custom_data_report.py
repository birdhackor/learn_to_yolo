"""Per-image TP/FP matching, figure captions and axis steps of scripts/run_custom_data_learning.py,
on hand-made boxes whose IoU can be checked by hand."""

import json

import pytest
import torch

from miniyolo.metrics import evaluate_ap
from scripts.run_custom_data_learning import (caption_lines, create_fixture, fixture_foreground_boxes,
                                              match_image, nice_step, render_svg, tensor_rows)

CLASSES = ["red_rectangle", "blue_rectangle", "yellow_rectangle"]
NAMES = ["紅", "藍", "黃"]
IOU = .5


def target(boxes=(), labels=()):
    return {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.long)}


def prediction(boxes=(), scores=(), labels=()):
    return {**target(boxes, labels), "scores": torch.tensor(scores, dtype=torch.float32)}


# One image each: its predictions, its GT and the caption lines the figure prints under it.
CASES = {
    "empty input": (prediction(), target(), ["沒有 GT，也沒有預測框"]),
    "image without GT": (prediction([[4, 4, 20, 20]], [.8], [0]), target(), ["紅 0.80：FP（圖中沒有 GT）"]),
    # The lower score is listed first: matching goes by score, and the GT is matched only once.
    "two boxes on one GT": (prediction([[11, 10, 31, 30], [10, 10, 30, 30]], [.6, .9], [0, 0]),
                            target([[10, 10, 30, 30]], [0]),
                            ["紅 0.60：FP（同類 GT 已被較高分的框配對）", "紅 0.90：TP，IoU 1.000"]),
    "two objects, one found": (prediction([[0, 0, 20, 20], [40, 40, 60, 60]], [.9, .4], [0, 1]),
                               target([[0, 0, 20, 20], [30, 0, 50, 20]], [0, 1]),
                               ["紅 0.90：TP，IoU 1.000", "藍 0.40：FP，IoU 0.000 < 0.5", "GT 藍：漏檢（FN）"]),
    # The box takes the GT it overlaps best, not the first one listed.
    "two GTs of one class": (prediction([[30, 30, 50, 50]], [.8], [0]),
                             target([[0, 0, 20, 20], [30, 30, 50, 50]], [0, 0]),
                             ["紅 0.80：TP，IoU 1.000", "GT 紅：漏檢（FN）"]),
    "same class, IoU below threshold": (prediction([[10, 0, 30, 20]], [.5], [0]), target([[0, 0, 20, 20]], [0]),
                                        ["紅 0.50：FP，IoU 0.333 < 0.5", "GT 紅：漏檢（FN）"]),
    # IoU exactly .5 is enough for a TP.
    "same class, IoU at threshold": (prediction([[0, 0, 20, 10]], [.7], [0]), target([[0, 0, 20, 20]], [0]),
                                     ["紅 0.70：TP，IoU 0.500"]),
    "other class, no overlap": (prediction([[40, 40, 60, 60]], [.7], [0]), target([[0, 0, 16, 16]], [1]),
                                ["紅 0.70：FP（沒有同類 GT，最大 IoU 0.000）", "GT 藍：漏檢（FN）"]),
    "other class, IoU below threshold": (prediction([[0, 0, 20, 9]], [.7], [0]), target([[0, 0, 20, 20]], [1]),
                                         ["紅 0.70：FP（沒有同類 GT，最大 IoU 0.450）", "GT 藍：漏檢（FN）"]),
    # It is also enough for a box to count as covering another class's object.
    "other class, IoU at threshold": (prediction([[0, 0, 20, 10]], [.7], [2]), target([[0, 0, 20, 20]], [1]),
                                      ["黃 0.70：FP（類別錯，IoU 0.500）", "GT 藍：漏檢（FN）"]),
}
# Four far-away boxes and the GT they miss: five lines, more than a panel shows.
CROWDED = (prediction([[40, 40, 60, 60]] * 4, [.9, .8, .7, .6], [0] * 4), target([[0, 0, 10, 10]], [0]))


def example(pred, gt, index=0):
    """A validation_examples entry as run() records it."""
    return {"validation_index": index, "target": tensor_rows([gt])[0], "prediction": tensor_rows([pred])[0],
            "matches": match_image(pred, gt, IOU)}


def ap_true_positives(predictions, targets):
    """evaluate_ap's micro TP count, read back from both its precision and its recall."""
    result = evaluate_ap(predictions, targets, num_classes=len(CLASSES), iou_threshold=IOU)
    from_precision = round(result["precision"] * sum(len(p["labels"]) for p in predictions))
    assert from_precision == round(result["recall"] * sum(len(t["labels"]) for t in targets))
    return from_precision


def true_positives(pred, gt):
    return sum(row["match"] == "TP" for row in match_image(pred, gt, IOU)["predictions"])


@pytest.mark.parametrize("name", CASES)
def test_match_image_counts_the_same_true_positives_as_evaluate_ap(name):
    pred, gt, _ = CASES[name]
    found = true_positives(pred, gt)
    assert found == ap_true_positives([pred], [gt])
    assert len(match_image(pred, gt, IOU)["missed_gt_labels"]) == len(gt["labels"]) - found


def test_match_image_lets_the_higher_score_take_the_gt_first():
    pred, gt, _ = CASES["two boxes on one GT"]
    assert [row["match"] for row in match_image(pred, gt, IOU)["predictions"]] == ["FP", "TP"]


def test_match_image_agrees_with_evaluate_ap_over_all_images():
    predictions, targets = [case[0] for case in CASES.values()], [case[1] for case in CASES.values()]
    found = sum(true_positives(pred, gt) for pred, gt in zip(predictions, targets))
    assert found == ap_true_positives(predictions, targets) == 4


@pytest.mark.parametrize("name", CASES)
def test_caption_names_the_rule_that_decided_each_box(name):
    pred, gt, expected = CASES[name]
    assert [text for text, _ in caption_lines(example(pred, gt), NAMES, IOU)] == expected


def test_caption_colors_follow_tp_fp_fn():
    pred, gt, _ = CASES["two objects, one found"]
    assert [color for _, color in caption_lines(example(pred, gt), NAMES, IOU)] == ["#15803d", "#c2410c", "#b91c1c"]


def test_long_caption_keeps_three_lines_and_counts_the_rest():
    assert [text for text, _ in caption_lines(example(*CROWDED), NAMES, IOU)] == [
        "紅 0.90：FP，IoU 0.000 < 0.5", "紅 0.80：FP，IoU 0.000 < 0.5", "紅 0.70：FP，IoU 0.000 < 0.5",
        "……另有 2 項，見下方紀錄檔"]


def figure(tmp_path, examples, record):
    """render_svg's text for a two-step run whose validation figure shows these examples."""
    result = {"config": {"seed": 7, "batch_size": 8, "learning_rate": .01, "score_threshold": .1, "nms_iou": .5,
                         "eval_iou": IOU},
              "data": {"split_counts": {split: {"images": 1} for split in ("train", "validation", "test")}},
              "steps_completed": 2, "predeclared_protocol": {}, "fixture": True,
              "final_train": {"map": None}, "final_validation": {"map": None}, "final_test": {"map": None},
              "reload_checks": {"raw_predictions_exact": True},
              "loss_history": [{"total": 1.5}, {"total": .5}], "validation_examples": examples}
    path = tmp_path / "learning.svg"
    render_svg(path, result, torch.zeros(len(examples), 3, 64, 64), CLASSES, record)
    return path.read_text(encoding="utf-8")


def test_cut_caption_points_to_the_record_path_render_svg_receives(tmp_path):
    record = "artifacts/checks/curriculum/custom-data-learning.json"
    cut = figure(tmp_path, [example(*CROWDED)], record)
    assert "……另有 2 項，見下方紀錄檔" in cut
    assert f"列在紀錄檔 {record} 的 validation_examples" in cut
    assert "report.json" not in cut
    assert record not in figure(tmp_path, [example(*CASES["two objects, one found"][:2])], record)


@pytest.mark.parametrize("raw, step", [(.03, .05), (.15, .2), (.21, .25), (.25, .25), (.31, .5), (.372, .5),
                                       (1, 1), (3, 5), (7, 10), (10, 10), (12, 20)])
def test_nice_step_rounds_up_to_1_2_2_5_or_5_times_a_power_of_ten(raw, step):
    assert nice_step(raw) == pytest.approx(step)


def test_fixture_foreground_boxes_recover_every_annotated_box(tmp_path):
    records = json.loads(create_fixture(tmp_path).read_text())["images"]
    assert [fixture_foreground_boxes(tmp_path / r["path"]) for r in records] == [r["boxes"] for r in records]
    assert sum(not r["boxes"] for r in records) == 9  # three background-only images per split

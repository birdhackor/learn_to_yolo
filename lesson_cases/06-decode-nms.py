"""Artificial logits make decoding/filtering/NMS independently checkable."""
import torch


def iou(box, boxes):
    left = torch.maximum(box[:2], boxes[:, :2])
    right = torch.minimum(box[2:], boxes[:, 2:])
    intersection = (right - left).clamp(min=0).prod(1)
    area = (box[2:] - box[:2]).clamp(min=0).prod()
    areas = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0).prod(1)
    return intersection / (area + areas - intersection).clamp(min=1e-8)


def class_nms(boxes, scores, labels, threshold=0.5):
    keep = []
    for label in labels.unique():
        indices = (labels == label).nonzero().flatten()
        order = indices[scores[indices].argsort(descending=True, stable=True)]
        while order.numel():
            current = order[0]
            keep.append(current.item())
            remaining = order[1:]
            order = remaining[iou(boxes[current], boxes[remaining]) <= threshold]
    kept = torch.tensor(keep, dtype=torch.long)
    return kept[scores[kept].argsort(descending=True, stable=True)]


def decode(logits, score_threshold=0.25, image_size=64):
    grid = logits.shape[0]
    row, col = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    center = (torch.stack((col, row), -1) + logits[..., :2].sigmoid()) / grid * image_size
    size = logits[..., 2:4].sigmoid() * image_size
    boxes = torch.cat((center - size / 2, center + size / 2), -1)
    class_prob, labels = logits[..., 5:].softmax(-1).max(-1)
    scores = logits[..., 4].sigmoid() * class_prob
    selected = scores >= score_threshold
    return boxes[selected], scores[selected], labels[selected]


def fill(logits, row, col, xywh, obj, class_prob):
    logits[row, col, :4] = torch.logit(torch.tensor(xywh))
    logits[row, col, 4] = torch.logit(torch.tensor(obj))
    logits[row, col, 5:] = torch.tensor(class_prob).log()


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    logits = torch.zeros(4, 4, 7)
    logits[..., 4] = -12
    fill(logits, 1, 2, [.25, .75, .25, .125], .9, [.8, .2])
    fill(logits, 1, 1, [.95, .75, .25, .125], .8, [.8, .2])
    fill(logits, 3, 0, [.5, .5, .125, .125], .95, [.9, .1])
    boxes, scores, labels = decode(logits)
    assert len(boxes) == 3
    assert torch.allclose(boxes[1], torch.tensor([28., 24., 44., 32.]), atol=1e-5)
    assert torch.allclose(scores, torch.tensor([.64, .72, .855]), atol=1e-6)
    overlap = iou(boxes[1], boxes[:1]).item()
    assert abs(overlap - 7 / 13) < 1e-5
    kept = class_nms(boxes, scores, labels)
    assert kept.tolist() == [2, 1]
    print(f"candidate_boxes={boxes.tolist()}")
    print(f"scores={[round(s.item(), 3) for s in scores]}, duplicate_IoU={overlap:.4f}")
    print(f"NMS keep_indices={kept.tolist()}, kept_scores={[round(s.item(), 3) for s in scores[kept]]}")
    strict_boxes, strict_scores, _ = decode(logits, score_threshold=.75)
    assert len(strict_boxes) == 1 and torch.allclose(strict_boxes[0], boxes[2])
    print(f"threshold .75 keeps only artificial false positive, score={strict_scores.item():.3f}")
    # Independent exercises: preserve the baseline arrays, thresholds, and assertions.
    keep06 = class_nms(boxes, scores, labels, threshold=.6)
    assert keep06.tolist() == [2, 1, 0]
    print(f"exercise NMS .6 keeps original candidate indices={keep06.tolist()}")
    boxes70, scores70, labels70 = decode(logits, score_threshold=.70)
    assert len(boxes70) == 2 and labels70.tolist() == [0, 0]
    assert torch.allclose(scores70, torch.tensor([.72, .855]), atol=1e-6)
    print(f"exercise score .70 keeps scores={[round(s.item(), 3) for s in scores70]}")
    same_boxes = torch.stack((boxes[1], boxes[1]))
    class_keep = class_nms(same_boxes, torch.tensor([.9, .8]), torch.tensor([0, 1]))
    assert class_keep.tolist() == [0, 1]
    assert class_nms(torch.empty(0, 4), torch.empty(0), torch.empty(0, dtype=torch.long)).numel() == 0
    print("identical boxes of different classes both survive class-wise NMS; empty input passed")
    print("All logits are hand-constructed, not a trained detector.")


if __name__ == "__main__":
    main()

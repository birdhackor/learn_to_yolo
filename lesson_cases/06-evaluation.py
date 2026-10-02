"""Image-aware, class-aware, one-use GT matching and all-points interpolated AP."""
import torch


def iou(box, boxes):
    intersection = (torch.minimum(box[2:], boxes[:, 2:]) -
                    torch.maximum(box[:2], boxes[:, :2])).clamp(min=0).prod(1)
    area = (box[2:] - box[:2]).clamp(min=0).prod()
    areas = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0).prod(1)
    return intersection / (area + areas - intersection).clamp(min=1e-8)


def interpolated_ap(recall, precision):
    r = torch.cat((torch.tensor([0.]), recall, torch.tensor([1.])))
    p = torch.cat((torch.tensor([0.]), precision, torch.tensor([0.])))
    for i in range(len(p) - 2, -1, -1):
        p[i] = torch.maximum(p[i], p[i + 1])
    changes = (r[1:] != r[:-1]).nonzero().flatten()
    return ((r[changes + 1] - r[changes]) * p[changes + 1]).sum().item()


def evaluate(predictions, targets, num_classes=2, match_iou_threshold=.5):
    aps, records = [], {}
    total_tp = total_fp = total_gt = 0
    for cls in range(num_classes):
        gt_count = sum((t["labels"] == cls).sum().item() for t in targets)
        candidates = [(p["scores"][j].item(), b, j)
                      for b, p in enumerate(predictions)
                      for j in range(len(p["boxes"])) if p["labels"][j] == cls]
        candidates.sort(key=lambda entry: (-entry[0], entry[1], entry[2]))
        matched = [set() for _ in targets]
        flags = []
        for _, b, j in candidates:
            eligible = [k for k in range(len(targets[b]["boxes"]))
                        if targets[b]["labels"][k] == cls and k not in matched[b]]
            is_tp = False
            if eligible:
                overlaps = iou(predictions[b]["boxes"][j], targets[b]["boxes"][eligible])
                best = overlaps.argmax().item()
                if overlaps[best] >= match_iou_threshold:
                    matched[b].add(eligible[best])
                    is_tp = True
            flags.append(int(is_tp))
        flags_tensor = torch.tensor(flags, dtype=torch.float32)
        tp = flags_tensor.cumsum(0)
        precision = tp / torch.arange(1, len(flags) + 1)
        recall = tp / max(gt_count, 1)
        ap = interpolated_ap(recall, precision) if gt_count else None
        aps.append(ap)
        records[cls] = {"flags": flags, "precision": precision, "recall": recall,
                        "scores": [entry[0] for entry in candidates]}
        total_tp += int(flags_tensor.sum().item())
        total_fp += len(flags) - int(flags_tensor.sum().item())
        total_gt += gt_count
    valid_aps = [ap for ap in aps if ap is not None]
    return {"ap_per_class": aps, "map": sum(valid_aps) / len(valid_aps) if valid_aps else None,
            "precision": total_tp / max(total_tp + total_fp, 1),
            "recall": total_tp / max(total_gt, 1),
            "tp": total_tp, "fp": total_fp, "fn": total_gt - total_tp, "records": records}


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    targets = [
        {"boxes": torch.tensor([[0., 0., 10., 10.], [20., 0., 30., 10.]]), "labels": torch.tensor([0, 0])},
        {"boxes": torch.tensor([[0., 20., 10., 30.]]), "labels": torch.tensor([0])},
    ]
    predictions = [
        {"boxes": torch.tensor([[40., 40., 50., 50.], [0., 0., 10., 10.], [1., 0., 11., 10.]]),
         "scores": torch.tensor([.95, .9, .8]), "labels": torch.tensor([0, 0, 0])},
        {"boxes": torch.tensor([[0., 20., 10., 30.]]), "scores": torch.tensor([.7]), "labels": torch.tensor([0])},
    ]
    result = evaluate(predictions, targets)
    record = result["records"][0]
    assert record["flags"] == [0, 1, 0, 1]
    assert (result["tp"], result["fp"], result["fn"]) == (2, 2, 1)
    assert abs(result["precision"] - .5) < 1e-6 and abs(result["recall"] - 2 / 3) < 1e-6
    assert abs(result["map"] - 1 / 3) < 1e-6 and result["ap_per_class"][1] is None
    for rank, (score, flag, p, r) in enumerate(zip(record["scores"], record["flags"],
                                                 record["precision"], record["recall"]), 1):
        print(f"rank={rank}, score={score:.2f}, {'TP' if flag else 'FP'}, "
              f"precision={p.item():.4f}, recall={r.item():.4f}")
    print(f"TP={result['tp']}, FP={result['fp']}, FN={result['fn']}, "
          f"AP50={result['map']:.6f}, ap_per_class={result['ap_per_class']}")
    truncated = [{key: values[p["scores"] >= .85] for key, values in p.items()} for p in predictions]
    short_result = evaluate(truncated, targets)
    assert abs(short_result["map"] - 1 / 6) < 1e-6
    print(f"candidate threshold .85: AP50={short_result['map']:.6f}, recall={short_result['recall']:.4f}")
    # Remove a known artificial false positive, without changing GT.
    cleaned = [{"boxes": predictions[0]["boxes"][1:], "scores": predictions[0]["scores"][1:],
                "labels": predictions[0]["labels"][1:]}, predictions[1]]
    clean_result = evaluate(cleaned, targets)
    assert abs(clean_result["map"] - 5 / 9) < 1e-6
    print(f"remove known high-score FP: AP50={clean_result['map']:.6f}")
    empty = {"boxes": torch.empty(0, 4), "scores": torch.empty(0), "labels": torch.empty(0, dtype=torch.long)}
    no_predictions = evaluate([empty, empty], targets)
    assert no_predictions["map"] == 0 and no_predictions["fn"] == 3
    # An empty image can still contribute false positives.
    empty_target = {"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long)}
    background = evaluate([predictions[1]], [empty_target])
    assert background["fp"] == 1 and background["map"] is None
    print("no predictions => AP=0 when GT exists; no-GT class AP=None; background FP counted")
    print("Artificial scoring exercise; not measured detector performance.")


if __name__ == "__main__":
    main()

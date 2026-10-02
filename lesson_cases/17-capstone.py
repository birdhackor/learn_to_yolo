"""Two paired detector runs: change box-loss weight, decide on validation, test once."""
import base64
import io
import json
import time
from pathlib import Path
import torch
from PIL import Image, ImageDraw
from miniyolo.data import ShapeDataset, collate
from miniyolo.models import GridDetector
from miniyolo.targets import build_targets
from miniyolo.losses import grid_loss
from miniyolo.inference import decode_grid
from miniyolo.metrics import evaluate_ap
from miniyolo.geometry import box_iou


def batch(seed, n):
    data = ShapeDataset(n=n, size=64, seed=seed, max_objects=2, allow_empty=True)
    return collate([data[i] for i in range(n)])


def fit(images, targets, box_weight, steps=160):
    torch.manual_seed(7)
    model = GridDetector(width=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    encoded = build_targets(targets)
    first = None
    start = time.perf_counter()
    for step in range(steps):
        ids = torch.tensor([(step * 8 + i) % len(images) for i in range(8)])
        target = {key: value[ids] for key, value in encoded.items()}
        optimizer.zero_grad()
        losses = grid_loss(model(images[ids]), target)
        loss = box_weight * losses['box'] + losses['objectness'] + losses['classification']
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        if first is None:
            first = float(loss.detach())
    elapsed = time.perf_counter() - start
    assert float(loss.detach()) < first
    return model.eval(), elapsed


@torch.no_grad()
def evaluate(model, images, targets):
    predictions = decode_grid(model(images), score_threshold=.05, nms_iou=.5)
    return evaluate_ap(predictions, targets, num_classes=2, iou_threshold=.5), predictions


def diagnostics(predictions, targets):
    localized, near_miss, absent = 0, 0, 0
    # GT coverage diagnostic, not AP matching: several GT may inspect the same prediction.
    for pred, target in zip(predictions, targets):
        for box, label in zip(target['boxes'], target['labels']):
            candidates = pred['boxes'][pred['labels'] == label]
            best = float(box_iou(box[None], candidates).max()) if len(candidates) else 0.
            if best >= .5:
                localized += 1
            elif best >= .1:
                near_miss += 1
            else:
                absent += 1
    return {'covered_iou50': localized, 'iou10_to_50': near_miss, 'below_iou10': absent}


def error_diagnostics(predictions, targets):
    """At score .05, count wrong-class GT coverage and explain every unmatched prediction."""
    classification_cases, false_positives = [], []
    counts = {'background': 0, 'localization': 0, 'wrong_class': 0, 'duplicate': 0}
    true_positives = 0
    for image_id, (pred, target) in enumerate(zip(predictions, targets)):
        overlaps = box_iou(pred['boxes'], target['boxes'])
        for gt_id, label in enumerate(target['labels']):
            same_class = pred['labels'] == label
            best_same = float(overlaps[same_class, gt_id].max()) if same_class.any() else 0.
            if len(pred['boxes']):
                best_any, prediction_id = overlaps[:, gt_id].max(0)
                if best_same < .5 and float(best_any) >= .5 and pred['labels'][prediction_id] != label:
                    classification_cases.append({'image': image_id, 'gt': gt_id, 'prediction': int(prediction_id),
                                                 'gt_class': int(label), 'predicted_class': int(pred['labels'][prediction_id]),
                                                 'best_any_iou': float(best_any), 'best_same_class_iou': best_same})
        used = torch.zeros(len(target['boxes']), dtype=torch.bool)
        # Descending score, then original order: one prediction and one GT per match.
        order = sorted(range(len(pred['boxes'])), key=lambda i: -float(pred['scores'][i]))
        for prediction_id in order:
            same_class = target['labels'] == pred['labels'][prediction_id]
            available = torch.where(same_class & ~used)[0]
            if len(available):
                best, position = overlaps[prediction_id, available].max(0)
                if float(best) >= .5:
                    used[available[position]] = True
                    true_positives += 1
                    continue
            best_any = float(overlaps[prediction_id].max()) if len(target['boxes']) else 0.
            best_same = float(overlaps[prediction_id, same_class].max()) if same_class.any() else 0.
            if best_any < .1:
                kind = 'background'
            elif best_any < .5:
                kind = 'localization'
            elif best_same >= .5:
                kind = 'duplicate'
            else:
                kind = 'wrong_class'
            counts[kind] += 1
            false_positives.append({'image': image_id, 'prediction': prediction_id, 'kind': kind,
                                   'class': int(pred['labels'][prediction_id]), 'score': float(pred['scores'][prediction_id]),
                                   'best_any_iou': best_any, 'best_same_class_iou': best_same})
    assert true_positives + len(false_positives) == sum(len(p['boxes']) for p in predictions)
    return {'score_threshold': .05, 'matching_iou': .5, 'background_iou_below': .1,
            'wrong_class_gt_count': len(classification_cases), 'wrong_class_gt_cases': classification_cases,
            'true_positives': true_positives, 'false_positive_counts': counts, 'false_positive_cases': false_positives}


def draw(image, target, prediction, threshold=.25):
    canvas = Image.fromarray((image.permute(1, 2, 0).numpy() * 255).round().astype('uint8')).resize((192, 192))
    pen = ImageDraw.Draw(canvas)
    for box in target['boxes']:
        pen.rectangle(tuple(float(v) * 3 for v in box), outline='#22c55e', width=2)
    for box, score, label in zip(prediction['boxes'], prediction['scores'], prediction['labels']):
        if score >= threshold:
            pen.rectangle(tuple(float(v) * 3 for v in box), outline='#f59e0b', width=2)
            pen.text((max(0, float(box[0]) * 3), max(0, float(box[1]) * 3)), f'{int(label)}:{float(score):.2f}', fill='white')
    return canvas


def save_panel(images, targets, baseline, changed, baseline_weight, changed_weight):
    # Select hardest baseline examples, then show the exact same images for both runs.
    ranks = []
    for i, (pred, target) in enumerate(zip(baseline, targets)):
        stats = diagnostics([pred], [target])
        ranks.append((stats['iou10_to_50'] + stats['below_iou10'], i))
    chosen = [i for _, i in sorted(ranks, reverse=True)[:4]]
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 860 810" role="img" aria-labelledby="t d">',
             f'<title id="t">Capstone actual validation examples: box weights {baseline_weight} and {changed_weight}</title>',
             f'<desc id="d">Same four validation images. Top: baseline weight{baseline_weight}; second row: weight{changed_weight}. Green is GT, orange is prediction with score at least .25. Evaluation uses .05. A baseline unmatched prediction is highlighted red below at the evaluation threshold.</desc>',
             '<rect width="860" height="810" fill="#0f172a"/>',
             '<g fill="white" font-family="sans-serif" font-size="20">',
             f'<text x="20" y="28">Baseline: box weight {baseline_weight} | green GT, orange prediction</text>',
             f'<text x="20" y="275">One change: box weight {changed_weight} | display score ≥ .25</text>']
    for row, predictions in enumerate((baseline, changed)):
        for col, i in enumerate(chosen):
            buffer = io.BytesIO()
            draw(images[i], targets[i], predictions[i]).save(buffer, format='PNG')
            data = base64.b64encode(buffer.getvalue()).decode()
            x, y = 20 + col * 210, 42 + row * 247
            parts.append(f'<image x="{x}" y="{y}" width="192" height="192" href="data:image/png;base64,{data}"/>')
            parts.append(f'<text x="{x}" y="{y+217}">validation #{i}</text>')
    errors = error_diagnostics(baseline, targets)
    cases = errors['false_positive_cases']
    if cases:
        example = next((case for case in cases if case['kind'] in ('background', 'wrong_class')), cases[0])
        i, prediction_id = example['image'], example['prediction']
        canvas = draw(images[i], targets[i], baseline[i], threshold=.05)
        pen = ImageDraw.Draw(canvas)
        pen.rectangle(tuple(float(v)*3 for v in baseline[i]['boxes'][prediction_id]), outline='#ef4444', width=3)
        buffer = io.BytesIO()
        canvas.save(buffer, format='PNG')
        data = base64.b64encode(buffer.getvalue()).decode()
        parts.extend(['<text x="20" y="540">Inspect an actual baseline FP at evaluation score .05</text>',
                      f'<image x="20" y="556" width="192" height="192" href="data:image/png;base64,{data}"/>',
                      f'<text x="235" y="596">validation #{i}: red = {example["kind"]} FP</text>',
                      f'<text x="235" y="636">class {example["class"]}, score {example["score"]:.3f}</text>',
                      f'<text x="235" y="676">best IoU with any GT: {example["best_any_iou"]:.3f}</text>',
                      '<text x="235" y="716">Orange shows all candidates with score ≥ .05.</text>'])
    parts.extend(['</g>', '<text x="20" y="790" fill="white" font-family="sans-serif" font-size="18">Actual CPU run; tiny synthetic split. AP is reported separately at score .05.</text>', '</svg>'])
    output = Path('docs/assets/diagrams/17-capstone.svg')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(parts))


@torch.no_grad()
def end_to_end_ms(model, image, target, iterations=12):
    rgb = (image.permute(1, 2, 0).numpy() * 255).round().astype('uint8')
    samples = []
    for i in range(iterations + 3):
        start = time.perf_counter()
        tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255
        pred = decode_grid(model(tensor[None]), score_threshold=.05)[0]
        draw(tensor, target, pred)
        if i >= 3:
            samples.append((time.perf_counter() - start) * 1000)
    return float(torch.tensor(samples).median())


def main(baseline_weight=5, changed_weight=10):
    torch.manual_seed(7)
    torch.set_num_threads(2)
    train_x, train_y = batch(1100, 32)
    val_x, val_y = batch(2200, 16)
    test_x, test_y = batch(3300, 16)
    baseline, base_seconds = fit(train_x, train_y, baseline_weight)
    base_metrics, base_preds = evaluate(baseline, val_x, val_y)
    changed, change_seconds = fit(train_x, train_y, changed_weight)
    change_metrics, change_preds = evaluate(changed, val_x, val_y)
    assert base_metrics['map'] > .05, 'This capstone must learn something; inspect the training pipeline.'
    # Pre-declared .01 validation-AP margin; never choose using test performance.
    keep = change_metrics['map'] >= base_metrics['map'] + .01
    chosen = changed if keep else baseline
    test_metrics, _ = evaluate(chosen, test_x, test_y)
    milliseconds = end_to_end_ms(chosen, val_x[0], val_y[0])
    save_panel(val_x, val_y, base_preds, change_preds, baseline_weight, changed_weight)
    report = {'data_seeds': [1100, 2200, 3300], 'train_samples': 32, 'validation_samples': 16, 'test_samples': 16,
              'steps_per_run': 160, 'box_weights': [baseline_weight, changed_weight], 'score_threshold': .05, 'display_threshold': .25,
              'eval_iou': .5, 'baseline_validation': base_metrics, 'changed_validation': change_metrics,
              'baseline_coverage': diagnostics(base_preds, val_y), 'changed_coverage': diagnostics(change_preds, val_y),
              'baseline_errors': error_diagnostics(base_preds, val_y), 'changed_errors': error_diagnostics(change_preds, val_y),
              'keep_change': keep, 'chosen_test': test_metrics, 'parameters': sum(p.numel() for p in chosen.parameters()),
              'train_seconds': [base_seconds, change_seconds], 'chosen_end_to_end_median_ms': milliseconds,
              'timing_scope': 'uint8 RGB -> tensor -> model -> decode/NMS -> drawing; CPU, batch1, threads2; no file I/O',
              'limits': 'single initialization, tiny synthetic rectangles; no real-image or multi-seed evidence'}
    output = Path('artifacts/lesson-17')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    print('actual validation panel: docs/assets/diagrams/17-capstone.svg')


if __name__ == '__main__':
    main()

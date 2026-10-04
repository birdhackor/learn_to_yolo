"""Render the website's two 160-step figures from the tracked record artifacts/checks/grid-learning.json.

    python scripts/render_learning_evidence.py

Reads only tracked inputs: the record's loss_history and validation_examples. The validation images are
regenerated from ShapeDataset with the recorded seed and size (and checked against the recorded ground
truth), so no training run directory is needed. Predictions are marked TP/FP, and GTs no prediction matched
FN, by the matching rule of miniyolo.metrics.evaluate_ap; each image's TP and FN counts are checked against
evaluate_ap itself. Writes grid-learning-curve.svg and grid-learning-predictions.svg (the validation images
are embedded in it) to docs/assets/diagrams/.
"""
import argparse
import base64
import io
import json
import math
from pathlib import Path
import sys
from xml.sax.saxutils import escape

from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from miniyolo.data import ShapeDataset  # noqa: E402
from miniyolo.figures import FONT_STACK  # noqa: E402
from miniyolo.geometry import box_iou  # noqa: E402
from miniyolo.metrics import evaluate_ap  # noqa: E402
from miniyolo.train import LOSS_COLORS  # noqa: E402

INK, PAPER, WHITE = '#0f172a', '#f8fafc', '#ffffff'
GT_COLOR, PRED_COLOR, GOOD, BAD = '#22c55e', '#fb923c', '#86efac', '#fca5a5'
# Font size stays an attribute (inherited from <svg>), so a smaller size on one <text> still applies.
STYLE = f'<style>text{{font-family:{FONT_STACK};fill:{INK}}}</style>'


def svg_open(width, height, title, desc):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" font-size="16" role="img" '
            'aria-labelledby="title desc">', f'<title id="title">{escape(title)}</title>',
            f'<desc id="desc">{escape(desc)}</desc>', STYLE, f'<rect width="{width}" height="{height}" fill="{PAPER}"/>']


# ---- loss curve ---------------------------------------------------------------------------------------

def nice_step(span, intervals):
    """Smallest 1, 2, 2.5, 4 or 5 x 10^k that covers span in at most `intervals` steps."""
    raw = span / intervals
    magnitude = 10 ** math.floor(math.log10(raw))
    return next(f * magnitude for f in (1, 2, 2.5, 4, 5, 10) if f * magnitude >= raw * (1 - 1e-9))


def decimals(step):
    return next(d for d in range(7) if abs(round(step, d) - step) < 1e-9 * max(1, step))


def loss_history(report):
    """The recorded per-step losses; the legend's total = 5*box + objectness + classification must hold."""
    history = report.get('loss_history')
    if history is None:
        raise SystemExit('The record has no loss_history; re-record it with python -m miniyolo.train ... '
                         '--report artifacts/checks/grid-learning.json')
    assert [row['step'] for row in history] == list(range(1, report['steps_completed'] + 1)), \
        'loss_history must hold one row per optimizer step, numbered from 1'
    assert all(history[-1][key] == value for key, value in report['final_train_loss'].items()), \
        'the last loss_history row must be final_train_loss'
    for row in history:
        weighted = 5 * row['box'] + row['objectness'] + row['classification']
        assert abs(row['total'] - weighted) <= 1e-5 * max(1, row['total']), \
            f"step {row['step']}: total is not 5*box + objectness + classification with box unweighted"
    return history


def curve_svg(report):
    config, history = report['config'], loss_history(report)
    steps, device, batch = report['steps_completed'], config['device'].upper(), config['batch_size']
    left, right, top, bottom = 90, 860, 80, 380
    maximum = max(row[key] for row in history for key in LOSS_COLORS)
    y_step = nice_step(maximum, 5)
    y_top = y_step * math.ceil(maximum / y_step - 1e-9)
    x_step = max(1, round(nice_step(steps, 4)))

    def x_of(step):
        return left + (right - left) * (step - 1) / max(1, steps - 1)

    def y_of(value):
        return bottom - (bottom - top) * value / y_top

    svg = svg_open(900, 548, f'{steps} 次 {device} 參數更新的實測 loss 曲線',
                   '藍線 total、紅線 box（還沒乘 5 的原始值）、綠線 objectness、紫線 classification。'
                   f'第 k 點是第 k 次參數更新前、那一批 {batch} 張訓練圖的 loss。'
                   '這是合成矩形任務的單次實測，不是原版 YOLO 的效能。')
    svg += [f'<text x="24" y="32" font-weight="bold">{steps} 次 {device} 參數更新的實測 training loss</text>',
            f'<text x="24" y="58">固定合成資料：{config["samples"]} 張訓練圖、batch {batch}、'
            f'seed {config["train_seed"]}、learning rate {config["learning_rate"]:g}</text>']
    digits = decimals(y_step)
    for i in range(round(y_top / y_step) + 1):
        y = y_of(i * y_step)
        svg += [f'<path d="M{left} {y:.1f}H{right}" stroke="#e2e8f0"/>',
                f'<text x="{left - 10}" y="{y + 5:.1f}" text-anchor="end">{i * y_step:.{digits}f}</text>']
    middle = (top + bottom) // 2
    svg.append(f'<text x="30" y="{middle}" text-anchor="middle" transform="rotate(-90 30 {middle})">loss</text>')
    ticks = [1] + list(range(x_step, steps + 1, x_step))
    if steps - ticks[-1] >= x_step / 2:
        ticks.append(steps)
    for value in ticks:
        svg += [f'<path d="M{x_of(value):.1f} {bottom}v6" stroke="#64748b"/>',
                f'<text x="{x_of(value):.1f}" y="{bottom + 24}" text-anchor="middle">{value}</text>']
    for key, color in LOSS_COLORS.items():
        points = ' '.join(f'{x_of(row["step"]):.1f},{y_of(row[key]):.1f}' for row in history)
        svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5" '
                   'stroke-linejoin="round"/>')
    svg.append(f'<text x="{(left + right) // 2}" y="{bottom + 50}" text-anchor="middle">'
               '橫軸：參數更新次序；第 k 點是第 k 次更新前、那一批訓練圖的 loss</text>')
    legend = {'total': 'total＝5×box＋objectness＋classification', 'box': 'box（還沒乘 5 的原始值）',
              'objectness': 'objectness', 'classification': 'classification'}
    for i, (key, color) in enumerate(LOSS_COLORS.items()):
        x, y = 90 + (i % 2) * 450, 466 + (i // 2) * 28
        svg += [f'<path d="M{x} {y - 5}h28" stroke="{color}" stroke-width="3"/>',
                f'<text x="{x + 36}" y="{y}">{legend[key]}</text>']
    svg += ['<text x="24" y="534">這條曲線只反映訓練圖；模型在沒參與訓練的圖上表現如何，'
            '要看另外在 validation／test 圖上算的 mAP50。</text>', '</svg>']
    return '\n'.join(svg) + '\n'


# ---- validation predictions ---------------------------------------------------------------------------

def verdicts(entry, num_classes, iou_threshold):
    """TP/FP and IoU per recorded prediction of one image, by the matching rule of miniyolo.metrics.evaluate_ap.

    By descending score, a prediction takes the same-class GT with the largest IoU among those not yet
    matched in this image, and is a TP if that IoU reaches the threshold. Returns (is_tp, iou, or None
    when no same-class GT is left) per prediction, and the indices of the unmatched GTs (FN).
    """
    pred, gt = entry['prediction'], entry['target']
    boxes = torch.tensor(pred['boxes'], dtype=torch.float64).reshape(-1, 4)
    gt_boxes = torch.tensor(gt['boxes'], dtype=torch.float64).reshape(-1, 4)
    used, marks = [False] * len(gt_boxes), [None] * len(boxes)
    for j in sorted(range(len(boxes)), key=lambda j: -pred['scores'][j]):
        available = [g for g, label in enumerate(gt['labels']) if label == pred['labels'][j] and not used[g]]
        if not available:
            marks[j] = (False, None)
            continue
        ious = box_iou(boxes[j:j + 1], gt_boxes[available])[0]
        iou, best = float(ious.max()), available[int(ious.argmax())]
        if iou >= iou_threshold:
            used[best] = True
        marks[j] = (iou >= iou_threshold, iou)
    # evaluate_ap must count the same number of TPs on this image, or the marks do not show what was scored.
    metrics = evaluate_ap([{'boxes': boxes, 'scores': torch.tensor(pred['scores'], dtype=torch.float64),
                            'labels': torch.tensor(pred['labels'], dtype=torch.long)}],
                          [{'boxes': gt_boxes, 'labels': torch.tensor(gt['labels'], dtype=torch.long)}],
                          num_classes=num_classes, iou_threshold=iou_threshold)
    true_positives = sum(is_tp for is_tp, _ in marks)
    assert round(metrics['precision'] * len(boxes)) == true_positives == round(metrics['recall'] * len(gt_boxes)), \
        f"image {entry['index']}: the number of TP marks differs from the TPs miniyolo.metrics.evaluate_ap counts"
    # Each TP mark took one GT and no FP took any, so the FN marks must be exactly the other len(gt) - TP GTs,
    # as many as the FNs of evaluate_ap's recall = TP / (TP + FN).
    missed = [g for g, matched in enumerate(used) if not matched]
    assert sum(used) == true_positives and len(missed) == len(gt_boxes) - true_positives \
        and not any(used[g] for g in missed), \
        (f"image {entry['index']}: the FN marks are not the unmatched GTs, as many as the FNs "
         'miniyolo.metrics.evaluate_ap counts')
    return marks, missed


def iou_text(iou, threshold):
    """Two decimals, or more when rounding would put the shown value on the other side of the threshold."""
    for digits in range(2, 8):
        text = f'{iou:.{digits}f}'
        if (float(text) >= threshold) == (iou >= threshold):
            return text
    return repr(iou)


def text_width(text, size):
    """Width estimate for a tag background: CJK and full-width characters 1 em, others 0.58 em (wider
    than the average Latin glyph of common sans fonts, DejaVu Sans included)."""
    return sum(size if ord(ch) >= 0x2E80 else .58 * size for ch in text)


def overlaps(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def cluster(box, boxes):
    """box grown to cover every box that overlaps it, so a tag beside it covers none of them."""
    x1, y1, x2, y2 = box
    for other in boxes:
        if overlaps(box, other):
            x1, y1, x2, y2 = min(x1, other[0]), min(y1, other[1]), max(x2, other[2]), max(y2, other[3])
    return x1, y1, x2, y2


def place(anchor, size, area, taken, others):
    """Top-left corner for a w x h tag beside anchor: inside area, off the tags already placed, preferably
    off the other boxes; above or below first, then left or right."""
    (x1, y1, x2, y2), (w, h), (ax1, ay1, ax2, ay2) = anchor, size, area
    beside = [(x1, y1 - h - 2), (x2 - w, y1 - h - 2), (x1, y2 + 2), (x2 - w, y2 + 2)]
    sides = [(x1 - w - 2, y1), (x2 + 2, y1), (x1 - w - 2, y2 - h), (x2 + 2, y2 - h)]
    candidates = ([(min(max(x, ax1), ax2 - w), y) for x, y in beside]
                  + [(x, min(max(y, ay1), ay2 - h)) for x, y in sides])
    inside = [(x, y) for x, y in candidates if ax1 <= x and x + w <= ax2 and ay1 <= y and y + h <= ay2]
    free = [(x, y) for x, y in inside if not any(overlaps((x, y, x + w, y + h), t) for t in taken)]
    clear = [(x, y) for x, y in free if not any(overlaps((x, y, x + w, y + h), o) for o in others)]
    x, y = (clear or free or inside or candidates)[0]
    taken.append((x, y, x + w, y + h))
    return x, y


def tag(anchor, lines, area, taken, others, border, size=15):
    """Dark label beside anchor; lines are lists of (text, color, bold) runs."""
    pad, step = 5, size + 4
    width = max(sum(text_width(text, size) for text, _, _ in line) for line in lines) + 2 * pad
    height = len(lines) * step + 2 * pad - 4
    x, y = place(anchor, (width, height), area, taken, others)
    parts = [f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height}" rx="3" fill="{INK}" '
             f'fill-opacity="0.85" stroke="{border}" stroke-width="1.5"/>']
    for i, line in enumerate(lines):
        runs = ''
        for text, color, bold in line:
            weight = ' font-weight="bold"' if bold else ''
            runs += f'<tspan fill="{color}"{weight}>{escape(text)}</tspan>'
        parts.append(f'<text x="{x + pad:.1f}" y="{y + pad + size - 2 + i * step:.1f}" font-size="{size}">{runs}</text>')
    return parts


def png_bytes(image):
    pixels = (image.permute(1, 2, 0).numpy() * 255).round().astype('uint8')
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format='PNG')
    return buffer.getvalue()


def to_panel(box, x0, y0, scale):
    """Image-pixel xyxy box to figure coordinates of an image drawn at (x0, y0)."""
    x1, y1, x2, y2 = box
    return x0 + x1 * scale, y0 + y1 * scale, x0 + x2 * scale, y0 + y2 * scale


def box_svg(box, color, dash=''):
    x1, y1, x2, y2 = box
    return (f'<rect x="{x1:.2f}" y="{y1:.2f}" width="{x2 - x1:.2f}" height="{y2 - y1:.2f}" fill="none" '
            f'stroke="{color}" stroke-width="2.5"{dash}/>')


def predictions_svg(report):
    """Panel of the recorded validation examples; also writes each regenerated image as a PNG."""
    config, examples = report['config'], report['validation_examples']
    threshold, num_classes = config['eval_iou'], len(report['class_names'])
    validation = ShapeDataset(n=config['validation_samples'], size=config['image_size'], seed=config['validation_seed'])
    side = 320
    scale = side / config['image_size']
    top = 176
    height = top + (math.ceil(len(examples) / 2) - 1) * 380 + side + 82
    body, summary = [], []
    for i, entry in enumerate(examples):
        image, target = validation[entry['index']]
        pred, gt = entry['prediction'], entry['target']
        assert all(a >= b for a, b in zip(pred['scores'], pred['scores'][1:])), \
            f"image {entry['index']}: the legend numbers predictions by descending score"
        assert all(score >= config['score_threshold'] for score in pred['scores']), \
            f"image {entry['index']}: the legend says every drawn box scores at least the score threshold"
        # The regenerated image must be the one the recorded predictions were made on.
        assert (target['boxes'].tolist(), target['labels'].tolist()) == (gt['boxes'], gt['labels']), \
            (f"ShapeDataset(seed={config['validation_seed']}) image {entry['index']} differs from the recorded ground "
             'truth, so it is not the image the recorded predictions were made on')
        data = png_bytes(image)
        x0, y0 = 80 + (i % 2) * 420, top + (i // 2) * 380
        gt_rects = [to_panel(box, x0, y0, scale) for box in gt['boxes']]
        pred_rects = [to_panel(box, x0, y0, scale) for box in pred['boxes']]
        rects = gt_rects + pred_rects
        body += [f'<text x="{x0}" y="{y0 - 12}">圖片 {entry["index"]}：{len(gt_rects)} 個真值、{len(pred_rects)} 個預測</text>',
                 f'<image href="data:image/png;base64,{base64.b64encode(data).decode("ascii")}" x="{x0}" y="{y0}" '
                 f'width="{side}" height="{side}" style="image-rendering:pixelated"/>']
        body += [box_svg(r, GT_COLOR, ' stroke-dasharray="7 4"') for r in gt_rects]
        body += [box_svg(r, PRED_COLOR) for r in pred_rects]
        marks, missed = verdicts(entry, num_classes, threshold)
        taken, area = [], (x0, y0, x0 + side, y0 + side)
        for j, (label, score) in enumerate(zip(pred['labels'], pred['scores'])):
            is_tp, iou = marks[j]
            verdict = 'TP' if is_tp else 'FP'
            detail = f' IoU {iou_text(iou, threshold)}' if iou is not None else ' 沒有可配對的同類真值'
            lines = [[(f'#{j} class {label} score {score:.3f}', WHITE, False)],
                     [(verdict, GOOD if is_tp else BAD, True), (detail, WHITE, False)]]
            own = len(gt_rects) + j
            body += tag(cluster(rects[own], rects), lines, area, taken, rects[:own] + rects[own + 1:], PRED_COLOR)
            summary.append(f'圖片 {entry["index"]} 的 #{j}：class {label}，score {score:.3f}，{verdict}，{detail.strip()}')
        for g in missed:
            body += tag(cluster(rects[g], rects), [[('FN', BAD, True)]], area, taken, rects[:g] + rects[g + 1:], GT_COLOR)
            summary.append(f'圖片 {entry["index"]} 有一個 class {gt["labels"][g]} 的真值沒被配對到，是 FN')
    indices = [entry['index'] for entry in examples]
    which = f'前 {len(examples)} 張' if indices == list(range(len(examples))) else f'其中 {len(examples)} 張'
    steps, device = report['steps_completed'], config['device'].upper()
    svg = svg_open(900, height, f'{steps} 次 {device} 參數更新後，{len(examples)} 張獨立 validation 圖的真值與預測框',
                   '原始圖片以 PNG 呈現，綠色虛線是真值，橙色實線是模型預測；每個預測框旁標出編號、類別、score 與評估判定。'
                   + '；'.join(summary) + '。')
    svg += [f'<text x="24" y="32" font-weight="bold">{steps} 次 {device} 參數更新後，模型在沒參與訓練的 validation 圖上的預測</text>',
            f'<text x="24" y="62" font-size="15">綠色虛線＝真值（GT）；橙色實線＝模型預測：score ≥ {config["score_threshold"]:g}、'
            '經同類 NMS 後留下的全部框。</text>',
            '<text x="24" y="86" font-size="15">矩形填色是類別：紅＝class 0，藍＝class 1。</text>',
            '<text x="24" y="110" font-size="15">預測框的標籤：#k 是第 k 個預測（依 score 由高到低，從 0 編號），接著是預測類別與 score。</text>',
            f'<text x="24" y="134" font-size="15">TP／FP：和同類、還沒被配對的真值算最大 IoU，≥ {threshold:g} 是 TP，'
            '否則是 FP；FN：沒被任何預測配對到的真值。</text>']
    svg += body
    svg += [f'<text x="24" y="{height - 42}" font-size="15">mAP50 用全部 {config["validation_samples"]} 張 validation 圖計算，'
            f'這裡畫的是{which}；它們都沒參與參數更新。</text>',
            f'<text x="24" y="{height - 18}" font-size="15">這是紅／藍矩形合成任務的一次實驗（單一 seed、小資料集），'
            '不能代表真實照片的效果。</text>', '</svg>']
    return '\n'.join(svg) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--report', type=Path, default=ROOT / 'artifacts/checks/grid-learning.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/assets/diagrams')
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'grid-learning-curve.svg').write_text(curve_svg(report))
    (args.output / 'grid-learning-predictions.svg').write_text(predictions_svg(report))
    print(f'Rendered grid-learning-curve.svg and grid-learning-predictions.svg from {args.report} into {args.output}')


if __name__ == '__main__':
    main()

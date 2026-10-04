"""Exact tiny gated matching, crossings, a missed detection, and measured ID switches."""
from dataclasses import dataclass
from pathlib import Path
import torch


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return torch.zeros(len(a), len(b))
    inter = (torch.minimum(a[:, None, 2:], b[None, :, 2:])
             - torch.maximum(a[:, None, :2], b[None, :, :2])).clamp(min=0).prod(-1)
    return inter / ((a[:, 2:] - a[:, :2]).prod(-1)[:, None] + (b[:, 2:] - b[:, :2]).prod(-1)[None] - inter)


def exact_gated_matching(quality, threshold=.1):
    # Enumerate unmatched and matched options, maximize valid pair count then sum IoU.
    # This is a true optimum for this tiny case, not a greedy or Hungarian implementation.
    best_key, best_pairs = (-1, -1.), []
    def search(row, used, pairs, total):
        nonlocal best_key, best_pairs
        if row == quality.shape[0]:
            key = (len(pairs), total)
            if key > best_key:
                best_key, best_pairs = key, pairs.copy()
            return
        search(row + 1, used, pairs, total)
        for col in range(quality.shape[1]):
            value = float(quality[row, col])
            if col not in used and value >= threshold:
                search(row + 1, used | {col}, pairs + [(row, col)], total + value)
    search(0, set(), [], 0.)
    return best_pairs


@dataclass
class Track:
    id: int
    box: torch.Tensor
    velocity: torch.Tensor
    last_frame: int


class Tracker:
    def __init__(self, motion=False, max_age=2):
        self.motion, self.max_age, self.next_id = motion, max_age, 1
        self.tracks = []

    def update(self, boxes, frame):
        self.tracks = [t for t in self.tracks if frame - t.last_frame <= self.max_age]
        predicted = [t.box + t.velocity * (frame - t.last_frame) if self.motion else t.box for t in self.tracks]
        predicted = torch.stack(predicted) if predicted else torch.empty(0, 4)
        pairs = exact_gated_matching(iou_matrix(predicted, boxes))
        ids = [-1] * len(boxes)
        for row, col in pairs:
            track = self.tracks[row]
            track.velocity = (boxes[col] - track.box) / (frame - track.last_frame)
            track.box, track.last_frame = boxes[col].clone(), frame
            ids[col] = track.id
        for col, box in enumerate(boxes):
            if ids[col] < 0:
                ids[col] = self.next_id
                self.tracks.append(Track(self.next_id, box.clone(), torch.zeros(4), frame))
                self.next_id += 1
        return ids


def truth_boxes(f):
    # Ground truth: both objects exist in all six frames; A moves right, B moves left.
    return {'A': [16 + 8*f, 20, 28 + 8*f, 32], 'B': [40 - 8*f, 20, 52 - 8*f, 32]}


def frames():
    for f in range(6):
        labels = ['A', 'B'] if f != 4 else ['A']  # B exists but its detector misses frame4
        yield labels, torch.tensor([truth_boxes(f)[label] for label in labels], dtype=torch.float32)


def run(motion, max_age=2):
    tracker, last_id, switches, rows = Tracker(motion=motion, max_age=max_age), {}, 0, []
    for f, (truth_ids, boxes) in enumerate(frames()):
        assigned = tracker.update(boxes, f)
        for truth, track_id in zip(truth_ids, assigned):
            if truth in last_id and track_id != last_id[truth]:
                switches += 1
            last_id[truth] = track_id
        rows.append(assigned)
    return switches, rows


def evaluate_detections(iou_threshold=.5):
    # Score the detections themselves (no tracker) with the TP/FP rule of section 6.2:
    # a detection matched one-to-one to a ground-truth box at IoU >= 0.5 is a true positive,
    # and a detection left unmatched is a false positive.
    true_positives = truth_count = detection_count = 0
    for f, (_, boxes) in enumerate(frames()):
        truth = torch.tensor(list(truth_boxes(f).values()), dtype=torch.float32)
        true_positives += len(exact_gated_matching(iou_matrix(truth, boxes), threshold=iou_threshold))
        truth_count, detection_count = truth_count + len(truth), detection_count + len(boxes)
    return true_positives, truth_count, detection_count - true_positives


def save_panel(raw_ids, motion_ids, raw_switches, motion_switches, max_age, path):
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1040 420" role="img" aria-labelledby="t d">',
             f'<title id="t">max_age={max_age} 時的 ID 切換：上一框法 {raw_switches} 次，速度預測法 {motion_switches} 次</title>',
             f'<desc id="d">六幀畫面，兩個真實身分 A 與 B。上列用上一框法配對，ID 切換 {raw_switches} 次；'
             f'下列用速度預測法，ID 切換 {motion_switches} 次。兩者都用 max_age={max_age}。第 4 幀漏檢 B。</desc>',
             '<rect width="1040" height="420" fill="#f8fafc"/>']
    for row, ids in enumerate((raw_ids, motion_ids)):
        y = 55 + row * 170
        name = (f'上一框法（直接用上一框算 IoU）：ID 切換 {raw_switches} 次' if row == 0
                else f'速度預測法（先預測位置再算 IoU）：ID 切換 {motion_switches} 次')
        parts.append(f'<text x="20" y="{y-20}" font-family="sans-serif" font-size="22" fill="#0f172a">{name}</text>')
        for f, (labels, boxes) in enumerate(frames()):
            x = 20 + f * 170
            parts.append(f'<rect x="{x}" y="{y}" width="155" height="120" fill="white" stroke="#94a3b8"/>')
            parts.append(f'<text x="{x+8}" y="{y+25}" font-family="sans-serif" font-size="20">第 {f} 幀</text>')
            for label, box, identity in zip(labels, boxes, ids[f]):
                bx = x + 6 + float(box[0]) * 1.9
                color = {1: '#dc2626', 2: '#2563eb', 3: '#7c3aed'}[identity]
                by = y + (55 if label == 'A' else 92)
                parts.append(f'<rect x="{bx}" y="{by}" width="23" height="18" fill="{color}"/>')
                parts.append(f'<text x="{x+8}" y="{by-4}" font-family="sans-serif" font-size="17">{label}：ID{identity}</text>')
            if f == 4:
                parts.append(f'<text x="{x+8}" y="{y+95}" font-family="sans-serif" font-size="17" fill="#64748b">B：漏檢</text>')
    parts.append('<text x="20" y="380" font-family="sans-serif" font-size="18">方塊顏色＝分配到的 track ID；A、B 只用來評分。兩種方法吃同一份偵測框。</text>')
    parts.append('<text x="20" y="408" font-family="sans-serif" font-size="18">水平位置是真的；A、B 分上下兩條只為了讓標籤分開，配對時兩者的 y 座標相同。</text>')
    parts.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(parts) + '\n')


def main(max_age=2):
    torch.manual_seed(7)
    torch.set_num_threads(2)
    assert max_age in (1, 2), 'This exercise compares the documented max_age=1 and 2 cases.'
    raw_switches, raw_ids = run(False, max_age=max_age)
    motion_switches, motion_ids = run(True, max_age=max_age)
    expected_motion_switches = 0 if max_age == 2 else 1
    assert raw_switches == 3 and motion_switches == expected_motion_switches
    assert motion_ids[-1] == ([1, 2] if max_age == 2 else [1, 3])
    # Both trackers get the same detections, so detector recall does not depend on the tracker.
    true_positives, truth_count, false_positives = evaluate_detections()
    assert (true_positives, truth_count, false_positives) == (11, 12, 0)
    # A physically realizable IoU matrix where a greedy maximum is suboptimal.
    tracks = torch.tensor([[2., 0., 12., 10.], [6., 0., 16., 10.]])
    detections = torch.tensor([[3., 0., 13., 10.], [0., 0., 10., 10.]])
    quality = iou_matrix(tracks, detections)
    pairs = exact_gated_matching(quality)
    assert pairs == [(0, 1), (1, 0)]
    assert quality[0, 1] + quality[1, 0] > quality[0, 0] + quality[1, 1]
    assert exact_gated_matching(torch.zeros(2, 0)) == []
    panel = Path('artifacts/lesson-19/ids.svg')
    save_panel(raw_ids, motion_ids, raw_switches, motion_switches, max_age, panel)
    print('max_age:', max_age)
    print('per-frame IDs in detection order A,B (frame4 has only A):')
    print('last-box IoU:', raw_ids, '; switches:', raw_switches)
    print('velocity IoU:', motion_ids, '; switches:', motion_switches)
    print(f'detector recall={true_positives}/{truth_count}, false positives={false_positives} '
          '(same detections feed both trackers)')
    print('exact small matching passes non-greedy counterexample and empty detections')
    print(f'actual ID panel: {panel}')


if __name__ == '__main__':
    main()

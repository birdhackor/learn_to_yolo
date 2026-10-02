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


def frames():
    for f in range(6):
        labels = ['A', 'B'] if f != 4 else ['A']
        boxes = [[16 + 8*f, 20, 28 + 8*f, 32]]
        if f != 4:  # B exists but its detector misses frame4
            boxes.append([40 - 8*f, 20, 52 - 8*f, 32])
        yield labels, torch.tensor(boxes, dtype=torch.float32)


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


def save_panel(raw_ids, motion_ids, raw_switches, motion_switches, max_age):
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1040 420" role="img" aria-labelledby="t d">',
             f'<title id="t">ID switches at max_age={max_age}: last-box {raw_switches}, velocity {motion_switches}</title>',
             f'<desc id="d">Six frames, two ground-truth identities A and B. Top row matches last boxes with {raw_switches} switches. Bottom uses velocity prediction with {motion_switches} switches. Both use max_age={max_age}. Frame4 misses B.</desc>',
             '<rect width="1040" height="420" fill="#f8fafc"/>']
    for row, ids in enumerate((raw_ids, motion_ids)):
        y = 55 + row * 170
        name = f'Last-box IoU: {raw_switches} switches' if row == 0 else f'Velocity + IoU: {motion_switches} switches'
        parts.append(f'<text x="20" y="{y-20}" font-family="sans-serif" font-size="22" fill="#0f172a">{name}</text>')
        for f, (labels, boxes) in enumerate(frames()):
            x = 20 + f * 170
            parts.append(f'<rect x="{x}" y="{y}" width="155" height="120" fill="white" stroke="#94a3b8"/>')
            parts.append(f'<text x="{x+8}" y="{y+25}" font-family="sans-serif" font-size="20">frame {f}</text>')
            for label, box, identity in zip(labels, boxes, ids[f]):
                bx = x + 6 + float(box[0]) * 1.9
                color = {1: '#dc2626', 2: '#2563eb', 3: '#7c3aed'}[identity]
                by = y + (55 if label == 'A' else 92)
                parts.append(f'<rect x="{bx}" y="{by}" width="23" height="18" fill="{color}"/>')
                parts.append(f'<text x="{x+8}" y="{by-4}" font-family="sans-serif" font-size="17">{label}: ID{identity}</text>')
            if f == 4:
                parts.append(f'<text x="{x+8}" y="{y+95}" font-family="sans-serif" font-size="17" fill="#64748b">B: missed</text>')
    parts.append('<text x="20" y="380" font-family="sans-serif" font-size="18">Colors = assigned track IDs. A/B labels are used only for evaluation. Same detections feed both trackers.</text>')
    parts.append('<text x="20" y="408" font-family="sans-serif" font-size="18">Horizontal position is real. A/B display lanes separate labels; matching uses the same y coordinate.</text>')
    parts.append('</svg>')
    Path('docs/assets/diagrams/19-tracking.svg').write_text('\n'.join(parts))


def main(max_age=2):
    torch.manual_seed(7)
    torch.set_num_threads(2)
    assert max_age in (1, 2), 'This exercise compares the documented max_age=1 and 2 cases.'
    raw_switches, raw_ids = run(False, max_age=max_age)
    motion_switches, motion_ids = run(True, max_age=max_age)
    expected_motion_switches = 0 if max_age == 2 else 1
    assert raw_switches == 3 and motion_switches == expected_motion_switches
    assert motion_ids[-1] == ([1, 2] if max_age == 2 else [1, 3])
    # A physically realizable IoU matrix where a greedy maximum is suboptimal.
    tracks = torch.tensor([[2., 0., 12., 10.], [6., 0., 16., 10.]])
    detections = torch.tensor([[3., 0., 13., 10.], [0., 0., 10., 10.]])
    quality = iou_matrix(tracks, detections)
    pairs = exact_gated_matching(quality)
    assert pairs == [(0, 1), (1, 0)]
    assert quality[0, 1] + quality[1, 0] > quality[0, 0] + quality[1, 1]
    assert exact_gated_matching(torch.zeros(2, 0)) == []
    save_panel(raw_ids, motion_ids, raw_switches, motion_switches, max_age)
    print('max_age:', max_age)
    print('per-frame IDs in detection order A,B (frame4 has only A):')
    print('last-box IoU:', raw_ids, '; switches:', raw_switches)
    print('velocity IoU:', motion_ids, '; switches:', motion_switches)
    print('detector recall=11/12 for both; no false positives in this artificial sequence')
    print('exact small matching passes non-greedy counterexample and empty detections')
    print('actual ID panel: docs/assets/diagrams/19-tracking.svg')


if __name__ == '__main__':
    main()

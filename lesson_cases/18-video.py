"""Lazy RGB frame stream -> letterbox -> a locally trained detector -> overlays."""
import base64
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from miniyolo.data import ShapeDataset, collate
from miniyolo.models import GridDetector
from miniyolo.targets import build_targets
from miniyolo.losses import grid_loss
from miniyolo.geometry import letterbox, undo_letterbox
from miniyolo.inference import decode_grid


@dataclass
class Frame:
    index: int
    timestamp_s: float
    rgb: np.ndarray  # uint8 HWC RGB; timestamp is source time, not processing latency


def synthetic_frames(count=12, fps=20):
    assert count >= 2 and fps > 0
    for i in range(count):
        rgb = np.full((64, 96, 3), 8, dtype=np.uint8)
        x = 4 + 4 * i
        # Clip at the right edge; once x >= width, this frame has no visible object.
        if x < rgb.shape[1]:
            rgb[20:34, x:min(x+14, rgb.shape[1])] = [242, 25, 25]
        yield Frame(i, i / fps, rgb)


def opencv_frames(source):
    """Optional real file/camera adapter. Never called by this lesson's main()."""
    import cv2  # optional: pip install opencv-python-headless
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f'Cannot open video source: {source}')
    fps = capture.get(cv2.CAP_PROP_FPS)
    started, index = time.perf_counter(), 0
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            timestamp = index / fps if isinstance(source, str) and fps > 0 else time.perf_counter() - started
            yield Frame(index, timestamp, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            index += 1
    finally:
        capture.release()


def fit_detector():
    torch.manual_seed(7)
    dataset = ShapeDataset(n=32, size=64, seed=4100)
    images, targets = collate([dataset[i] for i in range(len(dataset))])
    encoded = build_targets(targets)
    model = GridDetector(width=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    for step in range(160):
        ids = torch.tensor([(step * 8 + j) % 32 for j in range(8)])
        optimizer.zero_grad()
        loss = grid_loss(model(images[ids]), {k: v[ids] for k, v in encoded.items()})['total']
        loss.backward()
        optimizer.step()
    return model.eval()


@torch.no_grad()
def run_stream(frames, model):
    """Consume any iterable of Frame, yield one result at a time; no camera required."""
    for frame in frames:
        assert frame.rgb.dtype == np.uint8 and frame.rgb.ndim == 3 and frame.rgb.shape[-1] == 3
        started = time.perf_counter()
        image = torch.from_numpy(frame.rgb.copy()).permute(2, 0, 1).float() / 255
        image, _, metadata = letterbox(image, torch.empty(0, 4), size=64)
        t1 = time.perf_counter()
        raw = model(image[None])
        t2 = time.perf_counter()
        prediction = decode_grid(raw, score_threshold=.1, nms_iou=.5)[0]
        prediction['boxes'] = undo_letterbox(prediction['boxes'], metadata)
        t3 = time.perf_counter()
        overlay = Image.fromarray(frame.rgb.copy())
        pen = ImageDraw.Draw(overlay)
        for box, score in zip(prediction['boxes'], prediction['scores']):
            pen.rectangle(tuple(float(v) for v in box), outline='yellow', width=1)
            pen.text((max(0, float(box[0])), max(0, float(box[1])-9)), f'{float(score):.2f}', fill='white')
        t4 = time.perf_counter()
        yield {'index': frame.index, 'source_timestamp_s': frame.timestamp_s, 'image': overlay, 'prediction': prediction,
               'ms': {'preprocess': (t1-started)*1000, 'model': (t2-t1)*1000,
                      'postprocess': (t3-t2)*1000, 'drawing': (t4-t3)*1000, 'total': (t4-started)*1000}}


def save_panel(results, fps):
    selected = [0, (len(results)-1)//2, len(results)-1]
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 290" role="img" aria-labelledby="t d">',
             f'<title id="t">Actual moving-frame detector output: {len(results)} frames at {fps} source FPS</title>',
             f'<desc id="d">Generated RGB frames {selected} from this run, with yellow model predictions, source timestamps and processing time. The last frame is included; predictions are not ground truth.</desc>',
             '<rect width="960" height="290" fill="#0f172a"/>']
    for col, i in enumerate(selected):
        r = results[i]
        buffer = io.BytesIO()
        r['image'].save(buffer, format='PNG')
        data = base64.b64encode(buffer.getvalue()).decode()
        x = 15 + col * 315
        parts.append(f'<image x="{x}" y="20" width="288" height="192" href="data:image/png;base64,{data}"/>')
        parts.append(f'<text x="{x}" y="242" fill="white" font-family="sans-serif" font-size="21">frame {i}: source {r["source_timestamp_s"]:.2f}s</text>')
        parts.append(f'<text x="{x}" y="272" fill="white" font-family="sans-serif" font-size="19">pipeline {r["ms"]["total"]:.2f}ms | boxes {len(r["prediction"]["boxes"])}</text>')
    parts.append('</svg>')
    Path('docs/assets/diagrams/18-video.svg').write_text('\n'.join(parts))


def main(count=12, fps=20):
    torch.manual_seed(7)
    torch.set_num_threads(2)
    model = fit_detector()
    # Only this small GIF demo collects frames; run_stream itself remains lazy.
    results = list(run_stream(synthetic_frames(count=count, fps=fps), model))
    assert len(results) == count and all(r['image'].size == (96, 64) for r in results)
    assert [r['index'] for r in results] == list(range(count))
    assert abs(results[-1]['source_timestamp_s'] - (count-1)/fps) < 1e-8
    output = Path('artifacts/lesson-18')
    output.mkdir(parents=True, exist_ok=True)
    gif_duration_ms = round(1000/fps)
    results[0]['image'].save(output / 'stream.gif', save_all=True, append_images=[r['image'] for r in results[1:]], duration=gif_duration_ms, loop=0)
    # Exclude the first frame from summary because model/runtime initialization can distort timing.
    timing = {key: float(np.median([r['ms'][key] for r in results[1:]])) for key in results[0]['ms']}
    summary = {'frames': len(results), 'source_fps': fps, 'source_duration_s': count/fps,
               'source_last_timestamp_s': results[-1]['source_timestamp_s'], 'gif_frame_duration_ms': gif_duration_ms,
               'source_size_hw': [64, 96], 'object_visible_per_frame': [4+4*i < 96 for i in range(count)],
               'model_input_hw': [64, 64], 'boxes_per_frame': [len(r['prediction']['boxes']) for r in results],
               'median_ms_excluding_first': timing, 'camera_adapter': 'provided, not executed',
               'limits': 'synthetic lazy producer; no capture/codec/display/network queue latency measured'}
    (output / 'report.json').write_text(json.dumps(summary, indent=2)+'\n')
    save_panel(results, fps)
    print(json.dumps(summary, indent=2))
    print('GIF: artifacts/lesson-18/stream.gif; actual static panel: docs/assets/diagrams/18-video.svg')


if __name__ == '__main__':
    main()

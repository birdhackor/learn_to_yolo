"""Real ONNX export, checker, ORT CPU execution and decoded-box parity."""
import json
import statistics
import time
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument
import torch
from miniyolo.models import GridDetector
from miniyolo.geometry import letterbox, undo_letterbox
from miniyolo.inference import decode_grid
from miniyolo.targets import build_targets
from miniyolo.losses import grid_loss
from miniyolo.data import ShapeDataset, collate


def preprocess(rgb):
    assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[-1] == 3
    image = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255
    image, _, metadata = letterbox(image, torch.empty(0, 4), size=64)
    return image, metadata


def restore(raw, metadata):
    result = decode_grid(torch.from_numpy(raw.copy()), image_size=64, score_threshold=.05, nms_iou=.5)
    for pred, meta in zip(result, metadata):
        pred['boxes'] = undo_letterbox(pred['boxes'], meta)
    return result


def median_ms(function, iterations=20):
    for _ in range(3):
        function()
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        function()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    output = Path('artifacts/lesson-20')
    output.mkdir(parents=True, exist_ok=True)
    model = GridDetector(width=8)
    # A real forward/backward/step supplies locally learned weights; no pretrained download.
    data = ShapeDataset(n=4, size=64, seed=5100)
    images, targets = collate([data[i] for i in range(4)])
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    optimizer.zero_grad()
    grid_loss(model(images), build_targets(targets))['total'].backward()
    optimizer.step()
    model.eval()
    onnx_path = output / 'grid.onnx'
    torch.onnx.export(model, images[:1], str(onnx_path), input_names=['images'], output_names=['raw_grid'],
                      dynamic_axes={'images': {0: 'batch'}, 'raw_grid': {0: 'batch'}},
                      opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
    session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=['CPUExecutionProvider'])
    assert session.get_inputs()[0].shape == ['batch', 3, 64, 64]
    rng = np.random.default_rng(7)
    sources = [rng.integers(0, 256, shape, dtype=np.uint8)
               for shape in ((64, 96, 3), (48, 80, 3), (80, 48, 3))]
    processed = [preprocess(rgb) for rgb in sources]
    batch = torch.stack([item[0] for item in processed])
    metadata = [item[1] for item in processed]
    errors = []
    batches_checked = [1, 2, 3]
    with torch.no_grad():
        for b in batches_checked:
            assert batch[:b].shape == (b, 3, 64, 64)
            torch_raw = model(batch[:b]).numpy()
            ort_raw = session.run(['raw_grid'], {'images': batch[:b].numpy()})[0]
            assert torch_raw.shape == ort_raw.shape == (b, 4, 4, 7)
            error = float(np.abs(torch_raw - ort_raw).max())
            errors.append(error)
            np.testing.assert_allclose(torch_raw, ort_raw, rtol=1e-5, atol=1e-5)
            pytorch_predictions, ort_predictions = restore(torch_raw, metadata[:b]), restore(ort_raw, metadata[:b])
            for a, o in zip(pytorch_predictions, ort_predictions):
                torch.testing.assert_close(a['boxes'], o['boxes'], rtol=1e-5, atol=1e-4)
                torch.testing.assert_close(a['scores'], o['scores'], rtol=1e-5, atol=1e-6)
                assert torch.equal(a['labels'], o['labels'])
    try:
        session.run(['raw_grid'], {'images': np.zeros((1, 3, 80, 80), dtype=np.float32)})
    except InvalidArgument:
        spatial_rejected = True
    else:
        spatial_rejected = False
    assert spatial_rejected, 'Export is intentionally dynamic-batch, fixed spatial size.'
    with torch.no_grad():
        pytorch_ms = median_ms(lambda: model(batch[:1]))
    ort_ms = median_ms(lambda: session.run(['raw_grid'], {'images': batch[:1].numpy()}))
    def ort_pipeline():
        image, meta = preprocess(sources[0])
        raw = session.run(['raw_grid'], {'images': image[None].numpy()})[0]
        return restore(raw, [meta])
    def torch_pipeline():
        image, meta = preprocess(sources[0])
        with torch.no_grad():
            raw = model(image[None]).numpy()
        return restore(raw, [meta])
    end_to_end_ms = median_ms(ort_pipeline)
    torch_end_to_end_ms = median_ms(torch_pipeline)
    batch2_ms = median_ms(lambda: session.run(['raw_grid'], {'images': batch[:2].numpy()}))
    report = {'onnx': str(onnx_path), 'opset': 17, 'providers': session.get_providers(),
              'input_shape': session.get_inputs()[0].shape, 'batches_checked': batches_checked, 'max_abs_raw_errors': errors,
              'decoded_boxes_scores_labels_match': True, 'dynamic_spatial': False, 'spatial80_rejected': spatial_rejected,
              'median_ms': {'torch_raw_batch1': pytorch_ms, 'ort_raw_batch1': ort_ms,
                            'torch_preprocess_to_restored_boxes': torch_end_to_end_ms,
                            'ort_preprocess_to_restored_boxes': end_to_end_ms, 'ort_raw_batch2': batch2_ms},
              'ort_raw_batch2_images_per_second': 2000 / batch2_ms,
              'versions': {'torch': torch.__version__, 'onnx': onnx.__version__, 'onnxruntime': ort.__version__},
              'limits': 'CPU float32 parity; one training step is not detection-quality evidence; no TensorRT/GPU verification'}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    print('real ONNX export + checker + ORT CPU + restored-box parity completed')


if __name__ == '__main__':
    main()

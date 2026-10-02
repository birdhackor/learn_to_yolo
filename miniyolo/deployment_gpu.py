"""Bounded real TensorRT execution; no claim about real-photo detection quality."""
from pathlib import Path
import hashlib
import json
import statistics
import time


def verify(destination, code_commit):
    import numpy as np
    import onnx
    import onnxruntime as ort
    import tensorrt as trt
    import torch
    from miniyolo.data import ShapeDataset, collate
    from miniyolo.models import GridDetector
    from miniyolo.targets import build_targets
    from miniyolo.train import optimizer_step
    from miniyolo.inference import decode_grid

    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert "L4" in torch.cuda.get_device_name(0)
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    data = ShapeDataset(n=8, seed=7, max_objects=1, allow_empty=False)
    images, annotations = collate([data[i] for i in range(8)])
    images = images.cuda()
    targets = {k: v.cuda() for k, v in build_targets(annotations).items()}
    model = GridDetector(width=8).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    history = []
    for _ in range(40):
        parts, gradient_norm = optimizer_step(model, optimizer, images, targets, validate_gradients=True)
        history.append(parts)
    assert history[-1]["total"] < history[0]["total"] * .8
    model.eval()
    path = output / "grid.onnx"
    torch.onnx.export(model, images[:1], str(path), input_names=["images"], output_names=["raw_grid"],
                      dynamic_axes={"images": {0: "batch"}, "raw_grid": {0: "batch"}},
                      opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(path))
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    logger = trt.Logger(trt.Logger.WARNING)
    results = []
    for precision in ("fp32", "fp16"):
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        assert parser.parse(path.read_bytes()), "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        config = builder.create_builder_config()
        config.clear_flag(trt.BuilderFlag.TF32)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 * 1024 * 1024)
        if precision == "fp16":
            assert builder.platform_has_fast_fp16
            config.set_flag(trt.BuilderFlag.FP16)
        profile = builder.create_optimization_profile()
        profile.set_shape("images", (1, 3, 64, 64), (2, 3, 64, 64), (4, 3, 64, 64))
        config.add_optimization_profile(profile)
        serialized = builder.build_serialized_network(network, config)
        assert serialized is not None
        engine_path = output / f"grid-{precision}.engine"
        engine_path.write_bytes(bytes(serialized))
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(serialized)
        context = engine.create_execution_context()
        stream = torch.cuda.current_stream()

        def infer(x):
            assert x.is_cuda and x.dtype == torch.float32 and x.is_contiguous()
            assert context.set_input_shape("images", tuple(x.shape))
            shape = tuple(context.get_tensor_shape("raw_grid"))
            dtype = {trt.float32: torch.float32, trt.float16: torch.float16}[engine.get_tensor_dtype("raw_grid")]
            result = torch.empty(shape, dtype=dtype, device="cuda")
            assert context.set_tensor_address("images", x.data_ptr())
            assert context.set_tensor_address("raw_grid", result.data_ptr())
            assert context.execute_async_v3(stream.cuda_stream)
            stream.synchronize()  # also keeps buffers alive until execution completes
            return result

        batches = []
        with torch.inference_mode():
            for batch_size in (1, 2, 3, 4):
                x = images[:batch_size].contiguous()
                expected = model(x).cpu()
                ort_raw = torch.from_numpy(session.run(["raw_grid"], {"images": x.cpu().numpy()})[0])
                torch.testing.assert_close(expected, ort_raw, rtol=1e-4, atol=1e-5)
                actual = infer(x).float().cpu()
                assert actual.shape == expected.shape == (batch_size, 4, 4, 7)
                torch.testing.assert_close(actual, expected, rtol=2e-3 if precision == "fp16" else 1e-4,
                                           atol=2e-2 if precision == "fp16" else 1e-5)
                reference_boxes = decode_grid(expected, score_threshold=.05)
                actual_boxes = decode_grid(actual, score_threshold=.05)
                max_box, max_score = 0., 0.
                for a, b in zip(reference_boxes, actual_boxes):
                    assert torch.equal(a["labels"], b["labels"]), "Candidate identity/order changed"
                    torch.testing.assert_close(a["boxes"], b["boxes"], rtol=0,
                                               atol=.1 if precision == "fp16" else .001)
                    torch.testing.assert_close(a["scores"], b["scores"], rtol=0,
                                               atol=.002 if precision == "fp16" else 1e-5)
                    if a["boxes"].numel():
                        max_box = max(max_box, float((a["boxes"] - b["boxes"]).abs().max()))
                        max_score = max(max_score, float((a["scores"] - b["scores"]).abs().max()))
                batches.append({"batch": batch_size, "max_abs_raw_error": float((actual - expected).abs().max()),
                                "max_abs_box_pixel_error": max_box, "max_abs_score_error": max_score,
                                "decoded_candidates": sum(len(b["boxes"]) for b in actual_boxes),
                                "decoded_labels_and_order_match": True})
            assert sum(row["decoded_candidates"] for row in batches) > 0, "Empty predictions cannot verify decoding"
            for _ in range(3):
                infer(images[:1].contiguous())
            times = []
            for _ in range(20):
                torch.cuda.synchronize()
                start = time.perf_counter()
                infer(images[:1].contiguous())
                torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)
        results.append({"precision_flag": precision, "tf32_enabled": False, "batches": batches,
                        "raw_batch1_median_ms": statistics.median(times),
                        "timing_scope": "3 warmup, 20 measurements; shape/address setup, output allocation, enqueue and stream/device synchronization; excludes preprocessing, decode, copies, engine build and container startup",
                        "engine_sha256": hashlib.sha256(engine_path.read_bytes()).hexdigest()})
    report = {"passed": True, "code_commit": code_commit, "gpu": torch.cuda.get_device_name(0),
              "torch": str(torch.__version__), "torch_cuda": torch.version.cuda, "tensorrt": trt.__version__,
              "onnxruntime": ort.__version__, "parameters": sum(p.numel() for p in model.parameters()),
              "optimizer_steps": 40, "initial_loss": history[0]["total"], "final_loss": history[-1]["total"],
              "engines": results, "onnx_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "limits": "L4 single device, fixed synthetic training/evaluation images, not held-out quality or full YOLO performance; FP16 flag permits mixed precision, not a guarantee every layer uses FP16; INT8 untested"}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report

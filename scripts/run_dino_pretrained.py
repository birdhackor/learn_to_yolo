"""Extract official DINOv2 ViT-S/14 features on CPU; download is opt-in by running this CLI.

The default inputs are controlled RGB drawings, not a natural-image semantic benchmark.
Code and weights are cached under ignored artifacts/runs; torchvision is not required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import warnings
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import PIL
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from miniyolo.provenance import file_sha256, machine_info, repo_dependencies

COMMIT = "e1277af2ba9496fbadf7aec6eba56e8d882d1e35"
CODE_URL = f"https://codeload.github.com/facebookresearch/dinov2/zip/{COMMIT}"
CODE_SHA256 = "209c4f04fb0d63a869f7ea5c18d61dc7d8196c88cb1cb4a53bf3b0607146685c"
WEIGHTS_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
WEIGHTS_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
SIZE = 224
PATCH = 14


def verified_download(url: str, path: Path, expected_sha256: str) -> dict:
    """Require the recorded SHA-256 for both new downloads and existing cache files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cached = path.is_file()
    if not cached:
        temporary = path.with_suffix(path.suffix + ".part")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "learn-to-yolo-dinov2-probe"})
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as dest:
                while chunk := response.read(1024 * 1024):
                    dest.write(chunk)
            actual = file_sha256(temporary)
            if actual != expected_sha256:
                raise ValueError(f"SHA-256 mismatch for {path.name}: {actual}")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"Cached SHA-256 mismatch for {path.name}: {actual}")
    return {"url": url, "path": str(path), "sha256": actual,
            "expected_sha256": expected_sha256, "sha256_verified": True,
            "bytes": path.stat().st_size, "cache_hit": cached}


def official_code(cache: Path) -> tuple[Path, dict]:
    archive = cache / f"dinov2-{COMMIT}.zip"
    metadata = verified_download(CODE_URL, archive, CODE_SHA256)
    code_dir = cache / f"dinov2-{COMMIT}"
    # Re-extract the verified archive so edited cached Python cannot replace official code.
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            destination = (cache / member.filename).resolve()
            if not destination.is_relative_to(cache.resolve()):
                raise ValueError("Unsafe path in official source archive")
        bundle.extractall(cache)
    files = sorted(path for path in code_dir.rglob("*.py")) + [code_dir / "LICENSE"]
    hashes = {path.relative_to(code_dir).as_posix(): file_sha256(path) for path in files}
    metadata.update({"repository": "https://github.com/facebookresearch/dinov2",
                     "commit": COMMIT, "source": "official archive, unchanged",
                     "python_and_license_sha256": hashes,
                     "license": "Apache-2.0",
                     "license_url": f"https://github.com/facebookresearch/dinov2/blob/{COMMIT}/LICENSE"})
    return code_dir, metadata


def controlled_images() -> tuple[list[Image.Image], dict]:
    """Move the same colored shapes, preserving their sizes and RGB values."""
    images = []
    boxes = [
        {"red_square": [35, 35, 91, 91], "green_disk": [133, 35, 189, 91],
         "blue_triangle": [84, 126, 140, 182]},
        {"red_square": [133, 133, 189, 189], "green_disk": [35, 35, 91, 91],
         "blue_triangle": [133, 35, 189, 91]},
    ]
    for objects in boxes:
        image = Image.new("RGB", (SIZE, SIZE), (224, 224, 224))
        draw = ImageDraw.Draw(image)
        left, top, right, bottom = objects["red_square"]
        draw.rectangle([left, top, right - 1, bottom - 1], fill=(210, 40, 40))
        left, top, right, bottom = objects["green_disk"]
        draw.ellipse([left, top, right - 1, bottom - 1], fill=(40, 160, 60))
        left, top, right, bottom = objects["blue_triangle"]
        draw.polygon([(left, bottom - 1), ((left + right) // 2, top), (right - 1, bottom - 1)], fill=(40, 80, 210))
        images.append(image)
    return images, {"kind": "controlled synthetic RGB drawings", "objects_xyxy_pixels": boxes,
                    "box_convention": "xyxy pixels with exclusive right and bottom endpoints",
                    "background_rgb": [224, 224, 224],
                    "conclusion_scope": "Feature extraction and similarity for these drawings only; no natural-image semantic claim."}


def preprocess(images: list[Image.Image]) -> tuple[torch.Tensor, list[Image.Image]]:
    """RGB -> bicubic 224x224 -> float NCHW [0,1] -> per-channel ImageNet mean/std."""
    resized = [image.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BICUBIC) for image in images]
    arrays = np.stack([np.asarray(image, dtype=np.float32) / 255.0 for image in resized])
    tensor = torch.from_numpy(arrays).permute(0, 3, 1, 2).contiguous()
    tensor = (tensor - tensor.new_tensor(MEAN)[None, :, None, None]) / tensor.new_tensor(STD)[None, :, None, None]
    return tensor, resized


def run(args: argparse.Namespace) -> dict:
    start = time.perf_counter()
    torch.set_num_threads(args.threads)
    os.environ["XFORMERS_DISABLED"] = "1"
    warnings.filterwarnings("ignore", message=r"xFormers is (disabled|not available).*", category=UserWarning)
    cache = args.cache_dir.resolve()
    code_dir, code_metadata = official_code(cache)
    weight_path = cache / "dinov2_vits14_pretrain.pth"
    weight_metadata = verified_download(WEIGHTS_URL, weight_path, WEIGHTS_SHA256)
    # torch.hub executes the pinned, verified official entrypoint; load the verified state dict separately.
    model = torch.hub.load(str(code_dir), "dinov2_vits14", source="local", pretrained=False)
    model.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True), strict=True)
    model.eval().requires_grad_(False)
    if args.image_a is None:
        images, input_metadata = controlled_images()
    else:
        with Image.open(args.image_a) as image_a, Image.open(args.image_b) as image_b:
            images = [image_a.convert("RGB"), image_b.convert("RGB")]
        input_metadata = {"kind": "user-selected images", "original_paths": [str(args.image_a), str(args.image_b)],
                          "original_file_sha256": [file_sha256(args.image_a), file_sha256(args.image_b)],
                          "conclusion_scope": "Similarity for these two images; not a benchmark or a semantic guarantee."}
    tensor, resized = preprocess(images)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    output_dir = args.report.parent.resolve()
    if not output_dir.is_relative_to((ROOT / "artifacts/runs").resolve()):
        output_dir = ROOT / "artifacts/runs/dino-pretrained/outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    input_metadata["images"] = []
    for index, image in enumerate(resized):
        path = output_dir / f"input-{index}.png"
        image.save(path)
        input_metadata["images"].append({"path": str(path), "file_sha256": file_sha256(path),
                                         "rgb_pixels_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                                         "shape_hwc": [SIZE, SIZE, 3]})
    inference_start = time.perf_counter()
    with torch.inference_mode():
        outputs = model.forward_features(tensor)
    inference_seconds = time.perf_counter() - inference_start
    cls = outputs["x_norm_clstoken"]
    patch = outputs["x_norm_patchtokens"]
    registers = outputs["x_norm_regtokens"]
    if not torch.isfinite(cls).all() or not torch.isfinite(patch).all():
        raise ValueError("Non-finite official backbone features")
    cls_unit = F.normalize(cls, dim=-1)
    patch_unit = F.normalize(patch, dim=-1)
    row, column = args.query
    query_index = row * (SIZE // PATCH) + column
    similarities = patch_unit[1] @ patch_unit[0, query_index]
    values, indices = similarities.topk(8)
    neighbors = []
    for similarity, index in zip(values.tolist(), indices.tolist()):
        neighbor_row, neighbor_column = divmod(index, SIZE // PATCH)
        neighbors.append({"target_image": 1, "patch_index": index,
                          "row": neighbor_row, "column": neighbor_column, "cosine": similarity,
                          "xyxy_resized_pixels": [neighbor_column * PATCH, neighbor_row * PATCH,
                                                  (neighbor_column + 1) * PATCH, (neighbor_row + 1) * PATCH]})
    gram = patch_unit @ patch_unit.transpose(1, 2)
    channel_permuted = patch_unit.flip(-1)
    gram_permuted = channel_permuted @ channel_permuted.transpose(1, 2)
    npz_path = output_dir / "features.npz"
    np.savez(npz_path, cls=cls.numpy(), patch=patch.numpy(), patch_unit=patch_unit.numpy(), gram=gram.numpy())
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    report = {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "dependencies_sha256": repo_dependencies(__file__),
        "passed": True,
        "purpose": "Official frozen DINOv2 ViT-S/14 feature extraction on CPU; no training.",
        "source_record": {"git_commit": git.stdout.strip() or None,
                          "entrypoint": "scripts/run_dino_pretrained.py",
                          "entrypoint_sha256": file_sha256(__file__),
                          "repo_dependencies": repo_dependencies(__file__)},
        "machine": {**machine_info(args.threads), "numpy": np.__version__, "pillow": PIL.__version__},
        "official_code": code_metadata,
        "checkpoint": weight_metadata,
        "model": {"name": "dinov2_vits14", "parameters": sum(p.numel() for p in model.parameters()),
                  "register_tokens": model.num_register_tokens, "training": model.training,
                  "parameters_updated": False, "weights_only_load": True, "strict_state_dict_load": True,
                  "license": "Apache-2.0 (official README covers code and model weights)"},
        "inputs": input_metadata,
        "preprocessing": {"color_mode": "RGB", "resize_hw": [SIZE, SIZE], "resize": "bicubic",
                          "aspect_ratio": "resized directly to square", "scaling": "uint8 / 255",
                          "normalization_mean_rgb": MEAN, "normalization_std_rgb": STD,
                          "tensor_shape_nchw": list(tensor.shape), "device": str(tensor.device)},
        "features": {"cls_shape_nd": list(cls.shape), "patch_shape_npd": list(patch.shape),
                     "patch_grid_hw": [SIZE // PATCH, SIZE // PATCH], "patch_size_pixels": PATCH,
                     "register_shape_nrd": list(registers.shape), "all_finite": True,
                     "axis_note": "Patch sequence is row-major. x_norm_* is official LayerNorm output; cosine adds L2 normalization.",
                     "saved_npz": str(npz_path), "saved_npz_sha256": file_sha256(npz_path),
                     "cls_cosines": (cls_unit @ cls_unit.T).tolist()},
        "patch_neighbors": {"query_image": 0, "query_row": row, "query_column": column,
                            "query_index": query_index,
                            "query_xyxy_resized_pixels": [column * PATCH, row * PATCH,
                                                          (column + 1) * PATCH, (row + 1) * PATCH],
                            "target_image": 1, "metric": "L2-normalized patch cosine",
                            "top8": neighbors,
                            "cosine_map_16x16": similarities.reshape(SIZE // PATCH, SIZE // PATCH).tolist(),
                            "scope": "These query/target drawings only. No detection labels, mAP, or natural-image semantic evaluation."},
        "gram_observation": {"shape_npp": list(gram.shape), "minimum": gram.min().item(),
                             "maximum": gram.max().item(), "diagonal_mean": gram.diagonal(dim1=1, dim2=2).mean().item(),
                             "channel_permutation_max_abs_difference": (gram - gram_permuted).abs().max().item(),
                             "interpretation": "Permuting feature channels changes vectors but preserves patch-pair cosine relations (up to float error).",
                             "scope": "Algebra on DINOv2 frozen features. No DINOv3 checkpoint or Gram-anchor training is executed."},
        "timing": {"total_seconds": time.perf_counter() - start, "forward_seconds": inference_seconds,
                   "scope": "Total includes downloads on cache miss, cache verification/loading, drawing, forward and evidence preparation; excludes interpreter startup/imports and final JSON write. CPU forward needs no CUDA synchronization; not a performance benchmark."},
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "artifacts/runs/dino-pretrained/report.json")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "artifacts/runs/dino-pretrained/cache")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--image-a", type=Path)
    parser.add_argument("--image-b", type=Path)
    parser.add_argument("--query", type=int, nargs=2, default=(4, 4), metavar=("ROW", "COLUMN"))
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if (args.image_a is None) != (args.image_b is None):
        parser.error("supply both --image-a and --image-b")
    if any(index < 0 or index >= SIZE // PATCH for index in args.query):
        parser.error("--query row and column must be in [0, 15]")
    report = run(args)
    print(json.dumps({"report": str(args.report), "checkpoint_sha256_verified": True,
                      "cls_shape": report["features"]["cls_shape_nd"],
                      "patch_shape": report["features"]["patch_shape_npd"],
                      "nearest_target_patch": report["patch_neighbors"]["top8"][0]}, indent=2))


if __name__ == "__main__":
    main()

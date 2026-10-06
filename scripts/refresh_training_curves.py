"""Bind the displayed ViT/DINO training curves to the current saved CPU records."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

SVG = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG)


def load_record(path):
    record = json.loads(path.read_text())
    output = json.JSONDecoder().raw_decode(record["stdout"])[0]
    return record, output


def save(root, path):
    path.write_text(ET.tostring(root, encoding="unicode") + "\n")


def metadata(root, record_path, history):
    node = root.find(f"{{{SVG}}}metadata")
    if node is None:
        node = ET.Element(f"{{{SVG}}}metadata")
        root.insert(2, node)
    node.text = json.dumps({"source": record_path.as_posix(),
                            "sha256": hashlib.sha256(record_path.read_bytes()).hexdigest(),
                            "loss_history": history}, ensure_ascii=False)


def vit_curve(record_path: Path, target: Path):
    _, output = load_record(record_path)
    history = output["loss_history"]
    assert len(history) == 60
    root = ET.parse(target).getroot()
    points = [(78+370*(p["step"]-1)/59, 414-340*p["loss"]) for p in history]
    curve = root.find(f"{{{SVG}}}polyline")
    curve.set("points", " ".join(f"{x:.3f},{y:.3f}" for x, y in points))
    circles = root.findall(f"{{{SVG}}}circle")
    assert len(circles) == len(points)
    for node, (x, y) in zip(circles, points):
        node.set("cx", f"{x:.3f}")
        node.set("cy", f"{y:.3f}")
    metadata(root, record_path, history)
    root.find(f"{{{SVG}}}desc").text = "目前21-training.json的loss_history。橫軸為更新編號1到60，縱軸是當次更新前的訓練batch交叉熵。第30次更新後保存checkpoint。曲線不包含val/test。"
    save(root, target)


def dino_curve(record_path: Path, target: Path):
    record, output = load_record(record_path)
    history = record["figure_data"]["self_distillation_history"]
    assert len(history) == 160 and history[0] == output["first_step"] and history[-1] == output["last_step"]
    root = ET.parse(target).getroot()
    points = [(78+421.2*(p["step"]-1)/159, 435.6-144*p["loss"]) for p in history]
    curve = root.find(f".//{{{SVG}}}g[@id='line2d_21']/{{{SVG}}}path")
    assert curve is not None
    curve.set("d", "M " + " L ".join(f"{x:.6f} {y:.6f}" for x, y in points))
    metadata(root, record_path, history)
    root.find(f"{{{SVG}}}desc").text = "上半曲線取目前22.3的160步紀錄；下半test CLS統計取22.4另一次同設定自監督訓練後的評分。不是同一次執行，random與SSL的1-NN及linear probe全部64比64。"
    save(root, target)


def detection_predictions(record_path: Path, target: Path):
    _, output = load_record(record_path)
    examples = output["examples"]
    root = ET.parse(target).getroot()
    images = root.findall(f".//{{{SVG}}}image")
    predictions = [n for n in root.findall(f".//{{{SVG}}}rect") if n.get("stroke") == "#ffb43b"]
    truths = [n for n in root.findall(f".//{{{SVG}}}rect") if n.get("stroke") == "#22d395"]
    assert len(images) == len(predictions) == len(truths) == len(examples) == 4
    for image, predicted, truth, example in zip(images, predictions, truths, examples):
        x0, y0 = float(image.get("x")), float(image.get("y"))
        scale = float(image.get("width"))/32
        x1, y1, x2, y2 = example["truth_xyxy_pixel"]
        expected = [x0+x1*scale, y0+y1*scale, (x2-x1)*scale, (y2-y1)*scale]
        assert all(abs(float(truth.get(k))-v) < 1e-5 for k, v in zip(["x", "y", "width", "height"], expected))
        x1, y1, x2, y2 = example["predicted_xyxy_pixel"]
        for k, value in zip(["x", "y", "width", "height"], [x0+x1*scale, y0+y1*scale, (x2-x1)*scale, (y2-y1)*scale]):
            predicted.set(k, f"{value:.5f}")
    node = root.find(f"{{{SVG}}}metadata")
    if node is None:
        node = ET.Element(f"{{{SVG}}}metadata")
        root.insert(2, node)
    node.text = json.dumps({"source": record_path.as_posix(),
                            "sha256": hashlib.sha256(record_path.read_bytes()).hexdigest(),
                            "examples": examples}, ensure_ascii=False)
    save(root, target)

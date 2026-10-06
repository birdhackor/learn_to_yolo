"""Check SVG derivatives against current saved figure elements and custom JSON boxes."""
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import runpy
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.reflow_result_figures import capstone_panel, video_panel, tracking_panel, label_custom_predictions
NS = "{http://www.w3.org/2000/svg}"
SCRATCH = ROOT / "work/technical-applications/figure-rebuilds"
SCRATCH.mkdir(parents=True, exist_ok=True)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def node_signature(node):
    return (node.tag, sorted(node.attrib.items()), node.text)

def images(root):
    return [node_signature(n) for n in root.iter(NS+"image")]

def drawn_boxes(root):
    return [node_signature(n) for n in root.iter(NS+"rect") if n.get("stroke")]

checks = {}
for lesson, func in [("17-capstone", capstone_panel), ("18-video", video_panel), ("19-tracking", tracking_panel)]:
    original = ROOT / f"docs/assets/diagrams/{lesson}.svg"
    published = ROOT / f"docs/assets/diagrams/{lesson}-readable.svg"
    rebuilt = SCRATCH / f"{lesson}.svg"
    func(original, rebuilt)
    assert rebuilt.read_bytes() == published.read_bytes()
    old, new = ET.parse(original).getroot(), ET.parse(published).getroot()
    assert images(old) == images(new)
    assert drawn_boxes(old) == drawn_boxes(new)
    checks[lesson] = {"source_sha256": sha(original), "derived_sha256": sha(published),
                      "derived_rebuild_byte_identical": True,
                      "embedded_image_bytes_and_attributes_identical": True,
                      "all_box_coordinates_styles_identical": True,
                      "image_count": len(images(new)), "drawn_box_count": len(drawn_boxes(new))}
    if lesson == "18-video":
        old_texts = [node_signature(n) for n in old.findall(NS+"text")]
        new_texts = [node_signature(n) for n in new.iter(NS+"text")]
        assert old_texts == new_texts
        assert len(old_texts) == 6
        checks[lesson]["all_timestamps_frame_numbers_counts_measured_times_identical"] = True
        checks[lesson]["captions"] = [v[2] for v in old_texts]
    elif lesson == "19-tracking":
        old_cell_texts = [node_signature(n) for n in old.findall(NS+"text") if n.get("x") != "20"]
        new_cell_texts = [node_signature(n) for group in new.findall(NS+"g") for n in group.findall(NS+"text")]
        assert old_cell_texts == new_cell_texts
        old_squares = [node_signature(n) for n in old.iter(NS+"rect") if n.get("fill") in {"#dc2626", "#2563eb", "#7c3aed"}]
        new_squares = [node_signature(n) for n in new.iter(NS+"rect") if n.get("fill") in {"#dc2626", "#2563eb", "#7c3aed"}]
        assert old_squares == new_squares
        checks[lesson]["all_frame_labels_ID_colors_positions_identical"] = True
        checks[lesson]["colored_ID_square_count"] = len(new_squares)
        checks[lesson]["max_age_preserved_in_title_and_description"] = "max_age=2" in new.find(NS+"title").text and "max_age=2" in new.find(NS+"desc").text
    elif lesson == "17-capstone":
        old_scores = [node_signature(n) for n in old.iter(NS+"text") if n.text and n.text[:2] in {"0:", "1:"}]
        new_scores = [node_signature(n) for n in new.iter(NS+"text") if n.text and n.text[:2] in {"0:", "1:"}]
        assert old_scores == new_scores
        checks[lesson]["all_class_score_labels_preserved"] = True

record_path = ROOT / "artifacts/checks/curriculum/custom-data-learning.json"
record = json.loads(record_path.read_text())
original = ROOT / "docs/assets/diagrams/08-custom-predictions-readable.svg"
published = ROOT / "docs/assets/diagrams/08-custom-predictions-labelled.svg"
rebuilt = SCRATCH / "08-custom-predictions-labelled.svg"
label_custom_predictions(original, record_path, rebuilt)
assert rebuilt.read_bytes() == published.read_bytes()
old, new = ET.parse(original).getroot(), ET.parse(published).getroot()
assert images(old) == images(new)
assert drawn_boxes(old) == drawn_boxes(new)
examples = record["validation_examples"]
png_checks = []
for im, example in zip(new.iter(NS+"image"), examples):
    data = base64.b64decode(im.get("href").split(",", 1)[1])
    digest = hashlib.sha256(data).hexdigest()
    assert digest == example["letterbox_png_sha256"]
    assert example["prediction"]["scores"] == sorted(example["prediction"]["scores"], reverse=True)
    png_checks.append({"validation_index": example["validation_index"], "sha256": digest,
                       "predictions": len(example["prediction"]["boxes"]),
                       "matches": example["matches"]})
paths = list(new.iter(NS+"path"))
boxes = [n for n in new.iter(NS+"rect") if n.get("stroke") == "#f97316"]
labels = [n.text for n in new.iter(NS+"text") if n.text and n.text.startswith("#")]
assert len(paths) == len(boxes) == 4
for path, box in zip(paths, boxes):
    endpoint = path.get("d").split("L",1)[1].split()
    assert [float(v) for v in endpoint] == [float(box.get("x"))+4, float(box.get("y"))]
assert labels[-4:] == ["#0", "#1", "#0", "#0"]
checks["08-custom-predictions"] = {"source_sha256": sha(original), "derived_sha256": sha(published),
    "record_sha256": sha(record_path), "derived_rebuild_byte_identical": True,
    "original_embedded_images_and_all_saved_boxes_preserved": True,
    "custom_geometry_verified_against_report_by_derivation_assertions": True,
    "all_numbered_connector_endpoints_on_corresponding_original_box": True,
    "numbers_follow_report_score_order_and_reset_per_image": True, "examples": png_checks}
tracking = runpy.run_path(str(ROOT / "lesson_cases/19-tracking.py"))
assert tracking["run"](False, 2) == (3, [[1,2],[1,2],[2,1],[2,1],[2],[2,3]])
assert tracking["run"](True, 2) == (0, [[1,2],[1,2],[1,2],[1,2],[1],[1,2]])
assert tracking["run"](False, 1)[0] == 3
assert tracking["run"](True, 1) == (1, [[1,2],[1,2],[1,2],[1,2],[1],[1,3]])
assert tracking["evaluate_detections"]() == (11,12,0)
checks["19-tracking"]["independent_matching_run_max_age_1_and_2_and_recall_checked"] = True
report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "passed": True,
          "script_sha256": sha(Path(__file__)), "checks": checks,
          "limits": "逐項檢查 XML、內嵌 PNG 位元組、座標、ID 與時間；沒有以 XML 檢查代替桌面或手機的瀏覽器視覺驗證。"}
target = ROOT / "reviews/clear-tutorial/full-review-2026-10-06/rechecks/technical-applications-figures.json"
target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
print(json.dumps(report, ensure_ascii=False, indent=2))

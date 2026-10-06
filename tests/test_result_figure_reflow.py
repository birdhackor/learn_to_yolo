"""Site layouts must keep alternate experiment outcomes and measured frame data."""
from pathlib import Path
import runpy
import xml.etree.ElementTree as ET

import torch
import json
import pytest
from PIL import Image

from scripts.reflow_result_figures import capstone_panel, label_custom_predictions, tracking_panel, video_panel

ROOT = Path(__file__).resolve().parents[1]
NS = {"s": "http://www.w3.org/2000/svg"}


def test_capstone_layout_keeps_all_measured_boxes_and_alternate_weight(tmp_path):
    before = ET.parse(ROOT / "docs/assets/diagrams/17-capstone.svg")
    title = before.getroot().find("s:g/s:text[@y='312']", NS)
    title.text = "只改一項：box weight 2"
    original, readable = tmp_path / "raw.svg", tmp_path / "readable.svg"
    before.write(original, encoding="unicode")
    capstone_panel(original, readable)
    after = ET.parse(readable).getroot()
    assert [x.get("href") for x in before.findall(".//s:image", NS)] == [
        x.get("href") for x in after.findall(".//s:image", NS)]
    def boxes(root):
        return [dict(x.attrib) for x in root.findall(".//s:rect", NS) if x.get("stroke")]
    assert boxes(before) == boxes(after)
    assert "只改一項：box weight 2" in [x.text for x in after.findall("s:text", NS)]
    assert "1:0.07" in [x.text for x in after.findall(".//s:text", NS)]


def test_custom_labels_reject_a_report_from_different_box_geometry(tmp_path):
    record = json.loads((ROOT / "artifacts/checks/curriculum/custom-data-learning.json").read_text())
    record["validation_examples"][0]["prediction"]["boxes"][0][0] += 1
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(record))
    output = tmp_path / "labelled.svg"
    with pytest.raises(AssertionError):
        label_custom_predictions(ROOT / "docs/assets/diagrams/08-custom-predictions-readable.svg", changed, output)
    assert not output.exists()


def test_tracking_layout_keeps_max_age_one_outcome(tmp_path):
    case = runpy.run_path(str(ROOT / "lesson_cases/19-tracking.py"))
    raw_switches, raw_ids = case["run"](False, max_age=1)
    motion_switches, motion_ids = case["run"](True, max_age=1)
    original, readable = tmp_path / "raw.svg", tmp_path / "readable.svg"
    case["save_panel"](raw_ids, motion_ids, raw_switches, motion_switches, 1, original)
    tracking_panel(original, readable)
    result = ET.parse(readable).getroot()
    groups = result.findall("s:g", NS)
    assert len(groups) == 12
    last = [node.text for node in groups[-1].findall("s:text", NS)]
    assert "A：ID1" in last and "B：ID3" in last
    assert "ID 切換 1 次" in [node.text for node in result.findall("s:text", NS)]
    assert "max_age=1" in result.find("s:title", NS).text


def test_video_layout_preserves_frames_scores_and_measured_times(tmp_path):
    case = runpy.run_path(str(ROOT / "lesson_cases/18-video.py"))
    results = [{"image": Image.new("RGB", (96, 64), (i*40, 0, 0)),
                "source_timestamp_s": i/10, "ms": {"total": 13.25+i},
                "prediction": {"boxes": torch.zeros(i, 4)}} for i in range(3)]
    original, readable = tmp_path / "raw.svg", tmp_path / "readable.svg"
    case["save_panel"](results, 10, original)
    video_panel(original, readable)
    before, after = ET.parse(original).getroot(), ET.parse(readable).getroot()
    assert [x.get("href") for x in before.findall("s:image", NS)] == [
        x.get("href") for x in after.findall(".//s:image", NS)]
    captions = [node.text for node in after.findall(".//s:text", NS)]
    assert "第 2 幀：來源時間 0.20 秒" in captions
    assert "處理 15.25 ms｜框 2 個" in captions
    assert len(after.findall("s:g", NS)) == 3

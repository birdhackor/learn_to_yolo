"""Rearrange recorded SVG panels for reading; preserve their actual images, IDs and timings.

The lesson programs keep producing the original horizontal panels. This site-only layout
change does not change the model, matching calculation, or the fixed-version notebooks.
"""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

SVG = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG)


def element(tag, **attributes):
    return ET.Element(f"{{{SVG}}}{tag}", {k: str(v) for k, v in attributes.items()})


def canvas(source, width, height, background):
    root = element("svg", xmlns=SVG, viewBox=f"0 0 {width} {height}", role="img")
    root.set("aria-labelledby", "t d")
    for name in ("title", "desc"):
        root.append(deepcopy(source.find(f"{{{SVG}}}{name}")))
    root.append(element("rect", width=width, height=height, fill=background))
    return root


def write(root, path):
    # All nodes are namespace-qualified, so ElementTree already emits xmlns.
    root.attrib.pop("xmlns", None)
    path.write_text(ET.tostring(root, encoding="unicode") + "\n")


def video_panel(source_path: Path, target: Path):
    source = ET.parse(source_path).getroot()
    panels = source.findall(f"{{{SVG}}}image")
    assert len(panels) == 3, "Expected the first, middle and last recorded frames"
    root = canvas(source, 440, 1090, "#0f172a")
    root.find(f"{{{SVG}}}desc").text += " 網頁將三格依序上下排列，圖片、分數與計時沿用該次執行。"
    texts = source.findall(f"{{{SVG}}}text")
    for row, image in enumerate(panels):
        x = image.get("x")
        captions = [text for text in texts if text.get("x") == x]
        assert len(captions) == 2, "Each recorded frame needs both captions"
        group = element("g", transform=f"translate(20,{20+350*row}) scale(1.3333333333) translate(-{x},-20)")
        for child in [image, *captions]:
            group.append(deepcopy(child))
        root.append(group)
    write(root, target)


def capstone_panel(source_path: Path, target: Path):
    source = ET.parse(source_path).getroot()
    original = source.find(f"{{{SVG}}}g")
    cells, current, headings = [], None, []
    for child in original:
        if child.tag == f"{{{SVG}}}text" and child.get("y") in {"34", "312", "596"}:
            headings.append(child.text)
            continue
        if child.tag == f"{{{SVG}}}image":
            current = [child]
            cells.append(current)
        elif current is not None:
            current.append(child)
    assert len(cells) == 9 and len(headings) == 3
    root = canvas(source, 440, 4170, "#0f172a")
    root.find(f"{{{SVG}}}desc").text = (
        "同樣四張驗證圖，依 #14、#5、#4、#7 排列；每張先放基準 box weight 5，"
        "緊接著放只把權重改成 10 的結果。綠框是 GT，橘框是 score 至少 0.25 的預測，"
        "標籤寫『類別:score』。最後保留基準模型在候選截斷門檻 0.05 下的背景誤報，"
        "用紫色虛線框標出。圖片、所有框與分數沿用原結果。"
    )

    def label(value, y, size=24):
        node = element("text", x=20, y=y, fill="white")
        node.set("font-family", "sans-serif")
        node.set("font-size", str(size))
        node.text = value
        root.append(node)

    label("相同驗證圖：逐張比較", 32)
    label("綠框＝GT；橘框＝預測", 64, 20)
    label("以下四對只畫 score ≥ 0.25", 88, 20)
    for index in (0, 4, 1, 5, 2, 6, 3, 7):
        children = cells[index]
        method, row = divmod(index, 4)
        top = 140 + 440 * (2 * row + method)
        label(headings[method].split("（")[0], top - 20, 22)
        old_x, old_y = children[0].get("x"), children[0].get("y")
        group = element("g", transform=f"translate(20,{top}) scale(1.7) translate(-{old_x},-{old_y})")
        group.set("fill", "white")
        group.set("font-family", "sans-serif")
        group.set("font-size", "20")
        for child in children:
            group.append(deepcopy(child))
        root.append(group)
    label("baseline 的背景誤報：validation #10", 3672, 22)
    children = cells[-1]
    old_x, old_y = children[0].get("x"), children[0].get("y")
    group = element("g", transform=f"translate(20,3700) scale(1.7) translate(-{old_x},-{old_y})")
    group.set("fill", "white")
    group.set("font-family", "sans-serif")
    group.set("font-size", "20")
    for child in children:
        if child.tag == f"{{{SVG}}}text" and child.get("x") == "235":
            continue
        group.append(deepcopy(child))
    root.append(group)
    for row, value in enumerate(["紫色虛線框：class 1，score 0.068", "和所有 GT 的最大 IoU：0.000", "此格畫 score ≥ 0.05 的全部候選。", "CPU 小型合成圖；AP 候選門檻 0.05。"]):
        label(value, 4040+row*32, 19)
    write(root, target)


def label_custom_predictions(source_path: Path, record_path: Path, target: Path):
    """Add box-to-caption numbers after checking geometry against the saved report."""
    root = ET.parse(source_path).getroot()
    report = json.loads(record_path.read_text())
    group = root.find(f"{{{SVG}}}g")
    examples = report["validation_examples"]
    cells, current = [], None
    for child in list(group):
        if child.tag == f"{{{SVG}}}image":
            current = [child]
            cells.append(current)
        elif current is not None:
            current.append(child)
    assert len(cells) == len(examples)
    for children, example in zip(cells, examples):
        image = children[0]
        x0, y0 = float(image.get("x")), float(image.get("y"))
        scale = float(image.get("width"))/64
        boxes = [n for n in children if n.tag == f"{{{SVG}}}rect" and n.get("stroke") == "#f97316"]
        predictions = example["prediction"]
        assert len(boxes) == len(predictions["boxes"])
        captions = [n for n in children if n.tag == f"{{{SVG}}}text" and n.text and "：" in n.text and ("TP" in n.text or "FP" in n.text)]
        assert len(captions) == len(boxes)
        for j, (box, coordinates, caption) in enumerate(zip(boxes, predictions["boxes"], captions)):
            x1, y1, x2, y2 = coordinates
            expected = [x0+x1*scale, y0+y1*scale, (x2-x1)*scale, (y2-y1)*scale]
            assert all(abs(float(box.get(k))-v) < 1e-5 for k, v in zip(["x", "y", "width", "height"], expected))
            caption.text = f"#{j} " + caption.text
            tx = max(x0+4, min(float(box.get("x")), x0+180))
            ty = max(y0+4, float(box.get("y"))-30-j*28)
            group.append(element("path", d=f"M {tx+18} {ty+25} L {float(box.get('x'))+4} {box.get('y')}", stroke="#f97316", **{"stroke-width": 2}))
            group.append(element("rect", x=tx, y=ty, width=38, height=26, rx=3, fill="#0f172a"))
            label = element("text", x=tx+3, y=ty+21, fill="white")
            label.text = f"#{j}"
            group.append(label)
    root.find(f"{{{SVG}}}desc").text += " 每個預測框新增 # 編號與連線，對應圖下相同編號；框和圖片未改。"
    write(root, target)


def tracking_panel(source_path: Path, target: Path):
    source = ET.parse(source_path).getroot()
    cells, current = [], None
    for child in source:
        if child.tag == f"{{{SVG}}}rect" and child.get("width") == "155" and child.get("height") == "120":
            current = [child]
            cells.append(current)
        elif child.tag == f"{{{SVG}}}text" and child.get("x") == "20":
            current = None  # Method titles and footnotes are outside the frame cells.
        elif current is not None:
            current.append(child)
    assert len(cells) == 12, "Expected six frames for each of the two recorded methods"
    titles = [text for text in source.findall(f"{{{SVG}}}text") if text.get("font-size") == "22"]
    assert len(titles) == 2
    root = canvas(source, 480, 1480, "#f8fafc")
    root.find(f"{{{SVG}}}desc").text += " 網頁上下兩組各有六格，每组依左至右、上至下排列；只改版面。".replace("组", "組")
    for method in range(2):
        offset = method * 640
        first, count = titles[method].text.split("：")
        # Short title and subtitle avoid squeezing the original long title.
        short = first.split("（")[0]
        for y, value in [(30+offset, short), (62+offset, count)]:
            text = element("text", x=20, y=y, fill="#0f172a")
            text.set("font-family", "sans-serif")
            text.set("font-size", "24")
            text.text = value
            root.append(text)
        for frame in range(6):
            children = cells[method*6+frame]
            old_x, old_y = children[0].get("x"), children[0].get("y")
            new_x = 20+(frame % 2)*230
            new_y = 90+(frame//2)*176+offset
            group = element("g", transform=f"translate({new_x},{new_y}) scale(1.3) translate(-{old_x},-{old_y})")
            for child in children:
                group.append(deepcopy(child))
            root.append(group)
    notes = ["方塊顏色＝分配到的 track ID。", "A、B 只用來評分；第 4 幀 B 漏檢。",
             "兩方法吃同一份偵測框。", "水平位置是真的；上下分行只為標籤。", "配對時 A、B 的 y 座標相同。"]
    for row, note in enumerate(notes):
        text = element("text", x=20, y=1310+row*30, fill="#0f172a")
        text.set("font-family", "sans-serif")
        text.set("font-size", "22")
        text.text = note
        root.append(text)
    write(root, target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["capstone", "video", "tracking", "custom"], required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record", type=Path, help="saved 1600-step JSON, required for custom box labels")
    args = parser.parse_args()
    if args.kind == "custom":
        if args.record is None:
            parser.error("--kind custom needs --record; stale source geometry is rejected")
        label_custom_predictions(args.source, args.record, args.output)
    else:
        {"capstone": capstone_panel, "video": video_panel, "tracking": tracking_panel}[args.kind](args.source, args.output)
    print(f"Generated {args.output} from {args.source}")

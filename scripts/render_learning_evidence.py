"""Render SVG figures using actual recorded CPU training outputs."""
import json
import base64
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'docs/assets/diagrams'
report = json.loads((ROOT / 'artifacts/checks/grid-learning.json').read_text())
history = json.loads((ROOT / report['history_path']).read_text())
DEST.mkdir(parents=True, exist_ok=True)
colors = {'total': '#1d4ed8', 'box': '#dc2626', 'objectness': '#059669', 'classification': '#7c3aed'}
maximum = max(row[key] for row in history for key in colors)
svg = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 470" role="img" aria-labelledby="title desc">',
       '<title id="title">160 次 CPU 參數更新的實測 loss 曲線</title>',
       '<desc id="desc">藍線為 total，紅線為 box，綠線為 objectness，紫線為 classification。這是合成矩形任務的單次實測，不是原版 YOLO 的效能。</desc>',
       '<style>text{font-family:system-ui,sans-serif;font-size:16px;fill:#172554}</style>',
       '<rect width="900" height="470" fill="white"/>',
       '<text x="55" y="30" font-weight="bold">固定合成資料：32 張訓練圖、batch 8、seed 7</text>']
for i in range(5):
    value = maximum * i / 4
    y = 370 - 310 * i / 4
    svg += [f'<path d="M75 {y}H860" stroke="#e2e8f0"/>', f'<text x="12" y="{y+5}">{value:.2f}</text>']
for key, color in colors.items():
    points = ' '.join(f"{75+785*(row['step']-1)/(len(history)-1):.1f},{370-310*row[key]/maximum:.1f}" for row in history)
    svg += [f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>']
for value in [1, 40, 80, 120, 160]:
    x = 75 + 785 * (value - 1) / 159
    svg += [f'<text x="{x-10}" y="396">{value}</text>']
for i, (key, color) in enumerate(colors.items()):
    x = 80 + i * 195
    svg += [f'<path d="M{x} 420h24" stroke="{color}" stroke-width="3"/>', f'<text x="{x+30}" y="426">{key}</text>']
svg += ['<text x="55" y="456">橫軸：optimizer 更新次數；縱軸：loss。訓練下降仍需另看獨立資料。</text>', '</svg>']
(DEST / 'grid-learning-curve.svg').write_text('\n'.join(svg) + '\n')
svg = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 725" role="img" aria-labelledby="title desc">',
       '<title id="title">CPU 短訓練後，四張獨立 validation 圖的真值與模型框</title>',
       '<desc id="desc">原始圖片以 PNG 呈現，綠色虛線是真值，橙色實線是模型預測。框與分數來自固定 160 步實測，包含框邊緣偏差。</desc>',
       '<style>text{font-family:system-ui,sans-serif;font-size:16px;fill:#172554}</style>',
       '<rect width="900" height="725" fill="white"/>',
       '<text x="24" y="30" font-weight="bold">獨立 validation 圖：綠色虛線 GT，橙色實線 prediction</text>']
for i, entry in enumerate(report['validation_examples']):
    x, y = 34 + (i % 2) * 445, 74 + (i // 2) * 315
    filename = f'grid-validation-{i:02d}.png'
    shutil.copyfile(ROOT / entry['image_path'], DEST / filename)
    image_data = base64.b64encode((ROOT / entry['image_path']).read_bytes()).decode('ascii')
    svg += [f'<text x="{x}" y="{y-14}">圖片 {entry["index"]}（未參與參數更新）</text>',
            f'<image href="data:image/png;base64,{image_data}" x="{x}" y="{y}" width="248" height="248" style="image-rendering:pixelated"/>']
    for group, color, dash in [('target', '#22c55e', 'stroke-dasharray="7 4"'), ('prediction', '#fb923c', '')]:
        for box in entry[group]['boxes']:
            x1,y1,x2,y2 = box
            svg += [f'<rect x="{x+x1*248/64:.2f}" y="{y+y1*248/64:.2f}" width="{(x2-x1)*248/64:.2f}" height="{(y2-y1)*248/64:.2f}" fill="none" stroke="{color}" stroke-width="2.5" {dash}/>']
    for j, (label, score) in enumerate(zip(entry['prediction']['labels'], entry['prediction']['scores'])):
        svg += [f'<text x="{x+262}" y="{y+35+j*50}">pred {j}</text>', f'<text x="{x+262}" y="{y+55+j*50}">class {label} / {score:.3f}</text>']
svg += ['<text x="24" y="715">這是紅／藍矩形任務的早期證據。單一 seed、小資料集，不能代表真實照片的效果。</text>', '</svg>']
(DEST / 'grid-learning-predictions.svg').write_text('\n'.join(svg) + '\n')
print('Rendered actual CPU loss curve and validation predictions as accessible SVGs.')

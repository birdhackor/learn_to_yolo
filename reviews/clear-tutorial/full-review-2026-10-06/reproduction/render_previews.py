from pathlib import Path
import json
import os
import xml.etree.ElementTree as ET
from playwright.sync_api import sync_playwright
os.environ['PLAYWRIGHT_BROWSERS_PATH']='/workspace/work/browser-runtime'
WORK=Path('/workspace/work/full-curriculum-review')
manifest=json.loads((WORK/'manifest.json').read_text())
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1600,'height':1200},device_scale_factor=1,locale='zh-TW')
    count=0
    for item in manifest['figures'].values():
        source=Path(item['frozen']);preview=Path(item['preview'])
        if source.suffix!='.svg':continue
        preview.parent.mkdir(parents=True,exist_ok=True)
        page.goto(source.as_uri())
        page.locator('svg').first.screenshot(path=str(preview))
        count+=1
    browser.close()
print(json.dumps({'svg_previews_rendered':count}))

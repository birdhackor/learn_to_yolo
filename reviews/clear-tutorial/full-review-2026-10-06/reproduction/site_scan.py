from pathlib import Path
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from functools import partial
from datetime import datetime, timezone
import hashlib
import json
import os
import argparse
from playwright.sync_api import sync_playwright

os.environ['PLAYWRIGHT_BROWSERS_PATH']='/workspace/work/browser-runtime'
ROOT=Path('/workspace/learn_to_yolo')
WORK=Path('/workspace/work/full-curriculum-review')
p=argparse.ArgumentParser();p.add_argument('--stage',required=True);a=p.parse_args()
dest=WORK/f'site-{a.stage}';dest.mkdir(exist_ok=True)
class Handler(SimpleHTTPRequestHandler):
    def log_message(self,*args):pass
server=ThreadingHTTPServer(('127.0.0.1',0),partial(Handler,directory=str(ROOT/'site')))
Thread(target=server.serve_forever,daemon=True).start()
origin=f'http://127.0.0.1:{server.server_port}'
pages=['','learning-path/','glossary/','status/']+['lessons/'+s['id']+'/' for s in json.loads((ROOT/'section-map.json').read_text())['sections'] if s['kind']=='lesson']
results=[]
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
    for mode,width,height in [('desktop',1280,800),('mobile',390,844)]:
        context=browser.new_context(viewport={'width':width,'height':height},locale='zh-TW',device_scale_factor=1)
        context.route('**/*',lambda route:route.continue_() if route.request.url.startswith(origin) else route.abort())
        page=context.new_page()
        for path in pages:
            errors=[]
            on_error=lambda err:errors.append(str(err))
            page.on('pageerror',on_error)
            page.goto(origin+'/'+path,wait_until='networkidle')
            page.wait_for_function("!window.MathJax || !window.MathJax.startup || !!window.MathJax.startup.document", timeout=15000)
            page.evaluate("async () => { if (window.MathJax?.startup?.promise) await window.MathJax.startup.promise; }")
            info=page.evaluate('''() => ({title:document.title, horizontalOverflow:document.documentElement.scrollWidth-innerWidth, mathErrors:[...document.querySelectorAll('.katex-error, mjx-merror, [data-mjx-error]')].map(x=>x.textContent), typesetMath:document.querySelectorAll('mjx-container').length, images:[...document.querySelectorAll('.md-content img')].map(x=>({src:x.getAttribute('src'),alt:x.alt,loaded:x.complete&&x.naturalWidth>0,width:x.getBoundingClientRect().width,height:x.getBoundingClientRect().height,naturalWidth:x.naturalWidth,naturalHeight:x.naturalHeight})), contentText:document.querySelector('.md-content')?.innerText.slice(0,1400)})''')
            info.update(path=path or '/',mode=mode,errors=errors)
            results.append(info)
            # Capture the actual introduction and figure regions for review.
            name=(path.strip('/').replace('/','__') or 'index')+'-'+mode
            page.screenshot(path=str(dest/(name+'-opening.png')))
            for index,img in enumerate(page.locator('.md-content img').all()):
                if not img.is_visible():continue
                if img.bounding_box():img.screenshot(path=str(dest/(name+f'-figure-{index:02d}.png')))
            page.remove_listener('pageerror',on_error)
        context.close()
    browser.close()
server.shutdown()
built_files=[ROOT/'site'/path/'index.html' for path in pages]
built_files += [f for f in (ROOT/'site/assets/diagrams').iterdir() if f.is_file()]
report={'stage':a.stage,'captured_at':datetime.now(timezone.utc).isoformat(),'base_commit':os.popen('git -C /workspace/learn_to_yolo rev-parse HEAD').read().strip(),'built_files_sha256':{f.relative_to(ROOT/'site').as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in built_files},'scope':'56 actual built pages at desktop and mobile, introductions and full image regions; DOM checks alone are not a complete visual verdict. Third-party requests blocked, local MathJax/styles/scripts retained. base_commit is the Git base; built file fingerprints identify the actual rendered working tree.','results':results}
(dest/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'pages':len(pages),'viewport_page_checks':len(results),'unloaded_images':sum(not i['loaded'] for r in results for i in r['images']),'horizontal_overflow_pages':sum(r['horizontalOverflow']>1 for r in results),'math_error_pages':sum(bool(r['mathErrors']) for r in results),'runtime_error_pages':sum(bool(r['errors']) for r in results)},indent=2))

from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,threading,tomllib,subprocess,sys
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from functools import partial
from playwright.sync_api import sync_playwright
root=Path('/workspace/learn_to_yolo');out=root/'reviews/formatting/section-prefixes-2026-10-08/layout';out.mkdir(parents=True,exist_ok=True)
class Quiet(SimpleHTTPRequestHandler):
 def log_message(self,*args):pass
base=sys.argv[1] if len(sys.argv)>1 else 'http://127.0.0.1:8914/'
server=None
if base.startswith('http://127.0.0.1'):
 server=ThreadingHTTPServer(('127.0.0.1',8914),partial(Quiet,directory=str(root/'site')));threading.Thread(target=server.serve_forever,daemon=True).start()
items={'04-localization':'B.4.1','09-anchors':'C.9.1','17-capstone':'D.17','21-patches':'E.21.1'};records=[]
with sync_playwright() as pw:
 b=pw.chromium.launch(executable_path='/usr/bin/chromium',args=['--no-sandbox'])
 for width in (390,1280):
  p=b.new_page(viewport={'width':width,'height':844})
  for sid,prefix in items.items():
   response=p.goto(base+'lessons/'+sid+'/',wait_until='networkidle');p.evaluate('async()=>{await document.fonts.ready;if(window.MathJax?.startup?.promise)await window.MathJax.startup.promise}')
   title=p.locator('article h1').inner_text();assert title.startswith(prefix+' '),(sid,title)
   nav=p.locator('.md-nav a').all_text_contents();assert any(t.strip().startswith(prefix+' ') for t in nav),(sid,nav)
   assert p.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),sid
   file=sid+'-'+str(width)+'.png';p.screenshot(path=str(out/file));records.append({'id':sid,'width':width,'title':title,'nav_prefix_present':True,'http_status':response.status,'file':file,'sha256':hashlib.sha256((out/file).read_bytes()).hexdigest()})
  p.close()
 b.close()
if server:server.shutdown();server.server_close()
(out/'index.json').write_text(json.dumps({'captured_at_utc':datetime.now(timezone.utc).isoformat(),'base':base,'records':records,'manual_review':'capture/DOM alone not manual visual inspection'},ensure_ascii=False,indent=2)+'\n');print(len(records),'layout views passed')

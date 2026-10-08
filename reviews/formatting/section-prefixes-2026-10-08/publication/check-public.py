from pathlib import Path
from urllib.request import Request,urlopen
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json,hashlib,sys,tomllib,re,subprocess

root=Path('/workspace/learn_to_yolo');sys.path.insert(0,str(root))
from scripts.verify_release import article,navigation,site_path
cfg=tomllib.loads((root/'zensical.toml').read_text())['project'];base=cfg['site_url']
out=root/'reviews/formatting/section-prefixes-2026-10-08/publication'
def fetch(url):
    with urlopen(Request(url,headers={'User-Agent':'Codex-public-verification','Cache-Control':'no-cache'}),timeout=20) as r:return r.status,r.read()
def check_page(page):
    route=site_path(page);local=article((root/'site'/route/'index.html').read_text());status,body=fetch(base+route)
    return {'route':route,'http_status':status,'article_matches_build':status==200 and article(body.decode())==local}
def check_file(path):
    relative=str(path.relative_to(root/'site'));status,body=fetch(base+relative)
    return {'path':relative,'http_status':status,'bytes_match_build':status==200 and body==path.read_bytes()}
def check_notebook(name):
    page=(root/'site/lessons'/name/'index.html').read_text();ref=re.search(r'colab.research.google.com/github/birdhackor/learn_to_yolo/blob/([^/]+)/notebooks/',page).group(1)
    status,body=fetch(f'https://raw.githubusercontent.com/birdhackor/learn_to_yolo/{ref}/notebooks/{name}.ipynb')
    published=json.loads(body);local=json.loads((root/'notebooks'/(name+'.ipynb')).read_text())
    codes=lambda d:[''.join(c['source']) for c in d['cells'] if c['cell_type']=='code']
    return {'id':name,'ref':ref,'http_status':status,'all_code_cells_match':codes(published)==codes(local)}
with ThreadPoolExecutor(max_workers=8) as ex:
    pages=list(ex.map(check_page,navigation(cfg['nav'])))
    files=list(ex.map(check_file,[root/'site/assets/stylesheets/extra.css',*[p for p in sorted((root/'site/assets/diagrams').iterdir()) if p.suffix.lower() in {'.svg','.png','.jpg','.jpeg','.webp'}]]))
    notebooks=list(ex.map(check_notebook,[r['id'] for r in json.loads((root/'section-map.json').read_text())['sections'] if r['kind']=='lesson']))
d={'verified_at_utc':datetime.now(timezone.utc).isoformat(),'website':base,'build_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'deployment':json.loads((out/'deployment.json').read_text()),'mode':'Anonymous public HTTP comparison with local strict build; no Colab login/runtime execution','pages':pages,'files':files,'notebooks':notebooks}
d['passed']=all(p['article_matches_build'] for p in pages) and all(f['bytes_match_build'] for f in files) and all(n['all_code_cells_match'] for n in notebooks)
(out/'public-check.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'passed':d['passed'],'pages':len(pages),'files':len(files),'notebooks':len(notebooks),'failures':[p for p in pages if not p['article_matches_build']]+[f for f in files if not f['bytes_match_build']]+[n for n in notebooks if not n['all_code_cells_match']]}))
raise SystemExit(0 if d['passed'] else 1)

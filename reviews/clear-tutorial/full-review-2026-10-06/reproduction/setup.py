from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import shutil
import subprocess
from urllib.parse import unquote

ROOT=Path('/workspace/learn_to_yolo')
WORK=Path('/workspace/work/full-curriculum-review')
DATA=ROOT/'reviews/clear-tutorial/full-review-2026-10-06'
assert not DATA.exists(), 'Do not overwrite an earlier review.'
DATA.mkdir(parents=True)
WORK.mkdir(parents=True,exist_ok=True)
lessons=[s for s in json.loads((ROOT/'section-map.json').read_text())['sections'] if s['kind']=='lesson']
groups={
 'foundations':{'targets':list(range(0,6)),'prerequisites':[]},
 'detector':{'targets':list(range(6,19)),'prerequisites':list(range(0,6))},
 'evolution_a':{'targets':list(range(19,30)),'prerequisites':list(range(0,19))},
 'evolution_b':{'targets':list(range(30,38)),'prerequisites':list(range(0,30))},
 'applications':{'targets':list(range(38,42)),'prerequisites':list(range(0,19))},
 'vision':{'targets':list(range(42,52)),'prerequisites':list(range(0,11))+[33]},
}
assert sorted(i for g in groups.values() for i in g['targets'])==list(range(52))
sources={}
figures={}
pages={}
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def freeze(p):
    p=p.resolve();rel=p.relative_to(ROOT).as_posix()
    dest=DATA/'frozen'/rel
    dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():shutil.copy2(p,dest)
    sources[rel]=digest(p)
    return dest
def units(rel,page_id):
    p=ROOT/rel;freeze(p)
    text=p.read_text().split('\n<!-- curriculum-evidence:start -->')[0].rstrip()+'\n'
    lines=text.splitlines(keepends=True)
    cuts=[0]+[i for i,l in enumerate(lines) if i>0 and l.startswith('## ')]
    cuts.append(len(lines));result=[]
    for number,(a,b) in enumerate(zip(cuts,cuts[1:])):
        body=''.join(lines[a:b]);imgs=[]
        for target in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',body):
            target=target.split('#')[0]
            if target.startswith(('https://','http://')):continue
            img=(p.parent/unquote(target)).resolve()
            if not img.exists():continue
            dest=freeze(img);name=img.relative_to(ROOT).as_posix()
            preview=DATA/'previews'/(name.replace('/','__')+'.png') if img.suffix=='.svg' else dest
            figures[name]={'frozen':str(dest),'preview':str(preview)}
            imgs.append({'source':name,'path':str(preview),'kind':'raw source preview; website not verified'})
        result.append({'id':f'{page_id}/{number:02d}','page_id':page_id,'source':rel,'line_start':a+1,'line_end':b,'content':body,'images':imgs,'page_end':number==len(cuts)-2})
    return result
navigation=[]
for rel in ['docs/index.md','docs/learning-path.md']:
    navigation+=units(rel,Path(rel).stem)
for s in lessons:pages[s['id']]=units(s['page'],s['id'])
# Freeze related student and maintenance pages; do not reveal them early to readers.
for p in (ROOT/'docs').rglob('*.md'):freeze(p)
for rel in ['.agents/skills/clear-tutorial/SKILL.md','.agents/skills/clear-tutorial/references/review-protocol.md','section-map.json','AGENTS.md']:
    freeze(ROOT/rel)
for group,settings in groups.items():
    packets=[dict(u,target=False) for u in navigation]
    for i in settings['prerequisites']:
        packets += [dict(u,target=False) for u in pages[lessons[i]['id']]]
    for i in settings['targets']:
        packets += [dict(u,target=True) for u in pages[lessons[i]['id']]]
    (WORK/f'{group}-packets.json').write_text(json.dumps(packets,ensure_ascii=False,indent=2)+'\n')
    (WORK/group).mkdir()
    (WORK/group/'state.json').write_text(json.dumps({'cursor':0,'awaiting':False,'first_use_pages':[]})+'\n')
    settings.update(target_ids=[lessons[i]['id'] for i in settings['targets']],prerequisite_ids=[lessons[i]['id'] for i in settings['prerequisites']],packet_count=len(packets))
manifest={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),'created_at':datetime.now(timezone.utc).isoformat(),'groups':groups,'source_hashes':sources,'figures':figures,'scope':'All 52 authored lessons, navigation and related student/maintenance pages. Generated trailing execution appendices excluded from first-reader packets; original authored code and folded notes retained.','isolation':'Instructions and sequential gate, not technical filesystem access control.'}
(DATA/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
(WORK/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
(DATA/'first-read').mkdir()
(DATA/'rechecks').mkdir()
print(json.dumps({'source_commit':manifest['source_commit'],'groups':{g:{'targets':len(s['target_ids']),'prerequisite_pages':len(s['prerequisite_ids']),'units':s['packet_count']} for g,s in groups.items()},'frozen_files':len(sources),'figures':len(figures)},indent=2))

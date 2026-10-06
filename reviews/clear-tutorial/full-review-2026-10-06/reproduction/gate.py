from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json

WORK=Path('/workspace/work/full-curriculum-review')
DATA=Path('/workspace/learn_to_yolo/reviews/clear-tutorial/full-review-2026-10-06')
groups=['foundations','detector','evolution_a','evolution_b','applications','vision']
p=argparse.ArgumentParser();p.add_argument('action',choices=['next','record','status']);p.add_argument('reader',choices=groups);p.add_argument('--file',type=Path);a=p.parse_args()
packets=json.loads((WORK/f'{a.reader}-packets.json').read_text())
state_path=WORK/a.reader/'state.json';state=json.loads(state_path.read_text())
def append(name,obj):
    with (DATA/'first-read'/f'{a.reader}{name}.jsonl').open('a') as f:f.write(json.dumps(obj,ensure_ascii=False)+'\n')
if a.action=='status':
    print(json.dumps({'completed':state['cursor'],'total':len(packets),'awaiting':state['awaiting']},ensure_ascii=False));raise SystemExit
if a.action=='next':
    assert not state['awaiting'],'Save the current original note before another disclosure.'
    if state['cursor']==len(packets):print('READING COMPLETE');raise SystemExit
    packet=packets[state['cursor']];state['awaiting']=True;state_path.write_text(json.dumps(state)+'\n')
    append('-disclosures',{'unit_id':packet['id'],'disclosed_at':datetime.now(timezone.utc).isoformat(),'content_sha256':hashlib.sha256(packet['content'].encode()).hexdigest()})
    print(json.dumps(packet,ensure_ascii=False,indent=2));raise SystemExit
assert state['awaiting'] and a.file,'A disclosed unit and --file are required.'
packet=packets[state['cursor']];note=json.loads(a.file.read_text())
assert note['unit_id']==packet['id']
assert all(k in note for k in ['understanding','issues','visual_status','needed_visuals','checkpoints'])
for issue in note['issues']:
    assert all(k in issue for k in ['quote','reason','severity','needed','type'])
    assert issue['quote'] in packet['content'],'Issue quotation must be verbatim in current unit.'
    assert issue['severity'] in ['blocking','burden','optional']
visible={u['id']:u['content'] for u in packets[:state['cursor']+1]}
for checkpoint in note['checkpoints']:
    assert checkpoint['timing'] in ['first_use','scope_overview','page_end']
    assert len(checkpoint['answers'])==4
    for answer in checkpoint['answers']:
        assert all(k in answer for k in ['focus','answer','claims','unknown'])
        assert answer['focus'] and answer['answer']
        for c in answer['claims']:
            assert all(k in c for k in ['claim','status','unit_id','quote'])
            assert c['status'] in ['explicit_body','prior_or_navigation','inference']
            assert c['unit_id'] in visible and c['quote'] and c['quote'] in visible[c['unit_id']]
            if c['status']=='inference':assert c.get('inference')
    if checkpoint['timing'] in ['first_use','scope_overview']:
        if packet['page_id'] not in state['first_use_pages']:state['first_use_pages'].append(packet['page_id'])
if packet['target'] and packet['page_end']:
    assert any(c['timing']=='page_end' for c in note['checkpoints']),'Four page-end answers required.'
    assert packet['page_id'] in state['first_use_pages'],'Save a first-use or justified scope-overview checkpoint.'
append('',{'source':{k:v for k,v in packet.items() if k!='content'},'recorded_at':datetime.now(timezone.utc).isoformat(),'notes':note})
state['cursor']+=1;state['awaiting']=False;state_path.write_text(json.dumps(state)+'\n')
print(json.dumps({'saved':packet['id'],'completed':state['cursor'],'total':len(packets)},ensure_ascii=False))

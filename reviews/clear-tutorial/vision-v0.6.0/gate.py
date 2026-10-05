"""Sequential disclosure for this review; the shared filesystem is not isolated."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

LOGS = Path(__file__).resolve().parent / 'first-read'
ROOT = LOGS.parents[3]
RUN = ROOT / 'artifacts/runs/vision-firstread'
parser = argparse.ArgumentParser()
parser.add_argument('action', choices=['next', 'record', 'status'])
parser.add_argument('reader', choices=['vit', 'dino'])
parser.add_argument('--file', type=Path)
args = parser.parse_args()
packets = json.loads((RUN / f'{args.reader}-packets.json').read_text())
state_path = RUN / f'{args.reader}-state.json'
state = json.loads(state_path.read_text())
if args.action == 'status':
    print(json.dumps({'completed': state['cursor'], 'total': len(packets),
                      'awaiting_record': state['awaiting_record']}))
    raise SystemExit
if args.action == 'next':
    if state['awaiting_record']:
        raise SystemExit('Record the disclosed unit before requesting another.')
    if state['cursor'] == len(packets):
        print('READING COMPLETE')
        raise SystemExit
    state['awaiting_record'] = True
    state_path.write_text(json.dumps(state))
    print(json.dumps(packets[state['cursor']], ensure_ascii=False, indent=2))
    raise SystemExit
if not state['awaiting_record'] or args.file is None:
    raise SystemExit('A disclosed unit and --file are required.')
packet = packets[state['cursor']]
note = json.loads(args.file.read_text())
assert note['unit_id'] == packet['id']
assert all(key in note for key in ['understanding', 'elements', 'next_change_and_basis',
                                 'issues', 'needed_visuals', 'visual_status', 'variation'])
for issue in note['issues']:
    assert all(key in issue for key in ['quote', 'reason', 'severity', 'needed'])
    assert issue['quote'] in packet['content']
    assert issue['severity'] in ['blocking', 'burden', 'optional']
record = {'disclosed_source': {k: v for k, v in packet.items() if k != 'content'},
          'recorded_at_utc': datetime.now(timezone.utc).isoformat(), 'reader_notes': note}
LOGS.mkdir(parents=True, exist_ok=True)
with (LOGS / f'{args.reader}.jsonl').open('a') as stream:
    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
state['cursor'] += 1
state['awaiting_record'] = False
state_path.write_text(json.dumps(state))
print(json.dumps({'saved': packet['id'], 'completed': state['cursor'], 'total': len(packets)}))

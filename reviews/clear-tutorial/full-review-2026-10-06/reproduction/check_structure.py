"""Check frozen bytes, disclosure order and unchanged code; no semantic verdict."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[4]
ROUND = Path(__file__).resolve().parents[1]
manifest = json.loads((ROUND / 'manifest.json').read_text())
base = manifest['source_commit']
for name, expected in manifest['source_hashes'].items():
    original = subprocess.check_output(['git', 'show', f'{base}:{name}'], cwd=ROOT)
    assert hashlib.sha256(original).hexdigest() == expected, name
    assert hashlib.sha256((ROUND / 'frozen' / name).read_bytes()).hexdigest() == expected, name

units = checkpoints = answers = claims = 0
for name in manifest['groups']:
    rows = [json.loads(line) for line in (ROUND / 'first-read' / f'{name}.jsonl').read_text().splitlines()]
    disclosures = [json.loads(line) for line in (ROUND / 'first-read' / f'{name}-disclosures.jsonl').read_text().splitlines()]
    assert len(rows) == len(disclosures)
    for index, row in enumerate(rows):
        assert row['source']['id'] == disclosures[index]['unit_id']
        assert row['recorded_at'] >= disclosures[index]['disclosed_at']
        if index + 1 < len(rows):
            assert row['recorded_at'] < disclosures[index + 1]['disclosed_at']
        units += 1
        for checkpoint in row['notes']['checkpoints']:
            checkpoints += 1
            assert len(checkpoint['answers']) == 4
            answers += len(checkpoint['answers'])
            claims += sum(len(answer['claims']) for answer in checkpoint['answers'])

notebooks = 0
for path in (ROOT / 'notebooks').glob('*.ipynb'):
    old = json.loads(subprocess.check_output(['git', 'show', f'{base}:{path.relative_to(ROOT).as_posix()}'], cwd=ROOT))
    current = json.loads(path.read_text())
    assert [cell['source'] for cell in old['cells'] if cell['cell_type'] == 'code'] == [
        cell['source'] for cell in current['cells'] if cell['cell_type'] == 'code'], path
    notebooks += 1

assert (units, checkpoints, answers, claims) == (995, 212, 848, 1680)
print(json.dumps({'frozen_sources': len(manifest['source_hashes']), 'units': units,
                  'checkpoints': checkpoints, 'answers': answers, 'claims': claims,
                  'unchanged_notebook_code': notebooks,
                  'scope': 'Byte/structure checks only; semantic citation support and detection completeness require separate review.'}))

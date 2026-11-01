"""Compare les cles numeriques conservees a la cle certifiee, sans supposer la faisabilite."""
import hashlib
import json
from collections import Counter
from fractions import Fraction as F
from pathlib import Path

P=Path(__file__).resolve().parent
historical=P.parents[1]/'04_Code_et_experiences/results/v1/stress_profiles.jsonl'
verified=P/'verification_240.json'
data=json.loads(verified.read_text())
assert not data['errors'] and data['counts']['checked']==240
keys={(r['id'],r['model']):tuple(map(F,r['key'])) for r in data['records']}
reports={m:Counter() for m in ('lcs','enum','scip','eps')}
for r in map(json.loads,historical.read_text().splitlines()):
    if r.get('family')!='adjacent' or r['method'] not in reports: continue
    reference=keys[(r['id'],r['model'])]; val=r.get('key')
    if val is None or len(val)!=len(reference): reports[r['method']]['missing_or_partial']+=1; continue
    val=tuple(map(F,val))
    reports[r['method']]['above' if val>reference else 'equal' if val==reference else 'below']+=1
assert all(sum(c.values())==60 for c in reports.values())
print(json.dumps({'scope':'60 adjacent-profile pairs; recorded numerical keys, no feasibility proof',
                  'counts':{k:dict(v) for k,v in reports.items()},
                  'historical_sha256':hashlib.sha256(historical.read_bytes()).hexdigest(),
                  'verification_sha256':hashlib.sha256(verified.read_bytes()).hexdigest()},indent=2))

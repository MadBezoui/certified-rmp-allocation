"""Conserve un certificat par cas dans un nouveau fichier, sans modifier les sources."""
import hashlib
import json
from pathlib import Path
P=Path(__file__).parent
initial=[json.loads(line) for line in (P/'tous_sans_relaxation.jsonl').open()]
enhanced={r['id']:r for r in map(json.loads,(P/'dual_optimise.jsonl').open())}
out=P/'certificats_55.jsonl'
if out.exists(): raise SystemExit('output exists')
assert len(initial)==55 and len(enhanced)==24 and len({r['id'] for r in initial})==55
with out.open('x') as f:
    for r in initial:
        direct=r['new']['status']=='OPTIMAL_VERIFIED'
        cert=r['new']['certificate'] if direct else enhanced[r['id']]['certificate']
        raw=json.dumps(cert,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        f.write(json.dumps({'id':r['id'],'model':r['model'],'instance_sha':r['instance_sha'],
                           'historical_key':r['historical_key'],'method':'direct' if direct else 'quadratic',
                           'certificate_sha256':hashlib.sha256(raw).hexdigest(),'certificate':cert},
                           separators=(',',':'),allow_nan=False)+'\n')
print('55 certificats conserves dans',out)

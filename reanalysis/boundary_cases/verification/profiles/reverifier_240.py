"""Reverifie 185 certificats precedents et les 55 nouveaux, lies aux donnees gelees.

Les divergences historiques sont rapportees, jamais remplacees ni considerees comme
des erreurs de lecture si le nouveau certificat rationnel est valide.
"""
import hashlib
import json
import sys
from collections import Counter
from fractions import Fraction as F
from pathlib import Path

if not __debug__:
    raise SystemExit('Run without Python -O: verification requires assertions')

P=Path(__file__).resolve().parent
ROOT=P.parents[1]/'04_Code_et_experiences'
sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
from manifest import build_instance,instance_hash,objectives_for
from certrmp import certify,verify as prior
import verificateur_quadratique as vq
from controle_borne_independant import compare


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

manifest=ROOT/'results/v1/manifest_stress_profiles.json'
historical=ROOT/'results/v1/stress_profiles.jsonl'
baseline=ROOT/'results/replay_final_20261101/stress_profiles/records.jsonl'
new=P/'certificats_55.jsonl'
entries={e['id']:e for e in json.loads(manifest.read_text())['entries']}
old={(r['id'],r['model']):r for r in map(json.loads,historical.read_text().splitlines()) if r.get('method')=='lcs'}
base={(r['id'],r['model']):r for r in map(json.loads,baseline.read_text().splitlines())}
replacement={(r['id'],r['model']):r for r in map(json.loads,new.read_text().splitlines())}
required={(e['id'],i) for e in entries.values() for i in range(len(e['models']))}
assert len(required)==len(old)==len(base)==240 and set(old)==set(base)==required
missing={k for k,r in base.items() if not r['verification']['exact']}
assert len(missing)==55 and set(replacement)==missing
counts=Counter(); details=[]; errors=[]; eps=[]; comparisons=0
for key in sorted(required):
    entry=entries[key[0]]; inst=build_instance(entry)
    try:
        assert instance_hash(inst)==entry['instance_sha']
        original=base[key]
        row=replacement[key] if key in replacement else original
        cert=row['certificate']
        assert cert['instance']==certify.instance_dict(inst)
        assert cert['objectives']==certify.objectives_dict(objectives_for(inst,entry['p']))
        assert cert['model']==entry['models'][key[1]]['model']
        digest=hashlib.sha256(json.dumps(cert,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        assert digest==row['certificate_sha256']
        assert [F(v) for v in row['historical_key']]==[F(v) for v in old[key]['key']]
        result=vq.verify(cert) if key in replacement else prior.verify(cert)
        assert result['verified'] and result['exact']
        if key in replacement:
            for bound in cert['halfspaces']:
                if all(F(x)>=0 for x in bound['mu']): compare(cert,bound); comparisons+=1
            for bound in cert['refinement']['bounds']:
                compare(cert,bound,refinement=True); comparisons+=1
            if row['method']=='direct': assert prior.verify(cert)['exact']
            counts['new_verified']+=1; counts['new_'+row['method']]+=1
            eps.append(result['refinement_epsilon_relative'])
        else:
            counts['previous_verified']+=1
        a=tuple(F(v) for v in cert['key']); b=tuple(F(v) for v in old[key]['key'])
        relation='same' if a==b else 'historical_higher' if b>a else 'historical_lower'
        counts[relation]+=1
        details.append({'id':key[0],'model':key[1],'key':cert['key'],'historical_key':old[key]['key'],
                        'relation':relation,'new':key in replacement,'refinement_epsilon_relative':result.get('refinement_epsilon_relative')})
    except Exception as exc:
        errors.append([key[0],key[1],repr(exc)])
counts['checked']=len(details);counts['independent_bound_comparisons']=comparisons
files=[manifest,historical,baseline,new,P/'verificateur_quadratique.py',P/'controle_borne_independant.py',
       P/'reverifier_240.py',ROOT/'certrmp/verify.py',ROOT/'certrmp/certify.py',ROOT/'experiments/manifest.py']
summary={'counts':dict(counts),'max_relative_refinement_epsilon_new':max(eps) if eps else None,
         'sha256':{str(p.relative_to(P.parents[1])):sha(p) for p in files},'errors':errors,'records':details}
print(json.dumps(summary,indent=2,allow_nan=False))
if errors or counts['checked']!=240 or counts['new_verified']!=55 or counts['previous_verified']!=185:
    sys.exit(1)

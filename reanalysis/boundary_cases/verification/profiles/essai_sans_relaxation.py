"""Diagnostic local : resoudre les 55 cas sans relaxation des priorites.

Les resultats sont nouveaux, separes des fichiers figes. Un code non verifie n'est
jamais promu par ce script ; les certificats et leurs erreurs sont conserves.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[2] / '04_Code_et_experiences'
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments')]
from manifest import build_instance, instance_hash, objectives_for
from certrmp.model import model_from_dict
from certrmp import pipeline, oracle


class StrictOracle(oracle.Oracle):
    def __init__(self, inst, objs, sigma):
        super().__init__(inst, objs, sigma, tau_rel=0.0)


pipeline.Oracle = StrictOracle
oracle.SOLVER_OPTS.update(tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12)
manifest = json.loads((ROOT / 'results/v1/manifest_stress_profiles.json').read_text())
entries = {e['id']: e for e in manifest['entries']}
old = [json.loads(s) for s in (ROOT / 'results/replay_final_20261101/stress_profiles/records.jsonl').open()]
bad = [r for r in old if not r['verification']['exact']]
limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(bad)
out = Path(__file__).parent / (sys.argv[2] if len(sys.argv) > 2 else 'essai_sans_relaxation.jsonl')
if out.exists():
    raise SystemExit('output already exists; no overwrite')
with out.open('x') as stream:
    for row in bad[:limit]:
        e = entries[row['id']]
        inst = build_instance(e)
        assert instance_hash(inst) == e['instance_sha']
        model = model_from_dict(e['models'][row['model']]['model'])
        result = pipeline.solve(inst, objectives_for(inst, e['p']), model, time_limit=60, keep_certificate=True)
        result.pop('x', None)
        stored = {'id': row['id'], 'model': row['model'], 'instance_sha': e['instance_sha'],
                  'historical_key': row['historical_key'], 'previous_key': row['replay']['key'], 'new': result}
        stream.write(json.dumps(stored, allow_nan=False, separators=(',', ':'))+'\n')
        stream.flush()
        print(row['id'], result['status'], result.get('key'), result.get('checks'), flush=True)

"""Cherche des separateurs supplementaires ; seule la verification exacte decide."""
import copy
import json
import math
import os
import sys
from fractions import Fraction
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[2] / '04_Code_et_experiences'
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments')]
from manifest import build_instance, instance_hash, objectives_for
from certrmp import certify, oracle, search, verify
from certrmp.instance import priority_totals
from diagnostic_frontieres import missed

oracle.SOLVER_OPTS.update(tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12)
entries = {e['id']: e for e in json.loads((ROOT / 'results/v1/manifest_stress_profiles.json').read_text())['entries']}
path = Path(__file__).parent/'tous_sans_relaxation.jsonl'
out = Path(__file__).parent/'separateurs_supplementaires.jsonl'
rows = [json.loads(line) for line in path.open()]
if out.exists():
    raise SystemExit('output exists')
with out.open('x') as stream:
    for row in rows:
        if row['new']['status'] == 'OPTIMAL_VERIFIED':
            continue
        e = entries[row['id']]
        inst = build_instance(e)
        assert instance_hash(inst) == e['instance_sha']
        sigma, exact = priority_totals(inst)
        assert exact
        o = oracle.Oracle(inst, objectives_for(inst, e['p']), sigma, tau_rel=0.0)
        cert = copy.deepcopy(row['new']['certificate'])
        attempts = []
        for candidate in missed(cert):
            w = [math.inf if v == 'inf' else float(Fraction(v)) for v in candidate['w']]
            ans = o.achievement(w)
            attempts.append({'w': candidate['w'], 'solver_status': ans['status'], 't': ans.get('t')})
            if ans['status'] in ('optimal', 'optimal_inaccurate'):
                hs = search.Halfspace(ans['mu'], 0.0, ans['x'], ans['y_cap'], ans['y_tier'], 'targeted')
                cert['halfspaces'].append(certify.halfspace_dict(hs, inst))
        verified = verify.verify(cert)
        result = {'id': row['id'], 'model': row['model'], 'instance_sha': e['instance_sha'],
                  'historical_key': row['historical_key'], 'certificate': cert, 'verification': verified,
                  'attempts': attempts, 'remaining': missed(cert)}
        stream.write(json.dumps(result,allow_nan=False,separators=(',',':'))+'\n')
        stream.flush()
        print(row['id'], verified['exact'], 'remaining', len(result['remaining']), flush=True)

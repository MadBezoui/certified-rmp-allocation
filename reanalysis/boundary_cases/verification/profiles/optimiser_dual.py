"""Reserre les bornes duales, puis verifie les certificats en rationnels exacts.

Pour chaque tangente deja calculee, un LP maximise la borne separable sur les
multiplicateurs des capacites et des priorites. Le LP ne constitue pas une preuve :
seul le controle rationnel du certificat modifie decide de son acceptation.
"""
import copy
import json
import os
import sys
from fractions import Fraction as F
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[2] / '04_Code_et_experiences'
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.optimize import linprog
from certrmp import verify
from diagnostic_frontieres import missed


def improve(cert, h, scale=1e8):
    D = verify.Data(cert)
    mu = np.array([float(F(v)) for v in h['mu']])
    x = np.array([float(F(v)) for v in h['tangent_x']])
    s = np.array([float(F(v)) for v in h['tangent_s']])
    d = np.array([float(v) for v in D.d])
    xp = np.array([float(v) for v in D.xp])
    A = np.array([[float(a) for a in row] for row in D.A])
    lower = np.array([float(v) for v in D.lo]); upper = np.array([float(v) for v in D.hi])
    grad = np.zeros(D.n); V = np.zeros(D.n)
    for a, obj in zip(mu, D.objs):
        if not a:
            continue
        a /= float(F(obj['scale']))
        if obj['kind'] == 'shortfall':
            grad += -2*a*np.array([float(F(v)) for v in obj['weights']])*(d-x)
        elif obj['kind'] == 'movement':
            V += a*np.array([float(F(v)) for v in obj['weights']])
        else:
            for group, g in zip(obj['groups'], obj['gammas']):
                if len(group) >= 2:
                    ix=np.array(group); grad[ix] += 2*a*float(F(g))/len(ix)*(s[ix]-np.mean(s[ix]))/d[ix]
    tier = np.zeros((len(D.tiers),D.n))
    for k,group in enumerate(D.tiers): tier[k,group]=1
    C = np.vstack((A,-tier))
    nr = len(C); n = D.n
    rhs = np.r_[np.array([float(v) for v in D.c]),
                [-float(D.D[k]-F(cert['priority']['totals'][k])) for k in range(len(D.tiers))]]
    # Variable is (scale*y, scale*z); z_i is a lower envelope over the box.
    c = np.r_[rhs, -np.ones(n)]
    rows=[]; b=[]
    for i in range(n):
        for v in (lower[i], upper[i], min(max(xp[i],lower[i]),upper[i])):
            row=np.zeros(nr+n); row[:nr]=-v*C[:,i]; row[nr+i]=1
            rows.append(row); b.append(scale*(grad[i]*v+V[i]*abs(v-xp[i])))
    sol=linprog(c,A_ub=np.array(rows),b_ub=np.array(b),bounds=[(0,None)]*nr+[(None,None)]*n,
                method='highs',options={'dual_feasibility_tolerance':1e-10,'primal_feasibility_tolerance':1e-10})
    if not sol.success:
        return None,sol.message
    dual=np.maximum(sol.x[:nr]/scale,0)
    hs=copy.deepcopy(h)
    hs['y_cap']=[str(F(float(v))) for v in dual[:len(D.A)]]
    hs['y_tier']=[str(F(float(v))) for v in dual[len(D.A):]]
    hs['origin']='targeted_lp_dual'
    return hs, sol.message


if __name__=='__main__':
    parent=Path(__file__).parent
    src=parent/'separateurs_supplementaires.jsonl'; out=parent/'dual_optimise.jsonl'
    if out.exists(): raise SystemExit('output exists')
    with out.open('x') as stream:
        for row in (json.loads(line) for line in src.open()):
            cert=copy.deepcopy(row['certificate'])
            targeted=[h for h in cert['halfspaces'] if h.get('origin')=='targeted']
            messages=[]
            for h in targeted:
                hs,msg=improve(cert,h); messages.append(msg)
                if hs: cert['halfspaces'].append(hs)
            v=verify.verify(cert)
            record={'id':row['id'],'model':row['model'],'instance_sha':row['instance_sha'],
                    'historical_key':row['historical_key'],'certificate':cert,'verification':v,
                    'remaining':missed(cert),'lp_status':messages}
            stream.write(json.dumps(record,allow_nan=False,separators=(',',':'))+'\n'); stream.flush()
            print(row['id'],v['exact'],'remaining',len(record['remaining']),flush=True)

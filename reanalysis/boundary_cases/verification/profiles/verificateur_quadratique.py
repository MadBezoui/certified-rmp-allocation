"""Verification rationnelle avec minimisation exacte des termes quadratiques separables.

Variation du controleur conserve : les carres de shortfall ne sont plus linearises.
Chaque minimum univarie sur la boite, avec rupture en x_prev, est calcule sur les
extremites et les points stationnaires rationnels ; la disparite reste minoree par
ses tangentes rationnelles. Les autres controles du verificateur restent inchanges.
"""
import sys
from fractions import Fraction as F
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2] / '04_Code_et_experiences'
sys.path.insert(0, str(ROOT))
from certrmp import verify as original


def halfspace_bound(D, sigma, weights, tangent_x, tangent_s, y_cap, y_tier):
    n = D.n
    alpha = [F(0)] * n
    movement = [F(0)] * n
    lin = [F(0)] * n
    const = F(0)
    for a, obj in zip(weights, D.objs):
        if a == 0:
            continue
        sc = F(obj['scale'])
        if obj['kind'] == 'shortfall':
            for i, w in enumerate(obj['weights']):
                alpha[i] += a * F(w) / sc
        elif obj['kind'] == 'movement':
            for i, w in enumerate(obj['weights']):
                movement[i] += a * F(w) / sc
        else:
            for group, gamma in zip(obj['groups'], obj['gammas']):
                gamma = F(gamma)
                if len(group) < 2 or gamma <= 0:
                    continue
                kappa = a * gamma / (sc * len(group))
                s = [tangent_s[i] for i in group]
                mean = sum(s, F(0)) / len(s)
                for i, si in zip(group, s):
                    const += original.floor_grid(kappa * (mean*mean - si*si))
                    lin[i] += 2*kappa*(si-mean)/D.d[i]
    for r, y in enumerate(y_cap):
        if y:
            for i, a in enumerate(D.A[r]):
                lin[i] += y*a
            const -= y*D.c[r]
    for k, y in enumerate(y_tier):
        if y:
            for i in D.tiers[k]:
                lin[i] -= y
            const -= y*(sigma[k]-D.D[k])
    total = const
    for i in range(n):
        lo, hi, xp = D.lo[i], D.hi[i], D.xp[i]
        candidates = [lo, hi]
        if lo < xp < hi:
            candidates.append(xp)
        if alpha[i] > 0:
            for sign in (-1, 1):
                z = D.d[i] - (lin[i] + sign*movement[i])/(2*alpha[i])
                if lo <= z <= hi and ((sign < 0 and z <= xp) or (sign > 0 and z >= xp)):
                    candidates.append(z)
        value = min(alpha[i]*(D.d[i]-z)**2 + lin[i]*z + movement[i]*abs(z-xp) for z in candidates)
        total += original.floor_grid(value)
    return total


def verify(cert):
    previous = original.halfspace_bound
    original.halfspace_bound = halfspace_bound
    try:
        return original.verify(cert)
    finally:
        original.halfspace_bound = previous

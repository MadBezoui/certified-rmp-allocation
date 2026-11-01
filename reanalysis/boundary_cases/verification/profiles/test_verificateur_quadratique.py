"""Tests independants de bornes, cas limites et rejets adversariaux."""
import copy
import json
from fractions import Fraction as F
from pathlib import Path
from types import SimpleNamespace

import pytest
from certrmp import verify as prior
import verificateur_quadratique as vq


def data():
    return SimpleNamespace(n=2, d=[F(2),F(2)], xp=[F(1,2),F(3,2)],
        lo=[F(0),F(0)], hi=[F(2),F(2)], A=[[F(1),F(1)]], c=[F(2)],
        tiers=[[0,1]], D=[F(4)], objs=[
            {'kind':'shortfall','scale':'3','weights':['1','2']},
            {'kind':'movement','scale':'2','weights':['2','1']},
            {'kind':'disparity','scale':'1','groups':[[0,1]],'gammas':['3']}])


@pytest.mark.parametrize('weights', [[F(1),F(1),F(1)], [F(0),F(1),F(0)],
                                     [F(2),F(0),F(3)], [F(0),F(0),F(0)]])
@pytest.mark.parametrize('tangent', [[F(0),F(2)], [F(1,2),F(3,2)], [F(3,2),F(1,2)]])
def test_bound_never_exceeds_exact_values_on_feasible_points(weights,tangent):
    D=data(); sigma=[F(2)]; ts=[x/d for x,d in zip(tangent,D.d)]
    for yc,yt in ((F(0),F(0)), (F(2,7),F(3,11)), (F(10),F(1,20))):
        new=vq.halfspace_bound(D,sigma,weights,tangent,ts,[yc],[yt])
        old=prior.halfspace_bound(D,sigma,weights,tangent,ts,[yc],[yt])
        assert new >= old
        for i in range(81):
            x=[F(i,40), F(2)-F(i,40)]
            short=((2-x[0])**2+2*(2-x[1])**2)/3
            move=(2*abs(x[0]-D.xp[0])+abs(x[1]-D.xp[1]))/2
            a,b=x[0]/2,x[1]/2
            disp=F(3,2)*((a-(a+b)/2)**2+(b-(a+b)/2)**2)
            actual=weights[0]*short+weights[1]*move+weights[2]*disp
            assert new <= actual


def test_exact_quadratic_improves_tangent_bound():
    D=SimpleNamespace(n=1,d=[F(1)],xp=[F(0)],lo=[F(0)],hi=[F(1)],
        A=[],c=[],tiers=[[0]],D=[F(1)],objs=[{'kind':'shortfall','scale':'1','weights':['1']}])
    old=prior.halfspace_bound(D,[F(1)],[F(1)],[F(0)],[F(0)],[],[F(0)])
    new=vq.halfspace_bound(D,[F(1)],[F(1)],[F(0)],[F(0)],[],[F(0)])
    assert old == -1 and new == 0


@pytest.fixture(scope='module')
def certified():
    path=Path(__file__).parent/'dual_optimise.jsonl'
    row=json.loads(path.open().readline())
    cert=row['certificate']
    assert vq.verify(cert)['exact']
    return cert


@pytest.mark.parametrize('mutation', ['levels','witness','priority','refinement','weight','frontier'])
def test_rejects_tampered_certificate(certified,mutation):
    cert=copy.deepcopy(certified)
    if mutation == 'levels': cert['levels']=[]
    elif mutation == 'witness': cert['witness']['x'][0]='-1000'
    elif mutation == 'priority': cert['priority']['totals'][0]='-1000'
    elif mutation == 'refinement': cert['refinement']['bounds'][0]['u']=['-1000']*3
    elif mutation == 'weight': cert['model']['weights'][0]='-1'
    elif mutation == 'frontier': cert['levels'][-1]['next_frontier']=[['inf']*3]
    assert not vq.verify(cert)['verified']

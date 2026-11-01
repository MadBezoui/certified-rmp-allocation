"""Regression tests for the certificate trust boundary (audit 2026-11-01)."""
import copy
import numpy as np
import pytest

from certrmp.instance import Instance, SINGLE
from certrmp.objectives import Objective
from certrmp.model import RMPModel
from certrmp.pipeline import solve
from certrmp.verify import verify


@pytest.fixture(scope='module')
def valid_certificate():
    inst = Instance(d=[4, 2], x_prev=[1.5, 1.5], tiers=[[0, 1]],
                    A=np.ones((1, 2)), c=[3], structure=SINGLE)
    objs = [Objective('shortfall', 'shortfall', weights=np.ones(2), scale=1),
            Objective('movement', 'movement', weights=np.ones(2), scale=1),
            Objective('disparity', 'disparity', groups=[np.array([0, 1])], gammas=np.array([4.]), scale=1)]
    model = RMPModel.srmp([[5.2, 1.2, 0.02], [6.0, 1.8, 0.1]], [0, 1], [0.4, 0.2, 0.4])
    cert = solve(inst, objs, model, keep_certificate=True)['certificate']
    assert verify(cert)['exact']
    return cert


def test_rejects_understated_key_with_no_levels(valid_certificate):
    cert = copy.deepcopy(valid_certificate)
    cert['key'] = ['0', '0']
    cert['levels'] = []
    assert not verify(cert)['verified']


def test_rejects_non_candidate_frontier(valid_certificate):
    cert = copy.deepcopy(valid_certificate)
    cert['levels'][-1]['next_frontier'] = [['inf'] * 3]
    assert not verify(cert)['verified']


def test_rejects_unrelated_refinement_vector(valid_certificate):
    cert = copy.deepcopy(valid_certificate)
    cert['refinement']['bounds'][0]['u'] = ['-1000'] * 3
    result = verify(cert)
    assert not result['verified']
    assert 'refinement_epsilon_relative' not in result


def test_rejects_missing_refinement(valid_certificate):
    cert = copy.deepcopy(valid_certificate)
    del cert['refinement']
    assert not verify(cert)['verified']


def test_rejects_nonpositive_refinement_weight(valid_certificate):
    cert = copy.deepcopy(valid_certificate)
    cert['refinement']['lambda'][0] = '-1'
    assert not verify(cert)['verified']


def test_rejects_truncated_capacity_row(valid_certificate):
    cert = copy.deepcopy(valid_certificate)
    cert['instance']['A'][0].pop()
    assert not verify(cert)['verified']


def test_rejects_negative_capacity_coefficient(valid_certificate):
    cert = copy.deepcopy(valid_certificate)
    cert['instance']['A'][0][0] = '-1'
    assert not verify(cert)['verified']


def test_rejects_wrong_recorded_profile(valid_certificate):
    cert = copy.deepcopy(valid_certificate)
    cert['levels'][0]['profile'] = cert['levels'][1]['profile']
    assert not verify(cert)['verified']


def test_failed_priority_evidence_is_not_verified(valid_certificate):
    from fractions import Fraction
    cert = copy.deepcopy(valid_certificate)
    cert['priority']['totals'][0] = str(Fraction(cert['priority']['totals'][0]) + 1)
    result = verify(cert)
    assert not result['verified'] and not result['exact']


def test_rejects_imported_negative_srmp_weight(valid_certificate):
    from certrmp.model import model_from_dict
    model = copy.deepcopy(valid_certificate['model'])
    model['weights'][0] = '-1'
    with pytest.raises(ValueError, match='nonnegative'):
        model_from_dict(model)

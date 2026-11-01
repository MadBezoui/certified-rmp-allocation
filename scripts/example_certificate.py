"""Two-recipient example solved with the certified pipeline.

Requests (4, 2), capacity 3, previous allocation (1.5, 1.5); unscaled shortfall, movement and
squared difference of the two service fractions; profiles (5.2, 1.2, 0.02) then (6.0, 1.8, 0.1),
SRMP weights (0.4, 0.2, 0.4). Prints the best key, the refined allocation and the verifier's
checks, then shows that the verifier rejects a certificate claiming a better key.
"""
import copy
import json
import pathlib
import sys
import warnings

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np

from certrmp.instance import Instance, SINGLE
from certrmp.objectives import Objective
from certrmp.model import RMPModel
from certrmp.pipeline import solve
from certrmp import verify

inst = Instance(d=[4, 2], x_prev=[1.5, 1.5], tiers=[[0, 1]], A=np.ones((1, 2)), c=[3], structure=SINGLE)
objs = [Objective("shortfall", "shortfall", weights=np.ones(2), scale=1.0),
        Objective("movement", "movement", weights=np.ones(2), scale=1.0),
        Objective("disparity", "disparity", groups=[np.array([0, 1])], gammas=np.array([4.0]), scale=1.0)]
model = RMPModel.srmp([[5.2, 1.2, 0.02], [6.0, 1.8, 0.1]], [0, 1], [0.4, 0.2, 0.4])

rec = solve(inst, objs, model, keep_certificate=True)
print("status      ", rec["status"])
print("best key    ", rec["key"])
print("allocation  ", np.round(rec["x"], 6).tolist())
print("criteria    ", np.round(rec["f"], 6).tolist())
print("checks      ", rec["checks"])
print("certificate ", rec["certificate_bytes"], "bytes,", rec["halfspaces"], "halfspaces")

forged = copy.deepcopy(json.loads(json.dumps(rec["certificate"])))
forged["key"] = ["1", "2"]
out = verify.verify(forged)
print("forged key  ", "verified" if out["verified"] else "rejected",
      "- failed checks:", [k for k, v in out["checks"].items() if not v])

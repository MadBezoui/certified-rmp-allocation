"""Two-recipient example: requests (4,2), capacity 3, previous allocation (1.5,1.5)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from rmp.core import Instance, Objectives, RMPModel, solve

inst = Instance(d=np.array([4.0, 2.0]), tiers=[[0, 1]], A=np.ones((1, 2)), c=np.array([3.0]),
                x_prev=np.array([1.5, 1.5]))
# Unscaled: total squared shortfall, absolute movement, squared difference of the two fractions.
# gamma/|J| * sum (s - mean)^2 = gamma/2 * (s1-s2)^2/2, so gamma = 4 gives (s1-s2)^2.
obj = Objectives(w=np.ones(2), v=np.ones(2), gamma=np.array([4.0]))
for x in ([1.5, 1.5], [2.0, 1.0], [2.5, 0.5]):
    print(x, obj.values(inst, np.array(x)))
model = RMPModel.srmp(profiles=[[5.2, 1.2, 0.02], [6.0, 1.8, 0.1]], order=[0, 1], weights=[0.4, 0.2, 0.4])
for x in ([1.5, 1.5], [2.0, 1.0], [2.5, 0.5]):
    print(x, [str(k) for k in model.key(obj.values(inst, np.array(x)))])
res = solve(inst, obj, model)
print(res.status, [str(k) for k in res.key], res.x, res.f, res.patterns_tested, res.convex_calls)
print("weakest optimal patterns", res.optimal_patterns, "flags", res.boundary_flags)

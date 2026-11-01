# Boundary-profile reanalysis

The 55 new certificates in `certificats_55.jsonl` are bound to the frozen instance manifest and the historical records. The 185 previously certified pairs are stored in the separate replay records. Thirty-one new certificates pass the corrected verifier in `certrmp/verify.py`; 24 additionally use `verificateur_quadratique.py`, which exactly minimizes each separable shortfall quadratic over the box while keeping rational lower tangents for disparity. `controle_borne_independant.py` recomputes all 505 new certificate bounds independently by completing the square. The numerical optimizers suggest witnesses and multipliers; the rational check, not the optimizer status, establishes the key. Refinement is certified only up to the recorded tolerance.

From the **repository root**, after installing `requirements-lock.txt` in `.venv`, check all 240 certificates and run the supplementary verifier tests:

```sh
.venv/bin/python reanalysis/boundary_cases/verification/profiles/reverifier_240.py > boundary-verification.json
PYTHONPATH=reanalysis/boundary_cases/04_Code_et_experiences \
  .venv/bin/python -m pytest -q reanalysis/boundary_cases/verification/profiles/test_verificateur_quadratique.py
```

Expected: 185 previous and 55 new certificates verified, 200 historical keys matched, 40 historical keys higher, 505 bounds checked, and no errors. The recorded data are never overwritten. Historical keys, timings and statuses are observations of the earlier code; the historical allocations were not retained. `dual_optimise.jsonl` contains intermediate candidate records and is used as a test fixture; the final certificate verification is performed afresh by `reverifier_240.py`.

## Regenerating numerical candidates

The numerical replay used Python 3.13, CVXPY 1.9.3 and Clarabel 0.11.1 on macOS ARM64, with one worker and a 60-second limit per pair. `essai_sans_relaxation.py` builds the 55 tasks from the failed records of the first replay, verifies each regenerated instance hash, disables the default numerical priority relaxation and uses conic tolerances of `1e-12`. `ajouter_separateurs.py` and `optimiser_dual.py` propose further separating records and dual multipliers, which are checked by exact arithmetic. The original intermediate scripts are preserved in this directory.

Run regeneration only in a new disposable copy: it writes new files and refuses to overwrite previous ones. For example, from the repository root:

```sh
mkdir -p rerun
cp -R reanalysis/boundary_cases rerun/boundary_cases
cd rerun/boundary_cases/verification/profiles
mv dual_optimise.jsonl dual_optimise.reference.jsonl
mv certificats_55.jsonl certificats_55.reference.jsonl
../../../../.venv/bin/python essai_sans_relaxation.py 55 tous_sans_relaxation.jsonl
../../../../.venv/bin/python ajouter_separateurs.py
../../../../.venv/bin/python optimiser_dual.py
../../../../.venv/bin/python consolider_certificats.py
../../../../.venv/bin/python reverifier_240.py > verification_240.json
```

`consolider_certificats.py` expects the original 31/24 split; numerical outcomes can vary with solver version, platform and rounding, so a different split requires inspecting the candidate statuses and producing independently verified certificates, not forcing that count. A successful verification checks the newly generated evidence; it does not imply that numerical candidates, times, multipliers or refinement tolerances are bit-identical across machines. The archived historical and previously verified results remain distinct from any rerun.

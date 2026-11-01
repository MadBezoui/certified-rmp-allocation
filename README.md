# Certified multiobjective allocation with multiple reference profiles

Code, instance manifests and numerical results for a computational study of divisible-resource allocation under strict service priorities, in which a decision maker states preferences with multiple reference profiles (RMP, and its additive variant SRMP). Among all priority-optimal allocations, the solver searches for the best preference class with a lexicographic coalition search (LCS), constructs a witness and, when successful, a certificate that an independent verifier checks in rational arithmetic without a solver. The verified refinement bound limits the possible weighted-sum improvement; a positive bound does not establish exact nondominance.

This corrected code validates certificate structure, preference monotonicity, frontier membership and the association of each refinement bound with its frontier. The `results/v1` records are historical and retain their original code hash and run times. Their former verification flags were produced by an earlier verifier and their certificates were not stored; they are not, by themselves, independently replayable certificates. `scripts/replay_certificates.py` creates separately bound certificates from the frozen manifests, and `scripts/check_replay.py` checks those certificates in another process. Do not interpret a numerical or unresolved result as certified.

This is a research implementation, not a certified physical controller.

## Boundary-profile reanalysis

A separate reanalysis resolves the 55 previously uncertified profile-boundary cases. It produces 55 exact-rational certificates: 15 recover the historical numerical key and 40 establish a strictly lower key. Together with the previous 185 certified pairs, this verifies all 240 pairs in the profile design. The original results, solver statuses and timings are retained unchanged. The maximum relative refinement bound in the 55 new certificates is approximately `8.33e-6`; the preference keys are verified exactly, while the refinement guarantee has that recorded tolerance.

The 55-case replay disables the default numerical priority relaxation and proposes tighter witnesses and separating bounds. Thirty-one new certificates pass the existing corrected verifier; for the other 24 a supplementary verifier minimizes separable shortfall quadratics exactly over the box. A second rational implementation independently rechecks 505 bounds. A numerical or uncertified key must never be treated as an exact optimum.

After installing the locked dependencies below, verify the self-contained reanalysis and its tests from the repository root:

```sh
.venv/bin/python reanalysis/boundary_cases/verification/profiles/reverifier_240.py > boundary-verification.json
PYTHONPATH=reanalysis/boundary_cases/04_Code_et_experiences \
  .venv/bin/python -m pytest -q reanalysis/boundary_cases/verification/profiles/test_verificateur_quadratique.py
```

The first command verifies all 240 certificates, checks the 505 bounds and compares the keys with the frozen historical records. The expected counts are 185 previous and 55 new certificates, 200 matching historical keys, 40 historical keys higher, and no errors. The self-contained directory also includes the numerical replay scripts and their inputs; see its [reproduction notes](reanalysis/boundary_cases/verification/profiles/README.md). Other platforms may produce different numerical candidates; their certificates must pass independent exact verification. The historical allocations were not saved, so historical comparator feasibility cannot be rechecked from their recorded keys alone.

## Installation

Python 3.12 or later (results produced with Python 3.13). With [`uv`](https://docs.astral.sh/uv/):

```sh
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements-lock.txt
```

`requirements-lock.txt` pins the dependencies for this package, including the dataset downloader. Gurobi is optional for the optimization methods and requires a license; the historical results use a size-restricted license, hence only n = 30. Each `*_environment.json` file in `results/v1/` records versions, worker count, time limit and code hash for the historical run. These records use a different solver code version from the corrected verifier.

The original experiment code is separately preserved under `frozen/v1/`, with its own `SHA256SUMS`; its solver-code hash is `db89d903964c43fe`. Its old verifier has known deficiencies, so its historical verification flags must not be used as independent proof. The frozen direct-dependency list records the original versions but does not pin every transitive dependency. The historical multiprocessing runner applies a per-worker address-space limit intended for Linux and may fail on macOS. To regenerate a small historical run into a **new** file on either system, use this sequential wrapper; it calls the frozen task function without applying that limit:

```sh
mkdir -p rerun/historical
.venv/bin/python scripts/run_frozen_sequential.py \
  results/v1/manifest_main.json rerun/historical/main.jsonl --id main-00000
```

The original frozen runner is append-only; the sequential wrapper refuses to overwrite an output file. Its resource limits and worker scheduling differ from the historical run, so timings are not directly comparable. Numerical outcomes can vary across solvers and platforms. Verify current claims with the corrected certificates and the separate boundary reanalysis rather than inferring exactness from a historical status.

## Quick check

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python -m pytest -q
.venv/bin/python scripts/example_certificate.py
```

The example solves the two-recipient allocation used to explain the method, prints the best key, the refined allocation and the verifier's checks, and shows that the verifier rejects a certificate whose key was altered. `scripts/example_two_recipients.py` solves the same example with the small reference solver of `rmp/`.

## Reproduction

Run every command from the repository root. The benchmark runner is append-only and skips records that already exist, so write into a new file to rerun a design.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p rerun
for d in main stress_profiles stress_dimension sensitivity preference; do
  .venv/bin/python experiments/manifest.py $d rerun/manifest_$d.json
done
sha256sum rerun/manifest_*.json        # compare with results/v1/MANIFESTS.sha256
S="1000|1001|1002|1003|1004|1005|1006|1007|1008|1009"
.venv/bin/python experiments/runner.py rerun/manifest_main.json rerun/main.jsonl lcs,enum,scip,eps,gurobi 3 60
.venv/bin/python experiments/runner.py rerun/manifest_main.json rerun/main_ga.jsonl nsga,rmp_ga 3 60 "structure=single;seed=$S"
.venv/bin/python scripts/run_ablation.py rerun/manifest_main.json rerun/main_ablation.jsonl lcs_nopoints,lcs_nohalf,lcs_nocache 3 60 "n=100|1000;seed=$S"
for d in stress_profiles sensitivity preference; do
  .venv/bin/python experiments/runner.py rerun/manifest_$d.json rerun/$d.jsonl lcs,enum,scip,eps 3 60
done
.venv/bin/python experiments/runner.py rerun/manifest_stress_dimension.json rerun/stress_dimension.jsonl lcs,enum,scip 3 60
.venv/bin/python experiments/runner.py rerun/manifest_stress_dimension.json rerun/stress_dimension.jsonl eps 3 60 "p=2|3|5"
.venv/bin/python scripts/measure_memory.py rerun/manifest_main.json rerun/memory.jsonl 3
.venv/bin/python scripts/download_data.py              # about 250 MB from the UCI repository
.venv/bin/python experiments/electricity.py rerun/electricity 0.5 1
.venv/bin/python experiments/electricity.py rerun/electricity 2.0 1
.venv/bin/python scripts/analyze_benchmark.py rerun rerun/analysis
.venv/bin/python scripts/analyze_electricity.py rerun/electricity rerun/analysis
mkdir -p rerun/literature                              # comparators added after the freeze
RMP_CODE_ROOT=. .venv/bin/python scripts/run_literature.py rerun/manifest_main.json rerun/literature/literature.jsonl lcs,asf,lgp,scip_ind,eps60,nsga60,rmpga60,rnsga60 4 60 "seed=1000|1001|1002|1003|1004"
.venv/bin/python scripts/analyze_literature.py rerun rerun/analysis
```

Compute time of the published runs, as the sum of task times on single-threaded processes (Intel Xeon, 2.1 GHz): main design 19.6 core-hours (mostly MIQCP and enumeration tasks that reach the 60-second limit), evolutionary searches 3.5, ablation 0.2, stress designs 1.9 and 1.5, sensitivity 0.8, preference 2.3, peak-memory measurement 0.3, demand traces 0.5 per ramp limit, comparators added after the freeze 28.4. With three workers the main design takes about seven hours of wall time, and with four workers the comparators added after the freeze take about seven hours as well.

To analyze the published records without rerunning anything:

```sh
.venv/bin/python scripts/analyze_benchmark.py results/v1 figures/benchmark
.venv/bin/python scripts/analyze_electricity.py results/v1/electricity figures/electricity
.venv/bin/python scripts/analyze_literature.py results/v1 figures/literature
.venv/bin/python scripts/figures_illustrative.py figures/illustrative
```

`scripts/run_ablation.py` calls the same runner and keeps the method label of each ablation record (`docs/PROTOCOL.md`, section 8). `scripts/run_literature.py` runs the comparators added after the freeze through the same runner; `RMP_CODE_ROOT=.` points it to the solver code at the repository root, and each record carries a hash of the added code (`docs/PROTOCOL.md`, section 8). The epsilon-constraint sample has 8^(p-1) grid points and is not run at p = 8 (`docs/PROTOCOL.md`, section 8). Timings, statuses, keys, call counts and generated certificates can differ with the solver, platform or rounding near thresholds; compare them explicitly and report unresolved cases. Historical timings are retained as recorded.

If `results/replay/` is present, its configuration and certificate records can be checked independently against the historical inputs as follows:

```sh
for d in main stress_profiles stress_dimension sensitivity preference; do
  .venv/bin/python scripts/check_replay.py results/v1/manifest_$d.json results/v1/$d.jsonl results/replay/$d
done
.venv/bin/python scripts/download_data.py
.venv/bin/python scripts/check_electricity_replay.py data/processed/electricity.npz results/v1/electricity results/replay/electricity
```

The check exits nonzero if any pair remains unresolved, differs from its historical key, or is invalid; inspect its JSON counts rather than treating a nonzero exit as missing evidence. In that first replay the profile-boundary design leaves 55 cases unresolved; the separate boundary-profile reanalysis above resolves them and documents 40 different keys. Dynamic replay checks bind each new certificate to its regenerated epoch; historical allocations and certificates were not saved, so matching objective vectors and keys do not prove bitwise identity of the original decisions. The data archive and processed array are not redistributed.

The original `analyze_benchmark.py` retains legacy uncentred bootstrap tail probabilities to reproduce the historical analysis files. They are not used as calibrated confirmatory p-values. `scripts/recheck_statistics.py` provides a separate post-audit paired sign-flip sensitivity analysis under the stated cluster-exchangeability assumption; the third comparison is descriptive.

## Layout

| Path | Content |
|---|---|
| `certrmp/instance.py` | physical set, tiers, exact priority totals (polymatroid recursion for single and laminar capacities, sequential linear programs otherwise) |
| `certrmp/objectives.py` | shortfall, movement and disparity criteria |
| `certrmp/model.py` | RMP models: profiles, consultation order, additive weights or monotone importance tables |
| `certrmp/oracle.py` | conic achievement and weighted-sum problems (CVXPY and Clarabel) |
| `certrmp/search.py` | lexicographic coalition search, threshold enumeration, Pareto refinement |
| `certrmp/certify.py`, `certrmp/verify.py` | certificate construction and the independent exact verifier |
| `certrmp/baselines.py` | MIQCP with SCIP and Gurobi, epsilon-constraint sample, NSGA-II and RMP-guided NSGA-II |
| `certrmp/dynamic.py` | repeated decisions with reuse of separating halfspaces, operational rules |
| `comparators/` | comparators added after the freeze: reference point method, lexicographic goal programming, epsilon-constraint and evolutionary searches until the budget is spent, R-NSGA-II, MIQCP with indicator constraints |
| `rmp/` | a small independent reference solver used by the tests |
| `experiments/` | manifest generator, benchmark runner, demand-trace study |
| `scripts/` | data download, ablation and literature-comparator runs, analyses, illustrative figures, memory measurement |
| `results/v1/` | manifests and their hashes, raw records (one JSON line per instance, model and method), run environments, demand-trace trajectories, analysis tables |
| `frozen/v1/` | original historical experiment code and source hashes; earlier verifier retained only for provenance |
| `docs/PROTOCOL.md` | the computational protocol fixed before the runs |
| `reanalysis/boundary_cases/` | standalone inputs, verification code, replay scripts and certificates for the 55 boundary cases |

## Statuses

For the corrected code, `OPTIMAL_VERIFIED` means that priority totals, the witness, the complete exclusion argument and all refinement bounds passed the exact verifier. `FEASIBLE_UNCERTIFIED` reports a witness without a verified exclusion argument. `OPTIMAL_NUMERICAL` means that the search completed but an exact check failed, or no certificate was built (threshold enumeration). `PHYSICAL_INFEASIBLE` means that the physical set is empty, checked exactly. `UNRESOLVED` means that a convex call failed or the time limit was reached. Comparator statuses are those of the respective solver. Historical status strings remain as recorded and do not retroactively acquire the corrected meaning.

## Licenses

Software: MIT (`LICENSE`). Original results, manifests and documentation: CC BY 4.0 (`LICENSE-DATA.md`). The UCI electricity data keep their own CC BY 4.0 license and are downloaded by `scripts/download_data.py`, not redistributed (`DATA_SOURCES.md`).

## Citation

See `CITATION.cff`.

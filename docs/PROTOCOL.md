# Computational protocol, version 1

**Historical protocol.** The statements below describe the code and analysis as originally frozen. In particular, the claim in section 1 that only the electricity path differs from the release code is historical and does not describe the corrected verifier and model validation now in this repository. The later certificate replays and the boundary-profile reanalysis are separate, use the explicitly recorded code and inputs, and do not alter historical records or timings. See the repository README before interpreting historical verification flags or bootstrap tail probabilities.

Fixed on 30 October 2026 before any confirmatory run. It covers the benchmark designs, the comparators, the outcomes and the analysis of the results in `results/v1/`.

## 1. Code and environment

- Code: the packages `certrmp/`, `experiments/` and `rmp/` of this release. Every result record carries `code_sha` (hash of `certrmp/*.py`, value `db89d903964c43fe`) and `manifest_sha`. Any later code change produces new records; records of different code versions are never pooled. The only difference between the code that produced the results and this release is the default data path of `experiments/electricity.py` (now `data/processed/electricity.npz`).
- Tests: `python -m pytest -q tests`.
- Software: Python 3.13.17, numpy 2.5.4, scipy 1.18.1, cvxpy 1.9.3, Clarabel 0.11.1, HiGHS 1.15.1, SCIP 10.0 through PySCIPOpt 6.3.0, Gurobi 13.0.3 (size-restricted license), pymoo 0.6.2 (`requirements-lock.txt`).
- Hardware: Linux container, 4 virtual cores (Intel Xeon 2.10 GHz), 15 GB memory. Benchmark tasks run in 3 concurrent single-threaded worker processes; the fourth core runs the demand-trace study. All methods share these conditions, and tasks of all methods are interleaved.
- Limits: 60 s wall time per (instance, preference model, method) for the exact methods; 6 GiB address space per worker (a task that exceeds it is recorded as ERROR, never dropped).
- Conic solver: Clarabel with absolute and relative gap and feasibility tolerances 1e-10, at most 400 iterations; after a solver error or a non-optimal status, one retry with static regularization 1e-7, then one with the alternative direct solver. Retries and failures are logged. The sensitivity design repeats one cell at tolerance 1e-8.

## 2. Designs (manifests in `results/v1/manifest_<design>.json`)

Instance generator: `certrmp.generator.make_instance` (seeded, data rounded to four decimals so that every instance is exactly rational in decimal form). Objectives: shortfall, movement, disparity with the frozen scales of `standard_objectives` (sum of squared requests, sum of requests, number of tiers). Preference models: `certrmp.profiles.generate`, which only uses the instance's weighted-sum oracle, never a method under test. Payoff tables are lexicographic; a criterion without trade-off on the payoff table gets a span equal to 5 % of its minimum.

| Design | Content | Purpose |
|---|---|---|
| pilot (excluded) | n in {10, 30, 100}, p in {2, 3}, m in {1, 3, 5}, 3 structures, 3 seeds | failure search and limits; never used in confirmatory comparisons |
| main | p = 3, n in {30, 100, 300, 1000}, m in {2, 4, 8}, structures single, laminar, general, seeds 1000 to 1019; two SRMP models per instance (uniform, near_front) | 720 instances, 1440 instance-model pairs; models are nested within instances |
| stress_profiles | n = 100, m = 4, 3 structures x 20 seeds; families easy, unattainable, adjacent (SRMP) and near_front (nonadditive table) | uninformative, boundary and nonadditive models |
| stress_dimension | laminar, n = 100, m = 4, p in {2, 3, 5, 8} (p > 3: regional shortfalls + movement + disparity), 10 seeds, 2 models | dependence on the number of objectives |
| sensitivity | laminar, n = 100, m = 4, scarcity in {0.5, 0.75, 0.9} x ramp in {none, 0.3 d, 0.1 d}, 10 seeds; plus solver tolerance 1e-8 on the base cell | resource scarcity, ramp tightness, solver tolerance |
| preference | n = 100, m = 4, 3 structures x 10 seeds; one base near_front model and five variants: perturbed profiles (2 % of the payoff span), reversed order, concentrated weights (0.8, 0.1, 0.1), equal weights, nonadditive table | profile order, weight concentration, nonadditivity |

The structure called weighted (heterogeneous resource coefficients) is excluded from all designs: the objective audit (`results/v1/degeneracy_audit.json`) shows that its priority-optimal set is frequently a single allocation, so no preference choice remains.

## 3. Methods

| Method | Role | Budget |
|---|---|---|
| lcs | proposed lexicographic coalition search, refinement, exact certificate and independent verification | 60 s |
| enum | reference: one convex call per requirement vector of the threshold grid | 60 s |
| scip | generic exact MIQCP, binary attainment indicators with big-M from valid bounds, level-by-level optimization, feasibility tolerance 1e-9 | 60 s |
| gurobi | same formulation, 1 thread, feasibility and integrality tolerances 1e-9; license limits it to n = 30 | 60 s |
| eps | epsilon-constraint sample: 3 individual minima + 8^(p-1) grid points, RMP selection (finite sample, no completeness claim) | no limit, time recorded |
| nsga | NSGA-II, population 100, 200 generations, exact projection repair (single capacity only), RMP selection of the final population | 20,000 evaluations |
| rmp_ga | the same with RMP-key survival ranking (preference-guided evolutionary search) | 20,000 evaluations |
| lcs_nopoints, lcs_nohalf, lcs_nocache | ablations of lcs without the point cache, the halfspace cache, or both | 60 s |

Assignment: main runs lcs, enum, scip, eps on all pairs; gurobi on all n = 30 pairs; nsga and rmp_ga on single-capacity instances with seeds 1000 to 1009; the ablations on n in {100, 1000} with seeds 1000 to 1009. The stress, sensitivity and preference designs run lcs, enum, scip and eps.

## 4. Outcomes

Primary algorithmic outcome: the fraction of instance-model pairs whose best key is established within the budget. For lcs this means status OPTIMAL_VERIFIED or OPTIMAL_NUMERICAL; for enum, OPTIMAL_NUMERICAL; for scip and gurobi, solver status OPTIMAL at every level. Pairs with other statuses count as unsolved at every budget.

Secondary algorithmic outcomes: total time and time to key, verification time, certificate size in bytes, convex oracle calls, solver failures and retries, unresolved comparisons, and the share of lcs results whose every claim is checked exactly (OPTIMAL_VERIFIED).

Decision-quality outcomes (comparators against lcs under the same model): RMP wins, ties and losses of the key claimed by the method and of the key of its returned allocation evaluated in floating point; Pareto dominance between returned objective vectors; physical and priority residuals.

## 5. Analysis

- Paired comparisons on common instance-model pairs. Uncertainty by cluster bootstrap over instances (all models of an instance resampled together), 2,000 resamples, percentile 95 % intervals, fixed seed.
- Runtime: solved fraction as a function of the budget (0.01 s to 60 s, log scale) and performance profiles with unsolved runs at infinity; geometric mean time ratios computed only on pairs solved by both methods, always next to the solved fractions. No average over cases solved by only one method.
- Confirmatory family (Holm adjustment at 5 %): (H1) lcs solves more pairs than scip within 60 s at n = 1000; (H2) lcs needs fewer convex calls than enum (paired, all main pairs); (H3) the key of lcs is never worse than that of eps (wins minus losses of lcs). All other comparisons are descriptive.
- Ordinal keys: only wins, ties and losses are reported; no arithmetic on keys.
- Failure to detect a difference is not reported as equivalence.

## 6. Pilot observations

1. The SCIP solution was read after the transformed problem had been freed and could be a stale solution; fixed by reading it after every level.
2. Gurobi's default integrality tolerance (1e-5) let big-M rows claim thresholds that are not attainable (a gap of 3e-7 relative); tolerance set to 1e-9.
3. Payoff tables from single minimizations can be only weakly efficient; with a flat criterion, profiles were rounded within 5e-7 of the attainable boundary. Fixed by lexicographic payoff tables and a span floor.
4. The refined witness could sit on a threshold within rounding; fixed by a decreasing margin ladder (1e-7, 1e-9, 1e-11 of the scale) and, if needed, the most interior point of the requirement set.
5. A degenerate linear program made Clarabel stop with an error at tolerance 1e-10; fixed by the retry ladder.
6. Generic MIQCP witnesses often attain their thresholds only within the solver tolerance: in the second pilot, 15 of 145 SCIP allocations miss their claimed key by 1e-16 to 1e-10 relative in floating-point evaluation. Both the claimed and the evaluated keys are therefore reported.

## 7. Scope limits declared in advance

- Peak memory is not measured per task in the benchmark (worker processes are reused); a separate measurement in fresh processes covers lcs and scip at n = 100 and n = 1000.
- Gurobi is used only where its license allows (n = 30).
- No second operational dataset with verified service metadata was available; the demand-trace study uses UCI electricity traces with constructed constraints (see `DATA_SOURCES.md`).

## 8. Changes during the runs

- Scheduling (30 October 2026, before any record of these designs existed): once the demand-trace study had finished, its core ran the stress, sensitivity and preference designs in a separate runner process with one worker, while the main, evolutionary and ablation runs continued on three workers. The machine kept four busy single-threaded processes throughout, as before. The records do not depend on this order.
- Interruption (30 October 2026): the machine was paused from 21:20 to 23:53 UTC. All files were intact: every record line parses and no (instance, model, method) record appears twice. The runs resumed at 23:56 UTC with the restartable runner, which skips existing records; tasks that were running at the interruption had written nothing and were run again from the start. No record was changed or removed.
- Claims (same date, before any record of the stress_dimension, sensitivity and preference designs existed): each of the four n = 100 designs is run by exactly one runner, the one that claims it first through an atomic directory claim. The one-worker runner on the fourth core claims them in order; the three-worker runner claims any design still unclaimed after the main, evolutionary and ablation runs. The peak-memory measurement then runs on the fourth core with one worker.
- Epsilon-constraint sample at p = 8 (same date, before any record of stress_dimension existed): the frozen sample has 8^(p-1) grid points, that is 2,097,152 convex calls per pair at p = 8, about six hours per pair at the measured call time. It is therefore run on p in {2, 3, 5} only (runner filter `p=2|3|5`); lcs, enum and scip run on all four values of p. No code changed.
- Other load (31 October 2026): short analysis scripts and other light jobs ran on the same machine while the runs continued, for a few seconds to a few minutes at a time, most of them at the lowest scheduling priority (`nice -n 19`). They read the record files and never wrote to them. Times measured during these jobs may be slightly longer; no task was rerun because of them.
- Ablation labels (31 October 2026, after 797 of the 1,080 ablation records had been written): the frozen runner copies the record of the solver pipeline over its own, and the pipeline reports its base method, so every record of lcs_nopoints, lcs_nohalf and lcs_nocache was labeled "lcs". The run was stopped and the whole ablation design was run again with `scripts/run_ablation.py`, which calls the same frozen runner and restores the label of each record; the solver code and its hash are unchanged, and the rerun used four workers because nothing else was running. The unlabeled partial records are not analyzed and not published. All 797 of them reappear in the rerun with the same key, call counts, decision counts, objective vector and status. Ablations build no certificate, so the analysis compares their search times, not their total times, with those of LCS.
- Comparators from the literature (31 October 2026, after every frozen record existed; exploratory): to compare LCS with methods built for reference levels under the same budget, `comparators/methods.py` adds the reference point method (augmented achievement minimization for each profile in consultation order, then the best key), lexicographic goal programming with the profiles as goals and an efficiency step, an epsilon-constraint sample on nested grids refined until the budget is spent, NSGA-II, the RMP-guided NSGA-II and R-NSGA-II with the profiles as reference points, run until the budget is spent with the Euclidean projection onto the priority set on every structure, and the MIQCP with SCIP indicator constraints instead of big-M rows. `scripts/run_literature.py` runs them through the frozen runner, together with lcs, on the main pairs with seeds 1000 to 1004 (360 pairs, 2,880 tasks), on the machine of the frozen runs with four single-threaded workers and 60 s per task, and writes `results/v1/literature/literature.jsonl`. Every record carries the frozen code hash, unchanged, and `extra_sha`, the hash of the added code. Before the run, a smoke run of 24 pairs at a 10 s budget served only to find failures; it revealed two programming errors (the payoff-table allocations of the epsilon-constraint sample and the generation count of the evolutionary searches), which were fixed, and its records are excluded from every comparison. The outcome is the share of pairs where the key of the returned allocation, evaluated in floating point, equals the certified key of LCS, at 60 s and at the time LCS needed on the same pair. These results are descriptive and outside the confirmatory family. The machine was paused once during this run, from about 12:28 to 12:50 UTC: the files were intact (every line parses and no task appears twice), the restartable runner skipped the 72 completed tasks, and the tasks running at the pause had written nothing and were run again from the start. Short analysis jobs and document builds ran at the lowest scheduling priority (`nice -n 19`) during this run.

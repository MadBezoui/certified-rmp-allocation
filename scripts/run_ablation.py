"""Ablation runs of the benchmark runner with their method label (protocol v1, section 8).

Usage: python run_ablation.py <manifest.json> <results.jsonl> <methods> [workers] [time_limit] [filter]
Arguments are those of experiments/runner.py. The runner copies the record of the solver
pipeline over its own, and the pipeline reports its base method ("lcs"), so the records of
lcs_nopoints, lcs_nohalf and lcs_nocache would all be labeled "lcs". This script runs the
same runner and restores the label of each record. The solver code is unchanged. The code
root is the parent of this folder, or RMP_CODE_ROOT when set.
"""
import os
import pathlib
import sys

ROOT = pathlib.Path(os.environ.get("RMP_CODE_ROOT", pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "experiments"))

import runner  # noqa: E402

_run_task = runner.run_task


def run_task(args):
    rec = _run_task(args)
    rec["method"] = args[2]
    return rec


runner.run_task = run_task

if __name__ == "__main__":
    runner.main()

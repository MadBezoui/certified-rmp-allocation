"""Peak memory of single benchmark tasks, each in a fresh process (protocol v1, section 7).

Usage: python measure_memory.py <manifest_main.json> <out.jsonl> [workers] [code_root]
Runs lcs and scip on the m = 4 entries of seeds 1000 and 1001 at n = 100 and n = 1000
(three structures, both preference models) and records the peak resident set size of the
child process (ru_maxrss of the process that ran the task, in MiB).
"""
import json
import pathlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHILD = r"""
import json, resource, sys
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[1] + "/experiments")
from runner import run_task
entry = json.loads(sys.stdin.read())
rec = run_task((entry["entry"], entry["model"], entry["method"], 60.0))
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
print(json.dumps({"id": rec["id"], "model": rec["model"], "method": rec["method"], "n": rec["n"],
                  "status": rec.get("status"), "seconds": rec.get("seconds"), "peak_mib": peak}))
"""


def run(job, code_root=ROOT):
    out = subprocess.run([sys.executable, "-c", CHILD, str(code_root)], input=json.dumps(job), capture_output=True,
                         text=True, env={"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                                         "PATH": "/usr/bin:/bin"})
    lines = [l for l in out.stdout.splitlines() if l.startswith("{")]
    if not lines:
        return {"id": job["entry"]["id"], "model": job["model"], "method": job["method"], "status": "ERROR",
                "error": out.stderr[-300:]}
    return json.loads(lines[-1])


def main():
    manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())["entries"]
    out = pathlib.Path(sys.argv[2])
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    code_root = pathlib.Path(sys.argv[4]).resolve() if len(sys.argv) > 4 else ROOT
    jobs = [{"entry": e, "model": mi, "method": meth}
            for e in manifest if e["n"] in (100, 1000) and e["m"] == 4 and e["seed"] in (1000, 1001)
            for mi in range(len(e["models"])) for meth in ("lcs", "scip")]
    with ThreadPoolExecutor(workers) as pool, out.open("w") as fh:
        for rec in pool.map(lambda j: run(j, code_root), jobs):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(rec, flush=True)


if __name__ == "__main__":
    main()

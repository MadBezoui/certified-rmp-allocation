"""Download the UCI electricity archive, check its hash and prepare the demand traces.

Usage: python scripts/download_data.py
Writes data/raw/electricity.zip, data/processed/electricity.npz and data/source_manifest.json.
The processed demand array is checked against the content hash used in the experiments.
"""
import datetime
import hashlib
import json
import pathlib
import zipfile

import numpy as np
import pandas as pd
import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
PROCESSED = ROOT / "data/processed"
SOURCE = {
    "url": "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip",
    "doi": "10.24432/C58C86", "creator": "Artur Trindade (2015)", "license": "CC BY 4.0",
    "page": "https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014",
    "expected_sha256": "f6c4d0e0df12ecdb9ea008dd6eef3518adb52c559d04a9bac2e1b81dcfc8d4e1",
}
DEMAND_SHA256 = "f5492f007e61cfd402b7b83fe29aaa59c1349ed7e98d6c0f0df079fa678f151c"


def download():
    target = RAW / "electricity.zip"
    fresh = not target.exists()
    if fresh:
        partial = target.with_suffix(".partial")
        with requests.get(SOURCE["url"], stream=True, timeout=(30, 180)) as r:
            r.raise_for_status()
            with partial.open("wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    f.write(chunk)
        with zipfile.ZipFile(partial) as z:
            if z.testzip() is not None:
                raise RuntimeError("corrupt dataset archive")
        partial.rename(target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if digest != SOURCE["expected_sha256"]:
        raise RuntimeError("archive checksum mismatch; refusing changed inputs")
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return dict(SOURCE, sha256=digest, bytes=target.stat().st_size, verified_utc=now, downloaded=fresh)


def process():
    with zipfile.ZipFile(RAW / "electricity.zip") as z:
        member = next(n for n in z.namelist() if n.endswith("LD2011_2014.txt") and not n.startswith("__MACOSX"))
        sums, count, columns = None, 0, None
        with z.open(member) as f:
            for chunk in pd.read_csv(f, sep=";", decimal=",", index_col=0, chunksize=4096):
                train = chunk.loc[(chunk.index >= "2012-01-01") & (chunk.index < "2013-01-01")]
                if len(train):
                    values = train.to_numpy(float)
                    sums = values.sum(axis=0) if sums is None else sums + values.sum(axis=0)
                    count += len(train)
                    columns = list(chunk.columns)
        means = sums / count
        chosen = np.flatnonzero(means > 0)[:32]
        selected = [columns[i] for i in chosen]
        test, timestamps = [], []
        with z.open(member) as f:
            for chunk in pd.read_csv(f, sep=";", decimal=",", index_col=0,
                                     usecols=[0] + [int(i + 1) for i in chosen], chunksize=4096):
                sub = chunk.loc[(chunk.index >= "2013-01-01") & (chunk.index < "2014-01-01")]
                if len(sub):
                    test.append(sub.to_numpy(float) / means[chosen])
                    timestamps.extend(sub.index)
    data = np.concatenate(test)
    assert np.isfinite(data).all() and (data >= 0).all()
    digest = hashlib.sha256(np.ascontiguousarray(data, dtype=np.float64).tobytes()).hexdigest()
    if digest != DEMAND_SHA256:
        raise RuntimeError("processed demand differs from the array used in the experiments")
    np.savez_compressed(PROCESSED / "electricity.npz", demand=data, timestamps=np.asarray(timestamps),
                        train_total=32.0, critical=np.arange(8), columns=np.asarray(selected))
    return {"train_year": 2012, "train_rows": count, "test_year": 2013, "test_rows": len(data),
            "selected_columns": selected, "normalization": "each client divided by its 2012 mean kW",
            "critical_policy": "first 8 selected clients; simulated, not observed",
            "demand_sha256": digest}


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    manifest = {"electricity": download()}
    manifest["processing"] = process()
    (ROOT / "data/source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))

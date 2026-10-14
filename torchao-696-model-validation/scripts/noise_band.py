# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BF16 run-to-run noise band per sequence, and whether each candidate falls inside it.

Every run is scored against the same round-1 BF16 reference. The band for a
(resolution, label) is the worst value over all BF16 reruns of each gate metric;
a candidate is inside when no metric is worse than that band edge.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
ROUND1 = Path("<REPO>/artifacts/experiment-supplement-20261013/cluster-run")
HIGHER_IS_WORSE = ("mean_abs", "rmse", "max_frame_mean_abs", "max_frame_rmse", "mean_flip", "max_frame_flip")


def summarize(metrics):
    frames = metrics["frame_metrics"]
    flips = [frame["mean_flip"] for frame in frames if frame.get("mean_flip") is not None]
    return dict(mean_abs=metrics["mean_abs"], rmse=metrics["rmse"], psnr_db=float(metrics["psnr_db"]), mean_flip=metrics.get("mean_flip"),
                max_frame_mean_abs=max(frame["mean_abs"] for frame in frames), max_frame_rmse=max(frame["rmse"] for frame in frames),
                max_frame_flip=max(flips) if flips else None)


def inside(row, band):
    worse = [key for key in HIGHER_IS_WORSE if row[key] is not None and row[key] > band[key]]
    if row["psnr_db"] < band["psnr_db"]:
        worse.append("psnr_db")
    return worse


def band_of(rows):
    return {**{key: max(row[key] for row in rows) for key in HIGHER_IS_WORSE}, "psnr_db": min(row["psnr_db"] for row in rows)}


def load():
    """Yield (resolution, kind, variant, label, metrics) for every scored rollout."""
    for backend in ("flashdreams", "torchao"):
        for resolution in ("1280x704", "1168x640"):
            record = json.loads((ROUND1 / "696/quality" / f"{resolution}-{backend}-compile_graph" / "result.json").read_text())
            for rollout in record["rollouts"]:
                if rollout["label"].startswith("short") and rollout.get("metrics"):
                    yield resolution, "fp8", backend, rollout["label"], summarize(rollout["metrics"])
    for path in sorted((ROOT / "quality").glob("*/result.json")):
        resolution, variant = path.parent.name.split("-", 1)
        kind = "bf16" if variant.startswith("bf16_rerun") else "bf16_options" if variant.startswith("bf16_numeric") else "fp8"
        for rollout in json.loads(path.read_text())["rollouts"]:
            if rollout.get("metrics"):
                yield resolution, kind, variant, rollout["label"], summarize(rollout["metrics"])


def bitwise_pairs():
    """Exact RGB equality between BF16 reruns sharing a compile cache vs. not."""
    rows = []
    for resolution in ("1280x704", "1168x640"):
        runs = {name: ROOT / "raw-quality" / f"{resolution}-bf16_rerun-band-{name}" for name in ("cold0", "cold1", "cold2", "warm0", "warm1")}
        for a, b in (("cold0", "warm0"), ("cold0", "warm1"), ("warm0", "warm1"), ("cold0", "cold1"), ("cold0", "cold2"), ("cold1", "cold2")):
            if not (runs[a].exists() and runs[b].exists()):
                continue
            labels = sorted(path.name for path in runs[a].iterdir())
            equal = sum(all(np.array_equal(np.load(runs[a] / label / f"rgb-{i:03d}.npy"), np.load(runs[b] / label / f"rgb-{i:03d}.npy")) for i in range(10)) for label in labels)
            rows.append(dict(resolution=resolution, pair=f"{a}-{b}", same_compile_cache="warm" in a + b, bitwise_equal_labels=equal, labels=len(labels)))
    return rows


def main():
    if "--self-check" in sys.argv:
        bf16 = [dict(mean_abs=1, rmse=2, psnr_db=35, mean_flip=.05, max_frame_mean_abs=3, max_frame_rmse=4, max_frame_flip=.07),
                dict(mean_abs=2, rmse=3, psnr_db=33, mean_flip=.06, max_frame_mean_abs=4, max_frame_rmse=5, max_frame_flip=.09)]
        band = band_of(bf16)
        assert band["mean_abs"] == 2 and band["psnr_db"] == 33
        assert inside(dict(bf16[0]), band) == []
        assert inside(dict(bf16[1], mean_abs=2.5, psnr_db=32), band) == ["mean_abs", "psnr_db"]
        print("noise_band self-check passed")
        return
    records = list(load())
    bands, out = {}, []
    for resolution, kind, variant, label, metrics in records:
        if kind == "bf16":
            bands.setdefault((resolution, label), []).append(metrics)
    for resolution, kind, variant, label, metrics in records:
        reference = bands.get((resolution, label), [])
        others = [row for row in reference if row is not metrics]  # leave-one-out for BF16 members
        if len(others) < 2:
            continue
        worse = inside(metrics, band_of(others))
        out.append(dict(resolution=resolution, kind=kind, variant=variant, label=label, bf16_runs_in_band=len(others),
                        inside_band=not worse, worse_than_band="; ".join(worse), **metrics))
    with (ROOT / "noise-band.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    pairs = bitwise_pairs()
    (ROOT / "noise-band-bitwise.json").write_text(json.dumps(pairs, indent=2) + "\n")
    summary = {}
    for row in out:
        key = (row["kind"], row["variant"].split("-band")[0], row["label"].split("-")[1])
        summary.setdefault(key, [0, 0])[row["inside_band"]] += 1
    for (kind, variant, motion), (outside, inside_count) in sorted(summary.items()):
        print(f"{kind:13s} {variant:22s} {motion:9s} inside {inside_count}/{inside_count + outside}")
    for row in pairs:
        print(row)


if __name__ == "__main__":
    main()

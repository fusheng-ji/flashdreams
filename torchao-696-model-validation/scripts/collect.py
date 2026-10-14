# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flatten round-2 results (plus round-1 FP8 rows for comparison) into CSVs."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ROUND1 = Path("<REPO>/artifacts/experiment-supplement-20261013/cluster-run")


def write(name, rows):
    if not rows:
        print(f"{name}: no rows")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (ROOT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{name}: {len(rows)} rows")


def quality_row(source, resolution, variant, rollout):
    metrics = rollout.get("metrics") or {}
    frames = metrics.get("frame_metrics", [])
    flips = [frame["mean_flip"] for frame in frames if frame.get("mean_flip") is not None]
    return dict(source=source, resolution=resolution, variant=variant, label=rollout["label"], status=rollout["status"],
                gate="failed" if rollout.get("quality_failures") else ("passed" if metrics else "n/a"),
                bitwise_equal_reference=rollout.get("bitwise_equal_reference"), mean_abs=metrics.get("mean_abs"), rmse=metrics.get("rmse"),
                psnr_db=metrics.get("psnr_db"), mean_flip=metrics.get("mean_flip"),
                max_frame_mean_abs=max((frame["mean_abs"] for frame in frames), default=None), max_frame_flip=max(flips, default=None),
                failures="; ".join(rollout.get("quality_failures") or []), bf16_units=None)


def main():
    rows = []
    for backend in ("flashdreams", "torchao"):
        for resolution in ("1280x704", "1168x640"):
            record = json.loads((ROUND1 / "696/quality" / f"{resolution}-{backend}-compile_graph" / "result.json").read_text())
            rows += [quality_row("round1", resolution, backend, rollout) for rollout in record["rollouts"]
                     if rollout["label"].startswith(("short-straight", "short-turn"))]
    for path in sorted((ROOT / "quality").glob("*/result.json")):
        record = json.loads(path.read_text())
        resolution, variant = path.parent.name.split("-", 1)
        for rollout in record["rollouts"]:
            rows.append(dict(quality_row("round2", resolution, variant, rollout), bf16_units=",".join(record.get("bf16_units", [])) or None))
    write("quality-ablation.csv", rows)
    layer_rows = []
    for path in sorted((ROOT / "layers").glob("*/result.json")):
        record = json.loads(path.read_text())
        layer_rows += [dict(resolution=path.parent.name.split("-")[0], **row) for row in record.get("rows", [])]
    write("per-layer-error.csv", layer_rows)
    matrix = ROOT / "matrix-results.json"
    if matrix.exists():
        state = json.loads(matrix.read_text())
        write("ptxas-stress.csv", [dict(id=row["id"], phase=row["phase"], card=row["card"], tmpdir=(row.get("env") or {}).get("TMPDIR"),
                                        status=row["status"], exit_code=row.get("exit_code"), ptxas_255=row.get("ptxas_255"),
                                        missing_tmp_log=row.get("missing_tmp_log"), seconds=row.get("seconds"))
                                   for row in state["runs"] if row["phase"].startswith("stress")])
        write("matrix-runs.csv", [dict(id=row["id"], phase=row["phase"], card=row["card"], args=" ".join(row["args"]), status=row["status"],
                                       seconds=row.get("seconds"), reason=row.get("reason")) for row in state["runs"]])


if __name__ == "__main__":
    main()

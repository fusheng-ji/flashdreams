# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Slurm driver for round-2 #696 ablations; one card per resolution, wall-clock budget enforced."""

import concurrent.futures
import datetime
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = Path("<REPO>/.venv-coverage-cpu-isolated/bin/python")
RESOLUTIONS = ((1280, 704), (1168, 640))
STRESS_ROUNDS = 10
BUDGET_S = 115 * 60  # ponytail: 2 cards x ~1.9 h, under the 4 GPU-hour cap; later phases become not-executed
SNAPSHOT = {
    "findmnt_tmp": "findmnt -T /tmp", "df_tmp": "df -h /tmp", "ls_tmp": "ls -ld /tmp; ls /tmp | wc -l",
    "timers": "systemctl list-timers --all --no-pager 2>&1 | grep -iE 'tmp|clean' || true",
    "tmpfiles": "systemd-tmpfiles --cat-config 2>/dev/null | grep -vE '^#|^$' | grep -E ' /tmp|/var/tmp' || true",
    "epilogs": "ls -la <slurm-epilog-dir> <slurm-epilog-dir> 2>&1; grep -lE '/tmp' <slurm-epilog-dir>* 2>/dev/null || true",
    "slurm_tmpfs": "scontrol show config 2>/dev/null | grep -iE 'TmpFS|job_container|Epilog' || true",
    "inotifywait": "command -v inotifywait || echo missing",
    "nvidia_smi": "nvidia-smi --query-gpu=index,uuid,name,driver_version --format=csv,noheader",
    "other_jobs_on_node": "squeue -w $(hostname -s) -h -o '%i %u %T' 2>/dev/null || true",
}


def quality(card, variant, keep="", tag=""):
    width, height = RESOLUTIONS[card]
    cmd = ["quality", "--width", str(width), "--height", str(height), "--variant", variant]
    return cmd + (["--keep", keep, "--tag", tag] if keep else [])


def plan(name="ablation"):
    """Static phases; 1e is appended after the layers phase from its results."""
    rows = []
    def add(phase, card, args, private_tmp=True, cache_from=None):
        rows.append(dict(id=len(rows), phase=phase, card=card, args=args, private_tmp=private_tmp, cache_from=cache_from, status="not-executed"))
    if name == "noise-band":
        # 3 cold-cache reruns (autotune + runtime variation), then 2 reruns reusing rerun c0's cache (runtime only).
        for card in (0, 1):
            first = len(rows)
            for index in range(3):
                add("quality", card, quality(card, "bf16_rerun") + ["--tag", f"band-cold{index}", "--static"])
            for index in range(2):
                add("quality", card, quality(card, "bf16_rerun") + ["--tag", f"band-warm{index}", "--static"], cache_from=first)
        return rows
    for card, (width, height) in enumerate(RESOLUTIONS):
        add("layers", card, ["layers", "--width", str(width), "--height", str(height)])
        for variant in ("bf16_rerun", "bf16_numeric_control", "fd_self_only", "fd_cross_only"):
            add("quality", card, quality(card, variant))
    for round_index in range(STRESS_ROUNDS):
        for group in ("shared", "private"):
            for card in (0, 1):
                for slot in (0, 1):
                    add(f"stress-{round_index}-{group}", card, ["component", "--tag", f"{group}-r{round_index}-c{card}-s{slot}"], private_tmp=group == "private")
    return rows


def keep_units(width, height, k=4):
    """Top-k projections by FlashDreams FP8 teacher-forced error, or None without a clear outlier."""
    rows = json.loads((ROOT / "layers" / f"{width}x{height}-layers" / "result.json").read_text())["rows"]
    per_unit = {}
    for row in rows:
        per_unit.setdefault(f"{row['block']}:{row['kind']}", []).append(row["fd_rel_l2"])
    score = {unit: statistics.fmean(values) for unit, values in per_unit.items()}
    ranked = sorted(score, key=score.get, reverse=True)
    # Self-attn errors sit ~3x above cross-attn uniformly; an outlier must stand out within its own kind.
    kind = ranked[0].split(":")[1]
    median = statistics.median(value for unit, value in score.items() if unit.endswith(kind))
    outlier = score[ranked[0]] >= 2 * median
    return dict(units=ranked[:k], scores={unit: score[unit] for unit in ranked[:k]}, median=median, outlier=outlier)


def main():
    name = sys.argv[sys.argv.index("--plan") + 1] if "--plan" in sys.argv else "ablation"
    rows = plan(name)
    suffix = "" if name == "ablation" else f"-{name}"
    results = ROOT / f"matrix-results{suffix}.json"
    if "--self-check" in sys.argv and name == "noise-band":
        assert len(rows) == 10 and sum(row["cache_from"] is not None for row in rows) == 4
        assert all(rows[row["cache_from"]]["card"] == row["card"] for row in rows if row["cache_from"] is not None)
        assert len({tuple(row["args"]) for row in rows}) == len(rows)
        print("noise-band: 6 cold + 4 warm-cache BF16 reruns")
        return
    if "--self-check" in sys.argv:
        assert len(rows) == 10 + STRESS_ROUNDS * 8
        assert sum(not row["private_tmp"] for row in rows) == STRESS_ROUNDS * 4
        assert len({tuple(row["args"]) for row in rows}) == len(rows)
        print(f"{len(rows)} planned processes + conditional keep-BF16 phase")
        return
    allocation = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    assert len(allocation) == 2, allocation
    assert not results.exists(), "results exist; do not rerun into this directory"
    state = dict(status="running", job_id=os.environ.get("SLURM_JOB_ID"), host=os.uname().nodename, allocation=allocation,
                 created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), budget_s=BUDGET_S, runs=rows)
    start = time.monotonic()
    def save():
        state["elapsed_s"] = time.monotonic() - start
        results.with_suffix(".tmp").write_text(json.dumps(state, indent=2) + "\n")
        results.with_suffix(".tmp").replace(results)
    state["node_snapshot"] = {key: subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, timeout=60).stdout for key, cmd in SNAPSHOT.items()}
    save()
    (ROOT / f"logs{suffix}").mkdir(exist_ok=True)

    def run(row):
        if time.monotonic() - start > BUDGET_S:
            row["reason"] = "budget exhausted"
            return
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=allocation[row["card"]], OMP_NUM_THREADS="4", MAX_JOBS="4", HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1",
                   TORCH_EXTENSIONS_DIR=str(ROOT / "extensions"), PATH=f"{PYTHON.parent}:{env['PATH']}")
        private = ROOT / f"private{suffix}" / f"{row['id']:03d}"
        (private / "tmp").mkdir(parents=True)
        cache = private if row.get("cache_from") is None else ROOT / f"private{suffix}" / f"{row['cache_from']:03d}"
        env.update(TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"), TRITON_CACHE_DIR=str(cache / "triton"))
        if row["private_tmp"]:
            env["TMPDIR"] = str(private / "tmp")
        else:
            env.pop("TMPDIR", None)
        row["env"] = {key: env.get(key) for key in ("CUDA_VISIBLE_DEVICES", "TMPDIR", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")}
        log = ROOT / f"logs{suffix}" / f"{row['id']:03d}-{row['phase']}.log"
        row.update(status="running", started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), log=str(log.relative_to(ROOT)))
        begin = time.monotonic()
        with log.open("w") as handle:
            code = subprocess.run([str(PYTHON), str(ROOT / "ablation_runs.py"), *row["args"]], cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                  timeout=max(60, BUDGET_S - (begin - start))).returncode
        text = log.read_text(errors="replace")
        row.update(exit_code=code, status="passed" if code == 0 else "failed", seconds=time.monotonic() - begin,
                   ptxas_255="exit status 255" in text, missing_tmp_log="FileNotFoundError" in text and ".log'" in text)
        print(json.dumps({key: row[key] for key in ("id", "phase", "card", "status", "seconds")}), flush=True)

    def run_phase(prefix, concurrent_slots=1):
        selected = [row for row in rows if row["phase"].startswith(prefix)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2 * concurrent_slots) as pool:
            if concurrent_slots == 1:
                list(pool.map(lambda card: [run(row) for row in selected if row["card"] == card], (0, 1)))
            else:
                list(pool.map(run, selected))
        save()

    try:
        if name == "noise-band":
            run_phase("quality")
            return
        run_phase("layers")
        state["keep_bf16"] = {}
        for card, (width, height) in enumerate(RESOLUTIONS):
            try:
                selection = keep_units(width, height)
            except Exception as error:
                selection = dict(error=repr(error), outlier=False)
            state["keep_bf16"][f"{width}x{height}"] = selection
            if selection.get("outlier"):
                rows.append(dict(id=len(rows), phase="quality-keep", card=card, args=quality(card, "fd_keep_bf16", ",".join(selection["units"]), "top4"),
                                 private_tmp=True, status="not-executed"))
        save()
        run_phase("quality")  # includes quality-keep rows, run after the four ablations per card
        for round_index in range(STRESS_ROUNDS):
            for group in ("shared", "private"):
                run_phase(f"stress-{round_index}-{group}", concurrent_slots=2)
    finally:
        state["status"] = "completed-execution" if time.monotonic() - start <= BUDGET_S else "budget-exhausted"
        state["counts"] = {status: sum(row["status"] == status for row in rows) for status in ("passed", "failed", "not-executed", "running")}
        save()


if __name__ == "__main__":
    main()

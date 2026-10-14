# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-chunk latent and RGB divergence of saved quality rollouts against BF16."""

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
ROUND1 = Path("<REPO>/artifacts/experiment-supplement-20261013/cluster-run/raw-quality")


def compare(reference, candidate):
    """Return relative L2, cosine and RGB MAE per chunk for one label."""
    rows = []
    for index in range(len(list(reference.glob("latent-*.pt")))):
        ref = torch.load(reference / f"latent-{index:03d}.pt", weights_only=True).float().flatten()
        cand = torch.load(candidate / f"latent-{index:03d}.pt", weights_only=True).float().flatten()
        rgb_ref = np.load(reference / f"rgb-{index:03d}.npy").astype(np.float32)
        rgb_cand = np.load(candidate / f"rgb-{index:03d}.npy").astype(np.float32)
        rows.append(dict(chunk=index, latent_rel_l2=((cand - ref).norm() / ref.norm()).item(),
                         latent_cosine=torch.nn.functional.cosine_similarity(cand, ref, dim=0).item(),
                         rgb_mae=float(np.abs(rgb_cand - rgb_ref).mean())))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, action="append", help="raw-quality roots (default: round 1)")
    parser.add_argument("--output", type=Path, default=ROOT / "latent-drift.csv")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp, "a"), Path(tmp, "b")
            for path, value in ((a, 1.0), (b, 2.0)):
                path.mkdir()
                torch.save(torch.full((4,), value), path / "latent-000.pt")
                np.save(path / "rgb-000.npy", np.full((1, 2, 2, 3), int(value), np.uint8))
            assert compare(a, a)[0] == dict(chunk=0, latent_rel_l2=0.0, latent_cosine=1.0, rgb_mae=0.0)
            row = compare(a, b)[0]
            assert abs(row["latent_rel_l2"] - 1) < 1e-6 and abs(row["latent_cosine"] - 1) < 1e-6 and row["rgb_mae"] == 1
        print("latent_drift self-check passed")
        return
    out = []
    for root in args.raw or [ROUND1]:
        for config in sorted(root.iterdir()):
            match = re.fullmatch(r"(\d+x\d+)-(.+?)(?:-(compile_graph|graph|eager))?", config.name)
            if not match:
                continue
            resolution, backend, execution = match.groups(default="compile_graph")
            reference = ROUND1 / f"{resolution}-bf16-{execution}"
            if config == reference:
                continue
            for label in sorted(path.name for path in config.iterdir() if path.name.startswith("short")):
                for row in compare(reference / label, config / label):
                    out.append(dict(resolution=resolution, backend=backend, label=label,
                                    motion=label.split("-")[1], seed=label.split("-")[2], **row))
            print(config.name, file=sys.stderr)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    print(f"{len(out)} rows -> {args.output}")


if __name__ == "__main__":
    main()

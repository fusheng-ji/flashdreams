# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small multiples of per-chunk latent drift vs BF16 (mean over 3 seeds)."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
sources = [Path(arg) for arg in sys.argv[1:]] or [ROOT / "latent-drift.csv"]
data = pd.concat(pd.read_csv(path) for path in sources).groupby(["backend", "resolution", "motion", "chunk"])["latent_rel_l2"].mean().reset_index()
colors = {"flashdreams": "#2a78d6", "torchao": "#eb6834", "bf16_rerun": "#1baf7a", "bf16_numeric_control": "#eda100",
          "fd_self_only": "#e87ba4", "fd_cross_only": "#4a3aa7", "fd_keep_bf16": "#008300"}
styles = {"1280x704": "-", "1168x640": "--"}
plt.rcParams.update({"axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e"})
motions = [motion for motion in ("static", "straight", "turn") if motion in set(data.motion)]
figure, axes = plt.subplots(1, len(motions), squeeze=False, figsize=(4 * len(motions), 3.8), sharey=True, facecolor="#fcfcfb")
axes = axes[0]
for axis, motion in zip(axes, motions):
    axis.set_facecolor("#fcfcfb")
    for (backend, resolution), group in data[data.motion == motion].groupby(["backend", "resolution"]):
        axis.plot(group.chunk, group.latent_rel_l2, styles[resolution], color=colors.get(backend, "#52514e"), linewidth=2, label=f"{backend} {resolution}")
    axis.set_title(motion, color="#0b0b0b")
    axis.set_xlabel("chunk")
    axis.grid(alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("latent relative L2 vs BF16")
axes[-1].legend(frameon=False, fontsize=8)
figure.tight_layout()
output = sources[-1].with_suffix(".png")
figure.savefig(output, dpi=150)
print(output)

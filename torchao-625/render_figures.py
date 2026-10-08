# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Wenbo Ji
import json
from collections import Counter
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
rows = json.loads((ROOT / 'data/current/measurements.json').read_text())
assert len(rows) == 54
assert set(Counter((r['backend'], r['execution'], r['cache_state']) for r in rows).values()) == {3}
for row in rows:
    raw = json.loads((ROOT / f"data/current/raw/{row['label']}.json").read_text())
    samples = raw['benchmarks'][0]['stats']['data']
    assert len(samples) == len(row['cuda_event_samples_ms']) == 50
    assert abs(median(samples) * 1000 - row['wall_median_ms']) < 1e-10
    assert row['returncode'] == 0 and row['eager_max_abs_error'] == 0

plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'svg.hashsalt': 'torchao-625'})
colors = ['#31688e', '#d28c22', '#7d4b96']
modes = ['eager', 'compile', 'graph', 'compile_graph']
fig, ax = plt.subplots(figsize=(11, 5.8))
for i, (backend, label, color) in enumerate(zip(
    ['bf16', 'flashdreams', 'torchao'], ['BF16', 'FlashDreams FP8', 'torchao FP8'], colors
)):
    groups = [[r['wall_median_ms'] * 1000 for r in rows
               if r['backend'] == backend and r['execution'] == mode and r['cache_state'] != 'cold']
              for mode in modes]
    assert all(len(g) == 3 for g in groups)
    values = [median(g) for g in groups]
    errors = [[v - min(g) for v, g in zip(values, groups)],
              [max(g) - v for v, g in zip(values, groups)]]
    bars = ax.bar(np.arange(4) + (i - 1) * .25, values, .23,
                  yerr=errors, capsize=3, color=color, label=label)
    ax.bar_label(bars, labels=[f'{v:.2f}' for v in values], padding=5, fontsize=9)
ax.set_xticks(range(4), ['Eager', 'Compile', 'CUDA Graph', 'Compile + Graph'])
ax.set_ylabel('Synchronized end-to-end latency (µs) · lower is better')
ax.set_ylim(0, 195)
ax.legend(loc='upper right', frameon=False)
ax.set_axisbelow(True)
ax.grid(axis='y', alpha=.18)
fig.suptitle('Numerical compatibility is fixed; this projection still favors BF16',
             fontsize=15, fontweight='bold', y=.97)
ax.set_title('B200 · M=4800, K=N=2048 · seed 42 · PyTorch 2.12.1 / torchao 0.18.0', fontsize=10, pad=14)
fig.text(.08, .06, 'Median of 3 process medians; whiskers = process min–max (not confidence intervals).\n'
         '5 warmups + 50 wall samples/process. Compiled: reused cache; others: fresh.\n'
         'Only compiled torchao uses both local numerical options. Component result, not model throughput.', fontsize=9)
fig.subplots_adjust(left=.08, right=.98, bottom=.22, top=.83)
for suffix in ('png', 'svg'):
    fig.savefig(ROOT / f'steady_latency.{suffix}', dpi=180, metadata={'Date': None} if suffix == 'svg' else {})
plt.close(fig)

policies = ['default', 'casts', 'division', 'both']
data = [json.loads((ROOT / f'data/diagnosis/{p}.json').read_text()) for p in policies]
assert all(d['identical_inputs_gemm']['exact'] for d in data)
assert data[-1]['qdata']['exact'] and data[-1]['scales']['exact'] and data[-1]['full_projection']['exact']
fig, axes = plt.subplots(1, 3, figsize=(12, 5.4))
labels = ['Default', 'Casts only', 'Division only', 'Both']
for ax, key, metric, title in zip(axes, ['scales', 'qdata', 'full_projection'],
                                 ['mismatches', 'mismatches', 'max_abs'],
                                 ['Differing scales / 4,800', 'Differing FP8 values / 9,830,400', 'Output max absolute error']):
    values = [d[key][metric] for d in data]
    bars = ax.barh(range(4), values, color=['#b55b50'] * 3 + ['#34856c'], height=.56)
    ax.bar_label(bars, labels=[f'{v:,g}' for v in values], padding=5, fontsize=11)
    ax.set_yticks(range(4), labels)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=10, pad=15)
    ax.set_xlim(0, max(values) * 1.38)
    if key == 'qdata':
        ax.set_xticks([0, 100000, 200000], ['0', '100k', '200k'])
    ax.grid(axis='x', alpha=.18)
    ax.set_axisbelow(True)
fig.suptitle('Both local options restore eager quantization and projection output',
             fontsize=14, fontweight='bold', y=.97)
fig.text(.05, .07, 'Independent processes · B200 · M=4800, K=N=2048 · seed 42 · linear axes\n'
         'Casts = emulate_precision_casts; Division = eager_numerics.division_rounding.\n'
         'Identical-operand GEMM is exact in all four policies: the observed difference originates in activation quantization.', fontsize=9)
fig.subplots_adjust(left=.1, right=.98, bottom=.25, top=.81, wspace=.65)
for suffix in ('png', 'svg'):
    fig.savefig(ROOT / f'numerical_compatibility.{suffix}', dpi=180, metadata={'Date': None} if suffix == 'svg' else {})
plt.close(fig)
print('Verified 54 processes and 5,400 timing samples; wrote 4 figures.')

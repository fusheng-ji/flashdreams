# FP8 attention output projections in OmniDreams: model-level validation

**Model.** OmniDreams single-view 2B distilled (720p, 30 fps, HD-map conditioned), using the frozen PR head `10fd89b`.

**Pipeline.** Single GPU, batch 1, chunk 2, window 6, sink 0. Optimized attention uses cuDNN SDPA with BF16 QKV. FP8 applies only to the self- and cross-attention output projections.

**Execution.** Compile + CUDA graph. Compiled torchao and the "BF16 + options" control use `emulate_precision_casts=True` and `eager_numerics.division_rounding=True`.

## Sequences and scoring

**Inputs.**
- Resolutions: 1280×704 and 1168×640.
- Motion: one scene, with fixed HD-map trajectories for static, straight and turn.
- Seeds: 0, 1 and 42, applied to a dedicated noise generator.
- Short sequences have 10 chunks (77 frames).
- Long sequences have 120 chunks (957 frames). A reset run follows each long run.

**Scoring.**
- Every run receives the same per-chunk initial noise as the single BF16 reference rollout; SHA-256 is checked per chunk.
- RGB is scored against that reference with the repository's clip comparison.
- Existing short-clip gates: MAE ≤ 4, RMSE ≤ 8, PSNR ≥ 30 dB, worst-frame MAE ≤ 8 and RMSE ≤ 14, mean FLIP ≤ 0.07, worst-frame FLIP ≤ 0.08.

## 1. Throughput

| Resolution | Fastest non-FP8 path | torchao FP8, compile + graph | Change |
| --- | ---: | ---: | ---: |
| 1280×704 | 34.30 FPS | 34.38 FPS | +0.23% |
| 1168×640 | 44.81 FPS | 44.94 FPS | +0.28% |

- These are medians of three independent processes, timing HD-map encode, model, decode and finalize.
- `data/pipeline-throughput.csv` has all 108 Slurm processes: five backends, three execution modes, and cold and reused caches.

## 2. Absolute gates cannot separate FP8 from BF16 nondeterminism

- **Both FP8 backends fail the gates.** They fail on 12 of 12 straight and turn short sequences per resolution and pass all static ones.
- **So does an unchanged BF16 rerun.** It fails 12 of 12; at 1280×704 straight, only the worst-frame FLIP limit (0.083–0.093) is exceeded.
- **BF16 is not bitwise reproducible.** Across 12 rerun pairs, including pairs sharing one compile cache, 0 of 9 sequences match bitwise (`noise-band-bitwise.json`).

Mean MAE / PSNR (dB) / mean FLIP against the reference, averaged over three seeds:

| Variant | 1280 straight | 1280 turn | 1168 straight | 1168 turn |
| --- | --- | --- | --- | --- |
| BF16 rerun | 2.13 / 32.9 / 0.048 | 3.52 / 28.1 / 0.068 | 2.83 / 30.5 / 0.059 | 3.86 / 27.9 / 0.074 |
| BF16 + options | 2.65 / 31.6 / 0.058 | 3.95 / 27.8 / 0.076 | 3.26 / 29.9 / 0.066 | 4.60 / 26.7 / 0.085 |
| FP8 self-attn only | 2.76 / 31.2 / 0.059 | 4.21 / 27.3 / 0.080 | 3.42 / 29.4 / 0.069 | 4.94 / 26.7 / 0.092 |
| FP8 cross-attn only | 2.76 / 31.4 / 0.059 | 4.38 / 27.1 / 0.082 | 3.53 / 29.3 / 0.071 | 4.94 / 26.4 / 0.091 |
| FlashDreams FP8 | 2.71 / 31.4 / 0.058 | 4.10 / 27.6 / 0.078 | 3.39 / 29.5 / 0.069 | 4.67 / 26.8 / 0.086 |
| torchao FP8 | 2.96 / 30.9 / 0.063 | 4.24 / 27.5 / 0.081 | 3.59 / 29.4 / 0.072 | 5.33 / 26.1 / 0.097 |

### Errors compound across chunks

Relative L2 of the model output against the reference, before decoding, averaged over seeds and resolutions:

| Variant | Chunk 0 | 1 | 2 | 5 | 9 straight | 9 turn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 rerun | 0.014 | 0.035 | 0.062 | 0.116 | 0.265 | 0.454 |
| BF16 + options | 0.023 | 0.055 | 0.083 | 0.154 | 0.279 | 0.473 |
| FlashDreams FP8 | 0.026 | 0.055 | 0.080 | 0.158 | 0.294 | 0.456 |
| torchao FP8 | 0.029 | 0.066 | 0.095 | 0.166 | 0.300 | 0.480 |

Chunks 0–5 show straight values; turn differs by less than 0.01 in that range. FP8 starts at about twice the BF16 rerun's divergence, but every variant converges to the same level by chunk 9 (see `latent_drift.png`).

**Long runs.** All 12 long runs finished (120 chunks, 31.9 s) with finite values, and all 12 reset checks passed. Long-run drift is reported descriptively only.

## 3. Relative to BF16 noise, FP8 is consistently worse

Each sequence has six BF16 reruns (five for static): three with cold compile caches, two reusing the first run's cache, plus one earlier rerun. MAE is shown as a multiple of the median BF16 rerun for the same sequence; the bracket counts sequences above every BF16 rerun.

| Variant | Static | Straight | Turn |
| --- | --- | --- | --- |
| BF16 rerun | 0.86–1.30 | 0.88–1.18 | 0.87–1.15 |
| BF16 + options | – | 1.11–1.49 (6/6) | 1.10–1.26 (5/6) |
| FP8 self-attn only | – | 1.20–1.45 (6/6) | 1.12–1.38 (6/6) |
| FP8 cross-attn only | – | 1.23–1.49 (6/6) | 1.19–1.40 (6/6) |
| FlashDreams FP8 | 1.44–1.67 (6/6) | 1.23–1.31 (6/6) | 1.04–1.30 (5/6) |
| torchao FP8 | 1.56–2.00 (6/6) | 1.18–1.73 (6/6) | 1.18–1.57 (6/6) |

BF16 rerun MAE ranges, in 8-bit units:

| Resolution | Static | Straight | Turn |
| --- | --- | --- | --- |
| 1280×704 | 0.49–0.85 | 2.00–2.38 | 3.21–3.82 |
| 1168×640 | 0.50–0.76 | 2.26–3.06 | 3.26–4.44 |

FP8 static MAE is 0.93 for FlashDreams and 1.06 for torchao. Mean FLIP follows the same pattern, at 1.2–1.5× the BF16 median. Worst-frame FLIP on turn sequences overlaps BF16 (1.00–1.04×).

**This band is not a gate.** A per-sequence rule "no metric worse than the other BF16 reruns" admits only 62/102 BF16 reruns, using leave-one-out. A usable gate needs a ratio-based tolerance against the BF16 median, which is a maintainer decision.

## 4. Error is uniform across layers

Method: a BF16 eager rollout (turn, seed 42, 10 chunks), with every one of the 56 output projections also evaluated through both FP8 backends on the same input (teacher-forced).

- Relative L2 is 3.1–3.7% for self-attention and about 1.1% for cross-attention.
- The two backends agree to four decimals.
- No nonzero activation underflows to zero.
- The largest projection error is 1.2× the median of its kind.
- With no outlier layers, keeping a subset of layers in BF16 was not run.

## 5. Unrelated infrastructure failure: `ptxas` exit 255

**Symptom.** In the first sweep, 1 of 1008 component captures failed: `ptxas` returned 255, then Triton raised `FileNotFoundError` for its temporary `.log`.

**Cause.** The cluster's Slurm epilog runs `find /tmp -maxdepth 1 -user "$SLURM_JOB_USER" -delete` whenever any job of that user ends. That removes `tmp*.ptx` and `tmp*.log` files belonging to the user's other jobs that are still compiling.

**Evidence.**
- Injecting deletion of both files into Triton's `make_cubin` reproduces the exact traceback. Deleting only the `.log` compiles fine, and deleting only the `.ptx` raises a plain `PTXASError` (`ptxas-mechanism.json`).
- Stress test, 10 rounds × 4 concurrent cold torchao compiles:
  - Shared `/tmp`: 5 of 40 hits, all in two rounds where another of the user's jobs ended on the node.
  - Private `TMPDIR`: 0 of 40.
  - One round matches the job's recorded end time to the second; the other precedes it by 9 s.
- The original failure window also contains two such job ends.

**Mitigation.** Use a private `TMPDIR` per process. This is unrelated to the PR's code. The "temporary-file failure" rerun disclosed in `../torchao-625` may have the same cause, but that was not verified.

## Limits

- One scene, single view, short rollouts; quality is measured as reference agreement, not perceptual preference.
- The runtime kernel behind BF16 nondeterminism was not isolated.
- The BF16 reference and the reruns ran on different B200 nodes.

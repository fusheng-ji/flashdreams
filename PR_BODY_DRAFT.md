Refs NVIDIA/flashdreams#625

This component investigation adds an experimental, explicitly selected torchao
FP8 backend for accelerated attention output projections. The default backend,
checkpoint parameters and BF16 fallback remain unchanged; torchao is optional
and imported lazily. Derived quantized weights are non-persistent and rebuilt
after checkpoint loading or device migration. Zero/underflow weight rows are
explicitly rejected, and weight changes require fresh compilation and graph
reset/recapture.

## Compiler compatibility

The default compiler policy changes activation quantization through eliminated
intermediate BF16 casts and approximate division. Four independent-process
stage comparisons isolate this from GEMM: identical quantized operands produce
identical GEMM results under every tested policy. Enabling both local options
restores exact scales, FP8 values and projection output in the seed-42 diagnosis:

```python
options={
    "emulate_precision_casts": True,
    "eager_numerics.division_rounding": True,
}
```

`compile_module` accepts explicit local options, merges them over the selected
mode configuration, and preserves its existing default path. Neither the
caller dictionary nor process-wide numerical settings are modified. Only the
experimental torchao benchmark's compiled paths enable these options.

![Four-policy numerical diagnosis](https://raw.githubusercontent.com/fusheng-ji/flashdreams/4cb453047cd66c9d447fad828c21cb0ca3b7cd40/torchao-625/numerical_compatibility.png)

## Component results and recommendation

On one allocated B200 with PyTorch 2.12.1+cu130 / torchao 0.18.0, the corrected
sweep passes **54/54 processes**: three repetitions across BF16, existing FP8
and torchao FP8, four execution modes, plus cold/reused compile caches.
The input is BF16, M=4800, K=N=2048, no bias, seed 42. Each process uses five
warmups, 50 synchronized wall measurements and a separate 50-sample CUDA event
pass. Activation quantization is included in steady-state time.

![Projection steady-state latency](https://raw.githubusercontent.com/fusheng-ji/flashdreams/4cb453047cd66c9d447fad828c21cb0ca3b7cd40/torchao-625/steady_latency.png)

Bars show the median of three process medians; whiskers show process min/max,
not confidence intervals. Compiled bars use reused caches in new processes;
eager and graph bars use fresh caches. All cold-cache results are retained.

- BF16 eager: **42.32 µs**; native FP8 eager: **64.36 µs**; torchao eager: **158.23 µs**.
- Compile + graph: BF16 **53.66 µs**, native FP8 **56.43 µs**, torchao **59.59 µs**.
- Torchao compiled first execution: **50.93 s cold / 4.94 s reused cache**
  (compile-only, medians); wrapping, preparation and capture are recorded separately.
- Native and torchao FP8 retain **12.0078 MiB** of source plus derived weights,
  versus **8 MiB** for BF16. This is resident weight payload, not total model memory.

**Not recommended for adoption on performance grounds for this tested component.**
The local options pass compatibility checks, but do not establish a speed or
memory advantage over BF16. Original and corrected sweeps used different
allocations, so their timing difference is not a controlled estimate of option
cost. Model quality and whole-model benefit remain untested; no generation was run.

## Validation

- 44 CPU tests; 15 GPU regressions; 6 compiled GPU cases spanning seeds 0/1/42
  and bias/no bias, including 3D/zero inputs, shape recompilation and old-output ownership.
- Real attention output-projection checkpoint reload, strict fresh-instance load,
  migration and graph reset/recapture checks.
- Existing `rtol=atol=0.02` cross-mode and BF16 relative-L2 limits retained;
  the formal sweep records zero max error against each backend's own eager output.
- Warm torchao caches record AOT/FX hits; fixed-shape compilations have no graph
  breaks. Subsequent default compilation reproduces the original numerical
  difference, verifying local-option isolation.
- Ruff, targeted type checks and documentation build passed.

## Reproducible evidence

All links below pin the asset commit `4cb453047cd66c9d447fad828c21cb0ca3b7cd40` on the fork's `pr_asset` history:

- [Full English report](https://github.com/fusheng-ji/flashdreams/blob/4cb453047cd66c9d447fad828c21cb0ca3b7cd40/torchao-625/REPORT.md): capability inventory, lifecycle,
  diagnosis, startup/memory/error tables and limitations.
- [Measurements JSON](https://github.com/fusheng-ji/flashdreams/blob/4cb453047cd66c9d447fad828c21cb0ca3b7cd40/torchao-625/data/current/measurements.json),
  [CSV](https://github.com/fusheng-ji/flashdreams/blob/4cb453047cd66c9d447fad828c21cb0ca3b7cd40/torchao-625/data/current/measurements.csv),
  [54 raw pytest-benchmark exports](https://github.com/fusheng-ji/flashdreams/tree/4cb453047cd66c9d447fad828c21cb0ca3b7cd40/torchao-625/data/current/raw),
  [commands and provenance](https://github.com/fusheng-ji/flashdreams/blob/4cb453047cd66c9d447fad828c21cb0ca3b7cd40/torchao-625/data/current/manifest.json).
- [Four-policy diagnostic data and mismatch examples](https://github.com/fusheng-ji/flashdreams/tree/4cb453047cd66c9d447fad828c21cb0ca3b7cd40/torchao-625/data/diagnosis),
  [original failure evidence](https://github.com/fusheng-ji/flashdreams/blob/4cb453047cd66c9d447fad828c21cb0ca3b7cd40/torchao-625/data/original/measurements.json),
  [validation logs](https://github.com/fusheng-ji/flashdreams/tree/4cb453047cd66c9d447fad828c21cb0ca3b7cd40/torchao-625/validation).
- [Figure regeneration script](https://github.com/fusheng-ji/flashdreams/blob/4cb453047cd66c9d447fad828c21cb0ca3b7cd40/torchao-625/render_figures.py),
  [methodology and redactions](https://github.com/fusheng-ji/flashdreams/blob/4cb453047cd66c9d447fad828c21cb0ca3b7cd40/torchao-625/README.md),
  [SHA-256 checksums](https://github.com/fusheng-ji/flashdreams/blob/4cb453047cd66c9d447fad828c21cb0ca3b7cd40/torchao-625/SHA256SUMS).

Public exports remove local paths, hostnames and detailed CPU identifiers;
all numeric observations are preserved. The base commit and source-diff hash are
recorded; this asset branch contains evidence, not the experimental implementation.

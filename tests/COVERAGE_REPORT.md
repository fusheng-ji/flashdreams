# Diagnostic CI coverage validation (#673)

Status: **partial — full CPU validation completed; full GPU validation blocked**.
Three complete CPU control/coverage pairs passed with identical per-case results.
The median paired collection overhead was **22.16%** (range **17.39–24.13%**).
No GPU suite or GPU overhead measurement ran: the required quality reference
returned HTTP 403. These results do not establish a complete repository or
upstream-hardware baseline.

## Implementation and provenance

The isolated `ci/diagnostic-coverage` branch starts at upstream
`fd52f1a1f4076f86f3fcb86cf5a4425497804b65` and contains no #696 commits.
All six timed runs and their report exports use the fixed checkout
`791e19b5b24dad98e636990e975240000d99f562`. Metadata records the complete source,
configuration and lock digests, tool/Python versions, selection, exit status,
wall/CPU time, memory and test outcomes. A subsequent test-only refinement
prevents a synthetic path-mapping sample from satisfying the real thread
coverage assertion; the collector and measured configuration are unchanged.

Coverage.py **7.16.2**, constrained by `coverage[toml]>=7.10,<8`, is the only
added package, in a separate development group. Its license is Apache-2.0.
There is no runtime dependency, external service, default pytest instrumentation,
percentage threshold or new publishing dependency. Native coverage APIs collect,
merge and export data; project code adds provenance checks and grouped summaries.
Raw inputs are retained. Failed, missing, corrupt or incompatible evidence is
explicitly partial, and report failures do not replace the test job's status.

The measurement scope includes the core library, production tools in both
`flashdreams/tools/` and `tools/`, applications and maintained integrations.
Namespace scanning includes unexecuted files. Tests, benchmarks, fixtures,
generated protobuf code, vendored native code and build products are excluded.
The reusable production benchmark harness remains in scope.

## Full CPU results and collection cost

Each of the six sessions collected **3,799** tests: **2,176 passed, 2 skipped,
1 manual, 1,620 deselected**, with exit status 0. All paired per-case JUnit
results, collection counts and manual-marker records matched. All three controls
contained no `.coverage` process files, including the control run after an
instrumented session.

| Pair | Order | Control wall seconds | Coverage wall seconds | Paired overhead |
|---|---|---:|---:|---:|
| 1 | Control, coverage | 667.024 | 828.000 | 24.13% |
| 2 | Coverage, control | 669.214 | 785.570 | 17.39% |
| 3 | Control, coverage | 650.646 | 794.835 | 22.16% |

The 22.16% figure is the median of the three paired ratios, not the ratio of
independently computed wall-time medians.

| Metric | Control median (range) | Coverage median (range) |
|---|---:|---:|
| Wall time, seconds | 667.024 (650.646–669.214) | 794.835 (785.570–828.000) |
| Waited-for child CPU time, seconds | 309.706 (309.267–316.199) | 427.519 (426.460–437.332) |
| Largest-child RSS, GiB | 11.305 (11.304–11.307) | 11.324 (11.301–11.329) |
| Uncompressed run evidence, MiB | 0.854 (0.854–0.854) | 18.780 (18.776–18.780) |

RSS is Linux's largest child-process high-water mark, **not aggregate simultaneous
host memory or GPU memory**. Raw evidence includes logs, metadata and JUnit;
instrumented evidence additionally includes the process coverage databases.
The local PNG/SVG `cpu-collection-overhead` figure plots all three pairs.

The two skips preserve existing behavior: Windows-only DLL search handling,
and third-party source verification requiring the existing
`OMNIDREAMS_SINGLEVIEW_RUN_THIRDPARTY_VERIFY` opt-in. The manual result is
`test_omnidreams_streaming_inference`. No extra model opt-ins were enabled.

## Coverage distribution and report cost

Each native report contains 642 Python files. Pair 1 records **45,222/78,463
lines (57.63%)** and **8,527/21,652 branches (39.38%)**. Coverage.py's combined
line/branch score is 53.69%; that score is not line coverage. Missing lines and
branches are listed in the native JSON, XML and HTML exports.

| Scope (pair 1 CPU evidence) | Lines executed/total | Branches executed/total |
|---|---:|---:|
| Core library | 18,079/24,484 | 3,730/6,868 |
| Production tools | 2,470/3,288 | 742/1,274 |
| Applications | 14,237/23,332 | 2,808/6,478 |
| Maintained integrations | 10,436/27,359 | 1,247/7,032 |

The detailed Markdown export further separates core subsystems, individual
applications and integrations. Low CPU execution in GPU-oriented integrations
is not evidence that those paths lack GPU tests.

| Pair | Executed lines / 78,463 | Executed branches / 21,652 | Export seconds | Report MiB | Report ZIP MiB | Raw ZIP MiB |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 45,222 | 8,527 | 372.298 | 322.013 | 27.081 | 3.391 |
| 2 | 45,222 | 8,527 | 242.153 | 322.013 | 27.082 | 3.390 |
| 3 | 45,220 | 8,526 | 291.151 | 321.999 | 27.082 | 3.390 |

Export time has a **291.151-second median** (range **242.153–372.298**), separate
from test time. Reports include total/CPU/GPU HTML, JSON and XML, the merged
database, overlap data, summary and provenance. Local ZIP archives use deflate
level 6; report compression took 7.24–7.45 seconds and raw-data compression
0.39–0.49 seconds. These are local sizes and times, not GitHub transfer
measurements. Both archives and original files are retained.

Pair 3 omits lines 612 and 615 and branch 611→612 in
`flashdreams/flashdreams/serving/webrtc/services.py` relative to pairs 1 and 2. Source
inspection identifies the catch-up skipped-input/window metadata branch.
Identical test outcomes therefore do not imply identical executed paths.
Completeness is recorded in each report's `status.json` and `summary.md`;
all three contain only the missing-GPU diagnostic, with no CPU merge errors.
GPU-only/shared execution cannot be established without GPU evidence. A zero
GPU count in a partial export means unavailable evidence, not an unexecuted GPU
suite. Observed CPU lines are not proven CPU-exclusive lines.

## Environment and interpretation limits

CPU: Python **3.12.13**, pytest **9.1.1**, locked CI dependency profile, native
Ubuntu **24.04.3**, Intel Xeon Platinum 8570 host, eight pinned logical CPUs,
and hidden CUDA devices. This is a shared native host, not the upstream CPU
container. Each run has fresh compiler, Triton, temporary and Python bytecode
cache directories; operating-system filesystem caches remain shared. Existing
test fixtures and seed behavior are unchanged. No global RNG override was
introduced, and bit-identical internal random samples were not recorded.

The timer includes child interpreter startup, coverage/pytest startup,
collection and test execution. Dependency installation, provenance preparation,
asset preparation and report export are outside that timer. These three pairs
are descriptive local measurements, not confidence bounds or a causal estimate
for every CI host. No arbitrary speed or coverage threshold is applied.

Python coverage measures interpreted line/branch execution. It does not measure
CUDA/Triton instructions, compiled-kernel execution or CUDA graph replay.
Import-time execution is not proof of behavioral testing. Forced termination
can lose a worker's in-flight data; no additional signal or `os._exit` patch is
installed. Line overlap cannot establish test redundancy.

GPU setup used the existing CI container
`nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04`, Python **3.11.15**, PyTorch **2.12.1**,
Transformer Engine **2.17.0** and coverage.py **7.16.2**. Dependencies installed
successfully using two Slurm-allocated B200 GPUs, eight CPUs, 128 GiB memory
and a four-hour job limit. The required quality reference in
`nvidia/omni-dreams-samples` then failed with **HTTP 403 / GatedRepoError**:
the current credential lacks access. Dependent GPU pair jobs were cancelled.
There are no GPU outcomes, hardware-specific test skips or performance numbers
to report. Upstream CI uses two RTX PRO 6000 GPUs; B200 setup is not an upstream
hardware validation. Full GPU/CPU merging across Python versions remains
unverified on real GPU-suite data.

## Diagnosed failures and completed checks

An initial full compatibility pair found one instrumentation-induced failure:
site initialization consumed a 300 ms timeout fixture's startup budget before
its child readiness marker was written. The timeout itself worked. Its two
stdlib-only dummy processes now use `python -S`, retaining the original timeout
and descendant-cleanup assertions. They execute no project code; normal project
subprocesses remain instrumented. The corrected case passed in all six formal
runs. The original failure evidence is retained and excluded from estimates.

Real child data also exposed nested absolute-path suffixes, such as
`flashdreams/tools/__init__.py` and `tools/__init__.py`. The exporter now chooses
the longest tracked suffix for absolute names, and requires unique shortened
relative names. Pair 1's 72 raw shards passed format/context/path validation.
The final scope also includes top-level production tools; an earlier successful
control before that scope correction is retained separately.

Preliminary environment failures are not labelled upstream test regressions:
a missing `python3` on PATH, container import temporary-directory loss, and an
inaccessible cache symlink inside the GPU container. Environment activation and
job-local container/cache directories resolved them. The later dataset 403 is
an unresolved access blocker, not a coverage failure.

- All 14 focused diagnostic checks passed again on the final test code (61.29
  seconds), including the stronger thread assertion:
  namespace denominator, exclusions, threads, subprocess/spawn, relocated
  checkouts, contexts, source compatibility, failure retention and control runs.
- Six existing dependency-policy and Agent Skills checks passed.
- Modified Python files passed Ruff formatting/lint and `ty`; lock validation
  passed without unrelated package upgrades. The wheel built successfully.
- Workflow YAML and artifact/release dependency invariants passed local checks.
  GitHub Actions itself has not been run remotely in this phase.
- Sphinx HTML built with warnings treated as errors. New sources have SPDX
  headers; implementation commits carry DCO sign-offs and no coauthor trailers.

## Reproduction and evidence

See [CI coverage diagnostics](../docs/source/developer_guides/ci_coverage.md)
for collection, paired runs and report interpretation. Re-export archived data
against its recorded commit, not a later documentation or test-only commit:

```bash
git worktree add --detach /tmp/flashdreams-coverage-source 791e19b
cd /tmp/flashdreams-coverage-source
uv sync --locked --only-group coverage
uv run --no-sync python tests/coverage_ci.py report \
  --inputs /path/to/pair-1/coverage \
  --output /path/to/new-report
```

The inputs directory contains the original `cpu/` evidence and, when available,
matching `gpu/` evidence. A missing GPU suite intentionally produces a partial
report and diagnostic exit code 1. Output directories must be new.

Local evidence is retained in the ignored `artifacts/coverage-673/` directory:

- `pair-{1,2,3}/{control,coverage}/cpu/`: all six raw sessions and provenance;
- `cpu-reports/pair-{1,2,3}/`: HTML, JSON, XML, grouped summary and status;
- `experiment-summary.json`, `EXPERIMENT_REPORT.md`, `coverage-distribution.json`,
  `report-costs.json` and
  `cpu-collection-overhead.{png,svg}`: measured summaries and visualization;
- `packages/`: local ZIP archives; `run_pairs.py`, `summarize.py`,
  `package_reports.py`, `summarize_coverage.py`, `plot_timing.py`: reproducible local experiment drivers;
- package/license inventories, static/docs logs, GPU setup and scheduler logs;
- `compatibility-probe/`, `scope-preflight/` and preliminary attempts: excluded
  probes and their original failure evidence.

## Conclusion and follow-ups

CPU instrumentation is compatible with the tested full suite after the timeout
fixture correction, with a measured median paired wall-time cost of 22.16%.
Full CPU/GPU acceptance is **not complete**. Restore access to the exact CI
quality reference, then run the three full GPU pairs with the existing opt-ins
and merge compatible evidence. Do not substitute a selected GPU subset or infer
GPU overlap from this CPU-only report. The following follow-ups are supported by the CPU evidence and source review:

1. Add focused, credential-free CPU tests for `core/io/s3_filesystem.py` using a
   mocked S3 client. Pair 1 executes only 35/132 lines and 1/32 branches; the
   existing direct test checks credential-file permissions. Stream closure,
   404 versus other `ClientError` responses, empty paginator pages and directory
   marker handling are concrete candidates. No real cloud account is needed.
2. After the GPU merge, review the remaining checkpoint-loader cache/preflight
   and error paths. Pair 1 executes 217/482 lines and 63/190 branches in
   `core/checkpoint/load.py`; GPU checkpoint loads may account for some missing
   paths, so this is not yet a repository-wide gap estimate.
3. Add a controlled-clock service-level assertion for the WebRTC catch-up
   metadata branch identified above. The existing manager test supplies this
   metadata directly; it does not establish deterministic execution of the
   service's emitting branch. This is an evidence-based test-strengthening
   candidate, not a claim of a functional regression.
4. Measure export/storage costs on the actual CI runner before deciding whether
   report detail or collection frequency should change. Local report cost is
   substantial and storage-dependent; this phase does not change the requested
   CI collection frequency.

No tests are removed based on overlap, and no issue claim, remote push or PR
has been made for this work.

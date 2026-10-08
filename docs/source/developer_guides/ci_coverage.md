# CI coverage diagnostics

CPU and GPU test jobs collect Python line and branch coverage separately. The
`coverage-report` job merges compatible process data and publishes HTML, JSON,
XML and a subsystem summary. Coverage has no percentage threshold and does not
add a release dependency. Ordinary `pytest` runs remain uninstrumented.

## Collection and reports

Install the locked `coverage` development group alongside the existing CI
profile. This adds coverage.py, an Apache-2.0 development tool; it is not a
FlashDreams runtime dependency. CPU CI uses Python 3.12 and GPU CI uses Python
3.11. Keep their existing dependency profiles, test selectors and model-quality
environment variables from `.github/workflows/ci.yml`.

From the repository root, activate the corresponding installed environment
first, or invoke each command through `uv run --no-sync`. This also ensures
subprocesses that invoke `python3` resolve to the same environment:

```bash
python tests/coverage_ci.py run --suite cpu --output artifacts/coverage-673/raw/cpu
# On allocated GPUs, with the existing GPU CI environment:
python tests/coverage_ci.py run --suite gpu --output artifacts/coverage-673/raw/gpu
python tests/coverage_ci.py report --inputs artifacts/coverage-673/raw --output artifacts/coverage-673/report
```

Each run requires a new output directory and preserves pytest's exit status.
It records raw `.coverage.*` process files, JUnit, a test log, per-case JUnit outcomes, pytest summary counts and manual-test nodes,
and provenance: commit, Python source/configuration/lock digests, coverage and
Python versions, selector, timing and exit status. A run interrupted before
completion leaves its initial metadata with `status: running` and all evidence
written so far. Logs are saved directly, without buffering the entire test run
in the launcher. Test failures do not prevent CI artifact upload.

The report requires matching commit, source/configuration/lock digests and
coverage versions. Different Python versions are allowed and recorded.
Missing suites, failed/incomplete tests, collection errors, restricted test
arguments, corrupt shards and incompatible metadata produce **partial** reports
and a nonzero diagnostic exit status. This does not replace the test job's
status. Raw input files are never deleted. Export failures leave a diagnostic
status and the original shards for investigation.

Open `total/html/index.html`, `cpu/html/index.html` or `gpu/html/index.html` after
extracting the report artifact. Each directory also contains `coverage.json`
and `coverage.xml` with missing lines and branches. `overlap.json` lists CPU-only,
GPU-only and shared executed lines per file. The summary groups results by core
subsystem, application and integration. A missing suite's zero counts mean
**unavailable evidence**, not that the suite executes no code. Artifacts are
retained for 30 days; each job summary links to its artifact.

## Measurement scope and limitations

The explicit scope and exclusions are in `[tool.coverage]` in `pyproject.toml`:
core library, production tools (including the reusable benchmark harness), applications and maintained
integrations. Namespace scanning includes unexecuted Python modules in the
denominator. Tests, fixtures, benchmarks, generated protobuf stubs, vendored
native sources, build outputs and local environments are excluded. Maintained
integration `impl` code is included; it is not all treated as third-party code.

Static `cpu` / `gpu` contexts are inherited by threads, spawned subprocesses and
multiprocessing workers. There are no dynamic per-test contexts. Native
subprocess support is enabled only during coverage runs and is intended for
isolated CI environments. Child processes must exit cleanly to save data;
SIGKILL, crashes and forcibly terminated workers can lose in-flight coverage.
No additional signal or `os._exit` patches are installed. The benchmark harness's
stdlib-only timeout fixture uses `python -S` so site initialization cannot consume
its short startup budget. Those dummy scripts execute no project code; normal
project subprocesses remain instrumented.

Coverage's `relative_files` setting makes checkouts portable. Children that
start in another working directory may emit shorter or absolute paths. The
report maps only unique tracked source suffixes through the public
`CoverageData.update` API; unknown or ambiguous paths make the report partial.
No coverage calculations or third-party internals are reimplemented.

Execution is not an assertion of correctness. Import-time execution can produce
substantial overlap. Line overlap alone cannot establish redundant tests, and
this implementation does not identify individual redundant test cases. Python
coverage does not measure CUDA/Triton instructions, CUDA graph replays, or the
internal execution of compiled kernels. Intentional hardware, checkpoint and
service skips remain visible in JUnit; even a complete collection is not proof
that every supported model or backend was tested.

## Measuring collection overhead

Use the same isolated environment for control and instrumented runs. Add
`--no-coverage` to the run command for the control. It still saves JUnit, logs,
outcomes and timing, allowing an equal comparison of launcher overhead.
Run three full pairs per suite, alternating control/instrumented order:

```bash
for pair in 1 2 3; do
  modes='control coverage'
  if [ "$pair" = 2 ]; then modes='coverage control'; fi
  for mode in $modes; do
    flags=()
    if [ "$mode" = control ]; then flags=(--no-coverage); fi
    python tests/coverage_ci.py run --suite cpu \
      --output "artifacts/coverage-673/pair-$pair/$mode/cpu" "${flags[@]}"
  done
done
```

Use `gpu` in an allocated GPU job for the GPU pairs. Provision dependencies and
prefetch required checkpoints and the quality reference before timing. Keep
inputs and opt-in flags identical, and give each run fresh compiler/Triton/tmp
cache directories. Do not enable real-model opt-in tests absent from normal CI.
Do not silently replace full suites with selected tests after a failure.

Compare collected counts and per-case outcomes as well as wall and CPU time.
The test timer includes child interpreter startup, coverage/pytest startup,
collection and execution; provenance preparation and report export are outside
that interval. Both control and coverage sessions use the same recorder.
`pytest_summary` and `manual_nodes` preserve the manual-marker plugin's custom
results, which are not standard JUnit skips and otherwise resemble passed cases.
`peak_child_rss_kib` is Linux's largest child-process resident-set high-water
mark, not the sum of simultaneously resident workers. Report this limitation;
it is neither GPU memory nor allocation-wide host memory. Report export time
separately from test time, alongside artifact size, medians, ranges and paired
relative overhead. Failed collection timings are **not** suite runtime overhead.

The local validation for issue #673 uses two allocated B200s; upstream GPU CI
uses two RTX PRO 6000 GPUs. Local results must not be labelled an upstream
hardware baseline. See the repository's `tests/COVERAGE_REPORT.md` for the
validation state and evidence limitations.

Coverage.py documents [subprocess collection](https://coverage.readthedocs.io/en/7.16.2/subprocess.html)
and the [public data API](https://coverage.readthedocs.io/en/7.16.2/api_coveragedata.html).

# Re-verification of NVIDIA/flashdreams#703 at head `5d8163b`

This re-verifies PR head `5d8163b9324927a8f69e5b46c385bcb8367f5013` against its current parent, `67f91af7e7744d0b208574581e0648a4c1a0d3dd`. All checks passed.

| Check | Result |
| --- | --- |
| First-party long-form Apache headers, first 20 lines | 475 at base → 0 at head ([list](data/long-form-headers-at-base-67f91af.txt)) |
| Files changed by the PR | 475 header-only source edits; workflow, CONTRIBUTING.md and two skills; 1 new test |
| Pre-existing Python files changed | 421, all with identical ASTs ([source-invariance.json](data/source-invariance.json)) |
| Generated and vendored files | 65, all byte-identical (same file) |
| LICENSE, LICENSES/ (3), NOTICE, THIRD-PARTY-NOTICES, REUSE.toml | Unchanged |
| REUSE 6.2.0 lint | 1407/1407 files compliant with REUSE 3.3 ([log](data/reuse-lint-with-encoding-support.log)) |
| The five shell steps of the actual workflow | All pass ([workflow-1…5](data)) |
| SPDX regression test | Passes ([log](data/spdx-regression.log)) |

The one remaining long-form header is in vendored `ludus_renderer/_cpp/cudaraster/`. The workflow excludes that path, and so does the PR.

## Why these counts differ from the original PR description

The original description reported 476 edits, 422 Python files, 1,030 source files and 1,383 REUSE files. Those were measured when the commit sat on an earlier base. After rebasing onto `67f91af`, the counts are:
- 475 header edits; one file with a long-form header no longer exists on main.
- 421 pre-existing Python files.
- 1,034 applicable source files, including the new test.
- 1,407 REUSE files, because main gained files.

The 72 protected files are the 65 generated/vendored files plus LICENSE, three LICENSES/ texts, NOTICE, THIRD-PARTY-NOTICES and REUSE.toml.

## Process notes

- The first REUSE run failed only because the temporary tool environment lacked an encoding-detection dependency ([reuse-lint.log](data/reuse-lint.log)); the rerun with that dependency passed.
- The upstream CI snapshot at that time showed only Greptile, because GitHub-hosted workflows were awaiting maintainer approval.

Local paths are redacted as `<REPO>`, `<HOME>`, `<TMP>` and `<CHECKOUT>`.

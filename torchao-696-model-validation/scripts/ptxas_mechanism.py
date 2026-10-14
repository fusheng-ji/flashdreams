# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU check: deleting Triton's ptxas temp files mid-compile yields the round-1 traceback.

Calls the installed ``CUDABackend.make_cubin`` with a stub ``subprocess.run``
that unlinks the ``.ptx`` and ``.log`` temp files and then runs the real
bundled ptxas. Compares the exception chain with round-1 row 76.
"""

import json
import subprocess
import sys
import traceback
import types
from pathlib import Path

import triton.backends.nvidia.compiler as compiler

ROOT = Path(__file__).resolve().parent
ROUND1 = Path("<REPO>/artifacts/experiment-supplement-20261013/cluster-run/696/components/1168x640-torchao-compile-r1/result.json")
real_run = subprocess.run


def deleting_run(cmd, **kwargs):
    """Simulate an external /tmp cleaner racing the ptxas subprocess."""
    ptx = Path(cmd[cmd.index("-o") - 1])
    log = Path(kwargs["stderr"].name)
    ptx.unlink()
    log.unlink()
    return real_run(cmd, **kwargs)


def main():
    compiler.subprocess = types.SimpleNamespace(run=deleting_run, CalledProcessError=subprocess.CalledProcessError)
    backend = types.SimpleNamespace(target=types.SimpleNamespace(arch=100))
    options = types.SimpleNamespace(enable_fp_fusion=False, ptx_options=None)
    src = ".version 8.7\n.target sm_100a\n.address_size 64\n.visible .entry k() { ret; }\n"
    try:
        compiler.CUDABackend.make_cubin(backend, src, {}, options, 100)
    except FileNotFoundError as error:
        chain = traceback.format_exception(error)
    else:
        raise AssertionError("expected FileNotFoundError")
    text = "".join(chain)
    original = next(row["failure"] for row in json.loads(ROUND1.read_text())["rows"] if row["status"] == "failed")
    signature = ("returned non-zero exit status 255", "During handling of the above exception",
                 "with open(flog.name) as log_file", "FileNotFoundError: [Errno 2] No such file or directory", ".log'")
    result = dict(reproduced={item: item in text for item in signature}, original={item: item in original for item in signature}, traceback=text)
    (ROOT / "ptxas-mechanism.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "traceback"}, indent=2))
    assert all(result["reproduced"].values()) and all(result["original"].values())
    print("PASS: injected deletion reproduces the round-1 failure signature")


if __name__ == "__main__":
    sys.exit(main())

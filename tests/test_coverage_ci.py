# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise coverage diagnostics using isolated, dependency-free toy suites."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

pytest.importorskip("coverage", minversion="7.10")

pytestmark = pytest.mark.ci_cpu
ROOT = Path(__file__).resolve().parents[1]


def command(
    root: Path, *args: str, expected: int = 0
) -> subprocess.CompletedProcess[str]:
    # Do not let the outer coverage process instrument the self-check's children.
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("COVERAGE", "FLASHDREAMS_COVERAGE"))
    }
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    result = subprocess.run([*args], cwd=root, env=env, text=True, capture_output=True)
    assert result.returncode == expected, result.stdout + result.stderr
    return result


def prepare(root: Path) -> None:
    for directory in (
        "tests",
        "apps/demo",
        "flashdreams/flashdreams",
        "flashdreams/tools",
        "tools",
        "integrations_v2",
    ):
        (root / directory).mkdir(parents=True)
    config = (
        ROOT.joinpath("pyproject.toml").read_text().split("[tool.coverage.run]", 1)[1]
    )
    (root / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\nmarkers = ["ci_cpu", "ci_gpu"]\n[tool.coverage.run]'
        + config
    )
    (root / "uv.lock").write_text("test lock")
    shutil.copy(ROOT / "tests/coverage_ci.py", root / "tests/coverage_ci.py")
    (root / "apps/demo/unexecuted.py").write_text("value = 99\n")
    (root / "tools/unexecuted.py").write_text("production_tool = 99\n")
    for name in (
        "tests/test_sentinel.py",
        "benchmarks/sentinel.py",
        "fixtures/sentinel.py",
        "protos/sentinel_pb2.py",
        "3rdparty/sentinel.py",
        "build/sentinel.py",
    ):
        path = root / "apps/demo" / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("excluded_sentinel = 99\n")
    tool = root / "flashdreams/tools/benchmarks/runner.py"
    tool.parent.mkdir()
    tool.write_text("production_tool = 99\n")
    (root / "apps/demo/worker.py").write_text("""import sys

def thread_work():
    return 11

def process_work():
    return 22

def choose(cpu):
    if cpu:
        return 33
    return 44

if __name__ == "__main__":
    subprocess_value = 55
""")
    (root / "tests/test_sample.py").write_text("""import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import threading
import pytest
from apps.demo import worker
pytest.importorskip("coverage", minversion="7.10")

pytestmark = pytest.mark.ci_cpu if os.environ.get("FLASHDREAMS_COVERAGE_CONTEXT") == "cpu" else pytest.mark.ci_gpu

def test_work():
    thread = threading.Thread(target=worker.thread_work)
    thread.start()
    thread.join()
    process = multiprocessing.get_context("spawn").Process(target=worker.process_work)
    process.start()
    process.join()
    assert process.exitcode == 0
    subprocess.run([sys.executable, str(Path(worker.__file__).resolve())], cwd="apps", check=True)
    assert worker.choose(os.environ["FLASHDREAMS_COVERAGE_CONTEXT"] == "cpu") in (33, 44)
""")
    command(root, "git", "init", "-q")
    command(root, "git", "add", ".")
    command(
        root,
        "git",
        "-c",
        "user.name=Coverage test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "Test fixture",
    )


def collect(root: Path, suite: str, expected: int = 0) -> None:
    command(
        root,
        sys.executable,
        "tests/coverage_ci.py",
        "run",
        "--suite",
        suite,
        "--output",
        f"raw/{suite}",
        expected=expected,
    )


def export(root: Path, expected: int = 0) -> dict:
    command(
        root,
        sys.executable,
        "tests/coverage_ci.py",
        "report",
        "--inputs",
        "raw",
        "--output",
        "report",
        expected=expected,
    )
    return json.loads((root / "report/status.json").read_text())


def test_processes_contexts_denominator_and_portable_merge() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory) / "original"
        prepare(root)
        # A shorter tracked suffix must not make the full absolute path ambiguous.
        (root / "worker.py").write_text("not_the_measured_file = True\n")
        command(root, "git", "add", "worker.py")
        collect(root, "cpu")
        collect(root, "gpu")
        # Namespace file discovery in child processes can save absolute paths.
        import coverage

        for suite in ("cpu", "gpu"):
            data = coverage.CoverageData(
                basename=str(root / f"raw/{suite}/.coverage.absolute")
            )
            data.set_context(suite)
            data.add_arcs({str(root / "apps/demo/worker.py"): [(-1, 1)]})
            data.write()
        moved = Path(directory) / "moved"
        shutil.copytree(root, moved)
        shutil.rmtree(root)
        status = export(moved)
        assert status["status"] == "complete"
        files = json.loads((moved / "report/total/coverage.json").read_text())["files"]
        assert files["apps/demo/unexecuted.py"]["missing_lines"] == [1]
        assert files["tools/unexecuted.py"]["missing_lines"] == [1]
        assert not [
            file for file in files if "test_sample" in file or "sentinel" in file
        ]
        assert files["flashdreams/tools/benchmarks/runner.py"]["missing_lines"] == [1]
        worker = files["apps/demo/worker.py"]
        assert {4, 7, 11, 12, 15} <= set(worker["executed_lines"])
        assert worker["summary"]["missing_branches"] == 0
        assert {"cpu", "gpu"} == set(worker["contexts"]["4"])
        overlap = json.loads((moved / "report/overlap.json").read_text())[
            "apps/demo/worker.py"
        ]
        assert 11 in overlap["cpu_only"] and 12 in overlap["gpu_only"]
        assert 4 in overlap["shared"]
        assert list((moved / "raw/cpu").glob(".coverage.*"))
        for suite in ("total", "cpu", "gpu"):
            assert (moved / "report" / suite / "html/index.html").is_file()
            assert (moved / "report" / suite / "coverage.xml").is_file()


@pytest.mark.parametrize(
    "fault",
    [
        "missing_suite",
        "missing_data",
        "corrupt",
        "commit",
        "config_sha256",
        "source_sha256",
        "coverage_version",
        "context",
        "failed",
        "junit",
    ],
)
def test_partial_reports_diagnose_bad_inputs(fault: str) -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        prepare(root)
        collect(root, "cpu")
        collect(root, "gpu")
        target = root / "raw/gpu"
        if fault == "missing_suite":
            shutil.rmtree(target)
        elif fault == "missing_data":
            for shard in target.glob(".coverage.*"):
                shard.unlink()
        elif fault == "corrupt":
            (target / ".coverage.corrupt").write_bytes(b"invalid sqlite")
        elif fault == "context":
            for shard in (root / "raw/cpu").glob(".coverage.*"):
                shutil.copy(shard, target / (shard.name + "wrong"))
        elif fault == "junit":
            (target / "junit.xml").unlink()
        else:
            path = target / "metadata.json"
            metadata = json.loads(path.read_text())
            metadata["exit_code" if fault == "failed" else fault] = (
                1 if fault == "failed" else "wrong"
            )
            path.write_text(json.dumps(metadata))
        status = export(root, expected=1)
        assert status["status"] == "partial" and status["errors"]
        assert (root / "report/total/coverage.json").is_file()


def test_failed_tests_retain_data_and_existing_output_is_not_overwritten() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        prepare(root)
        path = root / "tests/test_sample.py"
        path.write_text(path.read_text() + '\n    assert False, "deliberate failure"\n')
        collect(root, "cpu", expected=1)
        metadata = json.loads((root / "raw/cpu/metadata.json").read_text())
        assert metadata["exit_code"] == 1
        assert list((root / "raw/cpu").glob(".coverage.*"))
        assert "deliberate failure" in (root / "raw/cpu/pytest.log").read_text()
        collect(root, "cpu", expected=1)
        assert metadata == json.loads((root / "raw/cpu/metadata.json").read_text())


def test_uninstrumented_control_has_same_outcomes_without_process_data() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        prepare(root)
        collect(root, "cpu")
        command(
            root,
            sys.executable,
            "tests/coverage_ci.py",
            "run",
            "--suite",
            "cpu",
            "--output",
            "control",
            "--no-coverage",
        )
        assert not list((root / "control").glob(".coverage*"))
        assert json.loads((root / "control/outcomes.json").read_text()) == json.loads(
            (root / "raw/cpu/outcomes.json").read_text()
        )


def test_manual_marker_is_not_counted_as_a_pass_in_pytest_summary() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        prepare(root)
        config = root / "pyproject.toml"
        config.write_text(
            config.read_text().replace(
                "[tool.pytest.ini_options]",
                '[tool.pytest.ini_options]\naddopts = "-p pytest_manual_marker"',
            )
        )
        test = root / "tests/test_sample.py"
        test.write_text(
            test.read_text()
            + "\n@pytest.mark.manual\ndef test_manual():\n    assert False\n"
        )
        collect(root, "cpu")
        metadata = json.loads((root / "raw/cpu/metadata.json").read_text())
        assert metadata["pytest_summary"] == {"passed": 1, "manual": 1}
        assert metadata["manual_nodes"] == ["tests/test_sample.py::test_manual"]

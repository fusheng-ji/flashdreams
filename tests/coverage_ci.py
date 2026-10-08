# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Record CI test runs and export compatible coverage diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import resource
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def provenance() -> dict[str, str]:
    files = subprocess.check_output(
        [
            "git",
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            "*.py",
        ],
        text=True,
    ).split("\0")
    source_hash = hashlib.sha256()
    for filename in sorted(set(files) - {""}):
        source_hash.update(filename.encode() + b"\0")
        source_hash.update(hashlib.sha256(Path(filename).read_bytes()).digest())
    return {
        "source_sha256": source_hash.hexdigest(),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "config_sha256": hashlib.sha256(
            Path("pyproject.toml").read_bytes()
        ).hexdigest(),
        "lock_sha256": hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest(),
        "coverage_version": importlib.metadata.version("coverage"),
    }


def run_suite(
    suite: str, output: Path, instrument: bool, pytest_args: list[str]
) -> int:
    """Save test evidence and return pytest's exit status without a report gate."""
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(
        FLASHDREAMS_COVERAGE_CONTEXT=suite,
        COVERAGE_FILE=str(output / ".coverage"),
        COVERAGE_RCFILE=str(ROOT / "pyproject.toml"),
    )
    command = [sys.executable, "-m"]
    if instrument:
        command += ["coverage", "run", "-m"]
    command += [
        "pytest",
        "-m",
        f"ci_{suite}",
        "-v",
        f"--junitxml={output / 'junit.xml'}",
        *pytest_args,
    ]
    metadata: dict[str, Any] = {
        **provenance(),
        "suite": suite,
        "python_version": platform.python_version(),
        "pytest_version": importlib.metadata.version("pytest"),
        "selector": f"ci_{suite}",
        "instrumented": instrument,
        "pytest_args": pytest_args,
        "exit_code": None,
        "status": "running",
    }
    write_json(output / "metadata.json", metadata)
    started = time.perf_counter()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    with (output / "pytest.log").open("w") as log:
        result = subprocess.run(
            command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    metadata.update(
        exit_code=result.returncode,
        status="finished",
        wall_seconds=time.perf_counter() - started,
        cpu_seconds=after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
        # Linux ru_maxrss is KiB; this is the largest child, not simultaneous RSS.
        peak_child_rss_kib=after.ru_maxrss,
    )
    log_text = (output / "pytest.log").read_text(errors="replace")
    collected = re.findall(r"collected (\d+) items?", log_text)
    metadata["collected"] = int(collected[-1]) if collected else None
    try:
        cases = ElementTree.parse(output / "junit.xml").getroot().iter("testcase")
        outcomes = [
            {
                "name": case.get("classname", "") + "::" + case.get("name", ""),
                "outcome": next(
                    (
                        tag
                        for tag in ("failure", "error", "skipped")
                        if case.find(tag) is not None
                    ),
                    "passed",
                ),
            }
            for case in cases
        ]
        write_json(output / "outcomes.json", outcomes)
    except (OSError, ElementTree.ParseError):
        metadata["junit_error"] = "missing or invalid JUnit results"
    write_json(output / "metadata.json", metadata)
    print(f"{suite}: pytest exited {result.returncode}; evidence: {output}", flush=True)
    return result.returncode if result.returncode >= 0 else 128 - result.returncode


def group_name(filename: str) -> str:
    parts = Path(filename).parts
    if parts[:2] == ("flashdreams", "flashdreams"):
        return "/".join(parts[:3]) if len(parts) > 3 else "flashdreams/package"
    return "/".join(parts[:2])


def report(inputs: Path, output: Path) -> int:
    """Retain raw shards and export total and static-context reports."""
    import coverage

    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    expected = provenance()
    errors: list[str] = []
    origins: list[dict[str, Any]] = []
    accepted: list[str] = []
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "*.py"], text=True
    ).split("\0")
    aliases: dict[str, str] = {}
    for suite in ("cpu", "gpu"):
        directory = inputs / suite
        try:
            metadata = json.loads((directory / "metadata.json").read_text())
            if any(metadata.get(key) != value for key, value in expected.items()):
                raise ValueError(
                    "incompatible commit, configuration, lock or coverage version"
                )
            if (
                metadata.get("suite") != suite
                or metadata.get("selector") != f"ci_{suite}"
                or metadata.get("instrumented") is not True
            ):
                raise ValueError("unexpected suite, selector or uninstrumented run")
            if metadata.get("pytest_args"):
                raise ValueError(
                    "restricted pytest arguments do not establish a full suite baseline"
                )
            origins.append(metadata)
            if metadata.get("status") != "finished" or metadata.get("exit_code") != 0:
                errors.append(f"{suite}: tests failed or did not finish")
            try:
                junit = ElementTree.parse(directory / "junit.xml").getroot()
                if not list(junit.iter("testcase")) or any(
                    int(node.get("errors", "0")) for node in junit.iter("testsuite")
                ):
                    errors.append(f"{suite}: no test cases or collection errors")
            except (OSError, ValueError, ElementTree.ParseError):
                errors.append(f"{suite}: missing or invalid JUnit results")
            shards = sorted(directory.glob(".coverage.*"))
            if not shards:
                raise ValueError("missing process coverage data")
            for index, shard in enumerate(shards):
                try:
                    data = coverage.CoverageData(basename=str(shard))
                    data.read()
                    if not data.has_arcs() or data.measured_contexts() - {suite}:
                        raise ValueError("missing branch data or unexpected context")
                    # Native subprocess tracing relativizes to each child's cwd.
                    # Map only unique tracked suffixes through CoverageData.update.
                    for name in data.measured_files():
                        if name in tracked:
                            continue
                        candidates = [
                            path
                            for path in tracked
                            if path
                            and (
                                name.endswith("/" + path)
                                if Path(name).is_absolute()
                                else path.endswith("/" + name)
                            )
                        ]
                        if len(candidates) != 1 or ".." in Path(name).parts:
                            raise ValueError("unknown or ambiguous source path")
                        canonical = candidates[0]
                        aliases[name] = canonical
                    accepted.append(str(shard.resolve()))
                except (coverage.CoverageException, ValueError) as exc:
                    errors.append(
                        f"{suite}/process-{index}: {type(exc).__name__}: invalid coverage data"
                    )
        except (
            OSError,
            ValueError,
            TypeError,
            AttributeError,
            ElementTree.ParseError,
        ) as exc:
            errors.append(
                f"{suite}: {type(exc).__name__}: {exc}"
                if isinstance(exc, ValueError)
                else f"{suite}: missing or malformed suite evidence ({type(exc).__name__})"
            )

    status = "partial" if errors else "complete"
    summary = [
        f"# CI coverage diagnostics ({status})",
        "",
        "Python line/branch execution only; not CUDA/Triton instruction coverage.",
        "",
    ]
    reports: dict[str, Any] = {}
    try:
        if accepted:
            cov = coverage.Coverage(
                config_file="pyproject.toml", data_file=str(output / ".coverage")
            )
            for shard in accepted:
                data = coverage.CoverageData(basename=shard)
                data.read()
                cov.get_data().update(
                    data, map_path=lambda name: aliases.get(name, name)
                )
            cov.save()
            for name, contexts in (
                ("total", None),
                ("cpu", ["^cpu$"]),
                ("gpu", ["^gpu$"]),
            ):
                destination = output / name
                destination.mkdir()
                cov.html_report(directory=str(destination / "html"), contexts=contexts)
                cov.xml_report(
                    outfile=str(destination / "coverage.xml"), contexts=contexts
                )
                cov.json_report(
                    outfile=str(destination / "coverage.json"),
                    contexts=contexts,
                    show_contexts=True,
                )
                reports[name] = json.loads((destination / "coverage.json").read_text())
            groups: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
            overlap: dict[str, Any] = {}
            for filename, entry in reports["total"]["files"].items():
                group = groups[group_name(filename)]
                for key in (
                    "covered_lines",
                    "num_statements",
                    "missing_lines",
                    "covered_branches",
                    "num_branches",
                    "missing_branches",
                ):
                    group[key] += entry["summary"][key]
                cpu = set(reports["cpu"]["files"][filename]["executed_lines"])
                gpu = set(reports["gpu"]["files"][filename]["executed_lines"])
                overlap[filename] = {
                    "cpu_only": sorted(cpu - gpu),
                    "gpu_only": sorted(gpu - cpu),
                    "shared": sorted(cpu & gpu),
                }
                for key, value in overlap[filename].items():
                    group[key] += len(value)
            write_json(output / "overlap.json", overlap)
            summary += [
                "| Scope | Lines hit/total | Branches hit/total | CPU only | GPU only | Shared |",
                "|---|---:|---:|---:|---:|---:|",
            ]
            for name, group in sorted(groups.items()):
                summary.append(
                    f"| {name} | {group['covered_lines']}/{group['num_statements']} | {group['covered_branches']}/{group['num_branches']} | {group['cpu_only']} | {group['gpu_only']} | {group['shared']} |"
                )
            summary += [
                "",
                "See total/, cpu/ and gpu/ for HTML, JSON and XML with missing lines and branches; overlap.json lists suite-exclusive and shared lines.",
                "Overlap does not establish test redundancy or justify removing tests.",
            ]
        else:
            errors.append("no compatible coverage data; no coverage baseline exported")
            status = "partial"
    except (coverage.CoverageException, OSError, ValueError) as exc:
        errors.append(
            f"report export failed ({type(exc).__name__}); raw inputs retained"
        )
        status = "partial"
    summary[0] = f"# CI coverage diagnostics ({status})"
    summary += ["", *[f"- Diagnostic: {error}" for error in errors]]
    (output / "summary.md").write_text("\n".join(summary) + "\n")
    write_json(
        output / "status.json",
        {
            "status": status,
            "errors": errors,
            "sources": origins,
            "report_seconds": time.perf_counter() - started,
        },
    )
    return int(bool(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run")
    run.add_argument("--suite", choices=("cpu", "gpu"), required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--no-coverage", action="store_true")
    run.add_argument("pytest_args", nargs=argparse.REMAINDER)
    export = sub.add_parser("report")
    export.add_argument("--inputs", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # All relative source names and report paths are repository-relative.
    os.chdir(ROOT)
    if args.action == "run":
        extra = args.pytest_args
        if extra[:1] == ["--"]:
            extra = extra[1:]
        return run_suite(args.suite, args.output, not args.no_coverage, extra)
    return report(args.inputs, args.output)


if __name__ == "__main__":
    raise SystemExit(main())

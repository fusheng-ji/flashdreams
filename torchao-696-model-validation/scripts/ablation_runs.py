# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Round-2 #696 ablations on the round-1 frozen harness.

Modes (one GPU process each):
  quality    short straight/turn rollouts for one variant, gated against the round-1 BF16 reference
  layers     teacher-forced per-layer FP8 output-projection error on a BF16 eager rollout
  component  round-1 component_bench cell redirected to a private output directory (PTXAS stress)
"""

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ROUND1 = Path("<REPO>/artifacts/experiment-supplement-20261013/cluster-run")
sys.path.insert(0, str(ROUND1))
LABELS = [f"short-{motion}-s{seed}" for motion in ("straight", "turn") for seed in (0, 1, 42)]
VARIANTS = ("bf16_rerun", "bf16_numeric_control", "fd_self_only", "fd_cross_only", "fd_keep_bf16")
KINDS = ("self_attn", "cross_attn")


def bf16_units(variant, keep):
    """Return the (block, kind) output projections that stay BF16 in an FP8 variant."""
    if variant == "fd_self_only":
        return {(block, "cross_attn") for block in range(28)}
    if variant == "fd_cross_only":
        return {(block, "self_attn") for block in range(28)}
    if variant == "fd_keep_bf16":
        units = {(int(block), kind) for block, kind in (item.split(":") for item in keep.split(","))}
        assert units and all(kind in KINDS and 0 <= block < 28 for block, kind in units), keep
        return units
    return set()


def build(variant, execution, keep=""):
    """Build the round-1 pipeline, then route selected FP8 projections back to BF16 before first trace."""
    from pipeline_bench import build_pipeline
    backend = {"bf16_rerun": "bf16", "bf16_numeric_control": "bf16_numeric_control"}.get(variant, "bf16" if variant == "layers" else "flashdreams")
    pipeline = build_pipeline(backend, execution)
    blocks = pipeline.diffusion_model.transformer.network.blocks
    units = sorted(bf16_units(variant, keep))
    for block, kind in units:
        attention = getattr(blocks[block], kind)
        assert attention.quantized_output_projection is not None, (block, kind)
        attention._project_output = attention.output_projection
    return pipeline, units


def load_inputs(width, height):
    import torch
    condition = torch.load(ROUND1 / "696" / f"probe-{width}x{height}" / "conditioning.pt", map_location="cpu", weights_only=True)
    manifest = json.loads((ROUND1 / "696" / f"sequences-{width}x{height}" / "result.json").read_text())
    assert manifest["status"] == "passed"
    return condition, {row["name"]: row["chunks"] for row in manifest["sequences"]}


def hdmap(chunk):
    import torch
    return torch.load(chunk["path"], weights_only=True).cuda().to(torch.bfloat16).div(127.5).sub(1)


def run_quality(args, record, output, save):
    """Roll out LABELS, store RGB/latents, gate against round-1 BF16 without asserting."""
    import numpy as np
    import torch
    from flashdreams.quality.clip_compare import ClipComparisonThresholds, _threshold_failures, compare_video_arrays
    from quality_runs import dataclasses_asdict
    condition, sequences = load_inputs(args.width, args.height)
    pipeline, units = build(args.variant, "compile_graph", args.keep)
    record["bf16_units"] = [f"{block}:{kind}" for block, kind in units]
    reference_root = ROUND1 / "raw-quality" / f"{args.width}x{args.height}-bf16-compile_graph"
    reference_record = json.loads((ROUND1 / "696/quality" / f"{args.width}x{args.height}-bf16-compile_graph" / "result.json").read_text())
    raw = ROOT / "raw-quality" / output.name
    latent = [None]
    noise = []
    diffuse = pipeline.diffusion_model.generate
    initial_noise = pipeline.diffusion_model.transformer.initial_noise
    def generate(*a, **kw):
        value, state = diffuse(*a, **kw)
        latent[0] = value.detach().cpu()
        return value, state
    def hashed_noise(*a, **kw):
        value = initial_noise(*a, **kw)
        noise.append(hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
        return value
    pipeline.diffusion_model.generate = generate
    pipeline.diffusion_model.transformer.initial_noise = hashed_noise
    with torch.inference_mode():
        for label in ([f"short-static-s{seed}" for seed in (0, 1, 42)] if args.static else []) + LABELS:
            motion, seed = label.split("-")[1], int(label.split("-s")[-1])
            row = dict(label=label, status="running", noise_sha256=[])
            record["rollouts"].append(row)
            destination = raw / label
            destination.mkdir(parents=True, exist_ok=True)
            try:
                cache = pipeline.initialize_cache_from_embeddings(**condition["embeddings"], view_names=["camera_front_wide_120fov"])
                pipeline.diffusion_model.rng.manual_seed(seed)
                frames = []
                for index in range(10):
                    noise.clear()
                    generated = pipeline.generate(index, cache, input=hdmap(sequences[motion][index]))
                    pipeline.finalize(index, cache)
                    torch.cuda.synchronize()
                    assert torch.isfinite(generated).all() and torch.isfinite(latent[0]).all()
                    row["noise_sha256"].append(noise.copy())
                    rgb = generated[0, 0].permute(0, 2, 3, 1).float().cpu().add(1).mul(127.5).round().clamp(0, 255).byte().numpy()
                    np.save(destination / f"rgb-{index:03d}.npy", rgb)
                    torch.save(latent[0], destination / f"latent-{index:03d}.pt")
                    frames.append(rgb)
                reference_row = next(item for item in reference_record["rollouts"] if item["label"] == label)
                row["noise_matches_reference"] = row["noise_sha256"] == reference_row["noise_sha256"]
                assert row["noise_matches_reference"], "noise sequence differs from round-1 BF16 reference"
                ref = np.concatenate([np.load(reference_root / label / f"rgb-{index:03d}.npy") for index in range(10)])
                candidate = np.concatenate(frames)
                row["bitwise_equal_reference"] = bool(np.array_equal(ref, candidate))
                result = compare_video_arrays(ref, candidate, frame_indices=range(len(ref)), compute_flip=True)
                row["metrics"] = dataclasses_asdict(result)
                row["quality_failures"] = _threshold_failures(result, ClipComparisonThresholds())
                row["rgb_gate"] = "failed" if row["quality_failures"] else "passed"
                row["status"] = "passed"
            except Exception:
                row.update(status="failed", failure=traceback.format_exc())
            save()
    from torch._dynamo.utils import counters
    record["dynamo_counters"] = {key: dict(value) for key, value in counters.items()}
    assert counters["stats"]["unique_graphs"] > 0 and not counters["graph_break"], record["dynamo_counters"]
    return all(row["status"] == "passed" for row in record["rollouts"])


def run_layers(args, record, output, save):
    """Teacher-forced FP8-vs-BF16 output-projection error for all 56 projections on the turn rollout."""
    import torch
    import torch.nn.functional as F
    from flashdreams.accelerated.quantization.linear import QuantizedNonPersistentLinear, TorchaoNonPersistentLinear, WeightGranularity
    from flashdreams.accelerated.quantization.quantizer import Granularity
    condition, sequences = load_inputs(args.width, args.height)
    pipeline, _ = build("layers", "eager")
    blocks = pipeline.diffusion_model.transformer.network.blocks
    stats = {}
    chunk = [None]
    def error(out, ref):
        delta = (out.float() - ref.float()).flatten()
        return (delta.norm() / ref.float().norm()).item(), F.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0).item()
    for block in range(len(blocks)):
        for kind in KINDS:
            attention = getattr(blocks[block], kind)
            weight, bias = attention.output_projection.weight, attention.output_projection.bias
            fd = QuantizedNonPersistentLinear(weight.detach().contiguous(), None if bias is None else bias.detach(), WeightGranularity.PER_OUT_CHANNEL, torch.float8_e4m3fn)
            ta = TorchaoNonPersistentLinear(weight.detach(), None if bias is None else bias.detach())
            def project(x, attention=attention, fd=fd, ta=ta, key=(block, kind)):
                ref = attention.output_projection(x)
                fd_rel, fd_cos = error(fd(x, Granularity.SLICE, out_dtype=x.dtype), ref)
                ta_rel, ta_cos = error(ta(x), ref)
                rows = x.float().reshape(-1, x.shape[-1])
                row_amax = rows.abs().amax(dim=1)
                scale = (row_amax / 448).clamp_min(torch.finfo(torch.float32).tiny)[:, None]
                quantized = (rows / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()
                nonzero = rows != 0
                entry = stats.setdefault((chunk[0], *key), dict(calls=0, fd_rel_l2=0.0, ta_rel_l2=0.0, fd_cosine=0.0, ta_cosine=0.0, fd_rel_l2_max=0.0, input_amax=0.0, row_amax_ratio=0.0, underflow_fraction=0.0, output_rms=0.0))
                entry["calls"] += 1
                for name, value in (("fd_rel_l2", fd_rel), ("ta_rel_l2", ta_rel), ("fd_cosine", fd_cos), ("ta_cosine", ta_cos),
                                    ("row_amax_ratio", (row_amax.max() / row_amax.median().clamp_min(1e-30)).item()),
                                    ("underflow_fraction", ((quantized == 0) & nonzero).sum().item() / max(1, nonzero.sum().item())),
                                    ("output_rms", ref.float().pow(2).mean().sqrt().item())):
                    entry[name] += value
                entry["fd_rel_l2_max"] = max(entry["fd_rel_l2_max"], fd_rel)
                entry["input_amax"] = max(entry["input_amax"], row_amax.max().item())
                return ref
            attention._project_output = project
    with torch.inference_mode():
        cache = pipeline.initialize_cache_from_embeddings(**condition["embeddings"], view_names=["camera_front_wide_120fov"])
        pipeline.diffusion_model.rng.manual_seed(42)
        for index in range(10):
            chunk[0] = index
            generated = pipeline.generate(index, cache, input=hdmap(sequences["turn"][index]))
            pipeline.finalize(index, cache)
            torch.cuda.synchronize()
            assert torch.isfinite(generated).all()
            save()
    averaged = ("fd_rel_l2", "ta_rel_l2", "fd_cosine", "ta_cosine", "row_amax_ratio", "underflow_fraction", "output_rms")
    record["rows"] = [dict(chunk=c, block=b, kind=k, calls=v["calls"], fd_rel_l2_max=v["fd_rel_l2_max"], input_amax=v["input_amax"],
                           **{name: v[name] / v["calls"] for name in averaged}) for (c, b, k), v in sorted(stats.items())]
    assert len({(row["block"], row["kind"]) for row in record["rows"]}) == 56, "not every output projection was exercised"
    return True


def run_component(args):
    """Run one round-1 component cell with output redirected under ROOT."""
    import component_bench
    target = ROOT / "ptxas-stress" / args.tag
    (target / "696").mkdir(parents=True, exist_ok=False)
    (target / "696" / "probe-1168x640").symlink_to(ROUND1 / "696" / "probe-1168x640")
    component_bench.ROOT = target
    sys.argv = [sys.argv[0], "--width", "1168", "--height", "640", "--backend", "torchao", "--execution", "compile", "--repeat", "0"]
    component_bench.main()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("quality", "layers", "component", "self-check"))
    parser.add_argument("--width", type=int, choices=(1280, 1168))
    parser.add_argument("--height", type=int, choices=(704, 640))
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--keep", default="", help="fd_keep_bf16: comma list of block:kind kept in BF16")
    parser.add_argument("--tag", help="unique output tag (component, repeated quality variants)")
    parser.add_argument("--static", action="store_true", help="quality: also roll out the three static sequences")
    args = parser.parse_args()
    if args.mode == "self-check":
        assert len(LABELS) == 6 and len(bf16_units("fd_self_only", "")) == 28 and len(bf16_units("fd_cross_only", "")) == 28
        assert bf16_units("fd_keep_bf16", "3:self_attn,27:cross_attn") == {(3, "self_attn"), (27, "cross_attn")}
        assert not bf16_units("bf16_rerun", "")
        for path in (ROUND1 / "pipeline_bench.py", ROUND1 / "component_bench.py", ROUND1 / "696/probe-1168x640/projections.pt",
                     ROUND1 / "raw-quality/1168x640-bf16-compile_graph/short-turn-s42/rgb-009.npy", ROUND1 / "696/sequences-1280x704/result.json"):
            assert path.exists(), path
        print("ablation_runs self-check passed")
        return
    if args.mode == "component":
        return run_component(args)
    assert (args.width, args.height) in ((1280, 704), (1168, 640))
    name = f"{args.width}x{args.height}-" + ("layers" if args.mode == "layers" else args.variant + (f"-{args.tag}" if args.tag else ""))
    output = ROOT / args.mode / name
    output.mkdir(parents=True, exist_ok=False)
    record = dict(vars(args), status="running", command=sys.argv, rollouts=[], tmpdir=os.environ.get("TMPDIR"), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    start = time.monotonic()
    def save():
        (output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    save()
    try:
        import frozen_imports
        os.environ["FLASHDREAMS_SYNC_AND_PROFILE"] = "0"
        os.environ["HF_HUB_CACHE"] = str(ROUND1 / "assets/hub")
        frozen_imports.verify()
        import torch
        record.update(torch=torch.__version__, gpu=torch.cuda.get_device_name())
        ok = (run_quality if args.mode == "quality" else run_layers)(args, record, output, save)
        frozen_imports.verify()
        record["status"] = "passed" if ok else "failed"
    except Exception:
        record.update(status="failed", failure=traceback.format_exc())
    record["elapsed_s"] = time.monotonic() - start
    save()
    print(json.dumps(dict(status=record["status"], output=str(output), elapsed_s=record["elapsed_s"])))
    raise SystemExit(0 if record["status"] == "passed" else 1)


if __name__ == "__main__":
    main()

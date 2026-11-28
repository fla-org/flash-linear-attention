# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
# Copyright 2026 Co-developed by KDA (Kernel Design Agents)

"""Compare the official forward matrix using identical CUDA IPC input tensors."""

import argparse
import gc
import hashlib
import importlib.util
import json
import queue
import sys
import traceback
from pathlib import Path

import torch
from torch import multiprocessing

from benchmarks.ops import qla_ptx_timing as cupti_timing
from fla.ops.gated_delta_rule.backends.qla_ptx._runtime import ROOT


def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def errors(actual, expected):
    result = []
    for name, observed, reference in zip(("output", "final_state"), actual, expected):
        assert observed.shape == reference.shape
        assert torch.isfinite(observed).all() and torch.isfinite(reference).all()
        delta = observed.double() - reference.double()
        ratio = (delta.norm() / reference.double().norm().clamp_min(1e-30)).item()
        result.append(
            {
                "tensor": name,
                "relative_l2": ratio,
                "max_absolute": delta.abs().max().item(),
                "passed_fla_0_005": ratio <= 0.005,
                "passed_flashqla_0_02": ratio <= 0.02,
            }
        )
    return result


def strict_pass(checks):
    return all(item["passed_fla_0_005"] for item in checks)


@torch.inference_mode()
def baseline_worker(link, reports, options):
    try:
        sys.path[:0] = [str(options.deps), str(options.baseline)]
        official = module_from_file(
            "official_benchmark",
            options.baseline / "benchmark/bench_gated_delta_rule.py",
        )
        config = json.loads((ROOT / "benchmark_shared.json").read_text())
        assert (
            hashlib.sha256((options.baseline / "benchmark/bench_gated_delta_rule.py").read_bytes()).hexdigest()
            == config["official_script_sha256"]
        )
        torch.manual_seed(config["seed"])
        reports.put({"phase": "baseline_ready", "gpu": torch.cuda.get_device_name()})
        for index, workload in enumerate(config["workloads"]):
            data = official.prepare_tensors(workload["lengths"], workload["Hq"], workload["Hv"], swa_ratio=0.75)
            if data is None:
                raise RuntimeError(f"Official input allocation failed: {workload['name']}")
            data.pop("do")
            data.pop("dht")
            context = official.build_cp_context(
                data["cu_seqlens"],
                enable_intra=True,
                num_v_heads=workload["Hv"],
                chunk_size=official.CHUNK_SIZE,
                is_train=False,
            )

            def forward(data=data, context=context):
                return official.qla_fwd(
                    data["q"],
                    data["k"],
                    data["v"],
                    data["g"],
                    data["beta"],
                    scale=data["scale"],
                    initial_state=data["h0"],
                    output_final_state=True,
                    output_h=False,
                    cu_seqlens=data["cu_seqlens"],
                    cp_context=context,
                )

            computed = forward()
            expected = (computed[2], computed[4])
            del computed
            torch.cuda.synchronize()
            link.send(
                {
                    "index": index,
                    "workload": workload,
                    "inputs": data,
                    "baseline_outputs": expected,
                }
            )
            checks = link.recv()
            baseline_us = None
            if checks["baseline_passed"]:
                baseline_us = cupti_timing.do_bench(forward, warmup=5, rep=3, backend="cupti") * 1000
            link.send(
                {
                    "baseline_us": baseline_us,
                    "baseline_timing": cupti_timing.last_report() if baseline_us is not None else None,
                }
            )
            result = link.recv()
            reports.put(result)
            del forward, data, expected, context, result
            gc.collect()
            torch.cuda.empty_cache()
        link.send(None)
        assert link.recv() == "done"
        reports.put({"phase": "baseline_done"})
    except Exception:
        reports.put(
            {
                "phase": "error",
                "worker": "baseline",
                "traceback": traceback.format_exc(),
            }
        )
        raise
    finally:
        link.close()


@torch.inference_mode()
def candidate_worker(link, reports, options):
    try:
        from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
        from fla.ops.gated_delta_rule.backends.qla_ptx import run

        reports.put({"phase": "candidate_ready", "gpu": torch.cuda.get_device_name()})
        while True:
            packet = link.recv()
            if packet is None:
                link.send("done")
                break
            data = packet["inputs"]
            args = (
                data["q"],
                data["k"],
                data["v"],
                data["g"],
                data["beta"],
                data["scale"],
                data["h0"],
                data["cu_seqlens"],
            )
            row = {"index": packet["index"], "workload": packet["workload"]}
            try:
                observed = run(*args)
                assert observed[0].dtype == torch.bfloat16
                assert observed[1].dtype == torch.float32
                reference = fused_recurrent_gated_delta_rule(
                    q=data["q"].float(),
                    k=data["k"].float(),
                    v=data["v"].float(),
                    g=data["g"],
                    beta=data["beta"],
                    scale=data["scale"],
                    initial_state=data["h0"],
                    output_final_state=True,
                    cu_seqlens=data["cu_seqlens"],
                )
                row["baseline_errors"] = errors(packet["baseline_outputs"], reference)
                row["candidate_errors"] = errors(observed, reference)
                row["candidate_vs_baseline"] = errors(observed, packet["baseline_outputs"])
                row["baseline_passed"] = strict_pass(row["baseline_errors"])
                row["candidate_passed"] = strict_pass(row["candidate_errors"])
                row["status"] = "passed" if row["baseline_passed"] and row["candidate_passed"] else "correctness_failed"
                del observed, reference
            except (AssertionError, RuntimeError, ValueError, TypeError, KeyError):
                row.update(
                    status="candidate_error",
                    baseline_passed=False,
                    candidate_passed=False,
                    error=traceback.format_exc(),
                )
            torch.cuda.synchronize()
            link.send(
                {
                    "baseline_passed": row["baseline_passed"],
                    "candidate_passed": row["candidate_passed"],
                }
            )
            row.update(link.recv())
            if row["status"] == "passed":
                row["candidate_us"] = (
                    cupti_timing.do_bench(
                        lambda args=args: run(*args),
                        warmup=5,
                        rep=3,
                        backend="cupti",
                    )
                    * 1000
                )
                row["candidate_timing"] = cupti_timing.last_report()
                row["speedup"] = row["baseline_us"] / row["candidate_us"]
            torch.cuda.synchronize()
            del args, data, packet
            gc.collect()
            torch.cuda.empty_cache()
            link.send(row)
        reports.put({"phase": "candidate_done"})
    except Exception:
        reports.put(
            {
                "phase": "error",
                "worker": "candidate",
                "traceback": traceback.format_exc(),
            }
        )
        raise
    finally:
        link.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--deps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    options.baseline, options.deps = (p.resolve() for p in (options.baseline, options.deps))
    options.output.parent.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "benchmark_shared.json").read_text())
    context = multiprocessing.get_context("spawn")
    producer_link, consumer_link = context.Pipe()
    reports = context.Queue()
    workers = [
        context.Process(target=baseline_worker, args=(producer_link, reports, options)),
        context.Process(target=candidate_worker, args=(consumer_link, reports, options)),
    ]
    result = {
        "config": config,
        "transport": "CUDA IPC: exact shared input/output storage",
        "candidate_mode": "exported_ptx",
        "rows": [],
        "complete": False,
    }
    output = options.output
    for worker in workers:
        worker.start()
    finished = 0
    try:
        while finished < 2:
            try:
                report = reports.get(timeout=30)
            except queue.Empty:
                if any(worker.exitcode not in (None, 0) for worker in workers):
                    raise RuntimeError("An official comparison worker exited unexpectedly")
                continue
            if report.get("phase") == "error":
                raise RuntimeError(report["traceback"])
            if report.get("phase", "").endswith("_done"):
                finished += 1
            elif "workload" in report:
                result["rows"].append(report)
                result["complete"] = len(result["rows"]) == len(config["workloads"])
                output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(report), flush=True)
        for worker in workers:
            worker.join(timeout=30)
            assert worker.exitcode == 0
        assert result["complete"]
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=10)


if __name__ == "__main__":
    main()

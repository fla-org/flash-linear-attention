#!/usr/bin/env python3
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Reproduce the dynamic-metadata CUDA Graph failure in varlen chunk GDN."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback

import torch
import torch.nn.functional as F

import fla.utils as fla_utils
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.ops.utils import prepare_chunk_indices
from fla.utils import device


T_MAX = 1024
N_MAX = 4
H = 2
D = 32
CHUNK_SIZE = 64
CU_A = (0, 256, 512, 768, 1024)
CU_B = (0, 100, 400, 1024, 1024)


def _inputs(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device).manual_seed(seed)
    dtype = torch.float16
    q = F.normalize(torch.randn(1, T_MAX, H, D, dtype=torch.float32, device=device, generator=generator), dim=-1).to(dtype)
    k = F.normalize(torch.randn(1, T_MAX, H, D, dtype=torch.float32, device=device, generator=generator), dim=-1).to(dtype)
    return {
        "q": q,
        "k": k,
        "v": torch.randn(1, T_MAX, H, D, dtype=dtype, device=device, generator=generator),
        "g": F.logsigmoid(torch.randn(1, T_MAX, H, dtype=torch.float32, device=device, generator=generator)).to(dtype),
        "beta": torch.rand(1, T_MAX, H, dtype=dtype, device=device, generator=generator).sigmoid(),
        "h0": torch.randn(N_MAX, H, D, D, dtype=dtype, device=device, generator=generator),
    }


def _call(inputs: dict[str, torch.Tensor], cu_seqlens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.inference_mode():
        return chunk_gated_delta_rule(
            q=inputs["q"],
            k=inputs["k"],
            v=inputs["v"],
            g=inputs["g"],
            beta=inputs["beta"],
            initial_state=inputs["h0"],
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            chunk_size=CHUNK_SIZE,
        )


def _warm_and_capture(step):
    warm_stream = torch.cuda.Stream()
    warm_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm_stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(warm_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = step()
    torch.cuda.synchronize()
    return graph, outputs


def _parity(got: torch.Tensor, expected: torch.Tensor) -> tuple[bool, float]:
    max_abs = (got.float() - expected.float()).abs().max().item()
    try:
        torch.testing.assert_close(got, expected, rtol=5e-3, atol=5e-3)
        return True, max_abs
    except AssertionError:
        return False, max_abs


def _run_uncached() -> None:
    fla_utils.FLA_DISABLE_TENSOR_CACHE = True
    inputs = _inputs(1)
    cu = torch.tensor(CU_A, dtype=torch.long, device=device)

    try:
        graph, captured = _warm_and_capture(lambda: _call(inputs, cu))
    except BaseException as exc:
        print(f"capture_error={type(exc).__name__}: {exc}")
        traceback.print_exc(file=sys.stdout)
        print(json.dumps({"stage": "uncached", "capture": "failed", "error_type": type(exc).__name__}))
        return

    next_inputs = _inputs(2)
    for name, value in next_inputs.items():
        inputs[name].copy_(value)
    cu.copy_(torch.tensor(CU_B, dtype=torch.long, device=device))
    graph.replay()
    torch.cuda.synchronize()

    eager = _call(next_inputs, torch.tensor(CU_B, dtype=torch.long, device=device))
    o_ok, o_abs = _parity(captured[0], eager[0])
    ht_ok, ht_abs = _parity(captured[1], eager[1])
    print(json.dumps({
        "stage": "uncached",
        "capture": "succeeded",
        "replay_output_matches": o_ok,
        "replay_state_matches": ht_ok,
        "output_max_abs": o_abs,
        "state_max_abs": ht_abs,
    }))


def _run_cached() -> None:
    fla_utils.FLA_DISABLE_TENSOR_CACHE = False
    inputs = _inputs(3)
    cu = torch.tensor(CU_A, dtype=torch.long, device=device)

    try:
        graph, captured = _warm_and_capture(lambda: _call(inputs, cu))
    except BaseException as exc:
        print(f"capture_error={type(exc).__name__}: {exc}")
        traceback.print_exc(file=sys.stdout)
        print(json.dumps({"stage": "cached", "capture": "failed", "error_type": type(exc).__name__}))
        return

    next_inputs = _inputs(4)
    for name, value in next_inputs.items():
        inputs[name].copy_(value)
    cu.copy_(torch.tensor(CU_B, dtype=torch.long, device=device))
    graph.replay()
    torch.cuda.synchronize()

    fla_utils.FLA_DISABLE_TENSOR_CACHE = True
    eager = _call(next_inputs, torch.tensor(CU_B, dtype=torch.long, device=device))
    o_ok, o_abs = _parity(captured[0], eager[0])
    ht_ok, ht_abs = _parity(captured[1], eager[1])
    print(json.dumps({
        "stage": "cached",
        "capture": "succeeded",
        "replay_output_matches": o_ok,
        "replay_state_matches": ht_ok,
        "output_max_abs": o_abs,
        "state_max_abs": ht_abs,
    }))


def _print_metadata() -> None:
    fla_utils.FLA_DISABLE_TENSOR_CACHE = True
    cu_a = torch.tensor(CU_A, dtype=torch.long, device=device)
    cu_b = torch.tensor(CU_B, dtype=torch.long, device=device)
    indices_a = prepare_chunk_indices(cu_a, CHUNK_SIZE)
    indices_b = prepare_chunk_indices(cu_b, CHUNK_SIZE)
    print(f"T_MAX={T_MAX} N_MAX={N_MAX} chunk_size={CHUNK_SIZE}")
    print(f"A cu_seqlens={list(CU_A)} actual_nt={len(indices_a)} shape={tuple(indices_a.shape)}")
    print(f"B cu_seqlens={list(CU_B)} actual_nt={len(indices_b)} shape={tuple(indices_b.shape)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all", "uncached", "cached"), default="all")
    args = parser.parse_args()

    if not torch.cuda.is_available() or device != "cuda":
        raise RuntimeError("This reproduction requires an NVIDIA CUDA device")

    _print_metadata()
    if args.stage != "all":
        {"uncached": _run_uncached, "cached": _run_cached}[args.stage]()
        return 0

    for stage in ("uncached", "cached"):
        completed = subprocess.run([sys.executable, __file__, "--stage", stage], check=False)
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

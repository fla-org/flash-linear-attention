# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import platform
import random
import socket
import statistics
import subprocess

import torch
import triton
from einops import rearrange

from fla.layers.gated_deltanet import GatedDeltaNet
from fla.modules.convolution import causal_conv1d
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.utils import device

logger = logging.getLogger(__name__)

QUANTILES = (
    ("median", 0.5),
    ("p20", 0.2),
    ("p80", 0.8),
)
DTYPE = torch.bfloat16

# Logical order: full layer, then the isolated op, then the conv prologue.
PROVIDERS = [
    "layer_train_fwd",
    "layer_train_fwdbwd",
    "layer_eval_fwd",
    "op_fwd",
    "op_fwdbwd",
    "conv_separate_fwd",
    "conv_fused_fwd",
    "conv_separate_fwdbwd",
    "conv_fused_fwdbwd",
]

# (B, T, hidden_size, H, HV, D, V)
SHAPES = [
    (1, 128, 512, 6, 6, 64, 128),  # Small enough for a quick smoke run.
    (2, 2048, 2048, 6, 6, 256, 512),  # GatedDeltaNetConfig defaults used by the layer-level q/k/v conv work.
    (2, 4096, 2048, 6, 6, 256, 512),  # Longer default-config layer workload used for prologue/output-kernel changes.
    (4, 2048, 2048, 8, 8, 64, 64),  # Dense K=V=64 path targeted by state/output fusion work.
    (2, 2048, 2048, 4, 8, 64, 64),  # Grouped-value variant for dense K=V=64 guarded paths.
    (2, 4096, 2048, 16, 16, 128, 128),  # D128 chunk-output style workload.
]


# --------------------------------------------------------------------------- #
# Section builders
# --------------------------------------------------------------------------- #
def _build_layer(hidden_size, H, HV, D, V):
    return GatedDeltaNet(
        hidden_size=hidden_size,
        expand_v=V / D,
        head_dim=D,
        num_heads=H,
        num_v_heads=HV,
        mode="chunk",
        use_gate=True,
        use_short_conv=True,
        conv_size=4,
    ).to(device=device, dtype=DTYPE)


def _separate_conv(layer, q, k, v):
    q = layer.q_conv1d(q)[0]
    k = layer.k_conv1d(k)[0]
    v = layer.v_conv1d(v)[0]
    return q, k, v


def _pack_qkv_conv_weight(layer):
    """Concatenate the q/k/v depthwise conv weights/bias once.

    In a real fused module this packing is a one-time init cost, so callers that
    time the fused path pre-pack here and keep it out of the measured function.
    """
    weight = torch.cat(
        [
            rearrange(layer.q_conv1d.weight, "d 1 w -> d w"),
            rearrange(layer.k_conv1d.weight, "d 1 w -> d w"),
            rearrange(layer.v_conv1d.weight, "d 1 w -> d w"),
        ],
        dim=0,
    )
    bias = None
    if layer.conv_bias:
        bias = torch.cat([layer.q_conv1d.bias, layer.k_conv1d.bias, layer.v_conv1d.bias], dim=0)
    return weight, bias


def _fused_conv(layer, q, k, v, weight, bias):
    # Per-call cost only: concat the three (separately projected) inputs, one conv, split.
    # `weight`/`bias` are pre-packed by _pack_qkv_conv_weight so weight concat isn't timed.
    qkv = torch.cat([q, k, v], dim=-1)
    qkv = causal_conv1d(
        x=qkv,
        weight=weight,
        bias=bias,
        activation=layer.q_conv1d.activation,
        backend=layer.q_conv1d.backend,
    )[0]
    return torch.split(qkv, [layer.key_dim, layer.key_dim, layer.value_dim], dim=-1)


def _op_inputs(layer, x):
    """Build the post-conv ``chunk_gated_delta_rule`` inputs as grad-enabled leaves."""
    with torch.no_grad():
        q = layer.q_conv1d(layer.q_proj(x))[0]
        k = layer.k_conv1d(layer.k_proj(x))[0]
        v = layer.v_conv1d(layer.v_proj(x))[0]
        g = layer.a_proj(x)
        beta = layer.b_proj(x)
    q = rearrange(q, "... (h d) -> ... h d", d=layer.head_k_dim).detach().requires_grad_(True)
    k = rearrange(k, "... (h d) -> ... h d", d=layer.head_k_dim).detach().requires_grad_(True)
    v = rearrange(v, "... (h d) -> ... h d", d=layer.head_v_dim).detach().requires_grad_(True)
    g = g.detach().requires_grad_(True)
    beta = beta.detach().requires_grad_(True)
    return q, k, v, g, beta


def _op_call(layer, q, k, v, g, beta):
    o, _ = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=layer.A_log,
        dt_bias=layer.dt_bias,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=layer.allow_neg_eigval,
        state_v_first=True,
        cu_seqlens=None,
    )
    return o


def make_runnable(provider, B, T, hidden_size, H, HV, D, V):
    """Return a zero-arg callable that runs *provider* once.

    Tensors/layer are captured in the closure so they stay alive for the
    lifetime of the returned callable.
    """
    torch.manual_seed(42)
    layer = _build_layer(hidden_size, H, HV, D, V)

    if provider == "layer_eval_fwd":
        # NOTE: eval-mode *prefill* only. GatedDeltaNet.forward switches to the
        # fused_recurrent decode kernel only for q_len <= 64; all default SHAPES have
        # T >= 128, so this measures the chunk path under inference_mode, not the
        # short-token / cache decode path.
        layer.eval()
        x = torch.randn(B, T, hidden_size, device=device, dtype=DTYPE)

        def fn():
            with torch.inference_mode():
                return layer(x)[0]

        return fn

    layer.train()

    if provider == "layer_train_fwd":
        x = torch.randn(B, T, hidden_size, device=device, dtype=DTYPE, requires_grad=True)

        def fn():
            return layer(x)[0]

        return fn

    if provider == "layer_train_fwdbwd":
        x = torch.randn(B, T, hidden_size, device=device, dtype=DTYPE, requires_grad=True)
        do = torch.randn(B, T, hidden_size, device=device, dtype=DTYPE)

        def fn():
            layer.zero_grad(set_to_none=True)
            x.grad = None
            layer(x)[0].backward(do)

        return fn

    if provider in ("op_fwd", "op_fwdbwd"):
        x = torch.randn(B, T, hidden_size, device=device, dtype=DTYPE)
        q, k, v, g, beta = _op_inputs(layer, x)
        if provider == "op_fwd":

            def fn():
                return _op_call(layer, q, k, v, g, beta)

            return fn

        with torch.no_grad():
            do = torch.randn_like(_op_call(layer, q, k, v, g, beta))

        def fn():
            layer.zero_grad(set_to_none=True)
            for t in (q, k, v, g, beta):
                t.grad = None
            torch.autograd.backward(_op_call(layer, q, k, v, g, beta), do)

        return fn

    # conv providers
    x = torch.randn(B, T, hidden_size, device=device, dtype=DTYPE)
    q = layer.q_proj(x).detach().requires_grad_(True)
    k = layer.k_proj(x).detach().requires_grad_(True)
    v = layer.v_proj(x).detach().requires_grad_(True)

    if "fused" in provider:
        weight, bias = _pack_qkv_conv_weight(layer)  # one-time pack, kept out of the timed path

        def conv_fn(layer, q, k, v):
            return _fused_conv(layer, q, k, v, weight, bias)

        # Validate fused == separate before timing so we never benchmark a wrong kernel.
        with torch.no_grad():
            separate = _separate_conv(layer, q.detach(), k.detach(), v.detach())
            fused = conv_fn(layer, q.detach(), k.detach(), v.detach())
        for name, actual, expected in zip(("q", "k", "v"), fused, separate):
            torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3, msg=f"fused conv {name} mismatch")
    else:
        conv_fn = _separate_conv

    if provider.endswith("_fwd"):

        def fn():
            return conv_fn(layer, q, k, v)

        return fn

    if provider.endswith("_fwdbwd"):
        with torch.no_grad():
            grad_outputs = tuple(torch.randn_like(o) for o in _separate_conv(layer, q, k, v))

        def fn():
            layer.zero_grad(set_to_none=True)
            q.grad = None
            k.grad = None
            v.grad = None
            torch.autograd.backward(conv_fn(layer, q, k, v), grad_outputs)

        return fn

    raise ValueError(provider)


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
def run(providers, shapes, repeat, warmup_iters, local_warmup_iters, do_bench_kw, order_seed) -> tuple[list[dict], list[dict]]:
    """Return (results, failures). *failures* is non-empty if any cell errored."""
    has_cuda = torch.cuda.is_available()
    cases = [(provider, shape) for provider in providers for shape in shapes]

    # Phase 1: warm up every (provider, shape) so autotuning is cached before any timing.
    print(f"\n  Warming up {len(providers)} provider(s) x {len(shapes)} shape(s)...")
    for provider, shape in cases:
        try:
            fn = make_runnable(provider, *shape)
            for _ in range(warmup_iters):
                fn()
            if has_cuda:
                torch.cuda.synchronize()
            del fn
        except Exception as e:
            logger.warning(f"Warmup failed for {provider} @ {tuple(shape)}: {e}")
        finally:
            gc.collect()
            if has_cuda:
                torch.cuda.empty_cache()
    print("  Warmup done.")

    # Phase 2: time each cell repeatedly. The display order stays stable, but the
    # measurement order changes per repeat to reduce provider/shape order effects.
    timings = {case: {name: [] for name, _ in QUANTILES} for case in cases}
    failures = []
    print(f"\n  Timing {len(cases)} cell(s) x {repeat} repeat(s)...")
    for repeat_idx in range(repeat):
        ordered_cases = list(cases)
        random.Random(order_seed + repeat_idx).shuffle(ordered_cases)
        print(f"  Repeat {repeat_idx + 1}/{repeat}...")
        for provider, shape in ordered_cases:
            try:
                fn = make_runnable(provider, *shape)
                for _ in range(local_warmup_iters):
                    fn()
                if has_cuda:
                    torch.cuda.synchronize()
                ms = triton.testing.do_bench(fn, quantiles=[quantile for _, quantile in QUANTILES], **do_bench_kw)
                for (name, _), value in zip(QUANTILES, ms):
                    timings[(provider, shape)][name].append(float(value))
                del fn
            except Exception as e:
                logger.warning(f"Bench failed for {provider} @ {tuple(shape)} repeat {repeat_idx + 1}: {e}")
                failures.append(
                    {"provider": provider, "shape": list(shape), "repeat": repeat_idx, "error": f"{type(e).__name__}: {e}"}
                )
            finally:
                gc.collect()
                if has_cuda:
                    torch.cuda.empty_cache()

    # Phase 3: measure the per-call memory delta once per successful cell.
    results = []
    for provider, shape in cases:
        repeat_medians = timings[(provider, shape)]["median"]
        if not repeat_medians:
            continue

        # peak_mb = peak CUDA memory of one call minus the resident setup (layer,
        # inputs, ...). This per-call working set is comparable across providers,
        # unlike a raw peak which folds in each provider's different resident tensors.
        # Free leftover tensors first so the baseline (and thus the delta) is clean.
        peak_mb = 0.0
        if has_cuda:
            gc.collect()
            torch.cuda.empty_cache()
            try:
                mem_fn = make_runnable(provider, *shape)
                torch.cuda.synchronize()
                baseline = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                mem_fn()
                torch.cuda.synchronize()
                peak_mb = max(0.0, (torch.cuda.max_memory_allocated() - baseline) / (1024**2))
                del mem_fn
            finally:
                gc.collect()
                torch.cuda.empty_cache()

        median_ms = float(statistics.median(repeat_medians))
        spread_pct = 0.0
        if len(repeat_medians) > 1 and median_ms > 0:
            spread_pct = float((max(repeat_medians) - min(repeat_medians)) / median_ms * 100)
        results.append(
            {
                "provider": provider,
                "shape": list(shape),
                "median_ms": median_ms,
                "p20_ms": float(statistics.median(timings[(provider, shape)]["p20"])),
                "p80_ms": float(statistics.median(timings[(provider, shape)]["p80"])),
                "repeat_medians_ms": repeat_medians,
                "repeat_p20_ms": timings[(provider, shape)]["p20"],
                "repeat_p80_ms": timings[(provider, shape)]["p80"],
                "min_ms": min(repeat_medians),
                "max_ms": max(repeat_medians),
                "spread_pct": spread_pct,
                "num_repeats": len(repeat_medians),
                "peak_mb": peak_mb,
            }
        )
    return results, failures


# --------------------------------------------------------------------------- #
# Machine info / git
# --------------------------------------------------------------------------- #
def _get_machine_info() -> dict:
    # Branch+sha via subprocess, matching the convention in benchmarks/ops/run.py and
    # benchmarks/benchmark_training_throughput.py (no extra git dependency).
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        git_label = f"{branch}[{sha}]"
    except Exception:
        git_label = "unknown"

    info = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "N/A",
        "git_label": git_label,
        "gpu_name": "N/A",
    }
    try:
        info["triton_version"] = triton.__version__
    except Exception:
        info["triton_version"] = "N/A"
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
    return info


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _short_label(git_label: str) -> str:
    """``feature-branch[abc1234]`` -> ``feature...[abc1234]`` (branch capped at 8 chars)."""
    if "[" in git_label:
        branch, sha = git_label.split("[", 1)
        sha = "[" + sha
    else:
        branch, sha = git_label, ""
    if len(branch) > 8:
        branch = branch[:8] + "..."
    return f"{branch}{sha}"


def print_results(results, machine_info, baseline=None, baseline_info=None):
    if not results:
        print("\n  No results to display.")
        return

    has_base = bool(baseline)
    base_map = {(r["provider"], tuple(r["shape"])): r for r in baseline} if has_base else {}

    new_label = _short_label(machine_info.get("git_label", "new"))
    base_label = _short_label((baseline_info or {}).get("git_label", "base")) if has_base else None

    col_w = max(12, len(new_label), len(base_label) if has_base else 0)

    if has_base:
        header = (
            f"  {'provider':<22s} {base_label:>{col_w}s} {new_label:>{col_w}s} "
            f"{'speedup':>8s} {'spread':>8s} {'note':>7s} {'dmem(MB)':>10s}"
        )
    else:
        header = f"  {'provider':<22s} {new_label:>{col_w}s} {'spread':>8s} {'dmem(MB)':>10s}"
    width = len(header) + 2
    sep = "=" * width

    print(f"\n{sep}")
    print(
        f"  Machine: {machine_info.get('gpu_name', 'N/A')} | "
        f"CUDA {machine_info.get('cuda_version', 'N/A')} | "
        f"PyTorch {machine_info.get('pytorch_version', 'N/A')}"
    )
    print("  dmem(MB) = per-call peak CUDA memory minus resident setup (working set)")
    print(sep)

    # Group rows by shape, preserving the order shapes first appear in results
    # (so --custom-shapes that aren't in the module-level SHAPES still print).
    by_shape = {}
    for r in results:
        by_shape.setdefault(tuple(r["shape"]), []).append(r)

    for shape in by_shape:
        B, T, hidden, H, HV, D, V = shape
        print(f"\n  shape: B={B} T={T} hidden={hidden} H={H} HV={HV} D={D} V={V}")
        print(f"  {'-' * (width - 2)}")
        print(header)
        rows = sorted(by_shape[shape], key=lambda r: PROVIDERS.index(r["provider"]))
        for r in rows:
            new_ms = r["median_ms"]
            peak = r["peak_mb"]
            spread = float(r.get("spread_pct", 0.0))
            if "spread_pct" not in r and new_ms > 0:
                spread = max(0.0, float(r.get("p80_ms", new_ms)) - float(r.get("p20_ms", new_ms))) / new_ms * 100
            if has_base:
                br = base_map.get((r["provider"], shape))
                if br:
                    base_ms = br["median_ms"]
                    speedup = base_ms / new_ms if new_ms > 0 else float("inf")
                    base_spread = float(br.get("spread_pct", 0.0))
                    if "spread_pct" not in br and base_ms > 0:
                        base_spread = max(0.0, float(br.get("p80_ms", base_ms)) - float(br.get("p20_ms", base_ms))) / base_ms * 100
                    note = "noisy" if abs(speedup - 1.0) * 100 <= max(spread, base_spread) else ""
                    print(
                        f"  {r['provider']:<22s} {base_ms:>{col_w}.4f} {new_ms:>{col_w}.4f} "
                        f"{speedup:>7.2f}x {spread:>7.1f}% {note:>7s} {peak:>10.1f}"
                    )
                else:
                    print(
                        f"  {r['provider']:<22s} {'-':>{col_w}s} {new_ms:>{col_w}.4f} "
                        f"{'-':>8s} {spread:>7.1f}% {'':>7s} {peak:>10.1f}"
                    )
            else:
                print(f"  {r['provider']:<22s} {new_ms:>{col_w}.4f} {spread:>7.1f}% {peak:>10.1f}")

    if has_base:
        groups = {
            "all": PROVIDERS,
            "layer": ["layer_train_fwd", "layer_train_fwdbwd", "layer_eval_fwd"],
            "op": ["op_fwd", "op_fwdbwd"],
            "conv": ["conv_separate_fwd", "conv_fused_fwd", "conv_separate_fwdbwd", "conv_fused_fwdbwd"],
            "layer_train": ["layer_train_fwd", "layer_train_fwdbwd"],
        }
        print("\n  Geomean speedups (main/reference median divided by current median):")
        for name, providers in groups.items():
            speedups = []
            noisy = 0
            for r in results:
                if r["provider"] not in providers:
                    continue
                br = base_map.get((r["provider"], tuple(r["shape"])))
                if not br:
                    continue
                new_ms = float(r["median_ms"])
                base_ms = float(br["median_ms"])
                speedup = base_ms / new_ms if new_ms > 0 else float("inf")
                speedups.append(speedup)

                spread = float(r.get("spread_pct", 0.0))
                if "spread_pct" not in r and new_ms > 0:
                    spread = max(0.0, float(r.get("p80_ms", new_ms)) - float(r.get("p20_ms", new_ms))) / new_ms * 100
                base_spread = float(br.get("spread_pct", 0.0))
                if "spread_pct" not in br and base_ms > 0:
                    base_spread = max(0.0, float(br.get("p80_ms", base_ms)) - float(br.get("p20_ms", base_ms))) / base_ms * 100
                noisy += int(abs(speedup - 1.0) * 100 <= max(spread, base_spread))
            if speedups:
                print(f"    - {name:<11s} {statistics.geometric_mean(speedups):.3f}x ({noisy}/{len(speedups)} noisy cells)")
    print(f"\n{sep}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Layer-level GatedDeltaNet benchmark")
    parser.add_argument(
        "--providers", nargs="+", default=PROVIDERS, choices=PROVIDERS, help="Subset of providers to run (default: all)"
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Path to a JSON file from a previous run; its results become the baseline (speedup) column. "
        "Produce one with --json on the other checkout. Only the file is read; no code is executed.",
    )
    parser.add_argument(
        "--custom-shapes", default=None, help="JSON list of [B,T,hidden,H,HV,D,V] shapes overriding the defaults"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=int(os.environ.get("FLA_BENCH_REPEAT", "3")),
        help="Number of independent timing repeats per provider/shape (default: env FLA_BENCH_REPEAT or 3)",
    )
    parser.add_argument(
        "--warmup-ms",
        type=int,
        default=int(os.environ.get("FLA_BENCH_WARMUP_MS", "50")),
        help="triton.testing.do_bench warmup window in ms (default: env FLA_BENCH_WARMUP_MS or 50)",
    )
    parser.add_argument(
        "--rep-ms",
        type=int,
        default=int(os.environ.get("FLA_BENCH_REP_MS", "300")),
        help="triton.testing.do_bench measurement window in ms (default: env FLA_BENCH_REP_MS or 300)",
    )
    parser.add_argument(
        "--op-warmup-iters",
        type=int,
        default=int(os.environ.get("FLA_BENCH_OP_WARMUP_ITERS", "5")),
        help="Initial untimed warmup calls per provider/shape (default: env FLA_BENCH_OP_WARMUP_ITERS or 5)",
    )
    parser.add_argument(
        "--local-warmup-iters",
        type=int,
        default=int(os.environ.get("FLA_BENCH_LOCAL_WARMUP_ITERS", "1")),
        help="Untimed calls before each timing repeat for the exact closure being measured "
        "(default: env FLA_BENCH_LOCAL_WARMUP_ITERS or 1)",
    )
    parser.add_argument(
        "--order-seed",
        type=int,
        default=int(os.environ.get("FLA_BENCH_ORDER_SEED", "42")),
        help="Seed used to shuffle measurement order per repeat (default: env FLA_BENCH_ORDER_SEED or 42)",
    )
    parser.add_argument("--json", dest="json_file", default=None, help="Write results to this JSON path")
    args = parser.parse_args()

    shapes = [tuple(s) for s in json.loads(args.custom_shapes)] if args.custom_shapes else list(SHAPES)
    repeat = max(1, args.repeat)
    warmup_iters = max(0, args.op_warmup_iters)
    local_warmup_iters = max(0, args.local_warmup_iters)
    do_bench_kw = {"warmup": max(1, args.warmup_ms), "rep": max(1, args.rep_ms)}
    benchmark_config = {
        "providers": args.providers,
        "shapes": [list(s) for s in shapes],
        "repeat": repeat,
        "op_warmup_iters": warmup_iters,
        "local_warmup_iters": local_warmup_iters,
        "warmup_ms": do_bench_kw["warmup"],
        "rep_ms": do_bench_kw["rep"],
        "order_seed": args.order_seed,
        "quantiles": {name: quantile for name, quantile in QUANTILES},
    }

    machine_info = _get_machine_info()
    print(
        f"Machine: {machine_info['gpu_name']} | CUDA {machine_info['cuda_version']} | "
        f"PyTorch {machine_info['pytorch_version']} | {machine_info['git_label']}"
    )
    print(f"Providers: {args.providers}")
    print(f"Shapes: {len(shapes)}")
    print(
        f"Timing config: repeat={repeat}, warmup_ms={do_bench_kw['warmup']}, rep_ms={do_bench_kw['rep']}, "
        f"op_warmup_iters={warmup_iters}, local_warmup_iters={local_warmup_iters}, order_seed={args.order_seed}"
    )

    results, failures = run(
        providers=args.providers,
        shapes=shapes,
        repeat=repeat,
        warmup_iters=warmup_iters,
        local_warmup_iters=local_warmup_iters,
        do_bench_kw=do_bench_kw,
        order_seed=args.order_seed,
    )

    baseline, baseline_info = None, None
    if args.baseline:
        with open(args.baseline) as f:
            data = json.load(f)
        baseline = data.get("results", [])
        baseline_info = data.get("machine_info")

    print_results(results, machine_info, baseline=baseline, baseline_info=baseline_info)

    if args.json_file:
        with open(args.json_file, "w") as f:
            json.dump(
                {
                    "machine_info": machine_info,
                    "benchmark_config": benchmark_config,
                    "results": results,
                    "failures": failures,
                },
                f,
                indent=2,
            )
        print(f"\nResults saved to {args.json_file}")

    if failures:
        print(f"\n  {len(failures)} benchmark cell(s) FAILED:")
        for fail in failures:
            print(f"    - {fail['provider']} @ {tuple(fail['shape'])}: {fail['error']}")
        # Exit non-zero so a broken provider can't masquerade as a clean run in PR evidence/CI.
        raise SystemExit(1)

    return results


if __name__ == "__main__":
    main()

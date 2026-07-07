# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Forced gate-state stress reproducer for GDN2.

The script initializes gate parameters to the requested values before the first
forward/backward. It tests whether a gate state is numerically safe under a
chosen optimizer; it does not prove that the optimizer naturally reaches that
state during training.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fla.layers import GatedDeltaNet2  # noqa: E402
from fla.ops.gdn2 import chunk_gdn2  # noqa: E402


class TinyPrenormGDN2(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        safe_gate: bool,
        gate_lower_bound: float,
    ) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size)
        self.gdn = GatedDeltaNet2(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_heads=num_heads,
            num_v_heads=num_heads,
            mode="chunk",
            use_short_conv=False,
            safe_gate=safe_gate,
            gate_lower_bound=gate_lower_bound,
        )
        self.out = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _, _ = self.gdn(self.norm(x))
        return self.out(y)


class TinyGDN2Op(nn.Module):
    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        a_log: float,
        dt_bias: float,
        gate_preact: float,
        safe_gate: bool,
        gate_lower_bound: float,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.safe_gate = safe_gate
        self.gate_lower_bound = gate_lower_bound
        rows = batch_size * seq_len * num_heads
        state_rows = batch_size * num_heads * head_dim
        self.q = nn.Parameter(torch.randn(rows, head_dim) * 0.02)
        self.k = nn.Parameter(torch.randn(rows, head_dim) * 0.02)
        self.v = nn.Parameter(torch.randn(rows, head_dim) * 0.02)
        self.raw_g = nn.Parameter(torch.full((rows, head_dim), gate_preact))
        self.b_logit = nn.Parameter(torch.zeros(rows, head_dim))
        self.w_logit = nn.Parameter(torch.zeros(rows, head_dim))
        self.initial_state = nn.Parameter(torch.randn(state_rows, head_dim) * 0.02)
        self.A_log = nn.Parameter(torch.full((num_heads,), a_log, dtype=torch.float32))
        self.A_log._no_weight_decay = True
        self.dt_bias = nn.Parameter(torch.full((num_heads * head_dim,), dt_bias, dtype=torch.float32))
        self.dt_bias._no_weight_decay = True

    def _sequence_view(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(self.batch_size, self.seq_len, self.num_heads, self.head_dim)

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        q = self._sequence_view(self.q)
        k = self._sequence_view(self.k)
        v = self._sequence_view(self.v)
        raw_g = self._sequence_view(self.raw_g)
        b = self._sequence_view(self.b_logit).sigmoid()
        w = self._sequence_view(self.w_logit).sigmoid()
        initial_state = self.initial_state.view(self.batch_size, self.num_heads, self.head_dim, self.head_dim)
        if self.safe_gate:
            g = raw_g
            kwargs = {
                "use_gate_in_kernel": True,
                "safe_gate": True,
                "lower_bound": self.gate_lower_bound,
                "A_log": self.A_log,
                "dt_bias": self.dt_bias,
            }
        else:
            H, K = raw_g.shape[-2:]
            g = -self.A_log.float().exp().view(1, 1, H, 1) * F.softplus(
                raw_g.float() + self.dt_bias.view(1, 1, H, K)
            )
            kwargs = {}
        return chunk_gdn2(
            q=q,
            k=k,
            v=v,
            g=g,
            b=b,
            w=w,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            disable_recompute=True,
            **kwargs,
        )


class TinyGate(nn.Module):
    def __init__(
        self,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        a_log: float,
        dt_bias: float,
        gate_preact: float,
        safe_gate: bool,
        gate_lower_bound: float,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.safe_gate = safe_gate
        self.gate_lower_bound = gate_lower_bound
        self.raw_g = nn.Parameter(torch.full((seq_len * num_heads, head_dim), gate_preact))
        self.A_log = nn.Parameter(torch.full((num_heads,), a_log, dtype=torch.float32))
        self.A_log._no_weight_decay = True
        self.dt_bias = nn.Parameter(torch.full((num_heads * head_dim,), dt_bias, dtype=torch.float32))
        self.dt_bias._no_weight_decay = True

    def forward(self) -> torch.Tensor:
        raw_g = self.raw_g.view(self.seq_len, self.num_heads, self.head_dim)
        x = raw_g.float() + self.dt_bias.view(self.num_heads, self.head_dim)
        a = self.A_log.float().exp().view(1, self.num_heads, 1)
        if self.safe_gate:
            return self.gate_lower_bound * torch.sigmoid(a * x)
        return -a * F.softplus(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stress a tiny GDN2 step from a forced gate state.",
    )
    parser.add_argument("--module", choices=("gate", "op", "layer"), default="op", help="which tiny training module to run")
    parser.add_argument("--safe-gate", action="store_true", help="use the bounded in-kernel gate path")
    parser.add_argument("--expect-nonfinite", action="store_true", help="return success only if a non-finite is reproduced")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--a-log", type=float, default=20.0)
    parser.add_argument("--dt-bias", type=float, default=0.0)
    parser.add_argument("--gate-preact", type=float, default=32.0)
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--gate-lower-bound", type=float, default=-5.0)
    parser.add_argument("--loss-scale", type=float, default=1.0)
    parser.add_argument("--state-loss-weight", type=float, default=1.0)
    parser.add_argument("--optimizer", choices=("muon", "adam", "adamw"), default="muon")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.95)
    return parser.parse_args()


def finite_status(model: nn.Module) -> tuple[bool, str]:
    for name, p in model.named_parameters():
        if not torch.isfinite(p).all():
            return False, f"parameter {name}"
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return False, f"gradient {name}"
    return True, ""


def max_grad(model: nn.Module) -> float:
    value = 0.0
    for p in model.parameters():
        if p.grad is not None:
            value = max(value, p.grad.detach().float().abs().max().item())
    return value


def make_optimizers(model: nn.Module, args: argparse.Namespace) -> list[torch.optim.Optimizer]:
    no_weight_decay = set()
    for module in model.modules():
        for name, p in module.named_parameters(recurse=False):
            if getattr(p, "_no_weight_decay", False) or name.endswith("bias") or p.ndim < 2:
                no_weight_decay.add(id(p))

    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer != "muon":
        optimizer_cls = torch.optim.Adam if args.optimizer == "adam" else torch.optim.AdamW
        decayed = [p for p in params if id(p) not in no_weight_decay]
        nondecayed = [p for p in params if id(p) in no_weight_decay]
        groups = []
        if decayed:
            groups.append({"params": decayed, "weight_decay": args.weight_decay})
        if nondecayed:
            groups.append({"params": nondecayed, "weight_decay": 0.0})
        return [optimizer_cls(groups, lr=args.lr)]

    if not hasattr(torch.optim, "Muon"):
        raise RuntimeError("this PyTorch build does not expose torch.optim.Muon")

    muon_params = []
    adamw_params = []
    for p in params:
        if p.ndim >= 2:
            muon_params.append(p)
        else:
            adamw_params.append(p)

    optimizers: list[torch.optim.Optimizer] = [
        torch.optim.Muon(
            muon_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
        )
    ]
    decayed, nondecayed = [], []
    for p in adamw_params:
        if id(p) in no_weight_decay:
            nondecayed.append(p)
        else:
            decayed.append(p)
    if decayed:
        optimizers.append(torch.optim.AdamW(decayed, lr=args.lr, weight_decay=args.weight_decay))
    if nondecayed:
        optimizers.append(torch.optim.AdamW(nondecayed, lr=args.lr, weight_decay=0.0))
    return optimizers


def init_stress_gate(model: TinyPrenormGDN2, args: argparse.Namespace) -> None:
    with torch.no_grad():
        model.gdn.A_log.fill_(args.a_log)
        model.gdn.dt_bias.fill_(args.dt_bias)
        model.gdn.f_proj[0].weight.fill_(1.0 / args.hidden_size)
        model.gdn.f_proj[1].weight.fill_(args.gate_preact / model.gdn.head_v_dim)


def make_batch(args: argparse.Namespace, device: torch.device, step: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(args.seed + step)
    x = torch.ones(args.batch_size, args.seq_len, args.hidden_size, device=device)
    x = x.mul(args.input_scale)
    if args.noise:
        x = x + args.noise * torch.randn(x.shape, generator=generator, device=device)
    target = torch.zeros_like(x)
    return x, target


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the GDN2 Triton kernels")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    if args.module == "layer":
        model = TinyPrenormGDN2(
            hidden_size=args.hidden_size,
            head_dim=args.head_dim,
            num_heads=args.num_heads,
            safe_gate=args.safe_gate,
            gate_lower_bound=args.gate_lower_bound,
        ).to(device)
        init_stress_gate(model, args)
    elif args.module == "gate":
        model = TinyGate(
            seq_len=args.seq_len,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            a_log=args.a_log,
            dt_bias=args.dt_bias,
            gate_preact=args.gate_preact,
            safe_gate=args.safe_gate,
            gate_lower_bound=args.gate_lower_bound,
        ).to(device)
    else:
        model = TinyGDN2Op(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            a_log=args.a_log,
            dt_bias=args.dt_bias,
            gate_preact=args.gate_preact,
            safe_gate=args.safe_gate,
            gate_lower_bound=args.gate_lower_bound,
        ).to(device)
    model.train()
    optimizers = make_optimizers(model, args)

    mode = "safe_gate" if args.safe_gate else "vanilla"
    optimizer_name = {"muon": "Muon", "adam": "Adam", "adamw": "AdamW"}[args.optimizer]
    print(
        "config "
        f"scenario=forced_gate_state module={args.module} mode={mode} "
        f"torch={torch.__version__} cuda={torch.version.cuda} "
        f"gpu={torch.cuda.get_device_name(0)} a_log={args.a_log} "
        f"gate_preact={args.gate_preact} optimizer=torch.optim.{optimizer_name}"
    )
    if args.safe_gate:
        print(f"bounded gate output is in [{args.gate_lower_bound}, 0)")
    else:
        expected_gate = -math.exp(args.a_log) * torch.nn.functional.softplus(
            torch.tensor(args.gate_preact + args.dt_bias)
        ).item()
        print(f"first-step vanilla gate is approximately {expected_gate:.3e}")

    reproduced_nonfinite = False
    failure = ""
    for step in range(args.steps):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        try:
            if args.module == "layer":
                x, target = make_batch(args, device, step)
                y = model(x)
                loss = (y.float() - target.float()).square().mean()
            elif args.module == "gate":
                y = model()
                loss = y.float().square().mean()
            else:
                y, final_state = model()
                loss = y.float().square().mean()
                if final_state is not None:
                    loss = loss + args.state_loss_weight * final_state.float().square().mean()
            loss = loss * args.loss_scale
            loss.backward()
            torch.cuda.synchronize()
        except RuntimeError as exc:
            reproduced_nonfinite = True
            failure = f"runtime error at step {step}: {exc}"
            print(failure)
            break

        is_finite, where = finite_status(model)
        loss_finite = torch.isfinite(loss.detach()).item()
        print(
            f"step={step} loss={loss.detach().item():.6e} "
            f"max_grad={max_grad(model):.6e} finite={bool(is_finite and loss_finite)}"
        )
        if not loss_finite or not is_finite:
            reproduced_nonfinite = True
            failure = f"non-finite {'loss' if not loss_finite else where} at step {step}"
            print(failure)
            break

        for optimizer in optimizers:
            optimizer.step()
        torch.cuda.synchronize()
        is_finite, where = finite_status(model)
        if not is_finite:
            reproduced_nonfinite = True
            failure = f"non-finite {where} after optimizer step {step}"
            print(failure)
            break

    if args.expect_nonfinite:
        if reproduced_nonfinite:
            print("result: reproduced expected non-finite behavior")
            return 0
        print("result: expected non-finite behavior, but run stayed finite")
        return 1

    if reproduced_nonfinite:
        print(f"result: failed with {failure}")
        return 1
    print("result: completed with finite loss, gradients, and parameters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

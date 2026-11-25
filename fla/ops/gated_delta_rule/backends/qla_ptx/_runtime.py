# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
# Copyright 2026 Co-developed by KDA (Kernel Design Agents)

"""Load checked PTX modules and launch their exported CUDA ABI."""

import hashlib
import inspect
import json
from functools import cache
from pathlib import Path

from torch.utils.cpp_extension import CUDA_HOME

ROOT = Path(__file__).resolve().parent


def binding_key(frame, ordinal):
    names = inspect.getargvalues(frame).args
    signature = json.dumps(
        [
            frame.f_globals["__name__"].split(".")[-1],
            frame.f_code.co_name,
            {name: frame.f_locals[name] for name in names},
        ],
        sort_keys=True,
    )
    return f"{signature}:{ordinal}"


@cache
def records():
    return json.loads((ROOT / "manifest.json").read_text())["bindings"]


class KernelRef:
    def retrace_with_globals(self, _globals):
        return self


class Module:
    def __init__(self, record):
        from tvm_ffi import cpp

        payload = ROOT / record["ptx"]
        launcher = ROOT / record["shim"]
        for path, expected in (
            (payload, record["ptx_sha256"]),
            (launcher, record["shim_sha256"]),
        ):
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Artifact digest mismatch: {path.name}")
        module = cpp.load_inline(
            f"qla_ptx_{payload.stem}",
            cpp_sources=launcher.read_text(),
            embed_cubin={record["module_ident"]: payload.read_bytes() + b"\0"},
            extra_include_paths=[str(Path(CUDA_HOME) / "include")],
            extra_ldflags=["-lcuda"],
        )
        self.run = module[record["entry"]]
        self.module = module
        self.keys = [name for kind, name in record["arg_plan"] if kind != "grid"]

    def launch(self, *, grid, **bindings):
        if set(bindings) != set(self.keys):
            raise ValueError("Bindings do not match the exported kernel ABI")
        self.run(*(bindings[name] for name in self.keys), *grid)


def load_module(_schedule, *, _ordinal, **_options):
    key = binding_key(inspect.currentframe().f_back, _ordinal)
    try:
        record = records()[key]
    except KeyError as error:
        raise ValueError(f"Specialization was not exported: {key}") from error
    options = {name: value for name, value in _options.items() if name.isupper()}
    if options != record["specializations"] or _options.get("use_pdl", False) != record["use_pdl"]:
        raise ValueError("Exported specialization or launch policy mismatch")
    return Module(record)

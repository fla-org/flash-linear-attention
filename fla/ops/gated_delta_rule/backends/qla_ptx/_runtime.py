# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
"""Load exported PTX through its CUDA launch ABI."""

import inspect
import json
from pathlib import Path

from torch.utils.cpp_extension import CUDA_HOME

ROOT = Path(__file__).resolve().parent


class Module:
    def __init__(self, record):
        from tvm_ffi import cpp

        ptx = ROOT / record["ptx"]
        self.module = cpp.load_inline(
            f"qla_ptx_{ptx.stem}",
            cpp_sources=(ROOT / record["shim"]).read_text(),
            embed_cubin={record["module_ident"]: ptx.read_bytes() + b"\x00"},
            extra_include_paths=[str(Path(CUDA_HOME) / "include")],
            extra_ldflags=["-lcuda"],
        )
        self.run = self.module[record["entry"]]
        self.keys = [key for kind, key in record["arg_plan"] if kind != "grid"]

    def launch(self, *, grid, **bindings):
        if set(bindings) != set(self.keys):
            raise ValueError("Bindings do not match the exported kernel ABI")
        self.run(*(bindings[key] for key in self.keys), *grid)


def load_kernel(**_options):
    frame = inspect.currentframe().f_back
    key = json.dumps(
        [
            frame.f_globals["__name__"].split(".")[-1],
            frame.f_code.co_name,
            {name: frame.f_locals[name] for name in inspect.getargvalues(frame).args},
        ],
        sort_keys=True,
    )
    records = json.loads((ROOT / "manifest.json").read_text())["bindings"]
    specializations = {name: value for name, value in _options.items() if name.isupper()}
    matches = [
        record
        for name, record in records.items()
        if name.startswith(key + ":")
        and record["specializations"] == specializations
        and (record["use_pdl"] == _options.get("use_pdl", False))
    ]
    if len(matches) != 1:
        raise ValueError(f"Specialization was not exported: {key}")
    return Module(matches[0])

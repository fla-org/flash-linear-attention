# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import importlib
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
__version__ = "0.5.1"

__all__: list[str] = []


def _export_optional_public_api(module_name: str) -> None:
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            return
        raise

    globals()[module_name.rsplit(".", maxsplit=1)[-1]] = module
    for name in module.__all__:
        if name.endswith("Config"):
            continue
        globals()[name] = getattr(module, name)
        __all__.append(name)


_export_optional_public_api("fla.layers")
_export_optional_public_api("fla.models")

del _export_optional_public_api

# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import importlib
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
__version__ = "0.5.2"

__all__: list[str] = []


def _import_optional_public_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = exc.name
        # The extension package is optional. Treat its absence, or the absence
        # of an external runtime dependency, as the extension being unavailable.
        if missing == module_name or (missing is not None and missing.split('.', 1)[0] != 'fla'):
            return None
        raise


def _export_public_api(module) -> None:
    globals()[module.__name__.rsplit('.', maxsplit=1)[-1]] = module
    for name in module.__all__:
        if name.endswith('Config'):
            continue
        globals()[name] = getattr(module, name)
        __all__.append(name)


_layers = _import_optional_public_module('fla.layers')
_models = _import_optional_public_module('fla.models')
if _layers is not None and _models is not None:
    _export_public_api(_layers)
    _export_public_api(_models)

del _import_optional_public_module, _export_public_api, _layers, _models

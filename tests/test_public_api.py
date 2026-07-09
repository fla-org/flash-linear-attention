# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import fla
import fla.layers as layers
import fla.models as models


def test_top_level_exports_layers_and_non_config_models():
    expected_exports = [
        *layers.__all__,
        *[name for name in models.__all__ if not name.endswith("Config")],
    ]

    assert fla.__all__ == expected_exports

    for name in expected_exports:
        source = layers if hasattr(layers, name) else models
        assert getattr(fla, name) is getattr(source, name)

    config_exports = [name for name in models.__all__ if name.endswith("Config")]
    assert not set(config_exports).intersection(fla.__all__)
    assert not any(name in fla.__dict__ for name in config_exports)

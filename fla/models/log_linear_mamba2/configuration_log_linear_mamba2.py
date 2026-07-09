# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from fla.models.mamba2 import Mamba2Config


class LogLinearMamba2Config(Mamba2Config):

    model_type = "log_linear_mamba2"

    def __init__(
        self,
        residual_in_fp32: bool = False,
        chunk_size: int = 64,
        attnres_block_size: int | None = None,
        **kwargs,
    ):
        self.attnres_block_size = attnres_block_size

        if attnres_block_size is not None and attnres_block_size != 1:
            if attnres_block_size < 2 or attnres_block_size % 2 != 0:
                raise ValueError(
                    "`attnres_block_size` must be `None`, `1` (full mode), or an even integer (one block "
                    f"contains `attnres_block_size // 2` transformer layers); got {attnres_block_size}."
                )

        super().__init__(
            residual_in_fp32=residual_in_fp32,
            chunk_size=chunk_size,
            **kwargs,
        )

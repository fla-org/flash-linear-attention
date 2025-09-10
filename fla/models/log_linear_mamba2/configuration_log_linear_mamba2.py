from fla.models.mamba2 import Mamba2Config


class LogLinearMamba2Config(Mamba2Config):

    model_type = "log_linear_mamba2"

    def __init__(
        self,
        residual_in_fp32: bool = False,
        chunk_size: int = 64,
        **kwargs,
    ):
        super().__init__(
            residual_in_fp32=residual_in_fp32,
            chunk_size=chunk_size,
            **kwargs,
        )

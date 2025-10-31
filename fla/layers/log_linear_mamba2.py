import math
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
from einops import rearrange
from transformers.activations import ACT2FN
from transformers.utils import logging

from fla.layers.mamba2 import apply_mask_to_padding_states, causal_conv1d_fn, causal_conv1d_update, is_fast_path_available
from fla.modules.layernorm_gated import RMSNormGated, rmsnorm_fn
from fla.ops.log_linear_attn.chunk import LogLinearAttentionState, chunk_log_linear_attn

if TYPE_CHECKING:
    from fla.models.log_linear_mamba2.modeling_log_linear_mamba2 import LogLinearMamba2Cache

logger = logging.get_logger(__name__)


def ceil_log(x: int, b: int) -> int:
    return math.ceil(math.log(x, b))


def get_num_levels(length: int, base: int) -> int:
    return ceil_log(length, base) + 1


MAX_SEQUENCE_LENGTH = 2048 * 8
LAMBDA_LEVEL_BASE = 2
MAX_NUM_LEVELS = get_num_levels(length=MAX_SEQUENCE_LENGTH, base=LAMBDA_LEVEL_BASE)


def hmamba_chunk_scan_combined(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    dl: torch.Tensor,
    L: torch.Tensor,
    chunk_size: int,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    initial_states: LogLinearAttentionState | None = None,
    seq_idx: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    dt_softplus: bool = False,
    dt_limit: tuple[float, float] = (0.0, float("inf")),
    return_final_states: bool = False,
):
    if z is not None:
        raise NotImplementedError
    if seq_idx is not None:
        raise NotImplementedError
    if cu_seqlens is not None:
        raise NotImplementedError
    if dt_softplus is not True:
        raise NotImplementedError
    if tuple(dt_limit) != (0.0, float("inf")):
        raise NotImplementedError
    if chunk_size != 64:
        raise NotImplementedError
    if not B.shape == C.shape:
        raise ValueError("B and C must have the same shape")

    if D is not None:
        if D.dim() != 1:
            raise ValueError
        D = rearrange(D, "h -> 1 1 h 1")
        D_residual = x * D

    if dt_bias is not None:
        dt = dt + rearrange(dt_bias, "h -> 1 1 h")
    if dt_softplus:
        dt = torch.nn.functional.softplus(dt)
    if dt_limit != (0.0, float("inf")):
        dt = torch.clamp(dt, min=dt_limit[0], max=dt_limit[1])
    x = (x * rearrange(dt, "b l h -> b l h 1")).to(x.dtype)
    A = rearrange(A, "h -> 1 1 h") * dt

    L = torch.nn.functional.softplus(rearrange(L, "h ell -> 1 1 h ell") * dl).to(L.dtype)

    y, state = chunk_log_linear_attn(
        q=C,
        k=B,
        v=x,
        g=A,
        level_scales=L,
        initial_state=initial_states,
        output_final_state=return_final_states,
        cu_seqlens=cu_seqlens,
    )

    if D is not None:
        y = y + D_residual

    return y, state


def hmamba_split_conv1d_scan_combined(
    zxbcdtdl: torch.Tensor,
    conv1d_weight: torch.Tensor,
    conv1d_bias: torch.Tensor,
    dt_bias: torch.Tensor,
    A: torch.Tensor,
    L: torch.Tensor,
    D: torch.Tensor,
    chunk_size: int,
    initial_states: torch.Tensor | None = None,
    seq_idx: torch.Tensor | None = None,
    dt_limit: tuple[float, float] = (0.0, float("inf")),
    return_final_states: bool = False,
    activation: str = "silu",
    rmsnorm_weight: torch.Tensor | None = None,
    rmsnorm_eps: float = 1e-6,
    outproj_weight: torch.Tensor | None = None,
    outproj_bias: torch.Tensor | None = None,
    headdim: int | None = None,
    ngroups: int = 1,
    norm_before_gate: bool = True,
) -> torch.Tensor:
    """
    Argument:
        zxbcdtdl: (batch, seqlen, 2 * dim + 2 * ngroups * dstate + nheads) where dim == nheads * headdim
        conv1d_weight: (dim + 2 * ngroups * dstate, width)
        conv1d_bias: (dim + 2 * ngroups * dstate,)
        dt_bias: (nheads,)
        A: (nheads)
        L: (nheads, nlevels)
        D: (nheads, headdim) or (nheads,)
        initial_states: (batch, nheads, headdim, dstate)
        seq_idx: (batch, seqlen), int32
        rmsnorm_weight: (dim,)
        outproj_weight: (out_dim, dim)
        outproj_bias: (out_dim,)
        headdim: if D is 1D, headdim must be passed in
        norm_before_gate: if True, we do RMSNorm(x) * F.silu(z). If False, we do RMSNorm(x * F.silu(z))
    Return:
        out: (batch, seqlen, dim)
    """
    if initial_states is not None:
        raise NotImplementedError
    if seq_idx is not None:
        raise NotImplementedError
    if dt_limit != (0.0, float("inf")):
        raise NotImplementedError
    if return_final_states is not False:
        raise NotImplementedError
    if norm_before_gate is not False:
        raise NotImplementedError
    if rmsnorm_weight is None:
        raise NotImplementedError
    if activation not in ["silu", "swish"]:
        raise NotImplementedError

    batch, seqlen, _ = zxbcdtdl.shape
    dlambda = L.shape[-1]
    (nheads,) = D.shape
    dim = nheads * headdim
    dstate = (zxbcdtdl.shape[-1] - 2 * dim - nheads - nheads * dlambda) // ngroups // 2

    if D.dim() != 1:
        raise ValueError
    if headdim is None:
        raise ValueError
    if nheads % ngroups != 0:
        raise ValueError
    if zxbcdtdl.shape != (
        batch,
        seqlen,
        2 * dim + 2 * ngroups * dstate + nheads + nheads * dlambda,
    ):
        raise ValueError
    if dt_bias.shape != (nheads,):
        raise ValueError
    if A.shape != (nheads,):
        raise ValueError
    if L.shape != (nheads, dlambda):
        raise ValueError
    if D.shape != (nheads,):
        raise ValueError
    if rmsnorm_weight is None:
        raise ValueError

    zxBCdtl_splits = [dim, dim + 2 * ngroups * dstate, nheads, nheads * dlambda]
    xBC_splits = [dim, ngroups * dstate, ngroups * dstate]
    z, xBC, dt, dl = torch.split(zxbcdtdl, zxBCdtl_splits, dim=-1)
    xBC = rearrange(
        causal_conv1d_fn(
            rearrange(xBC, "b s d -> b d s"),
            conv1d_weight,
            bias=conv1d_bias,
            activation=activation,
            seq_idx=seq_idx,
        ),
        "b d s -> b s d",
    )
    x, B, C = torch.split(xBC, xBC_splits, dim=-1)
    x = rearrange(x, "b l (h p) -> b l h p", h=nheads, p=headdim)
    B = rearrange(B, "b l (g n) -> b l g n", g=ngroups, n=dstate)
    C = rearrange(C, "b l (g n) -> b l g n", g=ngroups, n=dstate)
    dl = rearrange(dl, "b l (h ell) -> b l h ell", h=nheads, ell=dlambda)
    y, _ = hmamba_chunk_scan_combined(
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        dl=dl,
        L=L,
        chunk_size=chunk_size,
        D=D,
        z=z if rmsnorm_weight is None else None,
        dt_bias=dt_bias,
        dt_softplus=True,
        seq_idx=seq_idx,
        cu_seqlens=None,
        dt_limit=dt_limit,
        return_final_states=return_final_states,
    )

    y = rearrange(y, "b l h p -> b l (h p)")
    if rmsnorm_weight is not None:
        y = rmsnorm_fn(
            x=y,
            weight=rmsnorm_weight,
            bias=None,
            z=z,
            eps=rmsnorm_eps,
            group_size=None,
            norm_before_gate=False,
        )
    out = torch.nn.functional.linear(y, outproj_weight, outproj_bias)
    return out


class LogLinearMamba2(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int = 64,
        hidden_size: int = 2048,
        state_size: int = 128,
        expand: int = 2,
        n_groups: int = 1,
        conv_kernel: int = 4,
        use_conv_bias: bool = False,
        hidden_act: str = "silu",
        rms_norm: bool = True,
        chunk_size: int = 64,
        time_step_rank: float = 256,
        time_step_limit: tuple[float, float] = (0.0, float("inf")),
        time_step_min: float = 0.001,
        time_step_max: float = 0.1,
        use_bias: bool = True,
        norm_eps: float = 1e-5,
        layer_idx: int = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.ssm_state_size = state_size
        self.conv_kernel_size = conv_kernel
        self.intermediate_size = int(expand * self.hidden_size)
        self.time_step_rank = int(time_step_rank)
        self.layer_idx = layer_idx
        self.use_conv_bias = use_conv_bias
        self.activation = hidden_act
        self.act = ACT2FN[hidden_act]

        self.layer_norm_epsilon = norm_eps
        self.rms_norm = rms_norm

        self.n_groups = n_groups
        self.head_dim = head_dim
        self.chunk_size = chunk_size

        self.time_step_limit = time_step_limit
        self.time_step_min = time_step_min
        self.time_step_max = time_step_max

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=use_conv_bias,
            kernel_size=conv_kernel,
            groups=self.conv_dim,
            padding=conv_kernel - 1,
        )

        self.num_lambda_dims = MAX_NUM_LEVELS
        self.lambda_level_module = None

        # projection of the input hidden states
        projection_size = (
            self.intermediate_size
            + self.conv_dim
            + self.num_heads * (self.num_lambda_dims + 1)
        )
        self.in_proj = nn.Linear(
            self.hidden_size,
            projection_size,
            bias=use_bias,
        )
        # selective projection used to make dt, B and C input dependant

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(torch.ones(self.num_heads))

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        A = torch.arange(1, self.num_heads + 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        self.lambda_mode = "positive"
        L = torch.ones(self.num_heads, self.num_lambda_dims)
        self.L = nn.Parameter(L)
        self.L._no_weight_decay = True

        self.norm = RMSNormGated(
            self.intermediate_size, eps=self.layer_norm_epsilon, norm_before_gate=False,
        )
        self.D = nn.Parameter(torch.ones(self.num_heads))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=use_bias,
        )
        self.use_bias = use_bias

        if not is_fast_path_available:
            logger.warning_once(
                "The fast path is not available because one of "
                "`(selective_state_update, causal_conv1d_fn, causal_conv1d_update)` is None. "
                "Falling back to the naive implementation. "
                "To install follow https://github.com/state-spaces/mamba/#installation and"
                "https://github.com/Dao-AILab/causal-conv1d",
            )

    def cuda_kernels_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional["LogLinearMamba2Cache"] = None,
        cache_position: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ):
        if attention_mask is not None:
            # only supporting this in decoding
            if cache_params is None:
                raise NotImplementedError
        if self.activation not in ["silu", "swish"]:
            raise ValueError

        # 1. Gated MLP's linear projection
        hidden_states = apply_mask_to_padding_states(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        projected_states = self.in_proj(hidden_states)

        # Set up dimensions for reshapes later
        batch_size, seq_len, _ = hidden_states.shape
        groups_time_state_size = self.n_groups * self.ssm_state_size
        d_mlp = (
            projected_states.shape[-1]
            - 2 * self.intermediate_size
            - 2 * self.n_groups * self.ssm_state_size
            - self.num_heads * (self.num_lambda_dims + 1)
        ) // 2
        if d_mlp != 0:
            raise ValueError

        # Single step calculations via cache
        if (
            cache_params is not None
            and cache_position is not None
            and cache_position[0] > 0
        ):
            if hidden_states.shape[1] != 1:
                raise ValueError

            gate, xBC, dt, dl = torch.split(
                projected_states.squeeze(1),
                [
                    self.intermediate_size,
                    self.conv_dim,
                    self.num_heads,
                    self.num_heads * self.num_lambda_dims,
                ],
                dim=-1,
            )

            # 2. Convolution sequence transformation
            xBC = causal_conv1d_update(
                xBC,
                cache_params.conv_states[self.layer_idx],
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

            x, B, C = torch.split(
                xBC,
                [
                    self.intermediate_size,
                    groups_time_state_size,
                    groups_time_state_size,
                ],
                dim=-1,
            )

            # 3. SSM transformation
            A = -torch.exp(self.A_log.float())  # (nheads,)
            B = rearrange(
                B,
                "b (g n) -> b g n",
                b=batch_size,
                g=self.n_groups,
                n=self.ssm_state_size,
            )
            C = rearrange(
                C,
                "b (g n) -> b g n",
                b=batch_size,
                g=self.n_groups,
                n=self.ssm_state_size,
            )
            x_reshaped = rearrange(
                x,
                "b (h p) -> b h p",
                b=batch_size,
                h=self.num_heads,
                p=self.head_dim,
            )
            dl_reshaped = rearrange(
                dl,
                "b (h ell) -> b h ell",
                b=batch_size,
                h=self.num_heads,
                ell=self.num_lambda_dims,
            )
            y, hssm_state = hmamba_chunk_scan_combined(
                x_reshaped,
                dt=dt,
                A=A,
                B=B,
                C=C,
                dl=dl_reshaped,
                L=self.L,
                D=self.D,
                z=None,
                dt_bias=self.dt_bias,
                dt_softplus=True,
                initial_states=cache_params.hssm_states[self.layer_idx],
                return_final_states=True,
            )
            cache_params.update_hssm_state(
                layer_idx=self.layer_idx,
                new_hssm_state=hssm_state,
            )
            y = rearrange(
                y,
                "b h p -> b (h p)",
                b=batch_size,
                h=self.num_heads,
                p=self.head_dim,
            )
            y = self.norm(y, gate)

            # 4. Final linear projection
            out = self.out_proj(y)[:, None, ...]

        # Fused calculations or step by step if no initialized cache is found
        else:
            A = -torch.exp(
                self.A_log.float(),
            )  # (num_heads) or (intermediate_size, state_size)
            dt_limit_kwargs = (
                {}
                if self.time_step_limit == (0.0, float("inf"))
                else {"dt_limit": self.time_step_limit}
            )

            # 2-4. Fused kernel for conv1d, SSM, and the final projection
            if self.training and cache_params is None:
                out = torch.utils.checkpoint.checkpoint(
                    hmamba_split_conv1d_scan_combined,
                    use_reentrant=False,
                    # function arguments
                    zxbcdtdl=projected_states,
                    conv1d_weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    conv1d_bias=self.conv1d.bias,
                    dt_bias=self.dt_bias,
                    A=A,
                    L=self.L,
                    D=self.D,
                    chunk_size=self.chunk_size,
                    seq_idx=None,  # was seq_idx
                    activation=self.activation,
                    rmsnorm_weight=self.norm.weight,
                    rmsnorm_eps=self.norm.eps,
                    outproj_weight=self.out_proj.weight,
                    outproj_bias=self.out_proj.bias,
                    headdim=self.head_dim,
                    ngroups=self.n_groups,
                    norm_before_gate=False,
                    return_final_states=False,
                    **dt_limit_kwargs,
                )

            else:
                gate, xBC, dt, dl = torch.split(
                    projected_states,
                    [
                        self.intermediate_size,
                        self.conv_dim,
                        self.num_heads,
                        self.num_heads * self.num_lambda_dims,
                    ],
                    dim=-1,
                )

                # 2. Convolution sequence transformation
                # Init cache
                if cache_params is not None:
                    xBC_t = rearrange(xBC, "b l d -> b d l")
                    conv_states = torch.nn.functional.pad(
                        xBC_t,
                        (cache_params.conv_kernel_size - xBC_t.shape[-1], 0),
                    )
                    cache_params.update_conv_state(
                        layer_idx=self.layer_idx,
                        new_conv_state=conv_states,
                        cache_init=True,
                    )

                xBC = causal_conv1d_fn(
                    x=xBC.transpose(1, 2),
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                ).transpose(1, 2)

                xBC = apply_mask_to_padding_states(
                    hidden_states=xBC,
                    attention_mask=attention_mask,
                )

                x, B, C = torch.split(
                    xBC,
                    [
                        self.intermediate_size,
                        groups_time_state_size,
                        groups_time_state_size,
                    ],
                    dim=-1,
                )

                # 3. SSM transformation
                y, hssm_state = hmamba_chunk_scan_combined(
                    rearrange(
                        x,
                        "b l (h p) -> b l h p",
                        b=batch_size,
                        l=seq_len,
                        p=self.head_dim,
                    ),
                    dt=dt,
                    A=A,
                    B=rearrange(
                        B,
                        "b l (g n) -> b l g n",
                        b=batch_size,
                        l=seq_len,
                        g=self.n_groups,
                    ),
                    C=rearrange(
                        C,
                        "b l (g n) -> b l g n",
                        b=batch_size,
                        l=seq_len,
                        g=self.n_groups,
                    ),
                    dl=rearrange(
                        dl,
                        "b l (h ell) -> b l h ell",
                        b=batch_size,
                        h=self.num_heads,
                        ell=self.num_lambda_dims,
                    ),
                    L=self.L,
                    chunk_size=self.chunk_size,
                    D=self.D,
                    z=None,
                    seq_idx=None,
                    return_final_states=True,
                    dt_bias=self.dt_bias,
                    dt_softplus=True,
                    **dt_limit_kwargs,
                )

                # Init cache
                if hssm_state is not None and cache_params is not None:
                    cache_params.update_hssm_state(
                        layer_idx=self.layer_idx,
                        new_hssm_state=hssm_state,
                    )

                y = rearrange(
                    y,
                    "b l h p -> b l (h p)",
                    b=batch_size,
                    l=seq_len,
                    h=self.num_heads,
                    p=self.head_dim,
                )
                # Multiply "gate" branch and apply extra normalization layer
                y = self.norm(y, gate)

                # 4. Final linear projection
                out = self.out_proj(y)

        return out

    def forward(
        self,
        hidden_states,
        cache_params: Optional["LogLinearMamba2Cache"] = None,
        cache_position: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ):
        if "cuda" in self.in_proj.weight.device.type:
            return self.cuda_kernels_forward(
                hidden_states=hidden_states,
                cache_params=cache_params,
                cache_position=cache_position,
                attention_mask=attention_mask,
            )
        raise NotImplementedError

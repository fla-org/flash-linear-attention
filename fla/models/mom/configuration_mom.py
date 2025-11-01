

from transformers.configuration_utils import PretrainedConfig


class MomConfig(PretrainedConfig):
    model_type = 'mom'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        attn_mode: str = "chunk",
        hidden_size: int = 2048,
        conv_size: int = 4,
        num_heads: int = 4,
        head_dim: int = 256,
        expand_v: float = 1.,
        use_output_gate: bool = True,
        use_short_conv: bool = True,
        max_position_embeddings: int = 2048,
        hidden_ratio: int | None = 4,
        intermediate_size: int | None = None,
        hidden_act: str = "swish",
        num_hidden_layers: int = 24,
        norm_eps: float = 1e-6,
        attn: dict | None = None,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        initializer_range: float = 0.02,
        num_memories: int = 4,
        topk: int = 2,
        capacity: float = 1.0,
        use_layer_wise_balance: bool = True,
        aux_loss_scale: float = 0.01,
        shared_mem: bool = True,
        single_kv_proj: bool = False,
        mom_backend: str = 'gated_deltanet',
        fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        fuse_cross_entropy: bool = True,
        vocab_size: int = 32000,
        **kwargs,
    ):
        self.attn_mode = attn_mode
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.expand_v = expand_v
        self.conv_size = conv_size
        self.use_output_gate = use_output_gate
        self.use_short_conv = use_short_conv
        self.max_position_embeddings = max_position_embeddings

        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_hidden_layers = num_hidden_layers
        self.norm_eps = norm_eps
        self.attn = attn
        self.use_cache = use_cache
        self.initializer_range = initializer_range

        self.num_memories = num_memories
        self.topk = topk
        self.capacity = capacity
        self.use_layer_wise_balance = use_layer_wise_balance
        self.aux_loss_scale = aux_loss_scale
        self.shared_mem = shared_mem
        self.single_kv_proj = single_kv_proj
        self.mom_backend = mom_backend

        self.fuse_norm = fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.vocab_size = vocab_size

        if self.mom_backend not in ['gated_deltanet']:
            raise NotImplementedError(f"The MoM backend {mom_backend} is not currently supported.")

        if attn is not None:
            if not isinstance(attn, dict):
                raise ValueError("attn must be a dictionary")
            if 'layers' not in attn:
                raise ValueError("Layer indices must be provided to initialize hybrid attention layers")
            if 'num_heads' not in attn:
                raise ValueError("Number of heads must be provided to initialize hybrid attention layers")
            attn['num_kv_heads'] = attn.get('num_kv_heads', attn['num_heads'])
            attn['window_size'] = attn.get('window_size', None)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

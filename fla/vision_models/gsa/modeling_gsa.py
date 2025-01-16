import collections.abc
import math
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from typing import Optional, Set, Tuple, Union, List, Dict, Unpack
from transformers.utils import logging
from fla.layers.attn import Attention
from transformers.modeling_outputs import ImageClassifierOutput
from transformers.modeling_utils import PreTrainedModel
from .configuration_gsa import GSAVisionConfig
from fla.layers.gsa import GatedSlotAttention
from fla.models.utils import Cache
from ..utils import ImageEmbeddings, Pooler, prepare_hidden_states_for_cross_scan, prepare_hidden_states_for_cross_merge

logger = logging.get_logger(__name__)

class GSAMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            nn.Linear(config.hidden_size, config.mlp_dim),
            nn.GELU(),
            nn.Linear(config.mlp_dim, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob)
        )

    def forward(self, x):
        return self.net(x)

class GSABlock(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        
        if not config.norm_first:
            self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        if config.attn is not None and layer_idx in config.attn['layers']:
            self.attn = Attention(
                hidden_size=config.hidden_size,
                num_heads=config.attn['num_heads'],
                num_kv_heads=config.attn['num_kv_heads'],
                window_size=config.attn['window_size'],
                max_position_embeddings=config.max_position_embeddings,
                layer_idx=layer_idx
            )
        else:
            self.attn = GatedSlotAttention(
                hidden_size=config.hidden_size,
                expand_k=config.expand_k,
                expand_v=config.expand_v,
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                num_slots=config.num_slots,
                use_short_conv=config.use_short_conv,
                conv_size=config.conv_size,
                feature_map=config.feature_map,
                use_output_gate=config.use_output_gate,
                use_norm=config.use_norm,
                gate_fn=config.hidden_act,
                gate_logit_normalizer=config.gate_logit_normalizer,
                elementwise_affine=config.elementwise_affine,
                norm_first=config.norm_first,
                norm_eps=config.norm_eps,
                fuse_norm=config.fuse_norm,
                layer_idx=layer_idx
            )
            
        if not config.norm_first:
            self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            
        self.mlp = GSAMLP(config)

        self.scan_type = config.scan_type

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs: Unpack[Dict]
    ) -> Union[Tuple[torch.Tensor, Optional[torch.Tensor]], Tuple[torch.Tensor]]:
        residual = hidden_states

        # Pre-normalization if enabled
        if hasattr(self, 'ln_1'):
            hidden_states = self.ln_1(hidden_states)

        # Apply attention
        
        hidden_states = prepare_hidden_states_for_cross_scan(hidden_states, self.scan_type)
        
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs
        )
        
        hidden_states = prepare_hidden_states_for_cross_merge(hidden_states, self.scan_type)

        # First residual connection
        hidden_states = residual + hidden_states
        residual = hidden_states

        # Pre-normalization for MLP if enabled 
        if hasattr(self, 'ln_2'):
            hidden_states = self.ln_2(hidden_states)

        # MLP
        hidden_states = self.mlp(hidden_states)
        
        # Second residual connection
        hidden_states = residual + hidden_states

        outputs = (hidden_states, attentions, past_key_values)

        return outputs

class GSAVisionPreTrainedModel(PreTrainedModel):
    # this part of the code is adapted from huggingface/transformers vit implementation
    config_class = GSAVisionConfig
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, ImageEmbeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.position_embeddings.dtype)

class GSAForImageClassification(GSAVisionPreTrainedModel):
    config_class = GSAVisionConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_classes
        
        self.embeddings = ImageEmbeddings(config)
        self.blocks = nn.ModuleList([
            GSABlock(config, layer_idx) 
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = Pooler(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)
        self.interpolate_pos_encoding = config.interpolate_pos_encoding
        self.init_weights()
        
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Unpack[Dict]
    ) -> Union[Tuple, ImageClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        fuse_linear_and_cross_entropy = self.config.fuse_cross_entropy and self.training
        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=self.interpolate_pos_encoding)
        
        for block in self.blocks:
            hidden_states, attentions, past_key_values = block(
                hidden_states,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                **kwargs
            )
            
        hidden_states = self.norm(hidden_states)
        pooled_output = self.pooler(hidden_states)
        
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + (hidden_states,)
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
        )

import json
import os
import re
import warnings
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers.file_utils import to_py_obj
from transformers.tokenization_utils_base import (
    AddedToken,
    BatchEncoding,
    EncodedInput,
    PreTokenizedInput,
    TextInput,
    TruncationStrategy,
)
from transformers.utils import PaddingStrategy, TensorType, logging
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from fla.models.rodimus.chat_format import Chat


logger = logging.get_logger(__name__)


class RodimusTokenizer(PreTrainedTokenizerFast):
    slow_tokenizer_class = None
    padding_side = "left"
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = None

    SPECIAL_TOKENS_ATTRIBUTES = [
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
        "gmask_token",
        "additional_special_tokens",
    ]

    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        tokenizer_file=None,
        clean_up_tokenization_spaces=False,
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        cls_token="[CLS]",
        gmask_token="[gMASK]",
        add_bos_token=False,
        add_eos_token=False,
        **kwargs,
    ):
        self._gmask_token = (
            AddedToken(gmask_token, lstrip=False,
                       rstrip=False, normalized=False)
            if isinstance(gmask_token, str)
            else gmask_token
        )

        self._sop_token = (
            AddedToken(bos_token, lstrip=False, rstrip=False, normalized=False)
            if isinstance(bos_token, str)
            else bos_token
        )

        self._eop_token = (
            AddedToken(eos_token, lstrip=False, rstrip=False, normalized=False)
            if isinstance(eos_token, str)
            else eos_token
        )
        if 'pad_token' not in kwargs:
            kwargs['pad_token'] = eos_token

        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            tokenizer_file=tokenizer_file,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            bos_token=bos_token,
            eos_token=eos_token,
            cls_token=cls_token,
            gmask_token=gmask_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            **kwargs,
        )

        self.check_special_tokens()

    def check_special_tokens(self):
        '''
        eos_token, cls_token, mask_token
        special tokens should init, check special token is not None
        '''
        for name, special_token in zip(
            ['eos', 'bos', 'cls', 'gmask'],
            [self.eos_token, self.bos_token, self.cls_token, self.gmask_token],
        ):
            assert special_token is not None, f'should init special token [{name}] in tokenizer_config.json'

    @property
    def gmask_token(self) -> Optional[str]:
        if self._gmask_token is None:
            if self.verbose:
                logger.error("Using gmask_token, but it is not set yet.")
            return None
        return str(self._gmask_token)

    @gmask_token.setter
    def gmask_token(self, value):
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError(
                "Cannot set a non-string value as the gmask token")
        self._gmask_token = value

    @property
    def gmask_token_id(self) -> Optional[int]:
        if self._gmask_token is None:
            return None
        return self.convert_tokens_to_ids(self.gmask_token)

    @property
    def sop_token(self) -> Optional[str]:
        if self._sop_token is None:
            if self.verbose:
                logger.error("Using sop_token, but it is not set yet.")
            return None
        return str(self._sop_token)

    @sop_token.setter
    def sop_token(self, value):
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError("Cannot set a non-string value as the sop token")
        self._sop_token = value

    @property
    def sop_token_id(self) -> Optional[int]:
        if self._sop_token is None:
            return None
        return self.convert_tokens_to_ids(self.sop_token)

    @property
    def eop_token(self) -> Optional[str]:
        if self._eop_token is None:
            if self.verbose:
                logger.error("Using eop_token, but it is not set yet.")
            return None
        return str(self._eop_token)

    @eop_token.setter
    def eop_token(self, value):
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError("Cannot set a non-string value as the eop token")
        self._eop_token = value

    @property
    def eop_token_id(self) -> Optional[int]:
        if self._eop_token is None:
            return None
        return self.convert_tokens_to_ids(self.eop_token)

    @property
    def vocab_size(self):
        return len(self.get_vocab())

    def apply_chat_template(
        self,
        conversation: Union[List[Dict[str, str]], List[List[Dict[str, str]]]],
        system: str = None,
        tokenize=False,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_dict: bool = False,
        **kwargs,
    ):
        chat_format = kwargs.get('chat_format', 'rodimus_chat')

        is_batched = False

        if isinstance(conversation, List) and (
            isinstance(conversation[0], (list, tuple)
                       ) or "messages" in conversation[0]
        ):
            conversations = conversation
            is_batched = True

        if not is_batched:
            conversations = [conversation]

        rendered = []
        for chat in conversations:
            if "messages" not in chat:
                # Indicates it's a Conversation object
                chat = {'messages': chat}
            if system:
                chat['system_message'] = system
            rendered_chat = Chat.from_json(chat, name=chat_format).prompt_str
            rendered.append(rendered_chat)

        if not is_batched:
            rendered = rendered[0]

        if tokenize:
            out = self(
                rendered,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                add_special_tokens=False,
                return_tensors=return_tensors,
            )
            if return_dict:
                return out
            else:
                return out["input_ids"]
        else:
            return rendered
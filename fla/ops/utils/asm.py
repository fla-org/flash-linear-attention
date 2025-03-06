# -*- coding: utf-8 -*-

from fla.utils import device_platform


def fp32_to_tf32_asm() -> str:
    """
    Get the assembly code for converting FP32 to TF32.
    """
    ASM_DICT = {
        'nvidia': 'cvt.rna.tf32.f32 $0, $1;'
    }
    if device_platform in ASM_DICT:
        return ASM_DICT[device_platform]
    else:
        # return empty string if the device is not supported
        return ""

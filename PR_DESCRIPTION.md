# Pull Request: Add Quasar Attention and Standalone Model Implementation

## Summary
This PR introduces **Quasar Attention**, a highly optimized linear attention variant derived from Kimi Delta Attention (KDA) but featuring significant architectural optimizations and kernel refinements. Quasar achieves superior throughput and memory efficiency, particularly at long context lengths.

This PR includes:
1.  **Quasar Attention Triton Kernels**: Fused chunk-wise forward and backward kernels in `fla/ops/quasar`.
2.  **QuasarAttention Layer**: A standalone attention layer in `fla/layers/quasar.py`.
3.  **Quasar Model**: A complete HuggingFace-compatible model implementation in `fla/models/quasar`, including `QuasarConfig`, `QuasarModel`, and `QuasarForCausalLM`.
4.  **Library Integration**: Full registration of Quasar components in the `fla` library root interfaces.

## Benchmarks
Quasar demonstrates superior hardware efficiency compared to baseline linear attention architectures.

### High-Throughput Performance
**Setup**: 8x NVIDIA B200, 2B Model, 64k Context Length

| Architecture | Throughput (Tokens/sec) |
| :--- | :--- |
| **Quasar** | **478,559** |
| Kimi Delta Attention (KDA) | 456,163 |
| Gated Delta Attention | 447,784 |

### Scaling and Memory Efficiency
**Setup**: Single NVIDIA B200, 1B Model

| Context Length | Quasar Throughput | KDA Throughput | Speedup |
| :--- | :--- | :--- | :--- |
| 16k | 123,259 tok/s | 105,052 tok/s | **+17.3%** |
| 32k | 146,828 tok/s | 110,225 tok/s | **+33.2%** |

## References
- **Quasar Attention Repository**: [https://github.com/SILX-LABS/quasar-attention](https://github.com/SILX-LABS/quasar-attention)
- **Official Release**: Quasar Attention significantly improves upon KDA by optimizing the gating mechanism and kernel fusion for modern GPU architectures like Blackwell (B200).

## Implementation Details
- **Branding**: All components follow the `quasar` nomenclature to prevent symbol collisions with upstream KDA implementations.
- **Independence**: The Quasar module is self-contained, including its own recomputed kernels and configuration classes.
- **Compatibility**: Supports both standalone Quasar models and hybrid attention configurations within the FLA framework.

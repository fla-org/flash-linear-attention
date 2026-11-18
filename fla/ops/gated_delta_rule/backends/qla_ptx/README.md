# B300 PTX forward backend

This backend provides exported `sm_103a` PTX for the prefill specializations in
`workloads.json`. It accepts BF16 queries, keys, and values; FP32 precomputed log
decays and beta; and optional FP32 K-first initial states. Output is BF16 and the
optional final state is FP32. A missing scale uses the public operator's default.

Selection requires a CUDA 13 or newer PyTorch build, a compatible NVIDIA driver,
TVM FFI with `cpp.load_inline(embed_cubin=...)`, a CUDA toolkit, and a C++ compiler.
No kernel CUDA source is compiled at runtime. The CUDA driver
assembles PTX on first load, and TVM FFI compiles the host launchers. Initialization
must finish before CUDA graph capture.

Calls needing gradients, fused gate or beta activation, QK normalization,
V-first states, context parallelism, intermediate states, decoding, other
architectures, other dtypes, or unexported shapes continue to the existing
backends. Packed specializations additionally require `cu_seqlens_cpu` matching
the exported boundaries, so selection does not synchronize the GPU to inspect
sequence metadata. The operator expects consistent CPU and device offsets.

The packaged PTX and host launchers form one implementation. `manifest.json`
records each specialization, launch argument ordering, and PTX digest. Module
selection depends on the factory parameters and specialization values; retries
do not depend on prior load attempts.

See [RESULTS.md](RESULTS.md) for the same-GPU FlashQLA comparison and reproduction commands.

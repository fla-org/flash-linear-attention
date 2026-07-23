#pragma once

// TileLang 0.1.9 assumes CUDA 12.6 exposes the FP8 e8m0 CUDA C++ symbols.
// The CUDA 12.6 toolkit paired with this verifier does not define them, yet
// TileLang includes its FP8 helper header even for bf16-only kernels. These
// declarations let non-e8m0 kernels compile; the stubs must never be used for
// real e8m0 computation.

#include <cuda_runtime.h>
#include <cuda_fp8.h>

#if defined(__CUDACC_VER_MAJOR__) && defined(__CUDACC_VER_MINOR__) && \
    (__CUDACC_VER_MAJOR__ == 12) && (__CUDACC_VER_MINOR__ < 8)

struct __CUDA_ALIGN__(1) __nv_fp8_e8m0 {
  __nv_fp8_storage_t __x;

  __host__ __device__ __nv_fp8_e8m0() = default;
  __host__ __device__ explicit __nv_fp8_e8m0(__nv_fp8_storage_t x) : __x(x) {}
};

__host__ __device__ __forceinline__ __nv_bfloat16_raw
__nv_cvt_e8m0_to_bf16raw(const __nv_fp8_storage_t) {
  __nv_bfloat16_raw out;
  out.x = 0U;
  return out;
}

__host__ __device__ __forceinline__ __nv_bfloat162_raw
__nv_cvt_e8m0x2_to_bf162raw(const __nv_fp8x2_storage_t) {
  __nv_bfloat162_raw out;
  out.x = 0U;
  out.y = 0U;
  return out;
}

__host__ __device__ __forceinline__ __nv_fp8_storage_t
__nv_cvt_bfloat16raw_to_e8m0(
    const __nv_bfloat16_raw,
    const __nv_saturation_t,
    const cudaRoundMode) {
  return 0U;
}

__host__ __device__ __forceinline__ __nv_fp8x2_storage_t
__nv_cvt_bfloat162raw_to_e8m0x2(
    const __nv_bfloat162_raw,
    const __nv_saturation_t,
    const cudaRoundMode) {
  return 0U;
}

__host__ __device__ __forceinline__ __nv_fp8_storage_t
__nv_cvt_float_to_e8m0(const float, const __nv_saturation_t, const cudaRoundMode) {
  return 0U;
}

__host__ __device__ __forceinline__ __nv_fp8x2_storage_t
__nv_cvt_float2_to_e8m0x2(const float2, const __nv_saturation_t, const cudaRoundMode) {
  return 0U;
}

__host__ __device__ __forceinline__ __nv_fp8_storage_t
__nv_cvt_double_to_e8m0(const double, const __nv_saturation_t, const cudaRoundMode) {
  return 0U;
}

__host__ __device__ __forceinline__ __nv_fp8x2_storage_t
__nv_cvt_double2_to_e8m0x2(const double2, const __nv_saturation_t, const cudaRoundMode) {
  return 0U;
}

#endif

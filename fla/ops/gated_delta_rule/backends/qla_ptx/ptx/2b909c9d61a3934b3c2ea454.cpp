// Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#include <cuda.h>
#include <cuda_runtime.h>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/cubin_launcher.h>
#include <tvm/ffi/function.h>

#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

TVM_FFI_EMBED_CUBIN(prepare_bf16_80b8e6ac59);

namespace qla_ptx_host {

using tvm::ffi::TensorView;

inline void CheckCudaTensor(const TensorView& t, const char* name) {
  TVM_FFI_CHECK(t.device().device_type == kDLCUDA, ValueError)
      << name << " must be a CUDA tensor, got device_type=" << (int)t.device().device_type;
}

inline void CheckSameCudaDevice(
    const TensorView& t,
    const TensorView& reference,
    const char* name,
    const char* reference_name) {
  TVM_FFI_CHECK(t.device().device_id == reference.device().device_id, ValueError)
      << name << " must be on the same CUDA device as " << reference_name
      << ": got cuda:" << t.device().device_id
      << " versus cuda:" << reference.device().device_id;
}

inline void CheckContiguous(const TensorView& t, const char* name) {
  TVM_FFI_CHECK(t.IsContiguous(), ValueError) << name << " must be contiguous";
}

inline void CheckDtype(const TensorView& t, const char* name, int code, int bits, int lanes) {
  DLDataType d = t.dtype();
  TVM_FFI_CHECK((int)d.code == code && (int)d.bits == bits && (int)d.lanes == lanes, TypeError)
      << name << " dtype mismatch: expected DLDataType(code=" << code << ", bits=" << bits
      << ", lanes=" << lanes << "), got (code=" << (int)d.code << ", bits=" << (int)d.bits
      << ", lanes=" << (int)d.lanes << ")";
}

// A logical axis.outer(trailing) folds every source dim above the trailing
// dimensions. Shape products are independent of physical strides, so verify
// the leading dimensions form one dense row-major chain instead of inventing
// a "folded stride". The descriptor reads its exact adjacent physical step
// separately through stride[-(trailing + 1)].
inline void CheckDenseLeadingFold(const TensorView& t, int trailing, const char* name) {
  TVM_FFI_CHECK(trailing > 0 && t.ndim() >= trailing, ValueError)
      << name << " cannot fold leading dimensions above " << trailing
      << " trailing dims from ndim=" << t.ndim();
  int outer_last = t.ndim() - trailing - 1;
  if (outer_last <= 0) {
    return;
  }
  int64_t step = t.stride(outer_last);
  TVM_FFI_CHECK(step > 0, ValueError)
      << name << " physical strides must be positive";
  int64_t expected = step;
  for (int axis = outer_last - 1; axis >= 0; --axis) {
    expected *= t.size(axis + 1);
    if (t.size(axis) > 1) {
      TVM_FFI_CHECK(t.stride(axis) == expected, ValueError)
          << name << " leading dims are not physically foldable above " << trailing
          << " trailing dims: stride(" << axis << ")=" << t.stride(axis)
          << ", expected " << expected;
    }
  }
}

void Run(TensorView arg_q, TensorView arg_k, TensorView arg_v, TensorView arg_g, TensorView arg_beta, TensorView arg_cu, TensorView arg_zero, TensorView arg_prepared, TensorView arg_updates, TensorView arg_mixing, TensorView arg_decays, double arg_scale, int64_t grid_x, int64_t grid_y, int64_t grid_z) {
  CheckCudaTensor(arg_q, "q");
  CheckDtype(arg_q, "q", 4, 16, 1);
  CheckContiguous(arg_q, "q");
  CheckCudaTensor(arg_k, "k");
  CheckDtype(arg_k, "k", 4, 16, 1);
  CheckContiguous(arg_k, "k");
  CheckCudaTensor(arg_v, "v");
  CheckDtype(arg_v, "v", 4, 16, 1);
  CheckContiguous(arg_v, "v");
  CheckCudaTensor(arg_g, "g");
  CheckDtype(arg_g, "g", 2, 32, 1);
  CheckContiguous(arg_g, "g");
  CheckCudaTensor(arg_beta, "beta");
  CheckDtype(arg_beta, "beta", 2, 32, 1);
  CheckContiguous(arg_beta, "beta");
  CheckCudaTensor(arg_cu, "cu");
  CheckDtype(arg_cu, "cu", 0, 64, 1);
  CheckContiguous(arg_cu, "cu");
  CheckCudaTensor(arg_zero, "zero");
  CheckDtype(arg_zero, "zero", 2, 32, 1);
  CheckContiguous(arg_zero, "zero");
  CheckCudaTensor(arg_prepared, "prepared");
  CheckDtype(arg_prepared, "prepared", 4, 16, 1);
  CheckContiguous(arg_prepared, "prepared");
  CheckCudaTensor(arg_updates, "updates");
  CheckDtype(arg_updates, "updates", 4, 16, 1);
  CheckContiguous(arg_updates, "updates");
  CheckCudaTensor(arg_mixing, "mixing");
  CheckDtype(arg_mixing, "mixing", 4, 16, 1);
  CheckContiguous(arg_mixing, "mixing");
  CheckCudaTensor(arg_decays, "decays");
  CheckDtype(arg_decays, "decays", 2, 32, 1);
  CheckContiguous(arg_decays, "decays");
  CheckSameCudaDevice(arg_k, arg_q, "k", "q");
  CheckSameCudaDevice(arg_v, arg_q, "v", "q");
  CheckSameCudaDevice(arg_g, arg_q, "g", "q");
  CheckSameCudaDevice(arg_beta, arg_q, "beta", "q");
  CheckSameCudaDevice(arg_cu, arg_q, "cu", "q");
  CheckSameCudaDevice(arg_zero, arg_q, "zero", "q");
  CheckSameCudaDevice(arg_prepared, arg_q, "prepared", "q");
  CheckSameCudaDevice(arg_updates, arg_q, "updates", "q");
  CheckSameCudaDevice(arg_mixing, arg_q, "mixing", "q");
  CheckSameCudaDevice(arg_decays, arg_q, "decays", "q");
  TVM_FFI_CHECK(grid_x > 0 && grid_y > 0 && grid_z > 0, ValueError)
      << "launch grid dimensions must be positive, got (" << grid_x << ", " << grid_y
      << ", " << grid_z << ")";

  DLDevice dev = arg_q.device();
  cudaStream_t stream = (cudaStream_t)TVMFFIEnvGetStream(dev.device_type, dev.device_id);
  void* p_q = arg_q.data_ptr();
  void* p_k = arg_k.data_ptr();
  void* p_v = arg_v.data_ptr();
  void* p_g = arg_g.data_ptr();
  void* p_beta = arg_beta.data_ptr();
  void* p_cu = arg_cu.data_ptr();
  void* p_zero = arg_zero.data_ptr();
  void* p_prepared = arg_prepared.data_ptr();
  void* p_updates = arg_updates.data_ptr();
  void* p_mixing = arg_mixing.data_ptr();
  void* p_decays = arg_decays.data_ptr();
  float v_scale = (float)arg_scale;
  void* kargs[] = {&p_q, &p_k, &p_v, &p_g, &p_beta, &p_cu, &p_zero, &p_prepared, &p_updates, &p_mixing, &p_decays, &v_scale};

  static auto kernel = TVM_FFI_EMBED_CUBIN_GET_KERNEL(prepare_bf16_80b8e6ac59, "kernel_prepare_bf16");
  tvm::ffi::dim3 grid((uint32_t)grid_x, (uint32_t)grid_y, (uint32_t)grid_z);
  tvm::ffi::dim3 block(64u, 1u, 1u);

  TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(kernel.Launch(kargs, grid, block, stream, 14464u));
}

}  // namespace qla_ptx_host

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, qla_ptx_host::Run);

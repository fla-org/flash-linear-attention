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

TVM_FFI_EMBED_CUBIN(prefix_context_59ca1b1789);

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

void Run(TensorView arg_initial, TensorView arg_cu, TensorView arg_maps, TensorView arg_map_a, TensorView arg_entries, int64_t grid_x, int64_t grid_y, int64_t grid_z) {
  CheckCudaTensor(arg_initial, "initial");
  CheckDtype(arg_initial, "initial", 2, 32, 1);
  CheckContiguous(arg_initial, "initial");
  CheckCudaTensor(arg_cu, "cu");
  CheckDtype(arg_cu, "cu", 0, 64, 1);
  CheckContiguous(arg_cu, "cu");
  CheckCudaTensor(arg_maps, "maps");
  CheckDtype(arg_maps, "maps", 2, 32, 1);
  CheckContiguous(arg_maps, "maps");
  CheckCudaTensor(arg_map_a, "map_a");
  CheckDtype(arg_map_a, "map_a", 4, 16, 1);
  CheckContiguous(arg_map_a, "map_a");
  CheckCudaTensor(arg_entries, "entries");
  CheckDtype(arg_entries, "entries", 2, 32, 1);
  CheckContiguous(arg_entries, "entries");
  CheckSameCudaDevice(arg_cu, arg_initial, "cu", "initial");
  CheckSameCudaDevice(arg_maps, arg_initial, "maps", "initial");
  CheckSameCudaDevice(arg_map_a, arg_initial, "map_a", "initial");
  CheckSameCudaDevice(arg_entries, arg_initial, "entries", "initial");
  TVM_FFI_CHECK(grid_x > 0 && grid_y > 0 && grid_z > 0, ValueError)
      << "launch grid dimensions must be positive, got (" << grid_x << ", " << grid_y
      << ", " << grid_z << ")";

  DLDevice dev = arg_initial.device();
  cudaStream_t stream = (cudaStream_t)TVMFFIEnvGetStream(dev.device_type, dev.device_id);
  void* p_initial = arg_initial.data_ptr();
  void* p_cu = arg_cu.data_ptr();
  void* p_maps = arg_maps.data_ptr();
  void* p_map_a = arg_map_a.data_ptr();
  void* p_entries = arg_entries.data_ptr();
  void* kargs[] = {&p_initial, &p_cu, &p_maps, &p_map_a, &p_entries};

  static auto kernel = TVM_FFI_EMBED_CUBIN_GET_KERNEL(prefix_context_59ca1b1789, "kernel_prefix_context");
  tvm::ffi::dim3 grid((uint32_t)grid_x, (uint32_t)grid_y, (uint32_t)grid_z);
  tvm::ffi::dim3 block(32u, 1u, 1u);

  TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(kernel.Launch(kargs, grid, block, stream, 39168u));
}

}  // namespace qla_ptx_host

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, qla_ptx_host::Run);

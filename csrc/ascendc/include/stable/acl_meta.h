// Shared descriptor/metadata layer for the Stable-ABI launchers.
//
// Everything here talks to CANN through the dlopen'd acl* symbols and to torch
// exclusively through the aoti_torch_* C shims.  No ATen/c10 headers, no
// pybind11; the vendored headers pin the Stable ABI used by this launcher.
#pragma once

#include <torch/csrc/stable/stableivalue_conversions.h>
#include <torch/csrc/stable/tensor.h>

#include "stable/runtime.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace fla_ascendc {
namespace stable {

typedef struct aclTensor aclTensor;
typedef struct aclIntArray aclIntArray;
typedef struct aclOpExecutor aclOpExecutor;

constexpr int32_t kAclFormatNd = 2;
constexpr int32_t kAclFormatNchw = 0;
constexpr int32_t kAclFormatNcdhw = 30;
constexpr int32_t kAclFormatNcl = 47;
// "Not specified": the view infers the format from the rank, which is what the
// ctypes reference does when torch_npu does not report one.
constexpr int32_t kAclFormatAuto = -1;

// torch ScalarType -> ACL data type (same mapping the ctypes reference uses).
inline int32_t acl_dtype(int32_t scalar_type) {
  switch (scalar_type) {
    case 6:
      return 0;  // kFloat  -> ACL_FLOAT
    case 5:
      return 1;  // kHalf   -> ACL_FLOAT16
    case 1:
      return 2;  // kChar   -> ACL_INT8
    case 3:
      return 3;  // kInt    -> ACL_INT32
    case 0:
      return 4;  // kByte   -> ACL_UINT8
    case 2:
      return 6;  // kShort  -> ACL_INT16
    case 4:
      return 9;  // kLong   -> ACL_INT64
    case 7:
      return 11;  // kDouble -> ACL_DOUBLE
    case 11:
      return 12;  // kBool   -> ACL_BOOL
    case 15:
      return 27;  // kBFloat16 -> ACL_BF16
    default:
      throw std::runtime_error(
          "fla_ascendc: unsupported tensor dtype id " +
          std::to_string(scalar_type));
  }
}

inline int64_t element_size(int32_t scalar_type) {
  switch (scalar_type) {
    case 0:
    case 1:
    case 11:
      return 1;
    case 2:
    case 5:
    case 15:
      return 2;
    case 3:
    case 6:
      return 4;
    case 4:
    case 7:
      return 8;
    default:
      throw std::runtime_error(
          "fla_ascendc: unsupported dtype id " +
          std::to_string(scalar_type));
  }
}

using AclCreateTensorFn = aclTensor* (*)(const int64_t*, uint64_t, int32_t,
                                         const int64_t*, int64_t, int32_t,
                                         const int64_t*, uint64_t, void*);
using AclDestroyTensorFn = int (*)(aclTensor*);
using LaunchFn = int (*)(void*, uint64_t, aclOpExecutor*, void*);
using AclCreateIntArrayFn = aclIntArray* (*)(const int64_t*, uint64_t);
using AclDestroyIntArrayFn = int (*)(aclIntArray*);

// The descriptor entry points are looked up once per process rather than once
// per descriptor.  Every tensor argument of every operator goes through them,
// so a decode step would otherwise repeat the same string build and hash map
// lookup a few hundred times per layer.
inline AclCreateTensorFn acl_create_tensor() {
  static const auto fn = reinterpret_cast<AclCreateTensorFn>(
      Runtime::instance().symbol("aclCreateTensor"));
  return fn;
}

inline AclDestroyTensorFn acl_destroy_tensor() {
  static const auto fn = reinterpret_cast<AclDestroyTensorFn>(
      Runtime::instance().symbol("aclDestroyTensor"));
  return fn;
}

inline AclCreateIntArrayFn acl_create_int_array() {
  static const auto fn = reinterpret_cast<AclCreateIntArrayFn>(
      Runtime::instance().symbol("aclCreateIntArray"));
  return fn;
}

inline AclDestroyIntArrayFn acl_destroy_int_array() {
  static const auto fn = reinterpret_cast<AclDestroyIntArrayFn>(
      Runtime::instance().symbol("aclDestroyIntArray"));
  return fn;
}

struct TensorMeta {
  bool defined = false;
  void* data = nullptr;
  int64_t ndim = 0;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  int64_t storage_offset = 0;
  int64_t storage_numel = 0;
  int32_t scalar_type = 0;
  int32_t device_type = 0;
  int32_t device_index = 0;
  bool contiguous = false;
};

// Fills `meta` from a raw handle.  Deliberately does not construct a
// torch::stable::Tensor: that constructor *steals* ownership, so a temporary
// would release a reference this layer does not own -- the boxed entry point
// already consumed the stack's reference to every input, and the output
// handles are still owned by whoever created them.
// (Stealing one of them a second time was the root cause of the first two
// handle-unboxing attempts.)
inline void fill_meta(AtenTensorHandle handle, TensorMeta* meta) {
  // A missing optional arrives as a null handle; short-circuiting keeps the
  // needed aoti_torch_* set inside what older libtorch builds export.
  if (handle == nullptr) {
    return;
  }
  meta->defined = true;
  void* tensor_data = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_data_ptr(handle, &tensor_data));
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_dim(handle, &meta->ndim));
  int64_t* sizes = nullptr;
  int64_t* strides = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_sizes(handle, &sizes));
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_strides(handle, &strides));
  meta->sizes.assign(sizes, sizes + meta->ndim);
  meta->strides.assign(strides, strides + meta->ndim);
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_storage_offset(handle, &meta->storage_offset));
  // Storage extent in elements, exactly like the ctypes path computes it from
  // `untyped_storage().nbytes() // element_size` (aoti_torch_get_storage_numel
  // is the *view* numel and is wrong for paged/offset states).
  int64_t storage_bytes = 0;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_storage_size(handle, &storage_bytes));
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype(handle, &meta->scalar_type));
  const int64_t item_size = element_size(meta->scalar_type);
  meta->storage_numel = storage_bytes / item_size;
  // aclCreateTensor wants the *storage* base address and takes storage_offset
  // separately; the shim's data_ptr already includes the offset.
  meta->data = static_cast<uint8_t*>(tensor_data) -
               meta->storage_offset * item_size;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_device_type(handle, &meta->device_type));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_device_index(handle, &meta->device_index));
  // Computed locally rather than via aoti_torch_is_contiguous (2.9-only).
  int64_t expected = 1;
  meta->contiguous = true;
  for (int64_t dim = meta->ndim - 1; dim >= 0; --dim) {
    if (meta->sizes[dim] != 1 && meta->strides[dim] != expected) {
      meta->contiguous = false;
      break;
    }
    expected *= meta->sizes[dim];
  }
}

inline TensorMeta meta_of_handle(AtenTensorHandle handle) {
  TensorMeta meta;
  fill_meta(handle, &meta);
  return meta;
}

inline TensorMeta meta_of(const torch::stable::Tensor& tensor) {
  return meta_of_handle(tensor.get());
}

// A dimension of an already-collected `TensorMeta`.
//
// The callee only ever sees the metadata, so a wrong rank used to report a bare
// "size_of dim out of range" and leave the caller guessing which argument had
// the wrong rank -- every output-shape rule reads one dimension per argument,
// optional ones included.  C++ cannot recover the text of the tensor
// expression inside the callee, which is the one thing `SIZE_OF` is a macro
// for: it stringizes the argument, so the message names the tensor the adapter
// actually wrote, and the location builtins are expanded at the call site:
//
//   fla_ascendc: size_of dim 1 out of range at
//       csrc/src/stable_causal_conv1d_bwd.cpp:74 (tensor weight_meta, ndim=1, shape=[1536])
//
// Both halves come from the macro's own expansion, so both stay right only as
// far as the line that wrote `SIZE_OF`: a helper that reads the dimension
// itself can report nothing better than its own parameter and its own line.  A
// helper therefore answers with an axis number instead, and the operator reads
// the dimension -- `SIZE_OF(q_meta, layout_math::token_axis(layout))` -- which
// also settles the case of one helper serving several tensors, where no
// literal name is right for every caller (see layout_math.h).  GCC/Clang
// expand the location builtins at the call site, which is why the macro body
// can use them directly; C++20's `std::source_location::current()` is the
// standard spelling of the same trick (this tree builds with -std=c++17).
inline const char* short_location(const char* file) {
  if (file == nullptr) {
    return "?";
  }
  // The build passes absolute paths; keep the part a reader can find in the
  // tree ("csrc/src/stable_<op>.cpp").
  const char* tail = nullptr;
  for (const char* cursor = file; *cursor != '\0'; ++cursor) {
    if (std::strncmp(cursor, "csrc/", 5) == 0) {
      tail = cursor;
    }
  }
  return tail == nullptr ? file : tail;
}

inline std::string shape_detail(const TensorMeta& meta) {
  if (!meta.defined) {
    // An optional input that was not passed has no rank at all, which is the
    // other way to reach this branch.
    return "is None / undefined";
  }
  std::string shape = "[";
  for (int64_t axis = 0; axis < meta.ndim; ++axis) {
    if (axis != 0) {
      shape += ", ";
    }
    shape += std::to_string(meta.sizes[static_cast<size_t>(axis)]);
  }
  shape += "]";
  return "ndim=" + std::to_string(meta.ndim) + ", shape=" + shape;
}

// Names the tensor: the adapter passes the expression it wrote, which is the
// only spelling that is right for every caller when a helper serves several.
// A null `tensor_expr` reports the shape alone.
inline std::string arg_detail(const TensorMeta& meta, const char* tensor_expr) {
  if (tensor_expr == nullptr) {
    return meta.defined ? " (" + shape_detail(meta) + ")"
                        : " (tensor is None / undefined)";
  }
  return " (tensor " + std::string(tensor_expr) +
         (meta.defined ? ", " : " ") + shape_detail(meta) + ")";
}

inline int64_t size_of_impl(const TensorMeta& meta, int64_t dim,
                            const char* file, int line,
                            const char* tensor_expr = nullptr) {
  if (dim < 0 || dim >= meta.ndim) {
    throw std::runtime_error(
        "fla_ascendc: size_of dim " + std::to_string(dim) +
        " out of range at " + short_location(file) + ":" +
        std::to_string(line) + arg_detail(meta, tensor_expr));
  }
  return meta.sizes[static_cast<size_t>(dim)];
}

// Name-less entry point, for a caller that already holds the location and has
// no expression to report.  It is also what the adapters' `using
// fla_ascendc::stable::size_of;` lines bring into scope; the `SIZE_OF` macro
// below calls `size_of_impl` directly because only a macro can carry the
// tensor's name.
inline int64_t size_of(const TensorMeta& meta, int64_t dim,
                       const char* file = __builtin_FILE(),
                       int line = __builtin_LINE()) {
  return size_of_impl(meta, dim, file, line, nullptr);
}

// Call-site form, used by the adapters.  Function-like macro, so an argument
// that contains a comma needs parentheses of its own.
#define SIZE_OF(meta, dim)                                              \
  ::fla_ascendc::stable::size_of_impl((meta), (dim),                 \
                                         __builtin_FILE(),              \
                                         __builtin_LINE(), #meta)

inline TensorMeta meta_optional_handle(std::optional<AtenTensorHandle> handle) {
  TensorMeta meta;
  if (handle.has_value()) {
    fill_meta(*handle, &meta);
  }
  return meta;
}

// RAII wrapper around aclCreateTensor / aclDestroyTensor.  Same semantics as the
// ctypes path: contiguous tensors describe storage with the logical shape,
// non-contiguous ones fall back to a flat storage extent.
// The aclnn tiling reads the descriptor's format, so it has to be the one the
// reference would pass.  Measured on Ascend950: `torch_npu.get_npu_format`
// reports NCHW for a 4-D tensor, and handing aclnnChunkGatedDeltaRuleFwd an ND
// descriptor for the same input made its tiling reject the call (161002) where
// the reference's NCHW descriptor was accepted.  The rank-to-format fallback
// below is the reference's own (`_runtime.acl_format`); the operators whose
// reference asks for ND pass `kAclFormatNd` explicitly (see `nd_tensor` in
// exec.h).
inline int32_t inferred_format(const TensorMeta& meta) {
  switch (meta.ndim) {
    case 4:
      return kAclFormatNchw;
    case 5:
      return kAclFormatNcdhw;
    default:
      // Measured on Ascend950: torch_npu reports ND for the 3-D gate/beta
      // tensors and NCHW for the 4-D ones, so 3-D stays ND.  (The reference's
      // NCL branch only applies when torch_npu is not loaded at all, which is
      // not a configuration we run in.)
      return kAclFormatNd;
  }
}

class AclTensorView {
 public:
  // `logical_storage` picks between the reference's two storage spellings:
  // plain `ctx.tensor(...)` describes a contiguous tensor with its *flat*
  // extent, and `storage_shape_override=` with the logical shape.  The
  // difference is invisible when the framework treats the input as a plain
  // tensor, but the Ascend950 tiling of the fused GDN forward reads it (an ND
  // descriptor for the 3-D gate made it reject the call), so each adapter uses
  // whichever spelling its reference uses.
  explicit AclTensorView(const TensorMeta& meta,
                         int32_t format = kAclFormatAuto,
                         bool logical_storage = false)
      : format_(format < 0 ? inferred_format(meta) : format) {
    if (!meta.defined) {
      return;
    }
    // The flat extent is a single number and the logical spelling points at the
    // metadata's own sizes, so neither needs the heap allocation this used to
    // make once per descriptor.
    const bool logical = logical_storage && meta.contiguous;
    const int64_t flat_storage_numel = meta.storage_numel;
    const int64_t* storage_dims =
        logical ? meta.sizes.data() : &flat_storage_numel;
    const uint64_t storage_rank =
        logical ? static_cast<uint64_t>(meta.ndim) : 1;
    ptr_ = acl_create_tensor()(
        meta.sizes.data(), static_cast<uint64_t>(meta.ndim),
        acl_dtype(meta.scalar_type), meta.strides.data(), meta.storage_offset,
        format_, storage_dims, storage_rank, meta.data);
    // Off by default: `FLA_STABLE_DEBUG_DESC=1` prints what this descriptor
    // looks like, which is the only way to compare our arguments with the
    // ctypes reference's when a tiling accepts one and rejects the other.
    if (const char* debug = std::getenv("FLA_STABLE_DEBUG_DESC")) {
      if (debug[0] == '1') {
        std::fprintf(stderr, "[desc] shape=(");
        for (int64_t dim = 0; dim < meta.ndim; ++dim) {
          std::fprintf(stderr, "%s%lld", dim ? "," : "",
                       static_cast<long long>(meta.sizes[dim]));
        }
        std::fprintf(stderr, ") strides=(");
        for (int64_t dim = 0; dim < meta.ndim; ++dim) {
          std::fprintf(stderr, "%s%lld", dim ? "," : "",
                       static_cast<long long>(meta.strides[dim]));
        }
        std::fprintf(stderr, ") offset=%lld format=%d storage=(",
                     static_cast<long long>(meta.storage_offset), format_);
        for (uint64_t dim = 0; dim < storage_rank; ++dim) {
          std::fprintf(stderr, "%s%lld", dim ? "," : "",
                       static_cast<long long>(storage_dims[dim]));
        }
        std::fprintf(stderr, ") dtype=%d\n", meta.scalar_type);
      }
    }
    if (ptr_ == nullptr) {
      throw std::runtime_error(
          "fla_ascendc: aclCreateTensor returned nullptr");
    }
  }

  ~AclTensorView() {
    if (ptr_ != nullptr) {
      acl_destroy_tensor()(ptr_);
    }
  }

  AclTensorView(const AclTensorView&) = delete;
  AclTensorView& operator=(const AclTensorView&) = delete;

  // Movable so that an argument tuple can be handed to torch_npu's task queue
  // (see exec.h): the descriptor keeps exactly one owner, and the source is
  // left empty so its destructor cannot destroy the same descriptor twice.
  AclTensorView(AclTensorView&& other) noexcept
      : ptr_(other.ptr_), format_(other.format_) {
    other.ptr_ = nullptr;
  }
  AclTensorView& operator=(AclTensorView&& other) noexcept {
    if (this != &other) {
      if (ptr_ != nullptr) {
        acl_destroy_tensor()(ptr_);
      }
      ptr_ = other.ptr_;
      format_ = other.format_;
      other.ptr_ = nullptr;
    }
    return *this;
  }

  aclTensor* get() const { return ptr_; }

 private:
  aclTensor* ptr_ = nullptr;
  int32_t format_ = kAclFormatNd;
};

// Allocate a contiguous tensor of `meta`'s shape/dtype/device through the C
// shim.  torch::stable::empty_like would be an aten::empty_like dispatcher round
// trip (and needs a Tensor, reviving the stealing-temporary trap).
inline torch::stable::Tensor allocate_like(const TensorMeta& meta) {
  std::vector<int64_t> strides(meta.sizes.size(), 1);
  for (size_t dim = meta.sizes.size(); dim-- > 1;) {
    strides[dim - 1] = strides[dim] * meta.sizes[dim];
  }
  AtenTensorHandle handle = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(
      static_cast<int64_t>(meta.sizes.size()), meta.sizes.data(),
      strides.data(), meta.scalar_type, meta.device_type, meta.device_index,
      &handle));
  return torch::stable::Tensor(handle);  // steals the new reference
}

// Allocate a contiguous tensor with explicit sizes/dtype on `device_source`'s
// device (used by the generated adapters, whose output shapes come from the
// spec's structured `output` description).
inline torch::stable::Tensor allocate_sizes(
    const std::vector<int64_t>& sizes, int32_t dtype,
    const TensorMeta& device_source) {
  std::vector<int64_t> strides(sizes.size(), 1);
  for (size_t dim = sizes.size(); dim-- > 1;) {
    strides[dim - 1] = strides[dim] * sizes[dim];
  }
  AtenTensorHandle handle = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(
      static_cast<int64_t>(sizes.size()), sizes.data(), strides.data(), dtype,
      device_source.device_type, device_source.device_index, &handle));
  return torch::stable::Tensor(handle);
}

// Allocate a 1-D byte buffer (workspace) on `meta`'s device.
inline torch::stable::Tensor allocate_bytes(int64_t bytes,
                                            const TensorMeta& meta) {
  const int64_t sizes[1] = {bytes};
  const int64_t strides[1] = {1};
  constexpr int32_t kTorchByte = 0;
  AtenTensorHandle handle = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(
      1, sizes, strides, kTorchByte, meta.device_type, meta.device_index,
      &handle));
  return torch::stable::Tensor(handle);
}

// ---------------------------------------------------------------------------
// int[] arguments.
//
// torch 2.9's stable value conversions have no list support at all (no
// aoti_torch_*list* shim, no ToImpl<std::vector<T>>), so an `int[]` schema
// argument cannot be read on this side of the ABI.  The launchers therefore
// take such arrays as a *host int64 tensor* instead (the Python wrapper builds
// one) and the values are copied out here.  That keeps every op expressible
// without a list-capable shim.
// ---------------------------------------------------------------------------
inline std::vector<int64_t> host_int_values(AtenTensorHandle handle) {
  std::vector<int64_t> values;
  if (handle == nullptr) {
    return values;
  }
  TensorMeta meta;
  fill_meta(handle, &meta);
  if (!meta.defined) {
    return values;
  }
  const int64_t count = meta.storage_numel > 0 ? meta.storage_numel : 0;
  if (meta.scalar_type == 4) {  // kLong
    const auto* data = static_cast<const int64_t*>(meta.data);
    values.assign(data, data + count);
  } else if (meta.scalar_type == 3) {  // kInt
    const auto* data = static_cast<const int32_t*>(meta.data);
    values.reserve(static_cast<size_t>(count));
    for (int64_t index = 0; index < count; ++index) {
      values.push_back(static_cast<int64_t>(data[index]));
    }
  } else {
    throw std::runtime_error(
        "fla_ascendc: int[] argument must be an int32/int64 tensor");
  }
  return values;
}

// RAII aclIntArray built from host values (a null/empty vector yields nullptr,
// matching the ctypes path where an absent option is passed as nullptr).
class AclIntArrayView {
 public:
  explicit AclIntArrayView(const std::vector<int64_t>& values)
      : owned_(values) {
    if (owned_.empty()) {
      return;
    }
    ptr_ = acl_create_int_array()(owned_.data(),
                                  static_cast<uint64_t>(owned_.size()));
    if (ptr_ == nullptr) {
      throw std::runtime_error(
          "fla_ascendc: aclCreateIntArray returned nullptr");
    }
  }

  ~AclIntArrayView() {
    if (ptr_ != nullptr) {
      acl_destroy_int_array()(ptr_);
    }
  }

  AclIntArrayView(const AclIntArrayView&) = delete;
  AclIntArrayView& operator=(const AclIntArrayView&) = delete;

  // Movable for the same reason as AclTensorView.  `owned_` moves with the
  // handle because the aclIntArray points at the vector's buffer.
  AclIntArrayView(AclIntArrayView&& other) noexcept
      : owned_(std::move(other.owned_)), ptr_(other.ptr_) {
    other.ptr_ = nullptr;
  }
  AclIntArrayView& operator=(AclIntArrayView&& other) noexcept {
    if (this != &other) {
      if (ptr_ != nullptr) {
        acl_destroy_int_array()(ptr_);
      }
      owned_ = std::move(other.owned_);
      ptr_ = other.ptr_;
      other.ptr_ = nullptr;
    }
    return *this;
  }

  aclIntArray* get() const { return ptr_; }

 private:
  std::vector<int64_t> owned_;
  aclIntArray* ptr_ = nullptr;
};

}  // namespace stable
}  // namespace fla_ascendc

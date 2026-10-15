// An ATen-shaped facade over the stable metadata API.
//
// The op specs describe their outputs with ATen idioms
// (``at::empty({v.size(0), ...}, v.options())``, ``at::empty_like(x)``,
// ``at::Tensor()`` for "always null", plus ``helpers`` written against
// ``at::Tensor``).  Rather than rewrite 11 specs, the stable backend provides
// just enough of that surface here so the same spec text compiles.
//
// Everything is a *view*: no ATen headers, no libtorch C++ symbols, and the
// allocation goes through aoti_torch_empty_strided exactly like the hand-written
// adapters.  `Tensor` holds a copy of the stable tensor handle (shared_ptr), so
// nothing is stolen and nothing dangles.
#pragma once

#include "stable/acl_meta.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace fla_ascendc {
namespace stable {
namespace at_shim {

constexpr int32_t kByte = 0;
constexpr int32_t kChar = 1;
constexpr int32_t kShort = 2;
constexpr int32_t kInt = 3;
constexpr int32_t kLong = 4;
constexpr int32_t kHalf = 5;
constexpr int32_t kFloat = 6;
constexpr int32_t kDouble = 7;
constexpr int32_t kBool = 11;
constexpr int32_t kBFloat16 = 15;

struct TensorOptions {
  int32_t dtype_id = kFloat;
  int32_t device_type = 0;
  int32_t device_index = 0;

  TensorOptions dtype(int32_t value) const {
    TensorOptions copy = *this;
    copy.dtype_id = value;
    return copy;
  }
};

class Tensor {
 public:
  Tensor() = default;
  explicit Tensor(const torch::stable::Tensor& tensor)
      : holder_(std::make_shared<torch::stable::Tensor>(tensor)) {}

  bool defined() const { return holder_ && holder_->get() != nullptr; }
  int64_t size(int64_t dim) const { return meta().sizes[dim]; }
  std::vector<int64_t> sizes() const { return meta().sizes; }
  int64_t dim() const { return meta().ndim; }
  int32_t scalar_type() const { return meta().scalar_type; }

  TensorOptions options() const {
    const TensorMeta m = meta();
    return TensorOptions{m.scalar_type, m.device_type, m.device_index};
  }

  const torch::stable::Tensor& tensor() const { return *holder_; }

 private:
  TensorMeta meta() const {
    if (!defined()) {
      throw std::runtime_error(
          "fla_ascendc: undefined tensor used in a size/options "
          "expression");
    }
    return meta_of(*holder_);
  }

  std::shared_ptr<torch::stable::Tensor> holder_;
};

inline Tensor empty(const std::vector<int64_t>& sizes, TensorOptions options) {
  TensorMeta device_source;
  device_source.device_type = options.device_type;
  device_source.device_index = options.device_index;
  return Tensor(allocate_sizes(sizes, options.dtype_id, device_source));
}

inline Tensor empty_like(const Tensor& like) {
  const TensorMeta m = meta_of(like.tensor());
  return Tensor(allocate_sizes(m.sizes, m.scalar_type, m));
}

}  // namespace at_shim
}  // namespace stable
}  // namespace fla_ascendc

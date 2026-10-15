// Generic boxed-kernel adapter for the Stable-ABI launcher.
//
// torch's stable library registers boxed kernels only -- StableLibrary::impl
// takes `void (*)(StableIValue*, uint64_t, uint64_t)` and nothing else -- so
// every operator needs a function that pops its arguments off the stack, calls
// the typed adapter, and pushes the results back.  That shuffle is derived from
// the adapter's own signature, so an operator registers with one line:
//
//   m.impl("npu_x", &fla_ascendc::stable::boxed_adapter<run_npu_x>);
//
// and `run_npu_x` is a plain typed function.  Parameter types must be one of
// Tensor, std::optional<Tensor>, int64_t, double, bool; the return type must be
// Tensor, std::optional<Tensor>, or std::tuple of those.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include <torch/csrc/stable/stableivalue_conversions.h>

namespace fla_ascendc {
namespace stable {

using torch::stable::Tensor;

// --- stack -> C++ -----------------------------------------------------------

template <class T>
inline T unbox(StableIValue value);

template <>
inline Tensor unbox<Tensor>(StableIValue value) {
  return to<Tensor>(value);
}

template <>
inline std::optional<Tensor> unbox<std::optional<Tensor>>(StableIValue value) {
  return to<std::optional<Tensor>>(value);
}

template <>
inline int64_t unbox<int64_t>(StableIValue value) {
  return to<int64_t>(value);
}

template <>
inline double unbox<double>(StableIValue value) {
  return to<double>(value);
}

template <>
inline bool unbox<bool>(StableIValue value) {
  return to<bool>(value);
}

template <class Tuple, size_t... I>
inline Tuple unbox_all(StableIValue* stack, std::index_sequence<I...>) {
  return Tuple(unbox<std::tuple_element_t<I, Tuple>>(stack[I])...);
}

// --- C++ -> stack -----------------------------------------------------------

inline void pack(const Tensor& value, StableIValue* stack, uint64_t* index) {
  stack[(*index)++] = from(value);
}

inline void pack(const std::optional<Tensor>& value, StableIValue* stack,
                 uint64_t* index) {
  // A `Tensor?` return slot is filled with the *boxed* optional form, not with
  // a bare tensor handle: the dispatcher reads that slot as a pointer to the
  // box, so packing the handle directly crashes it as soon as the call returns.
  // `from(std::optional<Tensor>)` produces the boxed form for both a present
  // and an absent value.
  stack[(*index)++] = from(value);
}

template <class... Rs>
inline void pack(const std::tuple<Rs...>& values, StableIValue* stack,
                 uint64_t* index) {
  std::apply([&](const Rs&... value) { (pack(value, stack, index), ...); },
             values);
}

template <class R>
struct ReturnArity {
  static constexpr uint64_t value = 1;
};

template <class... Rs>
struct ReturnArity<std::tuple<Rs...>> {
  static constexpr uint64_t value = sizeof...(Rs);
};

// --- adapter ----------------------------------------------------------------

template <class Fn>
struct FunctionTraits;

template <class R, class... A>
struct FunctionTraits<R (*)(A...)> {
  using Return = R;
  using Args = std::tuple<A...>;
};

template <auto RunFn>
void boxed_adapter(StableIValue* stack, uint64_t num_inputs,
                   uint64_t num_outputs) {
  using Traits = FunctionTraits<decltype(RunFn)>;
  using Args = typename Traits::Args;
  using Return = typename Traits::Return;

  // The stack is read positionally, so a schema that gained or lost a parameter
  // would shift every following argument instead of failing to compile.
  constexpr uint64_t kInputs = std::tuple_size<Args>::value;
  if (num_inputs != kInputs) {
    throw std::runtime_error(
        "fla_ascendc: boxed adapter takes " + std::to_string(kInputs) +
        " inputs but the schema declares " + std::to_string(num_inputs));
  }

  Args args = unbox_all<Args>(
      stack, std::make_index_sequence<std::tuple_size<Args>::value>{});
  Return result = std::apply(RunFn, args);

  uint64_t packed = 0;
  pack(result, stack, &packed);
  // A mismatch means the schema's return list and the adapter disagree, which
  // would otherwise silently write past the stack (or leave slots unset).
  if (packed != num_outputs) {
    throw std::runtime_error(
        "fla_ascendc: boxed adapter packed " +
        std::to_string(packed) + " outputs but the schema declares " +
        std::to_string(num_outputs));
  }
}

}  // namespace stable
}  // namespace fla_ascendc

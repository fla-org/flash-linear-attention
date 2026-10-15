// Stable-ABI launch helper: the ABI-free counterpart of vLLM's EXEC_NPU_CMD.
//
// Every adapter is the same three steps -- build an acl descriptor per argument,
// ask the operator for its workspace size, launch it on the caller's stream --
// and only the argument list differs.  This header owns the mechanical part so
// an adapter stays ~3 lines plus its own shape logic.
//
// Two properties matter and are easy to lose:
//
//   * Descriptors must outlive the launch, not just the GetWorkspaceSize call.
//     The holders are therefore collected in a tuple that lives for the whole
//     helper, never as temporaries inside one expression.
//   * The GetWorkspaceSize pointer has an operator-specific signature, so it is
//     reinterpret_cast to the type derived from the tuple's holder types.  A
//     `const aclTensor*` parameter is spelled `aclTensor*` here; the pointer
//     representation is identical, which is what makes the cast safe.
//
// `FLA_STABLE_EXEC` must be called with arguments in exactly the order the
// aclnn entry point declares them.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "stable/acl_meta.h"

namespace fla_ascendc {
namespace stable {

// --- argument holders -------------------------------------------------------

class TensorArg {
 public:
  // `format` defaults to the rank-inferred one; pass `kAclFormatNd` for the
  // call sites whose reference asks for ND explicitly (see nd_tensor below).
  explicit TensorArg(const TensorMeta& meta,
                     int32_t format = kAclFormatAuto,
                     bool logical_storage = false)
      : view_(meta, format, logical_storage) {}
  aclTensor* get() const { return view_.get(); }

 private:
  AclTensorView view_;
};

// Same storage as an input view: aclnn declares outputs `const aclTensor*` in
// most operators and the cast below does not care, but naming it keeps the
// adapter readable.
using OutTensorArg = TensorArg;

class IntArrayArg {
 public:
  explicit IntArrayArg(std::vector<int64_t> values) : view_(values) {}
  aclIntArray* get() const { return view_.get(); }

 private:
  AclIntArrayView view_;
};

class CStrArg {
 public:
  explicit CStrArg(std::string name) : name_(std::move(name)) {}
  const char* get() const { return name_.c_str(); }

 private:
  std::string name_;
};

template <class T>
class ScalarArg {
 public:
  explicit ScalarArg(T value) : value_(value) {}
  T get() const { return value_; }

 private:
  T value_;
};

// --- constructors used at the call site ------------------------------------

inline TensorArg tensor(const TensorMeta& meta) { return TensorArg(meta); }
inline TensorArg tensor(const torch::stable::Tensor& value) {
  return TensorArg(meta_of(value));
}

// The same holders again, with the descriptor forced to ND.  A handful of
// operators' references pass `acl_format_override=ACL_FORMAT_ND` (they say the
// tensors are row-major and the format tag should not be read from the tensor):
// the upstream ABI validator cannot see this, so the list is kept explicit --
// chunk_fwd_h, chunk_gated_delta_rule_fwd_prepare, chunk_gdn_bwd_intra,
// chunk_kda_bwd, chunk_kda_bwd_intra and the two recurrent adapters.
inline TensorArg nd_tensor(const TensorMeta& meta) {
  return TensorArg(meta, kAclFormatNd, /*logical_storage=*/true);
}
inline TensorArg nd_tensor(const torch::stable::Tensor& value) {
  return TensorArg(meta_of(value), kAclFormatNd, /*logical_storage=*/true);
}
inline TensorArg nd_optional_tensor(const std::optional<TensorMeta>& meta) {
  return TensorArg(meta.value_or(TensorMeta()), kAclFormatNd,
                   /*logical_storage=*/true);
}
inline TensorArg nd_optional_tensor(std::nullopt_t) {
  return TensorArg(TensorMeta(), kAclFormatNd, /*logical_storage=*/true);
}
inline TensorArg nd_optional_tensor(
    const std::optional<torch::stable::Tensor>& value) {
  if (!value.has_value()) {
    return TensorArg(TensorMeta(), kAclFormatNd, /*logical_storage=*/true);
  }
  const TensorMeta meta = meta_of(*value);
  return TensorArg(meta.defined ? meta : TensorMeta(), kAclFormatNd,
                   /*logical_storage=*/true);
}

// The reference's third spelling: `storage_shape_override=_shape(tensor)` with
// the format left to the tensor.  Used by the operators that hand the tiling a
// logical storage shape without forcing ND: chunk_gated_delta_rule_bwd,
// chunk_gated_delta_rule_bwd_dhu and chunk_gated_delta_rule_bwd_finalize.
inline TensorArg logical_tensor(const TensorMeta& meta) {
  return TensorArg(meta, kAclFormatAuto, /*logical_storage=*/true);
}
inline TensorArg logical_tensor(const torch::stable::Tensor& value) {
  return TensorArg(meta_of(value), kAclFormatAuto, /*logical_storage=*/true);
}
inline TensorArg logical_optional_tensor(
    const std::optional<torch::stable::Tensor>& value) {
  if (!value.has_value()) {
    return TensorArg(TensorMeta(), kAclFormatAuto, /*logical_storage=*/true);
  }
  const TensorMeta meta = meta_of(*value);
  return TensorArg(meta.defined ? meta : TensorMeta(), kAclFormatAuto,
                   /*logical_storage=*/true);
}
inline TensorArg logical_optional_tensor(std::nullopt_t) {
  return TensorArg(TensorMeta(), kAclFormatAuto, /*logical_storage=*/true);
}
inline TensorArg logical_optional_tensor(
    const std::optional<TensorMeta>& meta) {
  return TensorArg(meta.value_or(TensorMeta()), kAclFormatAuto,
                   /*logical_storage=*/true);
}
inline OutTensorArg logical_out_tensor(const TensorMeta& meta) {
  return OutTensorArg(meta, kAclFormatAuto, /*logical_storage=*/true);
}

inline TensorArg optional_tensor(std::optional<TensorMeta> meta) {
  return TensorArg(meta.value_or(TensorMeta()));
}
// `std::nullopt` converts to both optional types, so the plain spelling is
// ambiguous; the adapters mean "no descriptor" when they pass it.
inline TensorArg optional_tensor(std::nullopt_t) {
  return TensorArg(TensorMeta());
}
inline TensorArg optional_tensor(
    const std::optional<torch::stable::Tensor>& value) {
  if (!value.has_value()) {
    return TensorArg(TensorMeta());
  }
  const TensorMeta meta = meta_of(*value);
  return TensorArg(meta.defined ? meta : TensorMeta());
}

inline OutTensorArg out_tensor(const TensorMeta& meta) {
  return OutTensorArg(meta);
}
inline OutTensorArg nd_out_tensor(const TensorMeta& meta) {
  return OutTensorArg(meta, kAclFormatNd);
}
// The same ND spelling with the reference's `storage_shape_override=` logical
// storage shape.  `aclnnChunkKdaFwdFinalize` is the one operator that reads it
// on an *output*: given a flat rank-1 storage shape it fails its own tiling and
// reports ACLNN_ERR_INNER_NULLPTR (561103) instead of writing anything.
inline OutTensorArg nd_logical_out_tensor(const TensorMeta& meta) {
  return OutTensorArg(meta, kAclFormatNd, /*logical_storage=*/true);
}
inline OutTensorArg out_tensor(const torch::stable::Tensor& value) {
  return OutTensorArg(meta_of(value));
}

// `int_array` arguments cross the Python boundary as a host int64 tensor (the
// stable conversions have no int[]), so the adapter copies the values out of
// it.  An absent/empty value becomes a null aclIntArray, exactly like the
// ctypes path passing nullptr.
//
// Adapters that need the metadata *before* the call -- an output shape that
// counts segments, for instance -- take the values with `int_values` and hand
// the same vector to `int_array`, so the values are copied once.
inline std::vector<int64_t> int_values(
    const std::optional<torch::stable::Tensor>& value) {
  std::vector<int64_t> values;
  if (value.has_value()) {
    const TensorMeta meta = meta_of(*value);
    if (meta.defined) {
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
            "fla_ascendc: int[] argument must be an int32/int64 "
            "tensor");
      }
    }
  }
  return values;
}

inline IntArrayArg int_array(const std::optional<torch::stable::Tensor>& value) {
  return IntArrayArg(int_values(value));
}
inline IntArrayArg int_array(std::vector<int64_t> values) {
  return IntArrayArg(std::move(values));
}

// Enum-coded `char*` argument: the name table is the single source of the legal
// values, so the same array feeds `cstr` and the coverage gate.
template <size_t N>
inline const char* enum_name(const char* const (&names)[N], int64_t code) {
  if (code >= 0 && static_cast<size_t>(code) < N) {
    return names[code];
  }
  throw std::runtime_error("fla_ascendc: bad enum code " +
                           std::to_string(code));
}

template <size_t N>
inline CStrArg cstr(const char* const (&names)[N], int64_t code) {
  return CStrArg(enum_name(names, code));
}

template <class T>
inline ScalarArg<T> scalar(T value) {
  return ScalarArg<T>(value);
}

// --- the launch itself ------------------------------------------------------

namespace detail {

// The GetWorkspaceSize pointer type this argument list implies.  Elements may
// be references (the macro forwards a tuple of them), so every step strips the
// reference before asking a holder for its C value.
template <class Element>
using ArgCType = decltype(
    std::declval<const std::remove_reference_t<Element>&>().get());

template <class Tuple, size_t... I>
inline int call_get_workspace(void* address, Tuple& args,
                              std::index_sequence<I...>, uint64_t* workspace,
                              aclOpExecutor** executor) {
  using GetWorkspaceFn = int (*)(ArgCType<std::tuple_element_t<I, Tuple>>...,
                                 uint64_t*, aclOpExecutor**);
  auto fn = reinterpret_cast<GetWorkspaceFn>(address);
  return fn(std::get<I>(args).get()..., workspace, executor);
}

// A launch that runs on torch_npu's task queue returns to no caller, so its
// status cannot be thrown where the caller could catch it.  The code is left
// here instead and raised by the next operator call, on whichever thread makes
// it.  Errors that a caller can act on (a rejected shape, a bad dtype) come out
// of GetWorkspaceSize synchronously; this covers what is left.
inline std::mutex& async_failure_mutex() {
  static std::mutex mutex;
  return mutex;
}

inline std::string& async_failure_message() {
  static std::string message;
  return message;
}

inline void report_async_failure(const std::string& api, int code) {
  std::lock_guard<std::mutex> guard(async_failure_mutex());
  if (async_failure_message().empty()) {
    async_failure_message() =
        api + " failed asynchronously: " + std::to_string(code);
  }
}

inline void check_async_failure() {
  std::string message;
  {
    std::lock_guard<std::mutex> guard(async_failure_mutex());
    if (async_failure_message().empty()) {
      return;
    }
    message.swap(async_failure_message());
  }
  throw std::runtime_error("fla_ascendc: " + message);
}

// The argument tuple the macro builds is a tuple of references to holders, so
// it cannot be handed to the task queue as it stands: the queue runs the launch
// after the calling full-expression is gone.  These turn it into a tuple of
// values (the holders are movable) that the queue entry owns.
template <class Tuple>
struct ValueTuple;

template <class... Elements>
struct ValueTuple<std::tuple<Elements...>> {
  using type = std::tuple<std::remove_reference_t<Elements>...>;
};

template <class Held, class Tuple, size_t... I>
inline Held hold_args(Tuple& args, std::index_sequence<I...>) {
  return Held(std::move(std::get<I>(args))...);
}

// What a queue entry owns while it waits its turn: the descriptors, which the
// launch reads when the consumer thread reaches it.
//
// The workspace is deliberately not owned here.  vLLM's EXEC_NPU_CMD allocates
// it as a local `at::Tensor` that dies on the calling thread as soon as the
// enqueue returns; keeping it in the entry instead moves that free onto the
// consumer thread, and on 910B that is what grew the caching allocator's
// reserved bytes by ~0.8 MiB per call (2000 decode-shaped recurrent calls:
// +1530 MiB, against 0 for the same calls submitted inline) until the service
// OOM'd.  Freeing on the calling thread costs no ordering: the launch joins the
// queue that every later operator on this stream also joins, so nothing can be
// submitted ahead of it.
template <class Held>
struct QueuedLaunch {
  std::optional<Held> held;

  void release() { held.reset(); }
};

// Hand one launch to torch_npu's task queue, and say whether it went there.
//
// Both the macro and the two hand-written recurrent adapters submit through
// here, so "queued" means the same thing in every adapter: the entry owns the
// descriptors until the launch has been submitted, and the ordering the caller
// gave up by not taking the queue's barrier now comes from the queue itself.
//
// `held` is borrowed, not consumed: on a refusal it is handed back untouched,
// so the caller can still launch inline.  The workspace is not passed in at all
// -- it stays owned by the caller's frame (see QueuedLaunch).  That refusal is
// not hypothetical -- `enCurrentNPUStream` rejects an *external* stream
// ("External NPU stream is not supported by task queue enqueue") before it
// submits anything, and a launcher that dropped the descriptors there would
// turn "this stream is not queueable" into a dangling one.
//
// Returns false when the caller has to submit inline: this host has no queue
// entry point, or torch_npu refused the enqueue for this stream.  An inline
// submission needs a stream read that first drains the queue -- the
// barrier-taking accessor -- because it would otherwise overtake whatever the
// host enqueued before it. The PoC Python adapter always uses the flushing
// accessor before entering this launcher.
template <class Held>
inline bool enqueue_launch(Runtime& runtime, const char* api, LaunchFn launch,
                           int64_t stream, void* workspace_ptr,
                           uint64_t workspace_size, aclOpExecutor* executor,
                           Held& held) {
  if (!runtime.enqueue_enabled()) {
    return false;
  }
  auto state = std::make_shared<QueuedLaunch<std::remove_reference_t<Held>>>();
  state->held.emplace(std::move(held));
  // `api` is a literal in every caller, so the pointer is enough to capture and
  // costs nothing per call.
  auto call = [state, launch, workspace_ptr, workspace_size, executor, api,
               stream]() -> int {
    const int ret = launch(workspace_ptr, workspace_size, executor,
                           reinterpret_cast<void*>(stream));
    if (ret != 0) {
      report_async_failure(api, ret);
    }
    // The descriptors have done their job once the launch has been submitted.
    state->release();
    return ret;
  };
  // The queue entry moves the callable out of `call`, so its state (and with it
  // every descriptor) lives exactly as long as the entry does.
  try {
    runtime.enqueue(api, call);
  } catch (const std::exception&) {
    // The enqueue was refused before it wrote to the queue, so nothing has been
    // submitted: hand the caller back its arguments and let it take the inline
    // path, which is how this stream was served before the queue existed.
    held = std::move(*state->held);
    state->release();
    return false;
  }
  return true;
}

}  // namespace detail

// `FLA_ASCENDC_STABLE_TRACE=1` prints one line per operator the first time each one
// runs: how much workspace its tiling asked for and which stream the glue read
// for it.  The environment is read once, so a traced build costs the same as an
// untraced one on the hot path.
inline bool trace_enabled() {
  static const bool enabled = std::getenv("FLA_ASCENDC_STABLE_TRACE") != nullptr;
  return enabled;
}

inline void trace_once(const char* api, uint64_t workspace_size,
                       int64_t stream) {
  static std::mutex mutex;
  static std::set<std::string> seen;
  std::lock_guard<std::mutex> guard(mutex);
  if (!seen.insert(api).second) {
    return;
  }
  std::fprintf(stderr,
               "[fla_ascendc] %s workspace=%llu bytes stream=%#llx\n", api,
               static_cast<unsigned long long>(workspace_size),
               static_cast<unsigned long long>(stream));
}

// `Tuple` is whatever the macro's std::forward_as_tuple produced; the holders
// inside it live until the end of the calling full-expression, which is what
// keeps every descriptor alive across both aclnn calls.
//
// `stream` is read by the Python glue on every call -- never cached, because a
// cached stream pointer is what sent kernels to another thread's stream in the
// vLLM run. The PoC adapter drains the task queue before reading the stream.
//
// The value is kept per thread (see fla_ascendc_last_launch_stream) so a test
// can read back *which* stream a call ran on, from the thread that made it.
inline thread_local int64_t t_last_launch_stream = -1;

// Remember which stream this call is about to launch on.
inline void note_launch_stream(int64_t stream) {
  t_last_launch_stream = stream;
}

template <class Tuple>
inline void exec(const char* api, const TensorMeta& workspace_meta,
                 int64_t stream, Tuple&& args) {
  using Elements = std::remove_reference_t<Tuple>;
  const std::string base(api);
  auto& runtime = Runtime::instance();
  void* get_workspace = runtime.symbol(base + "GetWorkspaceSize");
  auto launch = reinterpret_cast<LaunchFn>(runtime.symbol(base));

  note_launch_stream(stream);
  detail::check_async_failure();

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  const int get_ret =
      detail::call_get_workspace(get_workspace, args,
                                 std::make_index_sequence<
                                     std::tuple_size<Elements>::value>{},
                                 &workspace_size, &executor);
  if (get_ret != 0) {
    throw std::runtime_error("fla_ascendc: " + base +
                             "GetWorkspaceSize failed: " +
                             std::to_string(get_ret));
  }
  if (trace_enabled()) {
    trace_once(api, workspace_size, stream);
  }

  torch::stable::Tensor workspace;
  void* workspace_ptr = nullptr;
  if (workspace_size != 0) {
    // The workspace must be allocated on the operator's device, which is why
    // the caller passes a meta rather than letting this guess.
    // It is also this frame's, i.e. the calling thread's, exactly like the
    // local `at::Tensor` in vLLM's EXEC_NPU_CMD: see QueuedLaunch for why it
    // must not outlive the enqueue.
    workspace = allocate_bytes(static_cast<int64_t>(workspace_size),
                               workspace_meta);
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_data_ptr(workspace.get(), &workspace_ptr));
  }

  // Two ways to submit, picked by what the host's torch_npu offers.
  //
  // Enqueue: hand the launch to torch_npu's task queue, where it runs on the
  // consumer thread in queue order -- the same path vLLM's EXEC_NPU_CMD takes.
  // Ordering then comes from the queue, so the stream does not have to be read
  // through the barrier-taking accessor, and on a busy worker that barrier is
  // the difference between ~0.25 ms and ~1.2 ms per call.
  //
  // Inline: submit directly on the host thread.  Correct only together with a
  // stream read that first drains the queue -- the accessor the glue picks when
  // the launcher reports no queue entry point -- because a direct submission
  // would otherwise overtake work the host enqueued before it.  This is the
  // fallback for a torch_npu without that entry point.
  using Held = typename detail::ValueTuple<Elements>::type;
  auto held = detail::hold_args<Held>(
      args, std::make_index_sequence<std::tuple_size<Held>::value>{});
  if (detail::enqueue_launch(runtime, api, launch, stream, workspace_ptr,
                             workspace_size, executor, held)) {
    return;
  }

  const int launch_ret = launch(workspace_ptr, workspace_size, executor,
                                reinterpret_cast<void*>(stream));
  if (launch_ret != 0) {
    throw std::runtime_error("fla_ascendc: " + base + " failed: " +
                             std::to_string(launch_ret));
  }
}

}  // namespace stable
}  // namespace fla_ascendc

// Arguments go in aclnn order; see the header comment for why that is checked
// offline rather than at run time.
#define FLA_STABLE_EXEC(api, workspace_meta, stream, ...)         \
  ::fla_ascendc::stable::exec(                                   \
      (api), (workspace_meta), (stream),                          \
      std::forward_as_tuple(__VA_ARGS__))

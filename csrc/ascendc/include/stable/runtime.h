// Runtime symbol resolution for the launcher (no torch_npu headers).
//
// `dlopen`s the FLA custom `op_api` library and CANN's libopapi, and resolves
// `aclnn*` entry points with a per-name cache.  It lives with the Stable-ABI
// launcher because that is the one that ships.
#pragma once

#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>

namespace fla_ascendc {

class Runtime {
 public:
  static Runtime& instance();

  // dlopen the FLA custom op_api library (FLA_ASCENDC_OP_API_LIB or explicit path)
  // and CANN libopapi.so. Safe to call multiple times.
  void init(const std::string& custom_lib_path = "");

  // Custom-first symbol lookup with per-name cache. Throws std::runtime_error
  // when a symbol cannot be resolved.
  void* symbol(const std::string& name);

  // torch_npu's entry point for "run this callable on the task queue"
  // (`at_npu::native::OpCommand::RunOpApiV2`).  Null on a host whose torch_npu
  // does not export it; the launcher then stays on the inline path.
  void* queue_enqueue();

  // Whether this library will hand the launch to torch_npu's task queue.
  // The PoC Python adapter drains the task queue before either launch mode.
  bool enqueue_enabled();

  // Hand `fn` to torch_npu's task queue.  It runs on the queue's consumer
  // thread in queue order, which is what lets the launch skip the "drain the
  // queue first" barrier that a direct submission needs.  The callable is
  // *moved* into the queue entry, so `fn` is empty afterwards.
  void enqueue(const std::string& name, const std::function<int()>& fn);

 private:
  Runtime() = default;
  void* open_custom_library(const std::string& path);

  void* custom_handle_ = nullptr;
  void* cann_handle_ = nullptr;
  bool initialized_ = false;
  // `symbol()` is called from the queue's consumer thread as well as from the
  // operator threads, so the cache needs its own lock.
  std::mutex mutex_;
  std::unordered_map<std::string, void*> cache_;

  void* queue_enqueue_ = nullptr;
  bool queue_enqueue_looked_up_ = false;
  // RunOpApi (V1) takes the callable by value, RunOpApiV2 by const reference:
  // the two are not interchangeable through one function pointer type.
  bool queue_enqueue_takes_value_ = false;
};

}  // namespace fla_ascendc

#include "stable/runtime.h"

#include <dlfcn.h>

#include <cstdint>
#include <cstdlib>
#include <stdexcept>

namespace fla_ascendc {

namespace {

// torch_npu's entry point for "run this callable on the task queue", the same
// one its own inductor and mstx paths use.  Two spellings because the newer
// overload takes the callable by const reference and keeps the full operator
// name, while the older one takes it by value.
//
// This is what makes a submission that does not wait for the queue to drain
// ordering-safe: the callable runs on the queue's consumer thread, in queue
// order, so it is the queue -- not a host-side barrier -- that keeps our
// kernel behind everything the host enqueued before us.
constexpr const char* kQueueEnqueueV2 =
    "_ZN6at_npu6native9OpCommand10RunOpApiV2ERKNSt7__cxx1112basic_"
    "stringIcSt11char_traitsIcESaIcEEERKSt8functionIFivEEb";
constexpr const char* kQueueEnqueueV1 =
    "_ZN6at_npu6native9OpCommand8RunOpApiERKNSt7__cxx1112basic_"
    "stringIcSt11char_traitsIcESaIcEEESt8functionIFivEEb";

void* dlopen_required(const std::string& path, const char* what) {
  void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    throw std::runtime_error(
        std::string(what) + ": dlopen failed for " + path + ": " + dlerror());
  }
  return handle;
}

// torch_npu is loaded by the time an operator runs (the Python runtime imports it first),
// so the global namespace normally has the symbol.  The dlopen fallback covers
// a host that loaded the extension with RTLD_LOCAL instead.
void* find_exported(const char* name) {
  if (void* address = dlsym(RTLD_DEFAULT, name)) {
    return address;
  }
  static const char* const kLibraries[] = {"libtorch_npu.so",
                                           "libtorch_npu.so.2",
                                           "libtorch_npu.so.1"};
  for (const char* library : kLibraries) {
    void* handle = dlopen(library, RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
      continue;
    }
    if (void* address = dlsym(handle, name)) {
      return address;
    }
  }
  return nullptr;
}

}  // namespace

Runtime& Runtime::instance() {
  static Runtime runtime;
  return runtime;
}

void Runtime::init(const std::string& custom_lib_path) {
  std::lock_guard<std::mutex> guard(mutex_);
  if (initialized_) {
    return;
  }
  std::string cust_path = custom_lib_path;
  if (cust_path.empty()) {
    const char* env = std::getenv("FLA_ASCENDC_OP_API_LIB");
    if (env != nullptr && *env != '\0') {
      cust_path = env;
    }
  }
  if (cust_path.empty()) {
    throw std::runtime_error(
        "stable launcher: FLA_ASCENDC_OP_API_LIB is not set; call "
        "fla.utils.ascendc.load() before using the stable launcher");
  }
  custom_handle_ = open_custom_library(cust_path);
  cann_handle_ = dlopen_required("libopapi.so", "CANN opapi");
  initialized_ = true;
}

void* Runtime::open_custom_library(const std::string& path) {
  return dlopen_required(path, "FLA custom opapi");
}

void* Runtime::symbol(const std::string& name) {
  init();
  std::lock_guard<std::mutex> guard(mutex_);
  auto it = cache_.find(name);
  if (it != cache_.end()) {
    return it->second;
  }
  void* addr = nullptr;
  if (custom_handle_ != nullptr) {
    addr = dlsym(custom_handle_, name.c_str());
  }
  if (addr == nullptr && cann_handle_ != nullptr) {
    addr = dlsym(cann_handle_, name.c_str());
  }
  if (addr == nullptr) {
    throw std::runtime_error("stable launcher: symbol not found: " + name);
  }
  cache_[name] = addr;
  return addr;
}

void* Runtime::queue_enqueue() {
  if (!queue_enqueue_looked_up_) {
    queue_enqueue_ = find_exported(kQueueEnqueueV2);
    if (queue_enqueue_ != nullptr) {
      queue_enqueue_takes_value_ = false;
    } else {
      queue_enqueue_ = find_exported(kQueueEnqueueV1);
      queue_enqueue_takes_value_ = queue_enqueue_ != nullptr;
    }
    queue_enqueue_looked_up_ = true;
  }
  return queue_enqueue_;
}

bool Runtime::enqueue_enabled() {
  // retain one launch policy for the lifetime of this process.
  static const bool enabled = []() -> bool {
    const char* requested = std::getenv("FLA_ASCENDC_STABLE_LAUNCH");
    if (requested != nullptr && std::string(requested) == "inline") {
      return false;
    }
    return Runtime::instance().queue_enqueue() != nullptr;
  }();
  return enabled;
}

void Runtime::enqueue(const std::string& name, const std::function<int()>& fn) {
  void* entry = queue_enqueue();
  if (entry == nullptr) {
    throw std::runtime_error(
        "stable launcher: torch_npu's task queue entry point is not available");
  }
  if (queue_enqueue_takes_value_) {
    using EnqueueByValueFn = void (*)(const std::string&, std::function<int()>,
                                      bool);
    std::function<int()> copy = fn;  // the callee moves out of this one
    reinterpret_cast<EnqueueByValueFn>(entry)(name, std::move(copy), false);
    return;
  }
  using EnqueueFn = void (*)(const std::string&, const std::function<int()>&,
                             bool);
  reinterpret_cast<EnqueueFn>(entry)(name, fn, false);
}

}  // namespace fla_ascendc

// Adapted from flashserve/flash-linear-attention-npu, commit 066f47e4 (BSD-3-Clause).
#include <torch/csrc/stable/library.h>
#include "stable_chunk_local_cumsum.cpp"

extern "C" __attribute__((visibility("default")))
const char* fla_ascendc_source_hash() {
  return FLA_ASCENDC_SOURCE_HASH;
}

extern "C" __attribute__((visibility("default")))
int32_t fla_ascendc_queue_enqueue_available() {
  return fla_ascendc::Runtime::instance().enqueue_enabled() ? 1 : 0;
}

extern "C" __attribute__((visibility("default")))
int64_t fla_ascendc_last_launch_stream() {
  return fla_ascendc::stable::t_last_launch_stream;
}

STABLE_TORCH_LIBRARY(fla_ascendc, m) {
  m.def(kSchema_chunk_local_cumsum);
}

STABLE_TORCH_LIBRARY_IMPL(fla_ascendc, CompositeExplicitAutograd, m) {
  m.impl("npu_chunk_local_cumsum",
         &fla_ascendc::stable::boxed_adapter<run_npu_chunk_local_cumsum>);
}

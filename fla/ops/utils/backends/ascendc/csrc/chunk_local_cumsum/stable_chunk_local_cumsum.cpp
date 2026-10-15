// Adapted from flashserve/flash-linear-attention-npu (BSD-3-Clause; see NOTICE).
#include "stable/at_facade.h"
#include "stable/boxed.h"
#include "stable/exec.h"

#include <cstdint>
#include <optional>
#include <tuple>

namespace {

using torch::stable::Tensor;
using fla_ascendc::stable::TensorMeta;
using fla_ascendc::stable::at_shim::kBFloat16;
using fla_ascendc::stable::at_shim::kFloat;
using fla_ascendc::stable::allocate_like;
using fla_ascendc::stable::allocate_sizes;
using fla_ascendc::stable::cstr;
using fla_ascendc::stable::int_array;
using fla_ascendc::stable::meta_of;
using fla_ascendc::stable::optional_tensor;
using fla_ascendc::stable::out_tensor;
using fla_ascendc::stable::scalar;
using fla_ascendc::stable::size_of;
using fla_ascendc::stable::tensor;

constexpr const char* kChunkLocalCumsumOutputDtypeNames[] = {"float32",
                                                             "bfloat16"};

constexpr const char* kSchema_chunk_local_cumsum =
    "npu_chunk_local_cumsum(Tensor g, Tensor? cu_seqlens, "
    "Tensor? chunk_indices_out, int chunk_size, bool reverse, float scale, "
    "bool head_first, int output_dtype, int stream) -> Tensor";

Tensor run_npu_chunk_local_cumsum(Tensor g, std::optional<Tensor> cu_seqlens,
                                  std::optional<Tensor> chunk_indices_out,
                                  int64_t chunk_size, bool reverse, double scale,
                                  bool head_first, int64_t output_dtype,
                                  int64_t stream) {
  const TensorMeta g_meta = meta_of(g);
  Tensor out = allocate_sizes(g_meta.sizes,
                              output_dtype == 0 ? kFloat : kBFloat16, g_meta);
  FLA_STABLE_EXEC("aclnnChunkLocalCumsum", g_meta, stream, tensor(g_meta),
                  int_array(cu_seqlens), int_array(chunk_indices_out),
                  scalar(chunk_size), scalar(reverse), scalar(scale),
                  scalar(head_first),
                  cstr(kChunkLocalCumsumOutputDtypeNames, output_dtype),
                  out_tensor(meta_of(out)));
  return out;
}

}  // namespace

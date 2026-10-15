/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#include "aclnn_chunk_local_cumsum.h"

#include <string>

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ChunkLocalCumsum);

const aclTensor *ChunkLocalCumsum(
    const aclTensor *g,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOutOptional,
    int64_t chunkSize,
    bool reverse,
    double scale,
    bool headFirst,
    const char *outputDtypeOptional,
    const aclTensor *out,
    aclOpExecutor *executor)
{
    L0_DFX(ChunkLocalCumsum, g, cuSeqlensOptional, chunkIndicesOutOptional, chunkSize, reverse, scale, headFirst,
           outputDtypeOptional, out);

    const aclTensor *actualCuSeqlens = nullptr;
    if (cuSeqlensOptional != nullptr) {
        actualCuSeqlens = executor->ConvertToTensor(cuSeqlensOptional, DataType::DT_INT64);
        if (actualCuSeqlens == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Convert cu_seqlens int array to tensor failed.");
            return nullptr;
        }
        const_cast<aclTensor *>(actualCuSeqlens)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualCuSeqlens)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualCuSeqlens)->SetOriginalFormat(Format::FORMAT_ND);
    }

    const aclTensor *actualChunkIndicesOut = nullptr;
    if (chunkIndicesOutOptional != nullptr) {
        actualChunkIndicesOut = executor->ConvertToTensor(chunkIndicesOutOptional, DataType::DT_INT64);
        if (actualChunkIndicesOut == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Convert chunk_indices_out int array to tensor failed.");
            return nullptr;
        }
        const_cast<aclTensor *>(actualChunkIndicesOut)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualChunkIndicesOut)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualChunkIndicesOut)->SetOriginalFormat(Format::FORMAT_ND);
    }

    std::string outputDtypeStr(outputDtypeOptional ? outputDtypeOptional : "float32");
    float scaleFloat = static_cast<float>(scale);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ChunkLocalCumsum,
        OP_INPUT(g, actualCuSeqlens, actualChunkIndicesOut),
        OP_OUTPUT(out),
        OP_ATTR(chunkSize, reverse, scaleFloat, headFirst, outputDtypeStr));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return nullptr;
    }
    return out;
}

}  // namespace l0op

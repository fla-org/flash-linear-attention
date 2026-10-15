/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#ifndef OP_API_INC_LEVEL0_OP_CHUNK_LOCAL_CUMSUM_H
#define OP_API_INC_LEVEL0_OP_CHUNK_LOCAL_CUMSUM_H

#include <cstdint>

#include "aclnn/aclnn_base.h"
#include "opdev/op_executor.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default")))
aclnnStatus aclnnChunkLocalCumsumGetWorkspaceSize(
    const aclTensor *g,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOutOptional,
    int64_t chunkSize,
    bool reverse,
    double scale,
    bool headFirst,
    char *outputDtypeOptional,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

__attribute__((visibility("default")))
aclnnStatus aclnnChunkLocalCumsum(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

namespace l0op {
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
    aclOpExecutor *executor);
}

#endif

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_local_cumsum_tiling_data.h
 * \brief
 */

#ifndef CHUNK_LOCAL_CUMSUM_TILING_DATA_H
#define CHUNK_LOCAL_CUMSUM_TILING_DATA_H

#include <cstdint>

struct ChunkLocalCumsumTilingData {
    int64_t b;
    int64_t t;
    int64_t h;
    int64_t chunkSize;
    int64_t blockT;
    int64_t numBlocks;
    int64_t seqNum;
    int64_t totalElements;
    int64_t isVarlen;
    int64_t reverse;
    int64_t headFirst;
    int64_t optimizedHeadFirst;
    int64_t varlenSeqTask;
    int64_t enableCumSumFastPath;
    int64_t fastBufferLimit;
    int64_t rebalanceChunkTasks;
    int64_t inputDtype;
    int64_t outputDtype;
    float scale;
};

#endif // CHUNK_LOCAL_CUMSUM_TILING_DATA_H

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
 * \file chunk_local_cumsum.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "adv_api/math/cumsum.h"
#include "chunk_local_cumsum_tiling_data.h"
#include <type_traits>

using namespace AscendC;

namespace {
constexpr int64_t H_TILE_SIZE = 512;
constexpr int64_t UB_ALIGN_BYTES = 32;
constexpr int64_t FLOAT_ALIGN_ELEMS = UB_ALIGN_BYTES / static_cast<int64_t>(sizeof(float));
constexpr int64_t FAST_CHUNK_BUFFER_LIMIT = 160 * 1024;
constexpr int64_t FAST_CHUNK_SCAN_BUFFER_NUM = 2;
constexpr int64_t FAST_HEAD_FIRST_PIPE_BUFFER_NUM = 4;
constexpr int64_t FAST_HEAD_FIRST_CUMSUM_BUFFER_NUM = FAST_HEAD_FIRST_PIPE_BUFFER_NUM + 1;
constexpr int64_t FAST_HEAD_FIRST_MAX_CHUNK_GROUP_SIZE = 8;
constexpr int64_t FAST_HEAD_FIRST_RANGE_GROUPS = 1;
constexpr int64_t FP32_REPEAT_ELEMS = 64;
constexpr int64_t VECTOR_MAX_REPEAT_TIMES = 255;
constexpr int64_t VECTOR_MAX_CALC_ELEMS = FP32_REPEAT_ELEMS * VECTOR_MAX_REPEAT_TIMES;
constexpr int64_t CHUNK_128_SIZE = 128;
constexpr int64_t CHUNK_128_SEGMENT_COUNT = 4;
constexpr int64_t CHUNK_128_SEGMENT_SIZE = CHUNK_128_SIZE / CHUNK_128_SEGMENT_COUNT;
constexpr int64_t CHUNK_128_STRIDE_BLOCKS = CHUNK_128_SIZE / FLOAT_ALIGN_ELEMS;
constexpr int64_t CHUNK_128_MAX_CARRY_COUNT = 64;
constexpr int64_t CHUNK_128_SECOND_OFFSET_BYTES = 256;
constexpr int64_t CHUNK_128_THIRD_OFFSET_BYTES = 512;
constexpr int64_t CHUNK_128_CARRY_OFFSET_BYTES = 768;
constexpr int64_t CHUNK_128_BRCB_OFFSET_BYTES = 1280;
constexpr CumSumConfig CUMSUM_NO_LAST_ROW_CONFIG = {true, false, false};
constexpr int64_t DTYPE_FP32 = 0;
constexpr int64_t DTYPE_FP16 = 1;
constexpr int64_t DTYPE_BF16 = 2;
constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t FAST_BUFFER_NUM = 2;

__aicore__ inline int64_t MinInt64(int64_t a, int64_t b)
{
    return a < b ? a : b;
}

__aicore__ inline int64_t CeilDivInt64(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

__aicore__ inline int64_t AlignUpInt64(int64_t value, int64_t align)
{
    return ((value + align - 1) / align) * align;
}

__aicore__ inline int64_t AlignDownInt64(int64_t value, int64_t align)
{
    return (value / align) * align;
}

__aicore__ inline int64_t GetFastBufferLimit(const ChunkLocalCumsumTilingData *tiling)
{
    if (tiling != nullptr && tiling->fastBufferLimit > 0) {
        return tiling->fastBufferLimit;
    }
    return FAST_CHUNK_BUFFER_LIMIT;
}

__aicore__ inline int64_t GetFastHeadFirstChunkGroupSize(int64_t chunkSize, int64_t head, int64_t fastBufferLimit)
{
    if (chunkSize % FLOAT_ALIGN_ELEMS != 0) {
        return 1;
    }
    int64_t hLen = MinInt64(H_TILE_SIZE, head);
    if (hLen < 1) {
        hLen = 1;
    }
    int64_t groupSize = fastBufferLimit /
                        (FAST_HEAD_FIRST_CUMSUM_BUFFER_NUM * hLen * chunkSize *
                         static_cast<int64_t>(sizeof(float)));
    int64_t maxGroupSize = 2;
    if (chunkSize <= 16) {
        maxGroupSize = FAST_HEAD_FIRST_MAX_CHUNK_GROUP_SIZE;
    } else if (chunkSize <= 32) {
        maxGroupSize = 4;
    }
    return MinInt64(maxGroupSize, groupSize < 1 ? 1 : groupSize);
}

__aicore__ inline int64_t GetCumSumWorkspaceBytes(int64_t inner)
{
    constexpr int64_t cumsumWorkspaceRows = 16;
    constexpr int64_t cumsumWorkspacePlanes = 2;
    return AlignUpInt64(cumsumWorkspaceRows * inner * cumsumWorkspacePlanes *
                            static_cast<int64_t>(sizeof(float)),
                        UB_ALIGN_BYTES);
}

template <typename GType, typename OType, int64_t SEGMENTED_CHUNK_SIZE = 0>
class ChunkLocalCumsumKernel {
public:
    __aicore__ inline ChunkLocalCumsumKernel() = default;

    __aicore__ inline void Init(GM_ADDR g, GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR out,
                                const ChunkLocalCumsumTilingData *tiling)
    {
        tiling_ = tiling;
        gGm_.SetGlobalBuffer(reinterpret_cast<__gm__ GType *>(g), tiling_->totalElements);
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ OType *>(out), tiling_->totalElements);
        if (tiling_->isVarlen != 0) {
            cuSeqlensGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cuSeqlens));
            chunkIndicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(chunkIndices), tiling_->numBlocks * 2);
        }
        enableCumSumFastPath_ = tiling_->enableCumSumFastPath != 0;
        optimizedHeadFirst_ = (tiling_->optimizedHeadFirst != 0) && enableCumSumFastPath_;
        headFirstPipeline_ = optimizedHeadFirst_;
        const int64_t fastBufferLimit = GetFastBufferLimit(tiling_);
        fastChunkGroupSize_ = headFirstPipeline_
                                  ? GetFastHeadFirstChunkGroupSize(tiling_->chunkSize, tiling_->h, fastBufferLimit)
                                  : 1;
        rebalanceChunkTasks_ = tiling_->rebalanceChunkTasks != 0;
        int64_t fastBufferNum = headFirstPipeline_ ? FAST_HEAD_FIRST_CUMSUM_BUFFER_NUM : FAST_CHUNK_SCAN_BUFFER_NUM;
        int64_t maxFastHLen = fastBufferLimit /
                              (fastBufferNum * fastChunkGroupSize_ * tiling_->chunkSize *
                               static_cast<int64_t>(sizeof(float)));
        if (optimizedHeadFirst_) {
            fastHTileSize_ = MinInt64(MinInt64(H_TILE_SIZE, tiling_->h), maxFastHLen);
            if (fastHTileSize_ < 1) {
                fastHTileSize_ = 1;
            }
        } else {
            fastHTileSize_ = AlignDownInt64(MinInt64(MinInt64(H_TILE_SIZE, tiling_->h), maxFastHLen),
                                            FLOAT_ALIGN_ELEMS);
            if (tiling_->h == 1) {
                fastHTileSize_ = 1;
            }
        }
        // The log-step scan needs two chunk buffers; shrink only the fast H tile, not the whole fast path.
        bool alignedFastPath = optimizedHeadFirst_ || (((tiling_->h & (FLOAT_ALIGN_ELEMS - 1)) == 0) &&
                                                       (fastHTileSize_ >= FLOAT_ALIGN_ELEMS));
        bool scalarHeadFastPath = enableCumSumFastPath_ && !optimizedHeadFirst_ && (tiling_->headFirst != 0) &&
                                  (tiling_->h == 1) && (tiling_->reverse == 0);
        cumsumFastPath_ = optimizedHeadFirst_ || scalarHeadFastPath;
        chunkFastPath_ = std::is_same<GType, float>::value && std::is_same<OType, float>::value &&
                         (alignedFastPath || scalarHeadFastPath);
        if (chunkFastPath_) {
            int64_t chunkElems = tiling_->chunkSize * fastHTileSize_;
            if (optimizedHeadFirst_) {
                chunkElems = AlignUpInt64(tiling_->chunkSize * fastChunkGroupSize_, FLOAT_ALIGN_ELEMS) *
                             fastHTileSize_;
            }
            if (scalarHeadFastPath) {
                chunkElems = AlignUpInt64(tiling_->chunkSize, FLOAT_ALIGN_ELEMS);
            }
            int64_t chunkBufferBytes = AlignUpInt64(chunkElems * static_cast<int64_t>(sizeof(float)), UB_ALIGN_BYTES);
            if (headFirstPipeline_) {
                pipe_.InitBuffer(headFirstChunkQueue_, FAST_BUFFER_NUM, chunkBufferBytes);
                pipe_.InitBuffer(headFirstOutQueue_, FAST_BUFFER_NUM, chunkBufferBytes);
            } else {
                pipe_.InitBuffer(chunkQueue_, BUFFER_NUM, chunkBufferBytes);
                pipe_.InitBuffer(scanBuf_, chunkBufferBytes);
            }
            if (cumsumFastPath_) {
                int64_t maxCumSumInner = AlignUpInt64(tiling_->chunkSize * fastChunkGroupSize_, FLOAT_ALIGN_ELEMS);
                pipe_.InitBuffer(cumsumLastRowBuf_,
                                 AlignUpInt64(maxCumSumInner * static_cast<int64_t>(sizeof(float)), UB_ALIGN_BYTES));
                pipe_.InitBuffer(cumsumWorkspaceBuf_, GetCumSumWorkspaceBytes(maxCumSumInner));
            }
        } else {
            int64_t rowBufferBytes = AlignUpInt64(H_TILE_SIZE * static_cast<int64_t>(sizeof(float)), UB_ALIGN_BYTES);
            pipe_.InitBuffer(rowQueue_, BUFFER_NUM, rowBufferBytes);
            pipe_.InitBuffer(outQueue_, BUFFER_NUM, rowBufferBytes);
            pipe_.InitBuffer(accBuf_, rowBufferBytes);
            if constexpr (!std::is_same<GType, float>::value) {
                int64_t inputRowBufferBytes =
                    AlignUpInt64(H_TILE_SIZE * static_cast<int64_t>(sizeof(GType)), UB_ALIGN_BYTES);
                pipe_.InitBuffer(inputRowQueue_, BUFFER_NUM, inputRowBufferBytes);
            }
            if constexpr (!std::is_same<OType, float>::value) {
                int64_t outputRowBufferBytes =
                    AlignUpInt64(H_TILE_SIZE * static_cast<int64_t>(sizeof(OType)), UB_ALIGN_BYTES);
                pipe_.InitBuffer(outCastQueue_, BUFFER_NUM, outputRowBufferBytes);
            }
        }
        vToMte3Event_ = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        mte3ToVEvent_ = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    }

    __aicore__ inline void Process()
    {
        if (tiling_->isVarlen != 0) {
            ProcessVarlen();
        } else {
            ProcessFixed();
        }
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(vToMte3Event_);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(mte3ToVEvent_);
    }

private:
    __aicore__ inline void WaitVToMte3()
    {
        SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
    }

    __aicore__ inline void WaitMte3ToV()
    {
        SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
    }

    __aicore__ inline int64_t GetDenseBaseOffset(int64_t bIdx)
    {
        return bIdx * tiling_->t * tiling_->h;
    }

    __aicore__ inline int64_t GetVarlenBaseOffset(int64_t outerIdx, int64_t bos)
    {
        if (optimizedHeadFirst_) {
            return outerIdx * tiling_->h * tiling_->t + bos;
        }
        return outerIdx * tiling_->t * tiling_->h + bos * tiling_->h;
    }

    __aicore__ inline int64_t GetChunkEnd(int64_t chunkStart, int64_t tEnd)
    {
        return MinInt64(chunkStart + tiling_->chunkSize, tEnd);
    }

    __aicore__ inline LocalTensor<float> LoadRowToUb(int64_t gmOffset, int64_t elementCount)
    {
        LocalTensor<float> rowLocal = rowQueue_.AllocTensor<float>();
        if constexpr (std::is_same<GType, float>::value) {
            if ((elementCount & 7) == 0) {
                DataCopy(rowLocal, gGm_[gmOffset], static_cast<uint32_t>(elementCount));
            } else {
                DataCopyExtParams copyParams{1, static_cast<uint32_t>(elementCount * static_cast<int64_t>(sizeof(float))),
                                             0, 0, 0};
                DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
                DataCopyPad(rowLocal, gGm_[gmOffset], copyParams, padParams);
            }
        } else {
            LocalTensor<GType> inputLocal = inputRowQueue_.AllocTensor<GType>();
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(elementCount * static_cast<int64_t>(sizeof(GType))),
                                         0, 0, 0};
            DataCopyPadExtParams<GType> padParams{false, 0, 0, 0};
            DataCopyPad(inputLocal, gGm_[gmOffset], copyParams, padParams);
            inputRowQueue_.EnQue(inputLocal);
            inputLocal = inputRowQueue_.DeQue<GType>();
            Cast(rowLocal, inputLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(elementCount));
            PipeBarrier<PIPE_V>();
            inputRowQueue_.FreeTensor(inputLocal);
        }
        rowQueue_.EnQue(rowLocal);
        return rowQueue_.DeQue<float>();
    }

    __aicore__ inline void CopyUbToGm(int64_t gmOffset, LocalTensor<float> srcLocal, int64_t elementCount)
    {
        if constexpr (std::is_same<OType, float>::value) {
            if ((elementCount & 7) == 0) {
                DataCopy(outGm_[gmOffset], srcLocal, static_cast<uint32_t>(elementCount));
            } else {
                DataCopyExtParams copyParams{1, static_cast<uint32_t>(elementCount * static_cast<int64_t>(sizeof(float))),
                                             0, 0, 0};
                DataCopyPad(outGm_[gmOffset], srcLocal, copyParams);
            }
        } else {
            LocalTensor<OType> outLocal = outCastQueue_.AllocTensor<OType>();
            Cast(outLocal, srcLocal, RoundMode::CAST_RINT, static_cast<uint32_t>(elementCount));
            PipeBarrier<PIPE_V>();
            outCastQueue_.EnQue(outLocal);
            outLocal = outCastQueue_.DeQue<OType>();
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(elementCount * static_cast<int64_t>(sizeof(OType))),
                                         0, 0, 0};
            DataCopyPad(outGm_[gmOffset], outLocal, copyParams);
            outCastQueue_.FreeTensor(outLocal);
        }
    }

    __aicore__ inline LocalTensor<float> LoadChunkToUb(int64_t gmOffset, int64_t chunkLen, int64_t hLen,
                                                       int64_t hLenAlign)
    {
        LocalTensor<float> chunkLocal = chunkQueue_.AllocTensor<float>();
        int64_t totalElems = chunkLen * hLen;
        if (hLen == tiling_->h && ((gmOffset & (FLOAT_ALIGN_ELEMS - 1)) == 0) &&
            ((totalElems & (FLOAT_ALIGN_ELEMS - 1)) == 0)) {
            DataCopy(chunkLocal, gGm_[gmOffset], static_cast<uint32_t>(totalElems));
        } else {
            DataCopyExtParams copyParams{static_cast<uint16_t>(chunkLen),
                                         static_cast<uint32_t>(hLen * static_cast<int64_t>(sizeof(float))),
                                         static_cast<uint32_t>((tiling_->h - hLen) *
                                                              static_cast<int64_t>(sizeof(float))),
                                         0,
                                         0};
            DataCopyPadExtParams<float> padParams{hLenAlign != hLen, 0,
                                                  static_cast<uint8_t>(hLenAlign - hLen), 0.0f};
            DataCopyPad(chunkLocal, gGm_[gmOffset], copyParams, padParams);
        }
        chunkQueue_.EnQue(chunkLocal);
        return chunkQueue_.DeQue<float>();
    }

    __aicore__ inline void CopyChunkUbToGm(int64_t gmOffset, LocalTensor<float> chunkLocal, int64_t chunkLen,
                                           int64_t hLen, int64_t hLenAlign)
    {
        int64_t totalElems = chunkLen * hLen;
        if (hLen == tiling_->h && ((gmOffset & (FLOAT_ALIGN_ELEMS - 1)) == 0) &&
            ((totalElems & (FLOAT_ALIGN_ELEMS - 1)) == 0)) {
            DataCopy(outGm_[gmOffset], chunkLocal, static_cast<uint32_t>(totalElems));
        } else {
            DataCopyExtParams copyParams{static_cast<uint16_t>(chunkLen),
                                         static_cast<uint32_t>(hLen * static_cast<int64_t>(sizeof(float))),
                                         static_cast<uint32_t>((hLenAlign - hLen) *
                                                              static_cast<int64_t>(sizeof(float)) / UB_ALIGN_BYTES),
                                         static_cast<uint32_t>((tiling_->h - hLen) *
                                                              static_cast<int64_t>(sizeof(float))),
                                         0};
            DataCopyPad(outGm_[gmOffset], chunkLocal, copyParams);
        }
    }

    __aicore__ inline LocalTensor<float> LoadHeadFirstChunkToUb(int64_t gmOffset, int64_t chunkLen,
                                                                int64_t hLen, int64_t chunkLenAlign)
    {
        LocalTensor<float> chunkLocal = chunkQueue_.AllocTensor<float>();
        DataCopyExtParams copyParams{static_cast<uint16_t>(hLen),
                                     static_cast<uint32_t>(chunkLen * static_cast<int64_t>(sizeof(float))),
                                     static_cast<uint32_t>((tiling_->t - chunkLen) *
                                                          static_cast<int64_t>(sizeof(float))),
                                     0,
                                     0};
        DataCopyPadExtParams<float> padParams{chunkLenAlign != chunkLen,
                                              0,
                                              static_cast<uint8_t>(chunkLenAlign - chunkLen),
                                              0.0f};
        DataCopyPad(chunkLocal, gGm_[gmOffset], copyParams, padParams);
        chunkQueue_.EnQue(chunkLocal);
        return chunkQueue_.DeQue<float>();
    }

    __aicore__ inline void CopyHeadFirstChunkUbToGm(int64_t gmOffset, LocalTensor<float> chunkLocal,
                                                    int64_t chunkLen, int64_t hLen, int64_t chunkLenAlign)
    {
        DataCopyExtParams copyParams{static_cast<uint16_t>(hLen),
                                     static_cast<uint32_t>(chunkLen * static_cast<int64_t>(sizeof(float))),
                                     static_cast<uint32_t>((chunkLenAlign - chunkLen) *
                                                          static_cast<int64_t>(sizeof(float)) / UB_ALIGN_BYTES),
                                     static_cast<uint32_t>((tiling_->t - chunkLen) *
                                                          static_cast<int64_t>(sizeof(float))),
                                     0};
        DataCopyPad(outGm_[gmOffset], chunkLocal, copyParams);
    }

    __aicore__ inline int64_t GetHeadFirstGroupEnd(int64_t chunkStart, int64_t tEnd)
    {
        int64_t remaining = tEnd - chunkStart;
        int64_t fullChunks = remaining / tiling_->chunkSize;
        int64_t groupChunks = MinInt64(fastChunkGroupSize_, fullChunks);
        if (groupChunks > 1) {
            return chunkStart + groupChunks * tiling_->chunkSize;
        }
        return MinInt64(chunkStart + tiling_->chunkSize, tEnd);
    }

    __aicore__ inline void CopyInHeadFirstGroup(int64_t gmOffset, int64_t groupLen, int64_t hLen,
                                                int64_t groupLenAlign)
    {
        LocalTensor<float> groupLocal = headFirstChunkQueue_.AllocTensor<float>();
        DataCopyExtParams copyParams{static_cast<uint16_t>(hLen),
                                     static_cast<uint32_t>(groupLen * static_cast<int64_t>(sizeof(float))),
                                     static_cast<uint32_t>((tiling_->t - groupLen) *
                                                          static_cast<int64_t>(sizeof(float))),
                                     0,
                                     0};
        DataCopyPadExtParams<float> padParams{groupLenAlign != groupLen,
                                              0,
                                              static_cast<uint8_t>(groupLenAlign - groupLen),
                                              0.0f};
        DataCopyPad(groupLocal, gGm_[gmOffset], copyParams, padParams);
        headFirstChunkQueue_.EnQue(groupLocal);
    }

    __aicore__ inline void ComputeHeadFirstGroup(int64_t groupLen, int64_t hLen, int64_t groupLenAlign)
    {
        LocalTensor<float> inLocal = headFirstChunkQueue_.DeQue<float>();
        LocalTensor<float> outLocal = headFirstOutQueue_.AllocTensor<float>();
        int64_t groupChunks = groupLen / tiling_->chunkSize;
        if constexpr (SEGMENTED_CHUNK_SIZE == CHUNK_128_SIZE) {
            int64_t carryCount = hLen * groupChunks;
            bool useChunk128Split = groupLenAlign == groupLen && groupLen % CHUNK_128_SIZE == 0 &&
                                    carryCount > 0 && carryCount <= CHUNK_128_MAX_CARRY_COUNT &&
                                    (carryCount & (FLOAT_ALIGN_ELEMS - 1)) == 0;
            if (useChunk128Split) {
                ComputeChunk128SegmentedAndScale(outLocal, inLocal, carryCount, hLen * groupLenAlign);
                headFirstOutQueue_.EnQue(outLocal);
                headFirstChunkQueue_.FreeTensor(inLocal);
                return;
            }
        }
        if (groupLen > tiling_->chunkSize && groupLenAlign == groupLen) {
            ComputeCumSumAndScale(outLocal, inLocal, hLen * groupChunks, tiling_->chunkSize, hLen * groupLenAlign);
        } else {
            ComputeCumSumAndScale(outLocal, inLocal, hLen, groupLenAlign, hLen * groupLenAlign);
        }
        headFirstOutQueue_.EnQue(outLocal);
        headFirstChunkQueue_.FreeTensor(inLocal);
    }

    __aicore__ inline void CopyOutHeadFirstGroup(int64_t gmOffset, int64_t groupLen, int64_t hLen,
                                                 int64_t groupLenAlign)
    {
        LocalTensor<float> outLocal = headFirstOutQueue_.DeQue<float>();
        DataCopyExtParams copyParams{static_cast<uint16_t>(hLen),
                                     static_cast<uint32_t>(groupLen * static_cast<int64_t>(sizeof(float))),
                                     static_cast<uint32_t>((groupLenAlign - groupLen) *
                                                          static_cast<int64_t>(sizeof(float)) / UB_ALIGN_BYTES),
                                     static_cast<uint32_t>((tiling_->t - groupLen) *
                                                          static_cast<int64_t>(sizeof(float))),
                                     0};
        DataCopyPad(outGm_[gmOffset], outLocal, copyParams);
        headFirstOutQueue_.FreeTensor(outLocal);
    }

    __aicore__ inline void ProcessHeadFirstFastRange(int64_t baseOffset, int64_t tStart, int64_t tEnd,
                                                     int64_t hStart, int64_t hLen)
    {
        if (tStart >= tEnd) {
            return;
        }
        int64_t curStart = tStart;
        int64_t curEnd = GetHeadFirstGroupEnd(curStart, tEnd);
        int64_t curLen = curEnd - curStart;
        int64_t curLenAlign = AlignUpInt64(curLen, FLOAT_ALIGN_ELEMS);
        CopyInHeadFirstGroup(baseOffset + hStart * tiling_->t + curStart, curLen, hLen, curLenAlign);

        int64_t nextStart = curEnd;
        int64_t nextEnd = tEnd;
        if (nextStart < tEnd) {
            nextEnd = GetHeadFirstGroupEnd(nextStart, tEnd);
            int64_t nextLen = nextEnd - nextStart;
            int64_t nextLenAlign = AlignUpInt64(nextLen, FLOAT_ALIGN_ELEMS);
            CopyInHeadFirstGroup(baseOffset + hStart * tiling_->t + nextStart, nextLen, hLen, nextLenAlign);
        }

        bool hasPendingOut = false;
        int64_t pendingStart = 0;
        int64_t pendingLen = 0;
        int64_t pendingLenAlign = 0;
        while (curStart < tEnd) {
            if (hasPendingOut) {
                CopyOutHeadFirstGroup(baseOffset + hStart * tiling_->t + pendingStart, pendingLen, hLen,
                                      pendingLenAlign);
            }
            ComputeHeadFirstGroup(curLen, hLen, curLenAlign);
            pendingStart = curStart;
            pendingLen = curLen;
            pendingLenAlign = curLenAlign;
            hasPendingOut = true;

            curStart = nextStart;
            curEnd = nextEnd;
            curLen = curEnd - curStart;
            curLenAlign = AlignUpInt64(curLen, FLOAT_ALIGN_ELEMS);
            nextStart = curEnd;
            if (nextStart < tEnd) {
                nextEnd = GetHeadFirstGroupEnd(nextStart, tEnd);
                int64_t nextLen = nextEnd - nextStart;
                int64_t nextLenAlign = AlignUpInt64(nextLen, FLOAT_ALIGN_ELEMS);
                CopyInHeadFirstGroup(baseOffset + hStart * tiling_->t + nextStart, nextLen, hLen, nextLenAlign);
            }
        }
        if (hasPendingOut) {
            CopyOutHeadFirstGroup(baseOffset + hStart * tiling_->t + pendingStart, pendingLen, hLen, pendingLenAlign);
        }
    }

    __aicore__ inline LocalTensor<float> LoadScalarHeadChunkToUb(int64_t gmOffset, int64_t chunkLen,
                                                                 int64_t chunkLenAlign)
    {
        LocalTensor<float> chunkLocal = chunkQueue_.AllocTensor<float>();
        if (((gmOffset & (FLOAT_ALIGN_ELEMS - 1)) == 0) && ((chunkLen & (FLOAT_ALIGN_ELEMS - 1)) == 0)) {
            DataCopy(chunkLocal, gGm_[gmOffset], static_cast<uint32_t>(chunkLen));
        } else {
            DataCopyExtParams copyParams{1,
                                         static_cast<uint32_t>(chunkLen * static_cast<int64_t>(sizeof(float))),
                                         0,
                                         0,
                                         0};
            DataCopyPadExtParams<float> padParams{chunkLenAlign != chunkLen,
                                                  0,
                                                  static_cast<uint8_t>(chunkLenAlign - chunkLen),
                                                  0.0f};
            DataCopyPad(chunkLocal, gGm_[gmOffset], copyParams, padParams);
        }
        chunkQueue_.EnQue(chunkLocal);
        return chunkQueue_.DeQue<float>();
    }

    __aicore__ inline void CopyScalarHeadChunkUbToGm(int64_t gmOffset, LocalTensor<float> chunkLocal,
                                                     int64_t chunkLen)
    {
        if (((gmOffset & (FLOAT_ALIGN_ELEMS - 1)) == 0) && ((chunkLen & (FLOAT_ALIGN_ELEMS - 1)) == 0)) {
            DataCopy(outGm_[gmOffset], chunkLocal, static_cast<uint32_t>(chunkLen));
        } else {
            DataCopyExtParams copyParams{1,
                                         static_cast<uint32_t>(chunkLen * static_cast<int64_t>(sizeof(float))),
                                         0,
                                         0,
                                         0};
            DataCopyPad(outGm_[gmOffset], chunkLocal, copyParams);
        }
    }

    __aicore__ inline void AddBatched(LocalTensor<float> dstLocal, LocalTensor<float> src0Local,
                                      LocalTensor<float> src1Local, int64_t elementCount)
    {
        for (int64_t offset = 0; offset < elementCount; offset += VECTOR_MAX_CALC_ELEMS) {
            int64_t curCount = MinInt64(VECTOR_MAX_CALC_ELEMS, elementCount - offset);
            Add(dstLocal[offset], src0Local[offset], src1Local[offset], static_cast<uint32_t>(curCount));
        }
    }

    __aicore__ inline void AddsBatched(LocalTensor<float> dstLocal, LocalTensor<float> srcLocal, float scalar,
                                       int64_t elementCount)
    {
        for (int64_t offset = 0; offset < elementCount; offset += VECTOR_MAX_CALC_ELEMS) {
            int64_t curCount = MinInt64(VECTOR_MAX_CALC_ELEMS, elementCount - offset);
            Adds(dstLocal[offset], srcLocal[offset], scalar, static_cast<uint32_t>(curCount));
        }
    }

    __aicore__ inline void MulsBatched(LocalTensor<float> dstLocal, LocalTensor<float> srcLocal, float scalar,
                                       int64_t elementCount)
    {
        for (int64_t offset = 0; offset < elementCount; offset += VECTOR_MAX_CALC_ELEMS) {
            int64_t curCount = MinInt64(VECTOR_MAX_CALC_ELEMS, elementCount - offset);
            Muls(dstLocal[offset], srcLocal[offset], scalar, static_cast<uint32_t>(curCount));
        }
    }

    __aicore__ inline void ApplyScaleInplace(LocalTensor<float> local, int64_t elementCount)
    {
        if (tiling_->scale != 1.0f) {
            MulsBatched(local, local, tiling_->scale, elementCount);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ComputeCumSumAndScale(LocalTensor<float> dstLocal, LocalTensor<float> srcLocal,
                                                 int64_t outer, int64_t inner, int64_t elementCount)
    {
        CumSumInfo cumSumInfo{static_cast<uint32_t>(outer), static_cast<uint32_t>(inner)};
        LocalTensor<float> lastRowLocal = cumsumLastRowBuf_.Get<float>();
        LocalTensor<uint8_t> workspaceLocal = cumsumWorkspaceBuf_.Get<uint8_t>();
        CumSum<float>(dstLocal, lastRowLocal, srcLocal, workspaceLocal, cumSumInfo);
        PipeBarrier<PIPE_V>();
        ApplyScaleInplace(dstLocal, elementCount);
    }

    __aicore__ inline void ComputeChunk128SegmentedAndScale(LocalTensor<float> dstLocal,
                                                            LocalTensor<float> srcLocal, int64_t carryCount,
                                                            int64_t elementCount)
    {
        CumSumInfo cumSumInfo{static_cast<uint32_t>(carryCount * CHUNK_128_SEGMENT_COUNT),
                              static_cast<uint32_t>(CHUNK_128_SEGMENT_SIZE)};
        LocalTensor<float> lastRowLocal = cumsumLastRowBuf_.Get<float>();
        LocalTensor<uint8_t> workspaceLocal = cumsumWorkspaceBuf_.Get<uint8_t>();
        CumSum<float, CUMSUM_NO_LAST_ROW_CONFIG>(dstLocal, lastRowLocal, srcLocal, workspaceLocal, cumSumInfo);
        PipeBarrier<PIPE_V>();

        LocalTensor<int32_t> carryOffsets = workspaceLocal.ReinterpretCast<int32_t>();
        LocalTensor<int32_t> secondOffsets =
            workspaceLocal[CHUNK_128_SECOND_OFFSET_BYTES].ReinterpretCast<int32_t>();
        LocalTensor<int32_t> thirdOffsets =
            workspaceLocal[CHUNK_128_THIRD_OFFSET_BYTES].ReinterpretCast<int32_t>();
        LocalTensor<float> carryLocal = workspaceLocal[CHUNK_128_CARRY_OFFSET_BYTES].ReinterpretCast<float>();
        LocalTensor<float> carryBrcb = workspaceLocal[CHUNK_128_BRCB_OFFSET_BYTES].ReinterpretCast<float>();
        CreateVecIndex(carryOffsets, 0, static_cast<uint32_t>(carryCount));
        PipeBarrier<PIPE_V>();
        Muls(carryOffsets, carryOffsets,
             static_cast<int32_t>(CHUNK_128_SIZE * static_cast<int64_t>(sizeof(float))),
             static_cast<uint32_t>(carryCount));
        PipeBarrier<PIPE_V>();
        LocalTensor<uint32_t> gatherOffsets = carryOffsets.ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> secondGatherOffsets = secondOffsets.ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> thirdGatherOffsets = thirdOffsets.ReinterpretCast<uint32_t>();
        BinaryRepeatParams repeatParams{1, 1, 0,
                                        static_cast<uint8_t>(CHUNK_128_STRIDE_BLOCKS),
                                        static_cast<uint8_t>(CHUNK_128_STRIDE_BLOCKS), 1};

        // First merge (0 -> 1, 2 -> 3), then add the completed segment-1 carry to segments 2 and 3.
        Adds(secondOffsets, carryOffsets,
             static_cast<int32_t>((3 * CHUNK_128_SEGMENT_SIZE - 1) * static_cast<int64_t>(sizeof(float))),
             static_cast<uint32_t>(carryCount));
        Adds(thirdOffsets, carryOffsets,
             static_cast<int32_t>((2 * CHUNK_128_SEGMENT_SIZE - 1) * static_cast<int64_t>(sizeof(float))),
             static_cast<uint32_t>(carryCount));
        Adds(carryOffsets, carryOffsets,
             static_cast<int32_t>((CHUNK_128_SEGMENT_SIZE - 1) * static_cast<int64_t>(sizeof(float))),
             static_cast<uint32_t>(carryCount));
        PipeBarrier<PIPE_V>();
        Gather(carryLocal, dstLocal, gatherOffsets, 0, static_cast<uint32_t>(carryCount));
        Gather(carryLocal[carryCount], dstLocal, secondGatherOffsets, 0, static_cast<uint32_t>(carryCount));
        PipeBarrier<PIPE_V>();
        Brcb(carryBrcb, carryLocal, static_cast<uint8_t>(2 * carryCount / FLOAT_ALIGN_ELEMS), {1, 8});
        PipeBarrier<PIPE_V>();
        Add(dstLocal[CHUNK_128_SEGMENT_SIZE], dstLocal[CHUNK_128_SEGMENT_SIZE], carryBrcb,
            static_cast<uint64_t>(CHUNK_128_SEGMENT_SIZE), static_cast<uint8_t>(carryCount), repeatParams);
        Add(dstLocal[3 * CHUNK_128_SEGMENT_SIZE], dstLocal[3 * CHUNK_128_SEGMENT_SIZE],
            carryBrcb[carryCount * FLOAT_ALIGN_ELEMS], static_cast<uint64_t>(CHUNK_128_SEGMENT_SIZE),
            static_cast<uint8_t>(carryCount), repeatParams);
        PipeBarrier<PIPE_V>();

        Gather(carryLocal, dstLocal, thirdGatherOffsets, 0, static_cast<uint32_t>(carryCount));
        PipeBarrier<PIPE_V>();
        Brcb(carryBrcb, carryLocal, static_cast<uint8_t>(carryCount / FLOAT_ALIGN_ELEMS), {1, 8});
        PipeBarrier<PIPE_V>();
        Add(dstLocal[2 * CHUNK_128_SEGMENT_SIZE], dstLocal[2 * CHUNK_128_SEGMENT_SIZE], carryBrcb,
            static_cast<uint64_t>(2 * CHUNK_128_SEGMENT_SIZE), static_cast<uint8_t>(carryCount), repeatParams);
        PipeBarrier<PIPE_V>();
        ApplyScaleInplace(dstLocal, elementCount);
    }

    __aicore__ inline void ComputeForwardScanStep(LocalTensor<float> dstLocal, LocalTensor<float> srcLocal,
                                                  int64_t chunkLen, int64_t hLenAlign, int64_t step)
    {
        AddsBatched(dstLocal, srcLocal, 0.0f, step * hLenAlign);
        AddBatched(dstLocal[step * hLenAlign], srcLocal[step * hLenAlign], srcLocal,
                   (chunkLen - step) * hLenAlign);
    }

    __aicore__ inline void ComputeReverseScanStep(LocalTensor<float> dstLocal, LocalTensor<float> srcLocal,
                                                  int64_t chunkLen, int64_t hLenAlign, int64_t step)
    {
        int64_t activeRows = chunkLen - step;
        AddBatched(dstLocal, srcLocal, srcLocal[step * hLenAlign], activeRows * hLenAlign);
        AddsBatched(dstLocal[activeRows * hLenAlign], srcLocal[activeRows * hLenAlign], 0.0f, step * hLenAlign);
    }

    // Ping-pong Hillis-Steele scan; each step reads from one UB buffer and writes the other.
    __aicore__ inline bool ComputeChunkPrefixInUb(LocalTensor<float> chunkLocal, int64_t chunkLen, int64_t hLenAlign)
    {
        LocalTensor<float> scanLocal = scanBuf_.Get<float>();
        bool nextSrcIsChunk = true;
        for (int64_t step = 1; step < chunkLen; step <<= 1) {
            LocalTensor<float> srcLocal = nextSrcIsChunk ? chunkLocal : scanLocal;
            LocalTensor<float> dstLocal = nextSrcIsChunk ? scanLocal : chunkLocal;
            if (tiling_->reverse != 0) {
                ComputeReverseScanStep(dstLocal, srcLocal, chunkLen, hLenAlign, step);
            } else {
                ComputeForwardScanStep(dstLocal, srcLocal, chunkLen, hLenAlign, step);
            }
            PipeBarrier<PIPE_V>();
            nextSrcIsChunk = !nextSrcIsChunk;
        }

        LocalTensor<float> resultLocal = nextSrcIsChunk ? chunkLocal : scanLocal;
        ApplyScaleInplace(resultLocal, chunkLen * hLenAlign);
        return nextSrcIsChunk;
    }

    __aicore__ inline void ProcessSequenceChunkFast(int64_t baseOffset, int64_t chunkStart, int64_t chunkEnd,
                                                    int64_t hStart, int64_t hLen)
    {
        int64_t chunkLen = chunkEnd - chunkStart;
        if (tiling_->h == 1 && hLen == 1) {
            int64_t rowOffset = baseOffset + chunkStart;
            int64_t chunkLenAlign = AlignUpInt64(chunkLen, FLOAT_ALIGN_ELEMS);
            LocalTensor<float> chunkLocal = LoadScalarHeadChunkToUb(rowOffset, chunkLen, chunkLenAlign);
            LocalTensor<float> outLocal = scanBuf_.Get<float>();
            ComputeCumSumAndScale(outLocal, chunkLocal, 1, chunkLenAlign, chunkLenAlign);
            WaitVToMte3();
            CopyScalarHeadChunkUbToGm(rowOffset, outLocal, chunkLen);
            WaitMte3ToV();
            chunkQueue_.FreeTensor(chunkLocal);
            return;
        }
        int64_t hLenAlign = AlignUpInt64(hLen, FLOAT_ALIGN_ELEMS);
        int64_t rowOffset = baseOffset + chunkStart * tiling_->h + hStart;
        LocalTensor<float> chunkLocal = LoadChunkToUb(rowOffset, chunkLen, hLen, hLenAlign);
        bool resultInChunk = ComputeChunkPrefixInUb(chunkLocal, chunkLen, hLenAlign);
        LocalTensor<float> outLocal = resultInChunk ? chunkLocal : scanBuf_.Get<float>();
        WaitVToMte3();
        CopyChunkUbToGm(rowOffset, outLocal, chunkLen, hLen, hLenAlign);
        WaitMte3ToV();
        chunkQueue_.FreeTensor(chunkLocal);
    }

    __aicore__ inline void StoreAccumVector(int64_t outOffset, LocalTensor<float> accLocal, int64_t hLen)
    {
        if (tiling_->scale == 1.0f) {
            WaitVToMte3();
            CopyUbToGm(outOffset, accLocal, hLen);
            WaitMte3ToV();
            return;
        }

        LocalTensor<float> outLocal = outQueue_.AllocTensor<float>();
        Muls(outLocal, accLocal, tiling_->scale, static_cast<uint32_t>(hLen));
        outQueue_.EnQue(outLocal);
        outLocal = outQueue_.DeQue<float>();
        CopyUbToGm(outOffset, outLocal, hLen);
        outQueue_.FreeTensor(outLocal);
    }

    __aicore__ inline void ProcessSequenceChunk(int64_t baseOffset, int64_t chunkStart, int64_t chunkEnd,
                                                int64_t hStart, int64_t hLen)
    {
        LocalTensor<float> accLocal = accBuf_.Get<float>();
        if (tiling_->reverse != 0) {
            for (int64_t localT = chunkEnd - 1; localT >= chunkStart; --localT) {
                int64_t rowOffset = baseOffset + localT * tiling_->h + hStart;
                LocalTensor<float> rowLocal = LoadRowToUb(rowOffset, hLen);
                if (localT == chunkEnd - 1) {
                    Adds(accLocal, rowLocal, 0.0f, static_cast<uint32_t>(hLen));
                } else {
                    Add(accLocal, accLocal, rowLocal, static_cast<uint32_t>(hLen));
                }
                rowQueue_.FreeTensor(rowLocal);
                PipeBarrier<PIPE_V>();
                StoreAccumVector(rowOffset, accLocal, hLen);
            }
        } else {
            for (int64_t localT = chunkStart; localT < chunkEnd; ++localT) {
                int64_t rowOffset = baseOffset + localT * tiling_->h + hStart;
                LocalTensor<float> rowLocal = LoadRowToUb(rowOffset, hLen);
                if (localT == chunkStart) {
                    Adds(accLocal, rowLocal, 0.0f, static_cast<uint32_t>(hLen));
                } else {
                    Add(accLocal, accLocal, rowLocal, static_cast<uint32_t>(hLen));
                }
                rowQueue_.FreeTensor(rowLocal);
                PipeBarrier<PIPE_V>();
                StoreAccumVector(rowOffset, accLocal, hLen);
            }
        }
    }

    __aicore__ inline void ProcessSequenceChunkByPath(int64_t baseOffset, int64_t chunkStart, int64_t chunkEnd,
                                                      int64_t hStart, int64_t hLen)
    {
        if constexpr (std::is_same<GType, float>::value && std::is_same<OType, float>::value) {
            if (chunkFastPath_) {
                ProcessSequenceChunkFast(baseOffset, chunkStart, chunkEnd, hStart, hLen);
            } else {
                ProcessSequenceChunk(baseOffset, chunkStart, chunkEnd, hStart, hLen);
            }
        } else {
            ProcessSequenceChunk(baseOffset, chunkStart, chunkEnd, hStart, hLen);
        }
    }

    __aicore__ inline void ProcessChunkRange(int64_t baseOffset, int64_t tStart, int64_t tEnd,
                                             int64_t hStart, int64_t hLen)
    {
        if constexpr (std::is_same<GType, float>::value && std::is_same<OType, float>::value) {
            if (headFirstPipeline_) {
                ProcessHeadFirstFastRange(baseOffset, tStart, tEnd, hStart, hLen);
                return;
            }
        }
        for (int64_t chunkStart = tStart; chunkStart < tEnd; chunkStart += tiling_->chunkSize) {
            int64_t chunkEnd = GetChunkEnd(chunkStart, tEnd);
            ProcessSequenceChunkByPath(baseOffset, chunkStart, chunkEnd, hStart, hLen);
        }
    }

    __aicore__ inline void ProcessFixed()
    {
        int64_t blockNum = static_cast<int64_t>(GetBlockNum());
        int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        int64_t chunkNum = CeilDivInt64(tiling_->t, tiling_->chunkSize);
        int64_t hTileSize = chunkFastPath_ ? fastHTileSize_ : H_TILE_SIZE;
        int64_t hTileNum = CeilDivInt64(tiling_->h, hTileSize);
        if (rebalanceChunkTasks_) {
            int64_t totalChunks = tiling_->b * hTileNum * chunkNum;
            // Distribute the remainder across the first cores so no core is left empty
            // while other cores receive an avoidable extra chunk.
            int64_t baseChunksPerCore = totalChunks / blockNum;
            int64_t remainderChunks = totalChunks % blockNum;
            int64_t unit = blockIdx * baseChunksPerCore + MinInt64(blockIdx, remainderChunks);
            int64_t unitEnd = unit + baseChunksPerCore + (blockIdx < remainderChunks ? 1 : 0);
            while (unit < unitEnd) {
                int64_t tileLinear = unit / chunkNum;
                int64_t chunkIdx = unit % chunkNum;
                int64_t hTileIdx = tileLinear % hTileNum;
                int64_t bIdx = tileLinear / hTileNum;
                int64_t tileUnitEnd = MinInt64(unitEnd, (tileLinear + 1) * chunkNum);
                int64_t hStart = hTileIdx * hTileSize;
                int64_t hLen = MinInt64(hTileSize, tiling_->h - hStart);
                int64_t chunkStart = chunkIdx * tiling_->chunkSize;
                int64_t chunkEnd = MinInt64((tileUnitEnd - tileLinear * chunkNum) * tiling_->chunkSize,
                                            tiling_->t);
                ProcessChunkRange(GetDenseBaseOffset(bIdx), chunkStart, chunkEnd, hStart, hLen);
                unit = tileUnitEnd;
            }
            return;
        }
        int64_t rangeLen = fastChunkGroupSize_ * FAST_HEAD_FIRST_RANGE_GROUPS * tiling_->chunkSize;
        int64_t rangeNum = headFirstPipeline_ ? CeilDivInt64(tiling_->t, rangeLen) : chunkNum;
        int64_t taskNum = tiling_->b * rangeNum * hTileNum;
        for (int64_t taskIdx = blockIdx; taskIdx < taskNum; taskIdx += blockNum) {
            int64_t hTileIdx = taskIdx % hTileNum;
            int64_t rangeLinear = taskIdx / hTileNum;
            int64_t rangeIdx = rangeLinear % rangeNum;
            int64_t bIdx = rangeLinear / rangeNum;
            int64_t hStart = hTileIdx * hTileSize;
            int64_t hLen = MinInt64(hTileSize, tiling_->h - hStart);
            int64_t chunkStart = headFirstPipeline_ ? rangeIdx * rangeLen : rangeIdx * tiling_->chunkSize;
            int64_t chunkEnd = headFirstPipeline_ ? MinInt64(chunkStart + rangeLen, tiling_->t)
                                                  : GetChunkEnd(chunkStart, tiling_->t);
            ProcessChunkRange(GetDenseBaseOffset(bIdx), chunkStart, chunkEnd, hStart, hLen);
        }
    }

    __aicore__ inline void ProcessVarlen()
    {
        int64_t blockNum = static_cast<int64_t>(GetBlockNum());
        int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        int64_t hTileSize = chunkFastPath_ ? fastHTileSize_ : H_TILE_SIZE;
        int64_t hTileNum = CeilDivInt64(tiling_->h, hTileSize);
        if constexpr (std::is_same<GType, float>::value && std::is_same<OType, float>::value) {
            bool varlenSeqTask = headFirstPipeline_ && (tiling_->varlenSeqTask != 0);
            if (varlenSeqTask) {
                int64_t seqTaskNum = tiling_->b * tiling_->seqNum * hTileNum;
                for (int64_t taskIdx = blockIdx; taskIdx < seqTaskNum; taskIdx += blockNum) {
                    int64_t hTileIdx = taskIdx % hTileNum;
                    int64_t hStart = hTileIdx * hTileSize;
                    int64_t hLen = MinInt64(hTileSize, tiling_->h - hStart);
                    int64_t seqLinear = taskIdx / hTileNum;
                    int64_t seqId = seqLinear % tiling_->seqNum;
                    int64_t outerIdx = seqLinear / tiling_->seqNum;
                    int64_t bos = cuSeqlensGm_.GetValue(seqId);
                    int64_t eos = cuSeqlensGm_.GetValue(seqId + 1);
                    int64_t baseOffset = GetVarlenBaseOffset(outerIdx, bos);
                    ProcessHeadFirstFastRange(baseOffset, 0, eos - bos, hStart, hLen);
                }
                return;
            }
        }

        int64_t taskNum = tiling_->b * tiling_->numBlocks * hTileNum;
        for (int64_t taskIdx = blockIdx; taskIdx < taskNum; taskIdx += blockNum) {
            int64_t hTileIdx = taskIdx % hTileNum;
            int64_t blockLinear = taskIdx / hTileNum;
            int64_t globalBlock = blockLinear % tiling_->numBlocks;
            int64_t outerIdx = blockLinear / tiling_->numBlocks;
            int64_t hStart = hTileIdx * hTileSize;
            int64_t hLen = MinInt64(hTileSize, tiling_->h - hStart);
            int64_t seqId = chunkIndicesGm_.GetValue(globalBlock * 2);
            int64_t localBlock = chunkIndicesGm_.GetValue(globalBlock * 2 + 1);
            int64_t bos = cuSeqlensGm_.GetValue(seqId);
            int64_t eos = cuSeqlensGm_.GetValue(seqId + 1);
            int64_t seqLen = eos - bos;
            int64_t tStart = localBlock * tiling_->blockT;
            int64_t tEnd = MinInt64(tStart + tiling_->blockT, seqLen);
            int64_t baseOffset = GetVarlenBaseOffset(outerIdx, bos);
            ProcessChunkRange(baseOffset, tStart, tEnd, hStart, hLen);
        }
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, FAST_BUFFER_NUM> headFirstChunkQueue_;
    TQue<QuePosition::VECOUT, FAST_BUFFER_NUM> headFirstOutQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> chunkQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> rowQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputRowQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outCastQueue_;
    TBuf<> scanBuf_;
    TBuf<> cumsumLastRowBuf_;
    TBuf<> cumsumWorkspaceBuf_;
    TBuf<> accBuf_;
    GlobalTensor<GType> gGm_;
    GlobalTensor<OType> outGm_;
    GlobalTensor<int64_t> cuSeqlensGm_;
    GlobalTensor<int64_t> chunkIndicesGm_;
    const ChunkLocalCumsumTilingData *tiling_ = nullptr;
    int64_t fastHTileSize_ = H_TILE_SIZE;
    int64_t fastChunkGroupSize_ = 1;
    bool rebalanceChunkTasks_ = false;
    bool chunkFastPath_ = false;
    bool cumsumFastPath_ = false;
    bool enableCumSumFastPath_ = false;
    bool optimizedHeadFirst_ = false;
    bool headFirstPipeline_ = false;
    TEventID vToMte3Event_;
    TEventID mte3ToVEvent_;
};
} // namespace

extern "C" __global__ __aicore__ void chunk_local_cumsum(GM_ADDR g, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
                                                          GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ChunkLocalCumsumTilingData);
    if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_AIV_ONLY);
        GET_TILING_DATA_WITH_STRUCT(ChunkLocalCumsumTilingData, tilingData, tiling);
        ChunkLocalCumsumKernel<float, float, CHUNK_128_SIZE> op;
        op.Init(g, cuSeqlens, chunkIndices, out, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(0)) {
        KERNEL_TASK_TYPE(0, KERNEL_TYPE_AIV_ONLY);
        GET_TILING_DATA_WITH_STRUCT(ChunkLocalCumsumTilingData, tilingData, tiling);
        if (tilingData.inputDtype == DTYPE_FP16 && tilingData.outputDtype == DTYPE_FP16) {
            ChunkLocalCumsumKernel<half, half> op;
            op.Init(g, cuSeqlens, chunkIndices, out, &tilingData);
            op.Process();
        } else if (tilingData.inputDtype == DTYPE_FP16 && tilingData.outputDtype == DTYPE_BF16) {
            ChunkLocalCumsumKernel<half, bfloat16_t> op;
            op.Init(g, cuSeqlens, chunkIndices, out, &tilingData);
            op.Process();
        } else if (tilingData.inputDtype == DTYPE_FP16) {
            ChunkLocalCumsumKernel<half, float> op;
            op.Init(g, cuSeqlens, chunkIndices, out, &tilingData);
            op.Process();
        } else if (tilingData.inputDtype == DTYPE_BF16 && tilingData.outputDtype == DTYPE_FP16) {
            ChunkLocalCumsumKernel<bfloat16_t, half> op;
            op.Init(g, cuSeqlens, chunkIndices, out, &tilingData);
            op.Process();
        } else if (tilingData.inputDtype == DTYPE_BF16 && tilingData.outputDtype == DTYPE_BF16) {
            ChunkLocalCumsumKernel<bfloat16_t, bfloat16_t> op;
            op.Init(g, cuSeqlens, chunkIndices, out, &tilingData);
            op.Process();
        } else if (tilingData.inputDtype == DTYPE_BF16) {
            ChunkLocalCumsumKernel<bfloat16_t, float> op;
            op.Init(g, cuSeqlens, chunkIndices, out, &tilingData);
            op.Process();
        } else if (tilingData.outputDtype == DTYPE_FP16) {
            ChunkLocalCumsumKernel<float, half> op;
            op.Init(g, cuSeqlens, chunkIndices, out, &tilingData);
            op.Process();
        } else if (tilingData.outputDtype == DTYPE_BF16) {
            ChunkLocalCumsumKernel<float, bfloat16_t> op;
            op.Init(g, cuSeqlens, chunkIndices, out, &tilingData);
            op.Process();
        } else {
            ChunkLocalCumsumKernel<float, float> op;
            op.Init(g, cuSeqlens, chunkIndices, out, &tilingData);
            op.Process();
        }
    }
}

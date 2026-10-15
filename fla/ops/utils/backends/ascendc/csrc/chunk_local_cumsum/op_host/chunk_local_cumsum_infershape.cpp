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
 * \file chunk_local_cumsum_infershape.cpp
 * \brief
 */

#include <cstring>
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
namespace {
constexpr int64_t G_INDEX = 0;
constexpr int64_t OUT_INDEX = 0;
constexpr int64_t ATTR_OUTPUT_DTYPE_INDEX = 4;

static bool ResolveOutputDataType(const char *outputDtype, ge::DataType inputDtype, ge::DataType &resolvedDtype)
{
    if (outputDtype == nullptr || outputDtype[0] == '\0' ||
        std::strcmp(outputDtype, "float") == 0 ||
        std::strcmp(outputDtype, "float32") == 0 ||
        std::strcmp(outputDtype, "fp32") == 0 ||
        std::strcmp(outputDtype, "torch.float") == 0 ||
        std::strcmp(outputDtype, "torch.float32") == 0) {
        resolvedDtype = ge::DT_FLOAT;
        return true;
    }
    if (std::strcmp(outputDtype, "same") == 0 ||
        std::strcmp(outputDtype, "same_as_input") == 0 ||
        std::strcmp(outputDtype, "input") == 0 ||
        std::strcmp(outputDtype, "none") == 0 ||
        std::strcmp(outputDtype, "None") == 0 ||
        std::strcmp(outputDtype, "null") == 0) {
        resolvedDtype = inputDtype;
        return true;
    }
    if (std::strcmp(outputDtype, "float16") == 0 ||
        std::strcmp(outputDtype, "fp16") == 0 ||
        std::strcmp(outputDtype, "half") == 0 ||
        std::strcmp(outputDtype, "torch.float16") == 0 ||
        std::strcmp(outputDtype, "torch.half") == 0) {
        resolvedDtype = ge::DT_FLOAT16;
        return true;
    }
    if (std::strcmp(outputDtype, "bfloat16") == 0 ||
        std::strcmp(outputDtype, "bf16") == 0 ||
        std::strcmp(outputDtype, "torch.bfloat16") == 0) {
        resolvedDtype = ge::DT_BF16;
        return true;
    }
    return false;
}
} // namespace

static ge::graphStatus InferShapeChunkLocalCumsum(gert::InferShapeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeChunkLocalCumsum.");
    const gert::Shape *gShape = context->GetInputShape(G_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gShape);
    gert::Shape *outShape = context->GetOutputShape(OUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);

    outShape->SetDimNum(gShape->GetDimNum());
    for (size_t i = 0; i < gShape->GetDimNum(); ++i) {
        outShape->SetDim(i, gShape->GetDim(i));
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeChunkLocalCumsum(gert::InferDataTypeContext *context)
{
    ge::DataType inputDtype = context->GetInputDataType(G_INDEX);
    ge::DataType outputDtype = ge::DT_FLOAT;
    const char *outputDtypeAttr = nullptr;
    if (context->GetAttrs() != nullptr) {
        outputDtypeAttr = context->GetAttrs()->GetAttrPointer<char>(ATTR_OUTPUT_DTYPE_INDEX);
    }
    if (!ResolveOutputDataType(outputDtypeAttr, inputDtype, outputDtype)) {
        return GRAPH_FAILED;
    }
    context->SetOutputDataType(OUT_INDEX, outputDtype);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ChunkLocalCumsum)
    .InferShape(InferShapeChunkLocalCumsum)
    .InferDataType(InferDataTypeChunkLocalCumsum);
} // namespace ops

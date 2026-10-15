// CANN opdev/op_log.h declares these helpers, but some torch_npu/CANN
// combinations do not export them into the process. Keep the custom opapi
// library self-contained so ctypes loading does not depend on PyTorch
// dispatcher-side symbols; real runtime definitions override these weak
// no-op fallbacks when they are available.
#include <cstdint>

__attribute__((weak)) void DlogRecordInner(int32_t moduleId, int32_t level, const char *fmt, ...)
{
    (void)moduleId;
    (void)level;
    (void)fmt;
}

__attribute__((weak)) int32_t CheckLogLevelInner(int32_t moduleId, int32_t level)
{
    (void)moduleId;
    (void)level;
    return 0;
}

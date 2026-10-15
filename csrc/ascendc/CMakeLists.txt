cmake_minimum_required(VERSION 3.18)
project(fla_ascendc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_SKIP_RPATH TRUE)
set(OP_DIR "${FLA_SOURCE}/fla/ops/utils/backends/ascendc/csrc/chunk_local_cumsum")
set(HOST_DIR "${OP_DIR}/op_host")
set(CANN_ARCH "${CANN}/${CMAKE_SYSTEM_PROCESSOR}-linux")

add_library(cann_host INTERFACE)
target_include_directories(cann_host INTERFACE
    "${CANN}/include" "${CANN}/include/aclnn" "${CANN}/include/op_common"
    "${CANN}/include/experiment" "${CANN}/include/experiment/runtime"
    "${CANN_ARCH}/pkg_inc" "${CANN_ARCH}/pkg_inc/op_common"
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_directories(cann_host INTERFACE "${CANN}/lib64")
target_compile_definitions(cann_host INTERFACE _GLIBCXX_USE_CXX11_ABI=0)
target_compile_options(cann_host INTERFACE -O2)

add_library(fla_opdef SHARED "${HOST_DIR}/chunk_local_cumsum_def.cpp")
target_link_libraries(fla_opdef PRIVATE cann_host exe_graph register c_sec)

add_library(fla_opmaster SHARED
    "${HOST_DIR}/chunk_local_cumsum_tiling.cpp"
    "${HOST_DIR}/chunk_local_cumsum_infershape.cpp")
target_compile_definitions(fla_opmaster PRIVATE LOG_CPP PROCESS_LOG)
target_link_libraries(fla_opmaster PRIVATE cann_host
    -Wl,--whole-archive rt2_registry tiling_api -Wl,--no-whole-archive
    -Wl,--no-as-needed exe_graph graph graph_base platform register error_manager ascendalog nnopbase c_sec)
set_target_properties(fla_opmaster PROPERTIES OUTPUT_NAME cust_opmaster_rt2.0)

add_library(fla_opapi SHARED
    "${HOST_DIR}/op_api/aclnn_chunk_local_cumsum.cpp"
    "${HOST_DIR}/op_api/chunk_local_cumsum.cpp"
    src/cust_opapi_dlog_stub.cpp)
target_link_libraries(fla_opapi PRIVATE cann_host -Wl,-Bsymbolic
    opapi nnopbase profapi ge_common_base ascend_dump ascendalog dl)
set_target_properties(fla_opapi PROPERTIES OUTPUT_NAME cust_opapi NO_SONAME ON)

set(PROTO_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/autogen/chunk_local_cumsum_proto.cpp")
set_source_files_properties("${PROTO_SOURCE}" PROPERTIES GENERATED TRUE)
add_library(fla_proto SHARED EXCLUDE_FROM_ALL "${PROTO_SOURCE}")
target_include_directories(fla_proto PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/autogen")
target_link_libraries(fla_proto PRIVATE cann_host
    -Wl,--whole-archive rt2_registry -Wl,--no-whole-archive
    exe_graph graph graph_base register metadef ascendalog unified_dlog c_sec mmpa)
target_link_options(fla_proto PRIVATE -Wl,--no-undefined)
set_target_properties(fla_proto PROPERTIES OUTPUT_NAME cust_opsproto_rt2.0)

add_library(fla_ascendc SHARED src/registration.cpp src/runtime.cpp)
target_include_directories(fla_ascendc PRIVATE include include/torch_stable_abi "${OP_DIR}")
target_link_directories(fla_ascendc PRIVATE "${TORCH_LIB}")
target_link_libraries(fla_ascendc PRIVATE -Wl,--no-as-needed torch_cpu c10 torch dl)
target_compile_options(fla_ascendc PRIVATE -O2 -fvisibility=hidden)
target_compile_definitions(fla_ascendc PRIVATE FLA_ASCENDC_SOURCE_HASH="${SOURCE_HASH}")

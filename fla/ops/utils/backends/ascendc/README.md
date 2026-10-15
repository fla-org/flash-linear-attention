# AscendC chunk-local cumsum PoC

This optional backend integrates one AscendC operator with FLA's existing public API, backend registry, and setuptools wheel build. Native compilation and NPU execution require Linux, Ascend950, and a compatible CANN development toolkit. The native build, six-case public-API accuracy demo, and 438 selected tests have passed on Ascend950PR_9579 with CANN 9.2.0-beta.2, PyTorch 2.9.0+cpu, and torch_npu 2.9.0.post2. The tests cover 432 dense numerical cases, five fallback cases, and one nondefault-stream case. Other toolchain versions and devices remain unverified.

## Supported calls

The public entry remains `fla.ops.utils.chunk_local_cumsum(g, chunk_size, ...)`.

| Argument       | AscendC route                                              |
|----------------|-----------------------------------------------------------|
| `g`            | Nonempty dense `[B, T, H]` on Ascend950; FP32, FP16 or BF16 |
| Layout         | Contiguous or strided; adapter materializes `[B, H, T]`    |
| `chunk_size`   | 64 or 128, including partial final chunks                  |
| `reverse`      | `False`                                                   |
| `scale`        | `None` or a finite Python scalar representable in FP32     |
| `output_dtype` | `torch.float32`                                           |
| Metadata       | No `cu_seqlens`, `chunk_indices`, or `use_graph`            |
| Autograd       | No backward registration; grad-enabled inputs rejected   |

The adapter restores `[B, T, H]` after the native call. Transposes and contiguous copies are included in the execution cost; this PoC makes no performance claim. Four-dimensional input, reverse cumsum, varlen, other output types, and other chunk sizes retain existing dispatch. Inputs requiring gradients are eligible only under `torch.no_grad()`. Graph capture and editable native builds are outside this PoC.

`FLA_ASCENDC=1` opts in at Python process startup. Priority `-1` places the backend before Triton-Ascend's priority `0`. An unavailable backend or a rejected call continues through the existing registry; a selected native implementation's build/load/launch error propagates to the caller.

## Build and install

Install the backend dependencies described in the repository's [INSTALL.md](../../../../../INSTALL.md) first. The native build uses the active environment's PyTorch libraries and CANN Python compiler modules, so use `--no-build-isolation`. A compatible C++ compiler, CMake >= 3.18, `op_build`, `opc`, and the CANN development headers/libraries must be available.

From the repository root, in a dedicated environment on a supported host:

```bash
source /path/to/ascend-toolkit/set_env.sh
python -m pip install 'setuptools>=64' wheel build
FLA_BUILD_ASCENDC=1 FLA_ASCENDC_SOC=ascend950 MAX_JOBS=4 \
  python -m pip wheel . --no-build-isolation --no-deps --wheel-dir dist/ascendc
python -m pip install --no-deps --force-reinstall dist/ascendc/flash_linear_attention-*-linux_*.whl
fla_poc_source="$PWD"
cd "$(mktemp -d)"
PYTHONPATH="$ASCEND_HOME_PATH/python/site-packages" FLA_ASCENDC=1 FLA_CI_ENV=0 \
  python "$fla_poc_source/scripts/check_ascendc_cumsum.py"
PYTHONPATH="$ASCEND_HOME_PATH/python/site-packages" FLA_ASCENDC=1 FLA_CI_ENV=0 python -m pytest \
  --import-mode=importlib -o pythonpath= "$fla_poc_source/tests/ops/utils/test_cumsum.py" -k ascendc -v
```

The demo prints `fla.__file__`, the source hash stored in the wheel, and per-case errors. The commands run outside the checkout and retain only CANN's Python modules on `PYTHONPATH`; pytest uses importlib mode and disables the repository's `pythonpath` setting so these checks import the installed wheel. CANN's Python modules are needed for the Triton-Ascend fallback's device-capacity detection.

For the split release flow, `scripts/build_packages.py::build_split_packages()` copies the native build inputs into the generated `fla-core` source tree. Build that directory with the same environment flags and pip command. The extension package contains only models/layers and depends on `fla-core`. The generated generic release `build.sh` uses isolated builds; use the explicit `--no-build-isolation` command for native `fla-core` builds.

Without `FLA_BUILD_ASCENDC=1`, the normal pure-Python wheel builds without importing torch or CANN from the build helper. A native build failure fails wheel creation. `FLA_ASCENDC_OPS`, when supplied, must equal `chunk_local_cumsum`; this PoC does not build an arbitrary operator subset.

## Build and runtime chain

```text
pip wheel / pip install . --no-build-isolation
  -> setuptools.build_meta -> setup.py -> AscendCBuildPy
  -> CMake: op definition, ACLNN API, tiling/infer-shape, Stable ABI launcher
  -> CANN op_build: operator INI and generated proto source
  -> CMake fla_proto: generated operator proto library
  -> CANN ascendc_impl_build.py: dynamic compile entry
     kernel source staged under impl/ascendc/chunk_local_cumsum/
  -> CANN ascendc_bin_param_build.py: dtype combinations and compile scripts
  -> opc -> AscendC compiler -> device .o files + per-kernel JSON
  -> CANN ascendc_ops_config.py: binary/config indices
  -> validate indexed artifacts -> stage OPP and launcher -> platform wheel

new Python process with FLA_ASCENDC=1
  -> wheel's fla_ascendc.pth -> stdlib bootstrap -> ASCEND_CUSTOM_OPP_PATH
  -> FLA import -> AscendCUtilsBackend registration
  -> public chunk_local_cumsum -> input_guard -> dispatch -> verifier
  -> [B,T,H] to contiguous [B,H,T]
  -> packaged libcust_opapi.so + torch.ops.load_library(libfla_ascendc.so)
  -> current device and flushing current-stream accessor
  -> torch.ops.fla_ascendc.npu_chunk_local_cumsum
  -> boxed Stable ABI adapter -> FLA_STABLE_EXEC
  -> aclnnChunkLocalCumsumGetWorkspaceSize -> workspace allocation
  -> aclnnChunkLocalCumsum -> CANN executor/host tiling -> packaged device kernel
  -> restore contiguous [B,T,H] FP32 result
```

The `.pth` hook is installed only in a native wheel and imports no torch modules. It prepends the wheel's vendor path before CANN can initialize, retaining other custom OPP paths. Set `FLA_ASCENDC` before starting Python; changing it in a running process is unsupported. Runtime loading is lazy, checks the launcher's source hash against `build.json`, and never compiles a kernel.

Native files reside under `fla/utils/ascendc/_native/`:

```text
build.json
libfla_ascendc.so
opp/vendors/config.ini
opp/vendors/fla_ascendc/
  op_api/lib/libcust_opapi.so
  op_proto/lib/linux/<arch>/libcust_opsproto_rt2.0.so
  op_impl/ai_core/tbe/
    op_tiling/lib/linux/<arch>/libcust_opmaster_rt2.0.so
    op_tiling/liboptiling.so
    config/ascend950/aic-ascend950-ops-info.json
    kernel/config/ascend950/*.json
    kernel/ascend950/chunk_local_cumsum/*.{o,json}
```

This is a package-local OPP deployment: no `.run` installer or global CANN installation mutation is needed. CANN and PyTorch shared libraries remain external dependencies; the wheel uses a Linux platform tag and does not claim manylinux portability.

## Source ownership and local checks

The original `op_host` and `op_kernel` files are under `csrc/chunk_local_cumsum/` beside their owning backend. Shared native targets and the Stable ABI header closure are at repository-root `csrc/ascendc/`; packaging is in `build_helpers/ascendc.py`. `MANIFEST.in` includes the source closure for sdist rebuilds and excludes generated `_native` payloads. The upstream native sources and their license notices are retained; see root `NOTICE` and `LICENSES/`.

Host-only checks, which do not establish NPU correctness:

```bash
python -m pytest --confcutdir=tests/host tests/host/test_ascendc.py tests/test_split_package_release.py -q
python -m build --no-isolation
```

The host route tests stub device dependencies but execute FLA's real registry and the new verifier. The native-wheel staging test uses explicitly labeled fixture bytes, not compiled device binaries. The NPU cases cover dtype, scale, tail chunks, strided inputs, fallback boundaries, and producer/consumer ordering on a nondefault stream; an explicit AscendC test request fails if the native payload is missing.

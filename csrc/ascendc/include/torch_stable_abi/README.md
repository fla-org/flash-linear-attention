# Vendored torch Stable-ABI headers

Verbatim copy of the Stable-ABI header closure of **torch 2.9.0**
(`torch/include/**` of the manylinux wheel, byte-for-byte the same as the
sdist).  They are here so `libfla_npu_stable.so` is compiled against a *fixed*
interface instead of whatever torch the build machine happens to have.

**Do not edit these files.**  `tools/vendor_stable_headers.py --check` and
`tests/test_stable_gates.py::VendoredHeaderTest` compare the tree against
`MANIFEST.sha256` and fail on any drift, so a local "fix" to a header becomes a
red gate rather than a silent divergence.

## Why pinned, and why 2.9

Two requirements pull in opposite directions:

* the artifact has to load on torch 2.7.1, and
* it must not inherit symbols that a newer torch introduced.

`torch/csrc/stable/library.h` is where they meet.  From 2.10 on, `impl()` goes
through `torch_library_impl(lib, name, fn, TORCH_ABI_VERSION)`; 2.7.1 -- 2.9
export only `aoti_torch_library_impl(lib, name, fn)`.  Compiling against the
installed torch therefore tied the artifact to the *build* torch: a wheel built
on a 2.10+ machine fails to load on 2.7.1 -- 2.9 with an undefined symbol.

2.9 is the pin because it is the oldest release with the complete set:

| torch | `torch/include/torch/csrc/stable/` |
| --- | --- |
| 2.7.1 | `library.h` |
| 2.8.0 | `library.h`, `tensor.h` |
| 2.9.0 | `library.h`, `tensor.h`, `tensor_inl.h`, `tensor_struct.h`, `stableivalue_conversions.h`, `accelerator.h`, `ops.h` |

The runtime floor is then what 2.9's headers still reference rather than what
the build torch provides.  Measured: the launcher references 38 `aoti_torch_*`
symbols, all 38 exist in 2.7.1 / 2.8.0 / 2.9.0 / 2.10.0 / 2.11.0 / 2.12.0, and
the entry point it uses (`aoti_torch_library_impl`) is exported by all six.
`torch`'s pins in the wheel metadata come from `STABLE_ABI_MIN_TORCH` in
`scripts/build_wheel.py`; keep the two in sync.

## What is not vendored

Two headers of the closure stay with the build torch:
`torch/headeronly/cpu/vec/vec_half.h` and `ATen/cpu/vec/vec_half.h`.  They are
reached from `torch/headeronly/util/Half.h` only under `CPU_CAPABILITY_AVX2` or
`CPU_CAPABILITY_AVX512`, which the launcher never defines, so no translation
unit here includes them.  `build_stable.py` therefore keeps the build torch's
include dirs on the command line *after* this tree -- the fallback exists for
exactly these, and a build without it is still a supported build.

## Refreshing

`tools/vendor_stable_headers.py` takes the closure from the compiler rather than
from a walk over `#include` lines: half of what torch pulls in sits behind
capability guards (`CPU_CAPABILITY_AVX2` and friends) that never fire here, and
a textual walk copies those too.  `-MD` reports the headers the launcher's two
translation units actually open, which is the set worth pinning.

```sh
python tools/vendor_stable_headers.py --torch /path/to/site-packages/torch --write
python tools/vendor_stable_headers.py --torch /path/to/site-packages/torch --check
```

`--check --torch` compiles with this tree first on the include path and fails if
any torch/c10 header still comes from the build torch -- which is how a new
revision's extra include shows up as a gate rather than as a wheel that behaves
differently depending on where it was built.  Refreshing from torch 2.9.0
reproduces this tree byte for byte (verified).

Refresh only when the runtime floor moves, and record the measurement above
again in the same commit.

## Line endings

`.gitattributes` marks this tree `-text`.  Without it a Windows checkout
(`core.autocrlf=true`) rewrites the line endings, the copy stops matching
upstream, and `MANIFEST.sha256` fails for a reason that has nothing to do with
the headers.

## License

BSD-3-Clause; see `LICENSE` (copied from the torch wheel's
`torch-<version>.dist-info/licenses/LICENSE`, "From PyTorch:" section).  The
repository-level `NOTICE` records where these files come from.

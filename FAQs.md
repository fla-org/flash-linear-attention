# FAQs

## MMA->MMA Assertion Error on H100

```py
Assertion `!(srcMmaLayout && dstMmaLayout && !srcMmaLayout.isAmpere()) && "mma -> mma layout conversion is only supported on Ampere"' failed.
```

This issue has been fixed in [PR #4492](https://github.com/triton-lang/triton/pull/4492).
It is recommended to use the nightly version of triton (built by fla:).

```sh
pip install -U --index-url https://pypi.fla-org.com/simple/ triton --upgrade
```

## AttributeError: 'NoneType' object has no attribute 'start'

This is a known triton issue [triton-lang/triton#5224](https://github.com/triton-lang/triton/issues/5224).
Upgrading python to 3.10 or higher could solve the question.


## On H100: /project/lib/Tools/LinearLayout.cpp:562: mlir::triton::LinearLayout mlir::triton::LinearLayout::reshapeOuts(llvm::ArrayRef<std::pair<mlir::StringAttr, int> >) const: Assertion `getTotalOutDimSize() == std::accumulate( newOutDims.begin(), newOutDims.end(), 1, [&](int32_t acc, auto &outDim) { return acc * outDim.second; })' failed.

This is a known triton issue [triton-lang/triton#5609](https://github.com/triton-lang/triton/issues/5609).

Use the nightly version of triton
```sh
pip install -U --index-url https://pypi.fla-org.com/simple/ triton --upgrade
```
or build it from scratch.

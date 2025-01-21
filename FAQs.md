# FAQs

1. MMA->MMA Assertion Error on H100

```py
Assertion `!(srcMmaLayout && dstMmaLayout && !srcMmaLayout.isAmpere()) && "mma -> mma layout conversion is only supported on Ampere"' failed.
```

**Solution:**

This issue has been fixed in [PR #4492](https://github.com/triton-lang/triton/pull/4492).
It is recommended to use the nightly version of triton.

```sh
pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/triton-nightly
```
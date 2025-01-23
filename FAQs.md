# FAQs

## MMA->MMA Assertion Error on H100

```py
Assertion `!(srcMmaLayout && dstMmaLayout && !srcMmaLayout.isAmpere()) && "mma -> mma layout conversion is only supported on Ampere"' failed.
```

This issue has been fixed in [PR #4492](https://github.com/triton-lang/triton/pull/4492).
It is recommended to use the nightly version of triton.

```sh
pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly
```

## AttributeError: 'NoneType' object has no attribute 'start'

This is a known triton issue [triton-lang/triton#5224](https://github.com/triton-lang/triton/issues/5224).
Upgrading python to 3.10 or higher could solve the question.

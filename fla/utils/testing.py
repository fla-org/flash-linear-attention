import os

COMPILER_MODE = os.getenv("FLA_COMPILER_MODE") == "1"
FLA_CI_ENV = os.getenv("FLA_CI_ENV") == "1"


def get_abs_err(x, y):
    return (x.detach()-y.detach()).flatten().abs().max().item()


def get_err_ratio(x, y):
    err = (x.detach()-y.detach()).flatten().square().mean().sqrt().item()
    base = (x.detach()).flatten().square().mean().sqrt().item()
    return err / (base + 1e-8)


def assert_close(prefix, ref, tri, ratio, warning=False, err_atol=1e-6):
    abs_atol = get_abs_err(ref, tri)
    msg = f"{prefix} diff: {abs_atol:.6f} ratio: {get_err_ratio(ref, tri):.6f}"
    print(msg)
    error_rate = get_err_ratio(ref, tri)
    if abs_atol <= err_atol:
        return
    if warning or (FLA_CI_ENV and (error_rate < 0.01 or abs_atol <= 0.3)):
        if error_rate > ratio:
            import warnings
            warnings.warn(msg)
    else:
        assert error_rate < ratio, msg

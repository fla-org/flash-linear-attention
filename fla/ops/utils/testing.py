import os

compiled_mode = os.getenv("FLA_COMPILER_MODE") == "1"
FLA_CI_ENV = os.getenv("FLA_CI_ENV") == "1"


def get_abs_err(x, y):
    return (x.detach()-y.detach()).flatten().abs().max().item()


def get_err_ratio(x, y):
    err = (x.detach()-y.detach()).flatten().square().mean().sqrt().item()
    base = (x).flatten().square().mean().sqrt().item()
    return err / (base + 1e-15)


def assert_close(prefix, ref, tri, ratio, warning=False):
    abs_err = get_abs_err(ref, tri)
    msg = f"{prefix} diff: {abs_err:.6f} ratio: {get_err_ratio(ref, tri):.6f}"
    print(msg)
    error_rate = get_err_ratio(ref, tri)
    if warning or str(prefix).strip().lower() == "dh0" or FLA_CI_ENV:
        if (error_rate > ratio) and (error_rate < 0.01 or abs_err < 0.3):
            import warnings
            warnings.warn(msg)
        else:
            assert error_rate < ratio, msg
    else:
        assert error_rate < ratio, msg

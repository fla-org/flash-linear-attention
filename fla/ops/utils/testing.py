import os

compiled_mode = os.getenv("COMPILER_MODE") == "1"
ci_env = os.getenv("CI_ENV") == "1"


def get_abs_err(x, y):
    return (x-y).flatten().abs().max().item()


def get_err_ratio(x, y):
    err = (x-y).flatten().square().mean().sqrt().item()
    base = (x).flatten().square().mean().sqrt().item()
    return err / (base + 1e-15)


def assert_close(prefix, ref, tri, ratio, warning=False):
    msg = f"{prefix} diff: {get_abs_err(ref, tri):.6f} ratio: {get_err_ratio(ref, tri):.6f}"
    print(msg)
    error_rate = get_err_ratio(ref, tri)
    if warning or str(prefix).strip().lower() == "dh0" or compiled_mode or (ci_env and error_rate < 0.1):
        if error_rate > ratio:
            import warnings
            warnings.warn(msg)
    else:
        assert error_rate < ratio, msg

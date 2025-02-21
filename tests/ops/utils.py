def get_abs_err(x, y):
    return (x-y).flatten().abs().max().item()


def get_err_ratio(x, y):
    x = x.float()
    y = y.float()
    err = (x-y).flatten().square().mean().sqrt().item()
    base = (x).flatten().square().mean().sqrt().item()
    return err / (base+1e-10)


def assert_close(prefix, ref, tri, ratio):
    error_ratio = get_err_ratio(ref, tri)
    msg = f"{prefix} diff: {get_abs_err(ref, tri):.6f} ratio: {error_ratio:.6f}"
    print(msg)
    assert error_ratio < ratio, msg

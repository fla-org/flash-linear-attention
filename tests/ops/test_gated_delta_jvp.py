# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Tests of the forward-mode derivative (JVP) of the chunked gated delta rule."""

import pytest
import torch
import torch.nn.functional as F

from fla.ops.gated_delta_rule.chunk_jvp import chunk_gated_delta_rule_jvp_ref
from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule
from fla.utils import device


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative difference in the Frobenius norm: |a - b| / |b|."""
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def make_inputs(B, T, H, K, V, D, dtype, seed=0, dev=device):
    """Primal inputs as a trained model feeds them (unit-norm keys, log-sigmoid gates, beta in (0, 1)), and tangents."""
    gen = torch.Generator(device='cpu').manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float64).to(dev)
    q = randn(B, T, H, K)
    k = F.normalize(randn(B, T, H, K), dim=-1)
    v = randn(B, T, H, V)
    g = F.logsigmoid(randn(B, T, H))
    beta = torch.rand(B, T, H, generator=gen, dtype=torch.float64).to(dev)
    tangents = [randn(D, *x.shape) for x in (q, k, v, g, beta)]
    cast = [x.to(dtype) for x in (q, k, v, g, beta, *tangents)]
    return cast


@pytest.mark.parametrize('T', [64, 100, 200])
def test_chunk_jvp_reference_matches_naive_fp64(T):
    """The chunked reference equals torch.func.jvp of the token-by-token recurrence in fp64 (rel. diff < 1e-10)."""
    B, H, K, V, D = 2, 3, 32, 32, 3
    q, k, v, g, beta, *tans = make_inputs(B, T, H, K, V, D, torch.float64)
    o, o_dot = chunk_gated_delta_rule_jvp_ref(q, k, v, g, beta, *tans)

    def f(q, k, v, g, beta):
        return naive_recurrent_gated_delta_rule(q, k, v, beta, g)[0]
    for d in range(D):
        o_ref, o_dot_ref = torch.func.jvp(f, (q, k, v, g, beta), tuple(t[d] for t in tans))
        assert rel(o_dot[d], o_dot_ref) < 1e-10
    assert rel(o, o_ref) < 1e-10


# ---------------------------------------------------------------------------------------------------------------
# Fast path (Triton) against the fp64 chunked reference
# ---------------------------------------------------------------------------------------------------------------
# Tolerances (relative Frobenius difference to the fp64 reference, measured on an RTX 5090):
#   * default precision: the library's forward kernels use TF32 in their fp32 dots, and so do the three JVP kernels;
#     the primal o itself is off by ~1.5e-3, and o_dot by the same amount. Bound: 5e-3.
#   * input_precision='ieee' in the three JVP kernels (library forward still TF32): o_dot ~3e-5.
#   * TRITON_F32_DEFAULT=ieee for the whole process (the library's solve_tril block merge keeps explicit TF32):
#     o ~5e-6, o_dot ~5e-6 (not a test here: compiling every library kernel with IEEE dots takes many minutes).
JVP_TF32_TOL = 5e-3


@pytest.mark.parametrize('D', [1, 8])
@pytest.mark.parametrize('T', [64, 130, 1024, 2048])
@pytest.mark.parametrize('K', [64, 128])
@pytest.mark.parametrize('H', [2, 16])
@pytest.mark.parametrize('B', [1, 2])
def test_chunk_jvp_fast_matches_reference(B, H, K, T, D):
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from fla.ops.gated_delta_rule.chunk_jvp import chunk_gated_delta_rule_jvp
    ins = make_inputs(B, T, H, K, K, D, torch.float64)
    o64, o_dot64 = chunk_gated_delta_rule_jvp_ref(*ins)
    f32 = [x.float() for x in ins]
    o, o_dot = chunk_gated_delta_rule_jvp(*f32)
    assert o_dot.shape == (D, B, T, H, K) and o_dot.dtype == torch.float32
    assert torch.isfinite(o_dot).all()
    assert rel(o_dot, o_dot64) < JVP_TF32_TOL
    assert rel(o, o64) < JVP_TF32_TOL
    # the primal output is the library's own forward, bit for bit
    assert torch.equal(o, chunk_gated_delta_rule(*f32[:5])[0])


@pytest.mark.parametrize('fuse_output', [True, False])
@pytest.mark.parametrize('intra_impl', ['triton', 'torch'])
def test_chunk_jvp_fast_variants(intra_impl, fuse_output):
    """Both implementations of the intra-chunk tangents, with and without the output fused into the recurrence."""
    from fla.ops.gated_delta_rule.chunk_jvp import chunk_gated_delta_rule_jvp_fwd
    B, T, H, K, D = 2, 200, 4, 128, 3
    ins = make_inputs(B, T, H, K, K, D, torch.float64)
    _, o_dot64 = chunk_gated_delta_rule_jvp_ref(*ins)
    f32 = [x.float() for x in ins]
    _, o_dot = chunk_gated_delta_rule_jvp_fwd(*f32, scale=K ** -0.5, intra_impl=intra_impl, fuse_output=fuse_output)
    assert rel(o_dot, o_dot64) < JVP_TF32_TOL


def test_chunk_jvp_fast_strided_tangents():
    """Tangents as non-contiguous views (a direction-broadcast and a transposed layout) give the same result."""
    from fla.ops.gated_delta_rule.chunk_jvp import chunk_gated_delta_rule_jvp
    B, T, H, K, D = 2, 200, 4, 64, 3
    q, k, v, g, beta, *tans = make_inputs(B, T, H, K, K, D, torch.float32)
    o, o_dot = chunk_gated_delta_rule_jvp(q, k, v, g, beta, *tans)
    # same values, different strides: [D, B, H, T, K] storage viewed as [D, B, T, H, K]
    strided = [t.transpose(2, 3).contiguous().transpose(2, 3) for t in tans]
    assert not strided[0].is_contiguous()
    o2, o_dot2 = chunk_gated_delta_rule_jvp(q, k, v, g, beta, *strided)
    assert torch.equal(o_dot, o_dot2)
    # a single direction broadcast to D copies (stride 0 over D): every direction equals the D=1 result
    one = [t[:1] for t in tans]
    _, o_dot1 = chunk_gated_delta_rule_jvp(q, k, v, g, beta, *one)
    bcast = [t.expand(D, *t.shape[1:]) for t in one]
    _, o_dot3 = chunk_gated_delta_rule_jvp(q, k, v, g, beta, *bcast)
    for d in range(D):
        assert torch.equal(o_dot3[d], o_dot1[0])


def test_chunk_jvp_fast_ieee_kernels():
    """input_precision='ieee' (the three JVP kernels) is at least as accurate as TF32."""
    from fla.ops.gated_delta_rule.chunk_jvp import chunk_gated_delta_rule_jvp
    B, T, H, K, D = 1, 1024, 4, 128, 2
    ins = make_inputs(B, T, H, K, K, D, torch.float64)
    _, o_dot64 = chunk_gated_delta_rule_jvp_ref(*ins)
    f32 = [x.float() for x in ins]
    _, o_dot_tf32 = chunk_gated_delta_rule_jvp(*f32, input_precision='tf32')
    _, o_dot_ieee = chunk_gated_delta_rule_jvp(*f32, input_precision='ieee')
    assert rel(o_dot_ieee, o_dot64) < JVP_TF32_TOL
    assert rel(o_dot_ieee, o_dot64) <= rel(o_dot_tf32, o_dot64) * 1.05


# ---------------------------------------------------------------------------------------------------------------
# torch.library custom ops: eager equality, autograd, and torch.compile(fullgraph=True) without graph breaks
# ---------------------------------------------------------------------------------------------------------------

def test_custom_op_forward_backward_matches_library():
    """fla::chunk_gated_delta_rule equals the library's autograd Function (forward and gradients, bit for bit),
    supports repeated autograd.grad(..., retain_graph=True), and its gradients match the fp64 naive recurrence."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from fla.ops.gated_delta_rule.custom_ops import chunk_gated_delta_rule_op
    B, T, H, K = 2, 130, 4, 64
    q, k, v, g, beta = make_inputs(B, T, H, K, K, 1, torch.float32)[:5]
    xs = [x.clone().requires_grad_() for x in (q, k, v, g, beta)]
    ys = [x.clone().requires_grad_() for x in (q, k, v, g, beta)]
    zs = [x.double().clone().requires_grad_() for x in (q, k, v, g, beta)]
    o1 = chunk_gated_delta_rule_op(*xs)
    o2 = chunk_gated_delta_rule(*ys)[0]
    o3 = naive_recurrent_gated_delta_rule(zs[0], zs[1], zs[2], zs[4], zs[3])[0]
    assert torch.equal(o1, o2)
    do = torch.randn_like(o1)
    grads = [torch.autograd.grad(o1, xs, do, retain_graph=True) for _ in range(3)]
    ref = torch.autograd.grad(o2, ys, do)
    ref64 = torch.autograd.grad(o3, zs, do.double())
    for i in range(5):
        assert torch.equal(grads[0][i], ref[i])
        assert torch.equal(grads[1][i], grads[0][i]) and torch.equal(grads[2][i], grads[0][i])
        # TF32 in the library's kernels: ~1e-3 relative to fp64
        assert rel(grads[0][i], ref64[i]) < 1e-2


class _TinyGDN(torch.nn.Module):
    """Projections -> gated delta rule -> projection, as in a model layer."""

    def __init__(self, C=96, H=2, K=64):
        super().__init__()
        self.H, self.K = H, K
        self.qkv = torch.nn.Linear(C, 3 * H * K)
        self.gb = torch.nn.Linear(C, 2 * H)
        self.out = torch.nn.Linear(H * K, C)

    def parts(self, x):
        q, k, v = self.qkv(x).view(*x.shape[:2], 3, self.H, self.K).unbind(2)
        a, b = self.gb(x).view(*x.shape[:2], 2, self.H).unbind(2)
        return F.normalize(q, dim=-1), F.normalize(k, dim=-1), v, F.logsigmoid(a), b.sigmoid()

    def forward(self, x):
        from fla.ops.gated_delta_rule.custom_ops import chunk_gated_delta_rule_op
        return self.out(chunk_gated_delta_rule_op(*self.parts(x)).flatten(2))

    def jvp(self, x, x_dot):
        """(y, y_dot [D, ...]): tangents of the op inputs by torch.func.jvp through the projections, then the op."""
        from fla.ops.gated_delta_rule.custom_ops import chunk_gated_delta_rule_jvp_op
        prim = self.parts(x)
        tans = [torch.func.jvp(self.parts, (x,), (x_dot[d],))[1] for d in range(x_dot.shape[0])]
        tans = [torch.stack(t) for t in zip(*tans)]
        o, o_dot = chunk_gated_delta_rule_jvp_op(*prim, *tans)
        return self.out(o.flatten(2)), o_dot.flatten(3) @ self.out.weight.T


def _assert_no_graph_break(fn, *args):
    explanation = torch._dynamo.explain(fn)(*args)
    assert explanation.graph_break_count == 0, explanation.break_reasons
    assert explanation.graph_count == 1


def test_custom_ops_compile_fullgraph():
    torch._dynamo.reset()
    torch.manual_seed(0)
    B, T, C, D = 2, 130, 96, 3
    m = _TinyGDN(C).to(device)
    x = torch.randn(B, T, C, device=device)
    x_dot = torch.randn(D, B, T, C, device=device)

    # forward + backward
    y = m(x)
    gr = torch.autograd.grad(y.square().sum(), list(m.parameters()))
    mc = torch.compile(m, fullgraph=True, dynamic=False)
    yc = mc(x)
    grc = torch.autograd.grad(yc.square().sum(), list(m.parameters()))
    # compiled projections differ from eager in rounding (~1e-7); the recurrence carries that to ~1e-5
    assert rel(yc, y) < 1e-5
    assert max(rel(a, b) for a, b in zip(grc, gr)) < 1e-3
    _assert_no_graph_break(m, x)

    # forward-mode derivative (the op has no reverse-mode rule: call it without autograd recording)
    with torch.no_grad():
        y, y_dot = m.jvp(x, x_dot)
        yc, y_dotc = torch.compile(m.jvp, fullgraph=True, dynamic=False)(x, x_dot)
        assert rel(yc, y) < 1e-5 and rel(y_dotc, y_dot) < 1e-4
        _assert_no_graph_break(m.jvp, x, x_dot)
        # and the op's o_dot is the derivative of the layer: compare with torch.func.jvp of the reference layer
        f64 = _TinyGDN(C).to(device).double()
        f64.load_state_dict({n: p.double() for n, p in m.state_dict().items()})

        def ref_layer(x):
            q, k, v, g, beta = f64.parts(x)
            return f64.out(naive_recurrent_gated_delta_rule(q, k, v, beta, g)[0].flatten(2))
        for d in range(D):
            _, y_dot64 = torch.func.jvp(ref_layer, (x.double(),), (x_dot[d].double(),))
            assert rel(y_dot[d], y_dot64) < JVP_TF32_TOL


def test_custom_op_jvp_backward():
    """Reverse mode through fla::chunk_gated_delta_rule_jvp: gradients of <lam, o> + <lam_dot, o_dot> with respect
    to all ten inputs equal those of the fp64 chunked reference (itself checked against the naive recurrence)."""
    from fla.ops.gated_delta_rule.custom_ops import chunk_gated_delta_rule_jvp_op
    B, T, H, K, D = 1, 130, 2, 64, 2
    ins = make_inputs(B, T, H, K, K, D, torch.float64)
    lam = torch.randn(B, T, H, K, dtype=torch.float64, device=device)
    lam_dot = torch.randn(D, B, T, H, K, dtype=torch.float64, device=device)

    x64 = [x.clone().requires_grad_() for x in ins]
    o, o_dot = chunk_gated_delta_rule_jvp_ref(*x64)
    ref = torch.autograd.grad((o * lam).sum() + (o_dot * lam_dot).sum(), x64)

    x32 = [x.float().requires_grad_() for x in ins]
    o, o_dot = chunk_gated_delta_rule_jvp_op(*x32)
    got = torch.autograd.grad((o * lam.float()).sum() + (o_dot * lam_dot.float()).sum(), x32)
    for name, a, b in zip(('q', 'k', 'v', 'g', 'beta', 'q_dot', 'k_dot', 'v_dot', 'g_dot', 'beta_dot'), got, ref):
        assert rel(a, b) < JVP_TF32_TOL, name

    # tangents only (the primal does not require grad): the fast path alone
    xs = [x.float() for x in ins[:5]]
    ts = [t.float().requires_grad_() for t in ins[5:]]
    _, o_dot = chunk_gated_delta_rule_jvp_op(*xs, *ts)
    got = torch.autograd.grad((o_dot * lam_dot.float()).sum(), ts)
    for a, b in zip(got, ref[5:]):
        assert rel(a, b) < JVP_TF32_TOL


def test_custom_ops_opcheck():
    """torch.library.opcheck: schema, fake (meta) implementations, autograd registration, AOT dispatch."""
    import fla.ops.gated_delta_rule.custom_ops  # noqa: F401  (registers the operators)
    B, T, H, K, D = 1, 100, 2, 64, 2
    q, k, v, g, beta, *tans = make_inputs(B, T, H, K, K, D, torch.float32)
    scale = K ** -0.5
    do = torch.randn_like(v)
    xs = [x.requires_grad_() for x in (q, k, v, g, beta)]
    torch.library.opcheck(torch.ops.fla.chunk_gated_delta_rule.default, (*xs, scale, 64))
    torch.library.opcheck(torch.ops.fla.chunk_gated_delta_rule_jvp.default, (*xs, *tans, scale, 64, None))
    xs = [x.detach() for x in xs]
    torch.library.opcheck(torch.ops.fla.chunk_gated_delta_rule_vjp.default, (*xs, do, scale, 64))
    torch.library.opcheck(torch.ops.fla.chunk_gated_delta_rule_vjp.default,
                          (*xs, torch.randn(D, *v.shape, device=device), scale, 64))

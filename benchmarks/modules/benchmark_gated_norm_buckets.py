# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Compare gated norm row buckets with fresh Triton caches on one NVIDIA GPU."""

import argparse
import hashlib
import importlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from functools import partial
from pathlib import Path

BASE = '6511ea3ecf7c3b390b15391e670d5bdcb657d914'
MODULE = 'fla.modules.norm.fused_norm_gate'
SOURCE = 'fla/modules/norm/fused_norm_gate.py'
D = 64
EPS = 1e-6
DTYPE = 'float32'
IS_RMS_NORM = False


def git(root, *args):
    if root.joinpath('.git').exists():
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    if args == ('rev-parse', 'HEAD'):
        return root.joinpath('BASE_REVISION').read_text().strip()
    raise RuntimeError('Archive checkouts only support revision lookup')


def row_counts(sm):
    # keep backward BS in one integer-specialization class while covering three partial-tile row buckets.
    if sm < 2:
        raise ValueError(f'Expected at least two SMs, found {sm}')
    stride = 16 * sm
    lo = (65536 + 1) // stride * stride - 1
    hi = lo + stride
    third = (131072 // stride + 1) * stride - 1
    if not 0 < lo <= 65536 < hi <= 131072 < third <= 196608:
        raise ValueError(f'Cannot cover NB=1, NB=2 and NB=3 for {sm} SMs')
    return lo, hi, third


def tensors(torch, rows):
    rng = torch.Generator(device='cuda').manual_seed(42 + rows)

    def sample(shape):
        return torch.randn(shape, generator=rng, device='cuda', dtype=getattr(torch, DTYPE))

    shape = (rows, D)
    x, dy = sample(shape), sample(shape)
    params = (x, sample(shape), sample((D,)), None if IS_RMS_NORM else sample((D,)))
    return params, dy


def reference(torch, params, dy):
    x, g, w, b = [t.detach().double().requires_grad_() if t is not None else None for t in params]
    norm = x * (x.square().mean(-1, keepdim=True) + EPS).rsqrt() * w if IS_RMS_NORM else torch.nn.functional.layer_norm(
        x, (D,), w, b, EPS)
    y = norm * torch.nn.functional.silu(g)
    grads = torch.autograd.grad(y, [t for t in (x, g, w, b) if t is not None], dy.double())
    names = ('dx', 'dg', 'dw') if b is None else ('dx', 'dg', 'dw', 'db')
    return {'y': y.detach().to(params[0].dtype), **{k: v.detach().to(params[0].dtype) for k, v in zip(names, grads)}}


def compare(torch, actual, expected):
    result = {}
    if actual.keys() != expected.keys():
        raise AssertionError(f'Different output fields: {actual.keys()}, {expected.keys()}')
    for name, value in actual.items():
        ref = expected[name]
        if value.shape != ref.shape or value.dtype != ref.dtype:
            raise AssertionError(f'Unexpected {name} metadata: {value.shape}, {value.dtype}')
        if not torch.isfinite(value).all().item() or not torch.isfinite(ref).all().item():
            raise AssertionError(f'Non-finite {name}')
        err = value.detach().float() - ref.float()
        abs_max = err.abs().max().item()
        rel_rms = (err.square().mean().sqrt() / (ref.float().square().mean().sqrt() + 1e-8)).item()
        ratio = 0.001
        if not (abs_max <= 1e-6 or rel_rms < ratio):
            raise AssertionError(f'{name}: max absolute error {abs_max}, relative RMS {rel_rms}, limit {ratio}')
        raw = value.detach().contiguous().view(torch.uint8).cpu().numpy()
        result[name] = {'max_abs': abs_max, 'rel_rms': rel_rms, 'sha256': hashlib.sha256(raw).hexdigest()}
    return result


def calls(mod, params, dy):
    x, g, w, b = params

    def fwd():
        return mod.layer_norm_gated_fwd(x, g, w, b, activation='silu', eps=EPS, is_rms_norm=IS_RMS_NORM)

    def bwd(state):
        y, mean, rstd, _ = state
        dx, dg, dw, db, _ = mod.layer_norm_gated_bwd(
            dy, x, g, w, b, activation='silu', eps=EPS, mean=mean, rstd=rstd, is_rms_norm=IS_RMS_NORM)
        return {'y': y, 'dx': dx, 'dg': dg, 'dw': dw, **({'db': db} if db is not None else {})}

    return fwd, bwd


def measure(torch, invoke, tuner, kernel, nb, misses, cache, hot_samples):
    from triton.testing import do_bench

    def cubins():
        return set(cache.rglob(f'{kernel}.cubin'))

    keys = set(tuner.cache)
    before = cubins()
    n_before = misses[kernel]
    torch.cuda.synchronize()
    start = time.perf_counter()
    state = invoke()
    torch.cuda.synchronize()
    first_ms = (time.perf_counter() - start) * 1000
    keys = set(tuner.cache) - keys
    if len(keys) != 1:
        raise RuntimeError(f'Expected one new autotune entry for {kernel}, found {keys}')
    key = keys.pop()
    if key[tuner.keys.index('NB')] != nb:
        raise RuntimeError(f'Unexpected autotune key for {kernel}: {key}')
    new_cubins = len(cubins() - before)
    new_compiles = misses[kernel] - n_before
    if new_compiles != new_cubins or (nb == 1 and new_compiles == 0):
        raise RuntimeError(f'{kernel} NB={nb}: {new_compiles} compiler misses, {new_cubins} new cubins')

    for _ in range(3):
        invoke()
    torch.cuda.synchronize()
    hot = []
    for _ in range(hot_samples):
        start = time.perf_counter()
        invoke()
        torch.cuda.synchronize()
        hot.append((time.perf_counter() - start) * 1000)
    device_p20, device_median, device_p80 = do_bench(invoke, warmup=25, rep=100, quantiles=[0.2, 0.5, 0.8])
    if misses[kernel] - n_before != new_compiles:
        raise RuntimeError(f'{kernel} compiled during the hot window')
    row = {
        'NB': nb, 'key': list(key), 'config': tuner.cache[key].all_kwargs(), 'candidates': len(tuner.configs),
        'new_compiles': new_compiles, 'new_cubins': new_cubins, 'first_ms': first_ms,
        'hot_median_ms': statistics.median(hot), 'hot_samples_ms': hot,
        'device_p20_ms': device_p20, 'device_median_ms': device_median, 'device_p80_ms': device_p80,
    }
    return state, key, row


def worker(root, work, arm, samples):
    import torch
    import triton
    from triton import knobs
    from triton.runtime.autotuner import Autotuner

    if torch.version.cuda is None or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one NVIDIA GPU with CUDA_VISIBLE_DEVICES')
    props = torch.cuda.get_device_properties(0)
    torch.cuda.set_device(0)
    if not hasattr(knobs.compilation, 'listener'):
        raise RuntimeError('This benchmark needs the Triton compilation listener')
    misses = Counter()

    def on_compile(*, src, cache_hit, **kwargs):
        if not cache_hit:
            misses[src.name] += 1

    knobs.compilation.listener = on_compile
    cache = Path(os.environ['TRITON_CACHE_DIR'])
    uuid = getattr(props, 'uuid', None)
    if not uuid or not str(uuid).strip():
        raise RuntimeError('CUDA device UUID unavailable; cannot establish a same-device comparison')
    drivers = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True,
    ).splitlines()
    if len(set(drivers)) != 1 or not all(part.isdigit() for part in drivers[0].split('.')):
        raise RuntimeError('nvidia-smi did not provide a consistent numeric driver version')
    report = {
        'git_head': git(root, 'rev-parse', 'HEAD'), 'sources': {}, 'torch': str(torch.__version__),
        'triton': triton.__version__, 'cuda': torch.version.cuda, 'python': platform.python_version(),
        'device': props.name, 'sm': props.multi_processor_count, 'capability': [props.major, props.minor],
        'device_uuid': str(uuid), 'driver': drivers[0], 'D': D, 'dtype': DTYPE, 'is_rms_norm': IS_RMS_NORM, 'rows': [],
    }
    mod = importlib.import_module(MODULE)
    source = Path(mod.__file__).resolve()
    if source != root.joinpath(SOURCE):
        raise RuntimeError(f'Loaded {MODULE} outside the selected checkout')
    report['sources'][MODULE] = hashlib.sha256(source.read_bytes()).hexdigest()
    tuners = []
    for kind in ('fwd', 'bwd'):
        name = f'layer_norm_gated_{kind}_kernel'
        tuner = getattr(mod, name)
        while not isinstance(tuner, Autotuner):
            tuner = tuner.fn
        tuners.append((name, tuner))

    for nb, rows in enumerate(row_counts(props.multi_processor_count), 1):
        params, dy = tensors(torch, rows)
        fwd, bwd = calls(mod, params, dy)
        key_args = (nb, misses, cache, samples)
        state, fk, fm = measure(torch, fwd, tuners[0][1], tuners[0][0], *key_args)
        actual, bk, bm = measure(torch, partial(bwd, state), tuners[1][1], tuners[1][0], *key_args)
        expected = reference(torch, params, dy)
        natural = compare(torch, actual, expected)
        # replay one available configuration after timing each bucket's independent winner.
        for (_, tuner), key in zip(tuners, (fk, bk)):
            tuner.cache[key] = next(c for c in tuner.configs if c.kwargs == {'BT': 16} and c.num_warps == 4)
        fixed = compare(torch, bwd(fwd()), expected)
        for (name, tuner), key in zip(tuners, (fk, bk)):
            picked = tuner.best_config
            if picked is not tuner.cache[key] or picked.kwargs != {'BT': 16} or picked.num_warps != 4:
                raise RuntimeError(f'Fixed replay did not run the selected configuration for {name}')
        report['rows'].append({
            'op': 'gated', 'T': rows, 'BS': (rows + props.multi_processor_count - 1) // props.multi_processor_count,
            'forward': fm, 'backward': bm, 'autotuned_vs_reference': natural, 'fixed_vs_reference': fixed,
        })
        print(f"{arm} gated T={rows}: compiler misses {fm['new_compiles']} fwd, {bm['new_compiles']} bwd", flush=True)
        del state, actual, expected, params, dy, fwd, bwd
    (work / f'{arm}.json').write_text(json.dumps(report, allow_nan=False))


def compare_arms(base, patched):
    for field in ('device_uuid', 'torch', 'triton', 'cuda', 'python', 'device', 'sm', 'capability', 'driver', 'D', 'dtype',
                  'is_rms_norm'):
        if base[field] != patched[field] or (field == 'device_uuid' and not base[field]):
            raise RuntimeError(f'Different {field} in the two arms')
    if base['sources'].keys() != patched['sources'].keys() or any(len(a['rows']) != 3 for a in (base, patched)):
        raise RuntimeError('Incomplete result set')
    numerical = []
    for a, b in zip(base['rows'], patched['rows']):
        if (a['op'], a['T'], a['BS']) != (b['op'], b['T'], b['BS']):
            raise RuntimeError('The two arms selected different workloads')
        for phase in ('forward', 'backward'):
            if a[phase]['key'] != b[phase]['key'] or a[phase]['candidates'] != b[phase]['candidates']:
                raise RuntimeError('Autotune key or search space changed')
            if a[phase]['NB'] > 1 and not (a[phase]['new_compiles'] > 0 and b[phase]['new_compiles'] == 0):
                raise AssertionError(f"Unexpected cross-bucket compilations for {phase} NB={a[phase]['NB']}: "
                                     f"{a[phase]['new_compiles']} -> {b[phase]['new_compiles']}")
        entry = {'op': a['op'], 'T': a['T']}
        for mode in ('autotuned_vs_reference', 'fixed_vs_reference'):
            bits = {k: v['sha256'] == b[mode][k]['sha256'] for k, v in a[mode].items()}
            if mode == 'fixed_vs_reference' and not all(bits.values()):
                raise AssertionError(f"Different full-tensor bits at BT=16, num_warps=4: {entry}, {bits}")
            entry[mode.split('_')[0] + '_bitwise'] = bits
        numerical.append(entry)
    return numerical


def main():
    global D, DTYPE, IS_RMS_NORM
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True, help=f'clean upstream checkout at {BASE}')
    parser.add_argument('--patched', type=Path, required=True, help='same source with the NB change applied')
    parser.add_argument('--output', type=Path, default=Path('gated-norm-buckets.json'))
    parser.add_argument('--hot-samples', type=int, default=20)
    parser.add_argument('--features', type=int, choices=(64, 128), default=64)
    parser.add_argument('--dtype', choices=('float32', 'bfloat16'), default='float32')
    parser.add_argument('--rms', action='store_true', help='benchmark RMSNorm with no bias')
    parser.add_argument('--reverse', action='store_true', help='run patched first for an order-balanced repeat')
    parser.add_argument('--worker', choices=('base', 'patched'), help=argparse.SUPPRESS)
    parser.add_argument('--work', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    D, DTYPE, IS_RMS_NORM = args.features, args.dtype, args.rms
    roots = {'base': args.base.resolve(strict=True), 'patched': args.patched.resolve(strict=True)}
    if args.hot_samples < 1:
        parser.error('--hot-samples must be positive')
    if args.worker:
        if args.work is None or not args.work.is_dir():
            parser.error('Worker requires an existing temporary directory')
        worker(roots[args.worker], args.work.resolve(strict=True), args.worker, args.hot_samples)
        return
    if git(roots['base'], 'rev-parse', 'HEAD') != BASE:
        parser.error(f'The baseline must be {BASE}')
    patched = roots['patched']
    if patched.joinpath('.git').exists():
        if git(patched, 'merge-base', BASE, 'HEAD') != BASE:
            parser.error(f'The patched tree must be derived from {BASE}')
    elif git(patched, 'rev-parse', 'HEAD') != BASE:
        parser.error(f'The patched archive must be derived from {BASE}')
    sources = {}
    for arm, root in roots.items():
        sources[arm] = {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.joinpath('fla').rglob('*.py')
        }
    if sources['base'].keys() != sources['patched'].keys():
        parser.error('The two source trees contain different Python files')
    differences = {name for name, digest in sources['base'].items() if digest != sources['patched'][name]}
    if differences != {SOURCE}:
        parser.error(f'Unexpected source differences: {sorted(differences)}')
    output = args.output.resolve()
    if not output.parent.is_dir() or output.exists():
        parser.error('Choose a new output filename in an existing directory')
    with tempfile.TemporaryDirectory(prefix='gated-norm-buckets-') as tmp:
        work = Path(tmp)
        for arm in (('patched', 'base') if args.reverse else ('base', 'patched')):
            env = dict(os.environ)
            for key in ('TRITON_CACHE_MANAGER', 'TRITON_ALWAYS_COMPILE', 'TRITON_KERNEL_OVERRIDE', 'TRITON_INTERPRET'):
                env.pop(key, None)
            env.update(
                PYTHONPATH=str(roots[arm]), PYTHONDONTWRITEBYTECODE='1', FLA_CI_ENV='0',
                FLA_CACHE_MODE='disabled', FLA_CACHE_RESULTS='0', FLA_DISABLE_BACKEND_DISPATCH='1',
                TRITON_HOME=str(work / arm), TRITON_CACHE_DIR=str(work / arm / 'triton'),
                CUDA_CACHE_PATH=str(work / arm / 'cuda'), TORCHINDUCTOR_CACHE_DIR=str(work / arm / 'inductor'),
            )
            (work / arm / 'cuda').mkdir(parents=True)
            cmd = [sys.executable, str(Path(__file__).resolve()), '--base', str(roots['base']), '--patched',
                   str(roots['patched']), '--hot-samples', str(args.hot_samples), '--work', tmp, '--worker', arm,
                   '--features', str(D), '--dtype', DTYPE, *(['--rms'] if IS_RMS_NORM else [])]
            subprocess.run(cmd, env=env, cwd=roots[arm], check=True, timeout=600)
        base, patched = (json.loads((work / f'{arm}.json').read_text()) for arm in ('base', 'patched'))
        report = {'base': base, 'patched': patched, 'run_order': 'patched,base' if args.reverse else 'base,patched',
                  'same_gpu_verified': bool(base['device_uuid']) and base['device_uuid'] == patched['device_uuid']}
        try:
            report['full_tensor_comparison'] = compare_arms(base, patched)
        except (AssertionError, RuntimeError) as error:
            report['comparison_error'] = str(error)
        del base['device_uuid'], patched['device_uuid']
        with output.open('x') as f:
            json.dump(report, f, indent=2, allow_nan=False)
            f.write('\n')
    print(f'Wrote {output}')
    if 'comparison_error' in report:
        raise SystemExit(report['comparison_error'])


if __name__ == '__main__':
    main()

# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Host checks; run with --confcutdir=tests/host without accelerator dependencies."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from runpy import run_path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_module(monkeypatch, name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def route(monkeypatch):
    for name in ('fla', 'fla.ops', 'fla.ops.utils', 'fla.ops.utils.backends'):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    torch = ModuleType('torch')
    torch.float = torch.float32 = 'float32'
    torch.float16, torch.bfloat16 = 'float16', 'bfloat16'
    torch.strided = 'strided'
    torch.compiler = SimpleNamespace(disable=lambda fn: fn)
    torch.is_grad_enabled = lambda: True
    torch.npu = SimpleNamespace(get_device_name=lambda _: 'Ascend950PR_9579')
    monkeypatch.setitem(sys.modules, 'torch', torch)
    utils = ModuleType('fla.utils')
    utils.IS_NPU = True
    utils.find_spec_cached = lambda _: None
    utils.has_usable_nvcc = lambda: False
    monkeypatch.setitem(sys.modules, 'fla.utils', utils)
    native = ModuleType('fla.utils.ascendc')
    native.is_available = lambda: True
    monkeypatch.setitem(sys.modules, 'fla.utils.ascendc', native)
    monkeypatch.delenv('FLA_DISABLE_BACKEND_DISPATCH', raising=False)
    dispatcher = load_module(monkeypatch, 'fla.backends', 'fla/backends.py')
    prefix = 'fla.ops.utils.backends.ascendc'
    adapter = load_module(monkeypatch, prefix + '.cumsum', 'fla/ops/utils/backends/ascendc/cumsum.py')
    backend = load_module(monkeypatch, prefix, 'fla/ops/utils/backends/ascendc/__init__.py').AscendCUtilsBackend
    monkeypatch.setenv('FLA_ASCENDC', '1')
    tensor = SimpleNamespace(device=SimpleNamespace(type='npu'), shape=(2, 129, 3), ndim=3,
                             dtype='float32', requires_grad=False, layout='strided')
    registry = dispatcher.BackendRegistry('utils')

    class TritonCandidate(dispatcher.TritonAscendBackend):
        def chunk_local_cumsum(self, *args, **kwargs):
            return 'fallback'

    registry.register(TritonCandidate())
    registry.register(backend())
    monkeypatch.setattr(adapter, 'chunk_local_cumsum', lambda *args, **kwargs: 'native')

    def chunk_local_cumsum(*args, **kwargs):
        raise AssertionError('Expected one of the registered NPU backends')

    return SimpleNamespace(call=registry.dispatch(chunk_local_cumsum), tensor=tensor, torch=torch,
                           native=native, backend=backend, adapter=adapter)


@pytest.mark.parametrize('dtype', ['float32', 'float16', 'bfloat16'])
@pytest.mark.parametrize('chunk_size', [64, 128])
def test_supported_route(route, dtype, chunk_size):
    route.tensor.dtype = dtype
    assert route.call(route.tensor, chunk_size) == 'native'


@pytest.mark.parametrize('kwargs', [
    {'reverse': True}, {'chunk_size': 32}, {'chunk_size': 64.0}, {'output_dtype': None}, {'output_dtype': 'float16'},
    {'cu_seqlens': object()}, {'chunk_indices': object()}, {'use_graph': True},
    {'head_first': True}, {'scale': float('inf')}, {'scale': float('nan')}, {'scale': 1e40},
])
def test_rejected_route(route, kwargs):
    assert route.call(route.tensor, **dict({'chunk_size': 64}, **kwargs)) == 'fallback'


@pytest.mark.parametrize('field,value', [('ndim', 4), ('shape', (1, 0, 2)), ('dtype', 'float64'), ('layout', 'sparse_coo'),
                                         ('requires_grad', True), ('device', SimpleNamespace(type='cpu'))])
def test_rejected_input(route, field, value):
    setattr(route.tensor, field, value)
    assert route.call(route.tensor, 64) == 'fallback'


def test_availability_and_opt_in(route, monkeypatch):
    monkeypatch.delenv('FLA_ASCENDC')
    assert route.call(route.tensor, 64) == 'fallback'
    monkeypatch.setenv('FLA_ASCENDC', '1')
    monkeypatch.setattr(route.native, 'is_available', lambda: False)
    assert route.call(route.tensor, 64) == 'fallback'
    monkeypatch.setattr(route.native, 'is_available', lambda: True)
    monkeypatch.setattr(route.torch.npu, 'get_device_name', lambda _: 'Ascend910B')
    assert route.call(route.tensor, 64) == 'fallback'


def test_execution_error_is_not_hidden(route, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError('launch failed')
    monkeypatch.setattr(route.adapter, 'chunk_local_cumsum', fail)
    with pytest.raises(RuntimeError, match='launch failed'):
        route.call(route.tensor, 64)


def test_native_build_rejects_unsupported_host(monkeypatch, tmp_path):
    build = run_path(str(ROOT / 'build_helpers/ascendc.py'))
    monkeypatch.setattr(build['platform'], 'system', lambda: 'Darwin')
    with pytest.raises(RuntimeError, match='require Linux'):
        build['build_native'](ROOT, tmp_path)
    assert not list(tmp_path.iterdir())


def test_ops_info_retains_case_and_optional_inputs(tmp_path):
    source = tmp_path / 'ops.ini'
    source.write_text('[ChunkLocalCumsum]\ninput0.name=g\ninput0.paramType=required\n'
                      'input1.paramType=optional\ndynamicShapeSupport.flag=true\n')
    output = tmp_path / 'config/ops.json'
    run_path(str(ROOT / 'build_helpers/ascendc.py'))['write_ops_info'](source, output)
    assert json.loads(output.read_text()) == {'ChunkLocalCumsum': {
        'input0': {'name': 'g', 'paramType': 'required'}, 'input1': {'paramType': 'optional'},
        'dynamicShapeSupport': {'flag': 'true'},
    }}


@pytest.mark.parametrize('enabled', ['0', '1'])
def test_startup_registers_opp_without_importing_torch(tmp_path, enabled):
    site = tmp_path / 'site'
    vendor = site / 'fla/utils/ascendc/_native/opp/vendors/fla_ascendc'
    vendor.mkdir(parents=True)
    (site / 'fla_ascendc_bootstrap.py').write_bytes((ROOT / 'build_helpers/bootstrap.py').read_bytes())
    (site / 'fla_ascendc.pth').write_text('import fla_ascendc_bootstrap\n')
    program = (
        'import json, os, site, sys; site.addsitedir(sys.argv[1]); '
        'print(json.dumps([os.environ.get("ASCEND_CUSTOM_OPP_PATH"), "torch" in sys.modules]))'
    )
    result = subprocess.run([sys.executable, '-S', '-c', program, str(site)], check=True, capture_output=True,
                            text=True, env={**os.environ, 'FLA_ASCENDC': enabled, 'ASCEND_CUSTOM_OPP_PATH': '/other/opp'})
    paths, torch_imported = json.loads(result.stdout)
    assert paths == (f'{vendor}{os.pathsep}/other/opp' if enabled == '1' else '/other/opp')
    assert not torch_imported


@pytest.mark.parametrize('missing', [False, True])
def test_kernel_index_references_packaged_files(tmp_path, missing):
    config = tmp_path / 'config/ascend950'
    config.mkdir(parents=True)
    relative = 'ascend950/chunk_local_cumsum/kernel'
    entry = {'binPath': relative + '.o', 'jsonPath': relative + '.json'}
    (config / 'binary_info_config.json').write_text(json.dumps({'ChunkLocalCumsum': {'binaryList': [entry]}}))
    for path in entry.values():
        artifact = tmp_path / path
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b'test fixture')
    validate = run_path(str(ROOT / 'build_helpers/ascendc.py'))['validate_kernel_index']
    if missing:
        (tmp_path / entry['binPath']).unlink()
        with pytest.raises(RuntimeError, match='missing or external'):
            validate(tmp_path, 'ascend950')
    else:
        validate(tmp_path, 'ascend950')


def test_native_wheel_staging_and_clean_default_rebuild(tmp_path):
    """Exercise setuptools with fixture payloads, without pretending to compile CANN."""
    import zipfile

    project = tmp_path / 'project'
    package = project / 'fla/utils/ascendc'
    package.mkdir(parents=True)
    (package / '__init__.py').write_bytes((ROOT / 'fla/utils/ascendc/__init__.py').read_bytes())
    program = '''
import os, sys
from pathlib import Path
from runpy import run_path
from setuptools import setup
root, base, output = map(Path, sys.argv[1:])
commands = run_path(str(root / 'build_helpers/ascendc.py'))['CMDCLASS']
def fixture_native(root, destination):
    payload = destination / '_native'
    payload.mkdir(parents=True, exist_ok=True)
    artifact = payload / 'libfla_ascendc.so'
    artifact.write_bytes(b'PACKAGING TEST FIXTURE - NOT AN ELF LIBRARY')
    return [str(artifact)]
commands['build_py'].run.__globals__['build_native'] = fixture_native
setup(name='fla-ascendc-packaging-fixture', version='0.0.0', packages=['fla.utils.ascendc'],
      cmdclass=commands, options={'build': {'build_base': str(base)}},
      script_args=['bdist_wheel', '--dist-dir', str(output)])
'''
    for enabled in ('1', '0'):
        output = tmp_path / enabled
        result = subprocess.run([sys.executable, '-c', program, str(ROOT), str(tmp_path / 'build'), str(output)],
                                cwd=project, env={**os.environ, 'FLA_BUILD_ASCENDC': enabled},
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        wheel = next(output.glob('*.whl'))
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            assert any(name.endswith('fla/utils/ascendc/_native/libfla_ascendc.so') for name in names) == (enabled == '1')
            assert any(name.endswith('fla_ascendc.pth') for name in names) == (enabled == '1')
            metadata = archive.read(next(name for name in names if name.endswith('.dist-info/WHEEL'))).decode()
            assert f'Root-Is-Purelib: {str(enabled == "0").lower()}' in metadata
            assert ('-py3-none-any.whl' in wheel.name) == (enabled == '0')

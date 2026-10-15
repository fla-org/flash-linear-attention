# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Build the optional dense cumsum backend without importing FLA."""

import configparser
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from setuptools.command.build_py import build_py

try:
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:
    from wheel.bdist_wheel import bdist_wheel


def enabled():
    return os.environ.get('FLA_BUILD_ASCENDC', '0') == '1'


def run(command, *, cwd=None, env=None):
    print('[FLA AscendC]', ' '.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, env=env, check=True)


def write_ops_info(source, destination):
    """Convert op_build's section/group.key INI into the OPP JSON schema."""
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    with source.open() as stream:
        parser.read_file(stream)
    result = {}
    for section in parser.sections():
        groups = result.setdefault(section, {})
        for key, value in parser.items(section):
            group, field = key.split('.', 1)
            groups.setdefault(group, {})[field] = value
    if 'ChunkLocalCumsum' not in result:
        raise RuntimeError('op_build did not emit ChunkLocalCumsum metadata')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + '\n')


def validate_kernel_index(kernel_root, soc):
    index = kernel_root / 'config' / soc / 'binary_info_config.json'
    if not index.is_file():
        raise RuntimeError('CANN output is missing binary_info_config.json')
    entries = json.loads(index.read_text()).get('ChunkLocalCumsum', {}).get('binaryList', [])
    if not entries:
        raise RuntimeError('CANN output has no ChunkLocalCumsum kernel entries')
    for entry in entries:
        for key in ('binPath', 'jsonPath'):
            path = (kernel_root / entry[key]).resolve()
            if not path.is_relative_to(kernel_root.resolve()) or not path.is_file():
                raise RuntimeError(f'CANN kernel index references a missing or external artifact: {entry[key]}')


def build_native(root, destination):
    destination = destination.resolve()
    if platform.system() != 'Linux':
        raise RuntimeError('AscendC builds require Linux and the CANN development toolkit')
    soc = os.environ.get('FLA_ASCENDC_SOC', '')
    if soc != 'ascend950':
        raise RuntimeError('This PoC requires FLA_ASCENDC_SOC=ascend950')
    if os.environ.get('FLA_ASCENDC_OPS', 'chunk_local_cumsum') != 'chunk_local_cumsum':
        raise RuntimeError('This PoC builds only chunk_local_cumsum')
    cann_value = os.environ.get('ASCEND_HOME_PATH')
    if not cann_value:
        raise RuntimeError('Source the CANN set_env.sh before building AscendC')
    cann = Path(cann_value).resolve()
    utilities = cann / 'tools/op_project_templates/ascendc/common/util'
    if not (utilities / 'ascendc_impl_build.py').is_file():
        raise RuntimeError(f'CANN AscendC generation tools are missing under {utilities}')
    import torch

    jobs = max(1, int(os.environ.get('MAX_JOBS', '4')))
    op = root / 'fla/ops/utils/backends/ascendc/csrc/chunk_local_cumsum'
    native = root / 'csrc/ascendc'
    sources = sorted(p for folder in (native, op) for p in folder.rglob('*') if p.is_file())
    digest = hashlib.sha256()
    for path in sources:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    stamp = digest.hexdigest()
    build = root / 'build/ascendc'
    build.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    # custom OPPs from an existing installation must not enter this build.
    env.pop('ASCEND_CUSTOM_OPP_PATH', None)
    env.pop('FLA_NPU_OP_API_LIB', None)
    env.pop('FLA_ASCENDC_OP_API_LIB', None)
    run([
        'cmake', '-S', native, '-B', build, f'-DFLA_SOURCE={root}', f'-DCANN={cann}',
        f'-DTORCH_LIB={Path(torch.__file__).parent / "lib"}', f'-DSOURCE_HASH={stamp}',
    ], env=env)
    run(['cmake', '--build', build, '--parallel', jobs], env=env)
    autogen = build / 'autogen'
    autogen.mkdir(exist_ok=True)
    opbuild_env = dict(env, OPS_PROTO_SEPARATE='1', OPS_ACLNN_GEN='0', OPS_PROJECT_NAME='aclnn')
    run([cann / 'tools/opbuild/op_build', build / 'libfla_opdef.so', autogen], env=opbuild_env)
    run(['cmake', '--build', build, '--target', 'fla_proto', '--parallel', jobs], env=env)

    payload = destination / '_native'
    if payload.exists():
        shutil.rmtree(payload)
    vendor = payload / 'opp/vendors/fla_ascendc'
    tiling_dir = vendor / 'op_impl/ai_core/tbe/op_tiling'
    host_arch = platform.machine()
    for source, relative in (
        ('libcust_opapi.so', 'op_api/lib/libcust_opapi.so'),
        ('libcust_opmaster_rt2.0.so', f'op_impl/ai_core/tbe/op_tiling/lib/linux/{host_arch}/libcust_opmaster_rt2.0.so'),
        ('libcust_opsproto_rt2.0.so', f'op_proto/lib/linux/{host_arch}/libcust_opsproto_rt2.0.so'),
    ):
        target = vendor / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(build / source, target)
    # a regular file also survives wheel unpacking, which need not preserve symlinks.
    shutil.copy2(build / 'libcust_opmaster_rt2.0.so', tiling_dir / 'liboptiling.so')
    (vendor.parent / 'config.ini').write_text('load_priority=fla_ascendc\n')
    env['ASCEND_CUSTOM_OPP_PATH'] = str(vendor)

    impl = build / 'impl'
    impl.mkdir(exist_ok=True)
    info = autogen / f'aic-{soc}-ops-info.ini'
    write_ops_info(info, vendor / f'op_impl/ai_core/tbe/config/{soc}/aic-{soc}-ops-info.json')
    run([sys.executable, utilities / 'ascendc_impl_build.py', info, '', '', '', impl, autogen], env=env)
    dynamic = impl / 'dynamic'
    shutil.copytree(op / 'op_kernel', impl / 'ascendc/chunk_local_cumsum', dirs_exist_ok=True)
    gen = build / 'kernel_commands'
    if gen.exists():
        shutil.rmtree(gen)
    gen.mkdir()
    run([sys.executable, utilities / 'ascendc_bin_param_build.py', info, gen, soc], env=env)
    binaries = build / 'kernel' / soc
    if binaries.exists():
        shutil.rmtree(binaries)
    binaries.mkdir(parents=True)
    op_binaries = binaries / 'chunk_local_cumsum'
    op_binaries.mkdir()
    scripts = sorted(gen.glob('*.sh'))
    if not scripts:
        raise RuntimeError('CANN generated no kernel compile commands')
    env.update(TILINGKEY_PAR_COMPILE='1', BIN_FILENAME_HASHED='1')

    def compile_kernel(script):
        run(['bash', script, dynamic / 'chunk_local_cumsum.py', op_binaries], cwd=gen, env=env)

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(compile_kernel, scripts))
    run([sys.executable, utilities / 'ascendc_ops_config.py', '-p', binaries, '-s', soc], env=env)
    kernel_root = vendor / 'op_impl/ai_core/tbe/kernel'
    shutil.copytree(binaries, kernel_root / soc)
    config = kernel_root / 'config' / soc
    config.mkdir(parents=True)
    for path in binaries.glob('*.json'):
        shutil.copy2(path, config / path.name)
    validate_kernel_index(kernel_root, soc)
    shutil.copy2(build / 'libfla_ascendc.so', payload / 'libfla_ascendc.so')
    metadata = {
        'soc': soc, 'ops': ['chunk_local_cumsum'], 'source_hash': stamp,
        'host_arch': host_arch, 'torch_version': torch.__version__,
    }
    (payload / 'build.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return sorted(str(p) for p in payload.rglob('*') if p.is_file())


class AscendCBuildPy(build_py):
    def run(self):
        if enabled() and self.editable_mode:
            raise RuntimeError('The AscendC PoC requires a wheel install; editable native builds are unsupported')
        super().run()
        self.native_outputs = []
        root = Path(__file__).resolve().parents[1]
        package = Path('fla/utils/ascendc')
        destination = (root if self.editable_mode else Path(self.build_lib)) / package
        if enabled():
            self.native_outputs = build_native(root, destination)
            bootstrap = Path(self.build_lib) / 'fla_ascendc_bootstrap.py'
            shutil.copy2(root / 'build_helpers/bootstrap.py', bootstrap)
            startup = Path(self.build_lib) / 'fla_ascendc.pth'
            startup.write_text('import fla_ascendc_bootstrap\n')
            self.native_outputs.extend([str(bootstrap), str(startup)])
        elif not self.editable_mode and (destination / '_native').exists():
            shutil.rmtree(destination / '_native')
        if not enabled() and not self.editable_mode:
            for name in ('fla_ascendc_bootstrap.py', 'fla_ascendc.pth'):
                (Path(self.build_lib) / name).unlink(missing_ok=True)

    def get_outputs(self, include_bytecode=1):
        return super().get_outputs(include_bytecode) + getattr(self, 'native_outputs', [])


class AscendCBdistWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        if enabled():
            self.root_is_pure = False

    def get_tag(self):
        if enabled():
            return 'py3', 'none', sysconfig.get_platform().replace('-', '_').replace('.', '_')
        return super().get_tag()


CMDCLASS = {'build_py': AscendCBuildPy, 'bdist_wheel': AscendCBdistWheel}

# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import os
import subprocess
import sys
import textwrap
import venv
import zipfile
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import parse as parse_version

from scripts.build_packages import build_split_packages
from scripts.smoke_test_split_packages import find_wheel, read_wheel_metadata, venv_python

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    subprocess_env = os.environ.copy()
    subprocess_env.pop("PYTHONPATH", None)
    subprocess_env["PYTHONNOUSERSITE"] = "1"
    if env:
        subprocess_env.update(env)

    result = subprocess.run(
        cmd,
        check=False,
        cwd=cwd,
        env=subprocess_env,
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert result.returncode == 0, result.stdout


def _build_split_wheels(tmp_path: Path) -> tuple[Path, Path, str]:
    dist_dir, version = build_split_packages(tmp_path / "split")
    wheel_dir = tmp_path / "wheelhouse"
    wheel_dir.mkdir()

    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-index",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheel_dir),
            str(dist_dir / "fla-core"),
            str(dist_dir / "flash-linear-attention"),
        ],
        cwd=ROOT,
    )

    return find_wheel(wheel_dir, "fla_core"), find_wheel(wheel_dir, "flash_linear_attention"), version


def _wheel_names(wheel: Path) -> set[str]:
    with zipfile.ZipFile(wheel) as zf:
        return set(zf.namelist())


def _requires_by_name(wheel: Path) -> dict[str, Requirement]:
    return {
        requirement.name: requirement
        for requirement in _requires_dist_entries(wheel)
    }


def _requires_dist_entries(wheel: Path) -> list[Requirement]:
    return [Requirement(value) for value in read_wheel_metadata(wheel).get_all("Requires-Dist") or []]


def _base_requirement_names(wheel: Path) -> set[str]:
    return {requirement.name for requirement in _requires_dist_entries(wheel) if requirement.marker is None}


def _public_api_contract() -> str:
    return textwrap.dedent(
        """
        import fla
        exported = bool(fla.__all__)
        assert hasattr(fla, 'layers') == exported
        assert hasattr(fla, 'models') == exported
        if exported:
            assert 'GLAModel' in fla.__all__
            assert 'GatedLinearAttention' in fla.__all__
            assert 'GLAConfig' not in fla.__all__
            assert not hasattr(fla, 'GLAConfig')
        """
    ).strip()


def _create_venv(tmp_path: Path, *, system_site_packages: bool = False) -> Path:
    venv_dir = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True, system_site_packages=system_site_packages).create(venv_dir)
    return venv_python(venv_dir)


def test_split_wheels_match_release_contract(tmp_path: Path) -> None:
    core_wheel, ext_wheel, version = _build_split_wheels(tmp_path)

    core_names = _wheel_names(core_wheel)
    ext_names = _wheel_names(ext_wheel)

    assert "fla/__init__.py" in core_names
    assert "fla/ops/__init__.py" in core_names
    assert "fla/modules/__init__.py" in core_names
    assert "fla/utils/__init__.py" in core_names
    assert "fla/utils/_device.py" in core_names
    assert "fla/utils.py" not in core_names
    assert not any(name.startswith("fla/layers/") for name in core_names)
    assert not any(name.startswith("fla/models/") for name in core_names)

    assert "fla/__init__.py" not in ext_names
    assert "fla/layers/__init__.py" in ext_names
    assert "fla/models/__init__.py" in ext_names
    assert not any(name.startswith("fla/ops/") for name in ext_names)
    assert not any(name.startswith("fla/modules/") for name in ext_names)
    assert not any(name.startswith("fla/utils/") for name in ext_names)
    assert "fla/utils.py" not in ext_names

    assert read_wheel_metadata(core_wheel)["Version"] == version
    assert read_wheel_metadata(ext_wheel)["Version"] == version
    core_requires = _requires_by_name(core_wheel)
    ext_requires = _requires_by_name(ext_wheel)
    assert _base_requirement_names(core_wheel) == {'einops'}
    assert str(core_requires['torch'].specifier) == '>=2.7.0'
    assert core_requires['torch'].marker is not None
    assert str(core_requires['triton'].specifier) == '>=3.3'
    assert core_requires['triton'].marker is not None
    assert str(core_requires['einops'].specifier) == ''
    assert _base_requirement_names(ext_wheel) == {'fla-core', 'transformers'}
    assert str(ext_requires['fla-core'].specifier) == f'=={version}'
    assert str(ext_requires['transformers'].specifier) == '>=4.45.0'


def test_core_then_extension_install_sequence_without_runtime_deps(tmp_path: Path) -> None:
    core_wheel, ext_wheel, version = _build_split_wheels(tmp_path)
    python = _create_venv(tmp_path)

    _run([sys.executable, "scripts/smoke_test_split_packages.py", str(core_wheel.parent)], cwd=ROOT)

    _run([str(python), "-m", "pip", "install", "--no-index", "--no-deps", str(core_wheel)], cwd=tmp_path)
    core_check = textwrap.dedent(
        f"""
        import importlib.util
        import sys

        import fla

        assert fla.__version__ == {version!r}
        assert fla.__all__ == []
        assert not hasattr(fla, "layers")
        assert not hasattr(fla, "models")
        assert importlib.util.find_spec("fla.ops") is not None
        assert importlib.util.find_spec("fla.modules") is not None
        assert importlib.util.find_spec("fla.utils") is not None
        assert importlib.util.find_spec("fla.layers") is None
        assert importlib.util.find_spec("fla.models") is None
        assert "fla.layers" not in sys.modules
        assert "fla.models" not in sys.modules

        try:
            from fla import GLAModel
        except ImportError:
            pass
        else:
            raise AssertionError("core-only install must not expose model classes")
        """
    )
    _run([str(python), "-c", core_check], cwd=tmp_path)

    _run([str(python), "-m", "pip", "install", "--no-index", "--no-deps", str(ext_wheel)], cwd=tmp_path)
    extension_check = textwrap.dedent(
        """
        import importlib.metadata as metadata

        core_files = {str(file) for file in metadata.files("fla-core")}
        ext_files = {str(file) for file in metadata.files("flash-linear-attention")}

        assert "fla/__init__.py" in core_files
        assert "fla/utils/__init__.py" in core_files
        assert "fla/layers/__init__.py" in ext_files
        assert "fla/models/__init__.py" in ext_files
        """
    ) + "\n" + _public_api_contract()
    _run([str(python), "-c", extension_check], cwd=tmp_path)


def test_core_only_import_with_repo_root_on_path(tmp_path: Path) -> None:
    """fla-core installed alone should still be importable from the repo root.

    This guards against the case where the source tree (containing fla/layers/ and
    fla/models/) is on sys.path but the extension package is not installed and the
    runtime deps (torch/triton/transformers) are absent. The top-level fla package
    must gracefully skip the optional extension rather than surfacing an import
    error from inside fla.layers.
    """
    core_wheel, _, version = _build_split_wheels(tmp_path)
    python = _create_venv(tmp_path)

    _run([str(python), "-m", "pip", "install", "--no-index", "--no-deps", str(core_wheel)], cwd=tmp_path)

    # Put the repo root on PYTHONPATH (simulates running from the project directory
    # or using an editable-src checkout) and try to import fla with no torch/triton.
    core_check = textwrap.dedent(
        f"""
        import importlib.util
        import sys

        assert importlib.util.find_spec('fla.layers') is not None
        assert importlib.util.find_spec('fla.models') is not None

        import fla

        assert fla.__version__ == {version!r}
        assert fla.__all__ == []
        assert not hasattr(fla, 'layers')
        assert not hasattr(fla, 'models')
        assert 'fla.layers' not in sys.modules
        assert 'fla.models' not in sys.modules

        try:
            from fla import GLAModel
        except ImportError:
            pass
        else:
            raise AssertionError('core-only install must not expose model classes')
        """
    )
    _run([str(python), '-c', core_check], cwd=ROOT, env={'PYTHONPATH': str(ROOT)})


def test_split_namespace_across_sys_path_entries_without_runtime_deps(tmp_path: Path) -> None:
    core_wheel, ext_wheel, _ = _build_split_wheels(tmp_path)
    python = _create_venv(tmp_path)
    core_target = tmp_path / "core-target"
    ext_target = tmp_path / "ext-target"

    _run(
        [str(python), "-m", "pip", "install", "--no-index", "--no-deps", "--target", str(core_target), str(core_wheel)],
        cwd=tmp_path,
    )
    _run(
        [str(python), "-m", "pip", "install", "--no-index", "--no-deps", "--target", str(ext_target), str(ext_wheel)],
        cwd=tmp_path,
    )

    for first, second in ((core_target, ext_target), (ext_target, core_target)):
        check = textwrap.dedent(
            f"""
            import sys

            sys.path[:0] = [{str(first)!r}, {str(second)!r}]
            """
        ) + "\n" + _public_api_contract()
        _run([str(python), "-c", check], cwd=tmp_path)


def test_full_split_import_contract_when_runtime_dependencies_available(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    pytest.importorskip("transformers")
    pytest.importorskip("einops")
    if parse_version(torch.__version__.split("+", maxsplit=1)[0]) < parse_version("2.7.0"):
        pytest.skip("full split import smoke requires torch>=2.7.0")

    core_wheel, ext_wheel, version = _build_split_wheels(tmp_path)
    python = _create_venv(tmp_path, system_site_packages=True)

    _run([str(python), "-m", "pip", "install", "--no-index", "--no-deps", str(core_wheel)], cwd=tmp_path)
    _run([str(python), "-m", "pip", "install", "--no-index", "--no-deps", str(ext_wheel)], cwd=tmp_path)
    full_check = textwrap.dedent(
        f"""
        import fla
        import fla.layers
        import fla.models
        from fla import GLAModel, GatedLinearAttention
        from fla.models import GLAConfig, GLAForCausalLM, GLAModel as ModelsGLAModel
        from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

        assert fla.__version__ == {version!r}
        assert "GLAModel" in fla.__all__
        assert "GatedLinearAttention" in fla.__all__
        assert "GLAConfig" not in fla.__all__
        assert not hasattr(fla, "GLAConfig")
        assert GLAModel is ModelsGLAModel
        assert GatedLinearAttention is fla.layers.GatedLinearAttention
        assert isinstance(AutoConfig.for_model(GLAConfig.model_type), GLAConfig)
        config = GLAConfig(hidden_size=16, hidden_ratio=1, num_hidden_layers=1)
        assert isinstance(AutoModel.from_config(config), ModelsGLAModel)
        assert isinstance(AutoModelForCausalLM.from_config(config), GLAForCausalLM)
        """
    )
    _run([str(python), "-c", full_check], cwd=tmp_path)

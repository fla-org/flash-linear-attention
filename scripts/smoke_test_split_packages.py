# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import argparse
import os
import subprocess
import tempfile
import textwrap
import venv
import zipfile
from email.message import Message
from email.parser import Parser
from pathlib import Path


def run(
    cmd: list[str],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 600,
) -> None:
    subprocess.run(cmd, check=True, cwd=cwd, env=env, timeout=timeout)


def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def find_wheel(dist_dir: Path, prefix: str) -> Path:
    wheels = sorted(dist_dir.glob(f"{prefix}-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected one {prefix} wheel in {dist_dir}, found {len(wheels)}")
    return wheels[0]


def read_wheel_metadata(wheel: Path) -> Message:
    with zipfile.ZipFile(wheel) as zf:
        metadata_file = next(name for name in zf.namelist() if name.endswith(".dist-info/METADATA"))
        return Parser().parsestr(zf.read(metadata_file).decode())


def python_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def assert_core_only_import(python: Path, version: str) -> None:
    code = textwrap.dedent(
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
        """
    )
    run([str(python), "-c", code], cwd=tempfile.gettempdir(), env=python_env())


def assert_full_import(python: Path, version: str) -> None:
    code = textwrap.dedent(
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
    run([str(python), "-c", code], cwd=tempfile.gettempdir(), env=python_env())


def assert_split_namespace_installed(python: Path) -> None:
    code = textwrap.dedent(
        """
        import importlib.metadata as metadata

        core_files = {str(file) for file in metadata.files("fla-core")}
        ext_files = {str(file) for file in metadata.files("flash-linear-attention")}

        assert "fla/__init__.py" in core_files
        assert "fla/ops/__init__.py" in core_files
        assert "fla/modules/__init__.py" in core_files
        assert "fla/utils/__init__.py" in core_files
        assert "fla/layers/__init__.py" in ext_files
        assert "fla/models/__init__.py" in ext_files

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
    )
    run([str(python), "-c", code], cwd=tempfile.gettempdir(), env=python_env())


def install_wheel(python: Path, wheel: Path, *, with_deps: bool) -> None:
    cmd = [str(python), "-m", "pip", "install"]
    if not with_deps:
        cmd.extend(["--no-index", "--no-deps"])
    cmd.append(str(wheel))
    run(cmd, timeout=900)


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test locally built split package wheels.")
    parser.add_argument("dist_dir", type=Path, help="directory containing split package wheels")
    parser.add_argument("--version", help="expected package version")
    parser.add_argument("--with-deps", action="store_true", help="install runtime dependencies and run full import smoke")
    args = parser.parse_args()

    core_wheel = find_wheel(args.dist_dir, "fla_core")
    ext_wheel = find_wheel(args.dist_dir, "flash_linear_attention")
    core_metadata = read_wheel_metadata(core_wheel)
    ext_metadata = read_wheel_metadata(ext_wheel)
    version = args.version or core_metadata["Version"]
    if core_metadata["Version"] != version or ext_metadata["Version"] != version:
        raise RuntimeError(
            f"Wheel version mismatch: fla-core={core_metadata['Version']}, "
            f"flash-linear-attention={ext_metadata['Version']}, expected={version}"
        )

    with tempfile.TemporaryDirectory(prefix="fla-release-smoke-") as tmp:
        venv_dir = Path(tmp) / "venv"
        venv.EnvBuilder(with_pip=True).create(venv_dir)
        python = venv_python(venv_dir)

        install_wheel(python, core_wheel, with_deps=args.with_deps)
        assert_core_only_import(python, version)

        install_wheel(python, ext_wheel, with_deps=args.with_deps)
        assert_split_namespace_installed(python)
        if args.with_deps:
            run([str(python), "-m", "pip", "check"])
            assert_full_import(python, version)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

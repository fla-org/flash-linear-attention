# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import re
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

BACKEND_EXTRAS = {'cuda', 'rocm', 'xpu', 'npu', 'cpu'}


def extract_dependencies():
    """Read base dependencies and optional-dependency extras from pyproject.toml."""
    script_dir = Path(__file__).parent
    pyproject = script_dir.parent / 'pyproject.toml'

    with open(pyproject, 'rb') as f:
        data = tomllib.load(f)
    project = data.get('project', {})
    return list(project.get('dependencies', [])), dict(project.get('optional-dependencies', {}))


def categorize_dependencies(deps):
    """Split base deps: einops travels with the kernels, everything else with layers/models."""
    core_deps = []
    ext_deps = []

    for dep in deps:
        if 'einops' in dep:
            core_deps.append(dep)
        else:
            ext_deps.append(dep)

    return core_deps, ext_deps


def create_pyproject_toml(package_dir, name, version, dependencies, extras=None):
    """Create pyproject.toml for a package."""
    if extras is None:
        extras = {}

    extras_content = ""
    if extras:
        extras_content = "\n[project.optional-dependencies]\n"
        for key, values in extras.items():
            values_str = ', '.join(f'"{v}"' for v in values)
            extras_content += f"{key} = [{values_str}]\n"

    deps_content = ', '.join(f'"{dep}"' for dep in dependencies)

    # create description text
    if name == 'fla-core':
        desc_text = 'Core operations for flash-linear-attention'
    else:
        desc_text = 'Fast linear attention models and layers'

    content = f"""[build-system]
requires = ["setuptools>=64", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "{name}"
version = "{version}"
description = "{desc_text}"
readme = "README.md"
license = {{file = "LICENSE"}}
requires-python = ">=3.10"
dependencies = [{deps_content}]

[project.urls]
Homepage = "https://github.com/fla-org/flash-linear-attention"
Repository = "https://github.com/fla-org/flash-linear-attention"
"""

    content += extras_content

    # both split packages contribute subpackages under the shared `fla`
    # namespace, so both generated pyproject.toml files need namespace discovery.
    content += """

[tool.setuptools.packages.find]
include = ["fla*"]
namespaces = true
"""

    with open(package_dir / 'pyproject.toml', 'w') as f:
        f.write(content)


def build_split_packages(output_dir: str | Path | None = None):
    """Build split packages with proper dependency management."""
    # get script directory and find files relative to it
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent

    # get current version
    init_file = root_dir / 'fla' / '__init__.py'
    with open(init_file, encoding='utf-8') as f:
        content = f.read()
    version_match = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]\s*$", content, re.MULTILINE)
    if not version_match:
        raise RuntimeError(f"Could not find __version__ in {init_file}")
    version = version_match.group(1)

    # extract dependencies
    all_deps, extras = extract_dependencies()
    core_deps, ext_deps = categorize_dependencies(all_deps)
    core_extras = {k: v for k, v in extras.items() if k in BACKEND_EXTRAS}
    # extension forwards backend extras to fla-core so flash-linear-attention[cuda]
    # resolves the same torch/triton flavor as fla-core[cuda].
    ext_extras = {k: [f'fla-core[{k}]=={version}'] for k in extras if k in BACKEND_EXTRAS}
    ext_extras.update({k: v for k, v in extras.items() if k not in BACKEND_EXTRAS})

    # add version constraint for fla-core in extension package
    ext_deps.insert(0, f'fla-core=={version}')

    # create output directory
    if output_dir is None:
        output_dir = script_dir / 'dist'
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # create fla-core package
    core_dir = output_dir / 'fla-core'
    if core_dir.exists():
        shutil.rmtree(core_dir)
    core_dir.mkdir()

    # copy core files
    fla_core = core_dir / 'fla'
    ignore = shutil.ignore_patterns('__pycache__')
    shutil.copytree(root_dir / 'fla' / 'ops', fla_core / 'ops', ignore=ignore)
    shutil.copytree(root_dir / 'fla' / 'modules', fla_core / 'modules', ignore=ignore)
    shutil.copytree(root_dir / 'fla' / 'utils', fla_core / 'utils', ignore=ignore)

    # keep the source and split-wheel top-level import contract identical.
    # the source __init__.py is split-package-safe: it always exposes core
    # metadata, extends the namespace path, and only exports layers/models
    # when the extension package is installed.
    shutil.copy(root_dir / 'fla' / '__init__.py', fla_core / '__init__.py')

    # copy ancillary files (README.md, LICENSE) to core package
    for fname in ("README.md", "LICENSE"):
        src = root_dir / fname
        if src.exists():
            shutil.copy(src, core_dir / fname)

    # create fla-core configs
    create_pyproject_toml(core_dir, 'fla-core', version, core_deps, core_extras)

    # create flash-linear-attention package
    ext_dir = output_dir / 'flash-linear-attention'
    if ext_dir.exists():
        shutil.rmtree(ext_dir)
    ext_dir.mkdir()

    # copy extension files
    fla_ext = ext_dir / 'fla'
    shutil.copytree(root_dir / 'fla' / 'models', fla_ext / 'models', ignore=ignore)
    shutil.copytree(root_dir / 'fla' / 'layers', fla_ext / 'layers', ignore=ignore)

    # intentionally do not create fla/__init__.py in the extension package.
    # the top-level package is provided by fla-core (namespace via pkgutil).

    # copy ancillary files (README.md, LICENSE) to extension package
    for fname in ("README.md", "LICENSE"):
        src = root_dir / fname
        if src.exists():
            shutil.copy(src, ext_dir / fname)

    # create extension configs
    create_pyproject_toml(ext_dir, 'flash-linear-attention', version, ext_deps, ext_extras)

    # create build script
    build_script = output_dir / 'build.sh'
    with open(build_script, 'w') as f:
        f.write("""#!/bin/bash
# build both packages

echo "Building fla-core..."
cd fla-core
pip install -U build
python -m build

echo "Building flash-linear-attention..."
cd ../flash-linear-attention
python -m build

echo "Build complete! Packages in dist/"
""")

    build_script.chmod(0o755)

    print(f"✅ Split packages created in {output_dir}")
    print(f"✅ fla-core dependencies: {len(core_deps)} packages")
    print(f"✅ flash-linear-attention dependencies: {len(ext_deps)} packages")
    print(f"✅ Version: {version}")

    return output_dir, version


def build_packages(dist_dir):
    """Build wheels and source distributions for both packages."""
    print("Building packages...")

    # build fla-core (both wheel and sdist)
    print("Building fla-core packages...")
    try:
        subprocess.run(
            [sys.executable, "-m", "build", str(dist_dir / "fla-core")],
            check=True,
            timeout=1800,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print("Failed to build fla-core packages:")
        print(e.stdout)
        return False
    except subprocess.TimeoutExpired:
        print("Timed out building fla-core packages")
        return False

    # build flash-linear-attention (both wheel and sdist)
    print("Building flash-linear-attention packages...")
    try:
        subprocess.run(
            [sys.executable, "-m", "build", str(dist_dir / "flash-linear-attention")],
            check=True,
            timeout=1800,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print("Failed to build flash-linear-attention packages:")
        print(e.stdout)
        return False
    except subprocess.TimeoutExpired:
        print("Timed out building flash-linear-attention packages")
        return False

    print("✅ Packages built successfully")
    return True


def copy_packages_to_output(dist_dir):
    """Copy wheels and source distributions to output directory."""
    # get script directory (relative to this file)
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent

    # create output directory (relative to root)
    output_dir = root_dir / 'dist-packages'
    output_dir.mkdir(exist_ok=True)

    # find wheels and source distributions
    core_wheels = list((dist_dir / 'fla-core' / 'dist').glob('*.whl'))
    core_sdist = list((dist_dir / 'fla-core' / 'dist').glob('*.tar.gz'))
    ext_wheels = list((dist_dir / 'flash-linear-attention' / 'dist').glob('*.whl'))
    ext_sdist = list((dist_dir / 'flash-linear-attention' / 'dist').glob('*.tar.gz'))

    if not core_wheels:
        print("No fla-core wheel found")
        return False
    if not ext_wheels:
        print("No flash-linear-attention wheel found")
        return False

    # copy all packages to output directory
    all_packages = core_wheels + core_sdist + ext_wheels + ext_sdist
    for package in all_packages:
        target = output_dir / package.name
        shutil.copy2(package, target)
        if package.suffix == ".whl":
            package_type = "wheel"
        elif package.suffixes[-2:] == [".tar", ".gz"]:
            package_type = "sdist"
        else:
            package_type = "source"
        print(f"📦 Copied {package_type} package {package.name} to {output_dir}")

    print(f"\n✅ All packages copied to: {output_dir}")
    print("You can install wheels with:")
    print("  pip install dist-packages/*.whl")
    print("Source distributions are also available in:", output_dir)

    return True


def main():
    """Build split packages and copy to target directory."""

    print("Building split packages...")

    # build the split packages
    dist_dir, _ = build_split_packages()

    print("\nTo build packages manually:")
    print(f"cd {dist_dir}")
    print("./build.sh")

    # build packages (wheels and source distributions)
    if not build_packages(dist_dir):
        return 1

    # copy packages to output directory
    if not copy_packages_to_output(dist_dir):
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

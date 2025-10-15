#!/usr/bin/env python3
"""Build split packages with proper dependency management and copy them to target directory."""

import ast
import re
import shutil
import subprocess
import sys
from pathlib import Path


def extract_dependencies():
    """Extract dependencies from setup.py using AST."""
    # Get script directory and find setup.py in parent directory
    script_dir = Path(__file__).parent
    setup_py = script_dir.parent / 'setup.py'

    with open(setup_py, encoding='utf-8') as f:
        tree = ast.parse(f.read(), filename=str(setup_py))

    all_deps = []
    extras = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, 'id', '') == 'setup':
            for keyword in node.keywords:
                if (keyword.arg == 'install_requires' and
                        isinstance(keyword.value, (ast.List, ast.Tuple))):
                    all_deps.extend([
                        elt.value for elt in keyword.value.elts
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                    ])
                elif (keyword.arg == 'extras_require' and
                      isinstance(keyword.value, ast.Dict)):
                    for key_node, val_node in zip(keyword.value.keys, keyword.value.values, strict=False):
                        if (isinstance(key_node, ast.Constant) and
                            isinstance(key_node.value, str) and
                                isinstance(val_node, (ast.List, ast.Tuple))):
                            key = key_node.value
                            values = [
                                elt.value for elt in val_node.elts
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                            ]
                            extras[key] = values
            break  # Assume only one setup() call

    return all_deps, extras


def categorize_dependencies(deps):
    """Categorize dependencies based on core vs extension."""
    core_deps = []
    ext_deps = []

    for dep in deps:
        if any(core in dep for core in ['torch', 'einops']):
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

    # Create description text
    if name == 'fla-core':
        desc_text = 'Core operations for flash-linear-attention'
    else:
        desc_text = 'Fast linear attention models and layers'

    content = f"""[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "{name}"
version = "{version}"
description = "{desc_text}"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [{deps_content}]

[project.urls]
Homepage = "https://github.com/fla-org/flash-linear-attention"
Repository = "https://github.com/fla-org/flash-linear-attention"
"""

    content += extras_content

    # Add setuptools namespace package configuration for extension package
    if name == 'flash-linear-attention':
        content += """

[tool.setuptools.packages.find]
include = ["fla*"]
namespaces = true
"""

    with open(package_dir / 'pyproject.toml', 'w') as f:
        f.write(content)


def build_split_packages():
    """Build split packages with proper dependency management."""
    # Get script directory and find files relative to it
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent

    # Get current version
    init_file = root_dir / 'fla' / '__init__.py'
    with open(init_file, encoding='utf-8') as f:
        content = f.read()
    version_match = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]\s*$", content, re.MULTILINE)
    if not version_match:
        raise RuntimeError(f"Could not find __version__ in {init_file}")
    version = version_match.group(1)

    # Extract dependencies
    all_deps, extras = extract_dependencies()
    core_deps, ext_deps = categorize_dependencies(all_deps)

    # Add version constraint for fla-core in extension package
    ext_deps.insert(0, f'fla-core=={version}')

    # Create output directory
    output_dir = script_dir / 'dist'
    output_dir.mkdir(exist_ok=True)

    # Create fla-core package
    core_dir = output_dir / 'fla-core'
    if core_dir.exists():
        shutil.rmtree(core_dir)
    core_dir.mkdir()

    # Copy core files
    fla_core = core_dir / 'fla'
    shutil.copytree(root_dir / 'fla' / 'ops', fla_core / 'ops')
    shutil.copytree(root_dir / 'fla' / 'modules', fla_core / 'modules')
    shutil.copy(root_dir / 'fla' / 'utils.py', fla_core / 'utils.py')

    # Create fla-core __init__.py
    with open(fla_core / '__init__.py', 'w') as f:
        f.write(f"""# -*- coding: utf-8 -*-

__path__ = __import__('pkgutil').extend_path(__path__, __name__)
__version__ = '{version}'
""")

    # Copy ancillary files (README.md, LICENSE) to core package
    for fname in ("README.md", "LICENSE"):
        src = root_dir / fname
        if src.exists():
            shutil.copy(src, core_dir / fname)

    # Create fla-core configs
    create_pyproject_toml(core_dir, 'fla-core', version, core_deps)

    # Create flash-linear-attention package
    ext_dir = output_dir / 'flash-linear-attention'
    if ext_dir.exists():
        shutil.rmtree(ext_dir)
    ext_dir.mkdir()

    # Copy extension files
    fla_ext = ext_dir / 'fla'
    shutil.copytree(root_dir / 'fla' / 'models', fla_ext / 'models')
    shutil.copytree(root_dir / 'fla' / 'layers', fla_ext / 'layers')

    # Intentionally do NOT create fla/__init__.py in the extension package.
    # The top-level package is provided by fla-core (namespace via pkgutil).

    # Copy ancillary files (README.md, LICENSE) to extension package
    for fname in ("README.md", "LICENSE"):
        src = root_dir / fname
        if src.exists():
            shutil.copy(src, ext_dir / fname)

    # Create extension configs
    create_pyproject_toml(ext_dir, 'flash-linear-attention', version, ext_deps, extras)

    # Create build script
    build_script = output_dir / 'build.sh'
    with open(build_script, 'w') as f:
        f.write("""#!/bin/bash
# Build both packages

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


    return output_dir, version


def build_packages(dist_dir):
    """Build wheels and source distributions for both packages."""

    # Build fla-core (both wheel and sdist)
    try:
        subprocess.run(
            [sys.executable, "-m", "build", str(dist_dir / "fla-core")],
            check=True,
            timeout=1800,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except subprocess.CalledProcessError:
        return False
    except subprocess.TimeoutExpired:
        return False

    # Build flash-linear-attention (both wheel and sdist)
    try:
        subprocess.run(
            [sys.executable, "-m", "build", str(dist_dir / "flash-linear-attention")],
            check=True,
            timeout=1800,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except subprocess.CalledProcessError:
        return False
    except subprocess.TimeoutExpired:
        return False

    return True


def copy_packages_to_output(dist_dir):
    """Copy wheels and source distributions to output directory."""
    # Get script directory (relative to this file)
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent

    # Create output directory (relative to root)
    output_dir = root_dir / 'dist-packages'
    output_dir.mkdir(exist_ok=True)

    # Find wheels and source distributions
    core_wheels = list((dist_dir / 'fla-core' / 'dist').glob('*.whl'))
    core_sdist = list((dist_dir / 'fla-core' / 'dist').glob('*.tar.gz'))
    ext_wheels = list((dist_dir / 'flash-linear-attention' / 'dist').glob('*.whl'))
    ext_sdist = list((dist_dir / 'flash-linear-attention' / 'dist').glob('*.tar.gz'))

    if not core_wheels:
        return False
    if not ext_wheels:
        return False

    # Copy all packages to output directory
    all_packages = core_wheels + core_sdist + ext_wheels + ext_sdist
    for package in all_packages:
        target = output_dir / package.name
        shutil.copy2(package, target)
        if package.suffix == ".whl" or package.suffixes[-2:] == [".tar", ".gz"]:
            pass
        else:
            pass


    return True


def main():
    """Build split packages and copy to target directory."""


    # Build the split packages
    dist_dir, _ = build_split_packages()


    # Build packages (wheels and source distributions)
    if not build_packages(dist_dir):
        return 1

    # Copy packages to output directory
    if not copy_packages_to_output(dist_dir):
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

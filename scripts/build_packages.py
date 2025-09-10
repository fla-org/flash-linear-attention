#!/usr/bin/env python3
"""Build split packages with proper dependency management and copy them to target directory."""

import re
import shutil
import subprocess
from pathlib import Path


def extract_dependencies():
    """Extract dependencies from current setup.py."""
    # Get script directory and find setup.py in parent directory
    script_dir = Path(__file__).parent
    setup_py = script_dir.parent / 'setup.py'

    with open(setup_py) as f:
        content = f.read()

    # Extract install_requires
    match = re.search(r'install_requires=\[(.*?)\]', content, re.DOTALL)
    if not match:
        return [], {}

    deps_str = match.group(1)

    # Parse dependencies
    all_deps = []
    for line in deps_str.split('\n'):
        line = line.strip().strip("',")
        if line and not line.startswith('#'):
            all_deps.append(line)

    # Extract extras_require
    extras_match = re.search(r'extras_require=\{(.*?)\}', content, re.DOTALL)
    extras = {}
    if extras_match:
        extras_str = extras_match.group(1)
        for line in extras_str.split('\n'):
            if ':' in line and '[' in line:
                key = line.split(':')[0].strip().strip("'")
                values = re.findall(r"'(.*?)'", line)
                extras[key] = values

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
    desc_text = ('Core operations for flash-linear-attention' if name == 'fla-core'
                 else 'Fast linear attention models and layers')

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

    with open(package_dir / 'pyproject.toml', 'w') as f:
        f.write(content)


def build_split_packages():
    """Build split packages with proper dependency management."""
    # Get script directory and find files relative to it
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent

    # Get current version
    init_file = root_dir / 'fla' / '__init__.py'
    with open(init_file) as f:
        content = f.read()
    version_match = re.search(r"^__version__\s*=\s*'(.*)'$", content, re.MULTILINE)
    version = version_match.group(1) if version_match else '0.3.1'

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
        f.write(f"""__path__ = __import__('pkgutil').extend_path(__path__, __name__)
__version__ = '{version}'

from .ops import *
from .modules import *
from .utils import *
""")

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

    # Create flash-linear-attention __init__.py
    with open(fla_ext / '__init__.py', 'w') as f:
        f.write(f"""__path__ = __import__('pkgutil').extend_path(__path__, __name__)
__version__ = '{version}'

try:
    import fla
    if getattr(fla, '__version__', None) != __version__:
        raise ImportError(f"fla-core version {{fla.__version__}} != {{__version__}}")
except (ImportError, AttributeError) as e:
    raise ImportError(f"fla-core {{__version__}} is required") from e

from .layers import *
from .models import *
""")

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

    print(f"✅ Split packages created in {output_dir}")
    print(f"✅ fla-core dependencies: {len(core_deps)} packages")
    print(f"✅ flash-linear-attention dependencies: {len(ext_deps)} packages")
    print(f"✅ Version: {version}")

    return output_dir, version


def build_packages(dist_dir):
    """Build wheels and source distributions for both packages."""
    print("Building packages...")

    # Build fla-core (both wheel and sdist)
    print("Building fla-core packages...")
    result = subprocess.run([
        'python', '-m', 'build', str(dist_dir / 'fla-core')
    ])

    if result.returncode != 0:
        print("Failed to build fla-core packages")
        return False

    # Build flash-linear-attention (both wheel and sdist)
    print("Building flash-linear-attention packages...")
    result = subprocess.run([
        'python', '-m', 'build', str(dist_dir / 'flash-linear-attention')
    ])

    if result.returncode != 0:
        print("Failed to build flash-linear-attention packages")
        return False

    print("✅ Packages built successfully")
    return True


def copy_packages_to_output(dist_dir, version):
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
        print("No fla-core wheel found")
        return False
    if not ext_wheels:
        print("No flash-linear-attention wheel found")
        return False

    # Copy all packages to output directory
    all_packages = core_wheels + core_sdist + ext_wheels + ext_sdist
    for package in all_packages:
        target = output_dir / package.name
        shutil.copy2(package, target)
        package_type = "wheel" if package.suffix == '.whl' else "source"
        print(f"📦 Copied {package_type} package {package.name} to {output_dir}")

    print(f"\n✅ All packages copied to: {output_dir}")
    print("You can install wheels with:")
    print("  pip install dist-packages/*.whl")
    print(f"Source distributions are also available in: {output_dir}")

    return True


def main():
    """Build split packages and copy to target directory."""

    print("Building split packages...")

    # Build the split packages
    dist_dir, version = build_split_packages()

    print("\nTo build packages manually:")
    print(f"cd {dist_dir}")
    print("./build.sh")

    # Build packages (wheels and source distributions)
    if not build_packages(dist_dir):
        return 1

    # Copy packages to output directory
    if not copy_packages_to_output(dist_dir, version):
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

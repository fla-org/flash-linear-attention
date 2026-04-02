#!/usr/bin/env python
"""
Extract best configs from Triton's autotune cache.

This script searches Triton's cache directory (~/.triton/cache/fla_triton_cache/) for .autotune.json files,
extracts the best configuration for each kernel, and saves them to a human-readable format.

Usage:
    # Generate cache for KDA and extract configs (default head_dim=128)
    python scripts/extract_triton_autotune_cache.py -g

    # Generate cache for GDN
    python scripts/extract_triton_autotune_cache.py -g --op gdn

    # Generate cache for both KDA and GDN
    python scripts/extract_triton_autotune_cache.py -g --op both

    # Specify a single head_dim (affects autotune results)
    python scripts/extract_triton_autotune_cache.py -g -d 64

    # Generate cache for multiple head_dims in one run
    python scripts/extract_triton_autotune_cache.py -g -d 64 128 256

    # Generate cache for both ops across multiple head_dims
    python scripts/extract_triton_autotune_cache.py -g --op both -d 64 128 256

    # Extract from a custom Triton cache directory
    python scripts/extract_triton_autotune_cache.py --triton-cache-dir ~/.triton/cache

    # List available cache files without extracting
    python scripts/extract_triton_autotune_cache.py -l

The output files are saved to get_fla_config_dir()/{kernel_name}.json
Each file contains one or more autotune entries keyed by Triton's runtime key.
"""

import argparse
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault('FLA_CACHE_MODE', 'disabled')


def list_autotune_cache_files(triton_cache_dir: Path) -> None:
    if not triton_cache_dir.exists():
        print(f"Triton cache directory not found: {triton_cache_dir}")
        return

    autotune_files = list(triton_cache_dir.rglob("*.autotune.json"))
    print(f"Found {len(autotune_files)} .autotune.json files in {triton_cache_dir}:\n")

    for i, file in enumerate(autotune_files, 1):
        print(f"{i}. {file}")


def resolve_output_dir(output_dir: str | None, *, versioned: bool) -> Path:
    if output_dir is not None:
        return Path(output_dir)

    import triton

    from fla.ops.utils.cache import get_fla_config_dir

    resolved_output_dir = get_fla_config_dir()
    if versioned and "FLA_CONFIG_DIR" not in os.environ:
        return resolved_output_dir / triton.__version__
    return resolved_output_dir


def main():
    import triton

    from scripts.utils.autotune_export import extract_configs
    from scripts.utils.autotune_generate import generate_fla_cache, get_triton_cache_dir

    parser = argparse.ArgumentParser(description='Extract Triton autotune configs')
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory (default: $FLA_CONFIG_DIR or fla/configs/{GPU})'
    )
    parser.add_argument(
        '--triton-cache-dir',
        type=str,
        help='Triton cache directory (default: ~/.triton/cache)'
    )
    parser.add_argument(
        '--list-only', '-l',
        action='store_true',
        help='Only list the cache files without extracting'
    )
    parser.add_argument(
        '--generate-cache', '-g',
        action='store_true',
        help='Generate new cache with custom temporary directory'
    )
    parser.add_argument(
        '--op',
        choices=('kda', 'gdn', 'both'),
        default='kda',
        help='FLA op used to generate the Triton cache (default: kda)'
    )
    parser.add_argument(
        '--head-dim', '-d',
        type=int,
        nargs='+',
        default=[128],
        metavar='D',
        help='Head dimension(s) for cache generation (default: 128). '
             'Pass multiple values to cover several configs, e.g. -d 64 128 256.'
    )
    parser.add_argument(
        '--versioned', '-v',
        action='store_true',
        help=f'Include Triton version ({triton.__version__}) as a subdirectory in the output path'
    )
    args = parser.parse_args()

    # FLA_CONFIG_DIR already points at the final output directory when overridden.
    output_dir = resolve_output_dir(args.output_dir, versioned=args.versioned)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine directories
    if args.generate_cache:
        # Run FLA kernels to populate the fla_triton_cache subdirectory
        triton_cache_dir = Path(generate_fla_cache(args.op, args.head_dim, args.triton_cache_dir))
    else:
        triton_cache_dir = get_triton_cache_dir(args.triton_cache_dir)

    if args.list_only:
        list_autotune_cache_files(triton_cache_dir)
        return

    # Extract configs
    extract_configs(triton_cache_dir, output_dir)


if __name__ == "__main__":
    main()

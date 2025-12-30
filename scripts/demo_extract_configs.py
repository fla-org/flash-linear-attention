#!/usr/bin/env python
"""
Extract best configs from Triton's autotune cache.

This script searches Triton's cache directory (~/.triton/cache) for .autotune.json files,
extracts the best configuration for each kernel, and saves them to a human-readable format.

Usage:
    python extract_triton_autotune_cache.py [--output-dir DIR]

The output files are saved to fla/configs/{GPU}/{kernel_name}.json
Each file contains only the best_config for direct cache lookup.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

os.environ['FLA_DISABLE_CACHE'] = '1'


def get_gpu_info():
    """Get GPU model information.

    This function detects the GPU model and returns a sanitized string identifier.
    It prioritizes FLA_GPU_NAME environment variable if set, then detects from
    available hardware (CUDA, ROCm, Intel GPU, or CPU).
    """
    import torch

    # Check if GPU name is overridden via environment variable
    if "FLA_GPU_NAME" in os.environ:
        gpu_name = os.environ["FLA_GPU_NAME"]
        return gpu_name.replace(" ", "_").replace("(", "_").replace(")", "_").replace("-", "_")

    # Try to get device name based on availability
    if torch.cuda.is_available():
        # Works for both NVIDIA and AMD GPUs (ROCm)
        gpu_name = torch.cuda.get_device_name(0)
        gpu_name = gpu_name.replace(" ", "_").replace("(", "_").replace(")", "_").replace("-", "_")
        return gpu_name

    # Default to CPU if no GPU available
    return "cpu"


def get_triton_cache_dir() -> Path:
    """Get Triton's cache directory."""
    cache_dir = os.environ.get("TRITON_CACHE_DIR", "~/.triton")
    return Path(cache_dir).expanduser() / "cache"


def get_fla_config_dir() -> Path:
    """Get FLA's configs directory.

    The directory can be overridden by setting the FLA_CONFIG_DIR environment variable.
    If set, configs will be saved to $FLA_CONFIG_DIR/{GPU}/ instead of the default
    fla/configs/{GPU}/ in the project.
    """
    # Check if custom config dir is set via environment variable
    if "FLA_CONFIG_DIR" in os.environ:
        base_dir = Path(os.environ["FLA_CONFIG_DIR"])
    else:
        # Default: project_dir/fla/configs/
        project_dir = Path(__file__).parent.parent
        base_dir = project_dir / "fla" / "configs"

    gpu_name = get_gpu_info()
    config_dir = base_dir / gpu_name
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def process_autotune_file(autotune_file: Path) -> dict[str, Any]:
    """
    Process a single Triton autotune.json file and extract best config.

    Returns:
        Dictionary with kernel info and best config, or None if invalid
    """
    try:
        with open(autotune_file) as f:
            data = json.load(f)

        if not isinstance(data, dict) or "configs_timings" not in data:
            return None

        # Find the best config (minimum timing)
        configs_timings = data["configs_timings"]
        if not configs_timings:
            return None

        # configs_timings is a list of [config_dict, timing]
        best_entry = min(configs_timings, key=lambda x: x[1] if isinstance(x[1], (int, float)) else x[1][0])
        best_config_dict = best_entry[0]
        best_timing = best_entry[1]

        # Extract kernel info from the file path or content
        # Example path: ~/.triton/cache/a1b2c3d4/fused_recurrent_fwd_jit_functionn_12345.autotune.json
        parts = autotune_file.stem.split('.')
        if len(parts) >= 2:
            kernel_name = parts[0]
        else:
            kernel_name = "unknown_kernel"

        # Build output data structure
        result = {
            "kernel_name": kernel_name,
            "source_file": str(autotune_file),
            "cache_key": data.get("key", "unknown"),
            "timestamp": data.get("timestamp", 0),
            "best_config": best_config_dict,
            "best_timing": best_timing,
            "total_configs_tested": len(configs_timings),
        }

        return result

    except Exception as e:
        print(f"Error processing {autotune_file}: {e}")
        return None


def sync_hopper_configs(updated_dir: Path):
    """
    Sync Hopper architecture GPU configs (NVIDIA_H100, NVIDIA_H200, NVIDIA_H20).

    When one of these GPU configs is updated, copy its contents to the other directories.
    Creates target directories if they don't exist.

    Args:
        updated_dir: The directory that was just updated (source directory)
    """
    # Define the GPU groups that share the same configs (with NVIDIA_ prefix)
    hopper_gpus = ["NVIDIA_H100", "NVIDIA_H800", "NVIDIA_H20"]

    # Get the parent directory (e.g., fla/configs)
    parent_dir = updated_dir.parent

    # Check if the updated_dir name matches any of the hopper GPUs
    updated_gpu = updated_dir.name
    if updated_gpu not in hopper_gpus:
        return  # Not a hopper GPU, skip sync

    print(f"\nSyncing Hopper configs from {updated_gpu} to other GPUs: {hopper_gpus}")
    print("-" * 60)

    # Copy to other hopper GPU directories
    sync_count = 0
    for target_gpu in hopper_gpus:
        if target_gpu == updated_gpu:
            continue  # Skip self

        target_dir = parent_dir / target_gpu

        try:
            # Create target directory if it doesn't exist
            target_dir.mkdir(parents=True, exist_ok=True)

            # Copy all files from updated_dir to target_dir
            for config_file in updated_dir.glob("*.json"):
                if config_file.is_file():
                    shutil.copy2(config_file, target_dir / config_file.name)

            sync_count += 1
            print(f"  ✓ Synced to {target_gpu} ({target_dir})")

        except Exception as e:
            print(f"  ✗ Failed to sync to {target_gpu}: {e}")

    print(f"\nSuccessfully synced configs to {sync_count} GPU directories")
    print("=" * 60)


def extract_configs(triton_cache_dir: Path, output_dir: Path):
    """
    Extract all autotune configs from Triton cache.

    Args:
        triton_cache_dir: Triton's cache directory
        output_dir: Output directory for extracted configs
    """
    if not triton_cache_dir.exists():
        print(f"Triton cache directory not found: {triton_cache_dir}")
        return

    # Find all .autotune.json files
    autotune_files = list(triton_cache_dir.rglob("*.autotune.json"))

    if not autotune_files:
        print(f"No .autotune.json files found in {triton_cache_dir}")
        return

    print(f"Found {len(autotune_files)} autotune cache files")
    print(f"GPU: {get_gpu_info()}")
    print(f"Output directory: {output_dir}")
    print("-" * 60)

    # Process each file
    exported_count = 0
    for autotune_file in autotune_files:
        result = process_autotune_file(autotune_file)
        if result is None:
            continue

        # Save to output directory
        kernel_name = result["kernel_name"]
        output_file = output_dir / f"{kernel_name}.json"

        try:
            with open(output_file, 'w') as f:
                # Only save the best_config for direct lookup
                output_data = result["best_config"]
                json.dump(output_data, f, indent=2)

            exported_count += 1

            print(f"\n[{exported_count}] {kernel_name}")
            print(f"    Source: {autotune_file}")
            print(f"    Output: {output_file}")
            print(f"    Best config: {result['best_config']}")
            print(f"    Timing: {result['best_timing']}")

        except Exception as e:
            print(f"Error saving {output_file}: {e}")

    print("\n" + "=" * 60)
    print(f"Successfully exported {exported_count} configs to {output_dir}")
    print("=" * 60)
    sync_hopper_configs(output_dir)

    # Remove Custom Triton Cache
    try:
        shutil.rmtree(triton_cache_dir)
        print(f"Removed Triton cache directory: {triton_cache_dir}")
    except Exception as e:
        print(f"Warning: Failed to remove {triton_cache_dir}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Extract Triton autotune configs')
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Output directory (default: fla/configs/autotune/{GPU})'
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

    args = parser.parse_args()

    # Determine directories
    if args.generate_cache:
        # Generate cache with temporary directory
        triton_cache_dir = Path(generate_triton_cache())
    else:
        triton_cache_dir = Path(args.triton_cache_dir) if args.triton_cache_dir else get_triton_cache_dir()

    output_dir = Path(args.output_dir) if args.output_dir else get_fla_config_dir()

    if args.list_only:
        # Just list the files
        if not triton_cache_dir.exists():
            print(f"Triton cache directory not found: {triton_cache_dir}")
            return

        autotune_files = list(triton_cache_dir.rglob("*.autotune.json"))
        print(f"Found {len(autotune_files)} .autotune.json files in {triton_cache_dir}:\n")

        for i, file in enumerate(autotune_files, 1):
            print(f"{i}. {file}")
        return

    # Extract configs
    extract_configs(triton_cache_dir, output_dir)


def generate_triton_cache():
    """Generate Triton cache with custom directory."""
    import torch

    from fla.ops.kda import chunk_kda
    from fla.utils import device

    # Create a custom directory in the project for Triton cache
    project_dir = Path(__file__).parent.parent
    custom_cache_dir = project_dir / "tmp_triton_cache"
    cache_path = custom_cache_dir / "triton" / "cache"

    # Clear and create the directory
    if custom_cache_dir.exists():
        shutil.rmtree(custom_cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    print(f"Using custom Triton cache directory: {cache_path}")
    os.environ["TRITON_CACHE_DIR"] = str(cache_path)

    # Generate cache by running the kernels
    torch.manual_seed(42)
    dtype = torch.bfloat16
    # Just for DEMO.
    B, T, H, D = 1, 8192, 32, 128

    q = torch.rand(B, T, H, D, dtype=dtype)
    k = torch.rand(B, T, H, D, dtype=dtype)
    v = torch.rand(B, T, H, D, dtype=dtype)
    g = torch.randn(B, T, H, D, dtype=dtype)
    A_log = torch.randn(H, dtype=torch.float)
    dt_bias = torch.randn(H * D, dtype=torch.float)
    beta = torch.randn(B, T, H, dtype=dtype).sigmoid()
    h0 = torch.randn(B, H, D, D, dtype=torch.float32)
    A_log, dt_bias = map(lambda x: x.to(device).requires_grad_(True), (A_log, dt_bias))
    q, k, v, g, beta, h0 = map(lambda x: x.to(device).requires_grad_(True), (q, k, v, g, beta, h0))

    do = torch.randn_like(v)
    dht = torch.randn_like(h0)

    tri, tri_ht = chunk_kda(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=g.clone().float(),
        beta=beta.clone(),
        A_log=A_log.clone(),
        dt_bias=dt_bias.clone(),
        scale=None,
        initial_state=h0.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=False,
        safe_gate=True,
        lower_bound=-5,
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri0, tri_ht0 = chunk_kda(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        A_log=A_log.clone(),
        dt_bias=dt_bias.clone(),
        scale=None,
        initial_state=h0.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=-5,
    )
    ((tri0 * do).sum() + (tri_ht0 * dht).sum()).backward()

    W = 4
    x = torch.randn(B, T, H*D).to(device, dtype).requires_grad_(True)
    weight = torch.randn(H*D, W).to(device, dtype).requires_grad_(True)
    bias = None

    dy = torch.randn(B, T, H*D).to(device, dtype)

    from fla.modules.convolution import causal_conv1d
    tri, _ = causal_conv1d(x, weight, bias, residual=None, activation="silu")
    tri.backward(dy)

    return str(cache_path)


if __name__ == "__main__":
    # Check if we should extract to fla/configs
    if "--extract-to-fla-configs" in sys.argv:
        # Generate cache with temporary directory
        triton_cache_dir = Path(generate_triton_cache())

        # Compute output directory in fla/configs (relative to project root)
        project_dir = Path(__file__).parent.parent
        gpu_name = get_gpu_info()
        output_dir = project_dir / "fla" / "configs" / gpu_name
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nExtracting configs to: {output_dir}")
        extract_configs(triton_cache_dir, output_dir)
    else:
        main()

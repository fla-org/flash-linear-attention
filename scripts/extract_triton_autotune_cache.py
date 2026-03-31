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

    # Specify head_dim (affects autotune results)
    python scripts/extract_triton_autotune_cache.py -g -d 64

    # List available cache files without extracting
    python scripts/extract_triton_autotune_cache.py -l

The output files are saved to fla/configs/{GPU}/{kernel_name}.json
Each file contains the best_config fields plus kernel_name for inspection.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import triton

from fla.ops.utils.cache import get_fla_config_dir, get_gpu_info

os.environ['FLA_DISABLE_CACHE'] = '1'


def get_triton_cache_dir(path: str | None = None) -> Path:
    """Get Triton's cache directory via Triton's internal API."""
    if path is not None:
        return Path(path)
    from triton.runtime.cache import knobs
    return Path(knobs.cache.dir)


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

        def timing_key(entry):
            t = entry[1]
            return (t,) if isinstance(t, (int, float)) else tuple(t)

        # configs_timings is a list of [config_dict, timing]
        best_entry = min(configs_timings, key=timing_key)
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


def normalize_comparable_value(value):
    if isinstance(value, dict):
        return tuple((key, normalize_comparable_value(value[key])) for key in sorted(value))
    if isinstance(value, list):
        return tuple(normalize_comparable_value(item) for item in value)
    return value


def config_preference_key(config: dict[str, object] | None) -> tuple[object, ...]:
    if not isinstance(config, dict):
        return (float('inf'), (('__missing__', True),), float('inf'), float('inf'))
    return (
        config.get("num_stages", float('inf')),
        config.get("num_warps", float('inf')),
        normalize_comparable_value(config.get("kwargs", {})),
        config.get("num_ctas", float('inf')),
    )


def choose_preferred_config(
    existing_data: dict[str, object] | None,
    output_data: dict[str, object],
) -> dict[str, object]:
    existing_key = config_preference_key(existing_data)
    output_key = config_preference_key(output_data)
    return output_data if output_key < existing_key else existing_data


def backup_existing_file(output_file: Path) -> Path:
    backup_dir = output_file.parent / "bak"
    backup_dir.mkdir(parents=True, exist_ok=True)
    raw_bytes = output_file.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()[:16]
    backup_file = backup_dir / f"{output_file.stem}.{digest}{output_file.suffix}"
    if not backup_file.exists():
        backup_file.write_bytes(raw_bytes)
    return backup_file


def backup_config_data(output_file: Path, config_data: dict[str, object]) -> Path:
    backup_dir = output_file.parent / "bak"
    backup_dir.mkdir(parents=True, exist_ok=True)
    raw_bytes = json.dumps(config_data, indent=2, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(raw_bytes).hexdigest()[:16]
    backup_file = backup_dir / f"{output_file.stem}.{digest}{output_file.suffix}"
    if not backup_file.exists():
        backup_file.write_bytes(raw_bytes)
    return backup_file


def save_extracted_config(output_file: Path, output_data: dict[str, object]) -> tuple[str, Path | None]:
    backup_file = None

    if output_file.exists():
        try:
            with open(output_file) as f:
                existing_data = json.load(f)
        except Exception:
            existing_data = None

        if existing_data != output_data:
            chosen_data = choose_preferred_config(existing_data, output_data)
            if chosen_data != existing_data:
                backup_file = backup_existing_file(output_file)
                status = "updated"
            else:
                backup_file = backup_config_data(output_file, output_data)
                status = "unchanged"
            output_data = chosen_data
        else:
            status = "unchanged"
    else:
        status = "created"

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    return status, backup_file


def extract_configs(triton_cache_dir: Path, output_dir: Path):
    """
    Extract all autotune configs from Triton cache.

    Args:
        triton_cache_dir: Triton's cache directory
        output_dir: Output directory for extracted configs
    """
    if not triton_cache_dir.exists():
        print(f"Triton cache directory not found: {triton_cache_dir}. Exiting as there's nothing to extract.")
        sys.exit(1)

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
    created_count = 0
    overwritten_count = 0
    backup_files = []
    unchanged_count = 0
    for autotune_file in autotune_files:
        result = process_autotune_file(autotune_file)
        if result is None:
            continue

        # Save to output directory
        kernel_name = result["kernel_name"]
        output_file = output_dir / f"{kernel_name}.json"

        try:
            # Keep config fields at top level for cache lookup, plus kernel name for inspection.
            output_data = {
                **result["best_config"],
                "kernel_name": kernel_name,
            }
            status, backup_file = save_extracted_config(output_file, output_data)

            exported_count += 1
            if status == "created":
                created_count += 1
            else:
                overwritten_count += 1
                if status == "unchanged":
                    unchanged_count += 1
                elif status == "updated":
                    backup_files.append(backup_file)

            print(f"\n[{exported_count}] {kernel_name}")
            print(f"    Source: {autotune_file}")
            print(f"    Output: {output_file}")
            print(f"    Status: {status}")
            if status == "updated":
                print(f"    Backup: {backup_file}")
            print(f"    Best config: {result['best_config']}")
            print(f"    Timing: {result['best_timing']}")

        except Exception as e:
            print(f"Error saving {output_file}: {e}")

    print("\n" + "=" * 60)
    print(f"Successfully exported {exported_count} configs to {output_dir}")
    print(f"New files created: {created_count}")
    print(f"Existing files overwritten: {overwritten_count}")
    print(f"Existing files with identical content: {unchanged_count}")
    print(f"Backups created for changed files: {len(backup_files)}")
    for backup_file in backup_files:
        print(f"  {backup_file}")
    print("=" * 60)
    sync_hopper_configs(output_dir)


def main():
    parser = argparse.ArgumentParser(description='Extract Triton autotune configs')
    parser.add_argument(
        '--output-dir',
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
    parser.add_argument(
        '--op',
        choices=('kda', 'gdn', 'both'),
        default='kda',
        help='FLA op used to generate the Triton cache (default: kda)'
    )
    parser.add_argument(
        '--head-dim', '-d',
        type=int,
        default=128,
        help='Head dimension used when generating the Triton cache (default: 128). '
             'Different head_dim values produce different autotune configs.'
    )
    parser.add_argument(
        '--versioned', '-v',
        action='store_true',
        help=f'Include Triton version ({triton.__version__}) as a subdirectory in the output path'
    )

    args = parser.parse_args()

    # Determine directories
    if args.generate_cache:
        # Run FLA kernels to populate the fla_triton_cache subdirectory
        triton_cache_dir = Path(generate_fla_cache(args.op, args.head_dim, args.triton_cache_dir))
    else:
        triton_cache_dir = get_triton_cache_dir(args.triton_cache_dir)

    output_dir = Path(args.output_dir) if args.output_dir else get_fla_config_dir()
    # Only append the Triton version subdirectory when using the default output path.
    # If the user explicitly provided --output-dir, respect it as-is without modification.
    if not args.output_dir and args.versioned:
        output_dir = output_dir / triton.__version__

    output_dir.mkdir(parents=True, exist_ok=True)

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


def prepare_kernel_cache_tensors(head_dim: int, *, torch, device, op_name: str):
    # Generate cache by running the kernels
    torch.manual_seed(42)
    dtype = torch.bfloat16
    B, T, H, D = 1, 8192, 32, head_dim
    print(f"Generating {op_name} cache with head_dim={D}")

    q = torch.rand(B, T, H, D, dtype=dtype)
    k = torch.rand(B, T, H, D, dtype=dtype)
    v = torch.rand(B, T, H, D, dtype=dtype)
    g = torch.randn(B, T, H, D, dtype=dtype)
    A_log = torch.randn(H, dtype=torch.float)
    dt_bias = torch.randn(H * D, dtype=torch.float)
    beta = torch.randn(B, T, H, dtype=dtype).sigmoid()
    h0 = torch.randn(B, H, D, D, dtype=torch.float32)
    A_log, dt_bias = map(lambda x: x.to(device).requires_grad_(True), (A_log, dt_bias))
    q, k, v, beta, h0 = map(lambda x: x.to(device).requires_grad_(True), (q, k, v, beta, h0))
    g = g.to(device).requires_grad_(True)

    do = torch.randn_like(v)
    dht = torch.randn_like(h0)
    return q, k, v, g, beta, h0, do, dht, A_log, dt_bias


def generate_kda_cache(head_dim: int, *, torch, device):
    from fla.ops.kda import chunk_kda, fused_recurrent_kda
    from fla.ops.kda.gate import fused_kda_gate
    q, k, v, g, beta, h0, do, dht, A_log, dt_bias = prepare_kernel_cache_tensors(
        head_dim,
        torch=torch,
        device=device,
        op_name='kda',
    )

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

    g = fused_kda_gate(g=g.clone(), A_log=A_log.clone(), dt_bias=dt_bias.clone())
    fused_recurrent_kda(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=g,
        beta=beta.clone(),
        initial_state=h0.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )


def generate_gdn_cache(head_dim: int, *, torch, device):
    import torch.nn.functional as F

    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    q, k, v, g, beta, h0, do, dht, _, _ = prepare_kernel_cache_tensors(
        head_dim,
        torch=torch,
        device=device,
        op_name='gdn',
    )
    g = g[..., 0].float().detach().requires_grad_(True)

    for use_qk_l2norm_in_kernel in (False, True):
        tri, tri_ht = chunk_gated_delta_rule(
            q=(F.normalize(q.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else q.clone()),
            k=(F.normalize(k.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else k.clone()),
            v=v.clone(),
            g=g.clone(),
            beta=beta.clone(),
            scale=None,
            initial_state=h0.clone(),
            output_final_state=True,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=not use_qk_l2norm_in_kernel)
        q.grad = k.grad = v.grad = beta.grad = g.grad = h0.grad = None


def generate_conv_cache(head_dim: int, *, torch, device):
    torch.manual_seed(42)
    dtype = torch.bfloat16
    B, T, H, D = 1, 8192, 32, head_dim

    W = 4
    x = torch.randn(B, T, H * D).to(device, dtype).requires_grad_(True)
    weight = torch.randn(H * D, W).to(device, dtype).requires_grad_(True)
    bias = None

    dy = torch.randn(B, T, H * D).to(device, dtype)

    from fla.modules.convolution import causal_conv1d
    tri, _ = causal_conv1d(x, weight, bias, residual=None, activation="silu")
    tri.backward(dy)


def generate_fla_cache(op: str = 'kda', head_dim: int = 128, triton_cache_dir: str | None = None) -> str:
    """Generate Triton cache with custom directory."""
    import torch

    from fla.utils import device

    # Store FLA autotune results under fla_triton_cache/ to keep them separate from other Triton kernels
    fla_triton_cache = get_triton_cache_dir(triton_cache_dir) / "fla_triton_cache"

    # Clear and create the directory
    if fla_triton_cache.exists():
        shutil.rmtree(fla_triton_cache)
    fla_triton_cache.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(fla_triton_cache)

    print(f"Using FLA Triton cache directory: {fla_triton_cache}")

    if op in ('kda', 'both'):
        generate_kda_cache(head_dim, torch=torch, device=device)
    if op in ('gdn', 'both'):
        generate_gdn_cache(head_dim, torch=torch, device=device)
    elif op != 'kda':
        raise ValueError(f"Unsupported op: {op}")

    generate_conv_cache(head_dim, torch=torch, device=device)

    return str(fla_triton_cache)


if __name__ == "__main__":
    main()

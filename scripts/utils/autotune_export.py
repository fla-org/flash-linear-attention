import copy
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import triton

from fla.ops.utils.cache import AutotuneKey, KernelConfigFile, get_gpu_info


def timing_value_key(value: object) -> tuple[float, ...]:
    if isinstance(value, (int, float)):
        return (float(value),)
    if isinstance(value, (list, tuple)):
        return tuple(float(v) for v in value)
    return (float('inf'),)


@dataclasses.dataclass
class AutotuneResult:
    """Best config extracted from a single Triton .autotune.json file."""
    kernel_name: str
    source_file: str
    cache_key: Any
    timestamp: float
    best_config: dict[str, Any]
    best_timing: Any
    total_configs_tested: int

    @classmethod
    def from_file(cls, autotune_file: Path) -> "AutotuneResult | None":
        try:
            with open(autotune_file) as f:
                data = json.load(f)
            if not isinstance(data, dict) or "configs_timings" not in data:
                return None
            # Example path: ~/.triton/cache/a1b2c3d4/fused_recurrent_fwd_jit_functionn_12345.autotune.json
            parts = autotune_file.stem.split('.')
            kernel_name = parts[0] if len(parts) >= 2 else "unknown_kernel"
            configs_timings = data["configs_timings"]
            if not configs_timings:
                return None
            best_entry = min(configs_timings, key=lambda e: timing_value_key(e[1]))
            return cls(
                kernel_name=kernel_name,
                source_file=str(autotune_file),
                cache_key=data.get("key", "unknown"),
                timestamp=data.get("timestamp", 0),
                best_config=best_entry[0],
                best_timing=best_entry[1],
                total_configs_tested=len(configs_timings),
            )
        except Exception as e:
            print(f"Error processing {autotune_file}: {e}")
            return None

    def to_entry(self) -> dict[str, Any]:
        return {
            "autotune_key": AutotuneKey.normalize_autotune_key(self.cache_key),
            "config": self.best_config,
            "best_timing": self.best_timing,
        }


def extract_default_config(config: object) -> dict[str, object] | None:
    if not isinstance(config, dict):
        return None
    kwargs = config.get("kwargs")
    if not isinstance(kwargs, dict):
        return None
    return {
        "kwargs": copy.deepcopy(kwargs),
        "num_warps": config.get("num_warps"),
        "num_ctas": config.get("num_ctas"),
        "num_stages": config.get("num_stages"),
    }


def serialize_preserved_entry(entry: dict[str, object]) -> str:
    return json.dumps(
        {"autotune_key": entry.get("autotune_key"), "config": entry.get("config")},
        sort_keys=True, separators=(",", ":"),
    )


def load_kernel_config_file_uncached(config_file: Path) -> KernelConfigFile | None:
    """Read a kernel config file directly from disk.

    Extract runs in the same process as optional cache generation. During generation,
    FLA may cache a missing config file as ``None`` via ``fla.ops.utils.cache.load_config_file``.
    If extract then reuses that cached result, the first run incorrectly behaves as if the
    existing file does not exist and skips backup creation. Reading from disk here avoids
    that stale negative cache.
    """
    try:
        with open(config_file) as f:
            data = json.load(f)
    except Exception:
        return None
    return KernelConfigFile.from_dict(config_file, data)


class KernelConfigFileWriter:
    """Mutable builder for a {kernel_name}.json config file.

    Supports merging new AutotuneResult entries into an existing file,
    maintaining key ordering in the output JSON and backing up overwritten entries.
    """

    def __init__(self, kernel_name: str, existing: KernelConfigFile | None = None):
        self._data = self.normalize(existing, kernel_name)
        self.rejected_entries: list[dict[str, object]] = []

    @staticmethod
    def normalize(existing: KernelConfigFile | None, kernel_name: str) -> dict[str, object]:
        normalized: dict[str, object] = {
            "kernel_name": (existing.kernel_name if existing else None) or kernel_name,
            "triton_version": (existing.triton_version if existing else None) or triton.__version__,
        }
        if existing is not None and existing.default_config is not None:
            normalized["default_config"] = copy.deepcopy(existing.default_config)
        entries: dict[str, dict] = {}
        if existing is not None and existing.autotune_entries is not None:
            entries = copy.deepcopy(existing.autotune_entries)
        normalized["autotune_entries"] = entries
        return normalized

    def update_default_config(self) -> None:
        # Rewrite default_config in-place, keeping it ordered before autotune_entries in the output JSON.
        entries: dict = self._data.pop("autotune_entries", {})
        self._data.pop("default_config", None)
        for key in sorted(entries):
            default_config = extract_default_config(entries[key].get("config"))
            if default_config is not None:
                self._data["default_config"] = default_config
                break
        self._data["autotune_entries"] = entries

    def merge(self, result: AutotuneResult) -> None:
        self._data["triton_version"] = triton.__version__
        new_entry = result.to_entry()
        new_key = AutotuneKey.key_hash(new_entry["autotune_key"])
        entries: dict[str, dict] = self._data["autotune_entries"]

        def resource_key(cfg: dict) -> tuple:
            return (cfg["num_stages"], cfg["num_warps"], cfg["num_ctas"])

        if new_key in entries:
            existing_entry = entries[new_key]
            if resource_key(new_entry["config"]) < resource_key(existing_entry["config"]):
                if new_entry["config"] != existing_entry["config"]:
                    self.rejected_entries.append(copy.deepcopy(existing_entry))
                entries[new_key] = new_entry
            elif new_entry["config"] != existing_entry["config"]:
                self.rejected_entries.append(copy.deepcopy(new_entry))
        else:
            entries[new_key] = new_entry

        self.update_default_config()

    def to_dict(self) -> dict[str, object]:
        data = dict(self._data)
        entries: dict = data.get("autotune_entries", {})
        data["autotune_entries"] = {k: entries[k] for k in sorted(entries)}
        return data

    def find_changed_entries(self, existing: KernelConfigFile | None) -> list[dict[str, object]]:
        """Return entries from existing whose key+config differs from self (for backup)."""
        if existing is None or existing.autotune_entries is None:
            return []
        output_entries: dict = self._data.get("autotune_entries", {})
        backup_entries = []
        for h, entry in existing.autotune_entries.items():
            output_entry = output_entries.get(h)
            if output_entry is None:
                continue
            if serialize_preserved_entry(entry) != serialize_preserved_entry(output_entry):
                backup_entries.append(copy.deepcopy(entry))
        backup_entries.sort(key=lambda e: AutotuneKey.key_hash(e.get("autotune_key")))
        return backup_entries

    @staticmethod
    def write_backup(
        backup_file: Path,
        kernel_name: str,
        backup_entries: list[dict[str, object]],
    ) -> None:
        existing_backup: dict[str, dict[str, object]] = {}
        if backup_file.exists():
            try:
                with open(backup_file) as f:
                    existing_data = json.load(f)
                if isinstance(existing_data, dict) and isinstance(existing_data.get("autotune_entries"), dict):
                    for h, entry in existing_data["autotune_entries"].items():
                        if isinstance(entry, dict):
                            existing_backup[h] = copy.deepcopy(entry)
            except Exception:
                pass

        for entry in backup_entries:
            existing_backup[AutotuneKey.key_hash(entry.get("autotune_key"))] = copy.deepcopy(entry)

        backup_data = {
            "kernel_name": kernel_name,
            "autotune_entries": {k: existing_backup[k] for k in sorted(existing_backup)},
        }
        with open(backup_file, 'w') as f:
            json.dump(backup_data, f, indent=2)

    def save(self, output_file: Path, kernel_name: str, existing: KernelConfigFile | None) -> tuple[str, Path | None]:
        backup_entries = list(self.rejected_entries)
        if existing is not None:
            new_entries: dict = self._data.get("autotune_entries", {})
            old_entries = existing.autotune_entries or {}
            backup_entries = self.find_changed_entries(existing) + backup_entries
            if len(new_entries) == len(old_entries) and not backup_entries:
                return "unchanged", None
            status = "updated"
        else:
            status = "created"

        backup_file: Path | None = None
        if backup_entries:
            backup_file = output_file.parent / f"{output_file.name}.bak"
            self.write_backup(backup_file, kernel_name, backup_entries)

        with open(output_file, 'w') as f:
            json.dump(self._data, f, indent=2)
        return status, backup_file


def extract_configs(triton_cache_dir: Path, output_dir: Path) -> None:
    """Extract all autotune configs from Triton cache."""
    if not triton_cache_dir.exists():
        print(f"Triton cache directory not found: {triton_cache_dir}. Exiting as there's nothing to extract.")
        sys.exit(1)

    autotune_files = list(triton_cache_dir.rglob("*.autotune.json"))
    if not autotune_files:
        print(f"No .autotune.json files found in {triton_cache_dir}")
        return

    print(f"Found {len(autotune_files)} autotune cache files")
    print(f"GPU: {get_gpu_info()}")
    print(f"Output directory: {output_dir}")
    print("-" * 60)

    # Group results by kernel name so each kernel's file is read once, merged all at once, and written once.
    # Processing file-by-file would re-read the same cached (stale) KernelConfigFile on every pass
    # for the same kernel, causing later writes to silently overwrite entries added by earlier ones.
    from collections import defaultdict
    results_by_kernel: dict[str, list[AutotuneResult]] = defaultdict(list)
    for autotune_file in autotune_files:
        result = AutotuneResult.from_file(autotune_file)
        if result is not None:
            results_by_kernel[result.kernel_name].append(result)

    exported_count = 0
    created_count = 0
    updated_count = 0
    unchanged_count = 0
    backup_files: set[Path] = set()

    for kernel_name, results in results_by_kernel.items():
        output_file = output_dir / f"{kernel_name}.json"
        existing = load_kernel_config_file_uncached(output_file) if output_file.exists() else None

        try:
            writer = KernelConfigFileWriter(kernel_name, existing)
            for result in results:
                writer.merge(result)
            status, backup_file = writer.save(output_file, kernel_name, existing)

            for result in results:
                exported_count += 1
                print(f"\n[{exported_count}] {kernel_name}")
                print(f"    Autotune key: {AutotuneKey.normalize_autotune_key(result.cache_key)}")
                print(f"    Output: {output_file}")
                print(f"    Best config: {result.best_config}")
                print(f"    Timing: {result.best_timing}")

            print(f"    Status: {status}")
            if backup_file is not None:
                print(f"    Backup: {backup_file}")
                backup_files.add(backup_file)

            if status == "created":
                created_count += 1
            elif status == "updated":
                updated_count += 1
            else:
                unchanged_count += 1

        except Exception as e:
            print(f"Error saving {output_file}: {e}")

    print("\n" + "=" * 60)
    print(f"Successfully exported {exported_count} configs to {output_dir}")
    print(f"New files created: {created_count}")
    print(f"Existing files updated: {updated_count}")
    print(f"Existing files unchanged: {unchanged_count}")
    print(f"Backups created: {len(backup_files)}")
    for backup_file in sorted(backup_files):
        print(f"  {backup_file}")
    print("=" * 60)

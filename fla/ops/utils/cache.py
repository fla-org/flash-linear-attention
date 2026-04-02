import dataclasses
import enum
import json
import logging
import os
import re
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

import torch
import triton
from packaging import version
from triton.runtime.autotuner import Autotuner

TRITON_ABOVE_3_5_1 = version.parse(triton.__version__) >= version.parse("3.5.1")
TRITON_ABOVE_3_4_0 = version.parse(triton.__version__) >= version.parse("3.4.0")


class FlaCacheMode(enum.Enum):
    """Controls how FLA loads kernel configs from its config cache (FLA_CACHE_MODE env var).

    DISABLED    — skip all cache lookups, always fall back to Triton autotune (default when FLA_CACHE_MODE is unset)
    STRICT      — exact key match only; falls back to Triton autotune if no match
    FUZZY       — exact key match → fuzzy key match; falls back to Triton autotune if no match
    FULL        — exact key match → fuzzy key match → default_config fallback
    DEFAULT     — use only the top-level default_config field, skip key-based lookup
    ALWAYS      — like DEFAULT, but re-reads config files on every kernel call;
                  useful for debugging: edit default_config in a JSON file and the next
                  kernel call picks it up without restarting the process
    """
    DISABLED = "disabled"
    STRICT = "strict"
    FUZZY = "fuzzy"
    FULL = "full"
    DEFAULT = "default"
    ALWAYS = "always"

    def uses_default_config(self) -> bool:
        """Return True for modes that may fall back to default_config (FULL, DEFAULT, ALWAYS)."""
        return self in (FlaCacheMode.FULL, FlaCacheMode.DEFAULT, FlaCacheMode.ALWAYS)

    @classmethod
    def from_env(cls) -> "FlaCacheMode":
        mode_str = os.environ.get("FLA_CACHE_MODE", cls.DISABLED.value)
        try:
            return cls(mode_str)
        except ValueError:
            valid = [m.value for m in cls]
            raise ValueError(
                f"Invalid FLA_CACHE_MODE={mode_str!r}. Valid values: {valid}"
            ) from None


FLA_CACHE_MODE: FlaCacheMode = FlaCacheMode.from_env()
logger = logging.getLogger(__name__)


def sanitize_gpu_name(gpu_name: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z]+", "_", gpu_name)
    sanitized = sanitized.strip("_")
    return sanitized or "unknown_gpu"


@lru_cache(maxsize=1)
def get_gpu_info():
    """Get GPU model information.

    This function detects the GPU model and returns a sanitized string identifier.
    It prioritizes FLA_GPU_NAME environment variable if set, then detects from
    available hardware (CUDA, ROCm, Intel GPU, or CPU).
    """
    # Check if GPU name is overridden via environment variable
    gpu_name = None
    # Check if GPU name is overridden via environment variable
    if "FLA_GPU_NAME" in os.environ:
        gpu_name = os.environ["FLA_GPU_NAME"]
    # Try to get device name based on availability
    elif torch.cuda.is_available():
        # Works for both NVIDIA and AMD GPUs (ROCm)
        gpu_name = torch.cuda.get_device_name(0)
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        gpu_name = torch.xpu.get_device_name(0)

    if gpu_name:
        return sanitize_gpu_name(gpu_name)

    # Default to CPU if no GPU available
    return "cpu"


def get_fla_config_dir() -> Path:
    """Get FLA's configs directory.

    The directory can be overridden by setting the FLA_CONFIG_DIR environment variable.
    If set, configs will be loaded directly from $FLA_CONFIG_DIR/. Otherwise FLA
    falls back to the default fla/configs/{GPU}/ directory in the project.
    """
    # Check if custom config dir is set via environment variable
    if "FLA_CONFIG_DIR" in os.environ:
        return Path(os.environ["FLA_CONFIG_DIR"])

    # Default: project_dir/fla/configs/{GPU}/
    project_dir = Path(__file__).parent.parent.parent
    return project_dir / "configs" / get_gpu_info()


@dataclasses.dataclass(frozen=True)
class AutotuneKey:
    """Autotune key with exact/fuzzy matching, serialization, and construction helpers."""
    autotune_key: tuple[Any, ...]

    @staticmethod
    def normalize_autotune_key(value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            return [AutotuneKey.normalize_autotune_key(v) for v in value]
        if isinstance(value, dict):
            return {k: AutotuneKey.normalize_autotune_key(v) for k, v in value.items()}
        return value

    @staticmethod
    def serialize(key: Any) -> str:
        return json.dumps(AutotuneKey.normalize_autotune_key(key), separators=(",", ":"), sort_keys=True)

    @staticmethod
    def key_hash(key: Any) -> str:
        import hashlib
        return hashlib.md5(AutotuneKey.serialize(key).encode()).hexdigest()

    @staticmethod
    def is_numeric(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    @staticmethod
    def keys_fuzzy_match(cached_key: Any, requested_key: Any) -> bool:
        # Fuzzy match: numeric leaves are compatible regardless of their actual numeric values
        # (e.g. a config tuned for seq_len=1024 can apply to seq_len=2048).
        # Structure (type, length, dict keys) must still match exactly.
        if AutotuneKey.is_numeric(cached_key) and AutotuneKey.is_numeric(requested_key):
            return True
        if isinstance(cached_key, (list, tuple)) and isinstance(requested_key, (list, tuple)):
            return len(cached_key) == len(requested_key) and all(
                AutotuneKey.keys_fuzzy_match(c, r) for c, r in zip(cached_key, requested_key)
            )
        if isinstance(cached_key, dict) and isinstance(requested_key, dict):
            return cached_key.keys() == requested_key.keys() and all(
                AutotuneKey.keys_fuzzy_match(cached_key[k], requested_key[k]) for k in cached_key
            )
        return cached_key == requested_key

    @classmethod
    def build(
        cls,
        arg_names: list[str],
        key_names: list[str],
        positional_args: tuple[Any, ...],
        runtime_kwargs: dict[str, Any],
    ) -> "AutotuneKey":
        named_args = dict(zip(arg_names, positional_args))
        all_args = {**named_args, **runtime_kwargs}
        tracked_args = {k: v for (k, v) in all_args.items() if k in arg_names}
        tuning_key = [tracked_args[name] for name in key_names if name in tracked_args]
        for arg in tracked_args.values():
            if hasattr(arg, "dtype"):
                tuning_key.append(str(arg.dtype))
        return cls(autotune_key=tuple(tuning_key))

    def exact_matches(self, entry_key: Any) -> bool:
        return self.serialize(self.autotune_key) == self.serialize(entry_key)

    def fuzzy_matches(self, entry_key: Any) -> bool:
        self_normalized = self.normalize_autotune_key(self.autotune_key)
        entry_normalized = self.normalize_autotune_key(entry_key)
        return (
            isinstance(self_normalized, list)
            and isinstance(entry_normalized, list)
            and len(self_normalized) == len(entry_normalized)
            and AutotuneKey.keys_fuzzy_match(self_normalized, entry_normalized)
        )


@dataclasses.dataclass(frozen=True)
class KernelConfigFile:
    """Validated in-memory representation of a {kernel_name}.json config file."""
    kernel_name: str | None
    triton_version: str | None
    autotune_entries: dict[str, dict[str, Any]] | None
    default_config: dict[str, Any] | None

    @classmethod
    def from_dict(cls, config_file: Path, data: Any) -> "KernelConfigFile | None":
        """Parse and validate a raw JSON dict. Returns None (with a warning) if malformed."""
        def fail(msg, *args):
            logger.warning(msg, *args)
            raise ValueError

        try:
            if not isinstance(data, dict):
                fail("Malformed config %s: root is %s, expected dict", config_file, type(data).__name__)
            raw_entries = data.get("autotune_entries")
            entries: dict[str, dict[str, Any]] | None = None
            if raw_entries is not None:
                if not isinstance(raw_entries, dict):
                    fail("Malformed config %s: 'autotune_entries' is %s, expected dict",
                         config_file, type(raw_entries).__name__)
                for h, entry in raw_entries.items():
                    if not isinstance(entry, dict):
                        fail("Malformed config %s: autotune_entries[%r] is %s, expected dict",
                             config_file, h, type(entry).__name__)
                    if not isinstance(entry.get("config"), dict):
                        fail("Malformed config %s: autotune_entries[%r] missing valid 'config' field", config_file, h)
                entries = raw_entries
            default_config = data.get("default_config")
            if default_config is not None and not isinstance(default_config, dict):
                fail("Malformed config %s: 'default_config' is %s, expected dict", config_file, type(default_config).__name__)
            return cls(
                kernel_name=data.get("kernel_name"),
                triton_version=data.get("triton_version"),
                autotune_entries=entries,
                default_config=default_config,
            )
        except ValueError:
            return None

    @classmethod
    def from_file(cls, config_file: Path) -> "KernelConfigFile | None":
        """Read and validate a config file. Returns None if the file is missing or malformed."""
        config_data = read_config_file(config_file)
        if config_data is None:
            return None
        return cls.from_dict(config_file, config_data)

    def lookup_exact(self, key: AutotuneKey) -> dict[str, Any] | None:
        if self.autotune_entries is None:
            return None
        return self.autotune_entries.get(AutotuneKey.key_hash(key.autotune_key))

    def lookup_fuzzy(self, key: AutotuneKey) -> dict[str, Any] | None:
        if self.autotune_entries is None:
            return None
        for entry in self.autotune_entries.values():
            if key.fuzzy_matches(entry.get("autotune_key")):
                return entry
        return None


@cache
def load_config_file(config_file: Path) -> dict[str, Any] | None:
    try:
        with open(config_file) as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Error reading config file %s: %s", config_file, e)
        return None


def read_config_file(config_file: Path) -> dict[str, Any] | None:
    """Read a config file, bypassing the in-process cache in ALWAYS mode."""
    if FLA_CACHE_MODE is FlaCacheMode.ALWAYS:
        return load_config_file.__wrapped__(config_file)
    return load_config_file(config_file)


def load_cached_config(kernel_name: str, autotune_key: AutotuneKey | None = None) -> dict[str, Any] | None:
    """
    Load cached best config for a kernel from FLA configs directory.

    This function loads the cached best configuration for a given kernel name
    from get_fla_config_dir()/{kernel_name}.json.

    Cache files may contain multiple autotune entries keyed by Triton's
    runtime tuning key plus a top-level default config.

    If the config file is not found or cannot be loaded, a warning is printed
    and None is returned, allowing fallback to Triton's autotune.

    The lookup mode is controlled by the FLA_CACHE_MODE environment variable (see FlaCacheMode).

    Args:
        kernel_name: Name of the kernel (e.g., "causal_conv1d_fwd_kernel")
        autotune_key: Triton autotune key for the current invocation

    Returns:
        Best config dictionary or None if not found or disabled
    """
    if FLA_CACHE_MODE is FlaCacheMode.DISABLED:
        return None

    config_dir = get_fla_config_dir()
    config_file = config_dir / f"{kernel_name}.json"

    if not config_file.exists():
        return None

    config_data = read_config_file(config_file)
    if config_data is None:
        return None
    config = KernelConfigFile.from_dict(config_file, config_data)
    if config is None:
        return None

    if FLA_CACHE_MODE is FlaCacheMode.DEFAULT or FLA_CACHE_MODE is FlaCacheMode.ALWAYS:
        return config.default_config

    # STRICT mode: exact match only, no fuzzy fallback
    if FLA_CACHE_MODE is FlaCacheMode.STRICT:
        if autotune_key is not None:
            entry = config.lookup_exact(autotune_key)
            if entry is not None:
                return entry["config"]
        return None

    # FULL and FUZZY modes: try exact key match first, then fuzzy match
    if autotune_key is not None:
        entry = config.lookup_exact(autotune_key) or config.lookup_fuzzy(autotune_key)
        if entry is not None:
            return entry["config"]

    if FLA_CACHE_MODE is FlaCacheMode.FUZZY:
        return None

    # FULL mode: fall back to default_config, then legacy raw config (no autotune_entries)
    if config.default_config is not None:
        return config.default_config
    if config.autotune_entries is not None:
        return None
    return config_data


class CachedAutotuner(Autotuner):
    """
    A modified autotuner that loads best config from FLA's config directory.

    This class extends Triton's Autotuner but overrides the run method to
    try loading cached configuration first before falling back to autotune.
    """

    def __init__(self, fn, arg_names, configs, key, reset_to_zero, restore_value, **kwargs):
        super().__init__(fn, arg_names, configs, key, reset_to_zero, restore_value, **kwargs)
        self.kernel_name = fn.fn.__name__ if hasattr(fn, 'fn') else fn.__name__

    def should_check_fla_cache(self, key: AutotuneKey) -> bool:
        if FLA_CACHE_MODE is FlaCacheMode.DISABLED:
            return False
        if FLA_CACHE_MODE is FlaCacheMode.ALWAYS:
            return True
        return key.autotune_key not in self.cache

    def run(self, *args, **kwargs):
        key = AutotuneKey.build(self.arg_names, self.keys, args, kwargs)
        if self.should_check_fla_cache(key):
            self.maybe_load_cached_config(key)
        return super().run(*args, **kwargs)

    def maybe_load_cached_config(self, key: AutotuneKey):
        best_config = load_cached_config(self.kernel_name, key)

        if best_config is not None:
            kw = best_config["kwargs"]
            num_warps = best_config["num_warps"]
            num_stages = best_config["num_stages"]

            extra = {
                "num_ctas": best_config["num_ctas"],
                "maxnreg": best_config.get("maxnreg"),
                "pre_hook": None,
                "ir_override": best_config.get("ir_override"),
            } if TRITON_ABOVE_3_5_1 else {}
            cfg = triton.Config(kw, num_warps=num_warps, num_stages=num_stages, **extra)

            self.cache[key.autotune_key] = cfg
        else:
            logger.debug(
                "No cached config found for kernel %s and key %s; falling back to Triton autotune",
                self.kernel_name,
                list(key.autotune_key),
            )


def fla_cache_autotune(configs, key=None, prune_configs_by=None, reset_to_zero=None, restore_value=None,
                       pre_hook=None, post_hook=None, warmup=None, rep=None, use_cuda_graph=False,
                       do_bench=None, cache_results=False):
    """
    Decorator for auto-tuning a :code:`triton.jit`'d function with FLA config support.

    Extends Triton's autotune to load best configurations from FLA's config directory
    (default: fla/configs/{GPU}/, or FLA_CONFIG_DIR/ when overridden), keyed by kernel
    name from {kernel_name}.json. Lookup behaviour is controlled by FLA_CACHE_MODE.
    Falls back to normal Triton autotuning when no cached config is found.
    """
    # key can be None when we want to use cache only (no fallback autotune)
    if key is None:
        key = []

    def decorator(fn):
        kwargs = {}
        if TRITON_ABOVE_3_4_0:
            kwargs = {"cache_results": cache_results}

        return CachedAutotuner(fn, fn.arg_names, configs, key, reset_to_zero, restore_value,
                               pre_hook=pre_hook, post_hook=post_hook,
                               prune_configs_by=prune_configs_by, warmup=warmup, rep=rep,
                               use_cuda_graph=use_cuda_graph, do_bench=do_bench,
                               **kwargs,
                               )

    return decorator


def configure_fla_fla_cache_autotune():
    triton.autotune = fla_cache_autotune
    logger.info(
        "configure_fla_fla_cache_autotune() is enabling FLA fla_cache_autotune; "
        "triton.autotune will be replaced with fla_cache_autotune."
    )


def restore_autotune_backend():
    from triton.runtime.autotuner import autotune as original_autotune
    triton.autotune = original_autotune
    logger.info(
        "restore_autotune_backend() is restoring Triton's original autotune; "
        "triton.autotune will be replaced with triton.runtime.autotuner.autotune."
    )

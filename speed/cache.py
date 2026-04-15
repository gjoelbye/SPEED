"""File-level caching for SPEED pipeline runs."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional


def load_cache(out_path: str) -> dict:
    """
    Load the cache index from the output directory.

    Parameters
    ----------
    out_path : str
        Output directory containing .speed_cache.json.

    Returns
    -------
    dict
        Cache entries keyed by input file path.
    """
    cache_path = Path(out_path) / ".speed_cache.json"
    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_cache(out_path: str, cache: dict) -> None:
    """
    Atomically write the cache index to the output directory.

    Parameters
    ----------
    out_path : str
        Output directory.
    cache : dict
        Cache entries to write.
    """
    out_dir = Path(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / ".speed_cache.json"

    # Atomic write: write to temp file then rename
    fd, tmp_path = tempfile.mkstemp(dir=str(out_dir), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(cache, f, indent=2, default=str)
        os.replace(tmp_path, str(cache_path))
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def compute_input_hash(path: str) -> str:
    """
    Compute a fast hash of an input file.

    Uses first 1MB of content + file size + mtime for speed.

    Parameters
    ----------
    path : str
        Path to the input file.

    Returns
    -------
    str
        Hex digest of the hash.
    """
    p = Path(path)
    stat = p.stat()
    h = hashlib.md5()
    h.update(str(stat.st_size).encode())
    h.update(str(stat.st_mtime).encode())
    with open(p, "rb") as f:
        h.update(f.read(1024 * 1024))  # First 1MB
    return h.hexdigest()


def compute_config_hash(pipeline) -> str:
    """
    Compute a hash of the pipeline configuration.

    Parameters
    ----------
    pipeline : BasePipeline
        The pipeline instance.

    Returns
    -------
    str
        Hex digest of the config hash.
    """
    config_dict = {
        k: v for k, v in sorted(vars(pipeline).items())
        if not k.startswith("_")
    }
    config_str = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.md5(config_str.encode()).hexdigest()


def is_cached(
    cache: dict,
    input_path: str,
    input_hash: str,
    config_hash: str,
) -> bool:
    """
    Check if a file has a valid cache entry.

    Parameters
    ----------
    cache : dict
        The loaded cache index.
    input_path : str
        Path to the input file.
    input_hash : str
        Current hash of the input file.
    config_hash : str
        Current hash of the pipeline config.

    Returns
    -------
    bool
        True if the cache entry is valid (file can be skipped).
    """
    key = str(input_path)
    if key not in cache:
        return False
    entry = cache[key]
    return (
        entry.get("input_hash") == input_hash
        and entry.get("config_hash") == config_hash
    )


def update_cache(
    cache: dict,
    input_path: str,
    input_hash: str,
    config_hash: str,
    output_path: str,
) -> None:
    """
    Add or update a cache entry.

    Parameters
    ----------
    cache : dict
        The cache index to update (mutated in place).
    input_path : str
        Path to the input file.
    input_hash : str
        Hash of the input file.
    config_hash : str
        Hash of the pipeline config.
    output_path : str
        Path to the output file produced.
    """
    import datetime

    cache[str(input_path)] = {
        "input_hash": input_hash,
        "config_hash": config_hash,
        "output_path": str(output_path),
        "timestamp": datetime.datetime.now().isoformat(),
    }

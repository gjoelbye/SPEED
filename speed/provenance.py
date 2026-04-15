"""Config provenance tracking for SPEED pipeline runs."""

import datetime
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


def _get_git_hash() -> Optional[str]:
    """Get the current git commit hash, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _get_package_version(package: str) -> Optional[str]:
    """Get installed package version via importlib.metadata."""
    try:
        from importlib.metadata import version
        return version(package)
    except Exception:
        return None


def save_provenance(
    out_path: str,
    config: Dict[str, Any],
    n_input: int,
    n_output: int,
) -> Path:
    """
    Save a provenance record to the output directory.

    Parameters
    ----------
    out_path : str
        Output directory where provenance.yaml will be written.
    config : dict
        The full resolved config dict.
    n_input : int
        Number of input files discovered.
    n_output : int
        Number of output files/batches produced.

    Returns
    -------
    Path
        Path to the written provenance file.
    """
    provenance = {
        "timestamp": datetime.datetime.now().isoformat(),
        "speed_version": _get_package_version("speed-eeg"),
        "python_version": platform.python_version(),
        "mne_version": _get_package_version("mne"),
        "git_hash": _get_git_hash(),
        "platform": platform.platform(),
        "n_input_files": n_input,
        "n_output_files": n_output,
        "config": config,
    }

    out_dir = Path(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = out_dir / "provenance.yaml"

    try:
        import yaml
        with open(provenance_path, "w") as f:
            yaml.dump(provenance, f, default_flow_style=False, sort_keys=False)
    except ImportError:
        # Fallback to JSON if PyYAML not available
        provenance_path = out_dir / "provenance.json"
        with open(provenance_path, "w") as f:
            json.dump(provenance, f, indent=2, default=str)

    return provenance_path

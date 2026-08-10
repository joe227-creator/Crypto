"""Optional Stage 3 availability probes; no silent fallback is allowed."""

from __future__ import annotations

import importlib.util
from typing import Any


def availability_report() -> dict[str, Any]:
    return {
        "timesfm_package": importlib.util.find_spec("timesfm") is not None,
        "torch_package": importlib.util.find_spec("torch") is not None,
        "kronos_source": "https://github.com/shiyu-coder/Kronos",
        "timesfm_source": "https://github.com/google-research/timesfm",
        "status": "available" if importlib.util.find_spec("timesfm") is not None else "blocked_dependency",
    }


def require_available(model: str) -> None:
    report = availability_report()
    if model == "timesfm" and not report["timesfm_package"]:
        raise RuntimeError("TimesFM dependency/checkpoint unavailable; refusing silent replacement")
    if model == "kronos" and not report["torch_package"]:
        raise RuntimeError("Kronos requires torch; refusing silent replacement")

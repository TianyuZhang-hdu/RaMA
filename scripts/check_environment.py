#!/usr/bin/env python3
from __future__ import annotations

import importlib
import platform
import subprocess
import sys


def check_module(name: str) -> str:
    try:
        mod = importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - diagnostic script
        return f"MISSING ({type(exc).__name__}: {exc})"
    return f"OK {getattr(mod, '__version__', '')}".strip()


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    for name in ["numpy", "PIL", "scipy", "tqdm", "yaml", "torch"]:
        print(f"{name}: {check_module(name)}")
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=5,
        )
        print("GPU:")
        print(out.strip())
    except Exception as exc:  # pragma: no cover - diagnostic script
        print(f"GPU: unavailable ({exc})")


if __name__ == "__main__":
    main()

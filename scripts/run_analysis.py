"""Run the whole post-training analysis chain in dependency order."""
from __future__ import annotations

import os
import subprocess
import sys
import time

STEPS = [
    ("slices", "src.analysis.slices"),
    ("gates", "src.analysis.gates"),
    ("calibration", "src.analysis.calibration"),
    ("robustness", "src.analysis.robustness"),
    ("compare", "src.analysis.compare"),
    ("figures", "src.analysis.figures"),
]


def main() -> None:
    env = {**os.environ, "PYTHONPATH": ".", "EG_HORIZON": os.environ.get("EG_HORIZON", "24")}
    only = set(sys.argv[1:]) or None
    for name, mod in STEPS:
        if only and name not in only:
            continue
        print(f"\n===== {name} =====", flush=True)
        t0 = time.time()
        p = subprocess.run([sys.executable, "-m", mod], env=env)
        print(f"----- {name}: exit={p.returncode} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

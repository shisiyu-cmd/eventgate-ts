"""Sequential experiment driver: trains every model/seed once, logging to runs/_matrix.

Usage
-----
python -m scripts.run_matrix --stage main
python -m scripts.run_matrix --stage ablation
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

RUNS = Path("runs")
ENV = {"PYTHONPATH": ".", "EG_HORIZON": "24"}


def call(args: list[str]) -> None:
    print(f"\n>>> {' '.join(args)}", flush=True)
    t0 = time.time()
    p = subprocess.run([sys.executable, "-m", *args], env={**__import__("os").environ, **ENV})
    print(f"<<< exit={p.returncode} in {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="main")
    a = ap.parse_args()

    if a.stage == "main":
        for seed in (0, 1, 2):
            call(["src.train", "--model", "eventgate", "--seed", str(seed), "--epochs", "180", "--bs", "256"])
            call(["src.train", "--model", "patchtst", "--seed", str(seed), "--epochs", "180", "--bs", "256"])
        for m in ("lstm", "tcn", "mlp"):
            call(["src.train", "--model", m, "--seed", "0", "--epochs", "180", "--bs", "256"])
    elif a.stage == "ablation":
        for flag in ("--no_freq", "--no_gate", "--no_ev", "--no_revin"):
            call(["src.train", "--model", "eventgate", "--seed", "0", "--epochs", "180",
                  "--bs", "256", flag])
    elif a.stage == "quantile":
        call(["src.train", "--model", "eventgate", "--seed", "0", "--epochs", "180",
              "--bs", "256", "--n_q", "3", "--tag", "quantile"])
    print("done", flush=True)


if __name__ == "__main__":
    main()

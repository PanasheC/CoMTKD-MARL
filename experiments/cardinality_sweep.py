"""Launch fixed-cardinality CoMTKD-MARL experiments from M=1 through M_max."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-teachers", type=int, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--output-root", default="./checkpoints/cardinality_sweep")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for cardinality in range(1, args.max_teachers + 1):
            command = [
                sys.executable,
                "train_student_rl.py",
                "--seed",
                str(seed),
                "--trial",
                f"seed{seed}-M{cardinality}",
                "--forced-cardinality",
                str(cardinality),
                "--checkpoint-dir",
                args.output_root,
                *args.extra,
            ]
            print(" ".join(command), flush=True)
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

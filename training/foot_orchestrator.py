from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Run contact-pressure surrogate training jobs sequentially.")
    parser.add_argument("--models", nargs="+", default=["gno", "pinn", "ffn"], choices=["gno", "pinn", "ffn"])
    parser.add_argument("--config", default="training/foot_config.yaml")
    parser.add_argument("--shard-dir", default=None)
    parser.add_argument("--wandb-mode", default=None, choices=["online", "offline", "disabled"])
    args = parser.parse_args()

    script_by_model = {
        "gno": "training/train_gno.py",
        "pinn": "training/train_pinn.py",
        "ffn": "training/train_ffn.py",
    }
    for model in args.models:
        cmd = [sys.executable, script_by_model[model], "--config", args.config]
        if args.shard_dir:
            cmd += ["--shard-dir", args.shard_dir]
        if args.wandb_mode:
            cmd += ["--wandb-mode", args.wandb_mode]
        print(f"[RUN] {' '.join(cmd)}")
        subprocess.check_call(cmd)


if __name__ == "__main__":
    main()

"""
Run the full experiment pipeline:
  1. run_pretrain_fast.py  — pretrain a model (correlation=0)
  2. run_finetune_fast_w_metrics.py — finetune + reverse with gradient metrics

Each script is run as a subprocess so their module-level state is fully isolated.
"""

import subprocess
import sys


def run(script):
    print(f"\n{'='*80}")
    print(f"Running: {script}")
    print(f"{'='*80}\n")
    result = subprocess.run([sys.executable, script], check=True)
    return result


if __name__ == "__main__":
    run("run_pretrain_fast.py")
    run("run_finetune_fast_w_metrics.py")
    print(f"\n{'='*80}")
    print("Full pipeline complete.")
    print(f"{'='*80}")

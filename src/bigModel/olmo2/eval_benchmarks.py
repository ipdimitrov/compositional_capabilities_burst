#!/usr/bin/env python3
"""
Benchmark evaluation for OLMo-2-1B concentration experiment checkpoints.

Evaluates saved checkpoints on:
  - Pretraining retention: HellaSwag, ARC-Easy, ARC-Challenge, MMLU (subset)
  - Code capability: HumanEval (pass@1), MBPP (pass@1)

Can be run standalone on a checkpoint directory, or batch-evaluate all
checkpoints from a training run.

Usage:
    # Evaluate a single checkpoint
    python olmo2/eval_benchmarks.py --checkpoint_path olmo2/results/c1.0_lr5e-05_s42/checkpoints/step_500

    # Evaluate all checkpoints in a run
    python olmo2/eval_benchmarks.py --run_dir olmo2/results/c1.0_lr5e-05_s42

    # Evaluate the base model (no fine-tuning)
    python olmo2/eval_benchmarks.py --model_id allenai/OLMo-2-0425-1B --out_dir olmo2/results/baseline

    # Only run cheap benchmarks (skip code generation)
    python olmo2/eval_benchmarks.py --run_dir olmo2/results/c1.0_lr5e-05_s42 --skip_code
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

BASE_MODEL_ID = "allenai/OLMo-2-0425-1B"


# ---------------------------------------------------------------------------
# lm-evaluation-harness integration
# ---------------------------------------------------------------------------

def run_lm_eval(
    model_path: str,
    tasks: List[str],
    num_fewshot: int = 0,
    limit: Optional[int] = None,
    batch_size: str = "auto",
    output_dir: Optional[str] = None,
) -> Dict[str, float]:
    """
    Run lm-evaluation-harness on the given model checkpoint.
    Returns dict of {task_name: metric_value}.
    """
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},trust_remote_code=True,dtype=bfloat16",
        "--tasks", ",".join(tasks),
        "--num_fewshot", str(num_fewshot),
        "--batch_size", batch_size,
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    if output_dir:
        cmd.extend(["--output_path", output_dir])

    logger.info(f"Running: {' '.join(cmd)}")

    env = os.environ.copy()
    env["HF_ALLOW_CODE_EVAL"] = "1"

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=7200, env=env,
        )
        if result.returncode != 0:
            logger.error(f"lm_eval failed:\n{result.stderr}")
            return {}
    except subprocess.TimeoutExpired:
        logger.error("lm_eval timed out (2h)")
        return {}
    except FileNotFoundError:
        logger.error("lm_eval not found. Install: pip install lm-eval")
        return {}

    # Parse output JSON — lm-eval v0.4+ nests results in a model-named subdirectory
    scores = {}
    if output_dir:
        # Find the results JSON: could be at output_dir/results.json or
        # output_dir/{model_subdir}/results_{timestamp}.json
        results_files = list(Path(output_dir).rglob("results*.json"))
        if results_files:
            # Use the most recent one
            results_file = max(results_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Parsing results from {results_file}")
            with open(results_file) as f:
                data = json.load(f)
            for task, metrics in data.get("results", {}).items():
                for metric_name in ["acc_norm,none", "acc,none", "acc_norm", "acc"]:
                    if metric_name in metrics:
                        scores[task] = metrics[metric_name]
                        break

    # Fallback: parse from stdout table
    if not scores:
        for line in result.stdout.split("\n"):
            line = line.strip()
            if "|" in line and any(t in line for t in tasks):
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 4:
                    try:
                        scores[parts[1]] = float(parts[3])
                    except (ValueError, IndexError):
                        pass

    return scores


# ---------------------------------------------------------------------------
# HumanEval / MBPP via bigcode-evaluation-harness
# ---------------------------------------------------------------------------

def run_code_eval(
    model_path: str,
    task: str = "humaneval",
    n_samples: int = 1,
    temperature: float = 0.0,
    output_dir: Optional[str] = None,
) -> Dict[str, float]:
    """
    Run code generation benchmarks using bigcode-evaluation-harness.
    Returns dict with pass@1.
    """
    cmd = [
        sys.executable, "-m", "bigcode_eval.main",
        "--model", model_path,
        "--tasks", task,
        "--n_samples", str(n_samples),
        "--temperature", str(temperature),
        "--trust_remote_code",
        "--allow_code_execution",
        "--precision", "bf16",
        "--batch_size", "1",
    ]
    if output_dir:
        cmd.extend(["--save_generations_path", str(Path(output_dir) / f"{task}_generations.json")])
        cmd.extend(["--metric_output_path", str(Path(output_dir) / f"{task}_metrics.json")])

    logger.info(f"Running: {' '.join(cmd)}")

    env = os.environ.copy()
    env["HF_ALLOW_CODE_EVAL"] = "1"

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=7200, env=env,
        )
        if result.returncode != 0:
            logger.warning(f"bigcode_eval for {task} failed:\n{result.stderr[:500]}")
            return _fallback_code_eval(model_path, task, output_dir)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.warning(f"bigcode_eval not available, trying lm_eval fallback for {task}")
        return _fallback_code_eval(model_path, task, output_dir)

    # Parse metrics
    scores = {}
    if output_dir:
        metrics_file = Path(output_dir) / f"{task}_metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                data = json.load(f)
            for k, v in data.items():
                if "pass@1" in k.lower() or k == "pass@1":
                    scores[f"{task}_pass@1"] = v
    return scores


def _fallback_code_eval(model_path: str, task: str, output_dir: Optional[str]) -> Dict[str, float]:
    """Fallback: use lm-evaluation-harness for code tasks if available."""
    task_map = {
        "humaneval": "humaneval",
        "mbpp": "mbpp",
    }
    lm_task = task_map.get(task, task)
    scores = run_lm_eval(model_path, [lm_task], output_dir=output_dir)
    return {f"{task}_pass@1": scores.get(lm_task, float("nan"))}


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_checkpoint(
    model_path: str,
    out_dir: str,
    skip_code: bool = False,
    mmlu_limit: Optional[int] = 2000,
) -> Dict[str, float]:
    """Run all benchmarks on a single checkpoint."""
    os.makedirs(out_dir, exist_ok=True)
    all_scores: Dict[str, float] = {}

    # --- Pretraining retention benchmarks ---
    logger.info("=== Pretraining retention benchmarks ===")

    # HellaSwag (0-shot, acc_norm)
    logger.info("Running HellaSwag...")
    hs = run_lm_eval(model_path, ["hellaswag"], num_fewshot=0, output_dir=os.path.join(out_dir, "hellaswag"))
    all_scores.update({f"hellaswag_{k}": v for k, v in hs.items()} if hs else {"hellaswag": float("nan")})

    # ARC-Easy and ARC-Challenge (0-shot)
    logger.info("Running ARC...")
    arc = run_lm_eval(model_path, ["arc_easy", "arc_challenge"], num_fewshot=0, output_dir=os.path.join(out_dir, "arc"))
    all_scores.update(arc)

    # MMLU (5-shot, with limit for speed)
    logger.info("Running MMLU...")
    mmlu_args = {"num_fewshot": 5}
    if mmlu_limit:
        mmlu_args["limit"] = mmlu_limit
    mmlu = run_lm_eval(model_path, ["mmlu"], output_dir=os.path.join(out_dir, "mmlu"), **mmlu_args)
    all_scores.update(mmlu)

    # --- Code benchmarks ---
    if not skip_code:
        logger.info("=== Code capability benchmarks ===")

        logger.info("Running HumanEval...")
        he = run_code_eval(model_path, "humaneval", output_dir=os.path.join(out_dir, "humaneval"))
        all_scores.update(he)

        logger.info("Running MBPP...")
        mbpp = run_code_eval(model_path, "mbpp", output_dir=os.path.join(out_dir, "mbpp"))
        all_scores.update(mbpp)

    # Save results
    results = {
        "model_path": model_path,
        "scores": all_scores,
    }
    with open(os.path.join(out_dir, "benchmark_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Benchmark results: {json.dumps(all_scores, indent=2)}")
    return all_scores


def evaluate_all_checkpoints(run_dir: str, skip_code: bool = False, mmlu_limit: Optional[int] = 2000):
    """Evaluate all checkpoints in a training run directory."""
    run_path = Path(run_dir)
    ckpt_dir = run_path / "checkpoints"

    if not ckpt_dir.exists():
        logger.error(f"No checkpoints directory found at {ckpt_dir}")
        return

    # Find all checkpoint directories, sorted by step number
    ckpts = []
    for d in ckpt_dir.iterdir():
        if d.is_dir() and d.name.startswith("step_"):
            try:
                step = int(d.name.split("_")[1])
                ckpts.append((step, d))
            except ValueError:
                continue
    ckpts.sort(key=lambda x: x[0])

    if not ckpts:
        logger.error(f"No checkpoints found in {ckpt_dir}")
        return

    logger.info(f"Found {len(ckpts)} checkpoints: {[s for s, _ in ckpts]}")

    all_results = []
    for step, ckpt_path in ckpts:
        logger.info(f"\n{'='*70}\nEvaluating step {step}: {ckpt_path}\n{'='*70}")
        out_dir = str(run_path / "benchmark_evals" / f"step_{step}")
        scores = evaluate_checkpoint(str(ckpt_path), out_dir, skip_code=skip_code, mmlu_limit=mmlu_limit)
        all_results.append({"step": step, "checkpoint": str(ckpt_path), "scores": scores})

    # Save combined results
    combined_path = run_path / "all_benchmark_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nAll results saved to {combined_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark evaluation for OLMo-2 checkpoints")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Single checkpoint to evaluate")
    parser.add_argument("--run_dir", type=str, default=None, help="Run directory with checkpoints/ subdirectory")
    parser.add_argument("--model_id", type=str, default=None, help="HF model ID (for base model eval)")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory for results")
    parser.add_argument("--skip_code", action="store_true", help="Skip HumanEval/MBPP")
    parser.add_argument("--mmlu_limit", type=int, default=2000, help="MMLU sample limit (0 for full)")
    args = parser.parse_args()

    mmlu_limit = args.mmlu_limit if args.mmlu_limit > 0 else None

    if args.run_dir:
        evaluate_all_checkpoints(args.run_dir, skip_code=args.skip_code, mmlu_limit=mmlu_limit)
    elif args.checkpoint_path:
        out_dir = args.out_dir or str(Path(args.checkpoint_path) / "benchmark_evals")
        evaluate_checkpoint(args.checkpoint_path, out_dir, skip_code=args.skip_code, mmlu_limit=mmlu_limit)
    elif args.model_id:
        out_dir = args.out_dir or f"olmo2/results/baseline_{args.model_id.replace('/', '_')}"
        evaluate_checkpoint(args.model_id, out_dir, skip_code=args.skip_code, mmlu_limit=mmlu_limit)
    else:
        parser.error("Provide --checkpoint_path, --run_dir, or --model_id")


if __name__ == "__main__":
    main()

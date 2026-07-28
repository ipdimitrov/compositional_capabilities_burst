#!/usr/bin/env python3
"""Main entry point for the catastrophic forgetting experiment.

Usage:
    python run_experiment.py                                    # 70m, normal, chemistry
    python run_experiment.py --model 1b --preset deep           # 1b model
    python run_experiment.py --domain biomedical --grads        # with gradient analysis
    python run_experiment.py --plots_only --results_dir results/latest
"""

import argparse

from config import DOMAIN_PRESETS, MODEL_PRESETS, PRESETS, get_config
from plot import generate_all_plots
from train import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Catastrophic Forgetting Experiment")
    parser.add_argument("--preset", type=str, default="normal",
                        choices=list(PRESETS.keys()),
                        help="Config preset (quick/normal/full/deep/deepest)")
    parser.add_argument("--model", type=str, default="70m",
                        choices=list(MODEL_PRESETS.keys()),
                        help="Model preset (70m/1b)")
    parser.add_argument("--domain", type=str, default="chemistry",
                        choices=list(DOMAIN_PRESETS.keys()),
                        help="Domain dataset preset (chemistry/music/biomedical)")
    parser.add_argument("--ft_steps", type=int, default=None, help="Fine-tuning steps")
    parser.add_argument("--cpt_steps", type=int, default=None, help="Continued pretraining steps")
    parser.add_argument("--eval_every", type=int, default=None, help="Eval interval")
    parser.add_argument("--burst_levels", type=float, nargs="+", default=None,
                        help="Burst levels to run (e.g. 1.0 0.75 0.5 0.25)")
    parser.add_argument("--ft_budget_mode", type=str, default=None,
                        choices=["volume", "steps"],
                        help="'volume' scales ft_steps by 1/burst (same domain volume); "
                             "'steps' uses raw ft_steps for every burst. Default: volume.")
    parser.add_argument("--results_dir", type=str, default=None, help="Results directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--grads", action="store_true",
                        help="Enable gradient cosine similarity & norm analysis (slower)")
    parser.add_argument("--grad_every", type=int, default=None,
                        help="Compute gradient metrics every N steps (default: 100)")
    parser.add_argument("--save_checkpoints", action="store_true",
                        help="Save post-FT and post-CPT model checkpoints per burst "
                             "(off by default; disk-heavy for large models)")
    parser.add_argument("--max_train_chunks", type=int, default=None,
                        help="Cap training chunks per dataset (saves RAM for large-doc datasets)")
    parser.add_argument("--plots_only", action="store_true",
                        help="Only generate plots from existing metrics")

    args = parser.parse_args()

    # Build overrides from CLI args
    overrides = {}
    for key in ["ft_steps", "cpt_steps", "eval_every", "burst_levels", "results_dir",
                "seed", "max_train_chunks", "ft_budget_mode"]:
        val = getattr(args, key)
        if val is not None:
            overrides[key] = val
    if args.grads:
        overrides["compute_grad_metrics"] = True
    if args.grad_every is not None:
        overrides["grad_metrics_every"] = args.grad_every
    if args.save_checkpoints:
        overrides["save_checkpoints"] = True

    config = get_config(args.preset, domain=args.domain, model=args.model, **overrides)

    if args.plots_only:
        metrics_path = f"{config.results_dir}/metrics.json"
        generate_all_plots(metrics_path, config.results_dir)
        return

    if config.ft_budget_mode == "volume":
        pass
    else:
        pass
    if config.compute_grad_metrics:
        pass

    # Run experiment
    run_experiment(config)

    # Generate plots
    metrics_path = f"{config.results_dir}/metrics.json"
    generate_all_plots(metrics_path, config.results_dir)


if __name__ == "__main__":
    main()

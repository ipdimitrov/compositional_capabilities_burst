#!/usr/bin/env python3
"""Unified entry point for burst forgetting experiments.

Usage:
    # Run a config end-to-end (train + plot)
    python run_experiment.py --config configs/correlation_sweep.yaml

    # Train only (skip plots)
    python run_experiment.py --config configs/burst_simple.yaml --no-plot

    # Plot only (reuse existing results)
    python run_experiment.py --plot-only --out data/sweep_20260413_120000

    # Resume an interrupted run
    python run_experiment.py --config configs/correlation_sweep.yaml --resume --out data/sweep_20260412_200000

    # Override config values
    python run_experiment.py --config configs/correlation_sweep.yaml --override seeds=[42,123] ft_steps=2000
"""
import argparse
import ast
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Make `burst` importable when script is run directly
_HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(_HERE))

from burst.sweep import SweepConfig, run_sweep
from burst.plot import make_all_plots


def _load_config_file(path: str) -> dict:
    """Load YAML or JSON config into a dict."""
    p = Path(path)
    text = p.read_text()
    if p.suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise RuntimeError("pyyaml is required for YAML configs. pip install pyyaml")
        return yaml.safe_load(text)
    return json.loads(text)


def _apply_overrides(cfg_dict: dict, overrides: list[str]) -> dict:
    """Apply `key=value` overrides. Values are parsed with ast.literal_eval."""
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be key=value, got: {override}")
        key, raw_val = override.split("=", 1)
        try:
            val = ast.literal_eval(raw_val)
        except (ValueError, SyntaxError):
            val = raw_val  # fall back to string
        cfg_dict[key] = val
    return cfg_dict


def main():
    parser = argparse.ArgumentParser(
        description="Run a burst forgetting experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", type=str,
                        help="Path to YAML/JSON config file")
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory (default: data/sweep_YYYYMMDD_HHMMSS)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip existing result files (continues interrupted runs)")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Override config values: key=value (e.g., seeds=[42,123])")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plotting after training")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip training, only make plots from --out")

    args = parser.parse_args()

    # ── plot-only mode ─────────────────────────────────────────
    if args.plot_only:
        if args.out is None:
            parser.error("--plot-only requires --out pointing to a results dir")
        make_all_plots(args.out)
        return

    # ── training: load config ──────────────────────────────────
    if args.config is None:
        print("No config specified, using defaults (burst_simple style)")
        cfg_dict = {}
    else:
        cfg_dict = _load_config_file(args.config)

    cfg_dict = _apply_overrides(cfg_dict, args.override)
    cfg = SweepConfig.from_dict(cfg_dict)

    # Output dir
    if args.out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out = f"data/sweep_{ts}"
    out_dir = Path(args.out)

    # ── train ──────────────────────────────────────────────────
    run_sweep(cfg, out_dir, resume=args.resume)

    # ── plot ───────────────────────────────────────────────────
    if not args.no_plot:
        print()
        make_all_plots(out_dir)


if __name__ == "__main__":
    main()

"""Per-run combined analysis dashboard — thin wrapper around pres_pdf.py --full.

Kept for backwards compatibility. All output is now in:
  results/burst_report.html
  results/burst_report.txt

Usage:
    python burst/run_analysis.py data/burst_d3_pos3_<tag> [--n-seeds 3]
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathlib import Path


def run_analysis(run_dir: Path, n_seeds: int = 3, **kwargs) -> None:
    from burst.pres_pdf import main as pres_main
    sys.argv = [
        "burst/pres_pdf.py", str(run_dir),
        "--full", "--n-seeds", str(n_seeds),
    ]
    pres_main()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-run combined analysis dashboard (wrapper for pres_pdf.py --full).",
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--skip-unified", action="store_true")
    parser.add_argument("--skip-basin", action="store_true")
    parser.add_argument("--skip-extended", action="store_true")
    args = parser.parse_args()

    if args.skip_unified and args.skip_basin and args.skip_extended:
        from burst.pres_pdf import main as pres_main
        sys.argv = ["burst/pres_pdf.py", str(args.run_dir)]
        pres_main()
    else:
        run_analysis(args.run_dir, n_seeds=args.n_seeds)


if __name__ == "__main__":
    main()

"""Per-run combined analysis dashboard.

Runs unified_analysis and basin_metrics for a single run directory, then
assembles everything into one HTML + one machine-readable TXT in
<run_dir>/results/:

  results/analysis.html   — combined interactive Plotly dashboard:
                             1) plots/ PNGs (excluding files starting with 00-08)
                             2) unified_analysis charts
                             3) extended metrics charts
                             4) basin_metrics charts
  results/analysis.txt    — machine-readable companion text report

Usage:
    python burst/run_analysis.py data/burst_d3_pos3_<tag> [--n-seeds 3]

Dimension key:
    B: batch_size
    L: sequence_length (doc_len)
    N: n_embd (model dimension)
    V: vocab_size
"""
import sys, os, argparse, base64, re, time, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path



# ---------------------------------------------------------------------------
# PNG helpers
# ---------------------------------------------------------------------------

def _png_to_html_img(path: Path, max_width: int = 1000) -> str:
    data = base64.b64encode(path.read_bytes()).decode()
    return (
        f'<img src="data:image/png;base64,{data}" '
        f'style="max-width:{max_width}px;width:100%;border:1px solid #eee;border-radius:4px;">'
    )


# ---------------------------------------------------------------------------
# Collect plots/ PNGs (exclude files whose stem starts with 00-08)
# ---------------------------------------------------------------------------

_EXCLUDE_PREFIX_RE = re.compile(r"^0[0-8]")


def _collect_plot_pngs(run_dir: Path) -> list[tuple[str, Path]]:
    """Return (title, path) pairs for plots/ PNGs that don't start with 00-08."""
    plots_dir = run_dir / "results" / "plots"
    if not plots_dir.exists():
        plots_dir = run_dir / "plots"
    if not plots_dir.exists():
        return []
    pairs: list[tuple[str, Path]] = []
    for p in sorted(plots_dir.glob("*.png")):
        if _EXCLUDE_PREFIX_RE.match(p.stem):
            continue
        title = p.stem.replace("_", " ").title()
        pairs.append((title, p))
    return pairs


def _pngs_from_charts_dir(charts_dir: Path) -> list[tuple[str, Path]]:
    """Collect (title, path) from a charts/ directory of PNGs."""
    if not charts_dir.exists():
        return []
    return [
        (p.stem.replace("_", " ").title(), p)
        for p in sorted(charts_dir.glob("*.png"))
    ]


# ---------------------------------------------------------------------------
# Run unified_analysis for a single run
# ---------------------------------------------------------------------------

def _run_unified(
    run_dir: Path,
    n_seeds: int,
    n_prune_levels: int,
    relearn_steps: int,
    frank_seeds: int,
    xfrank_seeds: int,
    subsample_n: int,
    tmp_dir: Path,
) -> list[tuple[str, Path]]:
    """Run unified_analysis.analyse_run + make_dashboard, return PNG chart paths."""
    from burst.unified_analysis import analyse_run as ua_analyse, make_dashboard

    t0 = time.time()
    print(f"\n[unified_analysis] Analysing {run_dir.name}...", flush=True)
    r = ua_analyse(
        run_dir,
        n_seeds=n_seeds,
        n_prune_levels=n_prune_levels,
        relearn_steps=relearn_steps,
        frank_seeds=frank_seeds,
        xfrank_seeds=xfrank_seeds,
        subsample_n=subsample_n,
    )
    print(f"  Done in {time.time()-t0:.1f}s", flush=True)

    run_name = r["run_name"]
    metric_keys = [
        "ema_dual", "lmc_dual", "frankenstein", "cross_frankenstein",
        "transfer_dual", "pruning_dual", "relearning_dual",
        "trajectory_dim", "forgetting_decomposition", "grad_temporal",
        "layer_interference", "sharpness",
        "grad_norm_ratio", "grad_rank", "grad_snr", "conflict_rate",
        "token_pos_grad", "grad_attribution", "forgetting_grad_alignment",
        "weight_drift_per_layer", "effective_rank_per_layer",
        "cka_per_layer", "directional_pruning",
    ]
    combined: dict = {
        "run_names": [run_name],
        "burst_positions": {run_name: r["burst_pos"]},
        "n_layer": r.get("n_layer", 6),
    }
    for mk in metric_keys:
        if mk in r:
            combined[mk] = {run_name: r[mk]}

    tmp_dir.mkdir(parents=True, exist_ok=True)
    make_dashboard(combined, tmp_dir)
    return _pngs_from_charts_dir(tmp_dir / "charts")


# ---------------------------------------------------------------------------
# Run extended metrics dashboard for a single run
# ---------------------------------------------------------------------------

def _run_extended(run_dir: Path, tmp_dir: Path) -> list[tuple[str, Path]]:
    """Run make_extended_metrics_dashboard for a single run, return PNG chart paths."""
    from burst.unified_analysis import make_extended_metrics_dashboard

    tmp_dir.mkdir(parents=True, exist_ok=True)
    make_extended_metrics_dashboard([run_dir], tmp_dir)
    return _pngs_from_charts_dir(tmp_dir / "charts")


# ---------------------------------------------------------------------------
# Run basin_metrics for a single run
# ---------------------------------------------------------------------------

def _run_basin(
    run_dir: Path,
    n_seeds: int,
    skip_surface: bool,
    tmp_dir: Path,
) -> list[tuple[str, Path]]:
    """Run basin_metrics.analyse_run + make_dashboard, return PNG chart paths."""
    from burst.basin_metrics import analyse_run as bm_analyse, make_dashboard as bm_dashboard

    t0 = time.time()
    print(f"\n[basin_metrics] Analysing {run_dir.name}...", flush=True)
    r = bm_analyse(run_dir, n_seeds=n_seeds, skip_surface=skip_surface)
    print(f"  Done in {time.time()-t0:.1f}s", flush=True)

    tmp_dir.mkdir(parents=True, exist_ok=True)
    bm_dashboard({run_dir.name: r}, tmp_dir)
    return _pngs_from_charts_dir(tmp_dir / "charts")


# ---------------------------------------------------------------------------
# Assemble combined HTML
# ---------------------------------------------------------------------------

_CSS = """
body { font-family: Arial, sans-serif; margin: 20px; background: #f0f2f5; color: #1a1a2e; }
h1 { color: #1a1a2e; font-size: 1.8em; border-bottom: 3px solid #1565c0; padding-bottom: 8px; }
h2 { color: #16213e; font-size: 1.4em; margin-top: 2.5em; border-left: 4px solid #1565c0;
     padding-left: 10px; }
h3 { color: #16213e; font-size: 1.05em; margin-top: 1.2em; }
.chart-container {
  background: white; border-radius: 10px; padding: 20px;
  margin: 16px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
.toc { background: white; border-radius: 10px; padding: 20px; margin: 20px 0;
       box-shadow: 0 2px 8px rgba(0,0,0,0.1); max-width: 900px; }
.toc a { display: block; margin: 3px 0; color: #1565c0; text-decoration: none; font-size: 0.92em; }
.toc a:hover { text-decoration: underline; }
.section-header { background: #e8eaf6; border-radius: 8px; padding: 8px 16px;
                  margin: 30px 0 6px 0; }
.meta { color: #888; font-family: monospace; font-size: 0.85em; }
"""


def _build_html(
    run_dir: Path,
    run_name: str,
    plot_pairs: list[tuple[str, Path]],
    unified_pairs: list[tuple[str, Path]],
    extended_pairs: list[tuple[str, Path]],
    basin_pairs: list[tuple[str, Path]],
) -> str:
    parts: list[str] = []
    parts.append(
        f"<!DOCTYPE html>\n<html>\n<head>\n<meta charset='utf-8'>\n"
        f"<title>Analysis: {run_name}</title>\n"
        f"<style>{_CSS}</style>\n</head>\n<body>\n"
    )
    parts.append(f"<h1>Analysis Dashboard: {run_name}</h1>\n")
    parts.append(f'<p class="meta">Generated from {run_dir}</p>\n')

    toc_entries: list[tuple[str, str]] = []

    for i, (title, _) in enumerate(plot_pairs):
        toc_entries.append((f"plot_{i}", f"[Plots] {title}"))
    for i, (title, _) in enumerate(unified_pairs):
        toc_entries.append((f"ua_{i}", f"[Unified] {title}"))
    for i, (title, _) in enumerate(extended_pairs):
        toc_entries.append((f"ext_{i}", f"[Extended] {title}"))
    for i, (title, _) in enumerate(basin_pairs):
        toc_entries.append((f"basin_{i}", f"[Basin] {title}"))

    parts.append('<div class="toc"><strong>Contents</strong><br>\n')
    for anchor, label in toc_entries:
        parts.append(f'  <a href="#{anchor}">{label}</a>\n')
    parts.append("</div>\n")

    def _png_section(header: str, pairs: list[tuple[str, Path]], prefix: str) -> None:
        if not pairs:
            return
        parts.append(f'<div class="section-header"><h2>{header}</h2></div>\n')
        for i, (title, path) in enumerate(pairs):
            parts.append(f'<div class="chart-container" id="{prefix}_{i}">\n')
            parts.append(f"<h3>{title}</h3>\n")
            parts.append(_png_to_html_img(path))
            parts.append("\n</div>\n")

    _png_section("Training Plots", plot_pairs, "plot")
    _png_section("Unified Analysis", unified_pairs, "ua")
    _png_section("Extended Metrics", extended_pairs, "ext")
    _png_section("Basin Geometry Metrics", basin_pairs, "basin")

    parts.append("</body></html>\n")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Assemble combined TXT
# ---------------------------------------------------------------------------

def _build_txt(
    run_name: str,
    plot_pairs: list[tuple[str, Path]],
    unified_txt: Path | None,
    extended_txt: Path | None,
    basin_txt: Path | None,
) -> str:
    lines: list[str] = [
        "=" * 60,
        f"Analysis Dashboard: {run_name}",
        "=" * 60,
        "",
    ]

    if plot_pairs:
        lines.append("--- SECTION: Training Plots ---")
        lines.append("")
        for title, path in plot_pairs:
            lines.append(f"[PNG] {title}: {path.name}")
        lines.append("")

    for header, txt_path in [
        ("Unified Analysis", unified_txt),
        ("Extended Metrics", extended_txt),
        ("Basin Geometry Metrics", basin_txt),
    ]:
        if txt_path and txt_path.exists():
            lines.append(f"--- SECTION: {header} ---")
            lines.append("")
            lines.append(txt_path.read_text())
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_analysis(
    run_dir: Path,
    n_seeds: int = 3,
    n_prune_levels: int = 10,
    relearn_steps: int = 50,
    frank_seeds: int = 3,
    xfrank_seeds: int = 3,
    subsample_n: int = 256,
    skip_surface: bool = False,
    skip_unified: bool = False,
    skip_basin: bool = False,
    skip_extended: bool = False,
) -> None:
    run_dir = Path(run_dir)
    run_name = run_dir.name
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}", flush=True)
    print(f"run_analysis: {run_name}", flush=True)
    print(f"{'='*60}", flush=True)

    plot_pairs = _collect_plot_pngs(run_dir)
    print(f"  Found {len(plot_pairs)} plot PNGs (excluding 00-08 prefix)", flush=True)

    unified_tmp = results_dir / "_unified_tmp"
    extended_tmp = results_dir / "_extended_tmp"
    basin_tmp = results_dir / "_basin_tmp"

    unified_pairs: list[tuple[str, Path]] = []
    if not skip_unified:
        try:
            unified_pairs = _run_unified(
                run_dir,
                n_seeds=n_seeds,
                n_prune_levels=n_prune_levels,
                relearn_steps=relearn_steps,
                frank_seeds=frank_seeds,
                xfrank_seeds=xfrank_seeds,
                subsample_n=subsample_n,
                tmp_dir=unified_tmp,
            )
        except Exception:
            print(f"  WARNING: unified_analysis failed:\n{traceback.format_exc()}", flush=True)

    extended_pairs: list[tuple[str, Path]] = []
    if not skip_extended:
        try:
            extended_pairs = _run_extended(run_dir, tmp_dir=extended_tmp)
        except Exception:
            print(f"  WARNING: extended metrics failed:\n{traceback.format_exc()}", flush=True)

    basin_pairs: list[tuple[str, Path]] = []
    if not skip_basin:
        try:
            basin_pairs = _run_basin(
                run_dir, n_seeds=n_seeds, skip_surface=skip_surface, tmp_dir=basin_tmp,
            )
        except Exception:
            print(f"  WARNING: basin_metrics failed:\n{traceback.format_exc()}", flush=True)

    print("\nAssembling combined HTML...", flush=True)
    html = _build_html(run_dir, run_name, plot_pairs, unified_pairs, extended_pairs, basin_pairs)
    html_path = results_dir / "analysis.html"
    html_path.write_text(html)
    print(f"  HTML saved: {html_path}", flush=True)

    print("Assembling combined TXT...", flush=True)
    txt = _build_txt(
        run_name,
        plot_pairs,
        unified_txt=unified_tmp / "dashboard.txt",
        extended_txt=extended_tmp / "extended_metrics.txt",
        basin_txt=basin_tmp / "dashboard.txt",
    )
    txt_path = results_dir / "analysis.txt"
    txt_path.write_text(txt)
    print(f"  TXT saved: {txt_path}", flush=True)

    print(f"\nDone: {run_name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-run combined analysis dashboard (plots + unified + basin).",
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--n-prune-levels", type=int, default=10)
    parser.add_argument("--relearn-steps", type=int, default=50)
    parser.add_argument("--frank-seeds", type=int, default=3)
    parser.add_argument("--xfrank-seeds", type=int, default=3)
    parser.add_argument("--subsample-n", type=int, default=256)
    parser.add_argument("--skip-surface", action="store_true")
    parser.add_argument("--skip-unified", action="store_true")
    parser.add_argument("--skip-basin", action="store_true")
    parser.add_argument("--skip-extended", action="store_true")
    args = parser.parse_args()

    run_analysis(
        run_dir=args.run_dir,
        n_seeds=args.n_seeds,
        n_prune_levels=args.n_prune_levels,
        relearn_steps=args.relearn_steps,
        frank_seeds=args.frank_seeds,
        xfrank_seeds=args.xfrank_seeds,
        subsample_n=args.subsample_n,
        skip_surface=args.skip_surface,
        skip_unified=args.skip_unified,
        skip_basin=args.skip_basin,
        skip_extended=args.skip_extended,
    )


if __name__ == "__main__":
    main()

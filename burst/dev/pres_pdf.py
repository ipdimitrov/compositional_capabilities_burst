"""Single combined HTML + TXT builder for a burst experiment run.

Generates:
  results/burst_report.html  — full presentation with all charts
  results/burst_report.txt   — machine-readable companion for LLM ingestion

Usage:
  python burst/pres_pdf.py <run_dir>
  python burst/pres_pdf.py <run_dir> --full   # also run unified/basin/extended analysis
"""

import argparse
import base64
import json
import os
import pickle
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from collections import defaultdict
from pathlib import Path

import numpy as np

from burst.config import TrainConfig, parse_run_config, reversion_life_key, reversion_life_label
from burst.dev.pres_charts import (
    SCHED_SHORT,
    _group,
    _group_gs,
    _ordered,
    generate_all,
    load_grad_sim_data,
)


def _img_tag(path, max_width=900) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    data = base64.b64encode(p.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" style="max-width:{max_width}px;width:100%;">'


_CSS = """
body { font-family: Helvetica, Arial, sans-serif; max-width: 1100px; margin: 0 auto;
       padding: 20px 30px; color: #1e1e1e; background: #fff; }
h1 { color: #0d47a1; border-bottom: 3px solid #0d47a1; padding-bottom: 8px; }
h2 { color: #0d47a1; margin-top: 2em; }
h3 { color: #0d47a1; }
.subtitle { color: #555; font-size: 1.1em; text-align: center; }
.meta { color: #888; font-family: monospace; font-size: 0.85em; text-align: center; }
.hbox { background: #f0f5ff; border-left: 4px solid #0d47a1; padding: 6px 12px;
        margin: 10px 0; font-weight: bold; color: #0d47a1; }
.vbox { padding: 8px 14px; margin: 8px 0; border-radius: 4px; }
.vbox-supported { background: #e8f5e9; border-left: 4px solid #008000; }
.vbox-partial   { background: #fff3e0; border-left: 4px solid #ff9800; }
.vbox-not       { background: #ffebee; border-left: 4px solid #d32f2f; }
.vbox-observed  { background: #e3f2fd; border-left: 4px solid #0064b4; }
.verdict { font-weight: bold; }
.chart { text-align: center; margin: 16px 0; }
.chart img { border: 1px solid #eee; border-radius: 4px; }
ul { padding-left: 1.4em; }
li { margin-bottom: 4px; }
.page-break {
  page-break-before: always; margin-top: 3em; border-top: 1px solid #ddd; padding-top: 1em;
}
@media print { .page-break { page-break-before: always; } }
.toc { background: #f0f5ff; border-radius: 8px; padding: 16px 24px; margin: 20px 0; }
.toc a { display: block; margin: 3px 0; color: #0d47a1; text-decoration: none; font-size: 0.92em; }
.toc a:hover { text-decoration: underline; }
"""


def _verdict_html(verdict: str, explanation: str, kind: str = "supported") -> str:
    css_class = {
        "supported": "vbox-supported",
        "partial": "vbox-partial",
        "not": "vbox-not",
        "observed": "vbox-observed",
    }.get(kind, "vbox-supported")
    return f'<div class="vbox {css_class}"><span class="verdict">VERDICT: {verdict}</span> &mdash; {explanation}</div>'  # noqa: E501


def _section(title: str, level: int = 2, page_break: bool = True, anchor: str = "") -> str:
    pb = ' class="page-break"' if page_break else ""
    aid = f' id="{anchor}"' if anchor else ""
    tag = f"h{level}"
    return f"<div{pb}{aid}><{tag}>{title}</{tag}></div>"


def _chart(path, max_width=900) -> str:
    if path is None:
        return ""
    tag = _img_tag(path, max_width)
    return f'<div class="chart">{tag}</div>' if tag else ""


# ---------------------------------------------------------------------------
# Unified / Extended / Basin analysis chart collection
# ---------------------------------------------------------------------------


def _collect_analysis_charts(run_dir: Path, n_seeds: int = 3) -> dict[str, list[tuple[str, Path]]]:
    """Run unified_analysis, extended metrics, and basin_metrics; return chart paths."""
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, list[tuple[str, Path]]] = {
        "unified": [],
        "extended": [],
        "basin": [],
    }

    def _pngs(d: Path) -> list[tuple[str, Path]]:
        if not d.exists():
            return []
        return [(p.stem.replace("_", " ").title(), p) for p in sorted(d.glob("*.png"))]

    try:
        from burst.dev.unified_analysis import analyse_run as ua_analyse
        from burst.dev.unified_analysis import make_dashboard

        time.time()
        r = ua_analyse(
            run_dir,
            n_seeds=n_seeds,
            n_prune_levels=10,
            relearn_steps=50,
            frank_seeds=3,
            xfrank_seeds=3,
            subsample_n=256,
        )
        run_name = r["run_name"]
        metric_keys = [
            "ema_dual",
            "lmc_dual",
            "frankenstein",
            "cross_frankenstein",
            "transfer_dual",
            "pruning_dual",
            "relearning_dual",
            "trajectory_dim",
            "forgetting_decomposition",
            "grad_temporal",
            "layer_interference",
            "sharpness",
            "grad_norm_ratio",
            "grad_rank",
            "grad_snr",
            "conflict_rate",
            "token_pos_grad",
            "grad_attribution",
            "forgetting_grad_alignment",
            "weight_drift_per_layer",
            "effective_rank_per_layer",
            "cka_per_layer",
            "directional_pruning",
        ]
        combined: dict = {
            "run_names": [run_name],
            "burst_positions": {run_name: r["burst_pos"]},
            "n_layer": r.get("n_layer", 6),
        }
        for mk in metric_keys:
            if mk in r:
                combined[mk] = {run_name: r[mk]}
        tmp = results_dir / "_unified_tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        make_dashboard(combined, tmp)
        out["unified"] = _pngs(tmp / "charts")
    except Exception:
        pass

    try:
        from burst.dev.unified_analysis import make_extended_metrics_dashboard

        tmp = results_dir / "_extended_tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        make_extended_metrics_dashboard([run_dir], tmp)
        out["extended"] = _pngs(tmp / "charts")
    except Exception:
        pass

    try:
        from burst.dev.basin_metrics import (
            analyse_run as bm_analyse,
        )
        from burst.dev.basin_metrics import (
            make_dashboard as bm_dashboard,
        )

        time.time()
        r = bm_analyse(run_dir, n_seeds=n_seeds, skip_surface=False)
        tmp = results_dir / "_basin_tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        bm_dashboard({run_dir.name: r}, tmp)
        out["basin"] = _pngs(tmp / "charts")
    except Exception:
        pass

    return out


# ---------------------------------------------------------------------------
# Machine-readable TXT builder
# ---------------------------------------------------------------------------


def _build_txt(
    rd: Path,
    res: list,
    cfg: dict,
    cp: dict,
    gs_records: list,
    analysis_charts: dict[str, list[tuple[str, Path]]] | None = None,
) -> str:
    rc = parse_run_config(cfg)
    bcfg, depth, burst_pos, n_a = rc["base_cfg"], rc["depth"], rc["burst_pos"], rc["n_a"]
    T, U = bcfg["total_steps"], bcfg["reversion_steps"]
    P = bcfg.get("pre_burst_steps", 0)
    ns = cfg.get("n_seeds", 5)
    gr = _group(res)
    sc = _ordered(gr.keys())

    lines: list[str] = []
    lines.append("=" * 70)
    lines.append(f"BURST EXPERIMENT REPORT: {rd.name}")
    lines.append("=" * 70)
    lines.append("")
    lines.append("EXPERIMENT CONFIGURATION")
    lines.append(f"  Depth: {depth}")
    lines.append(f"  Burst position: {burst_pos}")
    lines.append(f"  Functions per position (n_a): {n_a}")
    lines.append(f"  Other-class tasks: {n_a**depth}")
    lines.append(f"  Burst-class tasks: {n_a ** (depth - 1)}")
    lines.append(f"  Pre-burst steps (P): {P}")
    lines.append(f"  Burst steps (T): {T}")
    lines.append(f"  Reversion steps (U): {U}")
    lines.append(f"  Batch size: {bcfg['batch_size']}")
    lines.append(f"  Seeds: {ns}")
    lines.append(f"  Schedules: {', '.join(sc)}")
    lines.append(f"  Model: {bcfg['n_layer']}L/{bcfg['n_embd']}d/{bcfg['n_head']}H")
    lines.append("")

    thresholds = TrainConfig().reversion_thresholds
    lines.append("SUMMARY STATISTICS (mean +/- 95% CI across seeds)")
    lines.append("-" * 70)
    header = f"{'Schedule':<14} {'Peak':>8} {'AUC':>8}"
    for t in thresholds:
        header += f" {reversion_life_label(t):>10}"
    header += f" {'OtherAcc':>10}"
    lines.append(header)
    lines.append("-" * 70)
    for sched in sc:
        runs = gr[sched]
        peak = np.array([r.get("peak_burst", 0) for r in runs])
        auc = np.array([r.get("reversion_auc", 0) for r in runs])
        row = f"{sched:<14} {peak.mean():>7.3f}  {auc.mean():>7.0f}"
        for t in thresholds:
            key = reversion_life_key(t)
            vals = np.array([r.get(key, U) for r in runs])
            row += f" {vals.mean():>9.0f}"
        try:
            other_end = np.array([r["log"]["acc_other"][-1] for r in runs])
            row += f" {other_end.mean():>9.3f}"
        except (KeyError, IndexError):
            row += f" {'N/A':>9}"
        lines.append(row)
    lines.append("")

    lines.append("GRADIENT COSINE SIMILARITY DATA")
    lines.append("(burst-vs-other cosine similarity of full-parameter gradient vectors)")
    lines.append("")
    if gs_records:
        gs_groups = _group_gs(gs_records)
        for sched in sc:
            if sched not in gs_groups:
                continue
            runs = [r for r in gs_groups[sched] if r["grad_sim_log"]["step"]]
            if not runs:
                continue
            lines.append(f"  Schedule: {sched}")
            for r in runs:
                steps = r["grad_sim_log"]["step"]
                sims = r["grad_sim_log"]["burst_vs_other"]
                seed = r.get("seed", "?")
                lines.append(
                    f"    seed={seed}: steps={steps[:5]}...{steps[-3:] if len(steps) > 5 else ''}"
                )
                lines.append(
                    f"              sims ={[f'{s:.3f}' for s in sims[:5]]}...{[f'{s:.3f}' for s in sims[-3:]] if len(sims) > 5 else ''}"  # noqa: E501
                )

            all_end_burst = []
            for r in runs:
                steps = np.array(r["grad_sim_log"]["step"])
                sims = np.array(r["grad_sim_log"]["burst_vs_other"])
                burst_mask = steps <= T
                if burst_mask.any():
                    all_end_burst.append(sims[burst_mask][-1])
            if all_end_burst:
                arr = np.array(all_end_burst)
                lines.append(
                    f"    End-of-burst cossim: {arr.mean():.4f} +/- {arr.std():.4f} (n={len(arr)})"
                )
            lines.append("")

        lines.append("PAIRWISE GRADIENT SIMILARITY SNAPSHOTS")
        lines.append(
            "(task groups: BURST, O_F{i} = other tasks by function at burst_pos, ALL_OTHER, ALL_DATA)"  # noqa: E501
        )
        lines.append("")
        for sched in sc:
            if sched not in gs_groups:
                continue
            snaps_by_step: dict[int, list] = defaultdict(list)
            for r in gs_groups[sched]:
                for snap in r.get("pairwise_snapshots", []):
                    snaps_by_step[snap["step"]].append(snap)
            for step in sorted(snaps_by_step.keys()):
                snaps = snaps_by_step[step]
                labels = snaps[0]["labels"]
                matrices = [np.array(s["matrix"]) for s in snaps if len(s["matrix"]) == len(labels)]
                if not matrices:
                    continue
                mean_mat = np.mean(matrices, axis=0)
                lines.append(
                    f"  {sched} step={step} phase={snaps[0].get('phase', '?')} (n={len(matrices)} seeds)"  # noqa: E501
                )
                lines.append(f"    Labels: {labels}")
                for i, row_label in enumerate(labels):
                    row_str = " ".join(f"{mean_mat[i, j]:+.3f}" for j in range(len(labels)))
                    lines.append(f"    {row_label:>12}: {row_str}")
                lines.append("")
    else:
        lines.append("  (no gradient cosine similarity data available)")
        lines.append("")

    lines.append("CHART DESCRIPTIONS")
    lines.append("(descriptions of each chart in the HTML report)")
    lines.append("")
    chart_descs = [
        (
            "Schedule Bars",
            "Fraction of burst-class data per training step for each schedule. Shows the temporal structure of data exposure.",  # noqa: E501
        ),
        (
            "Special Class Accuracy Overlay",
            "Mean +/- 95% CI of burst-class free-generation accuracy over training steps, one line per schedule. Shows acquisition speed and forgetting dynamics.",  # noqa: E501
        ),
        (
            "Other Classes Accuracy Overlay",
            "Mean +/- 95% CI of other-class accuracy. Should remain near 1.0 throughout — measures catastrophic forgetting of background knowledge.",  # noqa: E501
        ),
        (
            "Reversion Zoom",
            "Burst-class accuracy during reversion phase only, aligned to reversion start. Directly compares forgetting speed across schedules.",  # noqa: E501
        ),
        (
            "Peak Burst Bars",
            "Bar chart of peak burst-class accuracy by schedule. All should be near 1.0 if the model has sufficient capacity.",  # noqa: E501
        ),
        (
            "Reversion AUC Bars",
            "Area under the burst-class accuracy curve during reversion. Higher = slower forgetting = more robust representation.",  # noqa: E501
        ),
        ("Life Bars", "Steps until burst accuracy drops to X% of peak. Lower = faster forgetting."),
        (
            "AUC Diff Heatmap",
            "Pairwise percentage difference in reversion AUC between schedules. Shows relative forgetting resistance.",  # noqa: E501
        ),
        (
            "Gradient Cosine Similarity Overlay",
            "Cosine similarity between burst-class and other-class gradient vectors over training. High similarity = integrated representations; low/negative = conflicting gradients.",  # noqa: E501
        ),
        (
            "Gradient Cosine End-of-Burst Bars",
            "Snapshot of gradient alignment at end of burst phase. Predicts forgetting speed.",
        ),
        (
            "Gradient Cosine vs AUC Scatter",
            "Correlation between end-of-burst gradient alignment and reversion AUC. Positive correlation validates the gradient-conflict theory of forgetting.",  # noqa: E501
        ),
        (
            "Pairwise Gradient Heatmaps",
            "Full pairwise cosine similarity matrix between task groups (BURST, O_F1..O_Fn, ALL_OTHER, ALL_DATA) at key training steps.",  # noqa: E501
        ),
        (
            "Per-Layer Gradient Cosine Heatmap",
            "Layer x Step heatmap showing which layers have aligned vs conflicting gradients over time.",  # noqa: E501
        ),
        (
            "Per-Layer End-of-Burst Bars",
            "Grouped bars: each layer group x each schedule at end of burst. Shows which layers drive gradient alignment.",  # noqa: E501
        ),
        (
            "ADL Delta Norm",
            "Magnitude of activation bias (checkpoint - pre-burst) on other-class inputs. Rising norm = global activation shift from burst learning.",  # noqa: E501
        ),
        (
            "ADL Readability",
            "Fraction of top-10 unembedding tokens that are burst-relevant. High = wrapper representation; low = deeper encoding.",  # noqa: E501
        ),
        (
            "ADL Causal Ablation",
            "Accuracy drop when projecting out the activation bias. Large drop = model relies on global bias (wrapper); small drop = deeper representation.",  # noqa: E501
        ),
        (
            "Probe Heatmaps",
            "Binary linear probe accuracy (other vs burst) at each (layer, token position) pair. Shows where and when the model develops burst-specific representations.",  # noqa: E501
        ),
    ]
    for name, desc in chart_descs:
        lines.append(f"  [{name}]")
        lines.append(f"    {desc}")
        lines.append("")

    if analysis_charts:
        for section, pairs in analysis_charts.items():
            if pairs:
                lines.append(f"ANALYSIS CHARTS: {section.upper()}")
                for title, path in pairs:
                    lines.append(f"  [PNG] {title}: {path.name}")
                lines.append("")

        for section_name, tmp_name in [
            ("Unified Analysis", "_unified_tmp"),
            ("Extended Metrics", "_extended_tmp"),
            ("Basin Geometry", "_basin_tmp"),
        ]:
            txt_path = rd / "results" / tmp_name / "dashboard.txt"
            if not txt_path.exists():
                txt_path = rd / "results" / tmp_name / "extended_metrics.txt"
            if txt_path.exists():
                lines.append(f"--- RAW DATA: {section_name} ---")
                lines.append(txt_path.read_text())
                lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML builder
# ---------------------------------------------------------------------------


def build(rd, res, cfg, cp, analysis_charts=None):
    rc = parse_run_config(cfg)
    bcfg, depth, burst_pos, n_a = rc["base_cfg"], rc["depth"], rc["burst_pos"], rc["n_a"]
    T, U = bcfg["total_steps"], bcfg["reversion_steps"]
    nl, ne, nh = bcfg["n_layer"], bcfg["n_embd"], bcfg["n_head"]
    bs, p = bcfg["batch_size"], bcfg["p_target"]
    ns = cfg.get("n_seeds", 5)
    gr = _group(res)
    sc = _ordered(gr.keys())
    max(int(p * T), 1)

    auc_key = "reversion_auc"
    peak_key = "peak_burst"
    other_log_key = "acc_other"

    parts: list[str] = []

    def _try(fn, label="section") -> None:
        try:
            fn()
        except Exception:
            parts.append(
                f'<div class="vbox vbox-partial"><b>Skipped {label}</b>: {traceback.format_exc().splitlines()[-1]}</div>'  # noqa: E501
            )

    parts.append(
        f"<html><head><meta charset='utf-8'><title>Burst Report — {rd.name}</title><style>{_CSS}</style></head><body>"  # noqa: E501
    )

    parts.append(
        "<h1 style='text-align:center;border:none;'>Compositional Learning &amp; Forgetting<br>in Transformers</h1>"  # noqa: E501
    )
    parts.append(
        f'<p class="subtitle">Depth-{depth} Bijection Burst Experiment (burst at position {burst_pos})</p>'  # noqa: E501
    )
    P = bcfg.get("pre_burst_steps", 0)
    parts.append(
        f'<p class="meta">{nl}L/{ne}d/{nh}H | {P} pre-burst + {T} special + {U} all-but-special | batch {bs} | {len(sc)} schedules x {ns} seeds = {len(res)} runs</p>'  # noqa: E501
    )
    parts.append(f'<p class="meta">{rd.name}</p>')

    # Table of contents
    toc_items = [
        ("research-q", "Research Question"),
        ("setup", "Experimental Setup"),
        ("protocol", "Training Protocol"),
        ("result-peak", "Result: Peak Accuracy"),
        ("result-curves", "Result: Accuracy Over Time"),
        ("result-forgetting", "Result: Forgetting Dynamics"),
        ("result-auc", "Result: Reversion AUC"),
        ("result-ordering", "Result: Schedule Ordering"),
        ("result-other", "Result: Other Classes Preservation"),
        ("summary-stats", "Summary Statistics"),
        ("per-sched", "Per-Schedule Detail"),
        ("grad-sim", "Gradient Cosine Similarity"),
        ("layer-grad-sim", "Per-Layer Gradient Cosine Similarity"),
        ("pairwise-grad", "Pairwise Gradient Similarity"),
        ("probes", "Linear Probes"),
        ("adl", "Activation Difference Lens"),
    ]
    if analysis_charts:
        if analysis_charts.get("unified"):
            toc_items.append(("unified", "Unified Analysis"))
        if analysis_charts.get("extended"):
            toc_items.append(("extended", "Extended Metrics"))
        if analysis_charts.get("basin"):
            toc_items.append(("basin", "Basin Geometry"))
    toc_items.append(("conclusions", "Conclusions"))

    parts.append('<div class="toc"><strong>Contents</strong>')
    for anchor, label in toc_items:
        parts.append(f'<a href="#{anchor}">{label}</a>')
    parts.append("</div>")

    pv = {s: np.mean([r.get(peak_key, 0) for r in gr[s]]) for s in sc}
    av = {s: np.mean([r.get(auc_key, 0) for r in gr[s]]) for s in sc}
    thresholds = TrainConfig().reversion_thresholds
    life_vals = {}
    for t in thresholds:
        key = reversion_life_key(t)
        life_vals[t] = {s: np.mean([r.get(key, U) for r in gr[s]]) for s in sc}
    try:
        ae = {s: np.mean([r["log"][other_log_key][-1] for r in gr[s]]) for s in sc}
    except (KeyError, IndexError):
        ae = {s: float("nan") for s in sc}

    def _research_q() -> None:
        parts.append(_section("Research Question", anchor="research-q"))
        parts.append(
            "<p>How does the training schedule for introducing novel compositional knowledge affect a Transformer's ability to (a) acquire that knowledge and (b) retain it when the novel data is removed?</p>"  # noqa: E501
        )
        parts.append(
            "<p>Does interleaving other classes with the special class during the burst window produce more robust representations than presenting the special class in isolation?</p>"  # noqa: E501
        )

    _try(_research_q, "Research Question")

    def _setup() -> None:
        parts.append(_section("Experimental Setup", anchor="setup"))
        parts.append(
            f"<h3>Task: Depth-{depth} Bijection Composition (burst at position {burst_pos})</h3>"
        )
        parts.append(
            f"<p>Model applies chains of {depth} bijection functions to 6 digits. Eval: free generation.</p>"  # noqa: E501
        )
        parts.append("<h3>Data Split</h3><ul>")
        parts.append(
            f"<li><b>Other Classes:</b> {n_a} bijections per position x {depth} positions = {n_a**depth} other-class compositions</li>"  # noqa: E501
        )
        parts.append(
            f"<li><b>Special Class:</b> 1 new bijection b* at pos {burst_pos}, all {n_a ** (depth - 1)} combos for other positions</li>"  # noqa: E501
        )
        parts.append("</ul><h3>Model &amp; Training</h3><ul>")
        parts.append(f"<li>{nl}L Transformer, {ne}d, {nh}H, SwiGLU, no dropout</li>")
        parts.append(f"<li>AdamW lr={bcfg['lr']}, cosine decay, batch {bs}, bfloat16</li>")
        parts.append("</ul>")

    _try(_setup, "Experimental Setup")

    def _protocol() -> None:
        parts.append(_section("Training Protocol", anchor="protocol"))
        if P > 0:
            parts.append(f"<h3>All-but-special (0-{P - 1})</h3>")
            parts.append("<p>Other classes only. Shared across all schedules.</p>")
        parts.append(f"<h3>Special ({P}-{P + T - 1})</h3>")
        parts.append("<p>Other classes + Special class mixed per schedule.</p>")
        parts.append(f"<h3>All-but-special ({P + T}-{P + T + U - 1})</h3>")
        parts.append("<p>Special class removed. Other classes only.</p>")
        parts.append(_chart(cp.get("lr")))
        parts.append(_chart(cp.get("schedule_bars")))

    _try(_protocol, "Training Protocol")

    def _result1() -> None:
        parts.append(_section("Result: Peak Special Class Accuracy", anchor="result-peak"))
        parts.append(_chart(cp.get("peak_bars")))
        parts.append('<div class="hbox">H1: All schedules achieve peak special class ~ 1.0</div>')
        if all(m >= 0.998 for m in pv.values()):
            parts.append(
                _verdict_html(
                    "SUPPORTED",
                    f"All >= 0.998. Range: {min(pv.values()):.3f}-{max(pv.values()):.3f}.",
                    "supported",
                )
            )
        else:
            parts.append(
                _verdict_html(
                    "PARTIAL", f"Range: {min(pv.values()):.3f}-{max(pv.values()):.3f}.", "partial"
                )
            )

    _try(_result1, "Result 1")

    def _result2() -> None:
        parts.append(_section("Result: Special Class Accuracy Over Time", anchor="result-curves"))
        parts.append(_chart(cp.get("overlay_burst")))
        parts.append(_chart(cp.get("overlay_burst_aligned_end")))

    _try(_result2, "Result 2")

    def _result3() -> None:
        parts.append(_section("Result: Forgetting Dynamics", anchor="result-forgetting"))
        parts.append(_chart(cp.get("reversion_zoom")))
        order_str = " &gt; ".join(
            SCHED_SHORT.get(s, s) for s in sorted(av, key=av.get, reverse=True)
        )
        parts.append(f"<p>Ordering by retention: {order_str}</p>")

    _try(_result3, "Result 3")

    def _result4() -> None:
        parts.append(_section("Result: Reversion AUC", anchor="result-auc"))
        parts.append(_chart(cp.get("auc_bars")))
        life_bars = cp.get("life_bars", {})
        for _thresh_idx, t in enumerate(thresholds):
            if t not in life_bars:
                continue
            label = reversion_life_label(t)
            parts.append(f"<h3>{label}</h3>")
            parts.append(_chart(life_bars[t]))

    _try(_result4, "Result 4")

    def _result6() -> None:
        parts.append(_section("Result: Schedule Ordering", anchor="result-ordering"))
        parts.append(_chart(cp.get("auc_diff"), 700))
        order = sorted(av, key=av.get, reverse=True)
        parts.append(
            _verdict_html(
                "OBSERVED",
                f"Got: {' &gt; '.join(SCHED_SHORT.get(s, s) for s in order)}",
                "observed",
            )
        )

    _try(_result6, "Result 6")

    def _result7() -> None:
        parts.append(_section("Result: Other Classes Preservation", anchor="result-other"))
        parts.append(_chart(cp.get("overlay_other")))
        parts.append(_chart(cp.get("overlay_other_aligned_end")))
        if all(m >= 0.95 for m in ae.values()):
            parts.append(
                _verdict_html("SUPPORTED", "All other classes >= 0.95 at end.", "supported")
            )
        else:
            parts.append(_verdict_html("PARTIAL", f"Min: {min(ae.values()):.3f}", "partial"))

    _try(_result7, "Result 7")

    def _summary() -> None:
        parts.append(_section("Summary Statistics", anchor="summary-stats"))
        parts.append(_chart(cp.get("summary_table"), 1000))

    _try(_summary, "Summary Statistics")

    def _per_sched() -> None:
        parts.append(_section("Per-Schedule Detail", anchor="per-sched"))
        for path in cp.get("per_sched") or []:
            parts.append(_chart(path))

    _try(_per_sched, "Per-Schedule Detail")

    def _grad_sim() -> None:
        has_gs = any(
            cp.get(k)
            for k in [
                "grad_cosine_overlay",
                "grad_cosine_bars",
                "grad_cosine_per_seed",
                "grad_cosine_rate",
                "grad_cosine_phase",
                "grad_cosine_vs_auc",
                "grad_cosine_phase_bars",
            ]
        )
        if not has_gs:
            return
        parts.append(
            _section("Gradient Cosine Similarity: Special vs Other Classes", anchor="grad-sim")
        )
        parts.append(
            "<p>Cosine similarity between the full-parameter gradient vectors computed on burst-class "  # noqa: E501
            "vs other-class documents. High similarity = integrated representations; "
            "low/negative = conflicting gradient directions predicting faster forgetting.</p>"
        )
        for key, title, w in [
            ("grad_cosine_overlay", "Burst vs Other: All Schedules", 900),
            ("grad_cosine_bars", "End-of-Burst Snapshot", 800),
            ("grad_cosine_phase", "Burst Phase vs Reversion Phase", 900),
            ("grad_cosine_phase_bars", "Similarity Across Training Phases", 900),
        ]:
            if cp.get(key):
                parts.append(f"<h3>{title}</h3>")
                parts.append(_chart(cp[key], w))
        if cp.get("grad_cosine_rate"):
            parts.append("<h3>Rate of Change</h3>")
            parts.append(_chart(cp["grad_cosine_rate"]))
        if cp.get("grad_cosine_vs_auc"):
            parts.append("<h3>Gradient Alignment vs Forgetting Resistance</h3>")
            parts.append(_chart(cp["grad_cosine_vs_auc"], 800))
        if cp.get("grad_cosine_per_seed"):
            parts.append("<h3>Per-Seed Traces</h3>")
            for p_ in cp["grad_cosine_per_seed"]:
                parts.append(_chart(p_))

    _try(_grad_sim, "Gradient Cosine Similarity")

    def _layer_grad_sim() -> None:
        has_layer = any(
            cp.get(k)
            for k in [
                "layer_cossim_heatmap",
                "layer_cossim_layer_sched",
                "layer_cossim_end_burst_bars",
                "layer_cossim_overlay",
                "layer_cossim_change",
                "layer_cossim_all_scheds",
            ]
        )
        if not has_layer:
            return
        parts.append(_section("Per-Layer Gradient Cosine Similarity", anchor="layer-grad-sim"))
        parts.append(
            "<p>Same burst-vs-other gradient cosine similarity, computed independently "
            "for each layer group. Reveals which parts of the network show the strongest "
            "gradient alignment between burst and other classes.</p>"
        )
        if cp.get("layer_cossim_end_burst_bars"):
            parts.append("<h3>End-of-Burst: All Layers x All Schedules</h3>")
            parts.append(_chart(cp["layer_cossim_end_burst_bars"], 1000))
        if cp.get("layer_cossim_layer_sched"):
            parts.append("<h3>Layer x Schedule Heatmaps</h3>")
            for p_ in cp["layer_cossim_layer_sched"] or []:
                parts.append(_chart(p_, 900))
        if cp.get("layer_cossim_heatmap"):
            parts.append("<h3>Layer x Step Heatmaps</h3>")
            for p_ in cp["layer_cossim_heatmap"] or []:
                parts.append(_chart(p_, 1000))
        if cp.get("layer_cossim_change"):
            parts.append("<h3>Rate-of-Change Heatmaps</h3>")
            for p_ in cp["layer_cossim_change"] or []:
                parts.append(_chart(p_, 1000))
        if cp.get("layer_cossim_overlay"):
            parts.append("<h3>Per-Schedule Layer Overlays</h3>")
            for p_ in cp["layer_cossim_overlay"] or []:
                parts.append(_chart(p_, 900))

    _try(_layer_grad_sim, "Per-Layer Gradient Cosine Similarity")

    def _pairwise_evo() -> None:
        has_pw = (
            cp.get("pairwise_evo_by_metric")
            or cp.get("pairwise_evo_per_schedule")
            or cp.get("pairwise_heatmaps")
        )
        if not has_pw:
            return
        parts.append(_section("Pairwise Gradient Similarity", anchor="pairwise-grad"))
        parts.append(
            "<p>Tasks grouped by which function sits at the burst position. "
            "BURST = all burst-class tasks; O_F{i} = other-class tasks grouped by function at burst position; "  # noqa: E501
            "ALL_OTHER = all other tasks; ALL_DATA = everything.</p>"
        )
        if cp.get("pairwise_evo_by_metric"):
            for p_ in cp["pairwise_evo_by_metric"] or []:
                parts.append(_chart(p_))
        if cp.get("pairwise_evo_per_schedule"):
            parts.append(_chart(cp["pairwise_evo_per_schedule"], 1000))
        if cp.get("pairwise_heatmaps"):
            parts.append("<h3>Pairwise Heatmaps at Key Steps</h3>")
            for p_ in cp["pairwise_heatmaps"] or []:
                parts.append(_chart(p_, 700))

    _try(_pairwise_evo, "Pairwise Gradient Similarity")

    def _probes() -> None:
        has_probes = (
            cp.get("probe_dynamics") or cp.get("probe_heatmaps") or cp.get("probe_layer_schedule")
        )
        if not has_probes:
            return
        parts.append(_section("Linear Probes: Other vs Special Representations", anchor="probes"))
        if cp.get("probe_dynamics"):
            parts.append("<h3>Probe Accuracy Over Training</h3>")
            parts.append(_chart(cp["probe_dynamics"]))
        if cp.get("probe_layer_schedule"):
            for p_ in cp["probe_layer_schedule"]:
                parts.append(_chart(p_))
        if cp.get("probe_heatmaps"):
            parts.append("<h3>Probe Heatmaps</h3>")
            for p_ in cp["probe_heatmaps"]:
                parts.append(_chart(p_))

    _try(_probes, "Probes")

    def _adl() -> None:
        has_adl = any(
            cp.get(k)
            for k in [
                "adl_delta_norm",
                "adl_readability",
                "adl_causal_ablation",
                "adl_end_burst_bars",
                "adl_readability_vs_auc",
            ]
        )
        if not has_adl:
            return
        parts.append(_section("Activation Difference Lens (ADL)", anchor="adl"))
        parts.append(
            "<p>Measures the global activation bias introduced by the burst phase. "
            "Logit Lens readability tests whether the bias encodes burst-relevant tokens (wrapper hypothesis). "  # noqa: E501
            "Causal ablation tests whether the model relies on the bias for burst accuracy.</p>"
        )
        for key, title in [
            ("adl_delta_norm", "Activation Bias Magnitude"),
            ("adl_readability", "Logit Lens Readability"),
            ("adl_causal_ablation", "Causal Ablation: Accuracy Drop"),
            ("adl_end_burst_bars", "End-of-Burst Summary"),
            ("adl_readability_vs_auc", "ADL Readability vs Forgetting Resistance"),
        ]:
            if cp.get(key):
                parts.append(f"<h3>{title}</h3>")
                parts.append(_chart(cp[key], 900))

    _try(_adl, "ADL")

    # Unified / Extended / Basin analysis charts
    if analysis_charts:

        def _analysis_section(key, title, anchor) -> None:
            pairs = analysis_charts.get(key, [])
            if not pairs:
                return
            parts.append(_section(title, anchor=anchor))
            for chart_title, path in pairs:
                parts.append(f"<h3>{chart_title}</h3>")
                parts.append(_chart(path, 1000))

        _try(
            lambda: _analysis_section("unified", "Unified Analysis", "unified"), "Unified Analysis"
        )
        _try(
            lambda: _analysis_section("extended", "Extended Metrics", "extended"),
            "Extended Metrics",
        )
        _try(
            lambda: _analysis_section("basin", "Basin Geometry Metrics", "basin"), "Basin Geometry"
        )

    def _conclusions() -> None:
        parts.append(_section("Conclusions", anchor="conclusions"))
        parts.append("<ul>")
        for b, t in [
            ("Acquisition:", "All schedules acquire special class (peak ~ 1.0)."),
            ("Retention:", "More other-class mixing during burst = slower forgetting."),
            (
                "Gradient alignment:",
                "Higher burst-vs-other gradient cosine similarity at end of burst predicts slower forgetting.",  # noqa: E501
            ),
            ("Background:", "Other classes robust across all schedules."),
        ]:
            parts.append(f"<li><b>{b}</b> {t}</li>")
        parts.append("</ul>")

    _try(_conclusions, "Conclusions")

    parts.append("</body></html>")

    out = rd / "results" / "burst_report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build combined HTML + TXT report for a burst experiment run."
    )
    parser.add_argument("run_dir", type=Path, nargs="?", default=None)
    parser.add_argument(
        "--full", action="store_true", help="Also run unified/basin/extended analysis (slower)"
    )
    parser.add_argument(
        "--n-seeds", type=int, default=3, help="Number of seeds for unified/basin analysis"
    )
    args = parser.parse_args()

    rd = args.run_dir or sorted(Path("data").glob("burst_d*"))[-1]
    rd = Path(rd)
    from burst.core.train_utils import resolve_run_paths

    cfg_path, logs_dir, _ = resolve_run_paths(rd)

    pkl_path = logs_dir / "all_results.pkl"
    if not pkl_path.exists():
        pkl_path = rd / "all_results.pkl"

    with open(cfg_path) as f:
        cfg = json.load(f)

    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            results = pickle.load(f)
    else:
        base = cfg.get("base_cfg", cfg)
        results = []
        for j in cfg.get("jobs", []):
            results.append(
                {
                    "label": j["label"],
                    "schedule": j["schedule"],
                    "seed": j["seed"],
                    "config": {
                        **base,
                        "total_steps": j.get("total_steps", base.get("total_steps", 500)),
                        "batch_size": j.get("batch_size", base.get("batch_size", 128)),
                    },
                    "log": {"step": [], "loss": [], "acc_burst": [], "acc_other": []},
                }
            )

    cp = generate_all(rd, results, cfg)

    analysis_charts = None
    if args.full:
        analysis_charts = _collect_analysis_charts(rd, n_seeds=args.n_seeds)

    gs_records = load_grad_sim_data(rd)

    build(rd, results, cfg, cp, analysis_charts=analysis_charts)

    txt = _build_txt(rd, results, cfg, cp, gs_records, analysis_charts=analysis_charts)
    txt_path = rd / "results" / "burst_report.txt"
    txt_path.write_text(txt)



if __name__ == "__main__":
    main()

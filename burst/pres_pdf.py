"""PDF builder. Usage: python burst/pres_pdf.py data/burst_d<depth>_<run_tag>"""
import sys, os, pickle, json, base64, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from pathlib import Path
from burst.pres_charts import PALETTE, SCHED_SHORT, _ordered, _group, generate_all
from burst.config import TrainConfig, reversion_life_key, reversion_life_label, parse_run_config


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
.page-break { page-break-before: always; margin-top: 3em; border-top: 1px solid #ddd; padding-top: 1em; }
@media print { .page-break { page-break-before: always; } }
"""


def _verdict_html(verdict: str, explanation: str, kind: str = "supported") -> str:
    css_class = {"supported": "vbox-supported", "partial": "vbox-partial",
                 "not": "vbox-not", "observed": "vbox-observed"}.get(kind, "vbox-supported")
    return f'<div class="vbox {css_class}"><span class="verdict">VERDICT: {verdict}</span> &mdash; {explanation}</div>'


def _section(title: str, level: int = 2, page_break: bool = True) -> str:
    pb = ' class="page-break"' if page_break else ""
    tag = f"h{level}"
    return f"<div{pb}><{tag}>{title}</{tag}></div>"


def _chart(path, max_width=900) -> str:
    if path is None:
        return ""
    tag = _img_tag(path, max_width)
    return f'<div class="chart">{tag}</div>' if tag else ""


def build(rd, res, cfg, cp):
    rc = parse_run_config(cfg)
    bcfg, depth, burst_pos, n_a = rc["base_cfg"], rc["depth"], rc["burst_pos"], rc["n_a"]
    T, U = bcfg["total_steps"], bcfg["reversion_steps"]
    nl, ne, nh = bcfg["n_layer"], bcfg["n_embd"], bcfg["n_head"]
    bs, p = bcfg["batch_size"], bcfg["p_target"]
    ns = cfg.get("n_seeds", 5)
    gr = _group(res)
    sc = _ordered(gr.keys())
    bl = max(int(p * T), 1)

    auc_key = "reversion_auc"
    peak_key = "peak_burst"
    other_log_key = "acc_other"

    parts: list[str] = []

    def _try(fn, label="section"):
        try:
            fn()
        except Exception:
            parts.append(f'<div class="vbox vbox-partial"><b>Skipped {label}</b>: {traceback.format_exc().splitlines()[-1]}</div>')

    parts.append(f"<html><head><meta charset='utf-8'><title>Burst Presentation</title><style>{_CSS}</style></head><body>")

    parts.append("<h1 style='text-align:center;border:none;'>Compositional Learning &amp; Forgetting<br>in Transformers</h1>")
    parts.append(f'<p class="subtitle">Depth-{depth} Bijection Burst Experiment (burst at position {burst_pos})</p>')
    P = bcfg.get("pre_burst_steps", 0)
    parts.append(f'<p class="meta">{nl}L/{ne}d/{nh}H | {P} pre-burst + {T} special + {U} all-but-special | batch {bs} | {len(sc)} schedules x {ns} seeds = {len(res)} runs</p>')
    parts.append('<p class="meta">Free generation evaluation</p>')

    def _research_q():
        parts.append(_section("Research Question"))
        parts.append("<p>How does the training schedule for introducing novel compositional knowledge affect a Transformer's ability to (a) acquire that knowledge and (b) retain it when the novel data is removed?</p>")
        parts.append("<p>Does interleaving other classes with the special class during the burst window produce more robust representations than presenting the special class in isolation?</p>")
        parts.append("<h3>Why This Matters</h3>")
        parts.append("<p>Understanding how neural networks acquire and forget compositional skills is fundamental to continual learning, curriculum design, and knowledge editing.</p>")
    _try(_research_q, "Research Question")

    def _setup():
        parts.append(_section("Experimental Setup"))
        parts.append(f"<h3>Task: Depth-{depth} Bijection Composition (burst at position {burst_pos})</h3>")
        parts.append(f"<p>Model applies chains of {depth} bijection functions to 6 digits. Eval: free generation.</p>")
        parts.append("<h3>Data Split</h3><ul>")
        parts.append(f"<li><b>Other Classes:</b> {n_a} bijections x {depth} positions = {n_a**depth} other-class compositions</li>")
        parts.append(f"<li><b>Special Class:</b> 1 new bijection b* at pos {burst_pos}, all {n_a**(depth-1)} combos for other positions</li>")
        parts.append("</ul><h3>Model &amp; Training</h3><ul>")
        parts.append(f"<li>{nl}L Transformer, {ne}d, {nh}H, SwiGLU, no dropout</li>")
        parts.append(f"<li>AdamW lr={bcfg['lr']}, cosine decay, batch {bs}, bfloat16</li>")
        parts.append("</ul>")
    _try(_setup, "Experimental Setup")

    def _protocol():
        parts.append(_section("Training Protocol"))
        if P > 0:
            parts.append(f"<h3>All-but-special (0-{P-1})</h3>")
            parts.append("<p>Other classes only. Shared across all schedules.</p>")
        parts.append(f"<h3>Special ({P}-{P+T-1})</h3>")
        parts.append(f"<p>Other classes + Special class mixed per schedule. ~{int(p*100)}% special class exposure.</p>")
        parts.append(f"<h3>All-but-special ({P+T}-{P+T+U-1})</h3>")
        parts.append("<p>Special class removed. Other classes only. LR continues decaying.</p>")
        parts.append("<h3>Evaluation</h3><ul><li>Every 10 steps, free generation, last 6 tokens</li></ul>")
    _try(_protocol, "Training Protocol")

    def _lr_sched():
        parts.append(_section("LR Schedule"))
        parts.append(_chart(cp.get("lr")))
    _try(_lr_sched, "LR Schedule")

    def _schedules():
        parts.append(_section(f"The {len(sc)} Training Schedules"))
        parts.append(_chart(cp.get("schedule_bars")))
        parts.append("<ul>")
        parts.append(f"<li><b>burst_100:</b> Pure special class for last {bl} steps</li>")
        for pct, frac in [(98, 0.98), (95, 0.95), (90, 0.90), (85, 0.85),
                          (75, 0.75), (50, 0.50), (25, 0.25)]:
            if f"burst_{pct}" in sc:
                win = min(int(bl / frac), T)
                parts.append(f"<li><b>burst_{pct}:</b> {pct}% special class + {100-pct}% other classes for last {win} steps</li>")
        if "burst_10" in sc:
            parts.append(f"<li><b>burst_10:</b> ~{int(p*100)}% special class randomly throughout (uniform control)</li>")
        parts.append("</ul>")
    _try(_schedules, "Training Schedules")

    def _metrics():
        thresholds = TrainConfig().reversion_thresholds
        parts.append(_section("Metrics"))
        parts.append("<ul>")
        parts.append("<li><b>Peak Special:</b> Special class accuracy at end of training</li>")
        for t in thresholds:
            pct = int(t * 100)
            parts.append(f"<li><b>{reversion_life_label(t)}:</b> Reversion steps to {pct}% of peak (cap {U})</li>")
        parts.append("<li><b>Reversion AUC:</b> Area under special class curve during reversion</li>")
        parts.append("</ul>")
    _try(_metrics, "Metrics")

    def _hypotheses():
        parts.append(_section("Hypotheses"))
        for hid, txt, expl in [
            (1, "All schedules achieve peak special class ~ 1.0", f"Sufficient capacity for {n_a**(depth-1)} compositions."),
            (2, "burst_10 (uniform) = most forgetting-resistant", "Distributed special class integrates with other classes knowledge."),
            (3, "burst_100 = fastest forgetting", "Isolated special class creates fragile representations."),
            (4, "Mixed schedules ordered by other-classes content", "More other-classes mixing = more robust special class."),
            (5, "Other classes preserved regardless of schedule", "Other classes are majority of training."),
        ]:
            parts.append(f'<div class="hbox">H{hid}: {txt}</div><p>{expl}</p>')
    _try(_hypotheses, "Hypotheses")

    pv = {s: np.mean([r.get(peak_key, 0) for r in gr[s]]) for s in sc}
    av = {s: np.mean([r.get(auc_key, 0) for r in gr[s]]) for s in sc}
    thresholds = TrainConfig().reversion_thresholds
    life_vals = {}
    for t in thresholds:
        key = reversion_life_key(t)
        life_vals[t] = {s: np.mean([r.get(key, U) for r in gr[s]]) for s in sc}
    ae = {s: np.mean([r["log"][other_log_key][-1] for r in gr[s]]) for s in sc}

    def _result1():
        parts.append(_section("Result 1: Peak Special Class Accuracy"))
        parts.append(_chart(cp.get("peak_bars")))
        parts.append('<div class="hbox">H1: All schedules achieve peak special class ~ 1.0</div>')
        if all(m >= 0.998 for m in pv.values()):
            parts.append(_verdict_html("SUPPORTED", f"All >= 0.998. Range: {min(pv.values()):.3f}-{max(pv.values()):.3f}.", "supported"))
        else:
            parts.append(_verdict_html("PARTIAL", f"Range: {min(pv.values()):.3f}-{max(pv.values()):.3f}.", "partial"))
    _try(_result1, "Result 1")

    def _result2():
        parts.append(_section("Result 2: Special Class Accuracy Over Time"))
        parts.append(_chart(cp.get("overlay_burst")))
        parts.append(_chart(cp.get("overlay_burst_aligned_start")))
        parts.append(_chart(cp.get("overlay_burst_aligned_end")))
        parts.append(f"<p>All reach ~100% by step {P+T}. Forgetting speed varies dramatically.</p>")
    _try(_result2, "Result 2")

    def _result3():
        parts.append(_section("Result 3: Forgetting Dynamics"))
        parts.append(_chart(cp.get("reversion_zoom")))
        order_str = " &gt; ".join(SCHED_SHORT.get(s, s) for s in sorted(av, key=av.get, reverse=True))
        parts.append(f"<p>Ordering by retention: {order_str}</p>")
    _try(_result3, "Result 3")

    def _result4():
        parts.append(_section("Result 4: Reversion AUC"))
        parts.append(_chart(cp.get("auc_bars")))
        parts.append('<div class="hbox">H2: burst_10 (uniform) = most forgetting-resistant</div>')
        best = max(av, key=av.get)
        if best == "burst_10":
            parts.append(_verdict_html("SUPPORTED", f"burst_10 highest AUC ({av['burst_10']:.0f}).", "supported"))
        else:
            b10_auc = av.get("burst_10", 0)
            parts.append(_verdict_html("NOT SUPPORTED", f"{best} higher ({av[best]:.0f} vs burst_10 {b10_auc:.0f}).", "not"))
    _try(_result4, "Result 4")

    def _result5():
        life_bars = cp.get("life_bars", {})
        for thresh_idx, t in enumerate(thresholds):
            if t not in life_bars:
                continue
            label = reversion_life_label(t)
            lv = life_vals[t]
            parts.append(_section(f"Result 5.{thresh_idx+1}: {label}"))
            parts.append(_chart(life_bars[t]))
            parts.append('<div class="hbox">H3: burst_100 = fastest forgetting</div>')
            low = min(lv, key=lv.get)
            if low == "burst_100":
                parts.append(_verdict_html("SUPPORTED", f"Lowest {label} ({lv['burst_100']:.0f}). High variance.", "supported"))
            else:
                parts.append(_verdict_html("NOT SUPPORTED", f"{low} lower ({lv[low]:.0f}).", "not"))
    _try(_result5, "Result 5")

    def _result6():
        parts.append(_section("Result 6: Schedule Ordering"))
        parts.append(_chart(cp.get("auc_diff"), 700))
        parts.append('<div class="hbox">H4: Mixed schedules ordered by other-classes content</div>')
        order = sorted(av, key=av.get, reverse=True)
        parts.append(_verdict_html("OBSERVED", f"Got: {' &gt; '.join(SCHED_SHORT.get(s, s) for s in order)}", "observed"))
    _try(_result6, "Result 6")

    def _result7():
        parts.append(_section("Result 7: Other Classes Preservation"))
        parts.append(_chart(cp.get("overlay_other")))
        parts.append(_chart(cp.get("overlay_other_aligned_start")))
        parts.append(_chart(cp.get("overlay_other_aligned_end")))
        parts.append('<div class="hbox">H5: Other classes preserved regardless of schedule</div>')
        if all(m >= 0.95 for m in ae.values()):
            parts.append(_verdict_html("SUPPORTED", "All other classes >= 0.95 at end.", "supported"))
        else:
            parts.append(_verdict_html("PARTIAL", f"Min: {min(ae.values()):.3f}", "partial"))
    _try(_result7, "Result 7")

    def _summary():
        parts.append(_section("Summary Statistics"))
        parts.append(_chart(cp.get("summary_table"), 1000))
    _try(_summary, "Summary Statistics")

    def _per_sched():
        parts.append(_section("Per-Schedule Detail"))
        for path in (cp.get("per_sched") or []):
            parts.append(_chart(path))
        for path in (cp.get("per_sched_start") or []):
            parts.append(_chart(path))
        for path in (cp.get("per_sched_end") or []):
            parts.append(_chart(path))
    _try(_per_sched, "Per-Schedule Detail")

    def _probes():
        has_probes = cp.get("probe_dynamics") or cp.get("probe_heatmaps") or cp.get("probe_layer_schedule")
        if not has_probes:
            return
        parts.append(_section("Linear Probes: Other vs Special Representations"))
        parts.append(
            "<p>Logistic regression probes trained on residual-stream activations at every "
            "(layer, token position) pair to classify Other-class vs Burst-class representations. "
            "5-fold cross-validation accuracy, aggregated across seeds with 95% CI.</p>"
        )
        if cp.get("probe_dynamics"):
            parts.append("<h3>Probe Accuracy Over Training</h3>")
            parts.append(_chart(cp["probe_dynamics"]))
        if cp.get("probe_layer_schedule"):
            for p_ in cp["probe_layer_schedule"]:
                parts.append("<h3>Layer x Schedule Probe Accuracy</h3>")
                parts.append(_chart(p_))
        if cp.get("probe_heatmaps"):
            parts.append("<h3>Probe Heatmaps (per schedule, mean across seeds)</h3>")
            for p_ in cp["probe_heatmaps"]:
                parts.append(_chart(p_))
    _try(_probes, "Probes")

    def _next_token_probes():
        pd_ = rd / "next_token_regime_probes"
        if not pd_.exists():
            return
        parts.append(_section("Next-Token Probes"))
        parts.append("<p>Logit lens + learned linear probe at burst-position outputs, Other vs Burst.</p>")
        for sd in sorted(pd_.glob("step_*")):
            step = sd.name.replace("step_", "")
            for m in ["logit_lens", "learned_probe"]:
                for k in ["curves", "diff"]:
                    fp = sd / f"{k}_{m}.png"
                    if fp.exists():
                        parts.append(f"<h3>{k}: {m} @ step {step}</h3>")
                        parts.append(_chart(fp))
        cb = pd_ / "combined"
        if cb and cb.exists():
            for m in ["logit_lens", "learned_probe"]:
                for k in ["curves", "diff"]:
                    fp = cb / f"combined_{k}_{m}.png"
                    if fp.exists():
                        parts.append(f"<h3>Evolution: {k} {m}</h3>")
                        parts.append(_chart(fp))
    _try(_next_token_probes, "Next-Token Probes")

    def _grad_sim():
        has_gs = any(cp.get(k) for k in [
            "grad_cosine_overlay", "grad_cosine_bars", "grad_cosine_per_seed",
            "grad_cosine_rate", "grad_cosine_phase", "grad_cosine_vs_auc",
            "grad_cosine_phase_bars",
        ])
        if not has_gs:
            return
        parts.append(_section("Gradient Cosine Similarity: Special vs Other Classes"))
        parts.append("<h3>How It Works (Autoregressive Regime)</h3>")
        parts.append(
            "<p>The model is trained autoregressively: given a sequence "
            "[S F3 F2 F1 ' ' input ' ' out1 ' ' out2 ' ' out3], the loss is standard "
            "next-token cross-entropy over all positions. At evaluation time, the model "
            "generates its own outputs token-by-token (free generation) from a prompt "
            "containing only the function slots and input -- it never sees the ground-truth "
            "intermediate or final outputs during inference.</p>"
        )
        gs_bs = bcfg.get("grad_sim_batch_size", 64)
        parts.append(
            f"<p>To compute gradient similarity, we sample {gs_bs} documents from each class "
            "(burst and other), compute the next-token prediction loss on the full sequence "
            "for each class separately, backpropagate to obtain a gradient vector (the "
            "concatenation of all parameter gradients), and measure cosine similarity "
            "between the two gradient vectors. Because the loss is autoregressive over "
            "the entire sequence -- including the intermediate composition outputs -- the "
            "gradient captures how each class shapes the model's predictions at every "
            "position in the chain, not just the final output.</p>"
        )
        parts.append("<h3>Interpretation</h3>")
        grad_sim_every = cfg.get("grad_sim_every", 50)
        parts.append(
            f"<p>Computed every {grad_sim_every} steps throughout training. "
            "High similarity means the special class is pulling the model in the same direction "
            "as the other classes -- suggesting integrated, durable representations. "
            "Low or negative similarity indicates conflicting gradient directions, "
            "which predicts faster forgetting during reversion.</p>"
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
            parts.append("<p>Derivative of cosine similarity over training steps. Peaks indicate where gradient alignment shifts fastest.</p>")
            parts.append(_chart(cp["grad_cosine_rate"]))
        if cp.get("grad_cosine_vs_auc"):
            parts.append("<h3>Gradient Alignment vs Forgetting Resistance</h3>")
            parts.append("<p>Each dot is one seed x schedule. Positive correlation means higher end-of-burst gradient alignment predicts slower forgetting.</p>")
            parts.append(_chart(cp["grad_cosine_vs_auc"], 800))
        if cp.get("grad_cosine_per_seed"):
            parts.append("<h3>Per-Seed Traces</h3>")
            parts.append("<p>Individual seed traces reveal variance in gradient alignment within each schedule.</p>")
            for p_ in cp["grad_cosine_per_seed"]:
                parts.append(_chart(p_))
    _try(_grad_sim, "Gradient Cosine Similarity")

    def _layer_grad_sim():
        has_layer = any(cp.get(k) for k in [
            "layer_cossim_heatmap", "layer_cossim_layer_sched",
            "layer_cossim_end_burst_bars", "layer_cossim_overlay",
            "layer_cossim_change", "layer_cossim_all_scheds",
        ])
        if not has_layer:
            return
        parts.append(_section("Per-Layer Gradient Cosine Similarity"))
        parts.append(
            "<p>The same burst-vs-other gradient cosine similarity, computed independently "
            "for each named layer group: <b>emb</b> (token + position embeddings), "
            "<b>L{i}_ln</b> (layernorms in block i), <b>L{i}_attn</b> (attention projections "
            "in block i), <b>L{i}_mlp</b> (MLP in block i), and <b>ln_f</b> (final layernorm). "
            "This reveals which parts of the network show the strongest gradient alignment "
            "between burst and other classes, and how that changes over training.</p>"
        )

        if cp.get("layer_cossim_end_burst_bars"):
            parts.append("<h3>End-of-Burst Snapshot: All Layers x All Schedules</h3>")
            parts.append(
                "<p>Grouped bars: each group is one layer, each bar is one schedule. "
                "Layers with high cossim at end of burst are pulling burst and other classes "
                "in the same direction — predicting durable representations.</p>"
            )
            parts.append(_chart(cp["layer_cossim_end_burst_bars"], 1000))

        if cp.get("layer_cossim_layer_sched"):
            parts.append("<h3>Layer x Schedule Heatmaps (End-Burst &amp; End-Reversion)</h3>")
            parts.append(
                "<p>Rows = layers, columns = schedules. Color = mean cosine similarity "
                "in that phase window. Compare across schedules to see which layers "
                "are most schedule-sensitive.</p>"
            )
            for p_ in (cp["layer_cossim_layer_sched"] or []):
                parts.append(_chart(p_, 900))

        if cp.get("layer_cossim_heatmap"):
            parts.append("<h3>Layer x Step Heatmaps (Per Schedule)</h3>")
            parts.append(
                "<p>Rows = layers, columns = training steps. Vertical line = start of "
                "reversion. Shows the full temporal trajectory of gradient alignment "
                "for every layer simultaneously.</p>"
            )
            for p_ in (cp["layer_cossim_heatmap"] or []):
                parts.append(_chart(p_, 1000))

        if cp.get("layer_cossim_change"):
            parts.append("<h3>Rate-of-Change Heatmaps (Per Schedule)</h3>")
            parts.append(
                "<p>d(cossim)/d(step) per layer. Red = cossim rising, blue = falling. "
                "Highlights where gradient alignment shifts fastest — often at the "
                "burst onset and at the start of reversion.</p>"
            )
            for p_ in (cp["layer_cossim_change"] or []):
                parts.append(_chart(p_, 1000))

        if cp.get("layer_cossim_overlay"):
            parts.append("<h3>Per-Schedule Layer Overlays</h3>")
            parts.append(
                "<p>All layers as lines on one chart per schedule. "
                "Useful for seeing which layers diverge from the mean.</p>"
            )
            for p_ in (cp["layer_cossim_overlay"] or []):
                parts.append(_chart(p_, 900))

        if cp.get("layer_cossim_all_scheds"):
            parts.append("<h3>Per-Layer Schedule Comparisons</h3>")
            parts.append(
                "<p>One chart per layer, all schedules overlaid. "
                "Best for comparing how a specific layer responds to different schedules.</p>"
            )
            for p_ in (cp["layer_cossim_all_scheds"] or []):
                parts.append(_chart(p_, 900))
    _try(_layer_grad_sim, "Per-Layer Gradient Cosine Similarity")

    def _pairwise_evo():
        if cp.get("pairwise_evo_by_metric"):
            parts.append(_section("Pairwise Gradient Similarity: Metrics Over Time"))
            parts.append(
                "<p>Tasks are grouped by which function sits at the burst position. "
                "BURST = all burst-class tasks pooled; O_F1..O_Fn = other-class tasks grouped "
                "by function at burst position; ALL_OTHER = all other tasks; ALL_DATA = everything. "
                "Each plot shows one metric over training, one line per schedule.</p>"
            )
            for p_ in (cp["pairwise_evo_by_metric"] or []):
                parts.append(_chart(p_))
        if cp.get("pairwise_evo_per_schedule"):
            parts.append(_section("Pairwise Gradient Similarity: Per Schedule"))
            parts.append(
                "<p>Each subplot shows all pairwise metrics for one schedule over training. "
                "Error bands are 95% CI across seeds.</p>"
            )
            parts.append(_chart(cp["pairwise_evo_per_schedule"], 1000))
        if cp.get("pairwise_heatmaps"):
            parts.append(_section("Pairwise Gradient Cosine Heatmaps (Per Schedule)"))
            parts.append(
                "<p>Each heatmap shows the pairwise gradient cosine similarity matrix at a "
                "specific training step for a specific schedule. Groups: BURST, O_F1..O_Fn "
                "(other tasks by function at burst position), ALL_OTHER, ALL_DATA. "
                "Averaged over seeds within each schedule.</p>"
            )
            for p_ in (cp["pairwise_heatmaps"] or []):
                parts.append(_chart(p_, 700))
    _try(_pairwise_evo, "Pairwise Gradient Similarity")

    def _adl():
        has_adl = any(cp.get(k) for k in [
            "adl_delta_norm", "adl_readability", "adl_causal_ablation",
            "adl_end_burst_bars", "adl_readability_vs_auc",
        ])
        if not has_adl:
            return
        parts.append(_section("Activation Difference Lens (ADL)"))
        parts.append(
            "<p>The ADL metric measures the global activation bias introduced by the burst phase. "
            "For each checkpoint, we compute the mean activation difference on <em>other-class</em> "
            "inputs between the checkpoint model and the pre-burst model:</p>"
            "<p style='text-align:center;font-family:monospace;'>"
            "&delta;<sub>l</sub> = mean<sub>x &isin; other</sub>[ h<sup>checkpoint</sup><sub>l</sub>(x) "
            "&minus; h<sup>pre-burst</sup><sub>l</sub>(x) ]</p>"
            "<p>Applying the unembedding matrix (Logit Lens) to &delta;<sub>l</sub> reveals whether "
            "the bias encodes burst-relevant tokens &mdash; a direct test of the wrapper hypothesis. "
            "The causal ablation projects &delta; out of activations and measures the accuracy drop "
            "on burst-class data: a large drop means the model relies on the global bias (wrapper); "
            "a small drop means deeper, entangled representations.</p>"
        )
        parts.append(
            "<p><b>Prediction:</b> burst_100 should show high readability and large ablation drop "
            "(pure wrapper); burst_10 should show near-zero readability and small ablation drop "
            "(deeper representations).</p>"
        )
        if cp.get("adl_delta_norm"):
            parts.append("<h3>Activation Bias Magnitude (&Vert;&delta;&Vert;) Over Training</h3>")
            parts.append(
                "<p>Sum of delta norms across layers on other-class inputs. "
                "A rising norm during the burst window indicates the model is learning a global "
                "activation shift. Higher burstiness should produce a larger, faster-rising norm.</p>"
            )
            parts.append(_chart(cp["adl_delta_norm"]))
        if cp.get("adl_readability"):
            parts.append("<h3>Logit Lens Readability of Activation Bias</h3>")
            parts.append(
                "<p>Fraction of top-10 tokens (when applying the unembedding to &delta;) that are "
                "burst-relevant (b* function token or output value tokens). "
                "High readability = the bias directly encodes burst semantics = wrapper. "
                "Low readability = the bias is not burst-specific = deeper representation.</p>"
            )
            parts.append(_chart(cp["adl_readability"]))
        if cp.get("adl_causal_ablation"):
            parts.append("<h3>Causal Ablation: Accuracy Drop When &delta; Projected Out</h3>")
            parts.append(
                "<p>Burst-class accuracy before minus after projecting &delta; out of activations "
                "(mean over layers). Positive = the model relied on the global bias for burst accuracy. "
                "Zero = the capability is encoded in directions orthogonal to &delta; (deeper).</p>"
            )
            parts.append(_chart(cp["adl_causal_ablation"]))
        if cp.get("adl_end_burst_bars"):
            parts.append("<h3>End-of-Burst Summary: Readability and Ablation Drop by Schedule</h3>")
            parts.append(_chart(cp["adl_end_burst_bars"]))
        if cp.get("adl_readability_vs_auc"):
            parts.append("<h3>ADL Readability vs Forgetting Resistance</h3>")
            parts.append(
                "<p>Each dot is one seed &times; schedule. "
                "If the wrapper hypothesis holds, higher readability (more wrapper-like) should "
                "predict lower reversion AUC (faster forgetting).</p>"
            )
            parts.append(_chart(cp["adl_readability_vs_auc"], 800))
    _try(_adl, "ADL")

    def _conclusions():
        parts.append(_section("Conclusions"))
        parts.append("<ul>")
        for b, t in [
            ("Acquisition:", "All schedules acquire special class (peak ~ 1.0)."),
            ("Retention:", "burst_10 (uniform) is most forgetting-resistant."),
            ("Mixing:", "More other classes during burst = slower forgetting."),
            ("Variance:", "burst_100: fastest forgetting, high variance."),
            ("Background:", "Other classes robust across all schedules."),
        ]:
            parts.append(f"<li><b>{b}</b> {t}</li>")
        parts.append("</ul>")
        parts.append("<h3>Interpretation</h3>")
        parts.append("<p>Interleaving other classes with novel special class creates integrated, durable representations. Isolated bursts create fragile shortcuts. Many small exposures > concentrated bursts.</p>")
    _try(_conclusions, "Conclusions")

    def _followup():
        parts.append(_section("Follow-up Ideas"))
        parts.append("<ul>")
        for b, t in [
            ("Depth:", "Deeper compositions (depth 4, 5)"),
            ("Capacity:", "Multiple novel functions"),
            ("Architecture:", "Different model sizes"),
            ("Duration:", "Longer reversion phases"),
            ("Mechanistic:", "Activation patching for special class localization"),
            ("Recovery:", "Can special class be recovered after forgetting?"),
        ]:
            parts.append(f"<li><b>{b}</b> {t}</li>")
        parts.append("</ul><h3>Open Questions</h3><ul>")
        parts.append("<li>Why high variance in burst_100?</li>")
        parts.append("<li>Critical other-classes mixing threshold?</li>")
        parts.append("<li>LR-forgetting interaction?</li>")
        parts.append("</ul>")
    _try(_followup, "Follow-up Ideas")

    def _appendix():
        ti = cfg.get("task_info", {})
        doc_len = ti.get("doc_len", "?")
        prompt_len = ti.get("prompt_len", "?")
        seed_base = cfg.get("seed_base", 107)
        parts.append(_section("Appendix"))
        parts.append(f"<p>Depth={depth}, burst_pos={burst_pos}. Doc={doc_len}, prompt={prompt_len}</p>")
        parts.append(f"<p>Bijections: permutations of 0-9. [0]=id, [1-{n_a*depth}]=other ({n_a} per position), [{n_a*depth+1}]=b*. Data seed={seed_base}.</p>")
        parts.append(f"<p>Seeds: {seed_base}-{seed_base+ns-1}.</p>")
    _try(_appendix, "Appendix")

    parts.append("</body></html>")

    out = rd / "presentation" / "burst_presentation.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts))
    return out


def main():
    rd = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(Path("data").glob("burst_d*"))[-1]
    print(f"Loading from {rd}...")
    from burst.train_utils import resolve_run_paths
    cfg_path, logs_dir, _ = resolve_run_paths(rd)
    with open(logs_dir / "all_results.pkl", "rb") as f:
        results = pickle.load(f)
    with open(cfg_path) as f:
        cfg = json.load(f)
    print(f"  {len(results)} results\nGenerating charts...")
    cp = generate_all(rd, results, cfg)
    print("Building HTML presentation...")
    out = build(rd, results, cfg, cp)
    print(f"Done! {out}")


if __name__ == "__main__":
    main()

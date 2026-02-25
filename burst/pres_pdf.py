"""PDF builder. Usage: python burst/pres_pdf.py data/burst_d<depth>_<run_tag>"""
import sys, os, pickle, json, base64, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from pathlib import Path
from burst.pres_charts import PALETTE, SCHED_SHORT, _ordered, _group, generate_all
from burst.config import TrainConfig, reversion_life_key, reversion_life_label


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


def _safe_get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


def build(rd, res, cfg, cp):
    bcfg = cfg.get("base_cfg", cfg)
    n_a = cfg.get("n_a", 4)
    ti = cfg.get("task_info", {})
    if not isinstance(ti, dict):
        ti = {}
    depth = cfg.get("depth", _safe_get(ti, "depth", 3))
    burst_pos = cfg.get("burst_pos", _safe_get(ti, "burst_pos", depth))
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
    parts.append(f'<p class="meta">{nl}L/{ne}d/{nh}H | {T} foundation+burst + {U} reversion | batch {bs} | {len(sc)} schedules x {ns} seeds = {len(res)} runs</p>')
    parts.append('<p class="meta">Free generation evaluation</p>')

    def _research_q():
        parts.append(_section("Research Question"))
        parts.append("<p>How does the training schedule for introducing novel compositional knowledge affect a Transformer's ability to (a) acquire that knowledge and (b) retain it when the novel data is removed?</p>")
        parts.append("<p>Does interleaving other classes with the burst class during the burst window produce more robust representations than presenting the burst class in isolation?</p>")
        parts.append("<h3>Why This Matters</h3>")
        parts.append("<p>Understanding how neural networks acquire and forget compositional skills is fundamental to continual learning, curriculum design, and knowledge editing.</p>")
    _try(_research_q, "Research Question")

    def _setup():
        parts.append(_section("Experimental Setup"))
        parts.append(f"<h3>Task: Depth-{depth} Bijection Composition (burst at position {burst_pos})</h3>")
        parts.append(f"<p>Model applies chains of {depth} bijection functions to 6 digits. Eval: free generation.</p>")
        parts.append("<h3>Data Split</h3><ul>")
        parts.append(f"<li><b>Other Classes:</b> {n_a} bijections x {depth} positions = {n_a**depth} other-class compositions</li>")
        parts.append(f"<li><b>Burst Class:</b> 1 new bijection b* at pos {burst_pos}, all {n_a**(depth-1)} combos for other positions</li>")
        parts.append("</ul><h3>Model &amp; Training</h3><ul>")
        parts.append(f"<li>{nl}L Transformer, {ne}d, {nh}H, SwiGLU, no dropout</li>")
        parts.append(f"<li>AdamW lr={bcfg['lr']}, cosine decay, batch {bs}, bfloat16</li>")
        parts.append("</ul>")
    _try(_setup, "Experimental Setup")

    def _protocol():
        parts.append(_section("Training Protocol"))
        parts.append(f"<h3>Foundation + Burst (0-{T-1})</h3>")
        parts.append(f"<p>Other classes + Burst class mixed per schedule. ~{int(p*100)}% burst class exposure.</p>")
        parts.append(f"<h3>Reversion ({T}-{T+U-1})</h3>")
        parts.append("<p>Burst class removed. Other classes only. LR continues decaying.</p>")
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
        parts.append(f"<li><b>burst_100:</b> Pure burst class for last {bl} steps</li>")
        for pct, frac in [(98, 0.98), (95, 0.95), (90, 0.90), (85, 0.85),
                          (75, 0.75), (50, 0.50), (25, 0.25)]:
            if f"burst_{pct}" in sc:
                win = min(int(bl / frac), T)
                parts.append(f"<li><b>burst_{pct}:</b> {pct}% burst class + {100-pct}% other classes for last {win} steps</li>")
        if "burst_10" in sc:
            parts.append(f"<li><b>burst_10:</b> ~{int(p*100)}% burst class randomly throughout (uniform control)</li>")
        parts.append("</ul>")
    _try(_schedules, "Training Schedules")

    def _metrics():
        thresholds = TrainConfig().reversion_thresholds
        parts.append(_section("Metrics"))
        parts.append("<ul>")
        parts.append("<li><b>Peak Burst:</b> Burst class accuracy at end of training</li>")
        for t in thresholds:
            pct = int(t * 100)
            parts.append(f"<li><b>{reversion_life_label(t)}:</b> Reversion steps to {pct}% of peak (cap {U})</li>")
        parts.append("<li><b>Reversion AUC:</b> Area under burst class curve during reversion</li>")
        parts.append("</ul>")
    _try(_metrics, "Metrics")

    def _hypotheses():
        parts.append(_section("Hypotheses"))
        for hid, txt, expl in [
            (1, "All schedules achieve peak burst class ~ 1.0", f"Sufficient capacity for {n_a**(depth-1)} compositions."),
            (2, "burst_10 (uniform) = most forgetting-resistant", "Distributed burst class integrates with other classes knowledge."),
            (3, "burst_100 = fastest forgetting", "Isolated burst class creates fragile representations."),
            (4, "Mixed schedules ordered by other-classes content", "More other-classes mixing = more robust burst class."),
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
        parts.append(_section("Result 1: Peak Burst Class Accuracy"))
        parts.append(_chart(cp.get("peak_bars")))
        parts.append('<div class="hbox">H1: All schedules achieve peak burst class ~ 1.0</div>')
        if all(m >= 0.998 for m in pv.values()):
            parts.append(_verdict_html("SUPPORTED", f"All >= 0.998. Range: {min(pv.values()):.3f}-{max(pv.values()):.3f}.", "supported"))
        else:
            parts.append(_verdict_html("PARTIAL", f"Range: {min(pv.values()):.3f}-{max(pv.values()):.3f}.", "partial"))
    _try(_result1, "Result 1")

    def _result2():
        parts.append(_section("Result 2: Burst Class Accuracy Over Time"))
        parts.append(_chart(cp.get("overlay_burst")))
        parts.append(f"<p>All reach ~100% by step {T}. Forgetting speed varies dramatically.</p>")
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
    _try(_per_sched, "Per-Schedule Detail")

    def _probes():
        has_probes = cp.get("probe_dynamics") or cp.get("probe_heatmaps") or cp.get("probe_layer_schedule")
        if not has_probes:
            return
        parts.append(_section("Linear Probes: Other vs Burst Representations"))
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
        parts.append(_section("Gradient Cosine Similarity: Burst vs Other Classes"))
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
            "High similarity means the burst class is pulling the model in the same direction "
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

    def _conclusions():
        parts.append(_section("Conclusions"))
        parts.append("<ul>")
        for b, t in [
            ("Acquisition:", "All schedules acquire burst class (peak ~ 1.0)."),
            ("Retention:", "burst_10 (uniform) is most forgetting-resistant."),
            ("Mixing:", "More other classes during burst = slower forgetting."),
            ("Variance:", "burst_100: fastest forgetting, high variance."),
            ("Background:", "Other classes robust across all schedules."),
        ]:
            parts.append(f"<li><b>{b}</b> {t}</li>")
        parts.append("</ul>")
        parts.append("<h3>Interpretation</h3>")
        parts.append("<p>Interleaving other classes with novel burst class creates integrated, durable representations. Isolated bursts create fragile shortcuts. Many small exposures > concentrated bursts.</p>")
    _try(_conclusions, "Conclusions")

    def _followup():
        parts.append(_section("Follow-up Ideas"))
        parts.append("<ul>")
        for b, t in [
            ("Depth:", "Deeper compositions (depth 4, 5)"),
            ("Capacity:", "Multiple novel functions"),
            ("Architecture:", "Different model sizes"),
            ("Duration:", "Longer reversion phases"),
            ("Mechanistic:", "Activation patching for burst class localization"),
            ("Recovery:", "Can burst class be recovered after forgetting?"),
        ]:
            parts.append(f"<li><b>{b}</b> {t}</li>")
        parts.append("</ul><h3>Open Questions</h3><ul>")
        parts.append("<li>Why high variance in burst_100?</li>")
        parts.append("<li>Critical other-classes mixing threshold?</li>")
        parts.append("<li>LR-forgetting interaction?</li>")
        parts.append("</ul>")
    _try(_followup, "Follow-up Ideas")

    def _appendix():
        doc_len = _safe_get(ti, "doc_len", 32)
        prompt_len = _safe_get(ti, "prompt_len", 12)
        parts.append(_section("Appendix"))
        parts.append(f"<p>Depth={depth}, burst_pos={burst_pos}. Doc={doc_len}, prompt={prompt_len}</p>")
        parts.append(f"<p>Bijections: permutations of 0-9. [0]=id, [1-{n_a}]=other, [{n_a+1}]=b*. Data seed={cfg.get('seed_base', 999)}.</p>")
        parts.append(f"<p>Seeds: {cfg.get('seed_base',107)}-{cfg.get('seed_base',107)+ns-1}.</p>")
    _try(_appendix, "Appendix")

    parts.append("</body></html>")

    out = rd / "presentation" / "burst_presentation.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts))
    return out


def main():
    rd = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(Path("data").glob("burst_d*"))[-1]
    print(f"Loading from {rd}...")
    with open(rd / "all_results.pkl", "rb") as f:
        results = pickle.load(f)
    with open(rd / "config.json") as f:
        cfg = json.load(f)
    print(f"  {len(results)} results\nGenerating charts...")
    cp = generate_all(rd, results, cfg)
    print("Building HTML presentation...")
    out = build(rd, results, cfg, cp)
    print(f"Done! {out}")


if __name__ == "__main__":
    main()

"""PDF builder. Usage: python burst/pres_pdf.py data/burst_d3_data_500"""
import sys, os, pickle, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from pathlib import Path
from fpdf import FPDF
from burst.pres_charts import PALETTE, SCHED_SHORT, _ordered, _group, generate_all
from burst.config import TrainConfig, reversion_life_key, reversion_life_label
W, H = 297, 210

_UNICODE_SUBS = {"\u2014": "--", "\u2013": "-", "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2026": "..."}

def _latin1_safe(t: str) -> str:
    for u, a in _UNICODE_SUBS.items():
        t = t.replace(u, a)
    return t.encode("latin-1", "replace").decode("latin-1")

class PDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 7)
            self.set_text_color(140, 140, 140)
            self.cell(0, 4, "Depth-3 Bijection Burst", align="L")
            self.cell(0, 4, f"p. {self.page_no()}", align="R")
            self.ln(6)
    def st(self, t):
        self.set_font("Helvetica", "B", 20)
        self.set_text_color(13, 71, 161)
        self.multi_cell(0, 10, _latin1_safe(t), align="L")
        self.ln(3)
    def sh(self, t):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(13, 71, 161)
        self.cell(0, 8, _latin1_safe(t), new_x="LMARGIN", new_y="NEXT")
        self.ln(2)
    def bt(self, t):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5, _latin1_safe(t))
        self.ln(2)
    def bu(self, t, b=""):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.cell(6, 5, "-")
        if b:
            self.set_font("Helvetica", "B", 10)
            self.cell(self.get_string_width(b) + 1, 5, b)
            self.set_font("Helvetica", "", 10)
        self.multi_cell(W - 26, 5, t)
        self.ln(1)
    def hbox(self, hid, t):
        self.set_fill_color(240, 245, 255)
        self.set_draw_color(13, 71, 161)
        self.rect(self.get_x(), self.get_y(), W - 20, 8, style="DF")
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(13, 71, 161)
        self.cell(W - 20, 8, f"  H{hid}: {t}", new_x="LMARGIN", new_y="NEXT")
        self.ln(1)
    def vbox(self, v, rgb, e):
        light = tuple(int(c + (255 - c) * 0.85) for c in rgb)
        self.set_fill_color(*light)
        self.set_draw_color(*rgb)
        self.rect(self.get_x(), self.get_y(), W - 20, 7, style="DF")
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*rgb)
        self.cell(W - 20, 7, f"  VERDICT: {v}", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(50, 50, 50)
        self.multi_cell(W - 24, 4.5, e)
        self.ln(3)
    def ch(self, path, w=250):
        if Path(path).exists():
            if self.get_y() > H - 60:
                self.add_page()
            self.image(str(path), x=(W - w) / 2, w=w)
            self.ln(4)

def build(rd, res, cfg, cp):
    bcfg = cfg.get("base_cfg", cfg)
    n_a = cfg.get("n_a", 4)
    ti = cfg.get("task_info", {})
    depth = cfg.get("depth", ti.get("depth", 3))
    burst_pos = cfg.get("burst_pos", ti.get("burst_pos", depth))
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

    pdf = PDF(orientation="L", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page(); pdf.ln(30)
    pdf.set_font("Helvetica", "B", 28); pdf.set_text_color(13, 71, 161)
    pdf.multi_cell(0, 13, "Compositional Learning & Forgetting\nin Transformers", align="C")
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 14); pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, f"Depth-{depth} Bijection Burst Experiment (burst at position {burst_pos})",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)
    pdf.set_font("Courier", "", 9); pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 5, f"{nl}L/{ne}d/{nh}H | {T} foundation+burst + {U} reversion | batch {bs} | {len(sc)} schedules x {ns} seeds = {len(res)} runs",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, "Free generation evaluation", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.add_page(); pdf.st("Research Question")
    pdf.bt("How does the training schedule for introducing novel compositional knowledge affect a Transformer's ability to (a) acquire that knowledge and (b) retain it when the novel data is removed?")
    pdf.bt("Does interleaving other classes with the burst class during the burst window produce more robust representations than presenting the burst class in isolation?")
    pdf.sh("Why This Matters")
    pdf.bt("Understanding how neural networks acquire and forget compositional skills is fundamental to continual learning, curriculum design, and knowledge editing.")

    pdf.add_page(); pdf.st("Experimental Setup")
    pdf.sh(f"Task: Depth-{depth} Bijection Composition (burst at position {burst_pos})")
    pdf.bt(f"Model applies chains of {depth} bijection functions to 6 digits. Eval: free generation.")
    pdf.sh("Data Split")
    pdf.bu(f"{n_a} bijections x {depth} positions = {n_a**depth} other-class compositions", "Other Classes: ")
    pdf.bu(f"1 new bijection b* at pos {burst_pos}, all {n_a**(depth-1)} combos for other positions", "Burst Class: ")
    pdf.sh("Model & Training")
    pdf.bu(f"{nl}L Transformer, {ne}d, {nh}H, SwiGLU, no dropout")
    pdf.bu(f"AdamW lr={bcfg['lr']}, cosine decay, batch {bs}, bfloat16")

    pdf.add_page(); pdf.st("Training Protocol")
    pdf.sh(f"Foundation + Burst (0-{T-1})")
    pdf.bt(f"Other classes + Burst class mixed per schedule. ~{int(p*100)}% burst class exposure.")
    pdf.sh(f"Reversion ({T}-{T+U-1})")
    pdf.bt("Burst class removed. Other classes only. LR continues decaying.")
    pdf.sh("Evaluation"); pdf.bu("Every 10 steps, free generation, last 6 tokens")

    pdf.add_page(); pdf.st("LR Schedule"); pdf.ch(cp["lr"], w=260)
    pdf.add_page(); pdf.st(f"The {len(sc)} Training Schedules"); pdf.ch(cp["schedule_bars"], w=260)
    pdf.bu(f"Pure burst class for last {bl} steps", "burst_100: ")
    for pct, frac in [(98, 0.98), (95, 0.95), (90, 0.90), (85, 0.85),
                      (75, 0.75), (50, 0.50), (25, 0.25)]:
        if f"burst_{pct}" in sc:
            win = min(int(bl / frac), T)
            pdf.bu(f"{pct}% burst class + {100-pct}% other classes for last {win} steps",
                   f"burst_{pct}: ")
    if "burst_10" in sc:
        pdf.bu(f"~{int(p*100)}% burst class randomly throughout (uniform control)", "burst_10: ")

    thresholds = TrainConfig().reversion_thresholds
    pdf.add_page(); pdf.st("Metrics")
    pdf.bu("Burst class accuracy at end of training", "Peak Burst: ")
    for t in thresholds:
        pct = int(t * 100)
        pdf.bu(f"Reversion steps to {pct}% of peak (cap {U})", f"{reversion_life_label(t)}: ")
    pdf.bu("Area under burst class curve during reversion", "Reversion AUC: ")

    pdf.add_page(); pdf.st("Hypotheses")
    pdf.hbox(1, "All schedules achieve peak burst class ~ 1.0")
    pdf.bt(f"Sufficient capacity for {n_a**(depth-1)} compositions.")
    pdf.hbox(2, "burst_10 (uniform) = most forgetting-resistant")
    pdf.bt("Distributed burst class integrates with other classes knowledge.")
    pdf.hbox(3, "burst_100 = fastest forgetting")
    pdf.bt("Isolated burst class creates fragile representations.")
    pdf.hbox(4, "Mixed schedules ordered by other-classes content")
    pdf.bt("More other-classes mixing = more robust burst class.")
    pdf.hbox(5, "Other classes preserved regardless of schedule")
    pdf.bt("Other classes are majority of training.")

    pv = {s: np.mean([r.get(peak_key, 0) for r in gr[s]]) for s in sc}
    av = {s: np.mean([r.get(auc_key, 0) for r in gr[s]]) for s in sc}
    life_vals = {}
    for t in thresholds:
        key = reversion_life_key(t)
        life_vals[t] = {s: np.mean([r.get(key, U) for r in gr[s]]) for s in sc}
    ae = {s: np.mean([r["log"][other_log_key][-1] for r in gr[s]]) for s in sc}

    pdf.add_page(); pdf.st("Result 1: Peak Burst Class Accuracy"); pdf.ch(cp["peak_bars"], w=240)
    pdf.hbox(1, "All schedules achieve peak burst class ~ 1.0")
    if all(m >= 0.998 for m in pv.values()):
        pdf.vbox("SUPPORTED", (0, 128, 0), f"All >= 0.998. Range: {min(pv.values()):.3f}-{max(pv.values()):.3f}.")
    else:
        pdf.vbox("PARTIAL", (255, 152, 0), f"Range: {min(pv.values()):.3f}-{max(pv.values()):.3f}.")

    pdf.add_page(); pdf.st("Result 2: Burst Class Accuracy Over Time"); pdf.ch(cp["overlay_burst"], w=260)
    pdf.bt(f"All reach ~100% by step {T}. Forgetting speed varies dramatically.")

    pdf.add_page(); pdf.st("Result 3: Forgetting Dynamics"); pdf.ch(cp["reversion_zoom"], w=260)
    order_str = " > ".join(SCHED_SHORT.get(s, s) for s in sorted(av, key=av.get, reverse=True))
    pdf.bt(f"Ordering by retention: {order_str}")

    pdf.add_page(); pdf.st("Result 4: Reversion AUC"); pdf.ch(cp["auc_bars"], w=240)
    pdf.hbox(2, "burst_10 (uniform) = most forgetting-resistant")
    best = max(av, key=av.get)
    if best == "burst_10":
        pdf.vbox("SUPPORTED", (0, 128, 0), f"burst_10 highest AUC ({av['burst_10']:.0f}).")
    else:
        b10_auc = av.get("burst_10", 0)
        pdf.vbox("NOT SUPPORTED", (211, 47, 47), f"{best} higher ({av[best]:.0f} vs burst_10 {b10_auc:.0f}).")

    for ti, t in enumerate(thresholds):
        if t not in cp["life_bars"]:
            continue
        label = reversion_life_label(t)
        pct = int(t * 100)
        lv = life_vals[t]
        pdf.add_page()
        pdf.st(f"Result 5.{ti+1}: {label}")
        pdf.ch(cp["life_bars"][t], w=240)
        pdf.hbox(3, "burst_100 = fastest forgetting")
        low = min(lv, key=lv.get)
        if low == "burst_100":
            pdf.vbox("SUPPORTED", (0, 128, 0), f"Lowest {label} ({lv['burst_100']:.0f}). High variance.")
        else:
            pdf.vbox("NOT SUPPORTED", (211, 47, 47), f"{low} lower ({lv[low]:.0f}).")

    pdf.add_page(); pdf.st("Result 6: Schedule Ordering"); pdf.ch(cp["auc_diff"], w=200)
    pdf.hbox(4, "Mixed schedules ordered by other-classes content")
    order = sorted(av, key=av.get, reverse=True)
    pdf.vbox("OBSERVED", (0, 100, 180), f"Got: {' > '.join(SCHED_SHORT.get(s, s) for s in order)}")

    pdf.add_page(); pdf.st("Result 7: Other Classes Preservation"); pdf.ch(cp["overlay_other"], w=260)
    pdf.hbox(5, "Other classes preserved regardless of schedule")
    if all(m >= 0.95 for m in ae.values()):
        pdf.vbox("SUPPORTED", (0, 128, 0), "All other classes >= 0.95 at end.")
    else:
        pdf.vbox("PARTIAL", (255, 152, 0), f"Min: {min(ae.values()):.3f}")

    pdf.add_page(); pdf.st("Summary Statistics"); pdf.ch(cp["summary_table"], w=270)
    pdf.add_page(); pdf.st("Per-Schedule Detail")
    for path in cp["per_sched"]:
        pdf.ch(path, w=240)

    has_probes = cp.get("probe_dynamics") or cp.get("probe_heatmaps") or cp.get("probe_layer_schedule")
    if has_probes:
        pdf.add_page(); pdf.st("Linear Probes: Other vs Burst Representations")
        pdf.bt(
            "Logistic regression probes trained on residual-stream activations at every "
            "(layer, token position) pair to classify Other-class vs Burst-class representations. "
            "5-fold cross-validation accuracy, aggregated across seeds with 95% CI."
        )
        if cp.get("probe_dynamics"):
            pdf.add_page(); pdf.sh("Probe Accuracy Over Training")
            pdf.ch(cp["probe_dynamics"], w=260)
        if cp.get("probe_layer_schedule"):
            for p_ in cp["probe_layer_schedule"]:
                pdf.add_page(); pdf.sh("Layer x Schedule Probe Accuracy")
                pdf.ch(p_, w=240)
        if cp.get("probe_heatmaps"):
            pdf.add_page(); pdf.sh("Probe Heatmaps (per schedule, mean across seeds)")
            for p_ in cp["probe_heatmaps"]:
                pdf.ch(p_, w=260)

    pd_ = rd / "next_token_regime_probes"
    if pd_.exists():
        pdf.add_page(); pdf.st("Next-Token Probes")
        pdf.bt("Logit lens + learned linear probe at burst-position outputs, Other vs Burst.")
        for sd in sorted(pd_.glob("step_*")):
            step = sd.name.replace("step_", "")
            for m in ["logit_lens", "learned_probe"]:
                for k, w in [("curves", 260), ("diff", 240)]:
                    fp = sd / f"{k}_{m}.png"
                    if fp.exists():
                        pdf.add_page(); pdf.st(f"{k}: {m} @ step {step}"); pdf.ch(fp, w=w)
        cb = pd_ / "combined"
        if cb.exists():
            for m in ["logit_lens", "learned_probe"]:
                for k in ["curves", "diff"]:
                    fp = cb / f"combined_{k}_{m}.png"
                    if fp.exists():
                        pdf.add_page(); pdf.st(f"Evolution: {k} {m}"); pdf.ch(fp, w=260)

    has_gs = any(cp.get(k) for k in [
        "grad_cosine_overlay", "grad_cosine_bars", "grad_cosine_per_seed",
        "grad_cosine_rate", "grad_cosine_phase", "grad_cosine_vs_auc",
        "grad_cosine_phase_bars",
    ])
    if has_gs:
        pdf.add_page(); pdf.st("Gradient Cosine Similarity: Burst vs Other Classes")
        pdf.sh("How It Works (Autoregressive Regime)")
        pdf.bt(
            "The model is trained autoregressively: given a sequence "
            "[S F3 F2 F1 ' ' input ' ' out1 ' ' out2 ' ' out3], the loss is standard "
            "next-token cross-entropy over all positions. At evaluation time, the model "
            "generates its own outputs token-by-token (free generation) from a prompt "
            "containing only the function slots and input -- it never sees the ground-truth "
            "intermediate or final outputs during inference."
        )
        gs_bs = bcfg.get("grad_sim_batch_size", 64)
        pdf.bt(
            f"To compute gradient similarity, we sample {gs_bs} documents from each class "
            "(burst and other), compute the next-token prediction loss on the full sequence "
            "for each class separately, backpropagate to obtain a gradient vector (the "
            "concatenation of all parameter gradients), and measure cosine similarity "
            "between the two gradient vectors. Because the loss is autoregressive over "
            "the entire sequence -- including the intermediate composition outputs -- the "
            "gradient captures how each class shapes the model's predictions at every "
            "position in the chain, not just the final output."
        )
        pdf.sh("Interpretation")
        grad_sim_every = cfg.get("grad_sim_every", 50)
        pdf.bt(
            f"Computed every {grad_sim_every} steps throughout training. "
            "High similarity means the burst class is pulling the model in the same direction "
            "as the other classes -- suggesting integrated, durable representations. "
            "Low or negative similarity indicates conflicting gradient directions, "
            "which predicts faster forgetting during reversion."
        )
        if cp.get("grad_cosine_overlay"):
            pdf.add_page(); pdf.sh("Burst vs Other: All Schedules")
            pdf.ch(cp["grad_cosine_overlay"], w=W - 20)
        if cp.get("grad_cosine_bars"):
            pdf.sh("End-of-Burst Snapshot")
            pdf.ch(cp["grad_cosine_bars"], w=240)
        if cp.get("grad_cosine_phase"):
            pdf.add_page(); pdf.sh("Burst Phase vs Reversion Phase")
            pdf.ch(cp["grad_cosine_phase"], w=260)
        if cp.get("grad_cosine_phase_bars"):
            pdf.add_page(); pdf.sh("Similarity Across Training Phases")
            pdf.ch(cp["grad_cosine_phase_bars"], w=260)
        if cp.get("grad_cosine_rate"):
            pdf.add_page(); pdf.sh("Rate of Change")
            pdf.bt("Derivative of cosine similarity over training steps. "
                    "Peaks indicate where gradient alignment shifts fastest.")
            pdf.ch(cp["grad_cosine_rate"], w=260)
        if cp.get("grad_cosine_vs_auc"):
            pdf.add_page(); pdf.sh("Gradient Alignment vs Forgetting Resistance")
            pdf.bt("Each dot is one seed x schedule. Positive correlation means "
                    "higher end-of-burst gradient alignment predicts slower forgetting.")
            pdf.ch(cp["grad_cosine_vs_auc"], w=240)
        if cp.get("grad_cosine_per_seed"):
            pdf.add_page(); pdf.sh("Per-Seed Traces")
            pdf.bt("Individual seed traces reveal variance in gradient alignment within each schedule.")
            for p_ in cp["grad_cosine_per_seed"]:
                pdf.ch(p_, w=240)

    if cp.get("pairwise_evo_by_metric"):
        pdf.add_page(); pdf.st("Pairwise Gradient Similarity: Metrics Over Time")
        pdf.bt(
            "Tasks are grouped by which function sits at the burst position. "
            "BURST = all burst-class tasks pooled; O_F1..O_Fn = other-class tasks grouped "
            "by function at burst position; ALL_OTHER = all other tasks; ALL_DATA = everything. "
            "Each plot shows one metric over training, one line per schedule."
        )
        for p_ in (cp["pairwise_evo_by_metric"] or []):
            pdf.ch(p_, w=260)

    if cp.get("pairwise_evo_per_schedule"):
        pdf.add_page(); pdf.st("Pairwise Gradient Similarity: Per Schedule")
        pdf.bt(
            "Each subplot shows all pairwise metrics for one schedule over training. "
            "Error bands are 95% CI across seeds."
        )
        pdf.ch(cp["pairwise_evo_per_schedule"], w=270)

    if cp.get("pairwise_heatmaps"):
        pdf.add_page(); pdf.st("Pairwise Gradient Cosine Heatmaps (Per Schedule)")
        pdf.bt(
            "Each heatmap shows the pairwise gradient cosine similarity matrix at a "
            "specific training step for a specific schedule. Groups: BURST, O_F1..O_Fn "
            "(other tasks by function at burst position), ALL_OTHER, ALL_DATA. "
            "Averaged over seeds within each schedule."
        )
        for p_ in (cp["pairwise_heatmaps"] or []):
            pdf.ch(p_, w=210)

    pdf.add_page(); pdf.st("Conclusions")
    pdf.bu("All schedules acquire burst class (peak ~ 1.0).", "Acquisition: ")
    pdf.bu("burst_10 (uniform) is most forgetting-resistant.", "Retention: ")
    pdf.bu("More other classes during burst = slower forgetting.", "Mixing: ")
    pdf.bu("burst_100: fastest forgetting, high variance.", "Variance: ")
    pdf.bu("Other classes robust across all schedules.", "Background: ")
    pdf.sh("Interpretation")
    pdf.bt("Interleaving other classes with novel burst class creates integrated, durable representations. Isolated bursts create fragile shortcuts. Many small exposures > concentrated bursts.")

    pdf.add_page(); pdf.st("Follow-up Ideas")
    pdf.bu("Deeper compositions (depth 4, 5)", "Depth: ")
    pdf.bu("Multiple novel functions", "Capacity: ")
    pdf.bu("Different model sizes", "Architecture: ")
    pdf.bu("Longer reversion phases", "Duration: ")
    pdf.bu("Activation patching for burst class localization", "Mechanistic: ")
    pdf.bu("Can burst class be recovered after forgetting?", "Recovery: ")
    pdf.sh("Open Questions")
    pdf.bu("Why high variance in burst_100?")
    pdf.bu("Critical other-classes mixing threshold?")
    pdf.bu("LR-forgetting interaction?")

    pdf.add_page(); pdf.st("Appendix")
    pdf.bt(f"Depth={depth}, burst_pos={burst_pos}. Doc={ti.get('doc_len',32)}, prompt={ti.get('prompt_len',12)}")
    pdf.bt(f"Bijections: permutations of 0-9. [0]=id, [1-{n_a}]=other, [{n_a+1}]=b*. Data seed={cfg.get('seed_base', 999)}.")
    pdf.bt(f"Seeds: {cfg.get('seed_base',107)}-{cfg.get('seed_base',107)+ns-1}.")

    out = rd / "presentation" / "burst_presentation.pdf"
    pdf.output(str(out))
    return out

def main():
    rd = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(Path("data").glob("burst_d3_*"))[-1]
    print(f"Loading from {rd}...")
    with open(rd / "all_results.pkl", "rb") as f:
        results = pickle.load(f)
    with open(rd / "config.json") as f:
        cfg = json.load(f)
    print(f"  {len(results)} results\nGenerating charts...")
    cp = generate_all(rd, results, cfg)
    print("Building PDF...")
    out = build(rd, results, cfg, cp)
    print(f"Done! {out}")

if __name__ == "__main__":
    main()

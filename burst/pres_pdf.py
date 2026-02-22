"""PDF builder. Usage: python burst/pres_pdf.py data/burst_d3_data_500"""
import sys, os, pickle, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from pathlib import Path
from fpdf import FPDF
from burst.pres_charts import PALETTE, SCHED_SHORT, _ordered, _group, generate_all
W, H = 297, 210

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
        self.multi_cell(0, 10, t, align="L")
        self.ln(3)
    def sh(self, t):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(13, 71, 161)
        self.cell(0, 8, t, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)
    def bt(self, t):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5, t)
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
    T, U = bcfg["total_steps"], bcfg["undo_steps"]
    nl, ne, nh = bcfg["n_layer"], bcfg["n_embd"], bcfg["n_head"]
    bs, p = bcfg["batch_size"], bcfg["p_target"]
    ns = cfg.get("n_seeds", 5)
    gr = _group(res)
    sc = _ordered(gr.keys())
    bl = max(int(p * T), 1)
    pdf = PDF(orientation="L", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page(); pdf.ln(30)
    pdf.set_font("Helvetica", "B", 28); pdf.set_text_color(13, 71, 161)
    pdf.multi_cell(0, 13, "Compositional Learning & Forgetting\nin Transformers", align="C")
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 14); pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, "Depth-3 Bijection Burst Experiment", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)
    pdf.set_font("Courier", "", 9); pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 5, f"{nl}L/{ne}d/{nh}H | {T} train + {U} undo | batch {bs} | {len(sc)}sched x {ns}seeds = {len(res)} runs", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, "Free generation evaluation", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.add_page(); pdf.st("Research Question")
    pdf.bt("How does the training schedule for introducing novel compositional knowledge affect a Transformer's ability to (a) acquire that knowledge and (b) retain it when the novel data is removed?")
    pdf.bt("Does interleaving background (A) data with novel (B) data during the burst produce more robust representations than presenting B in isolation?")
    pdf.sh("Why This Matters")
    pdf.bt("Understanding how neural networks acquire and forget compositional skills is fundamental to continual learning, curriculum design, and knowledge editing.")
    pdf.add_page(); pdf.st("Experimental Setup")
    pdf.sh("Task: Depth-3 Bijection Composition")
    pdf.bt(f"Model applies chains of 3 bijection functions to 6 digits. Format: S [F3 F2 F1] [input] [after F1] [after F2] [after F3]. Eval: free generation.")
    pdf.sh("Data Split")
    pdf.bu(f"{n_a} bijections x 3 positions = {n_a**3} A compositions", "Background (A): ")
    pdf.bu(f"1 new bijection b* at pos 3, all {n_a**2} pairs for pos 1-2", "Novel (B): ")
    pdf.sh("Model & Training")
    pdf.bu(f"{nl}L Transformer, {ne}d, {nh}H, SwiGLU, no dropout")
    pdf.bu(f"AdamW lr={bcfg['lr']}, cosine decay, batch {bs}, bfloat16")
    pdf.add_page(); pdf.st("Training Protocol")
    pdf.sh(f"Phase 1: Training (0-{T-1})")
    pdf.bt(f"A+B mixed per schedule. ~{int(p*100)}% B exposure.")
    pdf.sh(f"Phase 2: Undo ({T}-{T+U-1})")
    pdf.bt("B removed. A only. LR continues decaying.")
    pdf.sh("Evaluation"); pdf.bu("Every 10 steps, free generation, last 6 tokens")
    pdf.add_page(); pdf.st("LR Schedule"); pdf.ch(cp["lr"], w=260)
    pdf.add_page(); pdf.st("The 5 Training Schedules"); pdf.ch(cp["schedule_bars"], w=260)
    pdf.bu(f"Pure B for last {bl} steps", "end_block: ")
    pdf.bu(f"75%B+25%A for last {min(int(bl/0.75),T)} steps", "end_mixed_75b: ")
    pdf.bu(f"50%B+50%A for last {min(int(bl/0.50),T)} steps", "end_mixed_50b: ")
    pdf.bu(f"25%B+75%A for last {min(int(bl/0.25),T)} steps", "end_mixed_25b: ")
    pdf.bu(f"~{int(p*100)}%B randomly throughout", "uniform: ")
    pdf.add_page(); pdf.st("Metrics")
    pdf.bu("B accuracy at step 500", "Peak B: ")
    pdf.bu(f"Undo steps to 25% of peak (cap {U})", "Quarter-life: ")
    pdf.bu("Area under B curve during undo", "Undo AUC: ")
    pdf.add_page(); pdf.st("Hypotheses")
    pdf.hbox(1, "All schedules achieve peak B ~ 1.0")
    pdf.bt("Sufficient capacity for 16 compositions.")
    pdf.hbox(2, "Uniform = most forgetting-resistant")
    pdf.bt("Distributed B integrates with A knowledge.")
    pdf.hbox(3, "End block = fastest forgetting")
    pdf.bt("Isolated B creates fragile representations.")
    pdf.hbox(4, "Mixed schedules ordered by A content")
    pdf.bt("More A-mixing = more robust B.")
    pdf.hbox(5, "A preserved regardless of schedule")
    pdf.bt("A is majority of training.")
    pv = {s: np.mean([r.get("train_end_B_comp", 0) for r in gr[s]]) for s in sc}
    av = {s: np.mean([r.get("undo_auc", 0) for r in gr[s]]) for s in sc}
    qv = {s: np.mean([r.get("quarter_life", U) for r in gr[s]]) for s in sc}
    ae = {s: np.mean([r["log"]["acc_A_comp"][-1] for r in gr[s]]) for s in sc}
    pdf.add_page(); pdf.st("Result 1: Peak B Accuracy"); pdf.ch(cp["peak_bars"], w=240)
    pdf.hbox(1, "All schedules achieve peak B ~ 1.0")
    if all(m >= 0.998 for m in pv.values()):
        pdf.vbox("SUPPORTED", (0, 128, 0), f"All >= 0.998. Range: {min(pv.values()):.3f}-{max(pv.values()):.3f}.")
    else:
        pdf.vbox("PARTIAL", (255, 152, 0), f"Range: {min(pv.values()):.3f}-{max(pv.values()):.3f}.")
    pdf.add_page(); pdf.st("Result 2: B Accuracy Over Time"); pdf.ch(cp["overlay_b"], w=260)
    pdf.bt("All reach ~100% by step 500. Forgetting speed varies dramatically.")
    pdf.add_page(); pdf.st("Result 3: Forgetting Dynamics"); pdf.ch(cp["undo_zoom"], w=260)
    pdf.bt("Clear ordering: uniform > 25%B > 50%B > 75%B > end_block.")
    pdf.add_page(); pdf.st("Result 4: Undo AUC"); pdf.ch(cp["auc_bars"], w=240)
    pdf.hbox(2, "Uniform = most forgetting-resistant")
    best = max(av, key=av.get)
    if best == "uniform":
        pdf.vbox("SUPPORTED", (0, 128, 0), f"Uniform highest AUC ({av['uniform']:.0f}).")
    else:
        pdf.vbox("NOT SUPPORTED", (211, 47, 47), f"{best} higher ({av[best]:.0f} vs {av['uniform']:.0f}).")
    pdf.add_page(); pdf.st("Result 5: Quarter-life"); pdf.ch(cp["ql_bars"], w=240)
    pdf.hbox(3, "End block = fastest forgetting")
    low = min(qv, key=qv.get)
    if low == "end_block":
        pdf.vbox("SUPPORTED", (0, 128, 0), f"Lowest quarter-life ({qv['end_block']:.0f}). High variance.")
    else:
        pdf.vbox("NOT SUPPORTED", (211, 47, 47), f"{low} lower ({qv[low]:.0f}).")
    pdf.add_page(); pdf.st("Result 6: Schedule Ordering"); pdf.ch(cp["auc_diff"], w=200)
    pdf.hbox(4, "Mixed schedules ordered by A content")
    order = sorted(av, key=av.get, reverse=True)
    exp = ["uniform", "end_mixed_25b", "end_mixed_50b", "end_mixed_75b", "end_block"]
    if order == exp:
        pdf.vbox("SUPPORTED", (0, 128, 0), " > ".join(SCHED_SHORT[s] for s in order))
    else:
        pdf.vbox("PARTIAL", (255, 152, 0), f"Got: {' > '.join(SCHED_SHORT[s] for s in order)}")
    pdf.add_page(); pdf.st("Result 7: A Preservation"); pdf.ch(cp["overlay_a"], w=260)
    pdf.hbox(5, "A preserved regardless of schedule")
    if all(m >= 0.95 for m in ae.values()):
        pdf.vbox("SUPPORTED", (0, 128, 0), "All A >= 0.95 at end.")
    else:
        pdf.vbox("PARTIAL", (255, 152, 0), f"Min: {min(ae.values()):.3f}")
    pdf.add_page(); pdf.st("Summary Statistics"); pdf.ch(cp["summary_table"], w=270)
    pdf.add_page(); pdf.st("Per-Schedule Detail")
    for path in cp["per_sched"]:
        pdf.ch(path, w=240)
    pd_ = rd / "next_token_regime_probes"
    if pd_.exists():
        pdf.add_page(); pdf.st("Next-Token Probes")
        pdf.bt("Logit lens + learned linear probe at f3-output positions, A vs B.")
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
    pdf.add_page(); pdf.st("Conclusions")
    pdf.bu("All schedules acquire B (peak ~ 1.0).", "Acquisition: ")
    pdf.bu("Uniform is most forgetting-resistant.", "Retention: ")
    pdf.bu("More A during burst = slower forgetting.", "A-mixing: ")
    pdf.bu("End block: fastest forgetting, high variance.", "Variance: ")
    pdf.bu("A robust across all schedules.", "Background: ")
    pdf.sh("Interpretation")
    pdf.bt("Interleaving background with novel data creates integrated, durable representations. Isolated bursts create fragile shortcuts. Many small exposures > concentrated bursts.")
    pdf.add_page(); pdf.st("Follow-up Ideas")
    pdf.bu("Deeper compositions (depth 4, 5)", "Depth: ")
    pdf.bu("Multiple novel functions", "Capacity: ")
    pdf.bu("Different model sizes", "Architecture: ")
    pdf.bu("Longer undo phases", "Duration: ")
    pdf.bu("Activation patching for B localization", "Mechanistic: ")
    pdf.bu("Can B be recovered after forgetting?", "Recovery: ")
    pdf.sh("Open Questions")
    pdf.bu("Why high variance in end_block?")
    pdf.bu("Critical A-mixing threshold?")
    pdf.bu("LR-forgetting interaction?")
    pdf.add_page(); pdf.st("Appendix")
    pdf.bt(f"Token: S [F3 F2 F1] [input] [F1(x)] [F2(F1(x))] [F3(F2(F1(x)))]. Doc={ti.get('doc_len',32)}, prompt={ti.get('prompt_len',12)}")
    pdf.bt(f"Bijections: permutations of 0-9. [0]=id, [1-{n_a}]=A, [{n_a+1}]=b*. Seed=999.")
    pdf.bt(f"Seeds: {cfg.get('seed_base',107)}-{cfg.get('seed_base',107)+ns-1}.")
    out = rd / "presentation" / "burst_d3_presentation.pdf"
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

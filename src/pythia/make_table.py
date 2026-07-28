"""Compute forgetting tables for each domain.

Outputs two LaTeX tables per domain:
1. Raw losses: Post-FT and Post-CPT domain/pile validation loss
2. Relative forgetting ratio: ℓ_CPT / ℓ_FT - 1

Also prints config differences across runs.

Usage:
    python make_table.py
"""

import json

RUNS = {
    # "Music": "results/20260411_145359_deepest_music",
    # "Biomedical": "results/20260417_111654_deepest_70m_biomedical",
    # "Chemistry": "results/20260418_095721_deepest_70m_chemistry",
    "Music": "results/20260422_191411_deep_70m_biomedical",
    "Biomedical": "results/20260410_082434_deepest",
    "Chemistry": "results/20260423_071726_deep_70m_music",
}

DOMAIN_LABELS = {
    "Music": "Music (ABC notation, Irishman)",
    "Biomedical": "Biomedical (PubMed articles)",
    "Chemistry": "Chemistry (SMILES, ChEMBL)",
}

TARGET_BURSTS = [1.0, 0.95, 0.90, 0.75, 0.50, 0.25]

CONFIG_KEYS = ["ft_steps", "cpt_steps", "ft_lr", "cpt_lr", "ft_budget_mode",
               "ft_batch_size", "ft_grad_accum", "cpt_batch_size", "cpt_grad_accum",
               "ft_warmup_steps", "cpt_warmup_steps", "seq_length", "eval_every",
               "max_grad_norm", "seed"]


def load_run(path):
    with open(f"{path}/metrics.json") as f:
        metrics = json.load(f)
    with open(f"{path}/config.json") as f:
        cfg = json.load(f)

    baseline = None
    post_ft = {}
    cpt_last = {}
    for r in metrics:
        bl = r.get("burst_level")
        if r["phase"] == "pretrained":
            baseline = r
        elif r["phase"] == "post_finetune":
            post_ft[bl] = r
        elif r["phase"] == "continued_pretraining":
            if bl not in cpt_last or r["step"] > cpt_last[bl]["step"]:
                cpt_last[bl] = r

    return baseline, post_ft, cpt_last, cfg


def fmt_k(v):
    """Format a number as e.g. '15k' if >= 1000, else as-is."""
    if isinstance(v, int) and v >= 1000:
        return f"{v // 1000}k"
    return str(v)


def print_loss_table(domain, baseline, post_ft, cpt_last, cfg):
    """Print a per-domain loss table in LaTeX."""
    label_safe = domain.lower().replace(" ", "-")
    long_label = DOMAIN_LABELS.get(domain, domain)

    ft_steps = fmt_k(cfg.get("ft_steps", "?"))
    cpt_steps = fmt_k(cfg.get("cpt_steps", "?"))

    print(r"\begin{table}[H]")
    print(r"  \centering\small")
    print(r"  \begin{tabular}{lcccc}")
    print(r"  \toprule")
    print(r"  & \multicolumn{2}{c}{Post-FT} & \multicolumn{2}{c}{Post-CPT} \\")
    print(r"  $c$ & Domain & Pile & Domain & Pile \\")
    print(r"  \midrule")

    if baseline:
        d = baseline["domain_val_loss"]
        p = baseline["pile_val_loss"]
        print(f"  Pretrained & {d:.2f} & {p:.2f} & --- & --- \\\\")
        print(r"  \midrule")

    for b in TARGET_BURSTS:
        ft = post_ft.get(b)
        cpt = cpt_last.get(b)
        if ft and cpt:
            print(f"  {b:.2f} & {ft['domain_val_loss']:.2f} & {ft['pile_val_loss']:.2f}"
                  f" & {cpt['domain_val_loss']:.2f} & {cpt['pile_val_loss']:.2f} \\\\")
        elif ft:
            print(f"  {b:.2f} & {ft['domain_val_loss']:.2f} & {ft['pile_val_loss']:.2f}"
                  f" & --- & --- \\\\")

    print(r"  \bottomrule")
    print(r"  \end{tabular}")
    print(f"  \\caption{{Validation cross-entropy loss for Pythia-70M after fine-tuning "
          f"(Post-FT) on {long_label} and after continued pretraining on the Pile (Post-CPT). "
          f"FT\\,=\\,{ft_steps} steps, CPT\\,=\\,{cpt_steps} steps.}}")
    print(f"  \\label{{tab:llm-{label_safe}-loss}}")
    print(r"\end{table}")


def print_ratio_table(domains, all_data, all_configs):
    """Print the combined relative-forgetting ratio table."""
    # Identify differing config keys
    diff_keys = []
    for key in CONFIG_KEYS:
        vals = [str(all_configs[d].get(key, "")) for d in domains]
        if len(set(vals)) > 1:
            diff_keys.append(key)

    print(r"\begin{table}[H]")
    print(r"  \centering\small")
    print(r"  \begin{tabular}{l" + "c" * len(domains) + "}")
    print(r"  \toprule")
    print("  $c$ & " + " & ".join(domains) + r" \\")

    if diff_keys:
        print(r"  \midrule")
        cells = []
        for d in domains:
            c = all_configs[d]
            parts = []
            if "ft_steps" in diff_keys:
                parts.append(f"FT={fmt_k(c['ft_steps'])}")
            if "cpt_steps" in diff_keys:
                parts.append(f"CPT={fmt_k(c['cpt_steps'])}")
            for key in diff_keys:
                if key not in ("ft_steps", "cpt_steps"):
                    parts.append(f"{key}={c.get(key, '?')}")
            cells.append(r"\scriptsize " + ", ".join(parts))
        print(r"  \multicolumn{1}{l}{} & " + " & ".join(cells) + r" \\")

    print(r"  \midrule")
    for b in TARGET_BURSTS:
        cells = [f"{b:.2f}"]
        for d in domains:
            val = all_data[d].get(b)
            cells.append(f"{val:.2f}" if val is not None else "---")
        print("  " + " & ".join(cells) + r" \\")
    print(r"  \bottomrule")
    print(r"  \end{tabular}")
    print(r"  \caption{Relative forgetting: $\ell_{\text{CPT}} / \ell_{\text{FT}} - 1$, "
          r"where $\ell_{\text{FT}}$ is the domain validation loss at the end of "
          r"fine-tuning and $\ell_{\text{CPT}}$ at the end of continued pretraining. "
          r"Higher values indicate more forgetting. Pythia-70M, budget mode\,=\,\emph{steps}, "
          r"lr\,=\,$5{\times}10^{-5}$, seq\_len\,=\,512.}")
    print(r"  \label{tab:llm-forgetting-ratio}")
    print(r"\end{table}")


def main():
    domains = list(RUNS.keys())
    all_data = {}
    all_configs = {}
    all_baselines = {}
    all_post_ft = {}
    all_cpt_last = {}

    for domain, path in RUNS.items():
        baseline, post_ft, cpt_last, cfg = load_run(path)
        all_configs[domain] = cfg
        all_baselines[domain] = baseline
        all_post_ft[domain] = post_ft
        all_cpt_last[domain] = cpt_last
        ratios = {}
        for b in TARGET_BURSTS:
            ft = post_ft.get(b)
            cpt = cpt_last.get(b)
            if ft and cpt:
                ratios[b] = (cpt["domain_val_loss"] / ft["domain_val_loss"]) - 1
        all_data[domain] = ratios

    # ── Config comparison (plain text) ──
    print("=" * 60)
    print("CONFIG COMPARISON")
    print("=" * 60)
    for key in CONFIG_KEYS:
        vals = [all_configs[d].get(key, "—") for d in domains]
        if len(set(str(v) for v in vals)) == 1:
            print(f"  {key}: {vals[0]} (same)")
        else:
            parts = "  /  ".join(f"{d}: {v}" for d, v in zip(domains, vals))
            print(f"  {key}: {parts}")

    # ── Ratio table (plain text) ──
    print()
    header = f"{'c':>6}" + "".join(f" {d:>12}" for d in domains)
    print(header)
    print("-" * len(header))
    for b in TARGET_BURSTS:
        row = f"{b:>6.2f}"
        for d in domains:
            val = all_data[d].get(b)
            row += f" {val:>12.4f}" if val is not None else f" {'---':>12}"
        print(row)

    # ── LaTeX: per-domain loss tables ──
    print("\n\n% ══════════════════════════════════════════════════════════")
    print("% Per-domain loss tables (copy-paste ready)")
    print("% ══════════════════════════════════════════════════════════")
    for domain in domains:
        print()
        print_loss_table(domain, all_baselines[domain],
                         all_post_ft[domain], all_cpt_last[domain],
                         all_configs[domain])

    # ── LaTeX: combined ratio table ──
    print("\n\n% ══════════════════════════════════════════════════════════")
    print("% Combined relative-forgetting ratio table")
    print("% ══════════════════════════════════════════════════════════")
    print()
    print_ratio_table(domains, all_data, all_configs)


if __name__ == "__main__":
    main()

"""
Count-A weighted accuracy over finetune->reverse training — one column per correlation.

Run from the PCFG directory:
    python plot_count_a_acc_over_training.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

sys.path.insert(0, os.path.dirname(__file__))

from style import apply_paper_style, save_figure, FULL_COL_WIDTH
from plot_helpers import (
    RESULTS_DIR, HISTORIES_DIR,
    CORRS, CONCS, CONC_COLORS,
    load_histories, load_pretrain_metrics,
    get_weighted_vals, WEIGHTED_SPLITS,
)

PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

apply_paper_style()
plt.rcParams.update({
    'font.size': 7,
    'axes.titlesize': 7,
    'axes.labelsize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
})

histories = load_histories(HISTORIES_DIR)
pretrain_accs, _ = load_pretrain_metrics(HISTORIES_DIR)

fig, axes = plt.subplots(
    1, len(CORRS),
    figsize=(FULL_COL_WIDTH * 2, FULL_COL_WIDTH * 2 / (len(CORRS) / 1.618)),
    sharey=True,
)

boundary_step = None
split_key = 'count_a_avg'
corr_split, uncorr_split = WEIGHTED_SPLITS[split_key]

for col, corr in enumerate(CORRS):
    ax = axes[col]
    for conc, color in zip(CONCS, CONC_COLORS):
        ft_key = f'finetune_corr_{corr:.2f}_conc_{conc:.2f}'
        rv_key = f'reverse_corr_{corr:.2f}_conc_{conc:.2f}'
        if ft_key not in histories or rv_key not in histories:
            continue
        ft_vals = get_weighted_vals(histories[ft_key], split_key, corr)
        rv_vals = get_weighted_vals(histories[rv_key], split_key, corr)
        if ft_vals is None or rv_vals is None:
            continue
        ft_steps = histories[ft_key]['steps']
        ft_last = ft_steps[-1]
        rv_steps = [ft_last + s for s in histories[rv_key]['steps']]
        if boundary_step is None:
            boundary_step = ft_last
        steps = list(ft_steps) + list(rv_steps)
        vals = list(ft_vals) + list(rv_vals)
        ax.plot(steps, vals, color=color, linewidth=0.8, label=f'conc={conc:.2f}')

    if corr in pretrain_accs:
        if corr_split in pretrain_accs[corr]:
            pval = (corr * pretrain_accs[corr][corr_split]
                    + (1 - corr) * pretrain_accs[corr][uncorr_split])
        else:
            pval = pretrain_accs[corr].get(split_key, None)
        if pval is not None:
            ax.plot(0, pval, marker='*', color='red', markersize=7, zorder=5,
                    markeredgecolor='darkred', markeredgewidth=0.4)

    if boundary_step is not None:
        ax.axvline(boundary_step, color='black', linestyle='--', alpha=0.4, linewidth=0.6)

    ax.grid(True, alpha=0.25)
    ax.set_title(f'corr = {corr:.2f}')
    ax.set_xlabel('Step')
    if col == 0:
        ax.set_ylabel('Count-A weighted acc')

    tick_vals = ax.get_xticks()
    ax.set_xticks(tick_vals)
    ax.set_xticklabels([f'{int(v/1000)}k' if v >= 1000 else str(int(v))
                        for v in tick_vals], rotation=45)

# Shared legend
conc_handles = [
    mlines.Line2D([], [], color=c, linewidth=1.0, label=f'conc={conc:.2f}')
    for conc, c in zip(CONCS, CONC_COLORS)
]
pretrain_handle = mlines.Line2D([], [], marker='*', color='red', markersize=6,
                                markeredgecolor='darkred', linestyle='None',
                                label='pretrained')
fig.legend(handles=conc_handles + [pretrain_handle],
           loc='lower center', ncol=len(CONCS) + 1,
           bbox_to_anchor=(0.5, -0.12))

out_path = os.path.join(PLOTS_DIR, 'count_a_acc_over_training')
save_figure(fig, out_path)
print(f'Saved -> {out_path}.pdf')

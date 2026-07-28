"""
Generate all metric plots and save them to results/plots/.

Run from the PCFG directory:
    python plot_metrics.py
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

sys.path.insert(0, os.path.dirname(__file__))

from plot_helpers import (
    RESULTS_DIR, HISTORIES_DIR, CORRS, CONCS, CONC_COLORS, WEIGHTED_SPLITS,
    load_histories, load_pretrain_metrics,
    plot_summary_heatmaps, plot_metric_grid,
    get_weighted_vals,
)

PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)


def save(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  Saved → {path}')


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

histories = load_histories(HISTORIES_DIR)
pretrain_accs, pretrain_losses = load_pretrain_metrics(HISTORIES_DIR)

# ---------------------------------------------------------------------------
# 1. Summary heatmaps — finetune final accuracy / loss
# ---------------------------------------------------------------------------

fig = plot_summary_heatmaps(
    phase='finetune',
    specs=[
        ('count_a_avg', 'answer_acc', 'Count-A acc',  'YlGn',   True),
        ('count_b_avg', 'answer_acc', 'Count-B acc',  'YlGn',   True),
        ('all_other_avg', 'answer_acc', 'Other acc',  'YlGn',   True),
    ],
    suptitle='Finetune — final accuracy (corr-weighted)',
    histories=histories,
)
save(fig, 'finetune_final_accuracy.png')

fig = plot_summary_heatmaps(
    phase='finetune',
    specs=[
        ('count_a_avg', 'loss', 'Count-A loss',   'YlOrRd', True),
        ('count_b_avg', 'loss', 'Count-B loss',   'YlOrRd', True),
        ('all_other_avg', 'loss', 'Other loss',   'YlOrRd', True),
    ],
    suptitle='Finetune — final loss (corr-weighted)',
    histories=histories,
)
save(fig, 'finetune_final_loss.png')

# ---------------------------------------------------------------------------
# 2. Summary heatmaps — reverse final accuracy / loss
# ---------------------------------------------------------------------------

fig = plot_summary_heatmaps(
    phase='reverse',
    specs=[
        ('count_a_avg', 'answer_acc', 'Count-A acc', 'YlGn', True),
        ('count_b_avg', 'answer_acc', 'Count-B acc', 'YlGn', True),
        ('all_other_avg', 'answer_acc', 'Other acc', 'YlGn', True),
    ],
    suptitle='Reverse — final accuracy (corr-weighted)',
    histories=histories,
)
save(fig, 'reverse_final_accuracy.png')

fig = plot_summary_heatmaps(
    phase='reverse',
    specs=[
        ('count_a_avg', 'loss', 'Count-A loss',   'RdBu', True),
        ('count_b_avg', 'loss', 'Count-B loss',   'RdBu', True),
        ('all_other_avg', 'loss', 'Other loss',   'RdBu', True),
    ],
    suptitle='Reverse — final loss (corr-weighted)',
    histories=histories,
)
save(fig, 'reverse_final_loss.png')

# ---------------------------------------------------------------------------
# 3. Accuracy curves: finetune → reverse, 3 metrics × N_CORRS
# ---------------------------------------------------------------------------

row_specs_acc = [
    ('count_a_avg',   'Count-A accuracy\n(weighted avg)'),
    ('count_b_avg',   'Count-B accuracy\n(weighted avg)'),
    ('all_other_avg', 'All other tasks accuracy\n(weighted avg)'),
]

fig, axes = plt.subplots(len(row_specs_acc), len(CORRS), figsize=(22, 12), sharey='row')
fig.suptitle('Finetune → Reverse — accuracy over training (corr-weighted averages)',
             fontsize=14, fontweight='bold')

boundary_step = None
for row, (split_key, ylabel) in enumerate(row_specs_acc):
    for col, corr in enumerate(CORRS):
        ax = axes[row, col]
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
            ft_last  = ft_steps[-1]
            rv_steps = [ft_last + s for s in histories[rv_key]['steps']]
            if boundary_step is None:
                boundary_step = ft_last
            steps = list(ft_steps) + list(rv_steps)
            vals  = list(ft_vals)  + list(rv_vals)
            ax.plot(steps, vals, color=color, linewidth=1.5, label=f'conc={conc:.1f}')

        if corr in pretrain_accs:
            corr_split, uncorr_split = WEIGHTED_SPLITS[split_key]
            if corr_split in pretrain_accs[corr]:
                pval = (corr * pretrain_accs[corr][corr_split] +
                        (1 - corr) * pretrain_accs[corr][uncorr_split])
            else:
                pval = pretrain_accs[corr].get(split_key, None)
            if pval is not None:
                ax.plot(0, pval, marker='*', color='red', markersize=14, zorder=5,
                        markeredgecolor='darkred', markeredgewidth=0.5)

        if boundary_step is not None:
            ax.axvline(boundary_step, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax.grid(True, alpha=0.3)
        if row == 0:
            ax.set_title(f'corr = {corr:.2f}', fontsize=11)
        if row == len(row_specs_acc) - 1:
            ax.set_xlabel('Step')
        if col == 0:
            ax.set_ylabel(ylabel)

handles, labels = axes[0, 0].get_legend_handles_labels()
pretrain_handle = mlines.Line2D([], [], marker='*', color='red', markersize=12,
                                markeredgecolor='darkred', linestyle='None',
                                label='pretrained')
handles.append(pretrain_handle)
labels.append('pretrained')
fig.legend(handles, labels, loc='lower center', ncol=len(CONCS) + 1,
           fontsize=9, bbox_to_anchor=(0.5, -0.04))
plt.tight_layout()
save(fig, 'accuracy_over_training.png')

# ---------------------------------------------------------------------------
# 4. Loss curves: finetune → reverse, 3 metrics × N_CORRS
# ---------------------------------------------------------------------------

row_specs_loss = [
    ('count_a_avg',   'Count-A loss\n(weighted avg)'),
    ('count_b_avg',   'Count-B loss\n(weighted avg)'),
    ('all_other_avg', 'All other tasks loss\n(weighted avg)'),
]

boundary_step = None
fig, axes = plt.subplots(len(row_specs_loss), len(CORRS), figsize=(22, 12), sharey='row')
fig.suptitle('Finetune → Reverse — loss over training (corr-weighted averages)',
             fontsize=14, fontweight='bold')

for row, (split_key, ylabel) in enumerate(row_specs_loss):
    for col, corr in enumerate(CORRS):
        ax = axes[row, col]
        for conc, color in zip(CONCS, CONC_COLORS):
            ft_key = f'finetune_corr_{corr:.2f}_conc_{conc:.2f}'
            rv_key = f'reverse_corr_{corr:.2f}_conc_{conc:.2f}'
            if ft_key not in histories or rv_key not in histories:
                continue
            ft_vals = get_weighted_vals(histories[ft_key], split_key, corr, metric='loss')
            rv_vals = get_weighted_vals(histories[rv_key], split_key, corr, metric='loss')
            if ft_vals is None or rv_vals is None:
                continue
            ft_steps = histories[ft_key]['steps']
            ft_last  = ft_steps[-1]
            rv_steps = [ft_last + s for s in histories[rv_key]['steps']]
            if boundary_step is None:
                boundary_step = ft_last
            steps = list(ft_steps) + list(rv_steps)
            vals  = list(ft_vals)  + list(rv_vals)
            ax.plot(steps, vals, color=color, linewidth=1.5, label=f'conc={conc:.1f}')

        if corr in pretrain_losses:
            corr_split, uncorr_split = WEIGHTED_SPLITS[split_key]
            if corr_split in pretrain_losses[corr]:
                pval = (corr * pretrain_losses[corr][corr_split] +
                        (1 - corr) * pretrain_losses[corr][uncorr_split])
            else:
                pval = pretrain_losses[corr].get(split_key, None)
            if pval is not None:
                ax.plot(0, pval, marker='*', color='red', markersize=14, zorder=5,
                        markeredgecolor='darkred', markeredgewidth=0.5)

        if boundary_step is not None:
            ax.axvline(boundary_step, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        if row == 0:
            ax.set_title(f'corr = {corr:.2f}', fontsize=11)
        if row == len(row_specs_loss) - 1:
            ax.set_xlabel('Step')
        if col == 0:
            ax.set_ylabel(ylabel)

handles, labels = axes[0, 0].get_legend_handles_labels()
pretrain_handle = mlines.Line2D([], [], marker='*', color='red', markersize=12,
                                markeredgecolor='darkred', linestyle='None',
                                label='pretrained')
handles.append(pretrain_handle)
labels.append('pretrained')
fig.legend(handles, labels, loc='lower center', ncol=len(CONCS) + 1,
           fontsize=9, bbox_to_anchor=(0.5, -0.04))
plt.tight_layout()
save(fig, 'loss_over_training.png')

# ---------------------------------------------------------------------------
# 5. Metric grids (layers × steps for every (corr, conc))
# ---------------------------------------------------------------------------

fig = plot_metric_grid(
    metric_path=('grad_count_a_vs_count_b', 'dot_product'),
    title='Dot product: ∇count_a · ∇count_b (per layer over training)',
    histories=histories,
    cmap='RdBu_r', symmetric=True, phase='both',
)
if fig:
    save(fig, 'grad_dot_product_a_vs_b.png')

fig = plot_metric_grid(
    metric_path=('grad_count_a_vs_all_other', 'dot_product'),
    title='Dot product: ∇count_a · ∇all_other (per layer over training)',
    histories=histories,
    cmap='RdBu_r', symmetric=True, phase='both',
)
if fig:
    save(fig, 'grad_dot_product_a_vs_other.png')

fig = plot_metric_grid(
    metric_path=('layerwise_drift',),
    title='Layerwise drift (||ΔW|| per layer between eval steps)',
    histories=histories,
    cmap='hot', symmetric=False, phase='finetune', same_vmax=False,
)
if fig:
    save(fig, 'layerwise_drift.png')

fig = plot_metric_grid(
    metric_path=('grad_count_a_vs_count_b', 'norm_a'),
    title='||∇L_count_a|| per layer over training',
    histories=histories,
    cmap='YlOrRd', symmetric=False, phase='both', same_vmax=True,
)
if fig:
    save(fig, 'grad_norm_a.png')

fig = plot_metric_grid(
    metric_path=('grad_count_a_vs_count_b', 'norm_b'),
    title='||∇L_count_b|| per layer over training',
    histories=histories,
    cmap='YlOrRd', symmetric=False, phase='both', same_vmax=True,
)
if fig:
    save(fig, 'grad_norm_b.png')

print(f'\nAll plots saved to {PLOTS_DIR}')

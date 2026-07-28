"""
Shared constants and helper functions for PCFG metric plots.

Import this in plot_metrics.py or directly in analyze_metrics.ipynb.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch

from config import CFG

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR   = '/workspace/PCFG/results'
HISTORIES_DIR = f'{RESULTS_DIR}/histories'

mcfg     = CFG['model']
N_LAYERS = mcfg['n_layer']

CORRS = [0.0, 0.25, 0.5, 0.66, 0.75, 0.85, 0.92, 0.95, 1.0]
CONCS = [0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.0]
SEEDS = [124]

CORR_PALETTE = {
    0.00: '#e74c3c',
    0.25: '#e67e22',
    0.50: '#f1c40f',
    0.66: '#a3be8c',
    0.75: '#2ecc71',
    0.85: '#3498db',
    0.92: '#2f009d',
    0.95: '#aa00ff',
    1.00: '#ff00e1',
}

CONC_ALPHA = {c: 0.25 + 0.75 * (c - min(CONCS)) / (max(CONCS) - min(CONCS))
              for c in CONCS}

CONC_COLORS = plt.cm.viridis(np.linspace(0.1, 0.9, len(CONCS)))

VAL_SPLITS = [
    'count_a_corr', 'count_a_uncorr',
    'count_b_corr', 'count_b_uncorr',
    'all_other_corr', 'all_other_uncorr',
]

SPLIT_LABELS = {
    'count_a_corr':   'Count-A (correlated)',
    'count_a_uncorr': 'Count-A (uncorrelated)',
    'count_b_corr':   'Count-B (correlated)',
    'count_b_uncorr': 'Count-B (uncorrelated)',
    'all_other_corr':   'All other (correlated)',
    'all_other_uncorr': 'All other (uncorrelated)',
}

WEIGHTED_SPLITS = {
    'count_a_avg':   ('count_a_corr',   'count_a_uncorr'),
    'count_b_avg':   ('count_b_corr',   'count_b_uncorr'),
    'all_other_avg': ('all_other_corr', 'all_other_uncorr'),
}

# ---------------------------------------------------------------------------
# History loading
# ---------------------------------------------------------------------------

def _avg_histories(hist_list):
    """Recursively average a list of history dicts element-wise."""
    if not hist_list:
        return {}
    first = hist_list[0]
    if isinstance(first, dict):
        out = {}
        for k in first:
            children = [h[k] for h in hist_list if k in h]
            if children:
                out[k] = _avg_histories(children)
        return out
    if isinstance(first, list):
        try:
            arrs = [np.array(h, dtype=float) for h in hist_list]
            return np.mean(arrs, axis=0).tolist()
        except (ValueError, TypeError):
            return first
    return first


def load_histories(histories_dir=HISTORIES_DIR, seeds=SEEDS):
    """Load and seed-average all history .pth files in histories_dir."""
    _per_seed = {}
    for seed in seeds:
        for path in sorted(glob.glob(f'{histories_dir}/*_seed{seed}_history.pth')):
            fname     = os.path.basename(path).replace('_history.pth', '')
            canonical = fname.replace(f'_seed{seed}', '')
            h = torch.load(path, map_location='cpu', weights_only=False)
            _per_seed.setdefault(canonical, []).append(h)

    histories = {}
    for name, hlist in _per_seed.items():
        histories[name] = _avg_histories(hlist) if len(hlist) > 1 else hlist[0]

    print(f'Loaded {sum(len(v) for v in _per_seed.values())} history files '
          f'across {len(seeds)} seed(s), averaged into {len(histories)} canonical histories.')
    return histories


def load_pretrain_metrics(histories_dir=HISTORIES_DIR):
    """Return (pretrain_accs, pretrain_losses) dicts keyed by correlation value."""
    pretrain_accs   = {}
    pretrain_losses = {}
    for path in sorted(glob.glob(f'{histories_dir}/pretrain_corr_*_history.pth')):
        fname    = os.path.basename(path)
        parts    = fname.replace('_history.pth', '').split('_')
        corr_val = float(parts[parts.index('corr') + 1])
        h = torch.load(path, map_location='cpu', weights_only=False)
        accs, losses = {}, {}
        for split_name, split_data in h['val'].items():
            if 'answer_acc' in split_data:
                accs[split_name]   = split_data['answer_acc'][-1]
            if 'loss' in split_data:
                losses[split_name] = split_data['loss'][-1]
        pretrain_accs[corr_val]   = accs
        pretrain_losses[corr_val] = losses
    return pretrain_accs, pretrain_losses

# ---------------------------------------------------------------------------
# Value helpers
# ---------------------------------------------------------------------------

def _get_vals(hist, split, metric, corr, use_weighted):
    """Get values from history, optionally using corr-weighted average."""
    if use_weighted and split in WEIGHTED_SPLITS:
        corr_split, uncorr_split = WEIGHTED_SPLITS[split]
        if corr_split in hist['val']:
            return (corr * np.array(hist['val'][corr_split][metric]) +
                    (1 - corr) * np.array(hist['val'][uncorr_split][metric]))
        if split in hist['val']:
            return np.array(hist['val'][split][metric])
        return None
    if split in hist['val']:
        return np.array(hist['val'][split][metric])
    return None


def get_weighted_vals(hist, weighted_key, corr, metric='answer_acc'):
    """Get corr-weighted average from history, with fallback for old key names."""
    corr_split, uncorr_split = WEIGHTED_SPLITS[weighted_key]
    if corr_split in hist['val']:
        return (corr * np.array(hist['val'][corr_split][metric]) +
                (1 - corr) * np.array(hist['val'][uncorr_split][metric]))
    old_key = weighted_key.replace('_avg', '') if weighted_key != 'all_other_avg' else 'all_other_avg'
    if old_key in hist['val']:
        return np.array(hist['val'][old_key][metric])
    old_key2 = weighted_key.replace('_avg', '')
    if old_key2 in hist['val']:
        return np.array(hist['val'][old_key2][metric])
    return None

# ---------------------------------------------------------------------------
# Summary heatmaps
# ---------------------------------------------------------------------------

def _final_val(hist, split, metric, corr, use_weighted=True):
    vals = _get_vals(hist, split, metric, corr, use_weighted)
    if vals is None or len(vals) == 0:
        return np.nan
    return vals[-1]


def build_final_grid(phase, split, metric, histories, use_weighted=True):
    """Build a (len(CORRS), len(CONCS)) array of final metric values."""
    grid = np.full((len(CORRS), len(CONCS)), np.nan)
    for i, corr in enumerate(CORRS):
        for j, conc in enumerate(CONCS):
            key = f'{phase}_corr_{corr:.2f}_conc_{conc:.2f}'
            if key not in histories:
                continue
            grid[i, j] = _final_val(histories[key], split, metric, corr, use_weighted)
    return grid


def plot_summary_heatmaps(phase, specs, suptitle, histories, figsize=None, fmt='.3f'):
    """
    Plot a row of summary heatmaps (one per spec).

    specs: list of (split, metric, title, cmap, use_weighted) tuples.
    Returns the figure.
    """
    n = len(specs)
    if figsize is None:
        figsize = (5.5 * n, 0.9 * len(CORRS) + 2)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    fig.suptitle(suptitle, fontsize=14, fontweight='bold')

    for ax, (split, metric, title, cmap, use_weighted) in zip(axes, specs):
        grid = build_final_grid(phase, split, metric, histories, use_weighted)
        im = ax.imshow(grid, aspect='auto', cmap=cmap, interpolation='nearest')
        for i in range(len(CORRS)):
            for j in range(len(CONCS)):
                v = grid[i, j]
                if not np.isnan(v):
                    mean_v = grid[~np.isnan(grid)].mean()
                    ax.text(j, i, f'{v:{fmt}}', ha='center', va='center', fontsize=7,
                            color='white' if v < mean_v else 'black')
        ax.set_xticks(range(len(CONCS)))
        ax.set_xticklabels([f'{c:.2f}' for c in CONCS], fontsize=7, rotation=45)
        ax.set_yticks(range(len(CORRS)))
        ax.set_yticklabels([f'{c:.2f}' for c in CORRS], fontsize=8)
        ax.set_xlabel('Concentration')
        if ax is axes[0]:
            ax.set_ylabel('Correlation')
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.04)

    plt.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# Layer heatmap helpers
# ---------------------------------------------------------------------------

def extract_layer_timeseries(history, metric_path, n_layers=N_LAYERS):
    """
    Extract a (n_layers, n_steps) array from a nested history dict.
    metric_path: e.g. ('grad_count_a_vs_count_b', 'cosine_sim')
    """
    d = history
    for key in metric_path:
        d = d[key]
    n_steps = len(d[0])
    arr = np.zeros((n_layers, n_steps))
    for l in range(n_layers):
        arr[l] = d[l]
    return arr


def plot_layer_heatmap(ax, arr, steps, title='', cmap='RdBu_r',
                       symmetric=False, vmin=None, vmax=None):
    """Plot a (layers x steps) heatmap on the given axes."""
    if symmetric:
        vabs = max(abs(arr.min()), abs(arr.max()))
        vmin, vmax = -vabs, vabs
    elif vmin is None:
        vmin, vmax = arr.min(), arr.max()
    im = ax.imshow(arr, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation='nearest', origin='lower')
    ax.set_yticks(range(arr.shape[0]))
    ax.set_yticklabels([f'L{l}' for l in range(arr.shape[0])], fontsize=8)
    n_ticks = min(8, len(steps))
    tick_idx = np.linspace(0, len(steps) - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([str(steps[i]) for i in tick_idx], fontsize=7, rotation=45)
    ax.set_title(title, fontsize=9)
    return im

# ---------------------------------------------------------------------------
# Per-model stacked heatmap
# ---------------------------------------------------------------------------

def plot_stacked_heatmaps_for_model(ft_key, rv_key, corr, conc, histories):
    """Plot stacked heatmaps for a single model's finetune + reverse trajectory."""
    ft_h = histories.get(ft_key)
    rv_h = histories.get(rv_key)
    if ft_h is None or rv_h is None:
        return
    if 'grad_count_a_vs_count_b' not in ft_h:
        return

    ft_steps = ft_h['steps']
    ft_last  = ft_steps[-1]
    rv_steps = [ft_last + s for s in rv_h['steps']]
    all_steps    = ft_steps + rv_steps
    boundary_idx = len(ft_steps)

    metric_specs = []

    if 'grad_count_a_vs_count_b' in ft_h and 'grad_count_a_vs_count_b' in rv_h:
        ft_cos = extract_layer_timeseries(ft_h, ('grad_count_a_vs_count_b', 'cosine_sim'))
        rv_cos = extract_layer_timeseries(rv_h, ('grad_count_a_vs_count_b', 'cosine_sim'))
        metric_specs.append(('Cosine sim (A vs B)', np.hstack([ft_cos, rv_cos]), 'RdBu_r', True))

        ft_dot = extract_layer_timeseries(ft_h, ('grad_count_a_vs_count_b', 'dot_product'))
        rv_dot = extract_layer_timeseries(rv_h, ('grad_count_a_vs_count_b', 'dot_product'))
        metric_specs.append(('Dot product (A vs B)', np.hstack([ft_dot, rv_dot]), 'RdBu_r', True))

        ft_na = extract_layer_timeseries(ft_h, ('grad_count_a_vs_count_b', 'norm_a'))
        rv_na = extract_layer_timeseries(rv_h, ('grad_count_a_vs_count_b', 'norm_a'))
        metric_specs.append(('||∇L_A|| (count_a)', np.hstack([ft_na, rv_na]), 'YlOrRd', False))

        ft_nb = extract_layer_timeseries(ft_h, ('grad_count_a_vs_count_b', 'norm_b'))
        rv_nb = extract_layer_timeseries(rv_h, ('grad_count_a_vs_count_b', 'norm_b'))
        metric_specs.append(('||∇L_B|| (count_b)', np.hstack([ft_nb, rv_nb]), 'YlOrRd', False))

    if 'grad_count_a_vs_all_other' in ft_h and 'grad_count_a_vs_all_other' in rv_h:
        ft_cos2 = extract_layer_timeseries(ft_h, ('grad_count_a_vs_all_other', 'cosine_sim'))
        rv_cos2 = extract_layer_timeseries(rv_h, ('grad_count_a_vs_all_other', 'cosine_sim'))
        metric_specs.append(('Cosine sim (A vs Other)', np.hstack([ft_cos2, rv_cos2]), 'RdBu_r', True))

    if 'layerwise_drift' in ft_h and 'layerwise_drift' in rv_h:
        ft_drift = extract_layer_timeseries(ft_h, ('layerwise_drift',))
        rv_drift = extract_layer_timeseries(rv_h, ('layerwise_drift',))
        metric_specs.append(('Layerwise drift', np.hstack([ft_drift, rv_drift]), 'hot', False))

    if not metric_specs:
        return

    n_metrics = len(metric_specs)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(14, 2.5 * n_metrics), sharex=True)
    if n_metrics == 1:
        axes = [axes]
    fig.suptitle(f'corr={corr:.2f}  conc={conc:.2f} — finetune → reverse',
                 fontsize=13, fontweight='bold')

    for ax, (title, arr, cmap, symmetric) in zip(axes, metric_specs):
        im = plot_layer_heatmap(ax, arr, all_steps, title=title, cmap=cmap, symmetric=symmetric)
        ax.axvline(boundary_idx - 0.5, color='white', linestyle='--', linewidth=1.5, alpha=0.8)
        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

    axes[-1].set_xlabel('Step')
    plt.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# Metric grid (layers × steps for every (corr, conc))
# ---------------------------------------------------------------------------

def plot_metric_grid(metric_path, title, histories, cmap='RdBu_r',
                     symmetric=False, phase='both', same_vmax=False):
    """
    Grid of (layers × steps) heatmaps: rows=correlations, cols=concentrations.
    phase: 'finetune', 'reverse', or 'both'.
    Returns the figure.
    """
    fig, axes = plt.subplots(len(CORRS), len(CONCS),
                             figsize=(3.5 * len(CONCS), 2.5 * len(CORRS)))
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    # First pass: global vmin/vmax
    all_vals = []
    for corr in CORRS:
        for conc in CONCS:
            for prefix in ['finetune', 'reverse']:
                key = f'{prefix}_corr_{corr:.2f}_conc_{conc:.2f}'
                if key in histories and metric_path[0] in histories[key]:
                    try:
                        all_vals.append(extract_layer_timeseries(histories[key], metric_path))
                    except (KeyError, TypeError):
                        pass
    if not all_vals:
        print(f'No data found for {metric_path}')
        plt.close(fig)
        return None

    all_concat = np.concatenate([a.flatten() for a in all_vals])
    if symmetric:
        vabs = max(abs(all_concat.min()), abs(all_concat.max()))
        gvmin, gvmax = -vabs, vabs
    else:
        gvmin, gvmax = all_concat.min(), all_concat.max()

    im = None
    for row, corr in enumerate(CORRS):
        for col, conc in enumerate(CONCS):
            ax = axes[row, col]
            ft_key = f'finetune_corr_{corr:.2f}_conc_{conc:.2f}'
            rv_key = f'reverse_corr_{corr:.2f}_conc_{conc:.2f}'

            arrs, steps_list, boundary_idx = [], [], None

            if phase in ('finetune', 'both') and ft_key in histories:
                try:
                    ft_arr = extract_layer_timeseries(histories[ft_key], metric_path)
                    arrs.append(ft_arr)
                    steps_list.extend(histories[ft_key]['steps'])
                    boundary_idx = ft_arr.shape[1]
                except (KeyError, TypeError):
                    pass

            if phase in ('reverse', 'both') and rv_key in histories:
                try:
                    rv_arr = extract_layer_timeseries(histories[rv_key], metric_path)
                    arrs.append(rv_arr)
                    offset = steps_list[-1] if steps_list else 0
                    steps_list.extend([offset + s for s in histories[rv_key]['steps']])
                except (KeyError, TypeError):
                    pass

            if not arrs:
                ax.set_visible(False)
                continue

            combined = np.hstack(arrs)
            if same_vmax:
                im = ax.imshow(combined, aspect='auto', cmap=cmap,
                               vmin=gvmin, vmax=gvmax,
                               interpolation='nearest', origin='lower')
            else:
                vmax = max(abs(combined.min()), abs(combined.max()))
                vmin = -vmax if symmetric else combined.min()
                im = ax.imshow(combined, aspect='auto', cmap=cmap,
                               vmin=vmin, vmax=vmax,
                               interpolation='nearest', origin='lower')

            if boundary_idx is not None and phase == 'both' and len(arrs) == 2:
                ax.axvline(boundary_idx - 0.5, color='white',
                           linestyle='--', linewidth=1, alpha=0.8)

            ax.set_yticks(range(N_LAYERS))
            ax.set_yticklabels([f'L{l}' for l in range(N_LAYERS)], fontsize=6)
            n_ticks = min(5, len(steps_list))
            tick_idx = np.linspace(0, len(steps_list) - 1, n_ticks, dtype=int)
            ax.set_xticks(tick_idx)
            ax.set_xticklabels([str(steps_list[i]) for i in tick_idx],
                               fontsize=6, rotation=45)
            if row == 0:
                ax.set_title(f'conc={conc:.2f}', fontsize=9)
            if col == 0:
                ax.set_ylabel(f'corr={corr:.2f}', fontsize=9)

    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02)
    plt.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# Line plots (layer-averaged metric over training)
# ---------------------------------------------------------------------------

def plot_metric_lines(metric_path, title, ylabel, histories, per_layer=False):
    """
    Line plots: one column per correlation, all concentrations overlaid.
    If per_layer=False: layer-averaged value over steps.
    If per_layer=True: one row per layer.
    Returns the figure.
    """
    n_rows = N_LAYERS if per_layer else 1
    fig, axes = plt.subplots(n_rows, len(CORRS),
                             figsize=(22, 3.5 * n_rows), sharey='row')
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(title, fontsize=14, fontweight='bold')

    for col, corr in enumerate(CORRS):
        for conc, color in zip(CONCS, CONC_COLORS):
            ft_key = f'finetune_corr_{corr:.2f}_conc_{conc:.2f}'
            rv_key = f'reverse_corr_{corr:.2f}_conc_{conc:.2f}'

            arrs, steps_list, boundary = [], [], None
            for hkey in [ft_key, rv_key]:
                if hkey not in histories:
                    continue
                try:
                    arr = extract_layer_timeseries(histories[hkey], metric_path)
                    arrs.append(arr)
                    h_steps = histories[hkey]['steps']
                    if steps_list:
                        boundary = steps_list[-1]
                        steps_list.extend([steps_list[-1] + s for s in h_steps])
                    else:
                        steps_list.extend(h_steps)
                except (KeyError, TypeError):
                    pass

            if not arrs:
                continue
            combined = np.hstack(arrs)

            if per_layer:
                for l in range(N_LAYERS):
                    ax = axes[l, col]
                    ax.plot(steps_list, combined[l], color=color, linewidth=1.2,
                            label=f'conc={conc:.1f}')
                    if boundary is not None:
                        ax.axvline(boundary, color='black', linestyle='--', alpha=0.4)
                    ax.axhline(0, color='grey', linestyle='-', alpha=0.2)
                    ax.grid(True, alpha=0.2)
                    if l == 0:
                        ax.set_title(f'corr={corr:.2f}', fontsize=10)
                    if col == 0:
                        ax.set_ylabel(f'Layer {l}')
                    if l == N_LAYERS - 1:
                        ax.set_xlabel('Step')
            else:
                ax = axes[0, col]
                mean_vals = combined.mean(axis=0)
                ax.plot(steps_list, mean_vals, color=color, linewidth=1.5,
                        label=f'conc={conc:.1f}')
                if boundary is not None:
                    ax.axvline(boundary, color='black', linestyle='--', alpha=0.4)
                ax.axhline(0, color='grey', linestyle='-', alpha=0.2)
                ax.grid(True, alpha=0.2)
                ax.set_title(f'corr={corr:.2f}', fontsize=10)
                if col == 0:
                    ax.set_ylabel(ylabel)
                ax.set_xlabel('Step')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(CONCS),
               fontsize=9, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout()
    return fig

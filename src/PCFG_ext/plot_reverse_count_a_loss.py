"""
Plot reverse-phase final Count-A loss heatmap with a log colour scale.

Run from the PCFG directory:
    python plot_reverse_count_a_loss.py
"""

import os
import sys
import numpy as np
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.dirname(__file__))

from style import apply_paper_style, figsize, save_figure
from plot_helpers import (
    RESULTS_DIR, HISTORIES_DIR,
    CORRS, CONCS,
    load_histories, build_final_grid,
)

import matplotlib.pyplot as plt

PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

apply_paper_style()
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
})

histories = load_histories(HISTORIES_DIR)

grid = build_final_grid('reverse', 'count_a_avg', 'loss', histories, use_weighted=True)

valid = grid[~np.isnan(grid)]
vmin = valid[valid > 0].min() if (valid > 0).any() else 1e-4
vmax = valid.max()
norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

fig, ax = plt.subplots(1, 1, figsize=(0.9 * len(CONCS), 0.7 * len(CORRS)))

im = ax.imshow(grid, aspect='auto', cmap='RdBu_r', norm=norm, interpolation='nearest')

cmap_obj = plt.get_cmap('RdBu_r')
for i in range(len(CORRS)):
    for j in range(len(CONCS)):
        v = grid[i, j]
        if not np.isnan(v) and v > 0:
            r, g, b, _ = cmap_obj(norm(v))
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=11,
                    color='white' if luminance < 0.6 else 'black')

ax.set_xticks(range(len(CONCS)))
ax.set_xticklabels([f'{c:.2f}' for c in CONCS], rotation=45)
ax.set_yticks(range(len(CORRS)))
ax.set_yticklabels([f'{c:.2f}' for c in CORRS])
ax.set_xlabel('Concentration')
ax.set_ylabel('Correlation')
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.8)
    spine.set_edgecolor('black')
# ax.set_title('Count-A loss (reverse, log scale)')

out_path = os.path.join(PLOTS_DIR, 'reverse_count_a_loss_log')
save_figure(fig, out_path)
print(f'Saved → {out_path}.pdf')

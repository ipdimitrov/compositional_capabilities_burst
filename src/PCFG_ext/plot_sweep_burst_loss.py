"""
Heatmap of final burst loss after the forget/reverse phase,
averaged across seeds, for the sweep_20260414_150029 experiment.
Analogous to reverse_count_a_loss_log.py.

Run from anywhere:
    python /workspace/PCFG/plot_sweep_burst_loss.py
"""

import os
import sys
import glob
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import apply_paper_style, save_figure

SWEEP_DIR = '/workspace/sweep_20260414_150029'
PLOTS_DIR = os.path.join(SWEEP_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---- Parse all fg files and collect final loss_burst ----
# filename pattern: corr{XX}_ftlr{lr}_frac{YY}_fg.pkl

pattern = os.path.join(SWEEP_DIR, 'seed_*', 'corr*_fg.pkl')
files = sorted(glob.glob(pattern))

# Accumulate per (corr_int, frac_int): list of final loss_burst values
data = {}
for path in files:
    fname = os.path.basename(path)
    m = re.match(r'corr(\d+)_ftlr[\de\-]+_frac(\d+)_fg\.pkl', fname)
    if not m:
        continue
    corr_int = int(m.group(1))
    frac_int = int(m.group(2))
    with open(path, 'rb') as f:
        d = pickle.load(f)
    log = d['log']
    if 'loss_burst' not in log or not log['loss_burst']:
        continue
    final_loss = log['loss_burst'][-1]
    key = (corr_int, frac_int)
    data.setdefault(key, []).append(final_loss)

# Seed-average
averaged = {k: float(np.mean(v)) for k, v in data.items()}

# Sorted axes
corr_ints = sorted({k[0] for k in averaged})
frac_ints = sorted({k[1] for k in averaged})

corr_labels = [f'{c/100:.2f}' for c in corr_ints]
frac_labels  = [f'{f/100:.2f}' for f in frac_ints]

# Build grid: rows = corr (ascending), cols = frac (ascending)
grid = np.full((len(corr_ints), len(frac_ints)), np.nan)
for i, c in enumerate(corr_ints):
    for j, f in enumerate(frac_ints):
        if (c, f) in averaged:
            grid[i, j] = averaged[(c, f)]

print(f'Grid shape: {grid.shape}  ({len(corr_ints)} corrs x {len(frac_ints)} fracs)')
print(f'Value range: {np.nanmin(grid):.4f} – {np.nanmax(grid):.4f}')

# ---- Plot ----
apply_paper_style()
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 15,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
})

valid = grid[~np.isnan(grid)]
vmin = (valid[valid > 0].min() if (valid > 0).any() else 1e-4) * 1.02
vmax = valid.max()
norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

fig, ax = plt.subplots(1, 1, figsize=(0.9 * len(frac_ints), 0.7 * len(corr_ints)))

im = ax.imshow(grid, aspect='auto', cmap='RdBu_r', norm=norm, interpolation='nearest')

cmap_obj = plt.get_cmap('RdBu_r')
for i in range(len(corr_ints)):
    for j in range(len(frac_ints)):
        v = grid[i, j]
        if not np.isnan(v) and v > 0:
            r, g, b, _ = cmap_obj(norm(v))
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=11,
                    color='white' if luminance < 0.6 else 'black')

ax.set_xticks(range(len(frac_ints)))
ax.set_xticklabels(frac_labels, rotation=45)
ax.set_yticks(range(len(corr_ints)))
ax.set_yticklabels(corr_labels)
ax.set_xlabel('Fraction (concentration)')
ax.set_ylabel('Correlation')

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.8)
    spine.set_edgecolor('black')

out_path = os.path.join(PLOTS_DIR, 'sweep_burst_loss_reverse')
save_figure(fig, out_path)
print(f'Saved -> {out_path}.pdf')

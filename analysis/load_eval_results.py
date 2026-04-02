"""Load evaluation pickles and plot autoregressive token accuracy over steps."""

# %%
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent

TEXT_CONTRAST_THRESHOLD = 0.4

# %%
# ============================================================
# In-order evaluation results
# ============================================================
inorder_path = PROJECT_ROOT / "data/inorder_eval_step_random50/accs.pkl"
with inorder_path.open("rb") as f:
    accs_inorder = pickle.load(f)  # noqa: S301

iterations = [entry[0][0] for entry in accs_inorder]

acc_mean, acc_std = [], []

for _, acc_map in accs_inorder:
    vals = np.array(list(acc_map.values()))
    acc_mean.append(vals.mean())
    acc_std.append(vals.std())

acc_mean = np.array(acc_mean)
acc_std = np.array(acc_std)

# %%
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(iterations, acc_mean, color="tab:blue", marker="o", markersize=4)
ax.fill_between(iterations, acc_mean - acc_std, acc_mean + acc_std, alpha=0.2, color="tab:blue")
ax.set_xlabel("Training Iteration")
ax.set_ylabel("Token Accuracy (Autoregressive)")
ax.set_ylim(-0.05, 1.05)
ax.set_title("In-Order Evaluation: Accuracy over Training")
ax.grid(visible=True, alpha=0.3)
fig.tight_layout()
plt.show()

# %%
# ============================================================
# Out-of-order evaluation results
# ============================================================
outorder_path = PROJECT_ROOT / "data/outorder_eval_step_random50/accs.pkl"
with outorder_path.open("rb") as f:
    accs_outorder = pickle.load(f)  # noqa: S301

all_keys = sorted(accs_outorder.keys())

num_ids = sorted({k[0] for k in all_keys})
displacements = sorted({k[1] for k in all_keys})

acc_grid = np.full((len(num_ids), len(displacements)), np.nan)

for (ni, disp), task_dict in accs_outorder.items():
    i = num_ids.index(ni)
    j = displacements.index(disp)
    task_accs = np.array([np.mean(v) for v in task_dict.values()])
    acc_grid[i, j] = task_accs.mean()

# %%
fig, ax = plt.subplots(figsize=(8, 5))

im = ax.imshow(acc_grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(displacements)))
ax.set_xticklabels(displacements)
ax.set_yticks(range(len(num_ids)))
ax.set_yticklabels(num_ids)
ax.set_xlabel("Displacement")
ax.set_ylabel("Num Identities")
ax.set_title("Token Accuracy (Autoregressive)")
for ii in range(len(num_ids)):
    for jj in range(len(displacements)):
        val = acc_grid[ii, jj]
        if not np.isnan(val):
            ax.text(
                jj, ii, f"{val:.2f}", ha="center", va="center", fontsize=8,
                color="black" if val > TEXT_CONTRAST_THRESHOLD else "white",
            )
plt.colorbar(im, ax=ax, shrink=0.8)

fig.suptitle(
    "Out-of-Order Evaluation: Accuracy by (Num Identities, Displacement)", fontsize=13, y=1.02,
)
fig.tight_layout()
plt.show()

# %%
fig, ax = plt.subplots(figsize=(8, 5))
for ni in num_ids:
    disps, means = [], []
    for disp in displacements:
        if (ni, disp) in accs_outorder:
            task_dict = accs_outorder[(ni, disp)]
            task_accs = np.array([np.mean(v) for v in task_dict.values()])
            disps.append(disp)
            means.append(task_accs.mean())
    ax.plot(disps, means, marker="o", markersize=5, label=f"Identities={ni}")

ax.set_xlabel("Displacement")
ax.set_ylabel("Mean Step Accuracy")
ax.set_title("Out-of-Order: Step Accuracy by Displacement (per Num Identities)")
ax.set_ylim(-0.05, 1.05)
ax.legend()
ax.grid(visible=True, alpha=0.3)
fig.tight_layout()
plt.show()

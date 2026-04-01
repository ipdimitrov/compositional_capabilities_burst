# %%
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent

# %%
# ============================================================
# In-order evaluation results
# ============================================================
inorder_path = PROJECT_ROOT / "data/inorder_eval_step_random50/accs.pkl"
with open(inorder_path, "rb") as f:
    accs_inorder = pickle.load(f)

iterations = [entry[0][0] for entry in accs_inorder]

token_acc_mean, token_acc_std = [], []
strict_acc_mean, strict_acc_std = [], []
teacher_acc_mean, teacher_acc_std = [], []

for _, acc_map in accs_inorder:
    vals = np.array(list(acc_map.values()))
    token_acc_mean.append(vals[:, 0].mean())
    token_acc_std.append(vals[:, 0].std())
    strict_acc_mean.append(vals[:, 1].mean())
    strict_acc_std.append(vals[:, 1].std())
    teacher_acc_mean.append(vals[:, 2].mean())
    teacher_acc_std.append(vals[:, 2].std())

token_acc_mean = np.array(token_acc_mean)
token_acc_std = np.array(token_acc_std)
strict_acc_mean = np.array(strict_acc_mean)
strict_acc_std = np.array(strict_acc_std)
teacher_acc_mean = np.array(teacher_acc_mean)
teacher_acc_std = np.array(teacher_acc_std)

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

metrics = [
    ("Token Accuracy (Autoregressive)", token_acc_mean, token_acc_std, "tab:blue"),
    ("Strict Accuracy (Autoregressive)", strict_acc_mean, strict_acc_std, "tab:orange"),
    ("Token Accuracy (Teacher-Forced)", teacher_acc_mean, teacher_acc_std, "tab:green"),
]

for ax, (title, mean, std, color) in zip(axes, metrics):
    ax.plot(iterations, mean, color=color, marker="o", markersize=4)
    ax.fill_between(iterations, mean - std, mean + std, alpha=0.2, color=color)
    ax.set_title(title)
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

fig.suptitle("In-Order Evaluation: Accuracy over Training", fontsize=14, y=1.02)
fig.tight_layout()
plt.show()

# %%
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(iterations, token_acc_mean, marker="o", markersize=4, label="Token Acc (Autoregressive)")
ax.plot(iterations, strict_acc_mean, marker="s", markersize=4, label="Strict Acc (Autoregressive)")
ax.plot(iterations, teacher_acc_mean, marker="^", markersize=4, label="Token Acc (Teacher-Forced)")
ax.fill_between(
    iterations, token_acc_mean - token_acc_std, token_acc_mean + token_acc_std, alpha=0.15
)
ax.fill_between(
    iterations, strict_acc_mean - strict_acc_std, strict_acc_mean + strict_acc_std, alpha=0.15
)
ax.fill_between(
    iterations,
    teacher_acc_mean - teacher_acc_std,
    teacher_acc_mean + teacher_acc_std,
    alpha=0.15,
)
ax.set_xlabel("Training Iteration")
ax.set_ylabel("Accuracy")
ax.set_ylim(-0.05, 1.05)
ax.set_title("In-Order Evaluation: All Metrics Compared")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()

# %%
# ============================================================
# Out-of-order evaluation results
# ============================================================
outorder_path = PROJECT_ROOT / "data/outorder_eval_step_random50/accs.pkl"
with open(outorder_path, "rb") as f:
    accs_outorder = pickle.load(f)

# Keys are (num_identities, displacement)
# Values are dicts mapping task_id -> (step_accs_array, final_accs_array)
all_keys = sorted(accs_outorder.keys())

num_ids = sorted({k[0] for k in all_keys})
displacements = sorted({k[1] for k in all_keys})

step_acc_grid = np.full((len(num_ids), len(displacements)), np.nan)
final_acc_grid = np.full((len(num_ids), len(displacements)), np.nan)
count_grid = np.full((len(num_ids), len(displacements)), 0)

for (ni, disp), task_dict in accs_outorder.items():
    i = num_ids.index(ni)
    j = displacements.index(disp)
    step_accs = np.array([np.mean(v[0]) for v in task_dict.values()])
    final_accs = np.array([np.mean(v[1]) for v in task_dict.values()])
    step_acc_grid[i, j] = step_accs.mean()
    final_acc_grid[i, j] = final_accs.mean()
    count_grid[i, j] = len(task_dict)

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, grid, title in zip(
    axes,
    [step_acc_grid, final_acc_grid],
    ["Step-by-Step Accuracy", "Final Output Accuracy"],
):
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(displacements)))
    ax.set_xticklabels(displacements)
    ax.set_yticks(range(len(num_ids)))
    ax.set_yticklabels(num_ids)
    ax.set_xlabel("Displacement")
    ax.set_ylabel("Num Identities")
    ax.set_title(title)
    for ii in range(len(num_ids)):
        for jj in range(len(displacements)):
            val = grid[ii, jj]
            if not np.isnan(val):
                ax.text(
                    jj,
                    ii,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black" if val > 0.4 else "white",
                )
    plt.colorbar(im, ax=ax, shrink=0.8)

fig.suptitle(
    "Out-of-Order Evaluation: Accuracy by (Num Identities, Displacement)", fontsize=13, y=1.02
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
            step_accs = np.array([np.mean(v[0]) for v in task_dict.values()])
            disps.append(disp)
            means.append(step_accs.mean())
    ax.plot(disps, means, marker="o", markersize=5, label=f"Identities={ni}")

ax.set_xlabel("Displacement")
ax.set_ylabel("Mean Step Accuracy")
ax.set_title("Out-of-Order: Step Accuracy by Displacement (per Num Identities)")
ax.set_ylim(-0.05, 1.05)
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()

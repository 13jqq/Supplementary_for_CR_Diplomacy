import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Linux Libertine O", "Libertine", "Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
})

row_labels = ["DipNet", "SearchBot", "Cicero\nno-press", "CR$^2$"]
col_labels = ["DipNet", "SearchBot", "Cicero", "CR$^2$"]

matrix = np.array([
    [np.nan,   0.015122, 0.021927, 0.010218],
    [0.451034, np.nan,   0.100000, 0.014339],
    [0.488583, 0.246083, np.nan,   0.062651],
    [0.398202, 0.256091, 0.158311, np.nan],
], dtype=float)

COLORMAP_NODES = [
    (0.00, "#2C7FB8"),
    (0.18, "#6FB6D7"),
    (0.34, "#C7E3EF"),
    (0.46, "#EEF5F8"),
    (0.50, "#FFF1EF"),
    (0.66, "#F8C9C3"),
    (0.83, "#F2958D"),
    (1.00, "#EC6A62"),
]

cmap = LinearSegmentedColormap.from_list(
    "custom_rb_soft",
    [(pos, mcolors.to_rgb(col)) for pos, col in COLORMAP_NODES],
    N=512,
).copy()

diag_color = "#C9CFD8"
cmap.set_bad(color=diag_color)
norm = TwoSlopeNorm(vmin=0.0, vcenter=0.20, vmax=0.50)

masked = np.ma.masked_invalid(matrix)
rgba = cmap(norm(masked))

red_cmap = LinearSegmentedColormap.from_list(
    "soft_reds_only",
    ["#FFF1EF", "#F8C9C3", "#F2958D", "#EC6A62"],
    N=256,
)
cr_row = matrix[-1, :]
cr_vals = cr_row[~np.isnan(cr_row)]
cr_min, cr_max = float(np.min(cr_vals)), float(np.max(cr_vals))
for j in range(matrix.shape[1]):
    v = matrix[-1, j]
    if np.isnan(v):
        rgba[-1, j, :] = mcolors.to_rgba(diag_color)
    else:
        t = (v - cr_min) / (cr_max - cr_min + 1e-12)
        rgba[-1, j, :] = red_cmap(0.25 + 0.75 * t)

for i in range(min(matrix.shape)):
    rgba[i, i, :] = mcolors.to_rgba(diag_color)

fig, ax = plt.subplots(figsize=(8.8, 7.6))
ax.imshow(rgba, aspect="equal", interpolation="nearest")

n = matrix.shape[0]
ax.set_xticks(np.arange(n))
ax.set_yticks(np.arange(n))
ax.set_xticklabels(col_labels, fontsize=18, color="black")
ax.set_yticklabels(row_labels, fontsize=18, color="black")
ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
ax.grid(which="minor", color="white", linewidth=1.6)
ax.tick_params(which="minor", length=0)
ax.tick_params(length=0, colors="black")

ax.set_xlabel("6 Agents", fontsize=19, labelpad=20, color="black")
ax.set_ylabel("1 Agent", fontsize=19, labelpad=18, color="black")

for spine in ax.spines.values():
    spine.set_visible(False)

for i in range(n):
    for j in range(n):
        v = matrix[i, j]
        if np.isnan(v):
            ax.text(j, i, "-", ha="center", va="center", fontsize=19, color="black")
        else:
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=18, color="black")

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="4.5%", pad=0.12)
mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
mappable.set_array([])
cbar = fig.colorbar(mappable, cax=cax)
cbar.ax.tick_params(labelsize=13, colors="black")
ticks = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50]
cbar.set_ticks(ticks)
cbar.set_ticklabels([f"{t:.2f}" for t in ticks])
for spine in cbar.ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.savefig("diplomacy_1v6_sos_heatmap.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig("diplomacy_1v6_sos_heatmap.pdf", dpi=300, bbox_inches="tight", facecolor="white")
plt.show()

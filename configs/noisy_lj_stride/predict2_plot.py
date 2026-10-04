"""Plot the original-seed one-frame/two-frame comparison."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

folder = Path(__file__).parent
result = json.loads((folder / "predict2_comparison.json").read_text())
rows = {(row["variant"], row["model"]): row for row in result["summary"]}
variants = ["published_head", "published_mst", "predict1_spread", "predict2_spread"]
labels = ["Published one-step\nHead windows", "Published MST\nHead windows",
          "Matched control\nSpread: 1 frame", "New two-frame\nSpread: 2 frames"]
plt.rcParams.update({"font.size": 17, "axes.titlesize": 19, "axes.labelsize": 18,
                     "xtick.labelsize": 16, "ytick.labelsize": 17})
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
for ax, metric, title in zip(axes, ["test_r2", "relative_mse"],
                              ["Test Poisson R² ↑", "Relative position MSE ↓"]):
    maximum = max(rows[variant, model][f"{metric}_mean"] + rows[variant, model][f"{metric}_sd"]
                  for variant in variants for model in ["gns", "mlp"])
    limit = 1 if metric == "test_r2" else maximum * 1.35
    for index, (model, color, hatch) in enumerate([("gns", "#0072b2", ""), ("mlp", "#d55e00", "//")]):
        values = np.array([rows[variant, model][f"{metric}_mean"] for variant in variants])
        sd = np.array([rows[variant, model][f"{metric}_sd"] for variant in variants])
        drawn = np.maximum(values, 0)
        lower = drawn - np.maximum(values - sd, 0)
        upper = np.maximum(values + sd, 0) - drawn
        bars = ax.bar(np.arange(4) + (index - .5) * .36, drawn, .36, color=color, hatch=hatch,
                      edgecolor="white", label="GNS" if model == "gns" else "MLP",
                      yerr=[lower, upper], capsize=4)
        for bar, value, error in zip(bars, values, upper):
            offset = limit * (.025 if index == 0 else .065)
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + error + offset,
                    f"{value:.3f}", ha="center", va="bottom", fontsize=17)
    ax.set_xticks(range(4), labels)
    ax.set_title(title, pad=14)
    ax.set_ylim(0, limit)
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
axes[0].legend(loc="upper right")
fig.suptitle("Noisy LJ: predict two frames per step with the original seed", fontsize=24, y=.97)
fig.text(.5, .075, "All inputs: frames 0–3 · Forecast and Poisson measurement: 3→103 · Same 50 train / 50 validation / 70 test networks\n"
         "Mean ± sample SD over 3 seeds. Matched spread pair uses early stopping; published one-step used fixed 40 epochs.\n"
         "Two-frame runs interpolate intermediate predictions. Negative R² is drawn at 0; labels retain actual values.",
         ha="center", fontsize=16)
fig.subplots_adjust(left=.06, right=.98, top=.84, bottom=.27, wspace=.2)
fig.savefig(folder / "predict2_comparison.png", dpi=160)
fig.savefig(folder / "predict2_comparison.pdf")
print("Saved predict2_comparison.png and predict2_comparison.pdf")

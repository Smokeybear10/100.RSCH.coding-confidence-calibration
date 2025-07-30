from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {"logistic": "Logistic", "platt": "Platt", "isotonic": "Isotonic",
          "group_logistic": "Group-aware logistic", "multicalibration": "Multicalibration"}
COLORS = {"logistic": "#245574", "platt": "#5492b3", "multicalibration": "#d1782d",
          "group_logistic": "#24776d"}


def read_table(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def make_figure(results: Path, output: Path) -> None:
    comparisons = {row["method"]: row for row in read_table(results / "comparisons.csv")}
    slices = read_table(results / "slices.csv")
    frontier = read_table(results / "risk_coverage.csv")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    order = list(LABELS)
    for y, method in enumerate(order):
        row = comparisons[method]
        center, low, high = [1000*float(row[key]) for key in ("estimate", "low", "high")]
        axes[0].errorbar(center, y, xerr=np.array([[center-low], [high-center]]),
                         fmt="o", color=COLORS.get(method, "#777777"), capsize=4)
    axes[0].set_yticks(range(len(order)), [LABELS[name] for name in order])
    axes[0].invert_yaxis()
    axes[0].axvline(0, color="#999999", linestyle="--", linewidth=1)
    axes[0].set(title="Probability quality", xlabel="Δ Brier vs Platt (×10⁻³)")
    axes[0].text(.03, -.29, "Task-bootstrap 95% intervals; lower is better",
                 transform=axes[0].transAxes, fontsize=8)
    axes[1].plot([0, 1], [0, 1], "--", color="#aaaaaa", linewidth=1)
    for method in ("logistic", "platt", "multicalibration"):
        rows = sorted((row for row in slices if row["split"] == "test" and row["method"] == method
                       and row["group"] == "all"), key=lambda row: int(row["bin"]))
        x, y = [float(row["mean"]) for row in rows], [float(row["rate"]) for row in rows]
        axes[1].plot(x, y, color=COLORS[method], label=LABELS[method], linewidth=1.5)
        axes[1].scatter(x, y, s=[15+float(row["n"])/25 for row in rows], color=COLORS[method], alpha=.8)
    axes[1].set(title="Observed reliability", xlabel="Mean predicted pass probability",
                ylabel="Observed pass rate", xlim=(0, 1), ylim=(0, 1))
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    axes[1].text(.03, -.29, "Five score bands; marker area increases with count",
                 transform=axes[1].transAxes, fontsize=8)
    for method in COLORS:
        rows = sorted((row for row in frontier if row["method"] == method and row["accepted_error"]),
                      key=lambda row: float(row["test_coverage"]))
        axes[2].plot([float(row["test_coverage"]) for row in rows],
                     [float(row["accepted_error"]) for row in rows], marker="o", markersize=3,
                     color=COLORS[method], label=LABELS[method])
    axes[2].set(title="Selective acceptance", xlabel="Fraction of primary answers accepted",
                ylabel="Error among accepted answers", xlim=(0, 1))
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].text(.03, -.29, "Cutoffs selected on validation; evaluated on test",
                 transform=axes[2].transAxes, fontsize=8)
    for axis in axes:
        axis.grid(alpha=.15)
    figure.suptitle("Coding confidence: 264 test tasks, 2,640 related attempts", fontsize=13, y=1.03)
    figure.tight_layout(w_pad=2.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render diagnostics from the recorded comparison tables")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/diagnostics.png"))
    args = parser.parse_args()
    make_figure(args.results, args.output)

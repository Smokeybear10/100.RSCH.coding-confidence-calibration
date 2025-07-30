from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS = {"raw": "Raw", "platt": "Platt", "fixed_mc": "Fixed groups",
          "tree_mc": "Tree audit", "confirmed_tree": "Confirmed tree"}
COLORS = {"platt": "#4c7892", "fixed_mc": "#8a7b99", "tree_mc": "#ce7e34", "confirmed_tree": "#24786e"}


def overview(results: Path) -> None:
    paired = pd.read_csv(results / "paired_comparisons.csv")
    paired = paired[paired["contrast"] == "overall"].set_index("method")
    figure, axes = plt.subplots(2, 2, figsize=(15, 11))
    order = [f"{base}/{variant}" for base in ("logistic", "histogram_boosting") for variant in LABELS]
    for y, method in enumerate(order):
        row = paired.loc[method]
        center, low, high = [1000*row[key] for key in ("brier_difference", "low", "high")]
        axes[0, 0].errorbar(center, y, xerr=[[center-low], [high-center]], fmt="o", capsize=3,
                            color="#245574" if method.startswith("logistic/") else "#8b587d")
    axes[0, 0].set_yticks(range(len(order)), [f"{'Logistic' if name.startswith('logistic/') else 'Boosted'} / {LABELS[name.split('/')[1]]}" for name in order])
    axes[0, 0].invert_yaxis()
    axes[0, 0].axvline(0, color="#888888", linestyle="--", linewidth=1)
    axes[0, 0].set(title="Repeated task-separated evaluation", xlabel="Δ Brier vs logistic/Platt (×10⁻³)")
    records = json.loads((results / "auditor_rules.json").read_text())
    audit = pd.DataFrame([{"base": base, "method": method, "patches": len(fit["patches"])}
                          for fold in records for base, details in fold["models"].items()
                          for method, fit in details["calibrators"].items() if method in ("tree_mc", "confirmed_tree")])
    for i, method in enumerate(("tree_mc", "confirmed_tree")):
        values = audit[audit["method"] == method].groupby("base")["patches"].mean()
        axes[0, 1].bar(np.arange(2)+(i-.5)*.32, [values.get(base, 0) for base in ("logistic", "histogram_boosting")],
                        width=.32, label=LABELS[method], color=COLORS[method])
    axes[0, 1].set_xticks([0, 1], ["Logistic", "Boosted trees"])
    axes[0, 1].set(title="Corrections that survive confirmation", ylabel="Mean accepted patches per fold", ylim=(0, 3.5))
    axes[0, 1].legend(frameon=False)
    samples = pd.read_csv(results / "sample_complexity.csv")
    for method in ("fixed_mc", "tree_mc", "confirmed_tree"):
        values = samples[samples["method"] == method].groupby("auditor_tasks")["test_brier"].mean()
        axes[1, 0].plot(values.index, values, marker="o", color=COLORS[method], label=LABELS[method])
    metrics = pd.read_csv(results / "fold_metrics.csv")
    reference = metrics.loc[metrics["method"] == "logistic/platt", "brier"].mean()
    axes[1, 0].axhline(reference, color=COLORS["platt"], linestyle="--", label="Platt, no audit")
    axes[1, 0].set(title="Auditor sample budget", xlabel="Available discovery + confirmation tasks", ylabel="Mean held-out Brier")
    axes[1, 0].legend(frameon=False, fontsize=9)
    costs = pd.read_csv(results / "decision_costs.csv")
    baseline = costs[(costs["method"] == "logistic/platt") & (costs["selector"] == "all_validation_boundaries")]
    choices = [("histogram_boosting/platt", "all_validation_boundaries", "Boosted / Platt", "#8b587d", "-"),
               ("histogram_boosting/confirmed_tree", "all_validation_boundaries", "Boosted / confirmed", "#24786e", "-"),
               ("benefit_model", "predicted_benefit", "Benefit model / plug-in", "#ce7e34", "--"),
               ("benefit_model", "all_validation_boundaries", "Benefit model / selected cutoff", "#ce7e34", "-")]
    for method, selector, label, color, style in choices:
        current = costs[(costs["method"] == method) & (costs["selector"] == selector)]
        joined = current.merge(baseline[["seed", "fold", "failure_loss", "total_objective"]],
                               on=["seed", "fold", "failure_loss"], suffixes=("", "_baseline"), validate="one_to_one")
        joined["difference"] = joined["total_objective"]-joined["total_objective_baseline"]
        values = joined.groupby("failure_loss")["difference"].mean()
        axes[1, 1].plot(values.index, values, marker="o", linestyle=style, color=color, label=label)
    axes[1, 1].axhline(0, color="#888888", linestyle=":", label="Logistic / exact cutoff")
    axes[1, 1].set(title="Decision-aware routing", xlabel="Assumed loss per failed answer",
                   ylabel="Δ compute + failure objective vs logistic")
    axes[1, 1].legend(frameon=False, fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=.15)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Learned calibration and routing · 3 repeats × 5 task-separated folds", fontsize=15)
    figure.tight_layout(rect=(0, .035, 1, .96), h_pad=3, w_pad=3)
    figure.text(.5, .012, "Intervals are descriptive, conditional on overlapping fitted models; costs are stipulated units, not measured bills.", ha="center", fontsize=10)
    figure.savefig(results / "advanced_diagnostics.png", dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def simulation_figure(results: Path) -> None:
    frame = pd.read_csv(results / "simulation_summary.csv")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for axis, scenario, title in zip(axes, ("calibrated_null", "hidden_interaction"), ("Already calibrated", "Hidden interaction bias")):
        for method in ("raw", "tree_mc", "confirmed_tree"):
            rows = frame[(frame["scenario"] == scenario) & (frame["method"] == method)].sort_values("calibration_tasks")
            x = rows["calibration_tasks"].to_numpy()
            mean, spread = rows["known_probability_mse"].to_numpy(), rows["mse_seed_sd"].fillna(0).to_numpy()
            color = COLORS.get(method, "#555555")
            axis.plot(x, mean, marker="o", color=color, label=LABELS[method])
            axis.fill_between(x, np.maximum(mean-spread, 0), mean+spread, color=color, alpha=.12)
        axis.set_xscale("log")
        axis.set_xticks(sorted(frame["calibration_tasks"].unique()), [str(value) for value in sorted(frame["calibration_tasks"].unique())])
        axis.set(title=title, xlabel="Calibration tasks (log scale)", ylabel="MSE against known probability")
        axis.grid(alpha=.15)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)
    figure.suptitle("Synthetic controls · task-correlated outcomes · mean ± seed SD", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, .95))
    figure.savefig(results / "simulation_diagnostics.png", dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def mixture_figure(results: Path) -> None:
    frame = pd.read_csv(results / "mixture_shift.csv")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    choices = ["logistic/raw", "logistic/platt", "logistic/confirmed_tree",
               "histogram_boosting/raw", "histogram_boosting/confirmed_tree"]
    for axis, feature in zip(axes, ("ast_depth", "log_prompt_chars")):
        for i, method in enumerate(choices):
            selected = frame[(frame["feature"] == feature) & (frame["method"] == method)]
            values = selected.groupby("target_high_fraction")["brier"].mean()
            axis.plot(values.index, values, marker="o", label=method.replace("histogram_boosting", "boosted"), color=plt.cm.tab10(i))
        axis.set(title="AST-depth mixture" if feature == "ast_depth" else "Prompt-length mixture",
                 xlabel="Target share above training-defined median", ylabel="Reweighted held-out Brier")
        axis.grid(alpha=.15)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle("Retrospective covariate-mixture stress test; no new outcomes or temporal-shift claim", fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, .95))
    figure.savefig(results / "mixture_diagnostics.png", dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def make_figures(results: Path, simulations: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    overview(results)
    simulation_figure(simulations)
    mixture_figure(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the advanced benchmark and known-probability controls")
    parser.add_argument("--results", type=Path, default=Path("advanced_results"))
    parser.add_argument("--simulations", type=Path, default=Path("simulation_results"))
    args = parser.parse_args()
    make_figures(args.results, args.simulations)

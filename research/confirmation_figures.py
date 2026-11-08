from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


SCENARIOS = ("calibrated_null", "weak_common", "strong_common", "strong_rare")
SCENARIO_LABELS = {"calibrated_null": "Already calibrated", "weak_common": "Weak common bias",
                   "strong_common": "Strong common bias", "strong_rare": "Strong rare bias"}
METHODS = ("no_update", "discovery_fixed", "independent_fixed", "independent_gated", "pooled_shrinkage")
LABELS = {"no_update": "No correction", "discovery_fixed": "Discovery estimate, fixed step",
          "independent_fixed": "Independent estimate, fixed step", "independent_gated": "Independent estimate + gate",
          "pooled_shrinkage": "Pooled, validation-selected step", "raw_logistic": "Raw logistic"}
COLORS = {"no_update": "#59636D", "discovery_fixed": "#CF812A", "independent_fixed": "#3D6DA5",
          "independent_gated": "#25765D", "pooled_shrinkage": "#885B96", "raw_logistic": "#222222"}
MARKERS = {"no_update": "o", "discovery_fixed": "s", "independent_fixed": "D",
           "independent_gated": "^", "pooled_shrinkage": "v"}


def _correlated(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame["shared_uniform"].astype(str).str.lower()
    if not set(values.unique()) <= {"true", "false", "1", "0"}:
        raise ValueError("shared_uniform must contain boolean CSV values")
    return frame.loc[values.isin(["true", "1"])]


def _curve(frame: pd.DataFrame, scenario: str, method: str) -> pd.DataFrame:
    selected = frame[(frame["scenario"] == scenario) & (frame["method"] == method)].sort_values("budget")
    if selected["budget"].duplicated().any():
        raise ValueError(f"Expected one summary row per budget for {scenario}/{method}")
    return selected


def _incomplete(results: Path, summary: pd.DataFrame) -> bool:
    if summary.empty:
        return True
    record = results / "summary.json"
    if record.exists() and json.loads(record.read_text()).get("smoke", False):
        return True
    protocol = results / "protocol.json"
    budgets = set(summary["budget"])
    if protocol.exists():
        budgets = set(json.loads(protocol.read_text())["simulation"]["budgets"])
    return any(set(_curve(summary, scenario, method)["budget"]) != budgets
               for scenario in SCENARIOS for method in METHODS)


def _unrun(axis: plt.Axes) -> None:
    axis.text(.5, .5, "Not run\n(incomplete study)", transform=axis.transAxes,
              ha="center", va="center", color="#777777", fontsize=12)


def _style(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=.16)
    axis.set_axisbelow(True)


def confirmation_diagnostics(results: Path) -> Path:
    summary = _correlated(pd.read_csv(results / "simulation_summary.csv"))
    incomplete = _incomplete(results, summary)
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 8.6), sharex=True, sharey=True)
    handles, upper_bounds = [], []
    shown_methods = set()
    budgets = sorted(summary["budget"].unique())
    for axis, scenario in zip(axes.flat, SCENARIOS):
        plotted = False
        for method in METHODS:
            rows = _curve(summary, scenario, method)
            if rows.empty:
                continue
            plotted = True
            x = rows["budget"].to_numpy(dtype=float)
            center = rows["known_probability_mse"].to_numpy(dtype=float)
            spread = rows["known_probability_mse_se"].to_numpy(dtype=float)
            upper_bounds.extend(center + 1.96 * np.where(np.isfinite(spread), spread, 0))
            line, = axis.plot(x, center, marker=MARKERS[method], markersize=4.5,
                              linestyle="--" if method == "no_update" else "-",
                              color=COLORS[method], label=LABELS[method], linewidth=1.7)
            valid = np.isfinite(center) & np.isfinite(spread)
            axis.fill_between(x[valid], np.maximum(center[valid] - 1.96 * spread[valid], 0),
                              center[valid] + 1.96 * spread[valid], color=COLORS[method], alpha=.11)
            if len(x) == 1 and valid[0]:
                axis.errorbar(x, center, yerr=[[min(center[0], 1.96*spread[0])], [1.96*spread[0]]],
                               fmt="none", capsize=3, color=COLORS[method], alpha=.75)
            if method not in shown_methods:
                handles.append(line)
                shown_methods.add(method)
        if not plotted:
            _unrun(axis)
        axis.set_xscale("log")
        axis.set_xticks(budgets, [str(int(value)) for value in budgets])
        axis.set_title(SCENARIO_LABELS[scenario])
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
        _style(axis)
    axes[0, 0].set_ylim(0, max(max(upper_bounds, default=0) * 1.08, 1e-6))
    if len(budgets) == 1:
        axes[0, 0].set_xlim(budgets[0]/1.25, budgets[0]*1.25)
    for axis in axes[1]:
        axis.set_xlabel("Available correction + validation tasks (log scale)")
    for axis in axes[:, 0]:
        axis.set_ylabel("MSE against known probability")
    subtitle = "Task-correlated synthetic outcomes" + (" · INCOMPLETE STUDY" if incomplete else "")
    figure.suptitle("Confirmation and shrinkage at a fixed available budget\n" + subtitle, fontsize=14)
    if handles:
        figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .075), ncol=3, frameon=False, fontsize=9)
    figure.tight_layout(rect=(0, .19, 1, .92), h_pad=2.3, w_pad=2)
    figure.text(.5, .037, "Intervals: mean ± 1.96 Monte Carlo SE across fitting seeds; omitted when SE is unavailable. Lower bounds are clipped at zero.", ha="center", fontsize=8.5)
    figure.text(.5, .014, "Fixed-step methods share a proposal; pooled shrinkage learns from the whole correction pool. These are synthetic results.", ha="center", fontsize=8.5)
    path = results / "confirmation_diagnostics.png"
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def gate_diagnostics(results: Path) -> Path:
    summary = _correlated(pd.read_csv(results / "simulation_summary.csv"))
    incomplete = _incomplete(results, summary)
    contrasts = _correlated(pd.read_csv(results / "simulation_contrasts.csv"))
    paired = contrasts[(contrasts["left"] == "independent_gated") &
                       (contrasts["right"] == "independent_fixed")]
    figure, axes = plt.subplots(1, 2, figsize=(14, 6.4))
    method_handles = []
    for method in METHODS:
        rows = _curve(summary, "calibrated_null", method)
        if rows.empty:
            continue
        x = rows["budget"].to_numpy(dtype=float)
        center = rows["update_rate"].to_numpy(dtype=float)
        low, high = rows["update_low"].to_numpy(dtype=float), rows["update_high"].to_numpy(dtype=float)
        line, = axes[0].plot(x, center, marker=MARKERS[method], markersize=5,
                             linestyle="--" if method == "no_update" else "-",
                             color=COLORS[method], label=LABELS[method], linewidth=1.7)
        axes[0].errorbar(x, center, yerr=np.maximum(np.vstack([center-low, high-center]), 0),
                         fmt="none", color=COLORS[method], capsize=3, alpha=.8)
        method_handles.append(line)
    budgets = sorted(summary["budget"].unique())
    axes[0].set_xscale("log")
    axes[0].set_xticks(budgets, [str(int(value)) for value in budgets])
    if len(budgets) == 1:
        axes[0].set_xlim(budgets[0]/1.25, budgets[0]*1.25)
    axes[0].set(title="False corrections under the calibrated null", ylim=(-.025, 1.025),
                xlabel="Available correction + validation tasks (log scale)", ylabel="Fraction of fitting runs with a correction")
    if method_handles:
        axes[0].legend(handles=method_handles, loc="upper center", bbox_to_anchor=(.5, -.29), ncol=2, frameon=False, fontsize=8.5)
    else:
        _unrun(axes[0])

    paired_budgets = sorted(paired["budget"].unique())
    offsets = np.linspace(-.24, .24, len(paired_budgets)) if len(paired_budgets) > 1 else [0]
    budget_markers = ("o", "s", "^")
    if len(paired_budgets) > len(budget_markers):
        raise ValueError("Gate forest plot supports up to three prespecified budgets")
    for position, scenario in enumerate(SCENARIOS):
        rows = paired[paired["scenario"] == scenario].set_index("budget", verify_integrity=True)
        if rows.empty:
            axes[1].text(.02, position, "Not run (incomplete study)", transform=axes[1].get_yaxis_transform(),
                          ha="left", va="center", color="#777777", fontsize=9)
        for index, budget in enumerate(paired_budgets):
            if budget not in rows.index:
                continue
            row = rows.loc[budget]
            y = position + offsets[index]
            center, low, high = [float(row[name]) for name in ("difference", "low", "high")]
            axes[1].hlines(y, low, high, color=COLORS["independent_gated"], linewidth=1.4)
            axes[1].vlines([low, high], y-.045, y+.045, color=COLORS["independent_gated"], linewidth=1)
            axes[1].scatter(center, y, marker=budget_markers[index], color=COLORS["independent_gated"], s=32, zorder=3)
    axes[1].axvline(0, color="#777777", linestyle="--", linewidth=1)
    axes[1].set_yticks(range(len(SCENARIOS)), [SCENARIO_LABELS[name] for name in SCENARIOS])
    axes[1].set_ylim(len(SCENARIOS)-.5, -.5)
    axes[1].set(title="Gate effect with an independent estimate",
                xlabel="Δ probability MSE: gated − ungated\nNegative favors the confirmation gate")
    axes[1].ticklabel_format(axis="x", style="sci", scilimits=(-2, 2), useMathText=True)
    budget_handles = [Line2D([], [], color=COLORS["independent_gated"], marker=budget_markers[i], linestyle="none",
                             label=f"{int(budget)} tasks") for i, budget in enumerate(paired_budgets)]
    if budget_handles:
        axes[1].legend(handles=budget_handles, loc="upper center", bbox_to_anchor=(.5, -.29), ncol=3, frameon=False, fontsize=9)
    for axis in axes:
        _style(axis)
    subtitle = "Task-correlated synthetic outcomes" + (" · INCOMPLETE STUDY" if incomplete else "")
    figure.suptitle("What the independent gate changes\n" + subtitle, fontsize=14)
    figure.subplots_adjust(left=.065, right=.99, bottom=.32, top=.8, wspace=.38)
    figure.text(.5, .025, "Left: exact 95% binomial intervals across fitting seeds. Right: paired 95% bootstrap intervals across fitting seeds. No simultaneous-coverage claim.", ha="center", fontsize=8.5)
    path = results / "gate_diagnostics.png"
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def source_external(results: Path) -> Path | None:
    if not (results / "source_metrics.csv").exists() or not (results / "external_metrics.csv").exists():
        return None
    source = pd.read_csv(results / "source_metrics.csv")
    external = pd.read_csv(results / "external_metrics.csv").set_index("method", verify_integrity=True)
    budget = int(source["budget"].max())
    source = source[source["budget"] == budget].groupby("method")["brier"].mean()
    methods = [name for name in ("raw_logistic", *METHODS) if name in source.index and name in external.index]
    if not methods:
        raise ValueError("Source and external tables have no shared plotted methods")
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharex=True, sharey=True)
    values = [float(source.loc[name]) for name in methods] + [float(external.loc[name, "brier"]) for name in methods]
    padding = max((max(values)-min(values))*.15, .002)
    for axis, estimates in ((axes[0], source), (axes[1], external["brier"])):
        for row, method in enumerate(methods):
            axis.scatter(float(estimates.loc[method]), row, color=COLORS[method], s=42, zorder=3)
        axis.set(xlabel="Brier score (lower is better)", xlim=(max(min(values)-padding, 0), max(values)+padding))
        _style(axis)
    axes[0].set_yticks(range(len(methods)),
                       ["Platt, no subgroup correction" if name == "no_update" else LABELS[name] for name in methods])
    axes[0].invert_yaxis()
    axes[0].set_title(f"Source: mean held-out fold Brier\n{budget} correction + validation tasks")
    axes[1].set_title("External benchmark: frozen-ensemble Brier\nOne evaluation after source fitting")
    figure.suptitle("Source evaluation and frozen external transfer", fontsize=14)
    figure.tight_layout(rect=(0, .1, 1, .9), w_pad=3)
    figure.text(.5, .025, "Separate test populations and estimands; no uncertainty intervals are shown. External outcomes do not choose models or corrections.", ha="center", fontsize=9)
    path = results / "source_external.png"
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def make_figures(results: Path = Path("confirmation_results")) -> list[Path]:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    paths = [confirmation_diagnostics(results), gate_diagnostics(results)]
    external = source_external(results)
    if external is not None:
        paths.append(external)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot matched confirmation experiments and frozen external evaluation")
    parser.add_argument("--results", type=Path, default=Path("confirmation_results"))
    arguments = parser.parse_args()
    for output in make_figures(arguments.results):
        print(output)

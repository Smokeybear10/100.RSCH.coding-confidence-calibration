from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import warnings

import numpy as np
from scipy.special import expit, logit
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from .data import FEATURES, ROOT, digest
from .methods import (budget_threshold, clustered_interval, fit_baselines, fit_patches,
                      group_thresholds, membership, metrics, policy, predict_classifier, replay)
from .reference import REFERENCE, load_reference


def partitions_from_reference(reference: dict) -> dict:
    frozen = reference["frozen"]
    partitions = {}
    for split in ("train", "calibration", "validation", "test"):
        rows = [row for row in reference["rows"] if row["split"] == split]
        x = np.array([row["features"] for row in rows])
        partitions[split] = {
            "x": x,
            "y": np.array([row["label"] for row in rows]),
            "alternate": np.array([row["alternate_label"] for row in rows]),
            "ids": np.array([row["task_id"] for row in rows]),
            "keys": np.array([row["tie_key"] for row in rows]),
            "groups": membership(x, frozen["group_thresholds"], frozen["group_names"]),
            "rows": rows,
        }
    return partitions


def replay_models(partitions: dict, frozen: dict) -> dict:
    models, names = frozen["models"], frozen["group_names"]
    predictions = {}
    for split, part in partitions.items():
        scores = {}
        columns = {name: part["x"][:, j] for j, name in enumerate(FEATURES)}
        columns.update({name: part["groups"][:, j] for j, name in enumerate(names)})
        for method in ("logistic", "group_logistic", "token_only", "structure_only"):
            artifact = models[method]
            scores[method] = predict_classifier(
                artifact, np.column_stack([columns[name] for name in artifact["features"]]),
            )
        raw = scores["logistic"]
        scores["platt"] = expit(models["platt"]["coefficient"] * logit(np.clip(raw, 1e-6, 1-1e-6))
                                + models["platt"]["intercept"])
        scores["isotonic"] = np.interp(raw, models["isotonic"]["x"], models["isotonic"]["y"])
        scores["likelihood"] = np.exp(part["x"][:, FEATURES.index("mean_logprob")])
        for method in ("multicalibration", "score_bins"):
            scores[method] = replay(scores["platt"], part["groups"], names, models[method])
        predictions[split] = scores
    return predictions


def refit_models(partitions: dict, protocol: dict) -> tuple[dict, dict]:
    names = protocol["groups"]
    thresholds = group_thresholds(partitions["train"]["x"])
    parts = {name: {**part, "groups": membership(part["x"], thresholds, names)}
             for name, part in partitions.items()}
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        with threadpool_limits(limits=1):
            predictions, models = fit_baselines(parts["train"], parts["calibration"], parts, names)
    cal, val = parts["calibration"], parts["validation"]
    selection = []
    for method, groups in (("multicalibration", names), ("score_bins", ["all"])):
        candidates = []
        indices = [names.index(name) for name in groups]
        for epsilon in protocol["epsilon_grid"]:
            for support in protocol["support_grid"]:
                fitted = fit_patches(predictions["calibration"]["platt"], cal["y"],
                                     cal["groups"][:, indices], cal["ids"], groups,
                                     protocol["bins"], support, epsilon, protocol["max_updates"])
                p = replay(predictions["validation"]["platt"], val["groups"], names, fitted)
                brier = float(np.mean((p - val["y"])**2))
                candidates.append((brier, len(fitted["patches"]), fitted))
                selection.append({"method": method, "epsilon": epsilon, "support": support,
                                  "validation_brier": brier, "updates": len(fitted["patches"]),
                                  "status": fitted["status"]})
        models[method] = min(candidates, key=lambda item: (item[0], item[1]))[2]
        for split, part in parts.items():
            predictions[split][method] = replay(predictions[split]["platt"], part["groups"], names, models[method])
    cutoffs = {method: budget_threshold(p, protocol["validation_escalation_budget"], val["keys"])
               for method, p in predictions["validation"].items()}
    return predictions, {"models": models, "group_names": names, "group_thresholds": thresholds,
                         "routing_thresholds": cutoffs, "selection": selection}


def result_tables(predictions: dict, partitions: dict, frozen: dict, protocol: dict) -> dict:
    names, bins = frozen["group_names"], protocol["bins"]
    support = min(protocol["support_grid"])
    rows, cells, comparisons, frontier, subsets = [], [], [], [], []
    for split, part in partitions.items():
        for method, scores in predictions[split].items():
            groups = membership(part["x"], frozen["group_thresholds"], names)
            result = metrics(scores, part["y"], groups, part["ids"], names, bins, support)
            routed = policy(scores, part["y"], part["alternate"], frozen["routing_thresholds"][method], part["keys"])
            rows.append({"split": split, "method": method, "tasks": len(np.unique(part["ids"])),
                         "candidates": len(scores),
                         **{key: result[key] for key in ("brier", "log_loss", "ece", "worst", "supported_cells", "occupied_cells")},
                         **{key: value for key, value in routed.items() if key != "threshold"}})
            cells.extend({"split": split, "method": method, **cell} for cell in result["cells"])
            if split != "test":
                continue
            differences = (scores - part["y"])**2 - (predictions[split]["platt"] - part["y"])**2
            comparisons.append({"method": method, "baseline": "platt",
                                **clustered_interval(differences, part["ids"], protocol["seed"], protocol["bootstrap_replicates"])})
            empty = part["x"][:, FEATURES.index("empty_code")] == 1
            for name, mask in (("empty", empty), ("nonempty", ~empty)):
                if not np.any(mask):
                    continue
                interval = clustered_interval(differences[mask], part["ids"][mask],
                                               protocol["seed"], protocol["bootstrap_replicates"])
                subsets.append({"method": method, "subset": name, "candidates": int(mask.sum()),
                                "pass_rate": float(part["y"][mask].mean()),
                                "brier": float(np.mean((scores[mask]-part["y"][mask])**2)),
                                "baseline": "platt", **interval})
            for budget in np.linspace(0, 1, 11):
                cutoff = budget_threshold(predictions["validation"][method], float(budget), partitions["validation"]["keys"])
                outcome = policy(scores, part["y"], part["alternate"], cutoff, part["keys"])
                frontier.append({"method": method, "validation_budget": float(budget),
                                 "test_coverage": 1-outcome["escalation"],
                                 **{key: value for key, value in outcome.items() if key != "threshold"}})
    return {"metrics": rows, "slices": cells, "comparisons": comparisons,
            "risk_coverage": frontier, "subsets": subsets, "selection": frozen["selection"]}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reproduce(output: Path, refit: bool = False) -> dict:
    output = output.resolve()
    if output == REFERENCE.resolve() or REFERENCE.resolve() in output.parents:
        raise ValueError("Choose an output directory outside the saved reference run")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new output directory to preserve earlier results")
    reference = load_reference()
    parts = partitions_from_reference(reference)
    protocol = reference["protocol"]
    if refit:
        predictions, frozen = refit_models(parts, protocol)
    else:
        frozen = reference["frozen"]
        predictions = replay_models(parts, frozen)
    comparisons = []
    for split, part in parts.items():
        for method, scores in predictions[split].items():
            expected = np.array([row["scores"][method] for row in part["rows"]])
            if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
                raise ValueError(f"Invalid probabilities: {split}/{method}")
            comparisons.append({"split": split, "method": method,
                                "maximum_probability_difference": float(np.max(np.abs(scores-expected))),
                                "within_1e-8": bool(np.allclose(scores, expected, rtol=0, atol=1e-8))})
    tables = result_tables(predictions, parts, frozen, protocol)
    metric_comparisons = []
    for row in tables["metrics"]:
        expected = reference["summary"]["partitions"][row["split"]]["methods"][row["method"]]
        for metric in ("brier", "log_loss", "ece", "worst"):
            original, observed = expected[metric], row[metric]
            metric_comparisons.append({"split": row["split"], "method": row["method"],
                                       "metric": metric, "reference": original, "reproduction": observed,
                                       "difference": observed-original if original is not None and observed is not None else None})
    tables["metric_comparison"] = metric_comparisons
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "refit from recorded features" if refit else "replay frozen models",
        "reference_run": reference["manifest"]["run_id"],
        "scope": "Existing public-data sample; no fresh held-out experiment or raw-feature re-extraction",
        "python": platform.python_version(), "platform": platform.platform(),
        "dependencies": {name: importlib.metadata.version(name) for name in
                         ("numpy", "scipy", "scikit-learn", "pyarrow", "joblib", "threadpoolctl")},
        "code_sha256": {f"research/{path.name}": digest(path) for path in sorted(ROOT.glob("*.py"))},
        "reference_comparison": comparisons,
        "all_probabilities_match": all(row["within_1e-8"] for row in comparisons),
        "maximum_metric_differences": {
            metric: max(abs(row["difference"]) for row in metric_comparisons
                        if row["metric"] == metric and row["difference"] is not None)
            for metric in ("brier", "log_loss", "ece", "worst")},
        "table_rows": {name: len(rows) for name, rows in tables.items()},
        "checks": {"reference_hashes": True, "bounded_finite_probabilities": True},
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        write_csv(output / f"{name}.csv", rows)
    (output / "reproduction.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    if refit:
        (output / "refitted-models.json").write_text(json.dumps(frozen, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay or refit the saved experiment without dataset downloads")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refit", action="store_true")
    args = parser.parse_args()
    report = reproduce(args.output, args.refit)
    print(json.dumps({"mode": report["mode"], "all_probabilities_match": report["all_probabilities_match"],
                      "maximum_probability_difference": max(row["maximum_probability_difference"] for row in report["reference_comparison"])}, indent=2))
    raise SystemExit(0 if report["all_probabilities_match"] else 1)

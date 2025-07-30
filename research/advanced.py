from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .auditors import apply_auditor, fit_auditor
from .data import FEATURES, ROOT, digest
from .methods import budget_threshold, fit_patches, group_thresholds, membership, metrics, replay, route_mask
from .reference import REFERENCE, load_reference

ROLES = ("train", "score_calibration", "discovery", "confirmation", "validation")
VARIANTS = ("raw", "platt", "fixed_mc", "tree_mc", "confirmed_tree")


def take(data: dict, mask: np.ndarray) -> dict:
    return {name: values[mask] for name, values in data.items()}


def pooled(first: dict, second: dict) -> dict:
    return {name: np.concatenate([first[name], second[name]]) for name in first}


def archive_data(reference: dict) -> dict:
    rows = reference["rows"]
    return {"x": np.array([row["features"] for row in rows]),
            "y": np.array([row["label"] for row in rows]),
            "alternate": np.array([row["alternate_label"] for row in rows]),
            "ids": np.array([row["task_id"] for row in rows]),
            "keys": np.array([row["tie_key"] for row in rows]),
            "candidate": np.array([row["candidate"] for row in rows])}


def task_folds(data: dict, protocol: dict) -> list[dict]:
    unique = np.unique(data["ids"])
    plans = []
    for seed in protocol["seeds"]:
        tests = np.array_split(np.random.default_rng(seed).permutation(unique), protocol["outer_folds"])
        for fold, test in enumerate(tests):
            remaining = np.setdiff1d(unique, test)
            remaining = np.random.default_rng(seed+1009*(fold+1)).permutation(remaining)
            roles, start = {}, 0
            for role in ROLES[:-1]:
                stop = start+int(len(remaining)*protocol["role_fractions"][role])
                roles[role] = [str(value) for value in remaining[start:stop]]
                start = stop
            roles["validation"] = [str(value) for value in remaining[start:]]
            roles["test"] = [str(value) for value in test]
            plans.append({"seed": seed, "fold": fold, "roles": roles})
    return plans


def make_parts(data: dict, plan: dict) -> dict:
    return {role: take(data, np.isin(data["ids"], ids)) for role, ids in plan["roles"].items()}


def fit_calibrators(parts: dict, platt: dict, thresholds: dict, groups: list[str],
                    protocol: dict, seed: int) -> tuple[dict, dict]:
    pool = pooled(parts["discovery"], parts["confirmation"])
    pool_scores = np.concatenate([platt["discovery"], platt["confirmation"]])
    fixed = protocol["fixed_groups"]
    fitted_fixed = fit_patches(pool_scores, pool["y"], membership(pool["x"], thresholds, groups), pool["ids"],
                               groups, protocol["auditor"]["bins"], fixed["min_tasks"],
                               fixed["epsilon"], fixed["max_updates"])
    naive = fit_auditor({**pool, "p": pool_scores}, None, protocol["auditor"], seed)
    confirmed = fit_auditor({**parts["discovery"], "p": platt["discovery"]},
                            {**parts["confirmation"], "p": platt["confirmation"]}, protocol["auditor"], seed)
    scores = {}
    for role, part in parts.items():
        scores[role] = {
            "platt": platt[role],
            "fixed_mc": replay(platt[role], membership(part["x"], thresholds, groups), groups, fitted_fixed),
            "tree_mc": apply_auditor(part["x"], platt[role], naive),
            "confirmed_tree": apply_auditor(part["x"], platt[role], confirmed),
        }
    return scores, {"fixed_mc": fitted_fixed, "tree_mc": naive, "confirmed_tree": confirmed}


def fit_fold(data: dict, plan: dict, protocol: dict, groups: list[str]) -> dict:
    parts = make_parts(data, plan)
    train, calibration = parts["train"], parts["score_calibration"]
    seed = plan["seed"]+plan["fold"]
    thresholds = group_thresholds(train["x"])
    models = {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(**protocol["models"]["logistic"], random_state=seed)),
        "histogram_boosting": HistGradientBoostingClassifier(**protocol["models"]["histogram_boosting"], random_state=seed),
    }
    predictions, details, platt_predictions = {role: {} for role in parts}, {}, {}
    with threadpool_limits(limits=1):
        for name, model in models.items():
            model.fit(train["x"], train["y"])
            raw = {role: model.predict_proba(part["x"])[:, 1] for role, part in parts.items()}
            scaling = LogisticRegression(C=1e6, max_iter=2000, random_state=seed).fit(
                logit(np.clip(raw["score_calibration"], 1e-6, 1-1e-6)).reshape(-1, 1), calibration["y"],
            )
            slope, intercept = float(scaling.coef_[0, 0]), float(scaling.intercept_[0])
            platt = {role: expit(slope*logit(np.clip(p, 1e-6, 1-1e-6))+intercept) for role, p in raw.items()}
            calibrated, fits = fit_calibrators(parts, platt, thresholds, groups, protocol, seed)
            platt_predictions[name] = platt
            for role in parts:
                predictions[role].update({f"{name}/{variant}": p for variant, p in {"raw": raw[role], **calibrated[role]}.items()})
            details[name] = {"platt": {"slope": slope, "intercept": intercept}, "calibrators": fits}
        benefit_model = HistGradientBoostingRegressor(**protocol["models"]["histogram_boosting"], random_state=seed)
        benefit_model.fit(train["x"], train["alternate"]-train["y"])
        benefit = {role: np.clip(benefit_model.predict(part["x"]), -1, 1) for role, part in parts.items()}
    return {"parts": parts, "predictions": predictions, "benefit": benefit, "details": details,
            "group_thresholds": thresholds, "platt_predictions": platt_predictions}


def route_result(part: dict, routed: np.ndarray) -> dict:
    accepted = ~routed
    outcome = np.where(routed, part["alternate"], part["y"])
    return {"escalation": float(routed.mean()), "success": float(outcome.mean()),
            "accepted_error": float(1-part["y"][accepted].mean()) if np.any(accepted) else None,
            "rescued": float(np.mean(routed & (part["y"] == 0) & (part["alternate"] == 1))),
            "regressed": float(np.mean(routed & (part["y"] == 1) & (part["alternate"] == 0)))}


def optimal_validation_cutoff(scores: np.ndarray, part: dict, settings: dict, loss: float) -> tuple[dict, float]:
    order = np.lexsort((part["keys"], scores))
    benefit = part["alternate"][order]-part["y"][order]
    n = len(scores)
    differences = np.concatenate([[0.0], settings["alternate_cost"]*np.arange(1, n+1)-loss*np.cumsum(benefit)])
    count = int(np.argmin(differences))
    if count == 0:
        return {"score": 0.0, "tie_key": 0.0}, 0.0
    if count == n:
        return {"score": 1.000001, "tie_key": 0.0}, 1.0
    boundary = order[count]
    return {"score": float(scores[boundary]), "tie_key": float(part["keys"][boundary])}, count/n


def routing_tables(fitted: dict, protocol: dict) -> tuple[list, list, list]:
    validation, test = fitted["parts"]["validation"], fitted["parts"]["test"]
    settings = protocol["routing"]
    ranking = {method: (p, fitted["predictions"]["test"][method])
               for method, p in fitted["predictions"]["validation"].items()}
    ranking["benefit_model"] = ((1-fitted["benefit"]["validation"])/2, (1-fitted["benefit"]["test"])/2)
    budgets, costs, task_costs = [], [], []

    def record(method: str, selector: str, loss: float, routed: np.ndarray,
               budget: float | None, threshold: dict | None, direct: bool = False) -> None:
        result = route_result(test, routed)
        compute = np.repeat(settings["alternate_cost"], len(routed)) if direct else settings["primary_cost"]+settings["alternate_cost"]*routed
        outcomes = np.where(routed, test["alternate"], test["y"])
        objective = compute+loss*(1-outcomes)
        costs.append({"method": method, "selector": selector, "failure_loss": loss, "validation_budget": budget,
                      "cutoff_score": threshold["score"] if threshold else None,
                      "cutoff_tie_key": threshold["tie_key"] if threshold else None,
                      "mean_compute": float(np.mean(compute)), "total_objective": float(np.mean(objective)), **result})
        by_task = pd.DataFrame({"task": test["ids"], "objective": objective}).groupby("task")["objective"].mean()
        task_costs.extend({"method": method, "selector": selector, "failure_loss": loss,
                           "task": str(task), "objective": float(value)} for task, value in by_task.items())

    for method, (val_scores, test_scores) in ranking.items():
        for budget in settings["budgets"]:
            threshold = budget_threshold(val_scores, budget, validation["keys"])
            result = route_result(test, route_mask(test_scores, threshold, test["keys"]))
            budgets.append({"method": method, "validation_budget": budget, **result})
        for loss in settings["failure_losses"]:
            if method == "benefit_model":
                record(method, "predicted_benefit", loss,
                       fitted["benefit"]["test"]*loss > settings["alternate_cost"], None, None)
            elif "budget_grid" in settings.get("selectors", ["budget_grid"]):
                candidates = []
                for budget in settings["cost_grid_budgets"]:
                    threshold = budget_threshold(val_scores, budget, validation["keys"])
                    result = route_result(validation, route_mask(val_scores, threshold, validation["keys"]))
                    value = settings["primary_cost"]+settings["alternate_cost"]*result["escalation"]+loss*(1-result["success"])
                    candidates.append((value, budget, threshold))
                _, chosen_budget, threshold = min(candidates, key=lambda row: (row[0], row[1]))
                record(method, "budget_grid", loss, route_mask(test_scores, threshold, test["keys"]), chosen_budget, threshold)
            if "all_validation_boundaries" in settings.get("selectors", []):
                threshold, budget = optimal_validation_cutoff(val_scores, validation, settings, loss)
                record(method, "all_validation_boundaries", loss, route_mask(test_scores, threshold, test["keys"]), budget, threshold)
    for loss in settings["failure_losses"]:
        record("primary_only", "fixed_action", loss, np.zeros(len(test["y"]), dtype=bool), None, None)
        record("alternate_direct", "fixed_action", loss, np.ones(len(test["y"]), dtype=bool), None, None, direct=True)
    return budgets, costs, task_costs


def shift_rows(fitted: dict, protocol: dict) -> list[dict]:
    train, test = fitted["parts"]["train"], fitted["parts"]["test"]
    rows = []
    for feature in protocol["shift"]["features"]:
        column = FEATURES.index(feature)
        training = pd.DataFrame({"task": train["ids"], "value": train["x"][:, column]}).groupby("task")["value"].mean()
        threshold = float(training.median())
        values = pd.DataFrame({"task": test["ids"], "value": test["x"][:, column]}).groupby("task")["value"].mean()
        high = np.array([values.loc[task] > threshold for task in test["ids"]])
        share = high.mean()
        if share == 0 or share == 1:
            continue
        for target in protocol["shift"]["target_high_fractions"]:
            weights = np.where(high, target/share, (1-target)/(1-share))
            task_weights = pd.DataFrame({"task": test["ids"], "weight": weights}).groupby("task")["weight"].first().to_numpy()
            effective = float(task_weights.sum()**2/np.sum(task_weights**2))
            for method, p in fitted["predictions"]["test"].items():
                bounded = np.clip(p, 1e-6, 1-1e-6)
                rows.append({"method": method, "feature": feature, "training_threshold": threshold,
                             "target_high_fraction": target, "effective_tasks": effective,
                             "brier": float(np.average((p-test["y"])**2, weights=weights)),
                             "log_loss": float(np.average(-test["y"]*np.log(bounded)-(1-test["y"])*np.log(1-bounded), weights=weights))})
    return rows


def sample_rows(fitted: dict, protocol: dict, groups: list[str], seed: int) -> list[dict]:
    parts = fitted["parts"]
    discovery = np.random.default_rng(seed).permutation(np.unique(parts["discovery"]["ids"]))
    confirmation = np.random.default_rng(seed+1).permutation(np.unique(parts["confirmation"]["ids"]))
    rows = []
    for size in protocol["sample_sizes"]:
        n = len(discovery) if size is None else min(size, len(discovery))
        count = max(1, int(len(confirmation)*n/len(discovery)))
        smaller = {role: part for role, part in parts.items()}
        smaller["discovery"] = take(parts["discovery"], np.isin(parts["discovery"]["ids"], discovery[:n]))
        smaller["confirmation"] = take(parts["confirmation"], np.isin(parts["confirmation"]["ids"], confirmation[:count]))
        original = fitted["platt_predictions"]["logistic"]
        p = {role: original[role] for role in original}
        p["discovery"] = original["discovery"][np.isin(parts["discovery"]["ids"], discovery[:n])]
        p["confirmation"] = original["confirmation"][np.isin(parts["confirmation"]["ids"], confirmation[:count])]
        scores, fits = fit_calibrators(smaller, p, fitted["group_thresholds"], groups, protocol, seed)
        for method in ("fixed_mc", "tree_mc", "confirmed_tree"):
            pool_p = np.concatenate([scores[role][method] for role in ("discovery", "confirmation")])
            pool_y = np.concatenate([smaller[role]["y"] for role in ("discovery", "confirmation")])
            rows.append({"method": method, "discovery_tasks": n, "confirmation_tasks": count,
                         "auditor_tasks": n+count, "patches": len(fits[method]["patches"]),
                         "fitting_brier": float(np.mean((pool_p-pool_y)**2)),
                         "test_brier": float(np.mean((scores["test"][method]-parts["test"]["y"])**2))})
    return rows


def paired_comparisons(frame: pd.DataFrame, repetitions: int, seed: int) -> pd.DataFrame:
    rows = []
    for method in sorted(frame["method"].unique()):
        base = method.split("/")[0]+"/platt"
        for label, baseline in (("overall", "logistic/platt"), ("within_base", base)):
            left = frame.loc[frame["method"] == method, ["seed", "task", "candidate", "p", "y"]]
            right = frame.loc[frame["method"] == baseline, ["seed", "task", "candidate", "p"]]
            joined = left.merge(right, on=["seed", "task", "candidate"], suffixes=("", "_baseline"), validate="one_to_one")
            joined["difference"] = (joined["p"]-joined["y"])**2-(joined["p_baseline"]-joined["y"])**2
            differences = joined.groupby("task")["difference"].mean().to_numpy()
            draws = np.random.default_rng(seed).integers(0, len(differences), (repetitions, len(differences)))
            interval = np.quantile(differences[draws].mean(axis=1), [.025, .975])
            rows.append({"method": method, "baseline": baseline, "contrast": label,
                         "brier_difference": float(differences.mean()), "low": float(interval[0]), "high": float(interval[1]),
                         "task_clusters": len(differences), "replicates": repetitions,
                         "interval_scope": "Descriptive task bootstrap conditional on overlapping fitted models; excludes refitting, split uncertainty and shared-training dependence"})
    return pd.DataFrame(rows)


def cost_comparisons(frame: pd.DataFrame, repetitions: int, seed: int) -> pd.DataFrame:
    rows = []
    for loss, current in frame.groupby("failure_loss"):
        baseline_selector = "all_validation_boundaries" if "all_validation_boundaries" in current["selector"].values else "budget_grid"
        reference = current[(current["method"] == "logistic/platt") & (current["selector"] == baseline_selector)]
        for (method, selector), selected in current.groupby(["method", "selector"]):
            paired = selected.merge(reference[["seed", "task", "objective"]], on=["seed", "task"],
                                    suffixes=("", "_reference"), validate="one_to_one")
            paired["difference"] = paired["objective"]-paired["objective_reference"]
            values = paired.groupby("task")["difference"].mean().to_numpy()
            draws = np.random.default_rng(seed).integers(0, len(values), (repetitions, len(values)))
            low, high = np.quantile(values[draws].mean(axis=1), [.025, .975])
            rows.append({"method": method, "selector": selector, "failure_loss": loss,
                         "baseline": "logistic/platt", "baseline_selector": baseline_selector,
                         "objective_difference": float(values.mean()), "low": float(low), "high": float(high),
                         "task_clusters": len(values),
                         "interval_scope": "Descriptive task bootstrap conditional on overlapping fitted models; excludes refitting, split uncertainty and shared-training dependence"})
    return pd.DataFrame(rows)


def run_advanced(output: Path, protocol_path: Path = ROOT / "advanced_protocol.json", smoke: bool = False) -> dict:
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new output directory")
    if output == REFERENCE.resolve() or REFERENCE.resolve() in output.parents:
        raise ValueError("Advanced runs cannot overwrite the original reference")
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()
    code_hashes = {f"research/{path.name}": digest(path) for path in sorted(ROOT.glob("*.py"))}
    reference = load_reference()
    data = archive_data(reference)
    plans = task_folds(data, protocol)
    if smoke:
        plans = plans[:1]
    groups = reference["protocol"]["groups"]
    fold_metrics, predictions, routes, costs, task_costs, shifts, samples, auditors = [], [], [], [], [], [], [], []
    for plan in plans:
        print(f"seed={plan['seed']} fold={plan['fold']}", flush=True)
        fitted = fit_fold(data, plan, protocol, groups)
        test = fitted["parts"]["test"]
        prefix = {"seed": plan["seed"], "fold": plan["fold"]}
        test_groups = membership(test["x"], fitted["group_thresholds"], groups)
        for method, p in fitted["predictions"]["test"].items():
            measured = metrics(p, test["y"], test_groups, test["ids"], groups, protocol["auditor"]["bins"], protocol["fixed_groups"]["min_tasks"])
            fold_metrics.append({**prefix, "method": method, "tasks": len(np.unique(test["ids"])),
                                 "candidates": len(p), "auc": float(roc_auc_score(test["y"], p)),
                                 **{key: value for key, value in measured.items() if key != "cells"}})
            for i, probability in enumerate(p):
                predictions.append({**prefix, "method": method, "task": str(test["ids"][i]),
                                    "candidate": int(test["candidate"][i]), "y": int(test["y"][i]),
                                    "alternate": int(test["alternate"][i]), "p": float(probability),
                                    "tie_key": float(test["keys"][i]),
                                    "estimated_benefit": float(fitted["benefit"]["test"][i])})
        budget_rows, cost_rows, task_rows = routing_tables(fitted, protocol)
        routes.extend({**prefix, **row} for row in budget_rows)
        costs.extend({**prefix, **row} for row in cost_rows)
        task_costs.extend({**prefix, **row} for row in task_rows)
        shifts.extend({**prefix, **row} for row in shift_rows(fitted, protocol))
        samples.extend({**prefix, **row} for row in sample_rows(fitted, protocol, groups, plan["seed"]+plan["fold"]))
        auditors.append({**prefix, "models": fitted["details"]})
    output.mkdir(parents=True, exist_ok=True)
    frames = {"fold_metrics": pd.DataFrame(fold_metrics), "routing": pd.DataFrame(routes),
              "decision_costs": pd.DataFrame(costs), "mixture_shift": pd.DataFrame(shifts),
              "sample_complexity": pd.DataFrame(samples)}
    frame = pd.DataFrame(predictions)
    frames["paired_comparisons"] = paired_comparisons(frame, protocol["bootstrap_replicates"], protocol["seeds"][0])
    cost_frame = pd.DataFrame(task_costs)
    frames["cost_comparisons"] = cost_comparisons(cost_frame, protocol["bootstrap_replicates"], protocol["seeds"][0])
    frames["repeat_summary"] = frames["fold_metrics"].groupby(["seed", "method"], as_index=False)[
        ["brier", "log_loss", "auc", "ece", "worst"]].mean()
    for name, table in frames.items():
        table.to_csv(output / f"{name}.csv", index=False)
    with gzip.open(output / "predictions.csv.gz", "wt", newline="") as handle:
        frame.to_csv(handle, index=False)
    with gzip.open(output / "task_costs.csv.gz", "wt", newline="") as handle:
        cost_frame.to_csv(handle, index=False)
    (output / "protocol.json").write_bytes(protocol_bytes)
    for name, value in (("splits", plans), ("auditor_rules", auditors)):
        (output / f"{name}.json").write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "smoke": smoke,
              "scope": protocol["status"], "reference_run": reference["manifest"]["run_id"],
              "protocol_sha256": protocol_hash, "python": platform.python_version(),
              "platform": platform.platform(), "folds": len(plans), "prediction_rows": len(frame),
              "task_cost_rows": len(cost_frame),
              "task_clusters": int(frame["task"].nunique()),
              "dependencies": {name: importlib.metadata.version(name) for name in
                               ("numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl")},
              "code_sha256": code_hashes,
              "interval_scope": "Descriptive task bootstrap conditional on overlapping fitted models; excludes refitting, split uncertainty and shared-training dependence",
              "tables": {name: len(table) for name, table in frames.items()},
              "caveats": protocol["restrictions"]}
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    manifest = {path.name: digest(path) for path in output.iterdir() if path.is_file()}
    (output / "manifest.json").write_text(json.dumps({"files": manifest}, indent=2)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Repeated grouped study of learned auditing and decision-aware routing")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "advanced_protocol.json")
    parser.add_argument("--smoke", action="store_true", help="Only the first outer fold; never a full-study result")
    args = parser.parse_args()
    print(json.dumps(run_advanced(args.output, args.protocol, args.smoke), indent=2))

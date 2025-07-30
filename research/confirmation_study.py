from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import beta, norm
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .advanced import archive_data, take
from .auditors import rule_mask
from .confirmation import METHODS, apply_confirmation, fit_confirmation
from .data import FEATURES, ROOT, digest
from .methods import export_classifier, group_thresholds, membership, metrics, predict_classifier
from .reference import load_reference

PROTOCOL = ROOT / "confirmation_protocol.json"
ROLES = ("discovery", "estimation", "confirmation", "validation")
PILOT_SEEDS = (17011, 29009)
CORE_FILES = ("confirmation.py", "confirmation_study.py", "external.py", "auditors.py",
              "advanced.py", "methods.py", "data.py", "reference.py")


def role_counts(budget: int, fractions: dict) -> dict[str, int]:
    counts = {role: int(budget * fractions[role]) for role in ROLES[:-1]}
    counts["validation"] = budget - sum(counts.values())
    return counts


def role_ids(ids: np.ndarray, budget: int, fractions: dict, seed: int) -> dict[str, list[str]]:
    shuffled = np.random.default_rng(seed).permutation(np.unique(ids))
    if len(shuffled) < budget:
        raise ValueError("Not enough independent tasks for the requested budget")
    result, start = {}, 0
    for role, count in role_counts(budget, fractions).items():
        result[role] = [str(value) for value in shuffled[start:start + count]]
        start += count
    return result


def budget_parts(data: dict, masters: dict, budget: int, fractions: dict) -> dict:
    counts = role_counts(budget, fractions)
    if any(counts[role] > len(masters[role]) for role in ROLES):
        raise ValueError("Nested task budget exceeds its master role")
    return {role: take(data, np.isin(data["ids"], masters[role][:counts[role]])) for role in ROLES}


def source_plans(data: dict, protocol: dict) -> list[dict]:
    settings = protocol["source"]
    seed = settings["seed"]
    tests = np.array_split(np.random.default_rng(seed).permutation(np.unique(data["ids"])),
                           settings["outer_folds"])
    plans = []
    for fold, test in enumerate(tests):
        remaining = np.random.default_rng(seed + 1009 * (fold + 1)).permutation(
            np.setdiff1d(np.unique(data["ids"]), test))
        train_count = int(len(remaining) * settings["train_fraction"])
        calibration_count = int(len(remaining) * settings["score_calibration_fraction"])
        end = train_count + calibration_count
        masters = role_ids(remaining[end:], max(settings["budgets"]),
                           protocol["budget_fractions"], seed + fold)
        plans.append({"fold": fold, "train": remaining[:train_count].tolist(),
                      "score_calibration": remaining[train_count:end].tolist(),
                      "test": test.tolist(), "audit_roles": masters})
    return plans


def fit_predictor(train: dict, calibration: dict, settings: dict, seed: int) -> dict:
    with threadpool_limits(limits=1):
        model = make_pipeline(StandardScaler(), LogisticRegression(**settings["base_model"], random_state=seed))
        model.fit(train["x"], train["y"])
        exported = export_classifier(model, FEATURES)
        raw = predict_classifier(exported, calibration["x"])
        platt = LogisticRegression(**settings["platt"], random_state=seed).fit(
            logit(np.clip(raw, 1e-6, 1 - 1e-6)).reshape(-1, 1), calibration["y"])
    return {"logistic": exported, "platt": {"slope": float(platt.coef_[0, 0]),
                                              "intercept": float(platt.intercept_[0])}}


def base_scores(x: np.ndarray, fitted: dict) -> tuple[np.ndarray, np.ndarray]:
    raw = predict_classifier(fitted["logistic"], x)
    scaling = fitted["platt"]
    platt = expit(scaling["slope"] * logit(np.clip(raw, 1e-6, 1 - 1e-6)) + scaling["intercept"])
    return raw, platt


def loss_metrics(part: dict, scores: np.ndarray, rule: dict | None, bins: int) -> dict:
    y = part["y"]
    bounded = np.clip(scores, 1e-6, 1 - 1e-6)
    result = {"tasks": int(len(np.unique(part["ids"]))), "candidates": len(y),
              "brier": float(np.mean((scores - y) ** 2)),
              "log_loss": float(np.mean(-y * np.log(bounded) - (1 - y) * np.log(1 - bounded))),
              "any_update": bool(np.any(scores != part["p"]))}
    mask = (rule_mask(part["x"], part["p"], rule, bins) if rule is not None
            else np.zeros(len(y), dtype=bool))
    result.update(selected_rule_tasks=int(len(np.unique(part["ids"][mask]))),
                  selected_rule_candidates=int(mask.sum()),
                  selected_rule_pre_residual=float(np.mean(y[mask] - part["p"][mask])) if mask.any() else None,
                  selected_rule_post_residual=float(np.mean(y[mask] - scores[mask])) if mask.any() else None)
    if "truth" in part:
        initial_mse = float(np.mean((part["p"] - part["truth"]) ** 2))
        known_mse = float(np.mean((scores - part["truth"]) ** 2))
        residual = float(np.mean(part["truth"][mask] - part["p"][mask])) if mask.any() else None
        result.update(known_probability_mse=known_mse, initial_probability_mse=initial_mse,
                      harmful_update=bool(known_mse > initial_mse + 1e-12),
                      selected_rule_true_residual=residual)
    else:
        nonempty = part["x"][:, FEATURES.index("empty_code")] == 0
        result.update(nonempty_candidates=int(nonempty.sum()),
                      nonempty_brier=float(np.mean((scores[nonempty] - y[nonempty]) ** 2)) if nonempty.any() else None)
    return result


def accounting(fitted: dict, method: str) -> dict:
    item = fitted["methods"][method]
    return {"eligible": bool(item["eligible"]), "proposal_eligible": bool(fitted["eligible"]),
            "proposal_found": fitted["rule"] is not None, "proposal_status": fitted["status"],
            "gate_accepted": bool(fitted["gate"]["accepted"]), "policy_uses_gate": bool(item["gated"]),
            "step": item["step"], "delta": item["delta"],
            "discovery_support": fitted["support"]["discovery"],
            "estimation_support": fitted["support"]["estimation"],
            "matched_estimation_tasks": fitted["matched_estimation"]["tasks"],
            "discovery_estimation_candidates": fitted["matched_estimation"]["discovery_candidates"],
            "independent_estimation_candidates": fitted["matched_estimation"]["estimation_candidates"],
            "confirmation_support": fitted["support"]["confirmation"],
            "available_label_tasks": fitted["available_label_tasks"]["total"],
            "queried_label_tasks": item["label_tasks"]["total"],
            "queried_with_diagnostics": item["actual_label_tasks"]["total"]}


def interval(values: np.ndarray, protocol: dict) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Paired intervals need finite, nonempty losses")
    draws = np.random.default_rng(protocol["bootstrap_seed"]).integers(
        0, len(values), (protocol["bootstrap_replicates"], len(values)))
    low, high = np.quantile(values[draws].mean(axis=1), [.025, .975])
    return float(values.mean()), float(low), float(high)


def contrasts(frame: pd.DataFrame, keys: list[str], unit: str,
              metric: str, protocol: dict) -> pd.DataFrame:
    rows = []
    groups = frame.groupby(keys, dropna=False) if keys else [((), frame)]
    for group, current in groups:
        group = group if isinstance(group, tuple) else (group,)
        identifiers = dict(zip(keys, group))
        wide = current.pivot(index=unit, columns="method", values=metric)
        if wide[list(METHODS)].isna().any().any():
            raise ValueError("Paired contrast is missing a method or replication")
        for comparison in protocol["contrasts"]:
            left, right = comparison["left"], comparison["right"]
            difference, low, high = interval((wide[left] - wide[right]).to_numpy(), protocol)
            rows.append({**identifiers, "contrast": comparison["name"], "left": left, "right": right,
                         "metric": metric, "difference": difference, "low": low, "high": high,
                         "seeds" if unit == "seed" else "task_clusters": len(wide), "primary": comparison["primary"]})
        interaction = ((wide["independent_gated"] - wide["independent_fixed"])
                       - (wide["discovery_gated"] - wide["discovery_fixed"]))
        difference, low, high = interval(interaction.to_numpy(), protocol)
        rows.append({**identifiers, "contrast": "gate_estimator_interaction", "metric": metric,
                     "left": "independent gate effect", "right": "discovery gate effect",
                     "difference": difference, "low": low, "high": high,
                     "seeds" if unit == "seed" else "task_clusters": len(wide), "primary": False})
    return pd.DataFrame(rows)


def save_frame(frame: pd.DataFrame, path: Path) -> None:
    compression = {"method": "gzip", "mtime": 0} if path.suffix == ".gz" else None
    frame.to_csv(path, index=False, compression=compression)


def source_experiment(data: dict, groups: list[str], protocol: dict,
                      output: Path, smoke: bool) -> list[dict]:
    plans = source_plans(data, protocol)
    selected_plans = plans[:1] if smoke else plans
    budgets = protocol["source"]["budgets"][-1:] if smoke else protocol["source"]["budgets"]
    fitted_models, rows, predictions = [], [], []
    for plan in selected_plans:
        fold = plan["fold"]
        train = take(data, np.isin(data["ids"], plan["train"]))
        calibration = take(data, np.isin(data["ids"], plan["score_calibration"]))
        base = fit_predictor(train, calibration, protocol["source"], protocol["source"]["seed"] + fold)
        raw, p = base_scores(data["x"], base)
        scored = {**data, "p": p}
        test_mask = np.isin(data["ids"], plan["test"])
        test = take(scored, test_mask)
        thresholds = group_thresholds(train["x"])
        cells = membership(test["x"], thresholds, groups)
        record = {"fold": fold, "base": base, "group_thresholds": thresholds, "budgets": {}}
        for budget in budgets:
            parts = budget_parts(scored, plan["audit_roles"], budget, protocol["budget_fractions"])
            fitted = fit_confirmation(**parts, settings=protocol["auditor"], seed=protocol["source"]["seed"] + fold)
            record["budgets"][str(budget)] = fitted
            scores = {**apply_confirmation(test["x"], test["p"], fitted), "raw_logistic": raw[test_mask]}
            for method, values in scores.items():
                fit = fitted["methods"].get(method)
                rule = (fit["rule"] if fit is not None else None) or fitted["rule"]
                measured = loss_metrics(test, values, rule, protocol["auditor"]["bins"])
                if method == "raw_logistic":
                    measured["any_update"] = False
                calibration_metrics = metrics(values, test["y"], cells, test["ids"], groups,
                                               protocol["auditor"]["bins"], 20)
                rows.append({"fold": fold, "budget": budget, "method": method, **measured,
                             "fixed_group_score_cell_error": calibration_metrics["worst"],
                             "ece": calibration_metrics["ece"],
                             **(accounting(fitted, method) if fit is not None else {})})
                predictions.extend({"fold": fold, "budget": budget, "method": method,
                                    "task": str(task), "candidate": int(candidate), "y": int(y),
                                    "p": float(value), "initial_p": float(initial), "empty_code": int(empty)}
                                   for task, candidate, y, value, initial, empty in zip(
                                       test["ids"], test["candidate"], test["y"], values, test["p"],
                                       test["x"][:, FEATURES.index("empty_code")]))
        fitted_models.append(record)
        print(f"Source fold {fold + 1}/{len(selected_plans)} fitted", flush=True)
    frame = pd.DataFrame(predictions)
    frame["brier"] = (frame["p"] - frame["y"]) ** 2
    bounded = frame["p"].clip(1e-6, 1 - 1e-6)
    frame["log_loss"] = -frame["y"] * np.log(bounded) - (1 - frame["y"]) * np.log(1 - bounded)
    losses = frame.groupby(["budget", "method", "task"], as_index=False)[["brier", "log_loss"]].mean()
    save_frame(frame, output / "source_predictions.csv.gz")
    save_frame(losses, output / "source_task_losses.csv.gz")
    save_frame(pd.DataFrame(rows), output / "source_metrics.csv")
    save_frame(contrasts(losses, ["budget"], "task", "brier", protocol), output / "source_contrasts.csv")
    (output / "splits.json").write_text(json.dumps(selected_plans, indent=2) + "\n")
    (output / "fitted_models.json").write_text(json.dumps(fitted_models, indent=2) + "\n")
    return fitted_models


def synthetic_data(tasks: int, seed: int, scenario: dict, shared: bool,
                   candidates: int = 10) -> dict:
    rng = np.random.default_rng(seed)
    task_x = rng.normal(size=(tasks, 16))
    x = (np.repeat(task_x, candidates, axis=0)
         + .25 * rng.normal(size=(tasks * candidates, 16))) / np.sqrt(1.0625)
    p = expit(.7 * x[:, 0])
    cutoff = norm.isf(np.sqrt(scenario["prevalence"]))
    region = (x[:, 1] > cutoff) & (x[:, 2] > cutoff)
    truth = np.clip(p - scenario["effect"] * region, 0, 1)
    individual = rng.uniform(size=tasks * candidates)
    common = np.repeat(rng.uniform(size=tasks), candidates)
    y = ((common if shared else individual) < truth).astype(int)
    return {"x": x, "p": p, "truth": truth, "y": y,
            "ids": np.repeat([f"{seed}:{task}" for task in range(tasks)], candidates)}


def binomial_interval(successes: int, trials: int) -> tuple[float, float]:
    if not 0 <= successes <= trials or trials < 1:
        raise ValueError("Invalid binomial counts")
    low = 0.0 if successes == 0 else float(beta.ppf(.025, successes, trials - successes + 1))
    high = 1.0 if successes == trials else float(beta.ppf(.975, successes + 1, trials - successes))
    return low, high


def summarize_simulations(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["scenario", "shared_uniform", "budget", "method"]
    for identifiers, current in frame.groupby(keys):
        low, high = binomial_interval(int(current["any_update"].sum()), len(current))
        mse = current["known_probability_mse"]
        eligible = current[current["proposal_eligible"]]
        signal = current[current["proposal_eligible"] & current["selected_rule_signal"]]
        uses_gate = bool(current["policy_uses_gate"].iloc[0])
        rows.append({**dict(zip(keys, identifiers)), "known_probability_mse": float(mse.mean()),
                     "known_probability_mse_se": float(mse.std(ddof=1) / np.sqrt(len(current))),
                     "brier": float(current["brier"].mean()), "update_rate": float(current["any_update"].mean()),
                     "update_low": low, "update_high": high,
                     "harmful_update_rate": float(current["harmful_update"].mean()),
                     "eligibility_rate": float(current["eligible"].mean()),
                     "common_discovery_rate": float(current["proposal_found"].mean()),
                     "common_proposal_eligibility_rate": float(current["proposal_eligible"].mean()),
                     "gate_acceptance_rate": float(current["gate_accepted"].mean()) if uses_gate else None,
                     "gate_acceptance_when_eligible": float(eligible["gate_accepted"].mean()) if uses_gate and len(eligible) else None,
                     "gate_acceptance_when_signal": float(signal["gate_accepted"].mean()) if uses_gate and len(signal) else None,
                     "eligible_signal_seeds": len(signal) if uses_gate else None, "seeds": len(current)})
    return pd.DataFrame(rows)


def simulation_experiment(protocol: dict, output: Path, smoke: bool) -> None:
    config = protocol["simulation"]
    seeds = list(PILOT_SEEDS) if smoke else config["seeds"]
    scenarios = config["scenarios"][:2] if smoke else config["scenarios"]
    budgets = config["budgets"][-1:] if smoke else config["budgets"]
    maximum = max(config["budgets"])
    rows = []
    with (output / "simulation_fits.jsonl.gz").open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as traces:
            for index, seed in enumerate(seeds):
                masters = None
                for scenario in scenarios:
                    for shared in config["shared_label_uniform"]:
                        data = synthetic_data(maximum, seed, scenario, shared, config["candidates_per_task"])
                        test = synthetic_data(config["test_tasks"], seed + 40000000, scenario, shared,
                                              config["candidates_per_task"])
                        if masters is None:
                            masters = role_ids(data["ids"], maximum, protocol["budget_fractions"], seed + 7)
                        for budget in budgets:
                            parts = budget_parts(data, masters, budget, protocol["budget_fractions"])
                            fitted = fit_confirmation(**parts, settings=protocol["auditor"], seed=seed)
                            identity = {"seed": seed, "scenario": scenario["name"],
                                        "shared_uniform": shared, "budget": budget}
                            traces.write((json.dumps({**identity, "fit": fitted}, separators=(",", ":")) + "\n").encode())
                            for method, scores in apply_confirmation(test["x"], test["p"], fitted).items():
                                rule = fitted["methods"][method]["rule"] or fitted["rule"]
                                measured = loss_metrics(test, scores, rule, protocol["auditor"]["bins"])
                                residual = measured["selected_rule_true_residual"]
                                rows.append({**identity, "method": method, **measured,
                                             "selected_rule_signal": residual is not None and abs(residual) > protocol["auditor"]["epsilon"],
                                             **accounting(fitted, method)})
                if index == 0 or (index + 1) % 10 == 0 or index + 1 == len(seeds):
                    print(f"Synthetic fitting seeds {index + 1}/{len(seeds)} complete", flush=True)
    frame = pd.DataFrame(rows)
    save_frame(frame, output / "simulation_runs.csv")
    save_frame(summarize_simulations(frame), output / "simulation_summary.csv")
    save_frame(contrasts(frame, ["scenario", "shared_uniform", "budget"], "seed",
                         "known_probability_mse", protocol), output / "simulation_contrasts.csv")


def external_experiment(directory: Path, protocol: dict, output: Path) -> dict:
    from .external import load_external

    seal_path = output / "source_freeze.json"
    seal = json.loads(seal_path.read_text())
    if digest(output / "fitted_models.json") != seal["fitted_models_sha256"]:
        raise ValueError("Frozen source models changed before external evaluation")
    if (digest(output / "protocol.json") != seal["protocol_sha256"]
            or json.loads((output / "protocol.json").read_text()) != protocol):
        raise ValueError("Frozen protocol changed before external evaluation")
    if seal["code_sha256"] != {f"research/{name}": digest(ROOT / name) for name in CORE_FILES}:
        raise ValueError("Study implementation changed before external evaluation")
    models = json.loads((output / "fitted_models.json").read_text())
    expected_folds = protocol["source"]["outer_folds"]
    budget = str(protocol["source"]["external_primary_budget"])
    if (len(models) != expected_folds or {model["fold"] for model in models} != set(range(expected_folds))
            or any(budget not in model["budgets"] for model in models)):
        raise ValueError("External evaluation requires every frozen source fold and the primary budget")
    external = load_external(directory)
    if external["manifest"]["protocol_sha256"] != digest(output / "protocol.json"):
        raise ValueError("External extraction used a different frozen protocol")
    rows = external["rows"]
    part = {"x": np.array([row["features"] for row in rows]),
            "y": np.array([row["label"] for row in rows]),
            "ids": np.array([row["task_id"] for row in rows])}
    if len(np.unique(part["ids"])) != protocol["external"]["expected_tasks"]:
        raise ValueError("Unexpected external task count")
    all_scores = {method: [] for method in (*METHODS, "raw_logistic")}
    for model in models:
        raw, p = base_scores(part["x"], model["base"])
        predicted = apply_confirmation(part["x"], p, model["budgets"][budget])
        for method, scores in {**predicted, "raw_logistic": raw}.items():
            all_scores[method].append(scores)
    ensemble = {method: np.mean(values, axis=0) for method, values in all_scores.items()}
    part["p"] = ensemble["no_update"]
    thresholds = {name: float(np.median([model["group_thresholds"][name] for model in models]))
                  for name in models[0]["group_thresholds"]}
    groups = load_reference()["protocol"]["groups"]
    cells = membership(part["x"], thresholds, groups)
    measured, task_losses = [], []
    for method, scores in ensemble.items():
        diagnostics = loss_metrics(part, scores, None, protocol["auditor"]["bins"])
        if method == "raw_logistic":
            diagnostics["any_update"] = False
        calibrated = metrics(scores, part["y"], cells, part["ids"], groups, protocol["auditor"]["bins"], 20)
        measured.append({"method": method, **diagnostics, "ece": calibrated["ece"],
                         "fixed_group_score_cell_error": calibrated["worst"]})
        per_task = pd.DataFrame({"task": part["ids"], "brier": (scores - part["y"]) ** 2}).groupby("task")["brier"].mean()
        task_losses.extend({"task": task, "method": method, "brier": float(value)} for task, value in per_task.items())
    save_frame(pd.DataFrame(measured), output / "external_metrics.csv")
    save_frame(contrasts(pd.DataFrame(task_losses), [], "task", "brier", protocol), output / "external_contrasts.csv")
    record = {"evaluated_at": datetime.now(timezone.utc).isoformat(), "tasks": len(np.unique(part["ids"])),
              "candidates": len(rows), "source_folds_averaged": len(models),
              "source_budget": int(budget), "source_freeze_sha256": digest(seal_path),
              "extraction_manifest_sha256": digest(directory / "manifest.json"),
              "extraction_rows_sha256": digest(directory / "rows.json.gz"),
              "source": protocol["external"], "publication": "Aggregate results only; no external row-level records published",
              "overlap_audit": "Task identities use distinct namespaces; exact source-prompt overlap not checked because original source prompts are not cached; semantic overlap and pretraining contamination remain unverified",
              "interval_scope": "Descriptive task bootstrap for 161 tasks conditional on the frozen five-pipeline ensemble"}
    (output / "external_record.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def run_study(output: Path, protocol_path: Path = PROTOCOL, external: Path | None = None,
              smoke: bool = False) -> dict:
    if smoke and external is not None:
        raise ValueError("Smoke runs never evaluate the held-out external benchmark")
    output = output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Use a fresh output directory")
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    if set(protocol["experimental_methods"]) != set(METHODS):
        raise ValueError("Protocol methods do not match the implementation")
    output.mkdir(parents=True, exist_ok=True)
    (output / "protocol.json").write_bytes(protocol_bytes)
    code = {f"research/{name}": digest(ROOT / name) for name in CORE_FILES}
    reference = load_reference()
    data = archive_data(reference)
    models = source_experiment(data, reference["protocol"]["groups"], protocol, output, smoke)
    (output / "source_freeze.json").write_text(json.dumps({
        "frozen_at": datetime.now(timezone.utc).isoformat(), "protocol_sha256": digest(output / "protocol.json"),
        "fitted_models_sha256": digest(output / "fitted_models.json"), "code_sha256": code,
        "external_outcomes_loaded_by_driver": False,
    }, indent=2) + "\n")
    simulation_experiment(protocol, output, smoke)
    external_record = external_experiment(external, protocol, output) if external is not None else None
    if code != {f"research/{name}": digest(ROOT / name) for name in CORE_FILES}:
        raise ValueError("Study implementation changed during evaluation; this run is incomplete")
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "smoke": smoke,
              "protocol_sha256": digest(output / "protocol.json"), "protocol_commit": "41d1599",
              "code_sha256": code, "python": platform.python_version(),
              "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn")},
              "source_folds": len(models), "source_task_clusters": len(np.unique(data["ids"])),
              "simulation_seeds": list(PILOT_SEEDS) if smoke else protocol["simulation"]["seeds"],
              "external_evaluated": external_record is not None,
              "scope": "One-correction mechanism and budget comparison; source exploratory, simulations newly replicated, external frozen-transfer evaluation",
              "intervals": {"source": "Descriptive task bootstrap conditional on overlapping fitted models",
                            "simulation": "Paired bootstrap across independent fitting seeds within a setting; no simultaneous coverage",
                            "external": "Descriptive task bootstrap conditional on the frozen ensemble"}}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    files = {path.name: digest(path) for path in sorted(output.iterdir()) if path.is_file()}
    (output / "manifest.json").write_text(json.dumps({"files": files}, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Matched confirmation, independent estimation and shrinkage study")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--external", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_study(args.output, args.protocol, args.external, args.smoke), indent=2))

from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from .data import FEATURES
from .methods import bin_ids


def rule_mask(x: np.ndarray, scores: np.ndarray, rule: dict, bins: int) -> np.ndarray:
    mask = bin_ids(scores, bins) == rule["bin"]
    for condition in rule["conditions"]:
        value = x[:, condition["column"]].astype(np.float32)
        mask &= value <= condition["threshold"] if condition["side"] == "le" else value > condition["threshold"]
    return mask


def leaf_rules(tree: DecisionTreeRegressor, node: int = 0, path: list | None = None) -> list[list]:
    path = [] if path is None else path
    left, right = tree.tree_.children_left[node], tree.tree_.children_right[node]
    if left == right:
        return [path]
    column, threshold = int(tree.tree_.feature[node]), float(tree.tree_.threshold[node])
    condition = {"column": column, "feature": FEATURES[column], "threshold": threshold}
    return leaf_rules(tree, left, path + [{**condition, "side": "le"}]) + leaf_rules(
        tree, right, path + [{**condition, "side": "gt"}],
    )


def find_rule(x: np.ndarray, y: np.ndarray, scores: np.ndarray, ids: np.ndarray,
              settings: dict, seed: int) -> dict | None:
    candidates = []
    residuals = y-scores
    for band in range(settings["bins"]):
        eligible = bin_ids(scores, settings["bins"]) == band
        if eligible.sum() < settings["min_samples_leaf"]:
            continue
        tree = DecisionTreeRegressor(max_depth=settings["max_depth"],
                                     min_samples_leaf=settings["min_samples_leaf"], random_state=seed)
        tree.fit(x[eligible], residuals[eligible])
        for conditions in leaf_rules(tree):
            rule = {"bin": band, "conditions": conditions}
            mask = rule_mask(x, scores, rule, settings["bins"])
            tasks = len(np.unique(ids[mask]))
            if tasks < settings["min_discovery_tasks"]:
                continue
            residual = float(residuals[mask].mean())
            if abs(residual) <= settings["epsilon"]:
                continue
            candidates.append({**rule, "discovery_tasks": tasks, "discovery_candidates": int(mask.sum()),
                               "discovery_residual": residual, "priority": float(mask.mean()*residual**2)})
    return max(candidates, key=lambda rule: rule["priority"]) if candidates else None


def residual_interval(values: np.ndarray, ids: np.ndarray, seed: int,
                      repetitions: int, alpha: float) -> dict:
    unique, inverse = np.unique(ids, return_inverse=True)
    sums, counts = np.bincount(inverse, weights=values), np.bincount(inverse)
    draws = np.random.default_rng(seed).integers(0, len(unique), (repetitions, len(unique)))
    samples = sums[draws].sum(axis=1)/counts[draws].sum(axis=1)
    low, high = np.quantile(samples, [alpha/2, 1-alpha/2])
    return {"estimate": float(values.mean()), "low": float(low), "high": float(high),
            "tasks": len(unique), "interval_level": 1-alpha}


def apply_auditor(x: np.ndarray, scores: np.ndarray, fitted: dict) -> np.ndarray:
    result = scores.copy()
    for patch in fitted["patches"]:
        mask = rule_mask(x, result, patch, fitted["bins"])
        result[mask] = np.clip(result[mask]+patch["delta"], 0, 1)
    return result


def fit_auditor(discovery: dict, confirmation: dict | None, settings: dict, seed: int) -> dict:
    scores = discovery["p"].copy()
    patches, trace = [], []
    blocks = []
    if confirmation is not None:
        unique = np.random.default_rng(seed).permutation(np.unique(confirmation["ids"]))
        blocks = list(np.array_split(unique, settings["max_rounds"]))
    status = "round_limit"
    for iteration in range(settings["max_rounds"]):
        rule = find_rule(discovery["x"], discovery["y"], scores, discovery["ids"], settings, seed+iteration)
        if rule is None:
            status = "no_discovered_violation"
            break
        record = {"round": iteration, **rule, "accepted": False}
        if confirmation is None:
            estimate, tasks = rule["discovery_residual"], rule["discovery_tasks"]
        else:
            block = blocks[iteration]
            available = np.isin(confirmation["ids"], block)
            current = apply_auditor(confirmation["x"], confirmation["p"],
                                    {"bins": settings["bins"], "patches": patches})
            mask = available & rule_mask(confirmation["x"], current, rule, settings["bins"])
            tasks = len(np.unique(confirmation["ids"][mask]))
            record.update({"confirmation_block": [str(value) for value in block],
                           "confirmation_tasks": tasks, "confirmation_candidates": int(mask.sum())})
            if tasks < settings["min_confirmation_tasks"]:
                trace.append({**record, "reason": "insufficient_confirmation_support"})
                continue
            interval = residual_interval(confirmation["y"][mask]-current[mask], confirmation["ids"][mask],
                                         seed+iteration, settings["bootstrap_replicates"],
                                         settings["alpha"]/settings["max_rounds"])
            record["confirmation_interval"] = interval
            estimate = interval["estimate"]
            agrees = estimate*rule["discovery_residual"] > 0
            excludes_zero = interval["low"] > 0 or interval["high"] < 0
            if not agrees or not excludes_zero:
                trace.append({**record, "reason": "unconfirmed_residual"})
                continue
        delta = estimate*tasks/(tasks+settings["shrinkage_tasks"])
        patch = {**rule, "delta": float(delta), "estimation_tasks": tasks, "round": iteration}
        patches.append(patch)
        mask = rule_mask(discovery["x"], scores, patch, settings["bins"])
        scores[mask] = np.clip(scores[mask]+delta, 0, 1)
        trace.append({**record, "accepted": True, "delta": float(delta), "reason": "patched"})
    return {"bins": settings["bins"], "patches": patches, "trace": trace, "status": status,
            "confirmation_blocks": [[str(value) for value in block] for block in blocks],
            "settings": settings, "confirmation_is_approximate": confirmation is not None}

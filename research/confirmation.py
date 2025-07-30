from __future__ import annotations

import numpy as np

from .auditors import find_rule, residual_interval, rule_mask


METHODS = (
    "no_update", "discovery_fixed", "independent_fixed", "discovery_gated",
    "independent_gated", "discovery_shrinkage", "independent_shrinkage", "pooled_shrinkage",
)


def _apply(x: np.ndarray, p: np.ndarray, rule: dict | None, delta: float, bins: int) -> np.ndarray:
    result = np.clip(p, 0, 1).copy()
    if rule is not None and delta != 0:
        mask = rule_mask(x, p, rule, bins)
        result[mask] = np.clip(p[mask] + delta, 0, 1)
    return result


def apply_confirmation(x: np.ndarray, p: np.ndarray, fitted: dict) -> dict[str, np.ndarray]:
    return {name: _apply(x, p, method["rule"], method["delta"], fitted["bins"])
            for name, method in fitted["methods"].items()}


def _task_count(ids: np.ndarray) -> int:
    return int(len(np.unique(ids)))


def fit_confirmation(discovery: dict, estimation: dict, confirmation: dict,
                     validation: dict, settings: dict, seed: int) -> dict:
    parts = {"discovery": discovery, "estimation": estimation,
             "confirmation": confirmation, "validation": validation}
    task_sets = {role: set(part["ids"]) for role, part in parts.items()}
    roles = list(parts)
    if any(task_sets[left] & task_sets[right] for i, left in enumerate(roles) for right in roles[i+1:]):
        raise ValueError("Discovery, estimation, confirmation and validation tasks must be disjoint")
    if not len(validation["y"]):
        raise ValueError("Validation needs at least one answer")
    if 0 not in settings["step_grid"]:
        raise ValueError("The validation step grid must include zero")

    available = {role: len(ids) for role, ids in task_sets.items()}
    available["total"] = sum(available.values())
    bins = settings["bins"]
    rule = find_rule(discovery["x"], discovery["y"], discovery["p"],
                     discovery["ids"], settings, seed)
    masks = {role: (rule_mask(part["x"], part["p"], rule, bins)
                    if rule is not None else np.zeros(len(part["p"]), dtype=bool))
             for role, part in parts.items() if role != "validation"}
    support = {role: _task_count(parts[role]["ids"][mask]) for role, mask in masks.items()}
    matched_tasks = min(support["discovery"], support["estimation"])
    estimate_masks, estimate_ids = {}, {}
    selection_seeds = {"discovery": seed + 17011, "estimation": seed + 29009}
    for role in ("discovery", "estimation"):
        ids = np.unique(parts[role]["ids"][masks[role]])
        selected = np.random.default_rng(selection_seeds[role]).permutation(ids)[:matched_tasks]
        estimate_ids[role] = [str(value) for value in selected]
        estimate_masks[role] = masks[role] & np.isin(parts[role]["ids"], selected)
    matched = {"tasks": matched_tasks, "selection_seeds": selection_seeds,
               "discovery_ids": estimate_ids["discovery"], "estimation_ids": estimate_ids["estimation"],
               "discovery_candidates": int(estimate_masks["discovery"].sum()),
               "estimation_candidates": int(estimate_masks["estimation"].sum())}
    eligible = (rule is not None and matched_tasks >= settings["min_independent_tasks"]
                and support["confirmation"] >= settings["min_independent_tasks"])
    status = ("no_discovered_rule" if rule is None else
              "eligible" if eligible else "insufficient_independent_support")
    gate = {"accepted": False, "reason": status, "interval": None}
    discovery_estimate = None
    independent_estimate = None
    if eligible:
        mask = estimate_masks["discovery"]
        discovery_estimate = float((discovery["y"][mask] - discovery["p"][mask]).mean())
        mask = estimate_masks["estimation"]
        independent_estimate = float((estimation["y"][mask] - estimation["p"][mask]).mean())
        mask = masks["confirmation"]
        interval = residual_interval(confirmation["y"][mask] - confirmation["p"][mask],
                                     confirmation["ids"][mask], seed,
                                     settings["bootstrap_replicates"], settings["alpha"])
        agrees = interval["estimate"] * rule["discovery_residual"] > 0
        excludes_zero = interval["low"] > 0 or interval["high"] < 0
        accepted = bool(agrees and excludes_zero)
        reason = "confirmed" if accepted else "opposite_discovery_sign" if not agrees else "interval_includes_zero"
        gate = {"accepted": accepted, "reason": reason, "interval": interval}

    methods = {}

    def record(name: str, selected_rule: dict | None, estimate: float | None,
               allowed: bool, gated: bool, tune: bool, queried: dict[str, int]) -> None:
        permitted = allowed and (not gated or gate["accepted"])
        candidates = settings["step_grid"] if tune else [settings["fixed_step"]]
        selection = []
        for value in candidates:
            step = float(value)
            delta = float(step * estimate) if permitted and estimate is not None else 0.0
            scores = _apply(validation["x"], validation["p"], selected_rule, delta, bins)
            selection.append({"step": step, "delta": delta,
                              "brier": float(np.mean((scores - validation["y"]) ** 2))})
        chosen = min(selection, key=lambda row: (row["brier"], row["step"]))
        fit_queries = {role: int(queried.get(role, 0)) for role in roles}
        fit_queries["total"] = sum(fit_queries.values())
        actual_queries = {role: fit_queries[role] for role in roles}
        actual_queries["validation"] = available["validation"]
        actual_queries["total"] = sum(actual_queries.values())
        methods[name] = {
            "rule": selected_rule, "estimate": estimate, "step": chosen["step"],
            "delta": chosen["delta"], "eligible": bool(allowed), "gated": gated,
            "gate_accepted": gate["accepted"] if gated else None,
            "gate": gate if gated else None, "applied": bool(chosen["delta"] != 0),
            "validation_brier": chosen["brier"], "selection": selection if tune else [],
            "label_tasks": fit_queries, "diagnostic_label_tasks": {"validation": available["validation"]},
            "actual_label_tasks": actual_queries,
        }

    record("no_update", None, None, True, False, False, {})
    methods["no_update"]["step"] = 0.0
    for name, independent, gated in (
        ("discovery_fixed", False, False), ("independent_fixed", True, False),
        ("discovery_gated", False, True), ("independent_gated", True, True),
    ):
        queried = {"discovery": available["discovery"]}
        if eligible and independent:
            queried["estimation"] = matched_tasks
        if eligible and gated:
            queried["confirmation"] = support["confirmation"]
        record(name, rule, independent_estimate if independent else discovery_estimate,
               eligible, gated, False, queried)

    for name, independent in (("discovery_shrinkage", False), ("independent_shrinkage", True)):
        queried = {"discovery": available["discovery"], "validation": available["validation"]}
        if eligible and independent:
            queried["estimation"] = matched_tasks
        record(name, rule, independent_estimate if independent else discovery_estimate,
               eligible, False, True, queried)

    pool = {key: np.concatenate([parts[role][key] for role in roles[:3]])
            for key in ("x", "p", "y", "ids")}
    pooled_rule = find_rule(pool["x"], pool["y"], pool["p"], pool["ids"], settings, seed)
    pooled_estimate = pooled_rule["discovery_residual"] if pooled_rule is not None else None
    record("pooled_shrinkage", pooled_rule, pooled_estimate, pooled_rule is not None,
           False, True, {role: available[role] for role in roles})
    return {
        "bins": bins, "rule": rule, "status": status, "eligible": bool(eligible),
        "support": support, "matched_estimation": matched,
        "gate": gate, "available_label_tasks": available,
        "methods": methods, "settings": dict(settings), "seed": int(seed),
        "label_accounting_scope": "label_tasks counts conceptual per-policy fitting/selection queries; actual_label_tasks also includes validation diagnostics; whole-study union is the available pool",
        "membership": "One frozen discovery rule, evaluated against original scores; no iterative score-band updates",
    }

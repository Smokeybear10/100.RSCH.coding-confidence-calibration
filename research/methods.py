from __future__ import annotations

import numpy as np
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import FEATURES


def fit_classifier(x: np.ndarray, y: np.ndarray):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=0))
    model.fit(x, y)
    return model


def export_classifier(model, names: list[str]) -> dict:
    scaler, classifier = model.steps[0][1], model.steps[1][1]
    return {"features": names, "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
            "coefficients": classifier.coef_[0].tolist(), "intercept": float(classifier.intercept_[0])}


def predict_classifier(artifact: dict, x: np.ndarray) -> np.ndarray:
    scaled = (x - np.array(artifact["mean"])) / np.array(artifact["scale"])
    return expit(np.einsum("ij,j->i", scaled, np.array(artifact["coefficients"]), optimize=False) + artifact["intercept"])


def group_thresholds(x: np.ndarray) -> dict:
    return {name: float(np.quantile(x[:, FEATURES.index(feature)], .75)) for name, feature in
            (("long_prompt", "log_prompt_chars"), ("long_code", "log_code_chars"), ("nested_code", "ast_depth"))}


def membership(x: np.ndarray, thresholds: dict, names: list[str]) -> np.ndarray:
    values = {name: x[:, FEATURES.index(feature)] > thresholds[name] for name, feature in
              (("long_prompt", "log_prompt_chars"), ("long_code", "log_code_chars"), ("nested_code", "ast_depth"))}
    values.update({"all": np.ones(len(x), dtype=bool), "uses_imports": x[:, FEATURES.index("imports")] > 0,
                   "empty_code": x[:, FEATURES.index("empty_code")] == 1})
    values["long_prompt&nested_code"] = values["long_prompt"] & values["nested_code"]
    return np.column_stack([values[name] for name in names])


def bin_ids(scores: np.ndarray, bins: int) -> np.ndarray:
    return np.minimum((np.clip(scores, 0, 1) * bins).astype(int), bins - 1)


def audit(scores: np.ndarray, y: np.ndarray, groups: np.ndarray, ids: np.ndarray,
          names: list[str], bins: int, support: int) -> list[dict]:
    assignments = bin_ids(scores, bins)
    cells = []
    for j, name in enumerate(names):
        for b in range(bins):
            mask = groups[:, j] & (assignments == b)
            n = int(mask.sum())
            if n == 0:
                continue
            tasks = len(np.unique(ids[mask]))
            mean, rate = float(scores[mask].mean()), float(y[mask].mean())
            cells.append({"group": name, "bin": b, "n": n, "tasks": tasks, "mean": mean,
                          "rate": rate, "residual": rate - mean, "supported": tasks >= support})
    return cells


def fit_patches(scores: np.ndarray, y: np.ndarray, groups: np.ndarray, ids: np.ndarray,
                names: list[str], bins: int, support: int, epsilon: float, limit: int) -> dict:
    scores = scores.copy()
    patches = []
    status = "update_limit"
    for _ in range(limit):
        cells = [c for c in audit(scores, y, groups, ids, names, bins, support) if c["supported"]]
        if not cells:
            status = "no_supported_cells"
            break
        worst = max(cells, key=lambda c: abs(c["residual"]))
        if abs(worst["residual"]) <= epsilon:
            status = "converged_on_fit_cells"
            break
        mask = groups[:, names.index(worst["group"])] & (bin_ids(scores, bins) == worst["bin"])
        scores[mask] = np.clip(scores[mask] + worst["residual"], 0, 1)
        patches.append({**worst, "delta": worst["residual"], "after": float(scores[mask].mean())})
    return {"patches": patches, "status": status, "epsilon": epsilon, "support": support, "bins": bins}


def replay(scores: np.ndarray, groups: np.ndarray, names: list[str], fit: dict) -> np.ndarray:
    scores = scores.copy()
    for patch in fit["patches"]:
        mask = groups[:, names.index(patch["group"])] & (bin_ids(scores, fit["bins"]) == patch["bin"])
        scores[mask] = np.clip(scores[mask] + patch["delta"], 0, 1)
    return scores


def metrics(scores: np.ndarray, y: np.ndarray, groups: np.ndarray, ids: np.ndarray,
            names: list[str], bins: int, support: int) -> dict:
    cells = audit(scores, y, groups, ids, names, bins, support)
    global_cells = [c for c in cells if c["group"] == "all"]
    supported = [c for c in cells if c["supported"]]
    bounded = np.clip(scores, 1e-6, 1 - 1e-6)
    return {"brier": float(np.mean((scores - y) ** 2)),
            "log_loss": float(-np.mean(y * np.log(bounded) + (1 - y) * np.log(1 - bounded))),
            "ece": sum(c["n"] * abs(c["residual"]) for c in global_cells) / len(y),
            "worst": max((abs(c["residual"]) for c in supported), default=None),
            "supported_cells": len(supported), "occupied_cells": len(cells), "cells": cells}


def clustered_interval(values: np.ndarray, ids: np.ndarray, seed: int, repetitions: int) -> dict:
    # Resample entire tasks, not their ten correlated sibling completions.
    unique, inverse = np.unique(ids, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), (repetitions, len(unique)))
    samples = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    low, high = np.quantile(samples, [.025, .975])
    return {"estimate": float(values.mean()), "low": float(low), "high": float(high),
            "tasks": len(unique), "replicates": repetitions}


def budget_threshold(scores: np.ndarray, budget: float, keys: np.ndarray) -> dict:
    if budget >= 1:
        return {"score": 1.000001, "tie_key": 0.0}
    if budget <= 0:
        return {"score": 0.0, "tie_key": 0.0}
    index = np.lexsort((keys, scores))[int(len(scores) * budget)]
    return {"score": float(scores[index]), "tie_key": float(keys[index])}


def route_mask(scores: np.ndarray, threshold: dict, keys: np.ndarray) -> np.ndarray:
    return (scores < threshold["score"]) | ((scores == threshold["score"]) & (keys < threshold["tie_key"]))


def policy(scores: np.ndarray, labels: np.ndarray, alternate: np.ndarray, threshold: dict, keys: np.ndarray) -> dict:
    routed = route_mask(scores, threshold, keys)
    accepted = ~routed
    outcome = np.where(routed, alternate, labels)
    return {"threshold": threshold, "escalation": float(routed.mean()), "success": float(outcome.mean()),
            "accepted_error": float(1 - labels[accepted].mean()) if np.any(accepted) else None,
            "accepted_failures_per_request": float(np.mean(accepted & (labels == 0))),
            "rescued": float(np.mean(routed & (labels == 0) & (alternate == 1))),
            "regressed": float(np.mean(routed & (labels == 1) & (alternate == 0)))}


def fit_baselines(train: dict, cal: dict, partitions: dict, group_names: list[str]) -> tuple[dict, dict]:
    x, y = train["x"], train["y"]
    base = fit_classifier(x, y)
    group_base = fit_classifier(np.column_stack([x, train["groups"][:, 1:]]), y)
    token_columns = [FEATURES.index(name) for name in FEATURES[3:8] + ["missing_logprobs"]]
    structural_columns = [i for i in range(len(FEATURES)) if i not in token_columns]
    token = fit_classifier(x[:, token_columns], y)
    structural = fit_classifier(x[:, structural_columns], y)
    raw_cal = base.predict_proba(cal["x"])[:, 1]
    platt = LogisticRegression(C=1e6, max_iter=2000).fit(logit(np.clip(raw_cal, 1e-6, 1-1e-6)).reshape(-1, 1), cal["y"])
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(raw_cal, cal["y"])
    predictions = {}
    for name, data in partitions.items():
        raw = base.predict_proba(data["x"])[:, 1]
        np.testing.assert_allclose(raw, predict_classifier(export_classifier(base, FEATURES), data["x"]), rtol=1e-12, atol=1e-12)
        predictions[name] = {
            "logistic": raw,
            "platt": platt.predict_proba(logit(np.clip(raw, 1e-6, 1-1e-6)).reshape(-1, 1))[:, 1],
            "isotonic": isotonic.predict(raw),
            "group_logistic": group_base.predict_proba(np.column_stack([data["x"], data["groups"][:, 1:]]))[:, 1],
            "token_only": token.predict_proba(data["x"][:, token_columns])[:, 1],
            "structure_only": structural.predict_proba(data["x"][:, structural_columns])[:, 1],
            "likelihood": np.exp(data["x"][:, FEATURES.index("mean_logprob")]),
        }
    artifacts = {"logistic": export_classifier(base, FEATURES),
        "group_logistic": export_classifier(group_base, FEATURES + group_names[1:]),
        "token_only": export_classifier(token, [FEATURES[i] for i in token_columns]),
        "structure_only": export_classifier(structural, [FEATURES[i] for i in structural_columns]),
        "platt": {"coefficient": float(platt.coef_[0, 0]), "intercept": float(platt.intercept_[0])},
        "isotonic": {"x": isotonic.X_thresholds_.tolist(), "y": isotonic.y_thresholds_.tolist()}}
    return predictions, artifacts

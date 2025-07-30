# Learned Calibration and Decision Value

The main experiment tests whether a residual tree can find useful subgroup corrections without fitting noise. Proposed rules are checked on separate task blocks, then compared with unconfirmed updates. Simulations with known probabilities test the resulting false-correction and detection-power tradeoff. A follow-up evaluates whether confidence or predicted escalation benefit gives better routing decisions.

These experiments reuse the inspected CALIBRI archive. The original single-split reference is documented in [METHODS.md](METHODS.md).

## Evaluation design

Three fixed seeds each produce five outer folds. Every one of the 1,055 tasks is evaluated once per repeat. All ten answers to a task stay together. An outer fold has 211 test tasks; its other 844 tasks are assigned separate roles:

| Role | Tasks | Purpose |
| --- | ---: | --- |
| Model training | 379 | Fit the correctness and escalation-benefit predictors; learn feature-group thresholds |
| Score calibration | 126 | Fit the shared Platt mapping |
| Auditor discovery | 126 | Find candidate feature rules from residuals |
| Auditor confirmation | 105 | Estimate/check corrections on three disjoint blocks |
| Policy validation | 108 | Choose routing thresholds |
| Outer test | 211 | Evaluate frozen predictions and policies |

Role assignment uses seeded permutations of task IDs, with no outcome-dependent assignment. The [protocol](research/advanced_protocol.json), [complete split manifest](advanced_results/splits.json), source hashes and dependency versions are recorded. Internal row-level early stopping is disabled for boosted models, so related answers cannot leak through an automatic validation split.

The study emits 316,500 probability rows: 10,550 answers × three repeats × ten confidence pipelines. These are still **1,055 underlying task clusters**, not 316,500 independent examples.

## Correctness predictors and calibration controls

The same sixteen pre-outcome measurements described in [METHODS.md](METHODS.md) feed two predictors:

- Standardized L2 logistic regression: C=1, at most 2,000 iterations.
- Histogram gradient boosting: 120 rounds, learning rate 0.05, at most seven leaves per tree, minimum 40 rows per leaf, L2 leaf regularization 10, no internal early stopping.

Each predictor has five variants: raw probabilities, Platt scaling, fixed-group residual corrections after Platt, a learned tree auditor, and a confirmation-filtered tree auditor. The boosted model is scikit-learn histogram boosting, not XGBoost.

These are fixed, recorded configurations rather than a claim that each method has been optimally tuned. Fixed-group corrections permit 30 updates; learned auditors permit three. Their capacities, estimation and shrinkage differ, so the comparison does not isolate one causal component of calibration.

## How the auditor learns a subgroup

For each current confidence band, a depth-two regression tree predicts the residual y minus p from the observable features. A leaf becomes a conjunction of feature tests; a root leaf can represent the score band alone. A rule stores the feature names, columns, thresholds, branch directions and current score band.

Candidates need at least 20 distinct discovery tasks, a mean residual exceeding 0.03 in magnitude, and at least 40 rows per tree leaf. The candidate priority is its sample fraction times squared residual, corresponding to the unconstrained squared-error improvement of a full constant update. This is a greedy heuristic search, not an exact optimization over every possible tree.

Rule replay uses float32 feature comparisons to match scikit-learn's tree traversal. Score-band assignment uses the current probability. Previous offsets can move an answer into or out of later rules.

### Reused-data auditor

The naïve auditor discovers and estimates corrections on the pooled discovery and confirmation tasks. Each accepted correction is shrunk by n/(n+20), where n counts represented tasks. It can repeatedly exploit noise in the same outcomes.

### Confirmation-filtered auditor

The confirmation tasks are randomly divided into three disjoint blocks before fitting. At round t:

1. Discover the next rule using discovery residuals only.
2. Apply previous frozen patches to the next unused confirmation block.
3. Require at least eight represented tasks in the proposed region.
4. Compute an approximate task-cluster percentile-bootstrap interval for its residual, using nominal alpha/3 per round.
5. Accept only if the interval excludes zero and its sign agrees with discovery. Shrink the independently estimated residual by n/(n+20).

Rejected rules consume their block. Future confirmation labels cannot change earlier decisions. The two auditor variants have the same available task pool, but the filtered variant uses only queried regions for estimation and typically consumes less label information. It trades fitting power for protection against selected noise.

Bootstrap intervals with eight or more represented tasks are approximate. The alpha allocation does not turn this into exact familywise error control, a population multicalibration theorem, or a guarantee under distribution shift. Accepted paths, confirmation blocks, support counts, residual intervals and rejection reasons are all saved in [auditor_rules.json](advanced_results/auditor_rules.json).

## Sample-budget experiment

The logistic predictor, Platt mapping and feature thresholds remain fixed while nested task prefixes vary the auditor data. Discovery/confirmation budgets are 30/25, 90/75 and 126/105 tasks. Thus the experiment varies 55, 165 and 231 available **auditor** tasks; it does not vary the base-training or Platt-calibration sample.

Each configuration is evaluated on the same outer test fold. The [learning-curve table](advanced_results/sample_complexity.csv) includes fitting-pool and held-out Brier plus accepted patch counts. This is descriptive sensitivity analysis, not test-based selection of a deployed sample size.

## Escalation benefit and decision costs

Correctness confidence estimates P(primary passes | X). Routing can need a different target:

$$u(X)=E[y_{alternate}-y_{primary}\mid X].$$

A regularized histogram regressor fits this signed outcome difference using **training tasks only**. Its inputs are primary-answer features, not alternate answers or hidden evaluation labels at decision time. Scores are clipped to [-1,1]. The target is an observed difference between recorded alternatives, not a causal treatment-effect claim.

For validation-budget comparisons, methods escalate at 10%, 30% and 50% validation budgets. Test rates may differ. Ties use the original outcome-independent keys.

For cost scenarios, primary generation costs one stipulated unit, alternate generation costs four, and failures cost L in {2,5,10,20,50}. A cascade is charged for both stages when escalating:

$$J=1+4P(escalate)+L\,P(final\ failure).$$

The plug-in benefit policy escalates when L times predicted benefit exceeds four. Confidence policies choose cutoffs using validation outcomes. Protocol v2 retains the initial eleven-budget grid and adds an exact search over all validation score/tie boundaries. A validation-selected benefit cutoff is included as well. The exact search computes every prefix objective from cumulative observed paired benefits; test outcomes never choose a boundary.

Primary-only costs one; a direct alternate-only baseline costs four, without a sunk primary stage. These cost units do not measure bills, latency, energy or confidence-estimation overhead. A different cost model can change the conclusion.

[Per-task policy objectives](advanced_results/task_costs.csv.gz), cutoffs, actual rescue/regression rates, [decision costs](advanced_results/decision_costs.csv) and [cost contrasts](advanced_results/cost_comparisons.csv) are included. The exact-threshold logistic/Platt policy is the contrast baseline. At L=20 the plug-in benefit policy's mean objective difference is approximately -0.365, with descriptive interval [-0.474,-0.258]. At L=50 the difference is approximately +0.045 and the interval spans zero. The benefit policy does not dominate every scenario.

## Covariate-mixture stress tests

Task-average AST depth and prompt length define high/low groups using training-task medians. Held-out task weights vary the high-group share from zero to one. The code reports weighted Brier/log loss and effective task count.

This retrospectively changes the feature mixture while preserving recorded outcomes. It does not simulate new conditional error mechanisms, establish temporal transfer, or evaluate a new benchmark. [Mixture results](advanced_results/mixture_shift.csv) remain separate from ordinary cross-validation metrics.

## Known-probability controls

Synthetic tasks have sixteen abstract numeric features and ten correlated candidate outcomes. One uniform draw per task is shared across candidates, preserving each known marginal success probability while inducing dependence.

- Under the calibrated null, the initial p is the generator's true probability.
- Under hidden interaction bias, truth is reduced by 0.25 when two feature coordinates are positive, then clipped to [0,1].

Ten seeds and three available calibration sizes (120, 360, 1,080 tasks) compare no correction, reused-data tree corrections and confirmed corrections. The evaluation target is MSE against the **known probability** on independent simulated tasks, not only a noisy binary-outcome Brier score. Shading in the simulation figure is seed standard deviation, not a confidence interval.

At 120 null calibration tasks, the reused-data auditor patches in all ten seeds and has mean probability MSE about 0.0130; the filtered auditor applies none. The filter still accepts a false correction in one of ten null runs at size 360. At size 1,080 with a real interaction, the reused-data method has lower MSE than the filter (about 0.00512 versus 0.01121). Independent confirmation costs power; it is not uniformly better.

These [simulations](simulation_results/simulation_summary.csv) explain a possible mechanism. They are not additional coding tasks or proof of behavior on real traffic.

## Uncertainty and interpretation

For each contrast, losses are averaged across a task's ten candidates and three repeats. The bootstrap resamples the 1,055 resulting task means. **The intervals are descriptive and conditional on overlapping fitted models. They omit refitting/split uncertainty and dependence induced by shared training sets.** Repeats are not independent replications. There is no simultaneous-coverage claim across methods, costs, or mixture scenarios.

The raw logistic model has the lowest mean Brier in this configuration. Learned tree corrections do not automatically improve it; confirmation usually declines to patch. Predicting escalation benefit changes some decision objectives even when probability post-processing adds little. These are findings on an inspected historical sample, not prospective validation or a new theorem.

## Reproduction

```bash
python -m research.advanced --output work/new-advanced
python -m research.simulations --output work/new-simulations
python -m research.advanced_figures --results work/new-advanced --simulations work/new-simulations
jupyter lab advanced_analysis.ipynb
```

The notebook reads the included full-study results. A `--smoke` run evaluates one outer fold or one simulation seed and is explicitly marked incomplete. New output directories are required. Numerical manifests, source hashes and literal protocol snapshots accompany both full runs.

Protocol v1 is retained under [research/protocols](research/protocols/advanced-v1.json). The exact-threshold cost comparator was added after the first exploratory run; original grid comparisons remain reported. No base-model or auditor parameter was retuned in response to those results. The original single-split study remains under `results/` and `research/runs/a8630092fa9e/`.

The methodology builds on [multicalibration](https://proceedings.mlr.press/v80/hebert-johnson18a.html), [learned auditing implementations](https://joss.theoj.org/papers/10.21105/joss.03453), and [LLM confidence multicalibration](https://proceedings.mlr.press/v235/detommaso24a.html). The question of when post-processing is useful is also studied in [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/43c6d9b0d11696f89c75157919a0334a-Paper-Conference.pdf). This repository applies existing ideas to recorded coding outcomes with a specific evaluation and decision model.

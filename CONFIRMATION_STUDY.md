# Confirmation Versus Conservative Calibration

Does fresh confirmation help beyond conservative updates? This study separates the gate from correction estimation and compares both with validation-selected shrinkage under the same available labeling budget.

This one-correction experiment gives no iterative multicalibration guarantee. The [protocol](research/confirmation_protocol.json) is frozen before external outcome evaluation, without independent registration.

## Data roles and budgets

Five task-separated outer folds reuse the inspected LiveCodeBench archive: 379 predictor-training, 126 Platt-calibration, 211 test and 339 available audit/tuning tasks per fold. All ten answers to a task stay together. Source results remain exploratory.

Discovery D, independent estimation E, confirmation C and tuning V use nested prefixes of fixed role assignments. Counts floor D/E/C fractions; V gets the remainder:

| Available tasks | D: 40% | E: 20% | C: 20% | V: remainder |
| --- | ---: | ---: | ---: | ---: |
| 120 | 48 | 24 | 24 | 24 |
| 240 | 96 | 48 | 48 | 48 |
| 339 | 135 | 67 | 67 | 70 |

The predictor and Platt mapping are fixed before these roles are used. The budget counts distinct tasks, not candidate rows. Methods receive the same available budget; some deliberately leave labels unused. Per-policy query counts describe fitting and selection requirements, with validation diagnostics recorded separately. The complete comparison computes all arms and uses the union of their labels. Equal availability is not equal information consumption.

## A common proposal and four matched arms

D selects one residual-tree rule inside five score bands. The tree has depth at most two, at least 40 rows per leaf, at least 20 represented discovery tasks and an absolute residual above 0.03. Supported rules are ranked by sample fraction times squared mean residual.

Let k be the smaller number of rule-represented tasks in D and E. Seeded membership-only permutations select k tasks from each role. Residual estimates rD and rE average qualifying candidate rows from those selected tasks; estimation IDs are saved. Shared eligibility requires k ≥ 8 and at least eight represented C tasks. All four primary arms and both D-rule shrinkage controls share this check, which never reads E/C labels.

The C gate passes when a 95% task-cluster percentile-bootstrap interval excludes zero and agrees with the full-D discovery residual. It uses 1,000 bootstrap draws. E labels cannot affect it. This approximate check gives no exact familywise error control. Matched rD/rE may have another sign; confirming the proposal direction does not validate the applied correction.

| Method | Estimate | Gate | Step |
| --- | --- | --- | ---: |
| `discovery_fixed` | rD | None | 0.5 |
| `independent_fixed` | rE | None | 0.5 |
| `discovery_gated` | rD | C | 0.5 |
| `independent_gated` | rE | C | 0.5 |

An accepted update adds step times residual within the rule, clipping to [0,1]. Ineligible arms retain scores. Within each estimator, gated versus ungated loss isolates the gate for the fixed proposal. Matching represented-task counts does not match qualifying candidate counts or variance; D estimation also retains rule-selection dependence.

## Conservative controls

`no_update` retains Platt probabilities; `raw_logistic` is an additional uncalibrated reference. `discovery_shrinkage` and `independent_shrinkage` tune the D-rule update using rD or rE. `pooled_shrinkage` learns its own one-step rule and residual on D+E+C.

All three select a step from {0, 0.125, 0.25, 0.5, 1} on the same V tasks. Zero permits no correction. The pooled control tests practical use of the available labels; its different discovery information prevents attributing its contrast solely to confirmation. No test outcomes choose steps or rules.

## Known-probability experiments

Each setting has 100 independent fitting seeds, ten candidates per task and 5,000 independent test tasks. Calibration budgets are 120, 360 and 1,080 tasks. Methods share draws within seeds.

Four scenarios use a calibrated null, a broad weak interaction, a broad strong interaction and a rare strong interaction. Broad regions have marginal prevalence 25%; the rare region has prevalence 5%. Interaction sizes subtract 0.10 or 0.25 from the initial probability, then clip to [0,1]. Candidatewise independent Uniform draws and a task-shared Uniform draw provide two outcome-dependence settings.

Primary loss is MSE against the generator's known probability. Report paired seed-level contrasts, eligibility, gate acceptance, subgroup residuals and update frequency. On the calibrated null, any score change is a false update; its Monte Carlo rate receives a Clopper–Pearson binomial interval across seeds. Under alternatives, harmful updates increase known-probability MSE and are reported separately. These intervals describe the generator, not universal error control. Keep missing support separate from a rejected eligible proposal.

## External evaluation

MultiPL-E Python provides 161 external tasks. At source budget 339, predictions from all five frozen source pipelines are averaged per answer and evaluated once. Upstream restrictions make these records evaluation-only: no fitting, recalibration, step selection or protocol tuning uses external labels. Source predictors and corrections are serialized before those labels are loaded. This tests frozen transfer. Benchmark separation does not establish absence of model pretraining contamination.

Source contrasts condition on overlapping fits; synthetic intervals use independent fitting seeds. External candidates stay task-grouped. Repeated source pipelines do not create independent external tasks.

## Reproduce

```bash
python -m research.external --output work/multipl-e-python --protocol research/confirmation_protocol.json
python -m research.confirmation_study --output work/new-confirmation --external work/multipl-e-python
python -m research.confirmation_figures --results work/new-confirmation
```

Use fresh paths. Omitting `--external` runs source and synthetic experiments; `--smoke` is incomplete. [confirmation_analysis.ipynb](confirmation_analysis.ipynb) reads published results.

## Related work

1. [Hebert-Johnson et al. (2018)](https://proceedings.mlr.press/v80/hebert-johnson18a.html) connect subgroup calibration with weak agnostic learning; residual trees are approximate auditors.
2. [MCBoost](https://joss.theoj.org/papers/10.21105/joss.03453) already provides [fresh splits, cross-validated auditors, smaller steps and fewer iterations](https://mlr-org.github.io/mcboost/articles/mcboost_basics_extensions.html).
3. [Detommaso et al. (2024)](https://proceedings.mlr.press/v235/detommaso24a.html) use validation-MSE stopping and minimum group mass against overfitting.
4. [Hansen et al. (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/43c6d9b0d11696f89c75157919a0334a-Abstract-Conference.html) find post-processing benefits depend on the initial predictor.
5. [Dwork et al. (2015)](https://proceedings.neurips.cc/paper_files/paper/2015/file/bad5f33780c42f2588878a9d07405083-Paper.pdf) address adaptive holdout reuse; this experiment tests one proposal.
6. [Shabat, Cohen and Mansour (2020)](https://proceedings.neurips.cc/paper/2020/file/9a96876e2f8f3dc4f3cf45f02c61c0c1-Paper.pdf) relate calibration-error generalization to subgroup and score-category mass.

The intended contribution is the controlled empirical comparison and its transfer evidence. Novelty is unestablished; the study claims no new multicalibration theorem.

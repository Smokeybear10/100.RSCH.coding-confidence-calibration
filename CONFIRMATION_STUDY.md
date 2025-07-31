# Confirmation Versus Conservative Calibration

Does checking a proposed correction on fresh labels help more than reducing its size? This study separates the confirmation gate from the residual estimate used for correction, then compares them with validation-selected shrinkage. Each method has the same pool of available labeled tasks.

The experiment applies at most one correction and gives no iterative multicalibration guarantee. The [protocol](research/confirmation_protocol.json) was frozen before external outcome evaluation but was not independently registered.

## Data roles and budgets

Five outer folds reuse the previously inspected LiveCodeBench archive. Each fold assigns 379 tasks to predictor training, 126 to Platt calibration, 211 to testing and 339 to auditing and tuning. These sets are disjoint within a fold, and all ten answers to a task stay together. Results on this source archive remain exploratory.

The audit/tuning pool is split into four disjoint roles:

- D (discovery) selects a rule and supplies the reused residual estimate.
- E (independent estimation) supplies a fresh residual estimate for that rule.
- C (confirmation) checks whether the rule's residual agrees with discovery.
- V (validation) selects the update size for the shrinkage controls.

Role assignments are fixed at the largest budget. Smaller budgets take prefixes of each role's task list, so tasks keep the same role as the budget grows. D, E and C counts are rounded down; V gets the remainder:

| Available tasks | D: 40% | E: 20% | C: 20% | V: remainder |
| --- | ---: | ---: | ---: | ---: |
| 120 | 48 | 24 | 24 | 24 |
| 240 | 96 | 48 | 48 | 48 |
| 339 | 135 | 67 | 67 | 70 |

The predictor and Platt mapping are fixed before D, E, C or V is used. Budgets count distinct tasks, not candidate rows. Some methods leave available labels unused. Per-method query counts record the labels needed for fitting and selection; validation diagnostics are recorded separately. Running the complete comparison uses the union of all methods' labels. The methods therefore have equal available budgets but do not consume equal amounts of label information.

## A common proposal and four matched arms

D searches five score bands and selects one residual-tree rule. Trees have depth at most two and at least 40 candidate rows per leaf. An eligible rule must include candidates from at least 20 discovery tasks and have an absolute mean residual above 0.03. Rules meeting these requirements are ranked by sample fraction times squared mean residual, where a residual is the outcome minus the predicted probability.

Let k be the smaller of the numbers of D and E tasks with at least one candidate inside the selected rule. Seeded permutations choose k such tasks from each role without reading their outcomes. Estimates rD and rE average the residuals of candidate rows inside the rule from those selected tasks; the task IDs are saved. An update is eligible only if k ≥ 8 and at least eight C tasks also have candidates inside the rule. All four primary arms and both shrinkage controls using the D-selected rule share this check. Eligibility uses rule membership without reading E or C labels.

The C gate uses a 95% task-cluster percentile-bootstrap interval for the rule's mean residual, with 1,000 draws. It passes when the interval excludes zero and the C residual has the same sign as the discovery residual computed from all D candidates inside the rule. E labels do not affect this decision. The check is approximate and gives no exact familywise error control. Either matched estimate, rD or rE, can have a different sign from the full-D residual, so passing the gate does not establish that the applied correction has the confirmed direction.

| Method | Estimate | Gate | Step |
| --- | --- | --- | ---: |
| `discovery_fixed` | rD | None | 0.5 |
| `independent_fixed` | rE | None | 0.5 |
| `discovery_gated` | rD | C | 0.5 |
| `independent_gated` | rE | C | 0.5 |

An applied update adds the step times its residual estimate to probabilities inside the rule, then clips them to [0,1]. Ineligible arms retain the original scores. For a given estimator, comparing gated and ungated loss isolates the gate's effect on the fixed proposal. Matching task counts does not equalize the numbers of qualifying candidates or the estimates' variances. The D estimate also reuses labels that selected the rule.

## Conservative controls

`no_update` retains Platt probabilities; `raw_logistic` supplies an additional uncalibrated reference. `discovery_shrinkage` and `independent_shrinkage` use the same D-selected rule and estimates rD and rE, respectively, but choose the step on V. `pooled_shrinkage` instead learns its own rule and residual from D+E+C and also applies at most one update.

All three shrinkage controls minimize Brier loss on the same V tasks over steps {0, 0.125, 0.25, 0.5, 1}; exact ties favor the smaller step. A step of zero leaves probabilities unchanged. The pooled control tests how well the available labels can be used together. Because it also has more labels for rule discovery, its comparison with the gated method cannot isolate confirmation's effect. Test outcomes never choose steps or rules.

## Known-probability experiments

Each setting uses 100 independent fitting seeds, ten candidates per task and 5,000 independent test tasks. Calibration budgets are 120, 360 and 1,080 tasks. Within a seed, all methods use the same draws.

The four scenarios are a calibrated null, a broad weak interaction, a broad strong interaction and a rare strong interaction. Broad regions have marginal prevalence 25%; the rare region has prevalence 5%. Within an interaction region, the generator subtracts 0.10 or 0.25 from the initial probability and clips the result to [0,1]. Outcomes are generated either with independent Uniform draws for each candidate or with one Uniform draw shared by all candidates in a task.

The primary loss is MSE against the generator's known probability. Reported measures include paired seed-level contrasts, eligibility, gate acceptance, subgroup residuals and update frequency. On the calibrated null, any score change is a false update; its Monte Carlo rate receives a Clopper–Pearson binomial interval across seeds. Under the interaction scenarios, harmful updates increase known-probability MSE and are reported separately. These intervals describe behavior under this generator and provide no universal error-control guarantee. Insufficient task support is reported separately from rejection of an eligible proposal.

## External evaluation

MultiPL-E Python provides 161 external tasks. At source budget 339, predictions from all five frozen source pipelines are averaged for each answer and evaluated once. Upstream restrictions limit these records to evaluation: external labels are never used for fitting, recalibration, step selection or protocol tuning. Source predictors and corrections are serialized before those labels are loaded. This tests transfer of the frozen pipelines. Separate benchmarks do not rule out model pretraining contamination.

Source-fold contrasts are descriptive because the fitted pipelines use overlapping tasks. Synthetic intervals use independent fitting seeds. External candidates stay grouped by task; evaluating several source pipelines does not create additional independent external tasks.

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

The study compares these correction policies under controlled conditions and evaluates their transfer to an external benchmark. Novelty has not been established, and the study claims no new multicalibration theorem.

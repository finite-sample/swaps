# Original validation-swaps implementation: correctness and demonstrated value

Assessed commit `ab1c17f230e2482e1f62887c45d708c26c190097`, dated 2026-03-30, titled “Partial Credit: Diagnosing Proxy Covariates with Validation Swaps.” This is the original covariate paper, not the later outcome-only rewrite. Extracted source files are unchanged snapshots in [original_snapshot](original_snapshot). No current manuscript or analysis code was changed.

## What the original actually proposes and implements

The original proposes a hybrid-covariate coefficient path, a Shapley row attribution (SIM), an observable prediction of absolute SIM for domain selection, a validation audit, and local regression calibration inside accepted domains. The diagnostic focus is downstream coefficient sensitivity, not predictive-model accuracy. See [original manuscript](original_snapshot/validation_swaps.tex:38).

The implementation genuinely trains a portable score: `predict_sim` regresses estimated absolute SIM on an intercept, absolute proxy value, both W variables, their squares, and an interaction, then truncates negative predictions to zero. It does **not** simply substitute a hardcoded proxy-risk feature for the proposed learner. See [original code](original_snapshot/replicate_final.py:230).

The path uses independent Bernoulli swaps at each p, refits OLS exactly for each draw, and reports Monte Carlo means and standard deviations. Only validated rows are swapped; the p=1 endpoint still uses proxies outside validation. This endpoint distinction is correctly acknowledged in the manuscript. See [code](original_snapshot/replicate_final.py:155) and [manuscript](original_snapshot/validation_swaps.tex:94).

The main correctness/value findings below do not by themselves prove that the full workflow lacks novelty. They establish that the present implementation and examples do not yet validate its stated contribution.

## Decisive findings

### 1. Implemented SIM is a Banzhaf value, not the stated Shapley value

The manuscript defines the usual Shapley coalition weights and claims that SIM sums to the hybrid endpoint gap. The implementation instead samples each validated row into a coalition independently with probability 1/2, toggles a sample of rows, orients each difference as adding the gold value, and averages. Conditional on a focal row, every coalition of other rows has equal probability. Its limiting estimand is therefore

\[
B_i=2^{-(m-1)}\sum_{S\subseteq V\setminus\{i\}}[v(S\cup\{i\})-v(S)],
\]

not

\[
\operatorname{Shapley}_i=\sum_S\frac{|S|!(m-|S|-1)!}{m!}[v(S\cup\{i\})-v(S)].
\]

These agree for some simple games, but OLS covariate replacements are generally nonadditive. More Monte Carlo draws do not fix the different weighting. See [SIM definition](original_snapshot/validation_swaps.tex:124) and [implementation](original_snapshot/replicate_final.py:193).

**Exact check:** use the original ugly DGP with n=30, five validation rows, seed 2026. Enumerate all 32 coalitions and evaluate the original OLS function. Results:

| Quantity | Value |
|---|---:|
| Hybrid endpoint gap | 0.0874319914 |
| Sum of exact Shapley values | 0.0874319914 |
| Sum of exact uniform-coalition/Banzhaf values | 0.0804696722 |
| Efficiency discrepancy | −0.0069623192, or −7.96% of the gap |
| Maximum error of original code at 20,000 draws relative to Banzhaf | 0.000350618 |
| Maximum error relative to Shapley | 0.002611108 |

Seed 2027 is used for the 20,000-draw check. All rowwise values are in [verification_results.json](verification_results.json). The exact enumeration, not the noisy convergence comparison, proves the mismatch. The original main configuration averages only about 6.4 marginal evaluations per discovery row (80 draws, 20 focal rows per draw, 250 discovery rows), so attribution noise is another issue beyond the wrong estimand.

**Contribution implication:** a Shapley attribution of nonlinear finite replacements remains a coherent mathematical object, but the original simulations do not implement that attribution. They cannot establish its claimed additive decomposition or the performance of a learner trained on it.

### 2. The “ugly” population coefficient is 2, and the calibration failure is lack of relevance

The stated DGP is

\[
X=W_2+U,\qquad Y=X+0.5W_2+0.3W_3+U+\epsilon.
\]

Substitution gives

\[
Y=2X-0.5W_2+0.3W_3+\epsilon.
\]

Therefore the population coefficient from regressing Y on X and W is **2**, not the claimed 1. If the nominal structural coefficient is kept at 1, the corresponding residual contains U and violates the stipulated conditional mean assumption. See [DGP and claim](original_snapshot/validation_swaps.tex:293) and [code](original_snapshot/replicate_final.py:101).

The code's main table compares methods with the estimated full-gold coefficient, so this is not a claim that all table bias calculations subtract the wrong target. The published ugly naive bias −1.999 and −100% actually reveal an oracle near 2. The manuscript's global statement that every target equals 1 and the horizontal figure reference at 1 are incorrect.

Also, with proxy \(\hat X=W_2+0.5\nu\),

\[
E[X\mid\hat X,W]=W_2.
\]

That conditional mean is **inside the fitted linear calibration family**. It is not misspecified because the proxy is uninformative. The failure is that its coefficient cannot be separated from W: the relevance/rank assumption fails exactly in population. With the correctly parameterized coefficient 2, the exogeneity restriction is satisfied, while calibration relevance fails. The manuscript conflates misspecification, lack of information, and conditional exogeneity. See [proposition and relevance assumption](original_snapshot/validation_swaps.tex:240) and [simulation interpretation](original_snapshot/validation_swaps.tex:301).

**Check:** 200,000 original ugly observations with seed 2028 yield gold coefficient 1.999004. The estimated calibration coefficients in order (intercept, proxy, W2, W3) are approximately (−0.00138, −0.00687, 1.00582, −0.00016).

The original --fast main table with a fixed hash seed produced ugly global-correction bias **+16.0475 (+803.7%)**, rather than the published −79.8%. This is consistent with an unstable near-zero calibration slope; it does not establish a different stable population bias. The named −80% failure should not be interpreted as a reliable property of a well-identified misspecified correction method.

**Contribution implication:** the strongest claimed advantage—refusal where “global correction silently fails”—is illustrated using a relevance failure detectable by ordinary calibration diagnostics. A comparison against a correction method with a relevance check is needed before crediting SIM with a unique decision advantage.

### 3. Heterogeneous calibration is not the true conditional mean, but its coefficient correction can still be consistent

For the heterogeneous DGP, normal conditioning gives

\[
E[X\mid\hat X,W]=\frac{\hat X}{1+(0.1+1.5|W_2|)^2}.
\]

This is not affine in the proxy and W. The original claim that its linear conditional-mean calibration is correctly specified is false. However, this does **not** imply that the coefficient correction must be biased.

The population linear projection \(\mu_L\) of X on (1, proxy, W) has a residual orthogonal to that design. Under the original linear outcome DGP and exogeneity, regressing Y on (1, \(\mu_L\), W) still recovers the target coefficient when the projected covariate is relevant. This projection argument is enough here even without the true conditional expectation. See the fitted calibration [code](original_snapshot/replicate_final.py:403).

**Check:** 200,000 heterogeneous observations, seed 2029, yield RMS discrepancy 0.452216 between the linear calibration and the actual conditional mean. The corrected coefficient is nevertheless 0.995245 with the linear projection and 0.996212 with the true conditional mean.

**Contribution implication:** the examples do not establish that SIM diagnoses conditional-mean misspecification. In fact, the heterogeneous example shows conditional-mean misspecification without material population coefficient error. The relevant diagnostic must target the deployed estimator, not the truth of a stronger unnecessary calibration model.

### 4. The audit checks raw proxy substitution, then deploys a different calibrated estimator

The projected branch audits the difference between regressions on X and raw proxy inside the selected audit rows. On passing, it fits local calibration and reports the regression on the predicted covariate. No audit of calibrated-versus-gold coefficients is performed. See [audit and subsequent deployment](original_snapshot/replicate_final.py:471).

Accordingly the audit is a check of raw substitution, not a test that the calibration model works. A steep raw path is compatible with perfectly valid calibration: even the elementary proxy \(\hat X=2X\) gives a coefficient half as large before calibration, while rescaling exactly restores X. Conversely a raw-proxy tolerance cannot simply be transferred to a newly fitted calibration. The manuscript is appropriately explicit that its point audit is not a formal guarantee; that caveat does not repair the mismatch between the audited and reported estimators.

The discovery SIM targets are calculated by refitting on **all outcomes**, including audit-fold outcomes. Thus the fold is held out from using its gold X values during discovery, but it is not an independent outcome holdout. The validation-only branch goes further and chooses audit members using SIM recomputed with their own gold values and outcomes. See [discovery call](original_snapshot/replicate_final.py:414) and [audit selection](original_snapshot/replicate_final.py:434). This needs careful interpretation before formal post-selection validity claims.

The manuscript's “flat path settles the question” language also overreaches the observable object: only the validation rows are replaced, and the mean averages configurations. It cannot establish adequacy of unseen corrections or of a subsequently calibrated estimator. This is not a claim that every flat path is misleading; it is a limit on what the diagnostic establishes.

### 5. Standard errors omit calibration uncertainty even without domain selection

The second-stage `ols` HC1 calculation treats predicted covariates as fixed. The paper acknowledges domain-selection uncertainty, but that is distinct from estimation of calibration parameters from only 500 rows. See [HC1 implementation](original_snapshot/replicate_final.py:120) and [global calibration](original_snapshot/replicate_final.py:403).

**Isolated check:** 300 independent heterogeneous DGP replications, n=2000, validation size 500, seed 2030. Use exactly the original global calibration followed by original OLS, with the full domain fixed and no SIM selection or audit.

| Metric | Result |
|---|---:|
| Mean coefficient | 1.007646 |
| Empirical coefficient standard deviation | 0.107728 |
| Mean reported HC1 SE | 0.065894 |
| Coverage of estimate ±1.96 reported SE for target 1 | 74.33% |

The Monte Carlo standard error of that coverage estimate is about 2.5 percentage points. This is a targeted counterexample to treating the reported SEs as uncertainty for the complete calibration pipeline, not a full coverage study of every domain policy. The precision/bias-budget comparisons require correct uncertainty for the actual estimator; ignoring only selection does not explain this fixed-domain result.

### 6. The validation-only correction branch never calibrates because of a set-membership bug

At [line 448](original_snapshot/replicate_final.py:448), `np.isin(val_idx, set(all_dom))` supplies a Python set to NumPy. NumPy treats that set as a single object rather than the collection of membership values, producing an all-false mask here. `local_correct` therefore sees fewer than ten calibration rows and returns the raw proxy unchanged.

The independent check gives `[False, False, False]` for `np.isin([1,2,3], {1,3})`, whereas passing `[1,3]` produces `[True, False, True]`. This behavior is documented in the official [NumPy isin notes and example](https://numpy.org/doc/stable/reference/generated/numpy.isin.html).

This affects the validation-only domain branch, not the projected branch's correctly constructed boolean mask. The fast benign table reports −7.9% bias for the branch labeled “SIM (val only)+corr,” consistent with uncorrected proxy attenuation. Do not attribute that branch's results to calibrated SIM.

### 7. Fallback policy value is discussed but not directly evaluated

`run_one` returns baseline coefficients, pass flags, and conditional domain metrics. It does not return a realized pass/fallback policy coefficient. Tables report domain bias only among passing replications. The manuscript's approximate unconditional bias calculation mixes pass-conditioned domain bias with the unconditional validation-only bias, rather than the validation-only bias conditional on rejection. The distinction matters when audit events select estimation errors. It also averages reported SEs rather than estimating the policy's sampling variance. See [returned quantities](original_snapshot/replicate_final.py:419), [conditional summaries](original_snapshot/replicate_final.py:539), and [policy discussion](original_snapshot/validation_swaps.tex:409).

No strong value claim about an implementable fallback policy is established by that mixture arithmetic. The modest descriptive claim that a low-pass-rate pipeline usually falls back is supported.

## Reproduction, limits, and research decision

The audit script and machine-readable results are [verify_original.py](verify_original.py) and [verification_results.json](verification_results.json). Dependencies were initially missing; I created an isolated standard uv environment in `/private/tmp/swaps-original-audit-venv` and installed NumPy, SciPy, and Matplotlib successfully. This avoided altering the repository environment.

Commands used:

```sh
uv venv /private/tmp/swaps-original-audit-venv
uv pip install --python /private/tmp/swaps-original-audit-venv/bin/python numpy scipy matplotlib
MPLCONFIGDIR=/private/tmp/swaps-original-mpl /private/tmp/swaps-original-audit-venv/bin/python 'research_notes/Original validation swaps novelty/verify_original.py'
MPLCONFIGDIR=/private/tmp/swaps-original-mpl PYTHONHASHSEED=0 /private/tmp/swaps-original-audit-venv/bin/python 'research_notes/Original validation swaps novelty/original_snapshot/replicate_final.py' --fast --outdir 'research_notes/Original validation swaps novelty/original_snapshot/fast_results'
```

The independent verification script completed. Black, isort, Flake8, and Ruff checks passed for the new script after formatting. The extracted original source was not reformatted or repaired.

**Full-replication status:** the original --fast command was launched and completed all three scenario main tables (50 replications each) and figures 1–3. It was then deliberately interrupted during the remaining sensitivity runs after the requested decisive checks were complete. Its output and log are preserved under `original_snapshot/fast_results` and `original_snapshot/fast_run.log`. This is **not** a completed full or --fast replication. The code uses `hash(scenario)` in seeds, so `PYTHONHASHSEED=0` was specified; published exact numbers cannot be expected from an unspecified hash seed. The original CLI also has no `--n_sims` argument despite the README's example.

The strongest original claim worth assessing is the combined **finite covariate substitution → row attribution → observable domain → audit → borrowing decision** workflow. Its components must still be compared with prior Shapley attribution, regression diagnostics, and two-phase measurement-error methods. These correctness findings do not decide novelty by themselves. They do mean the present implementation cannot support claims about Shapley decomposition, reliable calibration diagnostics, or valid precision gains.

**Decision: narrow and test, after correctness repairs.** A useful diagnostic note remains possible. Before a methods contribution is claimed, implement the stated attribution (or deliberately define the Banzhaf alternative), audit the actual deployed estimator, handle calibration uncertainty/relevance, and compare the policy with simple calibration diagnostics and validation-only estimation on independent paired data. Count domain changes as estimand changes, not merely precision improvements. Neither a coding fix nor a renamed standard attribution is a new methodological result.

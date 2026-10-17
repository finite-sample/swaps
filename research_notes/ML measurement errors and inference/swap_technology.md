# Finite measurement replacement, influence approximations, and novelty

Research date: 2026-10-17. Scope: the precise mathematical technology behind replacing ML measurements in downstream econometric analyses. This note does not evaluate prediction-model training influence. The accompanying calculation is a deterministic stress example and algebra check, not evidence of practical superiority.

## What technology is already established, and what does this repository implement?

### Takeaway

The repository implements exact finite-population sampling of **additive outcome corrections with fixed estimator weights**. Correcting ML covariates changes those weights and the fitted estimating equation, so a general extension requires distinguishing exact refits, local perturbations, paired-score approximations, and design-corrected estimating equations. None of those mathematical tools is new merely because the measurements originate from ML.

### Cited Findings

- The current README and manuscript define the raw change as a sum of cluster contributions, the design-scaled correction as a Horvitz–Thompson difference estimator, and exact enumeration up to 22 clusters. The manuscript explicitly extends exact additivity to OLS **with fixed covariates**, not replacement of the covariates. See equations `eq:add`, `eq:raw`, `eq:ht` and accompanying text. — [README](../../README.md); [manuscript source](../../validation_swaps.tex); [enumeration implementation](../../replicate_final.py).
- Measurement-error-specific local influence predates modern ML: Lee (1996), *Assessing Local Influence in Measurement Error Models*, studies how perturbing data, case weights, and assumptions changes generalized linear model parameters when covariates are measured with error. Only the publisher abstract was available in this search, so it supports this general characterization, not a specific formula. — [publisher DOI](https://doi.org/10.1002/bimj.4710380709).
- Giordano et al. (2019), *A Swiss Army Infinitesimal Jackknife*, §2 and Theorem 1, supplies finite-sample bounds for linear approximations to estimators under data reweighting. Its guarantees concern approximation to the refitted estimator, not automatic consistency for a population parameter. Smoothness, nondegeneracy, and perturbation-size conditions matter. — [full paper](https://proceedings.mlr.press/v89/giordano19a/giordano19a.pdf).
- Giordano, Jordan, and Broderick (2019), *A Higher-Order Swiss Army Infinitesimal Jackknife*, already develops higher-order reweighting expansions with finite-sample error bounds. Higher-order approximations to resampling distributions are therefore an existing technology. — [full paper](https://arxiv.org/pdf/1907.12116).
- Hu, Robinson, and Wang (2012), Theorems 2.1–2.2 and §4, develops finite-population tail approximations and compares them with Normal and saddlepoint approximations. The general problem of obtaining tails of sums sampled without replacement, including cases where Normal approximations are inaccurate, is established. — [full paper](https://www.numdam.org/item/10.1051/ps/2010027.pdf).
- Li and Ding (2017), Theorem 1, gives a sufficient finite-population Normal-approximation condition; Proposition 1 addresses consistency of the variance estimate under its conditions. — [full paper](https://arxiv.org/pdf/1610.04821).

### Inferences

- Four estimators/operations must be kept separate:
  1. **Raw partial replacement:** replace some proxy values with gold values and refit. Its target is the mixed-measurement dataset. It is generally not a consistent estimator of the all-gold target when an uncorrected proxy fraction remains.
  2. **Feature-amplitude derivative:** infinitesimally change feature values. Useful when those changes themselves are small.
  3. **Paired-score one-step approximation:** infinitesimally replace mass at a proxy record by mass at its gold counterpart. This retains the full finite record-level score difference; it is not the same as differentiating the feature value.
  4. **Design-corrected estimating equation:** inverse-probability weight the sampled gold-minus-proxy score contributions to estimate the all-gold score. This can yield valid all-gold estimation under the relevant assumptions, but its root is generally not exactly design-unbiased even though the estimated score is.
- Repeated partial refits define a sampling distribution for an existing statistic. Naming that distribution cannot itself establish novelty. Extending the diagnostic to covariates also does not establish novelty against target-specific two-phase sampling; the parent research covers directly overlapping allocation papers.
- One can write raw replacement as an estimating-equation perturbation with paired proxy/gold contributions. Thus generic reweighting expansions and their remainder analyses are close mathematical ancestors. A new theorem must improve on that machinery under a substantively different restriction, not just substitute new notation.
- Influence of an error on proxy bias is **signed**; choosing an unbiased correction sample usually concerns the conditional variance or conditional squared residual influence after available information has been used. Sampling only the largest signed changes can be inappropriate. An error can cancel in the endpoint gap yet materially increase validation uncertainty.
- The repository's text saying Normal planning is reliable “exactly when” corrections are diffuse is stronger than its cited sufficient theorem supports. Failure of the sufficient condition alone does not establish approximation failure in a particular finite sample. The all-equal contribution case is degenerate, with zero sampling variance. This is relevant to novelty framing, not a request to expand this task into an audit.

### Gaps

- Searches for the exact phrases `validation swap` and `validation-swap`, combined with statistics/econometrics/ML, did not locate an established relevant named method. This is only a terminology search, not evidence that the underlying method is new.
- I did not find a decisive prior paper for the exact combined proposal “finite group replacement interactions + an implementable tail-risk validation budget rule learned from a small pilot.” Absence from this scoped search does not establish novelty.
- The general optimal-allocation and causal-measurement papers are covered by other researchers in this project; their findings must be combined with this technology analysis.

## What changes mathematically when the ML variable is a covariate?

### Takeaway

Exact additivity generally disappears even for ordinary least squares. There is a simple exact identity, but it includes both a quadratic correction in feature error and a changed inverse Gram matrix. This identity is algebra, not a novel estimator.

### Cited Findings

- The repository's treatment-effect implementation is a difference in means, and its contributions are fixed before validation samples are enumerated. — [implementation](../../replicate_final.py).
- The independent script verifies the identity below against full OLS refits on 100 random cases and examines every two-record subset of a transparent four-record correction pattern. — [verification script](verify_covariate_substitution.py).

### Inferences

**Own derivation.** Let the proxy design matrix be \(X\), including the treatment indicator and intercept, and keep the outcome \(y\) fixed. Write

\[
A=X^TX,\quad \widehat\beta=A^{-1}X^Ty,\quad r=y-X\widehat\beta.
\]

For a validation set \(S\), let \(\Delta=\Delta_S\) contain gold-minus-proxy changes only in selected rows and in the measured covariate columns. Put

\[
B=X^T\Delta+\Delta^TX+\Delta^T\Delta,
\]
\[
d=\Delta^Tr-X^T\Delta\widehat\beta-\Delta^T\Delta\widehat\beta.
\]

If both design matrices have full column rank, direct subtraction of their normal equations gives

\[
\boxed{\widehat\beta(X+\Delta)-\widehat\beta(X)=(A+B)^{-1}d.}
\]

The treatment-coefficient change is \(e_D^T(A+B)^{-1}d\). This does not itself make that coefficient causal; the design and identification assumptions still have to justify the target.

The two approximate alternatives are:

\[
\text{feature derivative}
=A^{-1}\{\Delta^Tr-X^T\Delta\widehat\beta\},
\]
\[
\text{paired-score one-step}=A^{-1}d.
\]

For record \(i\), its complete score replacement is

\[
(x_i+\delta_i)(y_i-(x_i+\delta_i)^T\widehat\beta)
-x_i(y_i-x_i^T\widehat\beta)
=\delta_i r_i-x_i\delta_i^T\widehat\beta
-\delta_i\delta_i^T\widehat\beta.
\]

Thus the last, quadratic term belongs in the paired-score approximation even though it vanishes from the feature derivative. The paired-score approximation is first order in the **weight of replacement**, not in the **size of the feature error**. Comparing exact refits only with the weaker feature derivative would exaggerate an apparent improvement over influence-based methods.

Both \(B\) and \(d\) add across disjoint selected records, but \((A+B)^{-1}d\) does not. Therefore the sum of exact singleton replacements also fails to equal the exact group replacement. In outcome-only correction, \(B=0\), and the familiar fixed-weight exact additivity returns.

**Own elementary remainder calculation.** If \(C=A^{-1}B\) and an induced matrix norm satisfies \(\|C\|=q<1\), then

\[
\|(A+B)^{-1}d-A^{-1}d\|
\leq \frac{q}{1-q}\|A^{-1}d\|.
\]

This is a standard inverse-perturbation bound. It is informative only when the perturbation is controlled; it requires knowledge or a defensible bound on the corrections in \(B\). It is not a new finite-label design guarantee.

**Local numerical verification, not a benchmark.** All 100 random identity checks passed; maximum absolute discrepancy from independently refitted coefficients was \(3.24\times10^{-15}\). The deterministic example has 12 observations, an intercept, a binary treatment, and a proxy covariate. Outcomes, proxy values, and four correction directions are explicitly defined in the script. No model or error pattern was selected from a search for favorable performance.

| Scale of the four feature corrections | Exact change in treatment coefficient | Feature derivative | Paired-score one-step |
|---|---:|---:|---:|
| 0.001 | 0.000000187755 | approximately 0 | 0.000000187755 |
| 0.1 | 0.00187398 | approximately 0 | 0.00187755 |
| 1 | 0.157714 | approximately 0 | 0.187755 |
| 3 | 0.622556 | approximately 0 | 1.689796 |

The paired-score approximation is excellent for small corrections in this example, while large corrections can invalidate it. The vanishing feature derivative reflects a direction in which the leading change is quadratic; it does not show that all influence methods fail.

For the two-record subset \(\{6,7\}\), using zero-based script indices, the exact treatment-coefficient change is 0.137714. The sum of exact singleton changes is 0.104446, and the paired-score approximation is 0.163946. The script enumerates all six pairs, rather than reporting only one computed subset.

Local validation: script executed successfully; Black, isort, Flake8, and Ruff checks on the new script passed. No manuscript or existing analysis implementation was modified.

### Gaps

- These calculations establish algebra and a possible failure mechanism, not its empirical prevalence, a superior sampling design, or a novel theorem. We have not tested causal estimators with estimated nuisance functions, overlap violations, regularization, or nonsmooth matching.
- A general causal estimator may require influence through nuisance refitting and the evaluation sample. Orthogonality can eliminate some first-order nuisance sensitivity without eliminating finite measurement errors. A frozen-nuisance diagnostic and a full-pipeline refit answer different questions.
- Raw partially cleaned OLS and an inverse-probability corrected score estimator must be evaluated separately. The existing script concerns only the former; it must not be presented as a test of the latter's coverage or allocation performance.

## What novelty claim could survive, and what would decide it?

### Takeaway

Decision: **narrow and test**, with novelty unresolved for the narrow claim. Broad claims about parameter-specific measurement-error influence, target-aware validation sampling, or finite-sample reweighting approximations are already occupied. A credible contribution needs an additional decision result or guarantee that survives comparison with the closest active-inference and two-phase-design methods.

### Cited Findings

- Non-Normal finite-population tails and their numerical approximation already have dedicated theory and worked comparisons. — [Hu, Robinson, and Wang (2012)](https://www.numdam.org/item/10.1051/ps/2010027.pdf).
- Uniform control of refitting approximation error, including higher-order improvement, already exists under explicit regularity assumptions. — [Giordano et al. (2019)](https://proceedings.mlr.press/v89/giordano19a/giordano19a.pdf); [higher-order companion](https://arxiv.org/pdf/1907.12116).
- The repository acknowledges that all cluster discrepancies are required for its exact distribution, and that unseen contributions cannot be recovered nonparametrically from one small sample. — [README scope](../../README.md); [manuscript after equation `eq:htvar`](../../validation_swaps.tex).

### Inferences

- **Not sufficient for novelty:** replacing “prediction loss” by a coefficient in an influence formula; using squared score discrepancy to allocate labels; replacing outcome labels with covariate labels in a general estimating-equation formulation; calling a design-induced sampling distribution a swap distribution; showing a Normal approximation can fail; deriving the OLS identity above; simply adding second-order terms.
- **Plausible narrow problem:** construct a usable validation decision for a downstream causal/regression estimand when group corrections interact, the effective number of validation units is small, and the loss is a finite-sample probability of a consequential error. The method must improve decisions after paying for its pilot, not merely describe an already fully verified pool.
- **A falsifiable surviving claim could be:** under stated observable stability/overlap and bounded-error assumptions, an implementable pilot-based rule controls \(P(|\widehat\theta-\theta_{\rm gold}|>\delta)\) at level \(\alpha\), including uncertainty in error discovery and approximation, and achieves a meaningful budget improvement over existing target-IF allocation. This is a proposed claim to investigate, not a result established here.
- **Smallest decisive next test:** use an existing paired ML/gold dataset with a substantive downstream coefficient. Split off a pilot and an untouched assessment pool. Compare actual corrected estimators under uniform validation, the strongest feasible target-influence allocation, and a proposed finite-interaction/tail-aware rule. Include pilot cost; evaluate tail error, interval coverage, and budget to reach a prespecified tolerance across repeated samples. Diagnose whether failures arise from unknown correction values, approximation error, or the sampling design. Oracle fully labeled analyses are useful upper bounds, not deployable baselines.
- **Identification obstruction:** conditioned on a small pilot that misses an unobserved group, two finite populations can agree on everything observed and differ arbitrarily in that group's gold correction unless the problem imposes useful bounds or structure. Exact full-pool tails cannot be learned distribution-free in that setting. More clever influence mathematics cannot identify an unseen error.
- **Unbiased variance is not a risk certificate:** substituting an unbiased sample variance into Chebyshev's population-variance bound does not make the resulting realized number a high-probability upper bound. A pilot-derived guarantee needs uncertainty control on that variance or an alternative valid bound. This is particularly consequential when rare error groups are absent from the pilot.
- **Causal distinction:** proximity to the all-gold fitted coefficient is a measurement/estimation question. It does not establish that the all-gold coefficient identifies an ATE. Conversely, in randomized experiments, an imperfect baseline prognostic covariate may reduce precision without creating the same confounding bias as a mismeasured adjustment variable in an observational study. A proposed method needs one explicit causal setting before generalizing.

### Gaps

- The finite-interaction proposal could reduce to applying existing IJ bounds plus standard concentration/decision theory. Before calling it a paper, derive its strongest claim and attempt that reduction.
- It remains unknown whether corrections large enough to invalidate a good paired-score approximation occur often enough in realistic applications, and whether exact/higher-order corrections materially alter optimal validation decisions.
- This note does not establish a distinct theorem or algorithm. The honest present conclusion is an identified research question with specific novelty obstacles, not confirmed novelty.

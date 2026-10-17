# Learned domains, held-out audits, and local calibration in the original validation-swaps manuscript

## What directly precedes the proposed domain-selection workflow?

### Takeaway
As of March 30, 2026, the ingredients have substantial precedent, and a February 2026 paper already learns regions where predictions improve downstream inference. I did not find the exact combination of row-correction attribution, observable low-distortion domain selection, held-out coefficient-equivalence auditing, and calibration/fallback. That is a plausible synthesis contribution, not an established novel estimator or theorem.

### Cited Findings
- **Closest newly located comparator: Xu and Sun, Filtered Prediction-Powered Inference (FPPI), first submitted February 11, 2026.** It learns a feature-space region where predictions aid mean and GLM estimation; region recovery uses margin assumptions. The GLM objective, equations (12)–(13), retains the entire labeled-data loss and restricts only the prediction correction. Thus it retains the full-population target. Region choice improves variance/AMSE; it is not an equivalence certificate for the largest domain with small raw-proxy coefficient distortion. No Shapley attribution or the manuscript's discovery/holdout equivalence gate appears in the inspected method sections. **Strong comparator, not exact duplicate.** — [Submission record](https://arxiv.org/abs/2602.10464); [full text, Sections 2–4, Theorems 2, 6–7](https://arxiv.org/pdf/2602.10464).
- **Learn Then Test (LTT), Angelopoulos et al., 2021 preprint, inspected September 2022 version.** Given valid p-values for each candidate's unsafe-risk null, familywise-error control permits data-dependent choice among certified candidates. Its selective regression example controls prediction MSE on accepted cases. Its general theorem allows other risks with valid tests and does not require risk monotonicity. This supplies a natural architecture for auditing many candidate domains, but the paper does not derive coefficient-distortion tests for mismeasured regressors. — [Full text, Theorem 1 and Section 3.3](https://arxiv.org/pdf/2110.01052).
- **El-Yaniv and Wiener (2010):** selective classification formalizes trading retained population share against error and learning a reject rule. This is an antecedent of the acceptance/coverage objective, not downstream regression-coefficient measurement error. — [Primary paper](https://www.jmlr.org/papers/volume11/el-yaniv10a/el-yaniv10a.pdf).
- **Crump, Hotz, Imbens, and Mitnik (2009):** finds covariate-defined subpopulations permitting more precise ATE estimation, explicitly changing the estimand. Its restriction addresses overlap, not proxy distortion. — [Publication record](https://dash.harvard.edu/entities/publication/73120378-86bc-6bd4-e053-0100007fdf3b); [author paper](https://users.nber.org/~rdehejia/!@$AEM/Topic%2009%20Matching%20Advanced/readings/Crump_Hotz_Imbens_Mitnik_overlap.pdf).
- **Athey and Imbens (2016):** separates data for learning partitions and estimating effects within them. Honest subgroup estimation is therefore established; the manuscript's distinctive candidate target is measurement fidelity, not heterogeneity itself. — [Author paper](https://www.aeaweb.org/conference/2016/retrieve.php?pdfid=13619&tk=YBK4Shn5).

### Inferences
- The coherent research question is: **Can we find the largest observable population on which replacing truth with a specified proxy changes a chosen regression coefficient by at most a substantively meaningful amount, and certify that claim with affordable validation?** This differs from optimal verification allocation for a fixed population.
- FPPI poses an important challenge: why change the target through exclusion when filtering a valid correction might retain the original target? The original method must demonstrate a setting where its diagnostic or operational constraint adds value.
- Predicting absolute row attributions is a heuristic for proposing domains, not a proof of low coefficient distortion. Restriction changes regression geometry; the attributions computed in the original sample need not rank contributions to a restricted regression. Signed cancellation also makes coefficient distortion nonmonotone in the threshold even though accepted sets are nested.

### Gaps
- This scoped search found no exact predecessor, not evidence of uniqueness. Search terms included proxy/regression abstention, measurement-error trimming, selective downstream inference, and prediction-powered filtering.
- Later 2026 papers are not used to defeat March 30 priority. FPPI is a preprint; no peer-reviewed publication was established here.

## Are the audit, calibration proposition, and precision criterion new technology?

### Takeaway
They are primarily established statistical ideas adapted to this workflow. The strongest potential novelty lies in the object being diagnosed and the integrated procedure's demonstrated value.

### Cited Findings
- **Coefficient equivalence:** Counsell and Cribbie (2015) study equivalence tests for regression-coefficient differences via two one-sided tests. Their displayed regression method assumes independent coefficients, so it is not the exact paired gold/proxy test. It nevertheless directly precedes the principle of declaring a coefficient difference practically negligible only when an uncertainty interval lies inside a tolerance interval. — [Full paper, Section 4](https://yorkspace.library.yorku.ca/server/api/core/bitstreams/dd1ac706-b3e5-4df2-98fe-413c639df601/content).
- **Regression calibration:** Spiegelman, Logan, and Grove (2011) develop calibration under heterogeneous measurement error, review nondifferential-error assumptions, and note exactness of the conditional-mean calculation for linear regression. This predates the original manuscript's local-calibration proposition. — [Publisher full text, Section 2](https://www.degruyterbrill.com/document/doi/10.2202/1557-4679.1259/pdf).
- **Bias-aware intervals:** Armstrong, Kolesár, and Kwon, first posted 2020, optimize bias/variance tradeoffs and use critical values reflecting a justified worst-case bias bound. Equation (6) uses a folded-normal quantile, rather than simply adding a bias allowance to 1.96 standard errors. This is a comparator for the precision criterion, not a measurement-error domain-selection solution. — [Full paper, Section 2](https://arxiv.org/pdf/2012.14823).
- **Conformal Risk Control (2022 preprint):** the basic guarantee assumes bounded monotone losses; Section 2.4 explicitly shows failure without monotonicity and discusses modifications. A refitted coefficient discrepancy across nested domains is not automatically such a loss. — [Full paper, Section 2.4](https://arxiv.org/pdf/2208.02814).

### Inferences
- For a signed coefficient difference d, the audit `|d_hat| + 1.96 SE(d_hat) <= delta` is the requirement that its approximate 95% CI lie within `[-delta,delta]`. It is a conservative equivalence screen relative to conventional 5% TOST using a 90% two-sided CI. The paired SE must retain gold/proxy covariance. This controls an erroneous certification event under assumptions; it does not mean a 95% posterior probability of safety conditional on passing.
- LTT can organize multiple domain equivalence tests if valid p-values exist. The normal paired-coefficient approximation gives asymptotic rather than distribution-free finite-sample validity. A single frozen domain tested once on independent audit data avoids multiple-threshold testing on that audit fold.
- The calibration proposition is iterated expectations under a strong nondifferential restriction and relevance. For fixed observable `R=R(Z)`, `E[X|Z,R=1]=E[X|Z]` on the domain. Local fitting changes approximation and sampling, not this population identity.
- The proof uses the true conditional expectation even though the manuscript initially writes an estimated one. It supplies no guarantee for arbitrary fitted calibration, no generated-regressor standard error, and no learned-domain theorem. Crossfitting helps remove overfitting but alone does not remove first-stage bias or guarantee root-n inference.
- Adding delta to a conventional interval is conservative if delta is a genuine bound for the **deployed estimator's** bias and the sampling interval is valid. A point audit supplies neither condition. If the bound itself is estimated, uncertainty allocation must be handled jointly.

### Gaps
- Full-text primary sources were read for FPPI, LTT, equivalence, bias-aware intervals, selective classification, CRC, and calibration. Some PMC pages blocked access; publisher/author PDFs supplied alternatives. Detailed finite-sample proof extensions to paired coefficient equivalence were not established.

## What needs fixing before the synthesis can support a research claim?

### Takeaway
The original method warrants a targeted proof-and-benchmark exercise, not dismissal as identical to active inference. However, its current local-calibration theorem does not justify the advertised audit-to-borrowing chain.

### Cited Findings
- The inspected repository source is `git show ab1c17f:validation_swaps.tex`, titled *Partial Credit: Diagnosing Proxy Covariates with Validation Swaps*. It audits gold versus **raw proxy** coefficients, then deploys a fitted **calibrated** regressor, and explicitly acknowledges conditional selection and nonformal point-audit guarantees. These statements are from the user-owned manuscript, not external literature.

### Inferences
1. **Audit the deployed object.** Raw-proxy equivalence need not imply calibrated-estimator equivalence. Freeze the proposed calibration and domain before an independent audit, or prove a bridge. This is a correctness condition, not evidence that the research question lacks value.
2. **Separate stages genuinely.** If discovery SIMs use audit outcomes through full-sample regressions, a nominal discovery/audit split is not independent. Reusing audit data to refit calibration also changes what passed the audit. Conditional analysis of a fixed selected domain needs an appropriate independent estimation sample or a valid selection-aware argument.
3. **Compare like targets.** Under heterogeneity, a domain estimate and full-population validation-only estimate answer different questions. Comparing interval widths alone is not a pure efficiency comparison. Compare validation-only estimation on that same domain, or make the value of population coverage explicit.
4. **Prove a useful objective.** A promising claim is controlled false certification for the actual coefficient-distortion functional, plus near-maximal retained share under explicit regularity/margin conditions. Compare against simple proxy-error screening, influence approximations, and direct equivalence-based domain search—not only global correction.
5. **Do not overinterpret flat hybrid paths.** Correcting a small validation fraction leaves the rest uncorrected. A flat path can reflect dilution or cancellation; it does not establish full-population proxy adequacy.

### Gaps
- Audit researcher is checking the implementation against claimed Shapley weights and simulation assumptions; those results should be incorporated separately.
- **Decision:** continue only with a short, explicit test of whether attribution-based candidate construction yields better certified coverage or lower validation cost than direct domain-equivalence search and standard scores. If not, retain swaps as an explanatory diagnostic rather than promote a new inferential method.
- **Source ledger:** each external source carried a 200-word allowance; this note uses under 150 source-derived words per paper, no extended quotes. FPPI about 110, LTT about 90, selective classification 40, Crump 45, Athey–Imbens 45, equivalence 70, calibration 50, bias-aware 70, CRC 50. Subsequent mathematical interpretation and recommended repairs are our deductions. Treat versions/mirrors as one source when synthesizing.

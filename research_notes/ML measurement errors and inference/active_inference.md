# Active statistical inference and parameter-specific measurement validation

## What does Zrnic and Candes already establish?

### Takeaway

The relevant comparison is inferential sampling based on target-specific score discrepancies, not influence on ML training predictions. ASI directly occupies this territory; covariate measurement error is an algebraic specialization, subject to its regularity conditions.

### Cited Findings

- Zrnic and Candes (ICML 2024), §§5–7: their corrected objective is the proxy loss plus inverse-probability-weighted gold-minus-proxy loss (Eq. 6). Section 7.1 minimizes asymptotic variance for a selected parameter coordinate using

  \[
  \pi(X)\propto\sqrt{E[\{(\nabla\ell_{\theta^*}(X,Y)-\nabla\ell_{\theta^*}(X,f(X)))^\top H_{\theta^*}^{-1}e_j\}^2\mid X]}.
  \]

  For GLMs, the rule is conditional residual RMS times \(|X^\top H^{-1}e_j|\); Eq. 34 gives the stronger, unobservable oracle using actual residual magnitude. Section 7.2 estimates error magnitudes and the Hessian and stabilizes sampling through uniform mixing. Theorem 1 assumes estimator consistency, smoothness, positive-definite population Hessian, and stabilization of the sampling-scale parameter; Appendix A.2 proves its expansion. Sequential results add variance-convergence conditions. Appendix C gives some nonasymptotic extensions. The displayed GLM examples predict outcomes. Their general convex loss does not require the costly variable to be an economic outcome. [Full text, arXiv v2](https://arxiv.org/html/2403.03208v2), [official ICML entry](https://proceedings.mlr.press/v235/zrnic24a.html).

Access: read the main inferential/design sections, relevant proof and assumptions in arXiv v2. Official conference PDF links returned errors; therefore equation numbers above are verified against v2, not independently against the final PDF. The key general oracle formula in §7.1 is unnumbered. Source-derived description deliberately kept below 200 words.

### Inferences

The following are our reductions and algebra, not claimed new results in the paper.

**Covariate reduction.** Let the observed record be \(V=(Y,A,C,\widetilde U,\text{other cheap information})\); \(U\) is a costly true covariate and \(\widetilde U=f(V)\) its available proxy. Define \(q(V,U)=(1,A,C,U)\) and

\[
\ell_\beta(V,U)=\tfrac12\{Y-q(V,U)^\top\beta\}^2.
\]

Set the paper's observed \(X=V\) and costly label \(Y_{paper}=U\). Its general corrected loss and target-specific oracle now apply to a regression using the expensive covariate. The scalar-label setting already covers one missing covariate; vector missing components require checking the stated domain or extending the notation. Merely changing which side of the regression contains the ML prediction is not a novelty argument.

**Finite score replacement.** For the OLS estimating score \(s=q(Y-q^\top\beta)\), write \(q=\widetilde q+d\), \(\widetilde r=Y-\widetilde q^\top\beta\). Then exactly

\[
s_{gold}-s_{proxy}
=d\widetilde r-\widetilde q(d^\top\beta)-d(d^\top\beta).
\]

This contains a quadratic correction. A derivative with respect to the raw feature omits the last term and can fail for a binary relabeling or large continuous correction. Use the finite score difference inside the estimator's first-order expansion. An influence approximation is infinitesimal in the *mass of replaced observations*, not necessarily in the amount each feature changes.

For diagnostics of the existing proxy fit, evaluate at \(\widetilde\beta\) and the proxy Hessian: a single replacement changes coordinate \(j\) approximately by \(n^{-1}e_j^\top\widetilde H^{-1}(s_{gold}-s_{proxy})\). For efficient corrected inference, evaluate the limiting score and Hessian at the gold target. These are different objects; summing local proxy-fit effects need not recover a large simultaneous correction because the Hessian and fitted coefficients change.

The OLS finite score identity passed 1,000 randomized local numerical checks using Python's standard library (seed 17, tolerance 1e-12). This checks the algebra, not asymptotic guarantees.

### Gaps

- Relabeling does not prove all hypotheses automatically. With erroneous regressors, subtracting proxy losses can make the finite-sample corrected objective indefinite even though each original loss is convex. A compact parameter space or consistent estimating-equation solution may be needed.
- A causal interpretation additionally requires identification assumptions. A regression coefficient is not automatically an ATE.
- Cross-fitting, learned propensity/outcome regressions, and nuisance-estimation remainder control for a particular causal target require their own analysis. ASI's loss formula alone does not supply that proof.

## Which influence quantity should determine the validation design?

### Takeaway

Distinguish errors relative to a fixed proxy correction from the residual uncertainty after efficient augmentation. The first uses conditional mean-square error; the second uses conditional variance.

### Cited Findings

- Ma and Candes, OPAL (June 2026 preprint), §2.1 Eq. 7 learns smooth log-linear spline sampling policies through convex optimization. Section 2.2 Eq. 8 treats an observed-data EIF \(h(X)-\psi+R\zeta(X,Y)/\pi(X)\), with \(E[\zeta\mid X]=0\), yielding the policy-dependent variance \(E[E(\zeta^2\mid X)/\pi(X)]\). Appendix A.4 connects full-data and observed-data IFs. One-step and TMLE constructions are discussed; Appendix G Assumption 4 requires a negligible asymptotic remainder rather than establishing nuisance conditions for every target. Appendix H considers cross-fitting. This is a relevant recent baseline for learned sampling policies and general IF-based targets. [Full text](https://arxiv.org/html/2606.03211), [version/date](https://arxiv.org/abs/2606.03211).

### Inferences

Here is an independent derivation clarifying the distinction. Let \(\phi(W)\) be a scalar, mean-zero gold-data influence function, \(V\) the information available before validation, and \(g(V)\) any fixed augmentation. With known sampling probability \(\pi(V)\), use

\[
\phi_{obs}=g(V)+\frac{R}{\pi(V)}\{\phi(W)-g(V)\},\qquad R\perp W\mid V.
\]

Define \(m(V)=E[\phi(W)\mid V]\), \(v(V)=\operatorname{Var}(\phi(W)\mid V)\). Conditional expectation and variance give

\[
\operatorname{Var}(\phi_{obs})
=\operatorname{Var}(\phi)
+E[(\pi^{-1}-1)\{\phi-g\}^2].
\]

Consequently, for fixed \(g\), the policy-dependent term is \(E[c_g(V)/\pi(V)]\), where

\[
c_g(V)=E[(\phi-g)^2\mid V]=v(V)+\{m(V)-g(V)\}^2.
\]

Under \(E\pi\leq b\), Lagrange differentiation yields \(\pi(V)=\min\{1,\sqrt{c_g(V)/\lambda}\}\), with \(\lambda\) chosen for the budget. If validation costs vary, \(\pi\propto\sqrt{c_g/\text{cost}}\), capped at one. A positive floor protects against badly estimated near-zero contributions.

With optimal augmentation \(g=m\), the design criterion becomes \(c_g=v\), and

\[
\operatorname{Var}(\phi_{obs})=\operatorname{Var}(m)+E[v/\pi].
\]

For fixed proxy-score correction, \(g\) is the proxy's target-projected score, so the design also spends labels on squared conditional bias relative to that proxy. The oracle depends on the estimator being optimized. Predicting the missing feature and plugging it into a nonlinear score generally differs from predicting the gold score: \(s(V,E[U\mid V])\neq E[s(V,U)\mid V]\). For OLS covariates, second moments such as \(E[U^2\mid V]\) matter.

An implementable study should therefore compare (i) uniform validation, (ii) raw prediction-uncertainty sampling, (iii) projected score-discrepancy sampling with stabilization, and (iv) efficient IF-residual sampling with pilot-estimated augmentation. A comparison only to raw classifier confidence would omit the strongest established conceptual baseline.

### Gaps

- This derivation presumes a valid full-data IF and negligible nuisance remainder. For each proposed causal estimator, those must be checked.
- Asymptotic variance design does not automatically optimize rare-event sign reversals, finite-cluster tail probabilities, or the information gained for training future nuisance models. Such distinctions are possible research questions, not established novelty claims.
- The parent researcher is investigating the direct causal precedent Wang, Wang and Miao (arXiv:2312.10596), including expensive-confounder ATE examples; include that before concluding covariate/causal novelty.

## What newer work and practical baselines must be considered?

### Takeaway

Robust policy fitting and adaptive multiwave validation are already active research areas. A useful contribution needs a specific failure of these methods, or a decision criterion they do not address, with realistic pilot data and equal total labeling costs.

### Cited Findings

- Li, Zrnic and Candes, *Robust Sampling for Active Statistical Inference* (NeurIPS 2025), §§2–3: optimize along budget-preserving paths connecting uniform and active policies. Theorems 1–2 give asymptotic no-worse-than-either-endpoint variance when the optimizing path parameter is consistently estimated. Equation 4 adds a minimax uncertainty set for error-model misspecification; it is not an unconditional finite-sample dominance theorem. Section 3 uses target-projected squared score discrepancies for general M-estimators. Algorithm 1 is the practical minimax policy algorithm. Burn-in labels count against the total budget (§4.1). [Full text](https://arxiv.org/html/2511.08991v1), [official proceedings](https://papers.neurips.cc/paper_files/paper/2025/hash/6389470564214983604d1ac81631c2c5-Abstract-Conference.html).
- Kluger and Bates, *M-estimation under Two-Phase Multiwave Sampling with Applications to Prediction-Powered Inference* (February 2026, revised March): abstract reports valid adaptive multiwave estimation with pretrained proxy variables, a Multiwave Predict-Then-Debias estimator, asymptotic linearity/normality, and approximately greedy sampling. This agent accessed the abstract only; full-text assessment assigned elsewhere. [Abstract and version history](https://arxiv.org/abs/2602.16933).

### Inferences

Recommended practical experiment: start with a probability-sampled pilot; estimate corrected target, score/IF, conditional residual second moment, and a stabilized policy; acquire subsequent gold data using recorded probabilities; refit with the appropriate weighting/cross-fitting; compare precision and coverage at fixed total cost across budgets. Include the pilot in every budget. The specific machinery is an influence expansion, a two-phase/AIPW correction, and constrained allocation—not ML-training-data influence.

For sample size selection, invert a precision-versus-budget curve using the fitted design variance, while reporting uncertainty from learning that curve. To claim novelty beyond variance-optimal allocation, specify a substantively distinct objective and show an implementable advantage against target-specific and robust baselines, not only against uniform sampling.

### Gaps

The search is focused rather than exhaustive. No claim of absence is justified for finite-sample tail-optimal design, cluster-level validation, or adaptive stopping. Additional 2026 search hits on multiple predictors and adaptive stratification were not investigated deeply because they were less direct than the causal two-phase and OPAL precedents.

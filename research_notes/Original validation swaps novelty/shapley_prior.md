# Original validation swaps: Shapley and replacement-path prior art

## Are the manuscript's mathematical objects new?

### Takeaway

No new mathematical attribution object is established: the manuscript's SIM is ordinary baseline Shapley applied to a refitted coefficient; its Bernoulli swap path is the diagonal of the game's multilinear extension. The particular econometric diagnostic application remains distinguishable from ML predictor debugging.

### Cited Findings

- **Original object.** At commit `ab1c17f` (March 30, 2026 cutoff supplied by coordinator), manuscript §3 defines a coalition by replacing validated proxy covariates with their gold values while retaining every row, then refitting a target regression coefficient. The endpoint is a partially corrected hybrid, not the full-gold oracle when validation is incomplete. Source accessed with `git show ab1c17f:validation_swaps.tex`; implementation also read.
- **Baseline replacement Shapley.** Sundararajan and Najmi, ICML 2020, §2.2 Eq. 3 defines the game as a scalar function evaluated on input coordinates selected from an explicand or fixed baseline. Section 4 studies its axioms; the paper traces variants to earlier work. This is the precise algebraic construction underlying SIM once the coordinates are validated rows and the scalar function refits the regression. [Official full PDF](https://proceedings.mlr.press/v119/sundararajan20b/sundararajan20b.pdf). First preprint: August 22, 2019, verified on [arXiv](https://arxiv.org/abs/1908.08474).
- **Multilinear extension.** Owen's *Multilinear Extensions of Games*, published January 1, 1972, defines the extension on participation probabilities; publisher metadata and abstract accessed, original full proof not accessed. [Original publisher](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.18.5.64).
- **Probabilistic Shapley integral.** Okhrati and Lipani, first posted October 22, 2020, §III Eqs. 2–3 expresses Shapley as the integral over inclusion probability of expected marginal contribution under Bernoulli coalitions, attributing the representation to Owen. Section IV gives sampling algorithms and convergence results. [Full paper](https://arxiv.org/pdf/2010.12082), [verified date](https://arxiv.org/abs/2010.12082).

### Inferences

Exact reduction: fix all outcomes, other regressors, and nonvalidated rows. Let \(b=(\widehat X_i)_{i\in V}\), \(a=(X_i)_{i\in V}\), and let \(f(z)\) return the coefficient after inserting \(z\) in the validated positions. Then

\[
v(S)=f(a_S,b_{V\setminus S})
\]

is exactly baseline Shapley's coalition game. Calling it a coefficient rather than a prediction does not change this mathematics. The choice of function and baseline changes the interpretation and could motivate a useful application.

For \(m=|V|\), define

\[
F(q)=\sum_{S\subseteq V}v(S)\prod_{i\in S}q_i\prod_{i\notin S}(1-q_i).
\]

The manuscript path is \(\phi_V(p)=F(p\mathbf1)\). Set

\[
a_i(p)=E_{S\subseteq V\setminus i,\,\mathrm{Bern}(p)}[v(S\cup i)-v(S)].
\]

Then \(a_i(p)=\partial_iF(p\mathbf1)\), \(\phi'_V(p)=\sum_i a_i(p)\), and

\[
\mathrm{SIM}_i=\int_0^1 a_i(p)\,dp,
\quad
\sum_i\mathrm{SIM}_i=v(V)-v(\varnothing).
\]

The identity follows directly because integrating each coalition weight gives
\(\int_0^1p^k(1-p)^{m-1-k}dp=k!(m-1-k)!/m!\). Thus the path and SIM are two views of one established construction. Local exact-rational tests checked all 120 weight identities for \(1\leq m\leq15\).

### Gaps

No primary paper found in this focused search uses this exact rowwise gold/proxy replacement game specifically to explain an internally validated econometric coefficient gap. That supports an unresolved application-level novelty assessment, not a claim of first invention.

## How close are data valuation, repair explanation, and measurement-error paths?

### Takeaway

These are relevant neighboring literatures, but their actual coalition games differ. They establish the availability of the ingredients without establishing prior publication of the complete proposed workflow.

### Cited Findings

- **Data Shapley.** Ghorbani and Zou, ICML 2019, §2 and Proposition 2.1 values adding training records to subsets using a downstream performance score. Section 3.1 provides permutation Monte Carlo; §4 includes noisy-label identification and data acquisition. The original game's rows are included or excluded, rather than retained and corrected. [Official full paper](https://proceedings.mlr.press/v97/ghorbani19c/ghorbani19c.pdf).
- **DataScope.** Karlas et al., ICLR 2024 (preprint April 23, 2022), §2 Eqs. 1–2 uses Shapley over subsets of source training tuples, with utility measuring pipeline accuracy or fairness on validation data. Section 5 evaluates label-repair prioritization. It is a genuine data-cleaning application, but its player action is inclusion, and its target is prediction quality. [Official full paper](https://proceedings.iclr.cc/paper_files/paper/2024/file/ccdd1961dcd0245d9dadceccead0fad1-Paper-Conference.pdf), [first preprint](https://arxiv.org/abs/2204.11131).
- **Explaining data repairs.** Deutch, Frost, Gilad and Sheffer, CIKM 2021, §3 Definitions 3–5 attributes a particular repaired cell to database cells or integrity constraints. The value function records whether the repair algorithm still produces the same repaired value; excluded cells are replaced by null. It explains the *repair process*, not how verified corrections change a downstream coefficient. Full PDF confirms conference November 1–5, 2021. [Author-hosted full paper](https://amirgilad.github.io/publication/cikm21/CIKM21.pdf).
- **SIMEX.** Cook and Stefanski, JASA December 1994, adds known increments of simulated measurement error, fits the estimator-versus-noise relationship, then extrapolates toward zero measurement error. Abstract accessed. It precedes the general idea of plotting coefficient sensitivity to measurement quality but does not perform verified-row swaps or their Shapley decomposition. [Original article](https://www.jstor.org/stable/2290994).

### Inferences

The strongest novelty objection is the exact baseline-Shapley/multilinear reduction, not a loose analogy to Koh–Liang or DataScope. Conversely, baseline Shapley being available does not mean this particular diagnostic has already been empirically evaluated in econometrics. A paper needs to establish why the application requires more than putting an existing attribution method around a regression refit.

### Gaps

Searches covered Shapley with data correction, data replacement, dirty/clean coalitions, measurement error, regression-coefficient changes, and repair. They did not establish an exact prior application. A thorough bibliographic review could still find one. Later work, including June 2026 OPAL, cannot defeat novelty as of March 30, 2026; it is excluded from this historical comparison.

## What survives, and what should be tested next?

### Takeaway

**Decision: narrow and test the application; stop claiming a new Shapley/path technology.** A plausible surviving claim is that exact finite correction attribution identifies consequential nonlinear interactions missed by cheaper influence diagnostics, and materially improves a specified validation decision.

### Cited Findings

- Original manuscript §3 gives the Shapley formula, whereas `compute_sim` in the same commit samples all other corrections independently with probability 0.5 and averages marginal toggles. This targets the probabilistic Banzhaf value, not the manuscript's Shapley value. The coordinator's audit researcher is testing the implementation discrepancy. This is a correctness issue separate from conceptual novelty.

### Inferences

In path notation, the code estimates \(a_i(1/2)\); manuscript SIM integrates \(a_i(p)\) across all \(p\). Consequently, the sum of code scores targets \(\phi'_V(1/2)\), not generally the endpoint gap. A standard repair is permutation sampling or uniform sampling of \(p\in[0,1]\) followed by Bernoulli coalitions, with Monte Carlo uncertainty reported.

Required discriminating experiment: compare true Shapley, midpoint Banzhaf, finite one-row replacement effects, and projected influence contrasts on the *same* paired validation data. Use held-out decisions and equal computation/validation budgets. Demonstrate whether Shapley's treatment of interactions changes useful decisions enough to pay for repeated refits. Do not treat a decomposition identity as evidence of decision quality.

Further constraints: all coalition regressions must be well defined; near-singular hybrids can dominate values. Flat average paths can conceal opposing corrections, so report spread and signed attribution. SIM is conditional on the observed validation set and surrounding proxy dataset; it is not automatically a population unit property or an optimal sampling score. Learning \(E[|SIM|\mid Z]\) needs evidence of stability and transport to new rows. The manuscript's hybrid endpoint cannot by itself certify low full-population measurement distortion.

### Gaps

The initial manuscript's domain discovery, held-out audit, and precision decision may supply a distinct workflow, but this subtask does not establish its statistical validity or comparative value. Those claims need the separate audit and domain-selection literature review. Existing toy simulations alone do not settle whether the proposed diagnostics have a useful segment.

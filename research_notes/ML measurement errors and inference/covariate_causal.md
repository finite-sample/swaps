# ML-generated covariates, causal inference, and targeted validation

## Is parameter-specific validation for ML covariates or causal inference already covered?

### Takeaway
Yes, including remarkably direct 2026 work. The correct comparison is not training-data influence: it is two-phase measurement, error-prone regressors, and target-specific validation. A new contribution cannot consist solely of mapping ML errors through a coefficient's influence function and allocating labels accordingly.

### Cited Findings
- **Most direct: Kluger and Bates (2026), M-estimation under Two-Phase Multiwave Sampling with Applications to Prediction-Powered Inference.** Section 4.1, equation (15), defines the squared difference between the gold-data and proxy-data influence contributions for a specified coefficient. Equation (17) allocates labels according to its conditional expectation, with an adjustment for observations surviving prior waves. Section 4.2 estimates this conditional expectation from previously verified observations using ML or stratification. Section 3 supplies asymptotic inference accounting for adaptive dependencies. The theory concerns convex M-estimation, not a universal theorem for arbitrary nuisance-estimated causal estimators. The sampling rule is approximately greedy; it does not solve a full joint optimization of all waves and budgets. — [Full text, especially Sections 3–4 and 6](https://arxiv.org/html/2602.16933v1).
- **Wang, Wang, and Miao, A Maximin Optimal Approach for Sampling Designs in Two-phase Studies** (first posted December 2023; accepted Statistica Sinica preprint available by the research cutoff). Lemma 1 derives the observed-data efficient influence function from its full-data counterpart. Theorem 1 minimizes its efficiency bound using sampling probabilities proportional to the conditional standard deviation of the full-data influence function, capped at one. A pilot estimates the unknown conditional moments. Their multidimensional criterion protects the worst relative efficiency improvement across coordinates. This is an efficiency criterion, not a worst-tail guarantee. — [Accepted main paper, Lemma 1 and Theorem 1](https://www3.stat.sinica.edu.tw/preprint/SS-2024-0359_Preprint.pdf); [publication status](https://www3.stat.sinica.edu.tw/statistica/fp/SS-2024-0359.html).
- **Direct causal-confounder example:** Wang et al.'s supplement, Example S3, explicitly sets phase-one data to outcome, treatment, and cheap confounders; expensive confounders are verified in phase two. It uses the ATE efficient influence function under unconfoundedness given all confounders. Example S2 similarly treats expensive regression covariates. Thus these applications are explicit, not merely hypothetical consequences of a generic theorem. **Important limitation:** Section S3 acknowledges it does not rigorously establish the proposed one-step estimator's behavior when reusing pilot data under the induced dependence; it leaves that investigation open. This limits claims about fully proved adaptive implementation, not the population optimal-design result. — [Supplement, Examples S2–S3 and Section S3](https://www3.stat.sinica.edu.tw/preprint/supp/2024-0359_supp.pdf).
- **Chen and Lumley (2022):** For IPW regression estimation, optimal stratified allocation uses within-stratum standard deviations of target-coefficient influence functions. For generalized raking it uses residual influence functions after projection onto their best phase-one estimates. This makes design depend on both the parameter and the estimator/augmentation, not merely raw measurement error. They discuss multiwave estimation of unknown influences and residuals. Their measurement-error examples include continuous and discrete outcomes/covariates. — [Sections 4.2–4.3 and discussion](https://pmc.ncbi.nlm.nih.gov/articles/PMC8918008/).
- **Kluger, Lu, Zrnic, Wang, and Bates (2025): Prediction-Powered Inference with Imputed Covariates and Nonuniform Sampling.** Directly covers arbitrary subsets of variables being imputed, including covariates and both sides of a regression. Predict-Then-Debias combines estimates on gold and imputed data, with optional matrix tuning. Main assumptions include MAR conditional on proxies, known sampling probabilities bounded away from zero and one, and asymptotic weighted linearity of the component estimators. Bootstrap validity requires a corresponding bootstrap approximation. The paper discusses weighted, stratified, and clustered sampling; its main displayed theory starts from IID observations. It does not itself derive the adaptive selection rule supplied by the 2026 extension. — [Full text, Sections 2–4](https://arxiv.org/html/2501.18577v2).
- **Fong and Tyler (2021): Machine Learning Predictions as Regression Covariates.** Explicitly treats ML-imputed regressors, showing invalid naive inference and developing a sample-splitting/GMM method that combines labeled-sample information with predictions used as instruments under an exclusion restriction. This is older direct precedent on the substantive problem but is not the closest result on target-dependent optimal verification. — [Author-hosted paper](https://fongsurvey.polisci.lsa.umich.edu/papers/classification-error.pdf); [DOI](https://doi.org/10.1017/pan.2020.38).
- **Egami et al. (2023), Design-Based Supervised Learning:** its main logistic-regression exposition uses a costly outcome, known positive annotation probabilities, and crossfitted predictions. It also states extension to general moment estimators. Do not characterize the entire framework as intrinsically outcome-only because of its main example, or claim it establishes every causal extension without checking the moment and nuisance conditions. — [Full conference paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/d862f7f5445255090de13b825b880d59-Paper-Conference.pdf).
- **Recent direct econometrics:** Battaglia, Christensen, Hansen, and Sacher study regressions with AI/ML-generated variables and propose explicit bias correction or joint maximum likelihood. Zhang, Xue, Yu, and Tan (published online May 27, 2026) combine TSLS/GMM with DML for ML-generated regressors in partially linear models. These establish activity on generated covariates and causal estimation, but neither abstract establishes target-optimal validation design. — [Battaglia et al. primary preprint](https://arxiv.org/abs/2402.15585); [Zhang et al. publisher abstract](https://pubsonline.informs.org/doi/abs/10.1287/isre.2024.1370).

### Inferences
- **Novelty verdict for the broad idea: already covered.** The conjunction “target-specific influence of ML measurement errors, then adapt verification to reduce uncertainty about that target” is stated operationally in Kluger–Bates; “optimal verification of causal confounders” is explicit in Wang et al. Both must be compared mathematically before claiming a new method.
- The ML provenance of a proxy does not itself create a different identification/design problem. The target functional, missing-variable pattern, augmentation, measurement cost, dependence, and risk criterion determine whether an existing result applies.
- A useful diagnostic/application could still contribute if it shows a failure existing methods systematically miss or supplies validated implementation. A new name or replacing outcome errors with covariate errors is insufficient.
- **Budget versus allocation:** the closest results optimize allocation conditional on a specified expected sampling fraction or wave budget. They do not generally jointly choose the overall cohort size, total validation budget, wave count, and a scientific-error tolerance. Translating a variance curve into a sample-size choice is useful but does not automatically constitute a new design theorem.

### Gaps
- Fong–Tyler publisher/full PDF opening failed, although indexed author-paper passages were available. Its finer assumptions were not independently audited here.
- Zhang et al. and Battaglia et al. were checked at abstract level only; do not claim detailed coverage or noncoverage of selection/sample-size questions from those abstracts.
- No claim of exhaustive absence of a particular finite-sample/tail diagnostic is justified by this scoped search. Other researcher(s) are examining Active Statistical Inference and OPAL; integrate those results.
- **Bibliographic status:** Kluger–Bates (2026) and Kluger et al. (2025) are cited as arXiv preprints; journal publication was not established here. Wang et al. has an accepted Statistica Sinica preprint/DOI and a publisher page marked future publication, so avoid assigning an already-issued volume/year from the future issue listing. Battaglia et al. is cited as a preprint. Fong–Tyler, Chen–Lumley, and Zhang et al. have journal records; Egami et al. is a NeurIPS conference paper.
- **Source word-limit ledger:** All web sources used here carried a 200-word summary allowance, including each full-text paper. Treat mirrors/versions as the same source, not additional allowance. This note uses roughly 190 words of findings from Kluger–Bates (equations are referenced rather than reproduced), 130 from Wang's main paper, 180 from its supplement, 90 from Chen–Lumley, 130 from Kluger et al. (2025), 75 from Fong–Tyler, 65 from DSL, 50 each from Battaglia/Zhang, and 90 from the error-prone-exposure paper. No extended quotations are used. The explanatory algebra and proposed research tests are our own inferences. For synthesis, keep each paper's total derived prose under 200 words across the final report and final answer rather than repeating these blocks verbatim in several places.

## What exactly is the mathematical technology, and which errors matter?

### Takeaway
There are several distinct objects: a local change to a fitted estimator, the gold-minus-proxy estimator influence contrast, and the residual full-data efficient influence function after optimal use of cheap data. Their sampling rules are related but not interchangeable.

### Cited Findings
- Kluger–Bates equation (15) uses **different gold and proxy Hessians and parameter limits** in its influence contrast. This is not generally the same as replacing a covariate in one fixed fitted score. — [Equation (15)](https://arxiv.org/html/2602.16933v1).
- **Precise tuning qualification:** Equations (15)–(17) concern Multiwave Predict-Then-Debias with identity tuning matrix and fixed wave weights. Section 4.3 separately optimizes the tuning matrix for a given design; these formulas do not jointly optimize allocation and matrix tuning. — [Sections 4.1 and 4.3](https://arxiv.org/html/2602.16933v1).
- Wang et al. use the conditional mean and variance of the full-data efficient influence function; Chen–Lumley use residual influences appropriate to generalized raking. — [Lemma 1](https://www3.stat.sinica.edu.tw/preprint/SS-2024-0359_Preprint.pdf); [Section 4.3](https://pmc.ncbi.nlm.nih.gov/articles/PMC8918008/).

### Inferences
The following derivation is ours, to clarify the relationship, not a novelty claim.

Let V contain all cheap data and proxies, U the expensive true variable, R indicate verification, and pi(V) be its probability. Let phi(V,U) be a scalar full-data influence function and a(V) a fixed augmentation. The augmented influence contribution is

`h = a(V) + R/pi(V) * [phi(V,U) - a(V)]`.

Write mu(V)=E[phi|V] and sigma²(V)=Var(phi|V). Conditional on V, E[h|V]=mu(V), and

`Var(h) = Var(mu(V)) + E[sigma²(V)/pi(V) + (1/pi(V)-1)*(mu(V)-a(V))²]`.

For fixed a, the pi-dependent objective is therefore the conditional **second moment**

`E[(phi-a)²|V] = sigma²(V)+(mu(V)-a(V))²`.

With equal measurement costs and a fixed expected budget, the uncapped optimal probability is proportional to its square root. With optimal a=mu, this reduces to the conditional **standard deviation** of phi. With per-record cost c(V), the square-root rule also divides by sqrt(c(V)). Bounding probabilities and dependence across sampled observations require the corresponding constrained design, not unrestricted normalization.

For a fixed score approximation to a regression, a single replacement is approximately

`delta theta_i ≈ -A^{-1}[s(z_i_gold,theta)-s(z_i_proxy,theta)]/N`.

But when a covariate changes, its score contribution AND the design/Jacobian can change. A finite batch of replacements, refitting nuisance functions, poor overlap, or nonsmooth model selection can invalidate this first-order shortcut. Exact refits are a natural numerical benchmark for such a diagnostic; they do not by themselves constitute new theory.

This distinction also separates **signed contribution to current bias** from **expected benefit of collecting a label**. A positive and negative pair can cancel in aggregate bias while creating large sampling variance. A large predictable correction may be cheaply handled by augmentation; unresolved conditional variability determines the efficient sampling allocation.

More fundamentally, estimating the displacement between a proxy-only parameter and its gold-data counterpart is not identical to minimizing variance of a valid corrected estimator. Squared influence differences guide the latter; by themselves they neither identify the signed displacement nor reveal unknown errors on unverified records. A randomized pilot, historical validation, or defensible assumptions supplies information about those missing errors. A new diagnostic should state which of these targets it estimates.

### Gaps
- The source papers' local regularity, rate, and positivity assumptions do real work; raw classifier confidence is not a substitute for the relevant conditional influence moments.
- Neither a single replacement derivative nor the IF evaluated at proxy nuisance fits automatically estimates the discrepancy from a gold-data causal analysis. Need to specify which population/fitted functional is being perturbed and control remainder terms.
- Asymptotic variance-optimal design does not establish optimal finite-budget sign-error, tail, or stopping risk. Those would require separate comparisons and guarantees; novelty remains unverified.

## What additional care is needed for causal interpretations and implementation?

### Takeaway
Recovering the result of an analysis using true measurements is different from identifying a causal effect. A causal implementation must handle both verification missingness and the causal nuisance functions, with assumptions and validation appropriate to each.

### Cited Findings
- Wang et al.'s causal example assumes unconfoundedness conditional on both cheap and expensive confounders and uses a full-data ATE score involving propensity and outcome regressions. Their regularity conditions require positive verification probability and finite second moments. Their supplement does not claim a complete crossfitted-DML theorem for reused adaptive pilot data. — [Supplement, Example S3, S2.1, and S3](https://www3.stat.sinica.edu.tw/preprint/supp/2024-0359_supp.pdf).
- **Flexible and Efficient Estimation of Causal Effects with Error-Prone Exposures: A Control Variates Approach for Measurement Error** (available by cutoff) treats verified binary treatment/exposure in an internal subsample. Identification requires SUTVA, treatment positivity, and unconfoundedness, plus random verification or covariate-dependent verification and its positivity. Extra assumptions make proxy-exposure control variates useful; their estimand based on the proxy exposure is explicitly not necessarily causal. Nuisance models in simulations use Super Learner. This is direct error-prone-exposure estimation precedent, not an optimal sampling design paper. — [Sections 2–3](https://pmc.ncbi.nlm.nih.gov/articles/PMC12882774/).

### Inferences
- A credible extension should state whether ML predicts (i) an actual missing confounder/treatment/outcome or (ii) nuisance functions such as propensity and outcome regressions. These are different error channels. Orthogonality to nuisance estimation does not automatically protect against measuring a confounder incorrectly.
- For an ATE score, replacing a confounder changes propensity denominators, fitted outcome means, and potentially nuisance training. A fixed-nuisance score swap is a first-order diagnostic under an explicit target; a full-refit swap answers a different finite-sample question.
- Crossfitting can help control overfitting when learning nuisance functions and influence-moment models, but is not a universal requirement if another independence/empirical-process argument applies. Adaptive verification dependencies must be handled in the theory rather than silently treating the selected records as IID.
- Potential practical agenda: benchmark influence-based error rankings against gold-data refits, compare standard confidence-based labeling with existing target-dependent designs, and evaluate attainable tail/sample-size decisions under realistic cluster concentration. These are proposed tests, not established novelty.

### Gaps
- This research did not establish that broad nonparametric causal nuisance estimation plus exact fixed-size cluster verification plus finite-budget tail guarantees is already solved as a single package. It also did not establish that this conjunction is new.
- A fully verified pilot supplies empirical calibration; transport to the intended unlabeled population and the credibility of the “gold” measurement remain substantive assumptions.

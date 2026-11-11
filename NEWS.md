# MultinomFourier 1.1.0

## New features and bug fixes

* **Log-scale assembly for extreme p-values (< 5e-324)**: All Fourier series terms are now computed relative to a common magnitude scale $\exp(\text{log\_scale})$. The p-value is assembled in log scale via signed log-sum-exp, avoiding underflow of the series and ensuring finite, accurate `log_pval` even when `pval` underflows double precision (< 5e-324).
* **Defensive guard `status = 5`**: Added status code `5` ("First term of the series underflowed; no p-value") when the first term cannot be represented.
* **Correction for tied observation multiplicity at $S_0$**: The boundary atom at $S_0$ now accounts for the multiplicity of permutations of $x$ tied with equal null probabilities (`log_ties`), fixing an under-estimation of $\log(p)$ on symmetric boundary vertices.
* **Extrapolation default**: `extrapolate = 3` (deep median extrapolation) is now the default extrapolation method.

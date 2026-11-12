# MultinomFourier 1.1.0

## New features

* **Log-scale assembly for extreme p-values:** Fourier series terms are now computed relative to a common magnitude scale, and p-values are assembled using signed log-sum-exp. This avoids underflow during series accumulation and preserves finite, accurate `log_pval` values even when the corresponding `pval` is smaller than the minimum representable positive `double` (\(\approx 5 \times 10^{-324}\)).

* **Default extrapolation method:** Changed the default value of `extrapolate` to `3` (deep median extrapolation).

## Bug fixes

* **Underflow in the deep tail of exhaustive methods:** Fixed numerical inaccuracies in `pval_exact()` when computing extremely small p-values. The exhaustive bounds method now handles small binomial tail probabilities in log scale, avoiding underflow in the tail tables. Additionally, the log-probability shift is now adapted to the probability of the observed configuration through `exhaustive_log_shift()`, ensuring that the accumulated probability mass remains within the representable numerical range. This fixes previously underestimated p-values and incorrect `-Inf` results in the deep tail. For p-values above approximately \(10^{-304}\), the shift remains zero and the previous calculation is preserved.

* **Numerical stability checks in exhaustive methods:** Added a defensive check (`status = 4`) when the accumulated probability mass is non-finite or falls below the expected numerical threshold. In such cases, `pval_exact()` now returns `NA` for the p-value and its logarithm, with `converged = FALSE`, rather than reporting an unreliable result.

* **First-term underflow detection:** Added a defensive status code (`status = 5`) when the first term of the Fourier series underflows and no p-value can be computed.

* **Correction for tied observation multiplicity at \(S_0\):** The boundary atom at \(S_0\) now accounts for the multiplicity of permutations of the observed configuration with equal null probabilities (`log_ties`), correcting the underestimation of `log_pval` at symmetric boundary vertices.

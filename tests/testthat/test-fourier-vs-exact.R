test_that("Agreement between Fourier and Exact methods on benchmark instances", {
  # Benchmark instance from package documentation: K = 4, N = 200
  p <- c(0.1, 0.2, 0.3, 0.4)
  x <- c(35, 30, 60, 75)

  # 1. Log-likelihood ratio
  fit_fo_llr <- pval_fourier(x, p, stat = "llr")
  fit_ex_llr <- pval_exact(x, p, stat = "llr")
  expect_equal(fit_fo_llr$status, 0L)
  expect_equal(fit_ex_llr$status, 0L)
  # Documentation reports Fourier: 0.00688, Exact: 0.00684
  expect_equal(fit_fo_llr$pval, 0.00688, tolerance = 1e-3)
  expect_equal(fit_ex_llr$pval, 0.00684, tolerance = 1e-3)
  expect_lt(abs(fit_fo_llr$pval - fit_ex_llr$pval), 1e-4)

  # 2. Probability Mass Function (exact multinomial test)
  fit_fo_pmf <- pval_fourier(x, p, stat = "pmf")
  fit_ex_pmf <- pval_exact(x, p, stat = "pmf")
  expect_equal(fit_fo_pmf$status, 0L)
  expect_equal(fit_ex_pmf$status, 0L)
  # Documentation reports Fourier: 0.00577, Exact: 0.00578
  expect_equal(fit_fo_pmf$pval, 0.00577, tolerance = 1e-3)
  expect_equal(fit_ex_pmf$pval, 0.00578, tolerance = 1e-3)
  expect_lt(abs(fit_fo_pmf$pval - fit_ex_pmf$pval), 1e-4)

  # 3. Small instance: both converge without error; pval_flexible routes to exact
  x_small <- c(3, 2, 4)
  p_small <- c(0.3, 0.3, 0.4)
  fit_small_fo <- pval_fourier(x_small, p_small, stat = "llr")
  fit_small_ex <- pval_exact(x_small, p_small, stat = "llr")
  fit_small_fl <- pval_flexible(x_small, p_small, stat = "llr")
  expect_equal(fit_small_fo$status, 0L)
  expect_equal(fit_small_ex$status, 0L)
  expect_equal(fit_small_fl$method, "exact")
  expect_equal(fit_small_fl$pval, fit_small_ex$pval)
})

test_that("exact_terms: Poissonized Cauchy vs Polynomial Convolution", {
  x <- c(15, 20, 25)
  p <- c(0.2, 0.3, 0.5)

  fit_poisson <- pval_fourier(x, p, exact_terms = FALSE)
  fit_exact_t <- pval_fourier(x, p, exact_terms = TRUE)

  expect_equal(fit_poisson$pval, fit_exact_t$pval, tolerance = 1e-3)
  expect_true(fit_poisson$Poissonized)
  expect_false(fit_exact_t$Poissonized)
  expect_equal(fit_poisson$poissonized, TRUE)
  expect_equal(fit_exact_t$poissonized, FALSE)

  # Q and D_Q diagnostics are populated for Poisson, NA for polynomial convolution
  expect_false(is.na(fit_poisson$Q))
  expect_false(is.na(fit_poisson$D_Q))
  expect_true(is.na(fit_exact_t$Q))
  expect_true(is.na(fit_exact_t$D_Q))
})

test_that("pval_flexible automatic dispatch based on enum_cutoff", {
  x_small <- c(3, 2, 4)
  p_small <- c(0.3, 0.3, 0.4)

  # Small instance defaults to exact
  fit_flex_small <- pval_flexible(x_small, p_small)
  expect_equal(fit_flex_small$method, "exact")

  # Large instance defaults to fourier
  x_large <- c(100, 150, 200, 120, 180, 250)
  p_large <- rep(1/6, 6)
  fit_flex_large <- pval_flexible(x_large, p_large)
  expect_equal(fit_flex_large$method, "fourier")

  # Manual override of enum_cutoff
  fit_cutoff_fourier <- pval_flexible(x_small, p_small, enum_cutoff = 0)
  expect_equal(fit_cutoff_fourier$method, "fourier")

  fit_cutoff_exact <- pval_flexible(x_small, p_small, enum_cutoff = 50)
  expect_equal(fit_cutoff_exact$method, "exact")
})

test_that("Multi-threading yields identical results", {
  x <- c(12, 18, 25, 15)
  p <- c(0.2, 0.3, 0.3, 0.2)

  fit_1th <- pval_fourier(x, p, n_threads = 1)
  fit_2th <- pval_fourier(x, p, n_threads = 2)

  # The stopping point, the statistic and the tilt are thread-independent.
  expect_equal(fit_1th$stat_value, fit_2th$stat_value, tolerance = 1e-8)
  expect_equal(fit_1th$n_terms, fit_2th$n_terms)
  expect_equal(fit_1th$gamma, fit_2th$gamma, tolerance = 1e-8)

  # With extrapolate = 3 (the default) the reported value is a median of the deep
  # extrapolation table over the final window. That table holds one entry per chunk
  # of terms, and the chunk size is proportional to n_threads, so the entries being
  # averaged are not the same ones. The two values therefore agree only to within
  # the extrapolation error, not to round-off.
  expect_equal(fit_1th$pval, fit_2th$pval, tolerance = 1e-3)

  # With extrapolate = 2 the reported value is thread-independent to round-off.
  fit_1th_e2 <- pval_fourier(x, p, n_threads = 1, extrapolate = 2)
  fit_2th_e2 <- pval_fourier(x, p, n_threads = 2, extrapolate = 2)
  expect_equal(fit_1th_e2$pval, fit_2th_e2$pval, tolerance = 1e-8)
})

test_that("Statistic calculation and aliases (LLR, Chi2, PD, PMF)", {
  x <- c(10, 20, 30)
  p <- c(0.2, 0.3, 0.5)
  N <- sum(x)
  E <- N * p

  # 1. Log-likelihood ratio
  fit_llr <- pval_fourier(x, p, stat = "llr")
  expected_llr <- 2 * sum(ifelse(x > 0, x * log(x / E), 0))
  expect_equal(fit_llr$stat_value, expected_llr, tolerance = 1e-6)
  expect_equal(fit_llr$S0, fit_llr$stat_value)
  expect_equal(fit_llr[["S0"]], fit_llr$stat_value)
  expect_equal(fit_llr$stat, "Log-likelihood ratio")

  # 2. Pearson Chi-Square
  fit_chi2 <- pval_fourier(x, p, stat = "chi2")
  expected_chi2 <- sum((x - E)^2 / E)
  expect_equal(fit_chi2$stat_value, expected_chi2, tolerance = 1e-6)
  expect_equal(fit_chi2$S0, fit_chi2$stat_value)
  expect_equal(fit_chi2[["S0"]], fit_chi2$stat_value)
  expect_equal(fit_chi2$stat, "Pearson's Chi-Square")

  # 3. Power divergence (lambda = 2/3)
  lambda <- 2/3
  fit_pd <- pval_fourier(x, p, stat = "pd", lambda = lambda)
  expected_pd <- (2 / (lambda * (lambda + 1))) * sum(x * ((x / E)^lambda - 1))
  expect_equal(fit_pd$stat_value, expected_pd, tolerance = 1e-6)
  expect_equal(fit_pd$S0, fit_pd$stat_value)
  expect_equal(fit_pd[["S0"]], fit_pd$stat_value)
  expect_true(grepl("power-divergence", fit_pd$stat))

  # 4. Probability Mass Function (exact multinomial test)
  fit_pmf <- pval_fourier(x, p, stat = "pmf")
  expected_pmf <- dmultinom(x, prob = p, log = TRUE)
  expect_equal(fit_pmf$stat_value, expected_pmf, tolerance = 1e-6)
  expect_equal(fit_pmf$S0, fit_pmf$stat_value)
  expect_equal(fit_pmf[["S0"]], fit_pmf$stat_value)
  expect_equal(fit_pmf$stat, "Probability Mass Function")
})

test_that("stat_value matches between Fourier and Exact methods", {
  x_small <- c(3, 2, 4)
  p_small <- c(0.3, 0.3, 0.4)

  for (st in c("llr", "chi2", "pmf")) {
    fit_fo <- pval_fourier(x_small, p_small, stat = st)
    fit_ex <- pval_exact(x_small, p_small, stat = st)
    expect_equal(fit_fo$stat_value, fit_ex$stat_value, tolerance = 1e-8)
    expect_equal(fit_ex$S0, fit_ex$stat_value)
  }

  fit_fo_pd <- pval_fourier(x_small, p_small, stat = "pd", lambda = 2/3)
  fit_ex_pd <- pval_exact(x_small, p_small, stat = "pd", lambda = 2/3)
  expect_equal(fit_fo_pd$stat_value, fit_ex_pd$stat_value, tolerance = 1e-8)
  expect_equal(fit_ex_pd$S0, fit_ex_pd$stat_value)
})

test_that("Statistic calculation with zero counts (0 * log(0) = 0)", {
  x_zero <- c(10, 0, 20)
  p_zero <- c(0.3, 0.3, 0.4)
  E_zero <- sum(x_zero) * p_zero

  fit_llr_zero <- pval_fourier(x_zero, p_zero, stat = "llr")
  expected_llr_zero <- 2 * (10 * log(10 / E_zero[1]) + 20 * log(20 / E_zero[3]))
  expect_equal(fit_llr_zero$stat_value, expected_llr_zero, tolerance = 1e-6)
  expect_false(is.nan(fit_llr_zero$stat_value))
})


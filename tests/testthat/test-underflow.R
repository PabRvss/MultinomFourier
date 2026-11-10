test_that("Exact method underflow (< 5e-324) returns NA for pval and exact log_pval", {
  # 3 vertices of simplex: (750, 0, 0), (0, 750, 0), (0, 0, 750)
  # Theoretical p-value: 3 * (1/3)^750 = 3^(-749) ~ 10^(-357)
  # Theoretical log(p-value): -749 * log(3)
  x_tiny <- c(750, 0, 0)
  p_tiny <- rep(1/3, 3)

  expect_warning(
    fit_tiny_ex <- pval_exact(x_tiny, p_tiny),
    "The p-value is too small to represent as a positive double"
  )

  expect_true(is.na(fit_tiny_ex$pval))
  expect_true(!is.na(fit_tiny_ex$log_pval))
  expect_true(is.finite(fit_tiny_ex$log_pval))

  expected_log_pval <- -749 * log(3)
  expect_equal(fit_tiny_ex$log_pval, expected_log_pval, tolerance = 1e-4)

  # Check all accessor aliases
  expect_equal(fit_tiny_ex$`log(pval)`, expected_log_pval, tolerance = 1e-4)
  expect_equal(fit_tiny_ex$`log(p-value)`, expected_log_pval, tolerance = 1e-4)
  expect_equal(fit_tiny_ex$`log.pval`, expected_log_pval, tolerance = 1e-4)
  expect_equal(fit_tiny_ex[["log(pval)"]], expected_log_pval, tolerance = 1e-4)
  expect_equal(fit_tiny_ex[["log(p-value)"]], expected_log_pval, tolerance = 1e-4)
})

test_that("Fourier method underflow (< 5e-324) returns NA for pval and a finite log_pval", {
  # Extreme instance: N = 1000, 4 categories, observation at a vertex.
  # True p-value: the 4 vertices, 4 * (1/4)^1000 = 4^(-999), below 5e-324.
  # The series used to stop at the first term here and report log(p0) = -1000*log(4).
  # It now sums the series (thousands of terms). The estimate is still about 0.5 low in
  # log scale because of the tie at the observation (4 outcomes share the statistic and
  # only p0/2 is added), the same as for N = 50, which is in the double range.
  x_tiny <- c(1000, 0, 0, 0)
  p_tiny <- rep(0.25, 4)

  expect_warning(
    fit_tiny_fo <- pval_fourier(x_tiny, p_tiny),
    "The p-value is too small to represent as a positive double"
  )

  expect_true(is.na(fit_tiny_fo$pval))
  expect_true(!is.na(fit_tiny_fo$log_pval))
  expect_true(is.finite(fit_tiny_fo$log_pval))
  expect_gt(fit_tiny_fo$n_terms, 1)

  expected_log_pval <- -999 * log(4)
  expect_equal(fit_tiny_fo$log_pval, expected_log_pval, tolerance = 1e-3)

  expect_equal(fit_tiny_fo$`log(pval)`, fit_tiny_fo$log_pval)
  expect_equal(fit_tiny_fo$`log(p-value)`, fit_tiny_fo$log_pval)
})

test_that("Fourier log_pval below 5e-324 comes from the series, not from log(p0)", {
  # K = 5, N = 1300, p proportional to k^2.5, uniform observation, llr: p ~ 1e-392.
  # Before the log-scale assembly this returned log_pval = log(p0) = -922.2 with status 0
  # and 1 term, 8.5 orders of magnitude too small.
  p <- (1:5)^2.5; p <- p / sum(p)
  x <- rep(260, 5)
  fit <- suppressWarnings(pval_fourier(x, p, rel_eps = 1e-3))
  expect_equal(fit$status, 0L)
  expect_gt(fit$n_terms, 1)
  expect_true(is.na(fit$pval))
  expect_equal(fit$log_pval, -902.668, tolerance = 1e-5)
})

test_that("Fourier log_pval agrees with log(pval) in the double range", {
  p <- (1:5)^2.5; p <- p / sum(p)
  fit <- pval_fourier(rep(60, 5), p, rel_eps = 1e-3)
  expect_equal(fit$status, 0L)
  expect_equal(fit$log_pval, log(fit$pval), tolerance = 1e-12)
  fit_ex <- pval_exact(rep(60, 5), p)
  expect_equal(fit$pval, fit_ex$pval, tolerance = 1e-3)
})

test_that("Flexible method handles underflow consistently", {
  x_tiny <- c(750, 0, 0)
  p_tiny <- rep(1/3, 3)

  expect_warning(
    fit_tiny_fl <- pval_flexible(x_tiny, p_tiny),
    "The p-value is too small to represent as a positive double"
  )

  expect_true(is.na(fit_tiny_fl$pval))
  expect_true(is.finite(fit_tiny_fl$log_pval))
})

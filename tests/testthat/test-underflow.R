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

test_that("Fourier method underflow (< 5e-324) returns NA for pval and exact log_pval", {
  # Extreme instance: N = 1000, 4 categories
  # Underflows in standard double precision (< 5e-324).
  # Preserves exact log_pval from observed log_p0: ln((1/4)^1000) = -1000 * ln(4)
  x_tiny <- c(1000, 0, 0, 0)
  p_tiny <- rep(0.25, 4)

  expect_warning(
    fit_tiny_fo <- pval_fourier(x_tiny, p_tiny),
    "The p-value is too small to represent as a positive double"
  )

  expect_true(is.na(fit_tiny_fo$pval))
  expect_true(!is.na(fit_tiny_fo$log_pval))
  expect_true(is.finite(fit_tiny_fo$log_pval))

  expected_log_p0 <- -1000 * log(4)
  expect_equal(fit_tiny_fo$log_pval, expected_log_p0, tolerance = 1e-3)

  expect_equal(fit_tiny_fo$`log(pval)`, fit_tiny_fo$log_pval)
  expect_equal(fit_tiny_fo$`log(p-value)`, fit_tiny_fo$log_pval)
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

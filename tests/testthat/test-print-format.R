test_that("print.multinomfourier outputs exact expected layout and field names", {
  x <- c(10, 20, 30)
  p <- c(0.2, 0.3, 0.5)

  # Fourier method
  fit_fo <- pval_fourier(x, p, stat = "llr")
  out_fo <- capture.output(print(fit_fo))
  out_fo_str <- paste(out_fo, collapse = "\n")

  expect_true(grepl("Multinomial Goodness-of-Fit Test \\(MultinomFourier\\)", out_fo_str))
  expect_true(grepl("------------------------------------------------", out_fo_str))
  expect_true(grepl("Statistic       : Log-likelihood ratio", out_fo_str))
  expect_true(grepl("Statistic value :", out_fo_str))
  expect_true(grepl("p-value         :", out_fo_str))
  expect_true(grepl("log\\(p-value\\)    :", out_fo_str))
  expect_true(grepl("Computation", out_fo_str))
  expect_true(grepl("Method          : Fourier", out_fo_str))
  expect_true(grepl("Poissonized     : Yes", out_fo_str))
  expect_true(grepl("Status          : Converged", out_fo_str))
  expect_true(grepl("Gamma           :", out_fo_str))
  expect_true(grepl("W               :", out_fo_str))
  expect_true(grepl("Terms           :", out_fo_str))
  expect_true(grepl("Gamma time      :.* s", out_fo_str))
  expect_true(grepl("Total time      :.* s", out_fo_str))

  # Fourier exact_terms = TRUE
  fit_fo_conv <- pval_fourier(x, p, exact_terms = TRUE)
  out_conv_str <- paste(capture.output(print(fit_fo_conv)), collapse = "\n")
  expect_true(grepl("Poissonized     : No", out_conv_str))

  # Exact method
  fit_ex <- pval_exact(c(3, 2, 4), c(0.3, 0.3, 0.4))
  out_ex_str <- paste(capture.output(print(fit_ex)), collapse = "\n")
  expect_true(grepl("Method          : Exact", out_ex_str))
  # Exact method does not have Poissonized, Gamma, W, Terms, Gamma time
  expect_false(grepl("Poissonized", out_ex_str))
  expect_false(grepl("Gamma time", out_ex_str))
})

test_that("print.multinomfourier displays warning note on underflow", {
  x_tiny <- c(750, 0, 0)
  p_tiny <- rep(1/3, 3)

  suppressWarnings({
    fit_tiny <- pval_exact(x_tiny, p_tiny)
  })

  out_tiny <- capture.output(print(fit_tiny))
  out_tiny_str <- paste(out_tiny, collapse = "\n")

  expect_true(grepl("p-value         : NA", out_tiny_str))
  expect_true(grepl("log\\(p-value\\)    :", out_tiny_str))
  expect_true(grepl("The p-value is too small to represent as a positive double", out_tiny_str))
  expect_true(grepl("\\$pval was set to NA; use \\$log_pval", out_tiny_str))
})


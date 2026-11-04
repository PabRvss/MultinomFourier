test_that("Boundary dimensions: minimal K = 2 and small N", {
  # K = 2, N = 1: Exact calculation is 1.0 (both states have identical statistic)
  x_min <- c(1, 0)
  p_min <- c(0.5, 0.5)

  fit_fo_min <- pval_fourier(x_min, p_min)
  fit_ex_min <- pval_exact(x_min, p_min)

  expect_equal(fit_fo_min$status, 0L)
  expect_equal(fit_ex_min$status, 0L)
  expect_equal(fit_ex_min$pval, 1.0)
  expect_gt(fit_fo_min$pval, 0.0)
  expect_lte(fit_fo_min$pval, 1.0)

  # K = 2, binomial N = 200 (both methods agree closely in absolute terms)
  x_bin <- c(120, 80)
  p_bin <- c(0.5, 0.5)
  fit_fo_bin <- pval_fourier(x_bin, p_bin)
  fit_ex_bin <- pval_exact(x_bin, p_bin)
  expect_lt(abs(fit_fo_bin$pval - fit_ex_bin$pval), 1e-3)
})

test_that("Sparse count vectors with zeros", {
  x_sparse <- c(0, 0, 15)
  p_sparse <- c(0.2, 0.3, 0.5)

  fit_fo_sparse <- pval_fourier(x_sparse, p_sparse)
  fit_ex_sparse <- pval_exact(x_sparse, p_sparse)

  expect_false(is.nan(fit_fo_sparse$pval))
  expect_false(is.nan(fit_ex_sparse$pval))
  expect_equal(fit_fo_sparse$pval, fit_ex_sparse$pval, tolerance = 1e-2)
})

test_that("Fourier stopping rules: max_terms limit (status = 2)", {
  x <- c(10, 20, 30)
  p <- c(0.2, 0.3, 0.5)

  fit_terms <- pval_fourier(x, p, max_terms = 5, rel_eps = 1e-15)
  expect_equal(fit_terms$status, 2L)
  expect_equal(fit_terms$message, "Maximum number of terms reached")
  expect_lte(fit_terms$n_terms, 5L)
})

test_that("Fourier stopping rules: max_time limit (status = 1)", {
  x <- c(200, 300, 500)
  p <- c(0.2, 0.3, 0.5)

  fit_time <- pval_fourier(x, p, max_time = 0.0001, max_terms = 1e7)
  expect_equal(fit_time$status, 1L)
  expect_equal(fit_time$message, "Maximum time reached")
})

test_that("Exact method has NA in Fourier-specific components", {
  x <- c(3, 2, 4)
  p <- c(0.3, 0.3, 0.4)
  fit_ex <- pval_exact(x, p)

  expect_true(is.na(fit_ex$gamma))
  expect_true(is.na(fit_ex$time_gamma))
  expect_true(is.na(fit_ex$W))
  expect_true(is.na(fit_ex$n_terms))
  expect_true(is.na(fit_ex$Q))
  expect_true(is.na(fit_ex$D_Q))
  expect_true(is.na(fit_ex$Poissonized))
})

test_that("Vignette election test case (Escuela Centenario table 276 vs 278)", {
  votes <- matrix(
    c(
       96, 101, 173,  94, 106,  88,  96, 104,  88,  98, 109,  92, 104, 57, 50,
       37,  34,  31,  34,  37,  37,  35,  32,  40,  37,  37,  38,  36, 22, 25,
       39,  37,  36,  36,  26,  29,  28,  39,  39,  34,  28,  32,  35, 15, 27,
       99, 107, 110, 108, 127, 107, 108, 125,  89, 100, 109, 103,  98, 74, 70,
       46,  37,  42,  41,  34,  37,  45,  26,  62,  45,  48,  49,  41, 21, 21,
       83,  84,   8,  87,  70, 102,  88,  74,  82,  86,  69,  86,  86, 49, 44
    ),
    nrow = 6, byrow = TRUE,
    dimnames = list(
      c("Abstention", "Candidate A", "Candidate B", "Candidate C", "Null/blank", "Candidate D"),
      as.character(276:290)
    )
  )

  loo_p <- sapply(seq_len(ncol(votes)), function(m) {
    other <- rowSums(votes[, -m, drop = FALSE])
    other / sum(other)
  })
  colnames(loo_p) <- colnames(votes)

  # Mesa 278 (anomaly, highly significant)
  fit_278 <- pval_fourier(votes[, "278"], loo_p[, "278"], stat = "llr")
  expect_equal(fit_278$status, 0L)
  expect_lt(fit_278$pval, 1e-25)

  # Mesa 276 (normal)
  fit_276 <- pval_fourier(votes[, "276"], loo_p[, "276"], stat = "llr")
  expect_equal(fit_276$status, 0L)
  expect_gt(fit_276$pval, 0.5)
})

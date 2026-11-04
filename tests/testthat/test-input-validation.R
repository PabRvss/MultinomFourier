test_that("Input validation for counts vector x", {
  p_valid <- c(0.3, 0.3, 0.4)
  
  expect_error(pval_fourier("not_a_vector", p_valid), "'x' must be a numeric vector")
  expect_error(pval_fourier(numeric(0), numeric(0)), "'x' must be a numeric vector")
  expect_error(pval_fourier(c(10, NA, 20), p_valid), "'x' must not contain NA")
  expect_error(pval_fourier(c(10, Inf, 20), p_valid), "'x' must not contain NA")
  expect_error(pval_fourier(c(10, -1, 20), p_valid), "'x' must contain non-negative counts")
  expect_error(pval_fourier(c(10.5, 10, 20), p_valid), "'x' must contain integer counts")

  # Exact and flexible should enforce identical validations
  expect_error(pval_exact(c(10, -5, 20), p_valid), "'x' must contain non-negative counts")
  expect_error(pval_flexible(c(10.2, 5, 20), p_valid), "'x' must contain integer counts")
})

test_that("Input validation for probability vector p", {
  x_valid <- c(10, 20, 30)

  expect_error(pval_fourier(x_valid, "invalid"), "'p' must be a numeric vector")
  expect_error(pval_fourier(x_valid, c(0.5, 0.5)), "'x' and 'p' must have the same length")
  expect_error(pval_fourier(x_valid, c(0.3, NA, 0.7)), "'p' must not contain NA")
  expect_error(pval_fourier(x_valid, c(0.5, 0.6, -0.1)), "'p' must contain strictly positive probabilities")
  expect_error(pval_fourier(x_valid, c(0.5, 0.5, 0)), "'p' must contain strictly positive probabilities")

  # Probabilities not summing to 1
  expect_error(pval_fourier(x_valid, c(0.2, 0.2, 0.2)), "'p' must sum to 1")
  expect_error(pval_exact(x_valid, c(1, 2, 3)), "'p' must sum to 1")
  expect_error(pval_flexible(x_valid, c(1, 2, 3)), "'p' must sum to 1")

  # rescale_p validation and functionality
  expect_error(pval_fourier(x_valid, c(1, 2, 3), rescale_p = "yes"), "'rescale_p' must be TRUE or FALSE")
  fit_resc <- pval_fourier(x_valid, c(1, 2, 3), rescale_p = TRUE)
  expect_equal(fit_resc$p, c(1/6, 2/6, 3/6))
})

test_that("Statistic and lambda validation", {
  x <- c(5, 10, 15)
  p <- c(0.2, 0.3, 0.5)

  # Invalid stat
  expect_error(pval_fourier(x, p, stat = "unknown"))
  expect_error(pval_exact(x, p, stat = "unknown"))
  expect_error(pval_flexible(x, p, stat = "unknown"))

  # Warning when lambda provided for stat != 'pd'
  expect_warning(pval_fourier(x, p, stat = "llr", lambda = 1.5), "Parameter lambda is ignored")
  expect_warning(pval_exact(x, p, stat = "chi2", lambda = 1.5), "Parameter lambda is ignored")

  # lambda required for 'pd'
  expect_error(pval_fourier(x, p, stat = "pd"), "'lambda' must be given")
  expect_error(pval_exact(x, p, stat = "pd"), "'lambda' must be given")
  expect_error(pval_flexible(x, p, stat = "pd"), "'lambda' must be given")

  # lambda bounds (strictly > -1)
  expect_error(pval_fourier(x, p, stat = "pd", lambda = -1), "strictly greater than -1")
  expect_error(pval_fourier(x, p, stat = "pd", lambda = -2.5), "strictly greater than -1")
  expect_error(pval_exact(x, p, stat = "pd", lambda = -1), "strictly greater than -1")
  expect_error(pval_exact(x, p, stat = "pd", lambda = -1.2), "strictly greater than -1")
  expect_error(pval_flexible(x, p, stat = "pd", lambda = -1), "strictly greater than -1")
  expect_error(pval_flexible(x, p, stat = "pd", lambda = -1.5), "strictly greater than -1")
  expect_error(pval_fourier(x, p, stat = "pd", lambda = c(0.5, 0.5)), "single finite real number")
  expect_error(pval_fourier(x, p, stat = "pd", lambda = Inf), "single finite real number")
})

test_that("Numerical parameter boundaries and types", {
  x <- c(5, 10, 15)
  p <- c(0.2, 0.3, 0.5)

  expect_error(pval_fourier(x, p, undersampling = 0), "'undersampling' must be strictly positive")
  expect_error(pval_fourier(x, p, undersampling = -1), "'undersampling' must be strictly positive")
  expect_error(pval_fourier(x, p, rel_eps = -0.01), "'rel_eps' must be non-negative")
  expect_error(pval_fourier(x, p, B = -1), "'B' must be an integer greater than or equal to 0")
  expect_error(pval_fourier(x, p, max_terms = 0), "'max_terms' must be a positive integer")
  expect_error(pval_fourier(x, p, max_terms = -10), "'max_terms' must be a positive integer")
  expect_error(pval_fourier(x, p, exact_terms = "FALSE"), "'exact_terms' must be TRUE or FALSE")
  expect_error(pval_fourier(x, p, n_threads = 0), "'n_threads' must be a positive integer")
  expect_error(pval_fourier(x, p, n_threads = -2), "'n_threads' must be a positive integer")

  # Warnings on deprecated options
  expect_warning(pval_fourier(x, p, engine = "speedup"), "'engine' has been removed")
  expect_warning(pval_fourier(x, p, speedup = TRUE), "'speedup' has been removed")
})


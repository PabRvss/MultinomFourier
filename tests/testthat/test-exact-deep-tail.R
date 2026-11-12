# Exact below about 1e-500: the binomial tail tables used to underflow (p too small, 6.7 % at
# 1e-598) and the fixed offset exp(700) gave log_pval = -Inf below about 1e-628, both with
# status 0. Reference values: full enumeration in log scale (independent Python script).
deep_instance <- function(N, K, a) {
  p <- (1:K)^a; p <- p / sum(p)
  x <- rep(N %/% K, K); x[K] <- x[K] + N %% K
  list(x = x, p = p)
}

test_that("Exact log_pval matches full enumeration deep in the tail", {
  cases <- list(
    list(N = 200,  K = 5, a = 10, log_p = -1001.118896),  # p ~ 1e-435
    list(N = 300,  K = 4, a = 10, log_p = -1375.790340),  # p ~ 1e-598 (was 6.7 % low)
    list(N = 1000, K = 4, a = 5,  log_p = -1809.529776)   # p ~ 1e-786 (was -Inf)
  )
  for (cs in cases) {
    I <- deep_instance(cs$N, cs$K, cs$a)
    fit <- suppressWarnings(pval_exact(I$x, I$p, stat = "llr"))
    expect_equal(fit$status, 0L)
    expect_true(is.finite(fit$log_pval))
    expect_lt(abs(fit$log_pval - cs$log_p), 1e-5)
  }
})

test_that("Exact never reports a p-value below the probability of the observation", {
  for (cs in list(c(300, 4, 10), c(1000, 4, 5))) {
    I <- deep_instance(cs[1], cs[2], cs[3])
    fit <- suppressWarnings(pval_exact(I$x, I$p, stat = "llr"))
    expect_equal(fit$status, 0L)
    expect_gte(fit$log_pval, dmultinom(I$x, prob = I$p, log = TRUE) - 1e-9)
  }
})

test_that("Exact bounds and bisection variants agree deep in the tail", {
  I <- deep_instance(1000, 4, 10)  # p ~ 1e-1992
  fb <- suppressWarnings(pval_exact(I$x, I$p, stat = "llr"))
  fs <- suppressWarnings(pval_exact(I$x, I$p, stat = "llr", fast_exhaustive = FALSE))
  expect_true(is.finite(fb$log_pval))
  expect_lt(abs(fb$log_pval - fs$log_pval), 1e-8)
})

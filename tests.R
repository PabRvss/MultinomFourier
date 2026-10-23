library(MultinomFourier)

# Datos de prueba
x <- c(10, 20, 30)
p <- c(0.2, 0.3, 0.5)

# 1. Caso base (fourier default: llr)
fit <- pval_fourier(x, p)
print(fit)
stopifnot(inherits(fit, "multinomfourier"))

# 2. Distintos estadísticos
fit_chi2 <- pval_fourier(x, p, stat = "chi2")
fit_llr  <- pval_fourier(x, p, stat = "llr")
fit_pmf  <- pval_fourier(x, p, stat = "pmf")
fit_pd   <- pval_fourier(x, p, stat = "pd", lambda = 2/3)

# 2.1 Warning cuando se suministra lambda para stat != 'pd'
fit_warn <- pval_fourier(x, p, stat = "chi2", lambda = 1.5)

# 3. Límite de términos alcanzado (status = 2)
fit_terms <- pval_fourier(x, p, max_terms = 10, rel_eps = 1e-12)
print(fit_terms)

# 4. Límite de tiempo alcanzado (status = 1)
fit_time <- pval_fourier(x, p, max_time = 0.0001, max_terms = 1e6)
print(fit_time)

# 5. exact_terms: evaluador Poisson vs convolución polinomial exacta
fit_poisson <- pval_fourier(x, p, exact_terms = FALSE)
fit_exact_t <- pval_fourier(x, p, exact_terms = TRUE)
c(poisson = fit_poisson$pval, exact_terms = fit_exact_t$pval)
stopifnot(isTRUE(all.equal(fit_poisson$pval, fit_exact_t$pval, tolerance = 1e-3)))
stopifnot(isTRUE(fit_poisson$Poissonized))
stopifnot(isFALSE(fit_exact_t$Poissonized))
stopifnot(isTRUE(fit_poisson$poissonized))

# 5.1 Verificación de nuevos outputs Q y D_Q en Fourier
stopifnot(!is.null(fit$Q), !is.null(fit$D_Q))
cat(sprintf("Diagnósticos Fourier: Q = %s, D_Q = %s\n", fit$Q, fit$D_Q))

# 6. Variación de undersampling
fit_u05 <- pval_fourier(x, p, undersampling = 0.5)
fit_u15 <- pval_fourier(x, p, undersampling = 1.5)
c(W_05 = fit_u05$W, W_10 = fit$W, W_15 = fit_u15$W) #Valores de W para diferente undersampling

# 7. Ejecución con múltiples hilos
fit_th <- pval_fourier(x, p, n_threads = 4)
print(fit_th)

# 8. Bisección exhaustiva y método flexible
x_small <- c(3, 2, 4)
p_small <- c(0.3, 0.3, 0.4)
fit_ex <- pval_exact(x_small, p_small)
fit_fl <- pval_flexible(x_small, p_small)

# En pval_exact Q y D_Q deben ser NA
stopifnot(is.na(fit_ex$Q), is.na(fit_ex$D_Q), is.na(fit_ex$Poissonized))

# 9. lambda <= -1 no está soportado: debe dar un error claro
fit_bad <- tryCatch(pval_fourier(x, p, stat = "pd", lambda = -1), error = function(e) conditionMessage(e))
print(fit_bad)
stopifnot(grepl("strictly greater than -1", fit_bad))
fit_bad2 <- tryCatch(pval_fourier(x, p, stat = "pd", lambda = -2), error = function(e) conditionMessage(e))
stopifnot(grepl("strictly greater than -1", fit_bad2))
fit_bad_ex <- tryCatch(pval_exact(x, p, stat = "pd", lambda = -1), error = function(e) conditionMessage(e))
stopifnot(grepl("strictly greater than -1", fit_bad_ex))
fit_bad_fl <- tryCatch(pval_flexible(x, p, stat = "pd", lambda = -1), error = function(e) conditionMessage(e))
stopifnot(grepl("strictly greater than -1", fit_bad_fl))

# 9b. p que no suma 1: error por defecto; con rescale_p = TRUE se reescala (como chisq.test)
err_p <- tryCatch(pval_exact(x_small, c(3, 3, 4)), error = function(e) conditionMessage(e))
print(err_p)
stopifnot(grepl("must sum to 1", err_p))
fit_resc <- pval_exact(x_small, c(3, 3, 4), rescale_p = TRUE)
stopifnot(isTRUE(all.equal(fit_resc$pval, fit_ex$pval)), isTRUE(all.equal(fit_resc$p, p_small)))

# 10. Tests de la vignette (Escuela Centenario y leave-one-out)
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

stopifnot(all(abs(colSums(loo_p) - 1) < 1e-12))
stopifnot(all(loo_p >= 0))

fit_vignette_278 <- pval_fourier(votes[, "278"], loo_p[, "278"], stat = "llr")
stopifnot(fit_vignette_278$status == 0L)
stopifnot(fit_vignette_278$pval < 1e-25)

fit_vignette_276 <- pval_fourier(votes[, "276"], loo_p[, "276"], stat = "llr")
stopifnot(fit_vignette_276$status == 0L)
stopifnot(fit_vignette_276$pval > 0.5)
cat("Tests de la vignette pasaron exitosamente.\n")

# 11. Verificación de log_pval y p-valores menores a 5e-324
stopifnot(!is.null(fit$log_pval), !is.null(fit$`log(pval)`))
stopifnot(isTRUE(all.equal(fit$log_pval, log(fit$pval))))
stopifnot(isTRUE(all.equal(fit$`log(pval)`, log(fit$pval))))
stopifnot(isTRUE(all.equal(fit[["log(pval)"]], log(fit$pval))))
stopifnot(isTRUE(all.equal(fit_ex$log_pval, log(fit_ex$pval))))

# 11b. Instancia con p-valor < 5e-324 (debe retornar NA en pval y valor finito exacto en log_pval)
x_tiny <- c(750, 0, 0)
p_tiny <- rep(1/3, 3)
fit_tiny_ex <- pval_exact(x_tiny, p_tiny)
stopifnot(is.na(fit_tiny_ex$pval))
stopifnot(!is.na(fit_tiny_ex$log_pval), is.finite(fit_tiny_ex$log_pval))
# Teóricamente: 3 * (1/3)^750 = 3^(-749) -> ln(3^(-749)) = -749 * ln(3) = -822.8606
stopifnot(abs(fit_tiny_ex$log_pval - (-749 * log(3))) < 1e-4)

# Y lo mismo en Fourier:
fit_tiny_fo <- pval_fourier(c(1000, 0, 0, 0), rep(0.25, 4))
stopifnot(is.na(fit_tiny_fo$pval))
stopifnot(!is.na(fit_tiny_fo$log_pval), is.finite(fit_tiny_fo$log_pval))
cat("Tests de log_pval y underflow pasaron exitosamente.\n")


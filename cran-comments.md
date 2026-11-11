## Release summary
This is an update of MultinomFourier to version 1.1.0.

Key updates in this version:
* Implemented log-scale series assembly via signed log-sum-exp, avoiding underflow of Fourier terms and ensuring finite, accurate log p-values even when p-values underflow double precision (< 5e-324).
* Added status code 5 ("First term of the series underflowed; no p-value").
* Added boundary atom correction accounting for the multiplicity of permutations of x tied with equal null probabilities (`log_ties`).
* Set `extrapolate = 3` (deep median extrapolation) as default.

## Test environments
* local: Windows 11 x86_64, R 4.5.1, Rtools45
* local: Ubuntu 26.04 LTS x86_64 (WSL2), R 4.5.2, GCC 15.2.0
* local: macOS aarch64 (Apple Silicon M4), R 4.2.0, Apple Clang
* R-hub v2 (GitHub Actions):
  - linux (R-devel, Ubuntu container)
  - clang-asan (Linux R-devel, AddressSanitizer)
  - macos-arm64 (macOS R-devel, ARM64)
  - windows (Windows Server, R-devel)

## R CMD check results

0 errors | 0 warnings | 0 notes

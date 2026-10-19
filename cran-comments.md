## Resubmission
This is a resubmission. In this version I have addressed the feedback from the initial CRAN incoming check:

* Excluded the '.github' directory from the package tarball by adding it to '.Rbuildignore' (resolving the NOTE on hidden files and directories).
* Addressed the spelling NOTE for DESCRIPTION:
  - Replaced 'computable' with 'computationally feasible'.
  - Added 'inst/WORDLIST' and noted that 'Cressie' is the proper surname of Noel Cressie (author of the Cressie-Read power divergence family, Cressie & Read, 1984), not a misspelling.

## Test environments
* Windows 11 x86_64, R 4.5.1, Rtools45 (local)
* Ubuntu 26.04 LTS x86_64 (WSL2), R 4.5.2, GCC 15.2.0 (local)
* macOS aarch64 (Apple Silicon M4), R 4.2.0, Apple Clang (local)
* R-hub v2 (GitHub Actions): Linux (ubuntu-latest), macOS (macos-latest / ARM64), Windows (windows-latest)

## R CMD check results

0 errors | 0 warnings | 1 note

* This is a new submission.

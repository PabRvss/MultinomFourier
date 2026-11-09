#!/bin/bash
set -e
gcc-15 -x c -o pvalues.so src/main.c -shared -fPIC -pthread -Ofast -fno-finite-math-only -O3 -march=native -I/opt/homebrew/opt/simde/include/ -I/opt/homebrew/opt/opencl-headers/include/ -framework OpenCL -lquadmath \
    -x none ExactMultinom/exact_multinom.so BagFFT/bagfft.so

install_name_tool -change ExactMultinom/exact_multinom.so @loader_path/ExactMultinom/exact_multinom.so pvalues.so
install_name_tool -change BagFFT/bagfft.so @loader_path/BagFFT/bagfft.so pvalues.so

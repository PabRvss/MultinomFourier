#!/bin/bash
set -e

g++-15 -x c++ -o ExactMultinom/exact_multinom.so ExactMultinom/exact_multinom_wrap.cpp \
    -std=c++17 -shared -fPIC -O3 &
pid1=$!

g++-15 -x c++ -o BagFFT/bagfft.so BagFFT/bagfft_wrap.cpp \
    -std=c++14 -shared -fPIC -O3 \
    -L/opt/homebrew/lib -lfftw3 &
pid2=$!

wait "$pid1"
wait "$pid2"

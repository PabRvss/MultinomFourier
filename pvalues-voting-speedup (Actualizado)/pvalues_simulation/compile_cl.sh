#!/bin/bash
#echo "" | gcc -xc - -v -E
#g++ -std=c++17 -o pvalues_cl.so src/main_cl.cpp -Wall -shared -fPIC -pthread -I/opt/homebrew/opt/opencl-clhpp-headers/include/ -framework OpenCL -O3 -mcpu=apple-m1 -fdeclspec
gcc -x c -o pvalues_cl.so src/main_cl.c -Wall -shared -fPIC -pthread -I/opt/homebrew/opt/opencl-headers/include/ -framework OpenCL -O3 -mcpu=apple-m1 -fdeclspec

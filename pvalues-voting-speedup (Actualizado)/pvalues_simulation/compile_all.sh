#!/bin/bash
gcc -x c -o pvalues.so src/main.c -Wall -shared -fPIC -pthread -O3 -mcpu=apple-m1 -fdeclspec
gcc -x c -o pvalues_cl.so src/main_cl.c -Wall -shared -fPIC -pthread -I/opt/homebrew/opt/opencl-headers/include/ -framework OpenCL -O3 -mcpu=apple-m1 -fdeclspec

#!/bin/bash
gcc -x c -o pvalues.so src/main.c -Wall -shared -fPIC -pthread -O3 -mcpu=apple-m1 -fdeclspec

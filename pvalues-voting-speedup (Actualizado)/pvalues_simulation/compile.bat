@echo off
gcc -x c -o pvalues.so src/main.c -Wall -shared -fPIC -O3 -pthread -march=native -static
pause

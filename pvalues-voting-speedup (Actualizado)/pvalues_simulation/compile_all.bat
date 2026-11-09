@echo off
gcc -x c -o pvalues.so src/main.c -Wall -shared -fPIC -O3 -pthread -march=native -static
nvcc -O3 -o pvalues_cl.so src/main_cl.c -I"%CUDA_PATH%\include" -L"%CUDA_PATH%\lib\x64" -I"%PTHREADS_PATH%\include" -lOpenCL -Xcompiler="/LD /O2"
pause

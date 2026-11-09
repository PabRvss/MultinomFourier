@echo off
nvcc -O3 -o pvalues_cl.so src/main_cl.c -I"%CUDA_PATH%\include" -L"%CUDA_PATH%\lib\x64" -I"%PTHREADS_PATH%\include" -lOpenCL -Xcompiler="/LD /O2"
pause

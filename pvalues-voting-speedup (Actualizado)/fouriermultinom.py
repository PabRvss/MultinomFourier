import time
import subprocess

import numpy as np
from pvalues_fourier.fourierpy import fourier_calculate, fourier_init, fourier_finish

subprocess.run(["./compile.sh"], cwd="pvalues_fourier", check=True)

N_MAX = int(1e3)
fourier_init(N_MAX, verbose=False)

# x = np.array([20, 20, 10, 10, 5, 5, 2, 2], dtype=int)
x = np.array([46, 19, 10, 10, 5, 5, 2], dtype=int)
# x = np.array([5, 5, 5, 5], dtype=int)
p = np.array([1/len(x)]*len(x))
# p = np.array([0.115595415599955, 0.117955545366418, 0.134294521113883, 0.125837383149419, 0.126238166569396, 0.123399555704553, 0.116877407071095, 0.139802005425280])
# p = np.arange(1, len(x)+1)
# p = np.pow(p, 0.25)
# p = p*p
# p = p/p.sum()

t0 = time.perf_counter()
res = fourier_calculate(x, p)
t1 = time.perf_counter()
print("time:", t1-t0)

fourier_finish()

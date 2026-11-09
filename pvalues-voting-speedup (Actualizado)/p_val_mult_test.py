import ctypes
import pickle
import random

import numpy as np
import time
from scipy.special import loggamma, gamma
from scipy import linalg, stats

from tqdm import tqdm

np.random.seed(0)

# parametros
verbose = True  # imprimir informacion detallada de la ejecucion en cada mesa

file_in = 'data/2023_05_CCG_input_pvalues.pickle'  # datos de entrada
file_in_pval = 'data/2023_05_CCG_output_pvalues.pickle'  # datos de entrada
file_out = 'data/output.pickle'  # datos de salida
N = 0  # tomar muestra aleatoria de mesas, si es <= 0 se toman todas

threads = 20  # procesos paralelos en sampleo con tablas de alias

threshold = 1e-4  # umbral p-valor
error = 1e-8  # error tipo I
confidence = 0.95  # intervalo confianza p-valor
relative_error = 2e0  # umbral p_max/p_min para finalizar mesa

quick = False  # terminar rapido si p de la medicion >= umbral

S_min = int(1e8)  # minimo numero de muestras
S_max = int(1e8)  # maximo numero de muestras
S_np = int(1e3)  # muestras iniciales sampleadas sin tablas de alias, puede estar fuera del rango [S_min, S_max] para samplear de un solo modo

n_max = int(1e3)  # cota para el numero de votantes en una mesa

# fin parametros

random.seed(0)
size_type = np.int64
float_type_sim = np.float64

# float_type_fourier = np.float32
float_type_fourier = np.float64

# lib = np.ctypeslib.load_library('pvalues-simulation/pvalues.so', './')  # version CPU
lib = np.ctypeslib.load_library('pvalues-simulation/pvalues_cl.so', './')  # version OpenCL

lib_fourier = np.ctypeslib.load_library('pvalues-fourier/pvalues.so', './')  # version Fourier CPU


def lib_init():
    real_type = np.ctypeslib.ndpointer(size_type, ndim=1, flags='aligned, contiguous')
    trials_type = np.ctypeslib.ndpointer(size_type, ndim=1, flags='aligned, contiguous')
    probs_type_sim = np.ctypeslib.ndpointer(float_type_sim, ndim=1, flags='aligned, contiguous')
    probs_type_fourier = np.ctypeslib.ndpointer(float_type_fourier, ndim=1, flags='aligned, contiguous')

    res_pval_type_sim = np.ctypeslib.ndpointer(float_type_sim, ndim=1, flags='aligned, contiguous, writeable')
    res_pval_type_fourier = np.ctypeslib.ndpointer(float_type_fourier, ndim=1, flags='aligned, contiguous, writeable')
    res_trials_type = np.ctypeslib.ndpointer(size_type, ndim=1, flags='aligned, contiguous, writeable')

    lib.main_ext.restype = None
    lib.main_ext.argtypes = [
        ctypes.c_int64, ctypes.c_int64,
        real_type, probs_type_sim,
        res_pval_type_sim, res_pval_type_sim, res_pval_type_sim,
        res_trials_type,
    ]

    lib.init_ext.restype = None
    lib.init_ext.argtypes = [
        ctypes.c_int64, ctypes.c_int64,
        ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        trials_type, ctypes.c_int64
    ]

    lib.finish.restype = None
    lib.finish.argtypes = []

    lib_fourier.init_ext.argtypes = [
        ctypes.c_int64,  # int_t fact_max
        ctypes.c_int64,  # int_t verbose
    ]
    lib_fourier.finish.argtypes = []
    lib_fourier.main_ext.argtypes = [
        ctypes.c_int64, ctypes.c_int64,  # const int_t N, const int_t K,
        real_type, probs_type_fourier,  # const int_t* x0, const real_t* probs,
        res_pval_type_fourier,  # real_t* restrict p_value
    ]

    print("----- start init -----")
    t0 = time.perf_counter()
    i_min = int(round(np.log10(S_min)))
    i_max = int(round(np.log10(S_max)))
    trials_array = np.array([int(10**i) for i in range(i_min, i_max+1)], dtype=size_type, order="C")
    trials_n = len(trials_array)

    lib.init_ext(
        threads, n_max,
        threshold, error,
        confidence, relative_error,
        int(quick), trials_n, S_np,
        trials_array, int(verbose))

    lib_fourier.init_ext(n_max, int(verbose))

    t1 = time.perf_counter()
    print(f"----- end init: {t1-t0} seconds -----")
    print()

# compute a p-value using analytical fomrula integrating in one dimension only
def compute_p_value_analytical(z0):
    # z0 : current observed vector
    n = len(z0)
    a = linalg.norm(z0)
    # print("a = ", a)
    p_value = 0
    if n % 2 == 1:
        K_n = np.pi**(0.5) / (2**((n-3)/2) * gamma(n / 2))
        p_value += K_n * (1 - stats.norm.cdf(a))
    #log_Kn = np.log(np.pi**(0.5) / (2**((n-3)/2) * loggamma(n / 2)))
    for k in range(1, int(n / 2) + 1):
        aux_log = 0
        for i in range(1, k):
            aux_log += np.log(n - 2 * i)
        p_value += np.exp(-a**2 / 2 + (n - 2 * k) * np.log(a) + aux_log - loggamma(n / 2) + (1 - n/2) * np.log(2))
        # p_value += np.exp(log_Kn -a**2 / 2 + (n - 2 * k) * np.log(a) + aux_log - loggamma(n / 2) + (1 - n/2) * np.log(2))
    return p_value


# compute a p-value of a multinomial using analytical MVN method in one dimension integration
def compute_p_value_multinomial_with_MVN(x0, p, dim_removed=0):
    # x0 : current observed vector (we use one dimension less)
    # p : vector of the probabilities of the multinomial
    # dim_removed : index of the dimension to remove
    # x0 = np.array([11,  5,  5, 10,  2,  2, 11,  5, 21, 51,  7, 13, 15, 34,  5, 15, 13, 7, 12,  3, 85])
    # p = np.array([0.02388276, 0.02080695, 0.01845486, 0.02478741, 0.01356975, 0.00615162, 0.01755021, 0.01302696, 0.04885109, 0.15397141, 0.03672879, 0.05210783, 0.05482178, 0.09770219, 0.03383391, 0.0289488 , 0.04722272, 0.02315904, 0.02840601, 0.01067487, 0.24534105])
    # x0 = np.array([40, 60])
    # p = np.array([0.45, 0.55])
    J = int(sum(x0))
    # dim_removed = -1
    if dim_removed == 0:
        diferencias = np.abs(int(sum(x0)) * p - x0)
        dim_removed = np.where(diferencias == diferencias.min())[0][0]
    x0 = np.delete(x0, dim_removed)
    p = np.delete(p, dim_removed)
    mu = J * p
    sigma = J * (np.diag(p) - np.outer(p, p))
    if (abs(x0 - mu) < 1e-12).all():
        return 1

    # get the decomposition of sigma
    U, S, _ = np.linalg.svd(sigma)
    # take the square root of S
    S_sqrt = np.diag(np.sqrt(S))
    # U @ V
    # abs(B @ B.T - sigma).max().max()
    B = U @ S_sqrt
    if np.linalg.matrix_rank(B) != B.shape[0]:
        print("se ha terminado temprano")
        return -1
    z0 = np.linalg.inv(B) @ (x0 - mu)

    # compute the p-value
    p_value = compute_p_value_analytical(z0)
    return p_value


def compute_p_value_multinomial_simulate(n, p, S):
    J = sum(n)
    lgac_n = np.array([sum([np.log(max(k, 1)) for k in range(j + 1)]) for j in range(J+1)])
    log_p = np.log(p)
    x_samples = np.random.multinomial(J, p, size=S)
    beta_S = np.sum(x_samples * log_p, axis=1) - np.sum(lgac_n[x_samples], axis=1)
    beta_n = np.sum(n * log_p) - np.sum(lgac_n[n])
    less_p = np.sum(beta_S <= beta_n)
    return less_p/len(beta_S)


def exec_mult(n, p):
    nonzero = np.nonzero(p > 0)
    n = n[nonzero]
    p = p[nonzero]

    J = sum(n)
    c = len(p)

    pairs = sorted([(p[i], i) for i in range(c)], reverse=True)
    n = np.array([n[i] for _, i in pairs], dtype=size_type, order="C")
    p = np.array([p[i] for _, i in pairs], dtype=float_type_sim, order="C")

    pval = np.empty(1, dtype=float_type_sim, order="C")
    pval_fourier = np.empty(1, dtype=float_type_fourier, order="C")
    pval_chisq = np.empty(1, dtype=float_type_sim, order="C")
    pval_llr = np.empty(1, dtype=float_type_sim, order="C")
    trials = np.empty(1, dtype=size_type, order="C")

    lib.main_ext(J, c, n, p, pval, pval_chisq, pval_llr, trials)

    if verbose: print("")
    pairs = sorted([(p[i], i) for i in range(c)], reverse=False)
    n = np.array([n[i] for _, i in pairs], dtype=size_type, order="C")
    p = np.array([p[i] for _, i in pairs], dtype=float_type_fourier, order="C")
    lib_fourier.main_ext(J, c, n, p, pval_fourier)

    pval_mvn = compute_p_value_multinomial_with_MVN(n, p)

    pval = pval[0]
    pval_fourier = pval_fourier[0]
    pval_chisq = pval_chisq[0]
    pval_llr = pval_llr[0]
    trials = trials[0]
    result = {
        "p_value": pval,
        "p_value_fourier": pval_fourier,
        "p_value_mvn": pval_mvn,
        "p_value_chi": pval_chisq,
        "p_value_llr": pval_llr,
        "trials": trials,
    }
    return result


def main():
    lib_init()

    K = 5
    p = np.random.random(K)**3
    # p = np.ones(K)
    p = p/p.sum()

    x0 = p*25 + 200*np.random.random(K)*2/K
    # x0 = p*2 + 8*np.random.random(K)*2/K
    x0 = np.array(np.round(x0), dtype=size_type)
    result = exec_mult(x0, p)
    print(result)
    exit()

    data_out = dict()
    with open(file_in, 'rb') as f:
        data_in: dict = pickle.load(f)
    with open(file_in_pval, 'rb') as f:
        data_in_pval: dict = pickle.load(f)
    data_in: list = [(key, val) for key, val in data_in.items()]
    random.shuffle(data_in)

    if N > 0:
        data_in = data_in[:N]

    iterator = data_in if verbose else tqdm(data_in, mininterval=0.1, maxinterval=0.2, smoothing=0.0)
    t0 = time.perf_counter()
    for key, val in iterator:
        val_out = data_in_pval[key]
        pval_data = val_out["p_value"]
        # if key != ('LAS CONDES', 279):  # ~ 4.2e-10?
        if not (0 <= pval_data <= 2e-9):
            continue

        if verbose:
            print()
            print("-"*100)
            print(key)

        n = np.array(val["x"], dtype=size_type, order="C")
        p = np.array(val["r"], dtype=float, order="C")

        result = exec_mult(n, p)

        if verbose:
            print(result)
            print(val_out)

    t1 = time.perf_counter()
    print()
    print("-"*100)
    print("total time:", t1-t0, "seconds")
    with open(file_out, 'wb') as f:
        pickle.dump(data_out, f)

    lib.finish()
    lib_fourier.finish()


if __name__ == '__main__':
    main()

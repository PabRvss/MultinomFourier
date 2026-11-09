import ctypes
import numpy as np

size_type = np.int64

# float_type = np.float32
float_type = np.float64

lib_fourier = np.ctypeslib.load_library('pvalues_fourier/pvalues.so', './')

real_type = np.ctypeslib.ndpointer(size_type, ndim=1, flags='aligned, contiguous')
probs_type_fourier = np.ctypeslib.ndpointer(float_type, ndim=1, flags='aligned, contiguous')
res_pval_type_fourier = np.ctypeslib.ndpointer(np.float64, ndim=1, flags='aligned, contiguous, writeable')
int_type = np.ctypeslib.ndpointer(np.int64, ndim=1, flags='aligned, contiguous, writeable')


def fourier_init(n_max, verbose=False):
    lib_fourier.init_ext.argtypes = [
        ctypes.c_int64, ctypes.c_int64,  # int_t fact_max, int_t verbose_ext
    ]
    lib_fourier.finish.argtypes = []
    lib_fourier.main_ext.argtypes = [
        ctypes.c_int64, ctypes.c_int64,  # const int_t N, const int_t K,
        real_type, probs_type_fourier,  # const int_t* x0, const real_t* probs,
        res_pval_type_fourier,  # double* restrict p_value
        int_type,  # int_t* converged,
        int_type,  # int_t* terms,
        res_pval_type_fourier,  # double* restrict p0
    ]

    lib_fourier.init_ext(n_max, int(verbose))


def fourier_finish():
    lib_fourier.finish()


def fourier_calculate(x0, p):
    x0 = np.array(x0)
    p = np.array(p)

    nonzero = np.nonzero(p > 0)[0]
    x0 = x0[nonzero]
    p = p[nonzero]

    N = sum(x0)
    K = len(p)

    pairs = sorted([(p[i], i) for i in range(K)], reverse=False)
    x0 = np.array([x0[i] for _, i in pairs], dtype=size_type, order="C")
    p = np.array([p[i] for _, i in pairs], dtype=float_type, order="C")
    pval = np.full(1, fill_value=np.nan, dtype=np.float64, order="C")
    converged = np.full(1, fill_value=-1, dtype=size_type, order="C")
    terms = np.full(1, fill_value=-1, dtype=size_type, order="C")
    p0 = np.full(1, fill_value=-1, dtype=np.float64, order="C")

    lib_fourier.main_ext(N, K, x0, p, pval, converged, terms, p0)

    return pval[0], converged[0], terms[0], p0[0]

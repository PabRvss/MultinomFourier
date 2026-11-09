import ctypes
import pickle
import random
import time

import numpy as np
import pandas as pd

# from scipy.special import loggamma, gamma
# from scipy import linalg, stats
from tqdm import tqdm

from multinomial_mode import multinomial_mode, greedy_mode_find1
from fourierpy import fourier_init, fourier_finish, fourier_calculate

pd.options.display.width = 0
pd.options.display.min_rows = 100
pd.options.display.max_rows = 100
np.random.seed(0)

# parametros
verbose = True  # imprimir informacion detallada de la ejecucion en cada mesa
threads = 20  # procesos paralelos

threshold = 1e-4  # umbral p-valor
error = 1e-8  # error tipo I
confidence = 0.95  # intervalo confianza p-valor
relative_error = 2e0  # umbral p_max/p_min para finalizar mesa

quick = False  # terminar rapido si p de la medicion >= umbral

S_min = int(1e9)  # minimo numero de muestras
S_max = int(1e9)  # maximo numero de muestras
S_np = int(1e3)  # muestras iniciales sampleadas sin tablas de alias, puede estar fuera del rango [S_min, S_max] para samplear de un solo modo

n_max = int(1e4)  # cota para el numero de votantes en una mesa

# fin parametros

random.seed(0)
size_type = np.int64
float_type = np.float64

lib_sim = np.ctypeslib.load_library("pvalues-simulation/pvalues.so", "./")  # version CPU
# lib_sim = np.ctypeslib.load_library("pvalues-simulation/pvalues_cl.so", "./")  # version OpenCL


def lib_sim_init():
    real_type = np.ctypeslib.ndpointer(size_type, ndim=1, flags="aligned, contiguous")
    trials_type = np.ctypeslib.ndpointer(size_type, ndim=1, flags="aligned, contiguous")
    probs_type_sim = np.ctypeslib.ndpointer(float_type, ndim=1, flags="aligned, contiguous")

    res_pval_type_sim = np.ctypeslib.ndpointer(float_type, ndim=1, flags="aligned, contiguous, writeable")
    res_trials_type = np.ctypeslib.ndpointer(size_type, ndim=1, flags="aligned, contiguous, writeable")

    lib_sim.main_ext.restype = None
    lib_sim.main_ext.argtypes = [
        ctypes.c_int64, ctypes.c_int64,
        real_type, probs_type_sim,
        res_pval_type_sim, res_pval_type_sim, res_pval_type_sim,
        res_trials_type,
    ]

    lib_sim.init_ext.restype = None
    lib_sim.init_ext.argtypes = [
        ctypes.c_int64, ctypes.c_int64,
        ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        trials_type, ctypes.c_int64
    ]

    lib_sim.finish.restype = None
    lib_sim.finish.argtypes = []

    print("----- start init -----")
    t0 = time.perf_counter()
    i_min = int(round(np.log10(S_min)))
    i_max = int(round(np.log10(S_max)))
    trials_array = np.array([int(10**i) for i in range(i_min, i_max+1)], dtype=size_type, order="C")
    trials_n = len(trials_array)

    lib_sim.init_ext(
        threads, n_max,
        threshold, error,
        confidence, relative_error,
        int(quick), trials_n, S_np,
        trials_array, int(verbose))

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


def exec_fourier(n, p):
    if verbose: print("")
    pval, converged, terms, p0 = fourier_calculate(n, p)
    result = {
        "p_value": pval,
        "converged": bool(converged),
        "terms": terms,
        "p0": p0,
    }
    return result


def exec_mult(n, p):
    nonzero = np.nonzero(p > 0)
    n = n[nonzero]
    p = p[nonzero]

    J = sum(n)
    c = len(p)

    pairs = sorted([(p[i], i) for i in range(c)], reverse=True)
    n = np.array([n[i] for _, i in pairs], dtype=size_type, order="C")
    p = np.array([p[i] for _, i in pairs], dtype=float_type, order="C")

    pval = np.zeros(1, dtype=float_type, order="C")
    pval_chisq = np.zeros(1, dtype=float_type, order="C")
    pval_llr = np.zeros(1, dtype=float_type, order="C")
    trials = np.zeros(1, dtype=size_type, order="C")

    # lib_sim.main_ext(J, c, n, p, pval, pval_chisq, pval_llr, trials)

    if verbose: print("")
    pval_fourier, converged, terms = fourier_calculate(n, p)

    pval_mvn = compute_p_value_multinomial_with_MVN(n, p)

    pval = pval[0]
    pval_chisq = pval_chisq[0]
    pval_llr = pval_llr[0]
    trials = trials[0]
    result = {
        "p_value": pval,
        "p_fourier": pval_fourier,
        "p_value_mvn": pval_mvn,
        "p_value_chi": pval_chisq,
        "p_value_llr": pval_llr,
        "trials": trials,
    }
    return result


def main():
    K = 5
    p = np.random.random(K)**4
    # p = np.ones(K)
    p = p/p.sum()

    x0 = p*25 + 150*np.random.random(K)*2/K
    # x0 = p*2 + 8*np.random.random(K)*2/K
    x0 = np.array(np.round(x0), dtype=size_type)
    result = exec_mult(x0, p)
    print(result)


def main_arg():
    cols_group = ["DISTRITO_NOMBRE", "SECCION_NOMBRE", "CIRCUITO_ID"]
    cols_mesa = cols_group + ["MESA_ID"]

    df = pd.read_csv("data/ARG_2023_12_Presidencial_PAIS.csv", dtype={col: str for col in cols_mesa})
    df["P-VALOR-ALG"] = -1.0
    df["CONVERGED-ALG"] = -1
    df["TERMS-ALG"] = -1
    print(df)

    df[cols_mesa] = df[cols_mesa].astype(str)
    cols_votes = df.columns.tolist()[6:12]
    df_group = df.groupby(cols_group)
    sums = df_group[cols_votes].sum()
    totals = sums.sum(axis=1)
    probs = sums.divide(totals, axis=0)

    del df_group, sums, totals

    iterator = tqdm(df.iterrows(), total=len(df), smoothing=0.0, maxinterval=0.1) if not verbose else df.iterrows()
    for i, (_, row) in enumerate(iterator):
        pval_sim = row["P-VALOR"]
        mesa = tuple(row[cols_mesa])
        if pval_sim > 1e-8:
            continue

        local = tuple(row[cols_group])
        p = probs.loc[local].values
        x0 = row[cols_votes].values

        if verbose:
            print("#"*120)
            print(f"{i} {mesa}: {pval_sim}")
            print("prob local =", p)
            print("x0 =", x0)

        pval, converged, terms = fourier_calculate(x0, p)
        df["P-VALOR-ALG"].iat[i] = pval
        df["CONVERGED-ALG"].iat[i] = converged
        df["TERMS-ALG"].iat[i] = terms

        if verbose:
            print(f"\t{pval_sim}\t{pval}")

    # df.to_csv("data/ARG_2023_12_Presidencial_PAIS_ALG.csv", index=False)


def main_chl():
    file_in = "data/2023_05_CCG_input_pvalues.pickle"  # datos de entrada
    file_in_pval = "data/2023_05_CCG_output_pvalues.pickle"  # datos de entrada
    N = 0

    with open(file_in, "rb") as f:
        data_in: dict = pickle.load(f)
    with open(file_in_pval, "rb") as f:
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
        if not (0 <= pval_data <= 1e-8):
            continue

        if verbose:
            print()
            print("-"*100)
            print(key)

        x0 = np.array(val["x"], dtype=size_type, order="C")
        p = np.array(val["r"], dtype=float, order="C")

        result = exec_mult(x0, p)

        if verbose:
            print("current:", result)
            print("data:", val_out)

    t1 = time.perf_counter()
    print()
    print("-"*100)
    print("total time:", t1-t0, "seconds")


def main_fourier(df_full, df_zero, df_pvals, data_dir):
    df_pvals["p-fourier"] = np.NAN
    df_pvals["converged"] = ""
    df_pvals["terms"] = -2
    df_pvals["p0"] = np.NAN
    df_pvals["candidatos"] = ""

    df_mesas = pd.read_csv(f"data/{data_dir}/mesas.csv", sep=";", index_col="id_mesa")
    df_cand = pd.read_csv(f"data/{data_dir}/candidatos.csv", sep=";")
    cand_cols = ["candidato", "pacto", "partido"]

    # para preliminares
    cand_cols = ["candidato"]

    df_cand.drop_duplicates("id_candidato", inplace=True)
    df_cand.set_index("id_candidato", drop=True, inplace=True, verify_integrity=True)

    df_full["id_local"] = df_full["id_mesa"].map(df_mesas["id_local"])
    for c in cand_cols:
        df_full[c] = df_full["id_candidato"].map(df_cand[c])

    nulos = df_full.loc[df_full["id_candidato"].isin([-1, -2])].groupby("id_mesa")["votos"].sum()
    df_full = df_full.loc[df_full["id_candidato"] != -2].reset_index(drop=True)
    mask = df_full["id_candidato"] == -1
    df_full.loc[mask, "votos"] = df_full.loc[mask, "id_mesa"].map(nulos)

    df_grouped = df_full.groupby("id_mesa")
    df_grouped_neg = (df_grouped["votos"].min() < 0) | (df_grouped["votos"].sum() == df_grouped["votos"].max())
    mesas_remove = df_grouped_neg.loc[df_grouped_neg].index

    df_full = df_full.loc[~df_full["id_mesa"].isin(mesas_remove)]
    df_zero = df_zero.loc[~df_zero["id_mesa"].isin(mesas_remove)]
    df_pvals.loc[df_pvals.index.isin(mesas_remove), "terms"] = -3

    df_grouped = df_full.groupby("id_local")
    dict_mesas = {}
    for local_id, df_local in tqdm(df_grouped, total=len(df_grouped), smoothing=0.0, desc="freqs"):
        p = df_local.groupby("id_candidato")["votos"].sum()/df_local["votos"].sum()
        for id_mesa, df_mesa in df_local.groupby("id_mesa"):
            p_mesa = df_mesa["id_candidato"].map(p).values
            n_mesa = df_mesa["votos"].values
            dict_mesas[id_mesa] = (n_mesa, p_mesa)

    id_test = 712800093

    n_mesa, p_mesa = dict_mesas[id_test]
    N = n_mesa.sum()
    mode = greedy_mode_find1(N, p_mesa)

    df_mesa = df_full.loc[df_full["id_mesa"] == id_test].copy()
    df_mesa = df_mesa[cand_cols]
    df_mesa.set_index("candidato", drop=True, inplace=True)
    df_mesa["p"] = p_mesa
    df_mesa["N*p"] = N*p_mesa
    df_mesa["moda"] = mode
    df_mesa["obs"] = n_mesa

    df_mesa = df_mesa[["p", "N*p", "moda", "obs"]]
    print(df_mesa.sort_index())
    print("total:", N)
    print(df_mesa.sum())
    print(exec_fourier(n_mesa, p_mesa))
    exit()

    df_grouped = df_zero.groupby("id_mesa")
    for id_mesa, _ in tqdm(df_grouped, total=len(df_grouped), smoothing=0.0, maxinterval=0.1, desc="fourier"):
        if verbose:
            print()
            print("#"*150)
        n_mesa, p_mesa = dict_mesas[id_mesa]
        result = exec_fourier(n_mesa, p_mesa)
        df_pvals.loc[id_mesa, ["p-fourier", "converged", "terms", "p0", "candidatos"]] = \
            result["p_value"], str(result["converged"]), result["terms"], result["p0"], str(len(n_mesa))

    return df_pvals


if __name__ == "__main__":
    fourier_init(n_max, verbose)
    # lib_sim_init()

    # x0 = [0, 2, 0, 1, 0, 0, 1, 2, 0, 3, 0, 0, 4, 1, 5, 6, 2, 1, 2, 2, 1, 8, 1, 4, 1, 10, 6, 0, 2, 1, 7, 0, 2, 7, 1, 1, 4, 4, 0, 0, 0, 12, 4, 6, 4, 2, 3, 2, 0, 2, 4, 2, 1, 3, 1, 1, 6, 5, 3, 1, 8, 0, 4, 0, 0, 0, 1, 2, 0, 1, 2, 3, 1, 10, 2, 0, 3, 3, 3, 5, 4, 4, 1, 0, 1, 4, 8, 12, 1, 1, 94, 68]
    # p = [0.00195875100818066, 0.0100241963359834, 0.00391750201636133, 0.00334139877866113, 0.00184353036064062, 0.00241963359834082, 0.00241963359834082, 0.0047240465491416, 0.0014978684180205, 0.00518492913930176, 0.00126742712294043, 0.000921765180320313, 0.00564581172946191, 0.00195875100818066, 0.0106002995736836, 0.00990897568844336, 0.00483926719668164, 0.0053001497868418, 0.00322617813112109, 0.00426316395898144, 0.00241963359834082, 0.0248876598686484, 0.0053001497868418, 0.00944809309828321, 0.00414794331144141, 0.0197027307293467, 0.00898721050812305, 0.00334139877866113, 0.00403272266390137, 0.00334139877866113, 0.00956331374582325, 0.00311095748358105, 0.00334139877866113, 0.00806544532780274, 0.00403272266390137, 0.00460882590160156, 0.0102546376310634, 0.0100241963359834, 0.00357184007374121, 0.0014978684180205, 0.000691323885240235, 0.0157852287129853, 0.0109459615163037, 0.00679801820486231, 0.00391750201636133, 0.00299573683604101, 0.00645235626224219, 0.00426316395898144, 0.00161308906556054, 0.00841110727042286, 0.00864154856550293, 0.00322617813112109, 0.00241963359834082, 0.00576103237700195, 0.00230441295080078, 0.0115220647540039, 0.0153243461228252, 0.0101394169835234, 0.00288051618850097, 0.00322617813112109, 0.0150939048277451, 0.00334139877866113, 0.00714368014748242, 0.00184353036064062, 0.00230441295080078, 0.00679801820486231, 0.00195875100818066, 0.00990897568844336, 0.00299573683604101, 0.00311095748358105, 0.00380228136882129, 0.00380228136882129, 0.00403272266390137, 0.00299573683604101, 0.0100241963359834, 0.00184353036064062, 0.0107155202212236, 0.0117525060490839, 0.00610669431962207, 0.00990897568844336, 0.0116372854015439, 0.00679801820486231, 0.00345661942620117, 0.00218919230326074, 0.00564581172946191, 0.00426316395898144, 0.0144025809425048, 0.0332987671390713, 0.00495448784422168, 0.00299573683604101, 0.248415716096324, 0.186196566424703]
    # x0 = x0[:30]
    # p = p[:30]
    K = 20
    np.random.seed(0)
    p = np.random.random(K)
    p = np.exp(3*p)
    p = p/p.sum()
    x0 = 20*np.ones(K, dtype=int)

    print(x0)
    print(p)

    res = fourier_calculate(x0, p)
    print(res)

    # main()
    # main_arg()
    # main_chl()

    # G 1: gobernador
    # A 2: alcalde
    # C 3: concejal
    # R 7: consejero
    data_dir = "2024_prelim"
    file_pvals = "2024_10_ACGC_CA_A_PAIS"
    file_full = "resultados_2"

    df_pvals = pd.read_csv(f"data/{data_dir}/{file_pvals}.csv", index_col="ID_MESA")
    df_full = pd.read_csv(f"data/{data_dir}/{file_full}.csv", sep=";")

    ids_zero = df_pvals.loc[df_pvals["P-VALOR"] < 1e-4].index.unique()
    df_zero = df_full.loc[df_full["id_mesa"].isin(ids_zero)].copy()
    df_pvals = main_fourier(df_full, df_zero, df_pvals, data_dir)
    df_pvals.to_csv(f"data/{data_dir}/{file_pvals}_fourier_double_1e-3.csv")

    fourier_finish()
    # lib_sim.finish()

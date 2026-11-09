import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from fourierpy import fourier_init, fourier_finish, fourier_calculate
from multinomial_mode import greedy_mode_find1

pd.options.display.width = 0
verbose = True
n_max = int(1e4)  # cota para el numero de votantes en una mesa
data_dir = "../../mesas/2026_01"
id_eleccion = 6


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


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


def main():
    fourier_init(n_max, verbose)

    df_mesas = pd.read_csv(f"data/{data_dir}/mesas.csv", sep=";", index_col="id_mesa")
    df_cand = pd.read_csv(f"data/{data_dir}/candidatos.csv", sep=";", index_col="id_candidato")
    df_res = pd.read_csv(f"data/{data_dir}/resultados_{id_eleccion}.csv", sep=";")

    df_res = df_res.join(df_mesas[['id_local']], on='id_mesa', how='left')
    df_res = df_res.join(df_cand[['candidato']], on='id_candidato', how='left', rsuffix="-")

    df_res.loc[df_res['votos'] == -9999, 'votos'] = 0
    mask = df_res['id_candidato'] == -3  # abstenciones
    votos = df_res.loc[~mask].groupby("id_mesa")['votos'].sum()
    abstenciones = df_mesas[f'electores_{id_eleccion}'].astype(int) - votos
    print(abstenciones)
    df_res.loc[mask, 'votos'] = df_res.loc[mask, 'id_mesa'].map(abstenciones)

    id_test = 758900002
    # id_test = 758900009

    df_res_test = df_res.loc[df_res['id_mesa'] == id_test]
    print(df_res_test)
    print(df_res_test['votos'].sum())

    df_grouped = df_res.groupby("id_mesa")
    df_grouped_neg = df_grouped["votos"].min() < 0
    mesas_remove = df_grouped_neg.loc[df_grouped_neg].index
    df_res = df_res.loc[~df_res["id_mesa"].isin(mesas_remove)]

    df_grouped = df_res.groupby("id_local")
    dict_mesas = {}
    for id_local, df_local in tqdm(df_grouped, total=len(df_grouped), smoothing=0.0, desc="freqs"):
        p = df_local.groupby("id_candidato")["votos"].sum()/df_local["votos"].sum()
        for id_mesa, df_mesa in df_local.groupby("id_mesa"):
            p_mesa = df_mesa["id_candidato"].map(p).values
            n_mesa = df_mesa["votos"].values
            dict_mesas[id_mesa] = (n_mesa, p_mesa)

    n_mesa, p_mesa = dict_mesas[id_test]
    N = n_mesa.sum()
    mode = greedy_mode_find1(N, p_mesa)

    df_mesa = df_res.loc[df_res["id_mesa"] == id_test].copy()
    df_mesa.set_index("candidato", drop=True, inplace=True)
    df_mesa["p"] = p_mesa
    df_mesa["N*p"] = N*p_mesa
    df_mesa["moda"] = mode
    df_mesa["obs"] = n_mesa

    df_mesa = df_mesa[["p", "N*p", "moda", "obs"]]
    print(df_mesa.sort_index())
    print("total:", N)
    print(df_mesa.sum())
    p_res = exec_fourier(n_mesa, p_mesa)
    print(json.dumps(p_res, indent=4, cls=NpEncoder))

    fourier_finish()


if __name__ == '__main__':
    main()

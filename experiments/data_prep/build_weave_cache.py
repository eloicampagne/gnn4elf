"""Mise en cache des cinq cents sous-stations WEAVE pour le volet horizons.

Le dispositif reprend celui des quarante-huit horizons sur les régions
françaises. Un nœud est une sous-station, une observation est une origine de
prévision, les traits sont les quarante-huit retards de consommation disponibles
à cette origine plus le calendrier, et la cible à l'horizon h est la consommation
h demi-heures plus tard. Aucun retard postérieur à l'origine n'entre dans le plan
d'expérience, la prévision est donc authentiquement multi-horizon.

Aucune adjacence n'accompagne le jeu à cinq cents nœuds, contrairement aux
vingt-huit. On en construit une par corrélation de Pearson entre les séries
d'apprentissage, réduite aux k plus proches voisins puis symétrisée. Le graphe
complet serait dense à deux cent cinquante mille arêtes par échantillon, ce que
le passage de messages ne supporte pas à ce format de lot.

    python3.12 build_weave_cache.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import ROOT, DATA, CACHE  # noqa: E402  (puts every experiment folder on sys.path)
WEAVE = Path(DATA) / "weave"
OUT = Path(CACHE)
H = 48
KNN = int(os.environ.get("WEAVE_KNN", 8))


def load_split(path):
    df = pd.read_csv(path)
    df = df.sort_values(["id_unique", "date"]).reset_index(drop=True)
    nodes = sorted(df.id_unique.unique())
    lags = [f"consumption_l{t}" for t in range(1, H + 1)]
    per_node = {n: g.reset_index(drop=True) for n, g in df.groupby("id_unique")}
    T = len(per_node[nodes[0]])
    assert all(len(g) == T for g in per_node.values()), "séries de longueurs inégales"
    C = np.stack([per_node[n]["consumption"].values for n in nodes], axis=1)
    L = np.stack([per_node[n][lags].values for n in nodes], axis=1)   # [T, N, 48]
    ref = per_node[nodes[0]]
    inst = ref["instant"].values
    wd = ref["weekday"].values
    return nodes, C.astype(np.float64), L.astype(np.float64), inst, wd


def calendar(inst, wd):
    """Instant en coordonnées cycliques, jour de semaine en indicatrices."""
    a = 2 * np.pi * inst / 48.0
    dummies = np.zeros((len(wd), 7))
    dummies[np.arange(len(wd)), wd.astype(int)] = 1.0
    return np.column_stack([np.cos(a), np.sin(a), dummies])


def windows(C, L, cal):
    """Origines i, plan L[i] et calendrier cal[i], cibles C[i:i+48]."""
    T, N, _ = L.shape
    n = T - H + 1
    X = np.concatenate([L[:n], np.repeat(cal[:n, None, :], N, axis=1)], axis=2)
    Y = np.stack([C[i:i + H] for i in range(n)])          # [n, 48, N]
    return X.astype(np.float32), np.transpose(Y, (0, 2, 1)).astype(np.float32)


def knn_graph(C_train, k=KNN):
    """k plus proches voisins par corrélation de Pearson, symétrisé.

    Les poids sont les corrélations conservées, ramenées dans [0, 1]. Les séries
    constantes, s'il y en a, donnent une corrélation indéfinie et sont isolées
    par une boucle sur elles-mêmes plutôt que rattachées arbitrairement.
    """
    Z = C_train - C_train.mean(0, keepdims=True)
    sd = Z.std(0, keepdims=True)
    dead = (sd < 1e-9).ravel()
    Z = Z / np.where(sd < 1e-9, 1.0, sd)
    R = (Z.T @ Z) / len(Z)
    R = np.clip(np.nan_to_num(R, nan=0.0), -1.0, 1.0)
    np.fill_diagonal(R, -np.inf)
    N = R.shape[0]
    A = np.zeros((N, N))
    for i in range(N):
        if dead[i]:
            continue
        for j in np.argsort(-R[i])[:k]:
            w = max(R[i, j], 0.0)
            A[i, j] = max(A[i, j], w)
            A[j, i] = max(A[j, i], w)
    np.fill_diagonal(A, 1.0)
    return A, int(dead.sum())


def main():
    OUT.mkdir(exist_ok=True)
    ntr, Ctr, Ltr, itr, wtr = load_split(WEAVE / "train_weave_500.csv")
    nte, Cte, Lte, ite, wte = load_split(WEAVE / "test_weave_500.csv")
    assert ntr == nte, "les deux fichiers ne portent pas les mêmes sous-stations"
    Xtr, Ytr = windows(Ctr, Ltr, calendar(itr, wtr))
    Xte, Yte = windows(Cte, Lte, calendar(ite, wte))
    A, n_dead = knn_graph(Ctr)
    print(f"{len(ntr)} sous-stations, d={Xtr.shape[2]}, "
          f"origines train {Xtr.shape[0]}, test {Xte.shape[0]}", flush=True)
    print(f"graphe kNN k={KNN}: {int((A > 0).sum() - len(ntr))} arêtes dirigées, "
          f"{n_dead} séries constantes isolées", flush=True)
    np.savez_compressed(OUT / "weave500_tasks.npz", nodes=np.array(ntr),
                        Xtr=Xtr, Ytr=Ytr, Xte=Xte, Yte=Yte, A=A)
    print("écrit", OUT / "weave500_tasks.npz", flush=True)


if __name__ == "__main__":
    main()

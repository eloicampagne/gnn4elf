"""The additive expert, on the operational load formula.

Single-step setting, one model per node and per half-hour:

    load ~ s(Temp, k=5) + s(Nebu, k=5) + s(Wind, k=5) + s(TempLissFort, k=5)
         + s(TempLissFaible, k=5) + s(tempMax, k=5) + s(tempMin, k=5)
         + s(Posan, k=30, bs='cc') + offset:DayType,

a smooth of five basis functions per weather covariate, a cyclic smooth of thirty
on the position in the year, and a factor crossing the period with the day type.
Fitting per half-hour is how the formula is deployed, and a pooled fit costs
several hundred megawatts of national error on the twelve French regions.

Sequence-to-sequence setting, one model per node and per horizon, which is the
adaptation the chapter describes. The horizon-k model reads the weather of the
issue instant, the calendar of the instant it predicts, the last observed load
and the load of the same instant one day earlier, and predicts the load k steps
after the issue instant.

On the synthetic panel the weather block is whatever the generator produced, the
temperature and what derives from it.
"""

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _paths import ROOT, DATA, CACHE  # noqa: E402,F401  (puts every experiment folder on sys.path)
import aggreg_experts as A  # noqa: E402


def _cross(cal, fitmask_t):
    """offset:DayType as one factor, coded on the pairs seen while fitting.

    A pair the fitting window never showed falls back to the most frequent one,
    the factor having no level for it.
    """
    pairs = cal[:, 3].astype(int) * 100 + cal[:, 2].astype(int)
    code, count = np.unique(pairs[fitmask_t], return_counts=True)
    out = np.searchsorted(code, pairs)
    unseen = (out >= len(code)) | (code[np.minimum(out, len(code) - 1)] != pairs)
    out[unseen] = int(np.argmax(count))
    return out.astype(float)


def _weather_terms(k, n_splines=5):
    from pygam import s
    terms = s(0, n_splines=n_splines)
    for j in range(1, k):
        terms = terms + s(j, n_splines=n_splines)
    return terms


def fit_single(xn, cal, y, tt_fit, tt_te, n_nodes):
    """One model per node and per half-hour."""
    from pygam import LinearGAM
    from pygam import f as fac
    from pygam import s

    k = xn.shape[2]
    terms = _weather_terms(k) + s(k, n_splines=30, basis="cp") + fac(k + 1)
    cross = _cross(cal, tt_fit)
    inst = cal[:, 0].astype(int)
    pred = np.zeros((len(tt_te), n_nodes))
    for j in range(n_nodes):
        z = np.column_stack([xn[:, j, :], cal[:, 1], cross])
        for h in range(48):
            tr = tt_fit[inst[tt_fit] == h]
            te = tt_te[inst[tt_te] == h]
            if len(te) == 0:
                continue
            g = LinearGAM(terms).fit(z[tr], y[tr, j])
            pred[inst[tt_te] == h, j] = g.predict(z[te])
    return pred


def fit_seq(xn, cal, y, tt_fit, tt_te, n_nodes):
    """One model per node and per half-hour of the day, as the chapter's baseline.

    Every evaluation window is issued at midnight, so horizon h is the half-hour
    h and the per-horizon and per-instant formulations coincide; the fit is
    restricted to midnight issues so that the identity holds there too. Each
    model reads the weather and the calendar of the instant it predicts and the
    load of that instant on the previous day, which a forecast issued at midnight
    has in hand. Pooling the horizons instead, with the half-hour as a factor,
    costs a third of the validation error.

    The weather of the predicted instant is what the chapter's GAM baseline uses,
    where the networks read the weather of the issue instant alone. The asymmetry
    favours this expert, so the contribution measured for the graph experts in
    that setting is a lower bound.
    """
    from pygam import LinearGAM, l
    from pygam import f as fac
    from pygam import s

    H = A.HORIZON
    k = xn.shape[2]
    terms = _weather_terms(k) + s(k, n_splines=30, basis="cp") + fac(k + 1) + l(k + 2)
    inst = cal[:, 0].astype(int)
    tt_fit = tt_fit[inst[tt_fit] == 0]
    cross = _cross(cal, tt_fit)
    pred = np.zeros((len(tt_te), n_nodes, H))
    for j in range(n_nodes):
        for h in range(H):
            def design(tt):
                return np.column_stack([xn[tt + h, j, :], cal[tt + h, 1],
                                        cross[tt + h], y[tt + h - H, j]])
            g = LinearGAM(terms).fit(design(tt_fit), y[tt_fit + h, j])
            pred[:, j, h] = g.predict(design(tt_te))
    return pred


def fit_gam(setting, xn, cal, y, when, fitmask, temask, n_nodes):
    """Dispatch on the setting; masks are over the rows aggreg_experts built."""
    tt_fit, tt_te = when[fitmask], when[temask]
    if setting == "single":
        return fit_single(xn, cal, y, tt_fit, tt_te, n_nodes)
    return fit_seq(xn, cal, y, tt_fit, tt_te, n_nodes)


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "rfrance"
    setting = sys.argv[2] if len(sys.argv) > 2 else "single"
    xn, cal, y, nodes, dates = A.read_real()
    itr_all = (dates >= A.TRAIN_FROM) & (dates < A.TRAIN_TO)
    if dataset == "synth":
        A.NODE_FEATS = A.FEATS_SYNTH
        xn, y = A.make_synth(xn, cal, y, dates, itr_all, A.SYNTH_NOISE, A.SEED)
    x, tgt, when = A.build(setting, xn, cal, y)
    itr, iva, ite = A.split(dates, when, setting)

    t0 = time.time()
    p = fit_gam(setting, xn, cal, y, when, itr | iva, ite, len(nodes))
    nat = A.national(p, setting)
    truth = A.national(tgt[ite], setting)
    print(f"GAM  national RMSE {np.sqrt(((nat - truth) ** 2).mean()):8.1f} MW"
          f"  ({time.time() - t0:.0f}s)", flush=True)

    path = os.path.join(A.OUT, f"experts_{dataset}_{setting}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df["GAM"] = nat
        df.to_csv(path)
        print("reecrit", path)

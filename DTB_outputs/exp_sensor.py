
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, '/home/renku/work/DTB_outputs'); import dtb_regression as R
from sklearn.ensemble import HistGradientBoostingRegressor
OUT = '/home/renku/work/DTB_outputs/sensor_results'; os.makedirs(OUT, exist_ok=True)
F = R.F; df = R.df
EV = {b: {f: g.loc[g[f].notna(), ['elapsed_time', f]].values for f in F} for b, g in df.groupby('batch_id')}
VOLS = {b: g.loc[g.volume.notna(), ['elapsed_time', 'volume']].values for b, g in df.groupby('batch_id')}
PAST = [0.5, 1, 2]; FUT = [0.5, 1, 2]
def sensor_feats(b, t):
    x = {}; V = np.interp(t, VOLS[b][:, 0], VOLS[b][:, 1])
    for f in F:
        e = EV[b][f]; tt, am = (e[:, 0], e[:, 1]) if len(e) else (np.array([]), np.array([]))
        for w in PAST:
            m = (tt > t - w) & (tt <= t); x[f'{f}_past{w}_n'] = m.sum(); x[f'{f}_past{w}_amtV'] = am[m].sum() / V
        for w in FUT:
            m = (tt > t) & (tt <= t + w); x[f'{f}_fut{w}_n'] = m.sum(); x[f'{f}_fut{w}_amtV'] = am[m].sum() / V
        nxt = tt[tt > t]; prv = tt[tt <= t]
        x[f'{f}_to_next'] = min(nxt.min() - t, 3.0) if len(nxt) else 3.0
        x[f'{f}_since_last'] = min(t - prv.max(), 3.0) if len(prv) else 3.0
    return x
Zx = pd.concat([R.Z, pd.DataFrame([sensor_feats(r.batch, r.t) for r in R.Z.itertuples()], index=R.Z.index)], axis=1)
R.Z = Zx                                   # cvp() reads the module-level Z
SC = [c for c in Zx.columns if any(c.startswith(f + '_') for f in F) and ('past' in c or 'fut' in c or 'to_next' in c or 'since_last' in c)]
PASTC = [c for c in SC if 'past' in c or 'since_last' in c]; FUTC = [c for c in SC if 'fut' in c or 'to_next' in c]
A4 = R.S0 + R.VOL + sum([R.FEEDX(f) for f in F], [])
FSX = {'A4': A4, 'A4past': A4 + PASTC, 'A4fut': A4 + FUTC, 'A4both': A4 + SC}
mode, fs = sys.argv[1], sys.argv[2]
if mode == 'ridge':
    R.cvp(FSX[fs], pool=1, wnb=0.5).to_csv(f'{OUT}/ridge_{fs}.csv')
elif mode == 'hgb':
    mk = lambda: HistGradientBoostingRegressor(max_depth=4, learning_rate=0.03, max_iter=400, min_samples_leaf=10)
    R.cvp(FSX[fs], mk=mk, pool=14, wnb=1.0).to_csv(f'{OUT}/hgb_{fs}.csv')
elif mode == 'hgbphase':
    PH = {'early': (1, 4), 'mid': (5, 9), 'late': (10, 13)}; out = []
    for fold in R.FOLDS:
        tr = Zx[Zx.batch.isin(set(R.TRAIN) - set(fold))]; va = Zx[Zx.batch.isin(fold)]; P = pd.DataFrame(index=va.index, columns=R.T4, dtype=float)
        for a, b in PH.values():
            trr = tr[(tr.day >= a - 1) & (tr.day <= b + 1)]; m = (va.day >= a) & (va.day <= b); cols = FSX[fs] + ['t']
            for y in R.T4:
                P.loc[m, y] = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.03, max_iter=400, min_samples_leaf=10).fit(trr[cols], np.log(trr[y])).predict(va.loc[m, cols])
        out.append(P)
    pd.concat(out).sort_index().to_csv(f'{OUT}/hgbphase_{fs}.csv')
elif mode == 'knn':
    # analog batches: profile = day-0 state + volume and cumulative feed per volume at each day 1..13
    k = int(fs); days = np.arange(1, 14)
    def prof(b):
        g = Zx[Zx.batch == b].sort_values('t'); r0 = g.iloc[0]; v = list(r0[R.S0].values)
        for f in F: v += list(np.interp(days, g.t, g[f + '_cum'] / g.V))
        v += list(np.interp(days, g.t, g.V)); return np.array(v, float)
    PR = {b: prof(b) for b in R.ALL}; out = []
    for fold in R.FOLDS:
        trb = [b for b in R.TRAIN if b not in set(fold)]; M = np.array([PR[b] for b in trb]); mu, sd = M.mean(0), M.std(0) + 1e-9
        va = Zx[Zx.batch.isin(fold)]; P = pd.DataFrame(index=va.index, columns=R.T4, dtype=float)
        for b in fold:
            dist = np.sqrt((((M - PR[b]) / sd) ** 2).mean(1)); nn = np.argsort(dist)[:k]; w = 1 / (dist[nn] + 1e-3)
            for i in va.index[va.batch == b]:
                d = va.at[i, 'day']
                for y in R.T4:
                    vals = []; ws = []
                    for j, bb in enumerate(np.array(trb)[nn]):
                        r = Zx[(Zx.batch == bb) & (Zx.day == d)]
                        if len(r): vals.append(np.log(r[y].values[0])); ws.append(w[j])
                    P.at[i, y] = np.average(vals, weights=ws)
        out.append(P)
    pd.concat(out).sort_index().to_csv(f'{OUT}/knn_k{k}.csv')
print('done', sys.argv[1:])

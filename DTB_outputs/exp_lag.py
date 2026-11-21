
import sys, os, numpy as np, pandas as pd
variant = sys.argv[1]; sys.argv = ['x', 'none', 'none']
src = open('/home/renku/work/DTB_outputs/exp_target.py').read(); exec(src.rsplit("mode, fs = sys.argv[1], sys.argv[2]", 1)[0])   # Zx, A4, TS, R (unchanged pipeline)
from sklearn.ensemble import HistGradientBoostingRegressor
OUT = '/home/renku/work/DTB_outputs/lag_results'; os.makedirs(OUT, exist_ok=True)
EVL = {b: {f: g.loc[g[f].notna(), ['elapsed_time', f]].values for f in R.F} for b, g in R.df.groupby('batch_id')}
VL = {b: g.loc[g.volume.notna(), ['elapsed_time', 'volume']].values for b, g in R.df.groupby('batch_id')}
rows = []
for b, t in zip(Zx.batch, Zx.t):
    x = {'V_lag2': np.interp(t - 2, VL[b][:, 0], VL[b][:, 1])}
    for f in R.F:
        e = EVL[b][f]; tt, am = (e[:, 0], e[:, 1]) if len(e) else (np.array([]), np.array([]))
        for L in [1, 2, 3]: x[f'{f}_cum_lag{L}'] = am[tt <= t - L].sum()
        x[f'{f}_win_3to1'] = am[(tt > t - 3) & (tt <= t - 1)].sum(); x[f'{f}_win_2p5to1p5'] = am[(tt > t - 2.5) & (tt <= t - 1.5)].sum()
    rows.append(x)
Zx = pd.concat([Zx, pd.DataFrame(rows, index=Zx.index)], axis=1); R.Z = Zx
LAG2 = ['V_lag2'] + [f'{f}_cum_lag2' for f in R.F]
LAGALL = ['V_lag2'] + [c for c in Zx.columns if '_cum_lag' in c or '_win_' in c]
FS = {'A4lag2': A4 + LAG2, 'A4lagall': A4 + LAGALL, 'subS_lag2': TS['sub_futS'] + LAG2}
model, fs = variant.split('__')
if model == 'ridge':
    R.cvp(FS[fs], pool=1, wnb=0.5).to_csv(f'{OUT}/{variant}.csv')
else:
    out = []
    for fold in R.FOLDS:
        tr = Zx[Zx.batch.isin(set(R.TRAIN) - set(fold))]; va = Zx[Zx.batch.isin(fold)]; P = pd.DataFrame(index=va.index, columns=R.T4, dtype=float); cols = FS[fs] + ['t']
        for a, b in {'early': (1, 4), 'mid': (5, 9), 'late': (10, 13)}.values():
            trr = tr[(tr.day >= a - 1) & (tr.day <= b + 1)]; m = (va.day >= a) & (va.day <= b)
            for y in R.T4: P.loc[m, y] = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.03, max_iter=400, min_samples_leaf=10).fit(trr[cols], np.log(trr[y])).predict(va.loc[m, cols])
        out.append(P)
    pd.concat(out).sort_index().to_csv(f'{OUT}/{variant}.csv')
print('done', variant)

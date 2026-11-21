
"""v4 = per-target log-space blend of: v3, similar-batch kNN (k=5/8), ridge with upcoming-feed inputs, ridge with upcoming feed_substrate inputs, per-phase boosting with feed_c windows."""
import sys, json, numpy as np, pandas as pd, joblib, time
sys.argv = ['x', 'none', 'none']
src = open('/home/renku/work/DTB_outputs/exp_target.py').read(); exec(src.rsplit("mode, fs = sys.argv[1], sys.argv[2]", 1)[0])   # Zx, FSX, TS, R
from sklearn.ensemble import HistGradientBoostingRegressor
W = json.load(open('/home/renku/work/DTB_outputs/target_results/v4_weights.json'))
tr = Zx[Zx.batch.isin(R.TRAIN)]; te = Zx[Zx.batch.isin(R.TEST)]; t0 = time.time()
def ridge_fit_predict(cols, rows):
    P = pd.DataFrame(index=rows.index, columns=R.T4, dtype=float); models = {}
    for d in sorted(rows.day.unique()):
        trd = tr[(tr.day - d).abs() <= 1]; w = np.where(trd.day == d, 1.0, 0.5); m = rows.day.values == d
        for y in R.T4:
            mod = R._fit(R.ridge(), trd[cols + ['t']], np.log(trd[y]), w); models[(d, y)] = mod; P.loc[m, y] = mod.predict(rows.loc[m, cols + ['t']])
    return P, models
def hgbphase_fit_predict(cols, rows):
    PH = {'early': (1, 4), 'mid': (5, 9), 'late': (10, 13)}; P = pd.DataFrame(index=rows.index, columns=R.T4, dtype=float); models = {}
    for ph, (a, b) in PH.items():
        trr = tr[(tr.day >= a - 1) & (tr.day <= b + 1)]; m = (rows.day >= a) & (rows.day <= b)
        for y in R.T4:
            mod = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.03, max_iter=400, min_samples_leaf=10).fit(trr[cols + ['t']], np.log(trr[y])); models[(ph, y)] = mod
            if m.any(): P.loc[m, y] = mod.predict(rows.loc[m, cols + ['t']])
    return P, models
def knn_predict(k, rows):
    days = np.arange(1, 14)
    def prof(b):
        g = Zx[Zx.batch == b].sort_values('t'); v = list(g.iloc[0][R.S0].values)
        for f in R.F: v += list(np.interp(days, g.t, g[f + '_cum'] / g.V))
        return np.array(v + list(np.interp(days, g.t, g.V)), float)
    Mtr = np.array([prof(b) for b in R.TRAIN]); mu, sd = Mtr.mean(0), Mtr.std(0) + 1e-9
    P = pd.DataFrame(index=rows.index, columns=R.T4, dtype=float)
    for b in rows.batch.unique():
        dist = np.sqrt((((Mtr - prof(b)) / sd) ** 2).mean(1)); nn = np.argsort(dist)[:k]; w = 1 / (dist[nn] + 1e-3)
        for i in rows.index[rows.batch == b]:
            d = rows.at[i, 'day']
            for y in R.T4:
                vals, ws = [], []
                for j, bb in enumerate(np.array(R.TRAIN)[nn]):
                    r = tr[(tr.batch == bb) & (tr.day == d)]
                    if len(r): vals.append(np.log(r[y].values[0])); ws.append(w[j])
                P.at[i, y] = np.average(vals, weights=ws)
    return P
v3 = joblib.load('/home/renku/work/DTB_outputs/v3_models.joblib')
C = {'v3': np.log(v3['test_predictions'].loc[te.index])}
C['knn5'] = knn_predict(5, te); C['knn8'] = knn_predict(8, te)
C['ridge_fut'], m_rf = ridge_fit_predict(FSX['A4fut'], te)
C['t_ridge_sub_futS'], m_rs = ridge_fit_predict(TS['sub_futS'], te)
C['t_hgbphase_byp_Cfut'], m_hc = hgbphase_fit_predict(TS['byp_Cfut'], te)
pred = pd.DataFrame(index=te.index, columns=R.T4, dtype=float)
for y in R.T4: pred[y] = np.exp(sum(w * C[n][y].values for n, w in W[y]))
out = te[['batch', 't']].join(pred)
joblib.dump({'weights': W, 'test_predictions': out, 'components': {k: v for k, v in C.items()}, 'ridge_fut_models': m_rf, 'ridge_sub_futS_models': m_rs, 'hgbphase_byp_Cfut_models': m_hc,
             'cv_estimate_nested': {'vcd': 6.82, 'tcd': 6.63, 'substrate': 7.88, 'byproduct': 9.86, 'mean': 7.80}}, '/home/renku/work/DTB_outputs/v4_models.joblib')
p3 = v3['test_predictions'].loc[te.index]; rel = ((pred - p3) / p3 * 100)
print('trained in %.0fs | weights' % (time.time() - t0), W)
print('test rows %d finite %s tcd>=vcd %s' % (len(pred), bool(np.isfinite(pred.values).all()), bool((pred.tcd >= pred.vcd).all())))
print('ranges', {y: (round(pred[y].min(), 2), round(pred[y].max(), 2)) for y in R.T4})
print('v4 vs v3 mean |change| %', rel.abs().mean().round(2).to_dict(), 'max', rel.abs().max().round(1).to_dict())

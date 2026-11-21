
"""v3: 0.7 x mean(ridge over 6 day-borrowing settings x 2 input sets) + 0.3 x mean(per-phase boosting, margin +-1 day, 3 settings x 2 input sets), in log space."""
import sys, numpy as np, pandas as pd, joblib, time
sys.path.insert(0, '/home/renku/work/DTB_outputs'); import dtb_regression as R
from sklearn.ensemble import HistGradientBoostingRegressor
ORDER = ['feed_substrate', 'feed_b', 'feed_c', 'feed_d']
FS = {'A3': R.S0 + R.VOL + sum([R.FEEDX(f) for f in ORDER[:3]], []), 'A4': R.S0 + R.VOL + sum([R.FEEDX(f) for f in ORDER], [])}
RIDGE_SET = [(0, 1.0), (1, 0.25), (1, 0.5), (1, 1.0), (2, 0.25), (2, 0.5)]
HGB_SET = [(3, 0.05, 300), (2, 0.05, 400), (4, 0.03, 400)]
PHASES = {'early': (1, 4), 'mid': (5, 9), 'late': (10, 13)}; MARGIN = 1; W_RIDGE = 0.7
tr = R.Z[R.Z.batch.isin(R.TRAIN)]; DAYS = sorted(R.Z.day.unique())
t0 = time.time(); ridge_models = {}; hgb_models = {}
for fs, cols in FS.items():
    for pool, wnb in RIDGE_SET:
        for d in DAYS:
            trd = tr[(tr.day - d).abs() <= pool]; w = np.where(trd.day == d, 1.0, wnb)
            for y in R.T4: ridge_models[(fs, pool, wnb, d, y)] = R._fit(R.ridge(), trd[cols + ['t']], np.log(trd[y]), w)
    for depth, lr, it in HGB_SET:
        for ph, (a, b) in PHASES.items():
            trr = tr[(tr.day >= a - MARGIN) & (tr.day <= b + MARGIN)]
            for y in R.T4:
                hgb_models[(fs, depth, lr, it, ph, y)] = HistGradientBoostingRegressor(max_depth=depth, learning_rate=lr, max_iter=it, min_samples_leaf=10).fit(trr[cols + ['t']], np.log(trr[y]))
def phase_of(d): return next(p for p, (a, b) in PHASES.items() if a <= d <= b)
def predict_v3(rows):
    out = pd.DataFrame(index=rows.index, columns=R.T4, dtype=float)
    for y in R.T4:
        lr_ = np.zeros(len(rows)); lh = np.zeros(len(rows))
        for fs, cols in FS.items():
            X = rows[cols + ['t']]
            for pool, wnb in RIDGE_SET:
                for d in rows.day.unique():
                    m = rows.day.values == d; lr_[m] += ridge_models[(fs, pool, wnb, d, y)].predict(X[m])
            for depth, lrate, it in HGB_SET:
                for ph in PHASES:
                    m = np.array([phase_of(d) == ph for d in rows.day.values])
                    if m.any(): lh[m] += hgb_models[(fs, depth, lrate, it, ph, y)].predict(X[m])
        lr_ /= len(FS) * len(RIDGE_SET); lh /= len(FS) * len(HGB_SET)
        out[y] = np.exp(W_RIDGE * lr_ + (1 - W_RIDGE) * lh)
    return out
te = R.Z[R.Z.batch.isin(R.TEST)]
p_tr = predict_v3(tr); p_te = predict_v3(te)
ins = (np.abs(p_tr.values - tr[R.T4].values) / tr[R.T4].values).mean(0) * 100
joblib.dump({'ridge_models': ridge_models, 'hgb_models': hgb_models, 'FS': FS, 'RIDGE_SET': RIDGE_SET, 'HGB_SET': HGB_SET, 'PHASES': PHASES,
             'MARGIN': MARGIN, 'W_RIDGE': W_RIDGE, 'test_predictions': p_te, 'cv_estimate': {'vcd': 6.95, 'tcd': 6.89, 'substrate': 9.13, 'byproduct': 10.15, 'mean': 8.28}},
            '/home/renku/work/DTB_outputs/v3_models.joblib')
v1 = pd.read_csv('/home/renku/work/DTB_outputs/bioreaction_submission.csv')
te2 = te.copy(); te2[R.T4] = p_te[R.T4].values
mrg = v1.merge(te2[['batch', 't'] + R.T4].rename(columns={'batch': 'batch_id'}), on='batch_id', suffixes=('_v1', '_v3'))
mrg = mrg[(mrg.elapsed_time - mrg.t).abs() < 1e-6]
rel = {y: ((mrg[y + '_v3'] - mrg[y + '_v1']) / mrg[y + '_v1'] * 100) for y in R.T4}
print('models: %d ridge + %d boosting, trained in %.0fs' % (len(ridge_models), len(hgb_models), time.time() - t0))
print('in-sample MAPE', np.round(ins, 2), 'mean %.2f' % ins.mean())
print('test rows %d, finite %s, tcd>=vcd %s' % (len(p_te), bool(np.isfinite(p_te.values).all()), bool((p_te.tcd >= p_te.vcd).all())))
print('test ranges', {y: (round(p_te[y].min(), 2), round(p_te[y].max(), 2)) for y in R.T4})
print('v3 vs v1 on matched test rows (n=%d): mean |change| %%' % len(mrg), {y: round(float(rel[y].abs().mean()), 2) for y in R.T4}, '| max |change| %', {y: round(float(rel[y].abs().max()), 1) for y in R.T4})

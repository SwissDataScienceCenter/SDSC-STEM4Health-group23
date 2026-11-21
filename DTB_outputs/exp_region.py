
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, '/home/renku/work/DTB_outputs'); import dtb_regression as R
from sklearn.ensemble import HistGradientBoostingRegressor
OUT = '/home/renku/work/DTB_outputs/region_results'; os.makedirs(OUT, exist_ok=True)
ORDER = ['feed_substrate', 'feed_b', 'feed_c', 'feed_d']
FS = {'A3': R.S0 + R.VOL + sum([R.FEEDX(f) for f in ORDER[:3]], []), 'A4': R.S0 + R.VOL + sum([R.FEEDX(f) for f in ORDER], [])}
REG = {'early': (1, 4), 'mid': (5, 9), 'late': (10, 13)}
fs, depth, lr, it, margin = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
out = []
for fold in R.FOLDS:
    tr = R.Z[R.Z.batch.isin(set(R.TRAIN) - set(fold))]; va = R.Z[R.Z.batch.isin(fold)]; P = pd.DataFrame(index=va.index, columns=R.T4, dtype=float)
    for rg, (a, b) in REG.items():
        trr = tr[(tr.day >= a - margin) & (tr.day <= b + margin)]; m_va = (va.day >= a) & (va.day <= b)
        for y in R.T4:
            m = HistGradientBoostingRegressor(max_depth=depth, learning_rate=lr, max_iter=it, min_samples_leaf=10).fit(trr[FS[fs] + ['t']], np.log(trr[y]))
            P.loc[m_va, y] = m.predict(va.loc[m_va, FS[fs] + ['t']])
    out.append(P)
pd.concat(out).sort_index().to_csv(f'{OUT}/hgbreg_{fs}_d{depth}_lr{lr}_it{it}_m{margin}.csv'); print('done', sys.argv[1:])

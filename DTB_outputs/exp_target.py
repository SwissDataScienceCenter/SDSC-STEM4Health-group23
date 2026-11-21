
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, '/home/renku/work/DTB_outputs')
src = open('/home/renku/work/DTB_outputs/exp_sensor.py').read(); exec(src.split("mode, fs = sys.argv[1], sys.argv[2]")[0])   # builds Zx (sensor features) and sets R.Z = Zx
from sklearn.ensemble import HistGradientBoostingRegressor
OUT = '/home/renku/work/DTB_outputs/target_results'; os.makedirs(OUT, exist_ok=True)
for w in PAST: Zx[f'feed_c_past{w}_perEvent'] = Zx[f'feed_c_past{w}_amtV'] / Zx[f'feed_c_past{w}_n'].clip(lower=1)
for w in FUT:  Zx[f'feed_c_fut{w}_perEvent'] = Zx[f'feed_c_fut{w}_amtV'] / Zx[f'feed_c_fut{w}_n'].clip(lower=1)
Zx['feed_c_cumV'] = Zx.feed_c_cum / Zx.V; Zx['feed_c_meanEvent'] = Zx.feed_c_cum / Zx.feed_c_n.clip(lower=1)
R.Z = Zx
pick = lambda feeds, kinds: [c for c in Zx.columns if any(c.startswith(f + '_') for f in feeds) and any(k in c for k in kinds)]
TS = {'sub_futSB': A4 + pick(['feed_substrate', 'feed_b'], ['fut', 'to_next']),
      'sub_futS':  A4 + pick(['feed_substrate'], ['fut', 'to_next']),
      'byp_C':     A4 + pick(['feed_c'], ['past', 'fut', 'to_next', 'since_last', 'perEvent', 'cumV', 'meanEvent']),
      'byp_Cpast': A4 + pick(['feed_c'], ['past', 'since_last', 'cumV', 'meanEvent']),
      'byp_Cfut':  A4 + pick(['feed_c'], ['fut', 'to_next', 'cumV', 'meanEvent'])}
mode, fs = sys.argv[1], sys.argv[2]
if mode == 'ridge':
    R.cvp(TS[fs], pool=1, wnb=0.5).to_csv(f'{OUT}/ridge_{fs}.csv')
elif mode == 'hgbphase':
    PH = {'early': (1, 4), 'mid': (5, 9), 'late': (10, 13)}; out = []
    for fold in R.FOLDS:
        tr = Zx[Zx.batch.isin(set(R.TRAIN) - set(fold))]; va = Zx[Zx.batch.isin(fold)]; P = pd.DataFrame(index=va.index, columns=R.T4, dtype=float); cols = TS[fs] + ['t']
        for a, b in PH.values():
            trr = tr[(tr.day >= a - 1) & (tr.day <= b + 1)]; m = (va.day >= a) & (va.day <= b)
            for y in R.T4: P.loc[m, y] = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.03, max_iter=400, min_samples_leaf=10).fit(trr[cols], np.log(trr[y])).predict(va.loc[m, cols])
        out.append(P)
    pd.concat(out).sort_index().to_csv(f'{OUT}/hgbphase_{fs}.csv')
print('done', sys.argv[1:])

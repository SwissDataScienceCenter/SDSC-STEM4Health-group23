
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, '/home/renku/work/DTB_outputs'); import dtb_regression as R
from sklearn.ensemble import HistGradientBoostingRegressor
OUT = '/home/renku/work/DTB_outputs/tune_results'; os.makedirs(OUT, exist_ok=True)
ORDER = ['feed_substrate', 'feed_b', 'feed_c', 'feed_d']
FS = {'A2': R.S0 + R.VOL + R.FEEDX('feed_substrate') + R.FEEDX('feed_b'),
      'A3': R.S0 + R.VOL + sum([R.FEEDX(f) for f in ORDER[:3]], []),
      'A4': R.S0 + R.VOL + sum([R.FEEDX(f) for f in ORDER], [])}
kind, fs = sys.argv[1], sys.argv[2]
if kind == 'ridge':
    pool, wnb = int(sys.argv[3]), float(sys.argv[4])
    LP = R.cvp(FS[fs], pool=pool, wnb=wnb); LP.to_csv(f'{OUT}/ridge_{fs}_p{pool}_w{wnb}.csv')
else:
    depth, lr, it = int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
    mk = lambda: HistGradientBoostingRegressor(max_depth=depth, learning_rate=lr, max_iter=it, min_samples_leaf=10)
    LP = R.cvp(FS[fs], mk=mk, pool=14, wnb=1.0); LP.to_csv(f'{OUT}/hgb_{fs}_d{depth}_lr{lr}_it{it}.csv')
print('done', sys.argv[1:])

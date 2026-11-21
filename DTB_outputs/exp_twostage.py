
import sys, os, numpy as np, pandas as pd
variant = sys.argv[1]; sys.argv = ['x', 'none', 'none']
src = open('/home/renku/work/DTB_outputs/exp_target.py').read(); exec(src.rsplit("mode, fs = sys.argv[1], sys.argv[2]", 1)[0])   # Zx, FSX, TS, R
from sklearn.ensemble import HistGradientBoostingRegressor
OUT = '/home/renku/work/DTB_outputs/twostage_results'; os.makedirs(OUT, exist_ok=True)
S1COLS = TS['sub_futS'] + ['t']
def stage1(train_batches, rows):
    """log-substrate prediction for 'rows' from a per-day ridge (+-1 day, half weight) trained on train_batches"""
    tr = Zx[Zx.batch.isin(train_batches)]; out = pd.Series(index=rows.index, dtype=float)
    for d in sorted(rows.day.unique()):
        trd = tr[(tr.day - d).abs() <= 1]; w = np.where(trd.day == d, 1.0, 0.5); m = rows.day.values == d
        out[rows.index[m]] = R._fit(R.ridge(), trd[S1COLS], np.log(trd.substrate), w).predict(rows.loc[m, S1COLS])
    return out
def s2_features(rows, shat):
    """features from each batch's predicted substrate curve"""
    f = pd.DataFrame(index=rows.index)
    for b in rows.batch.unique():
        idx = rows.index[rows.batch == b]; g = rows.loc[idx].sort_values('t'); s = shat[g.index]; S = np.exp(s.values)
        f.loc[g.index, 'shat'] = s.values
        f.loc[g.index, 'shat_prev'] = np.r_[s.values[0], s.values[:-1]]
        f.loc[g.index, 'shat_mean_sofar'] = np.cumsum(S) / np.arange(1, len(S) + 1)
        f.loc[g.index, 'shat_min_sofar'] = np.minimum.accumulate(S)
        f.loc[g.index, 'shat_low_days_sofar'] = np.cumsum(S < 1.0)
        f.loc[g.index, 'shat_drop'] = np.r_[0, np.diff(s.values)]
        f.loc[g.index, 'shat_min_all'] = S.min(); f.loc[g.index, 'shat_mean_all'] = S.mean()
    return f
S2 = ['shat', 'shat_prev', 'shat_mean_sofar', 'shat_min_sofar', 'shat_low_days_sofar', 'shat_drop', 'shat_min_all', 'shat_mean_all']
BASECOLS = {'A4': A4, 'Cfut': TS['byp_Cfut']}[variant.split('_')[1]]
model = variant.split('_')[0]
out = []
for fold in R.FOLDS:
    trb = [b for b in R.TRAIN if b not in set(fold)]
    tr = Zx[Zx.batch.isin(trb)].copy(); va = Zx[Zx.batch.isin(fold)].copy()
    # stage 1 for training batches: inner 4-fold so they get "honest" predicted curves
    sh_tr = pd.Series(index=tr.index, dtype=float)
    for inner in np.array_split(np.random.default_rng(1).permutation(trb), 4):
        rows = tr[tr.batch.isin(inner)]; sh_tr[rows.index] = stage1([b for b in trb if b not in set(inner)], rows)
    sh_va = stage1(trb, va)
    tr = tr.join(s2_features(tr, sh_tr)); va = va.join(s2_features(va, sh_va))
    cols = BASECOLS + S2 + ['t']; P = pd.DataFrame(index=va.index, columns=['byproduct', 'substrate_stage1'], dtype=float); P['substrate_stage1'] = sh_va
    if model == 'ridge':
        for d in sorted(va.day.unique()):
            trd = tr[(tr.day - d).abs() <= 1]; w = np.where(trd.day == d, 1.0, 0.5); m = va.day.values == d
            P.loc[va.index[m], 'byproduct'] = R._fit(R.ridge(), trd[cols], np.log(trd.byproduct), w).predict(va.loc[m, cols])
    else:
        for a, b in {'early': (1, 4), 'mid': (5, 9), 'late': (10, 13)}.values():
            trr = tr[(tr.day >= a - 1) & (tr.day <= b + 1)]; m = (va.day >= a) & (va.day <= b)
            P.loc[m, 'byproduct'] = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.03, max_iter=400, min_samples_leaf=10).fit(trr[cols], np.log(trr.byproduct)).predict(va.loc[m, cols])
    out.append(P)
pd.concat(out).sort_index().to_csv(f'{OUT}/{variant}.csv'); print('done', variant)

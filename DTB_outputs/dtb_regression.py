
"""DTB per-day regression pipeline (Team 23). Rebuilds features, CV folds, ridge(+-1 day, half weight) + boosting blend."""
import numpy as np, pandas as pd, warnings, json, time, sys
warnings.filterwarnings('ignore')
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
SRC = '/home/renku/work/input-data/Project 1 - DTB/bioreaction_remaining.csv'
F = ['feed_substrate', 'feed_b', 'feed_c', 'feed_d']; T4 = ['vcd', 'tcd', 'substrate', 'byproduct']
df = pd.read_csv(SRC); df['to_predict'] = df.to_predict.astype(str).str.lower().eq('true')
TEST = sorted(set(df.loc[df.to_predict, 'batch_id']), key=lambda s: int(s[5:]))
ALL = sorted(df.batch_id.unique(), key=lambda s: int(s[5:])); TRAIN = [b for b in ALL if b not in TEST]
FOLDS = np.array_split(np.random.default_rng(0).permutation(TRAIN), 5)
def build_Z():
    rows = []
    for b in ALL:
        g = df[df.batch_id == b]; a = g[g.vcd.notna()].iloc[0]; vv = g[g.volume.notna()]
        for _, r in g[(g.vcd.notna() | g.to_predict) & (g.elapsed_time > 0.01)].iterrows():
            t = r.elapsed_time; h = g[g.elapsed_time <= t + 1e-9]
            x = dict(batch=b, day=round(t), t=t, lv0=np.log(a.vcd), ld0=np.log(a.tcd - a.vcd + 1e-3), ls0=np.log(a.substrate), lp0=np.log(a.byproduct),
                     lV0=np.log(a.volume), V=np.interp(t, vv.elapsed_time, vv.volume))
            for f in F: x[f + '_cum'] = h[f].sum(); x[f + '_n'] = h[f].notna().sum()
            for y in T4: x[y] = r[y]
            rows.append(x)
    return pd.DataFrame(rows)
Z = build_Z()
S0 = ['lv0', 'ld0', 'ls0', 'lp0']; VOL = ['lV0', 'V']; FEEDX = lambda f: [f + '_cum', f + '_n']
ridge = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 30)))
hgb = lambda: HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=10)
def _fit(m, X, y, w):
    if hasattr(m, 'steps'): m.fit(X, y, ridgecv__sample_weight=w)
    else: m.fit(X, y, sample_weight=w)
    return m
def cvp(xcols, mk=ridge, pool=1, wnb=0.5, target=None, offset=None):
    """out-of-fold predictions; if offset (log-pred DataFrame) is given, the model learns the residual ln y - offset"""
    out = []; tg = target or T4
    for fold in FOLDS:
        tr = Z[Z.batch.isin(set(TRAIN) - set(fold))]; va = Z[Z.batch.isin(fold)]; P = np.zeros((len(va), len(tg)))
        for d in sorted(va.day.unique()):
            trd = tr[(tr.day - d).abs() <= pool]; vad = va.day.values == d; cols = xcols + ['t']; w = np.where(trd.day == d, 1.0, wnb)
            for k, y in enumerate(tg):
                yy = np.log(trd[y]) - (offset.loc[trd.index, y] if offset is not None else 0)
                m = _fit(mk(), trd[cols], yy, w); P[vad, k] = m.predict(va.loc[vad, cols]) + (offset.loc[va.index[vad], y].values if offset is not None else 0)
        out.append(pd.DataFrame(P, index=va.index, columns=tg))
    return pd.concat(out).sort_index()   # LOG predictions
def score(LP):
    Y = Z.loc[LP.index, T4]; return (np.abs(np.exp(LP[T4].values) - Y.values) / Y.values).mean(0) * 100
def blend_cv(xcols): return 0.7 * cvp(xcols) + 0.3 * cvp(xcols, mk=hgb, pool=14, wnb=1.0)


import sys, json, os, time, numpy as np, pandas as pd
sys.path.insert(0, '/home/renku/work/DTB_outputs'); import dtb_regression as R
OUT = '/home/renku/work/DTB_outputs/stepwise_results'; os.makedirs(OUT, exist_ok=True)
ORDER = ['feed_substrate', 'feed_b', 'feed_c', 'feed_d']; BASE = R.S0 + R.VOL
mode = sys.argv[1]
def save(name, LP, extra=None):
    s = R.score(LP); d = {'name': name, 'mape': dict(zip(R.T4, np.round(s, 3).tolist())), 'mean': round(float(s.mean()), 3)}
    if extra: d.update(extra)
    json.dump(d, open(f'{OUT}/{name}.json', 'w')); LP.to_csv(f'{OUT}/{name}_oof_logpred.csv')
if mode == 'A':      # growing input set, full blend refit at each step
    k = int(sys.argv[2]); cols = BASE + sum([R.FEEDX(f) for f in ORDER[:k]], [])
    save('A%d_%s' % (k, 'state+volume' if k == 0 else '+'.join(['base'] + ORDER[:k])), R.blend_cv(cols))
if mode == 'B':      # chained: stage 0 = blend on state+volume, stage k = ridge on residual with ONLY feed k (fit inside each fold, no leakage)
    def fitpred(tr, va, cols, mk, pool, wnb, off_tr, off_va):
        Ptr = pd.DataFrame(0.0, index=tr.index, columns=R.T4); Pva = pd.DataFrame(0.0, index=va.index, columns=R.T4)
        for d in sorted(set(tr.day) | set(va.day)):
            trd = tr[(tr.day - d).abs() <= pool]; w = np.where(trd.day == d, 1.0, wnb); mtr = tr.day.values == d; mva = va.day.values == d
            for y in R.T4:
                m = R._fit(mk(), trd[cols + ['t']], np.log(trd[y]) - off_tr.loc[trd.index, y], w)
                if mtr.any(): Ptr.loc[tr.index[mtr], y] = m.predict(tr.loc[mtr, cols + ['t']]) + off_tr.loc[tr.index[mtr], y].values
                if mva.any(): Pva.loc[va.index[mva], y] = m.predict(va.loc[mva, cols + ['t']]) + off_va.loc[va.index[mva], y].values
        return Ptr, Pva
    stages = {k: [] for k in range(5)}
    for fold in R.FOLDS:
        tr = R.Z[R.Z.batch.isin(set(R.TRAIN) - set(fold))]; va = R.Z[R.Z.batch.isin(fold)]
        z_tr = pd.DataFrame(0.0, index=tr.index, columns=R.T4); z_va = pd.DataFrame(0.0, index=va.index, columns=R.T4)
        r_tr, r_va = fitpred(tr, va, BASE, R.ridge, 1, 0.5, z_tr, z_va)
        h_tr, h_va = fitpred(tr, va, BASE, R.hgb, 14, 1.0, z_tr, z_va)
        o_tr, o_va = 0.7 * r_tr + 0.3 * h_tr, 0.7 * r_va + 0.3 * h_va; stages[0].append(o_va)
        for k, f in enumerate(ORDER, start=1):
            o_tr, o_va = fitpred(tr, va, R.FEEDX(f), R.ridge, 1, 0.5, o_tr, o_va); stages[k].append(o_va)
    for k in range(5):
        save('B%d_chain_%s' % (k, 'state+volume' if k == 0 else '+'.join(ORDER[:k])), pd.concat(stages[k]).sort_index())
print('done', mode, sys.argv[2:] if len(sys.argv) > 2 else '')

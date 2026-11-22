"""Team 23 v14 research candidate: conservative byproduct plus compact RBF-SVR with Feed-C counts: friend's v9-SVR model remains the base.
VCD/TCD use a compact, stronger-regularized SVR and sparse feed-C count; substrate stays unchanged; byproduct retains the guarded v13 head.
No guarantee of improving the hidden leaderboard; choose using group-CV only.
"""
"""
Decode the Bioreactor - v9 + RBF-SVR (Team 23)
Blend v9 and RBF-SVR equally for VCD/TCD; preserve the original substrate; byproduct-specific robust fused ridge upgrade.

Original v9 notes:
v9 = ChatGPT v8 (Huber-IRLS fused ridge, un_next, alpha 3 / lam 10000 for cells)
     + test inputs clipped to the training range for substrate/byproduct linear models
     + soft limit on extreme vcd/tcd deviations (tau=0.3)
     + boosting share for vcd/tcd 0.25 -> 0.15 (trees go flat outside the training range)
CV (mean MAPE): random 5-fold  v7 6.874/6.820  gpt-v8 6.838/6.766  v9 6.838/6.771
                extreme-batch  v7 7.255        gpt-v8 7.129        v9 6.947
Original v7 reference, extended with robust fused ridge, regularization tuning,
and a forward residual-volume signal for byproduct.

v7 vs v6: vcd/tcd/byproduct linear part is a "fused" ridge: one linear model per sampling day, but the
coefficients of neighbouring days are pulled together (penalty lam*||b_k - b_{k+1}||^2). Batch effects persist
over the run, so sharing strength across days cuts estimation noise (49 batches only). Optional GP member for
byproduct. Substrate covariance is Ledoit-Wolf shrunk, boosting averaged over 3 seeds.

What changed vs v5 (5-fold x3 CV: 7.66 -> 6.99):
  * the process control system leaks state information into the inputs that are given for test batches:
      - substrate feed controller: on days 7-9 a 0.4%-of-volume substrate shot is given only if
        measured substrate < ~1.95; on days 11 and 12 the shot is 0.6% of volume if substrate < ~0.965,
        otherwise 0.4%.  -> we know on which side of the threshold the substrate was.
      - volume rises more than the recorded feeds explain on days 1-5 (base added for pH control while
        byproduct is produced). This "unexplained volume" tracks byproduct production.
      - feed_b amount (relative to volume) has a batch-specific deviation from the usual schedule.
  * substrate: ridge prediction is conditioned on the controller decisions (truncated Gaussian on the
    log residuals, covariance from an inner CV), then the MAPE-optimal point is taken.
  * byproduct / vcd / tcd: the two day-13 byproduct jumps (Batch29, Batch34) and two day-13 vcd jumps are
    left out of fitting; new features; gradient boosting blend.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import Ridge

RENKU = Path('/home/renku/work/team-workspace')
BASE_DIR = RENKU if (RENKU / 'bioreaction_remaining.csv').exists() else Path(__file__).resolve().parent
DATA = str(BASE_DIR / 'bioreaction_remaining.csv')
TEMPLATE = str(BASE_DIR / 'bioreaction_template.csv')
OUTPUT = str(BASE_DIR / 'TEAM23_submission_v14_metabolites.csv')

TARGETS = ["vcd", "tcd", "substrate", "byproduct"]
FEEDS = ["feed_substrate", "feed_b", "feed_c", "feed_d"]
C_FEED = np.array([197.0, 70.4, 0.0, 38.2])   # substrate per feed volume, from the mechanistic fit
S_TRIG, S_LOW = 1.95, 0.965                    # substrate controller thresholds (read off the training data)

BASE = ["y0", "v0", "dvol", "t"]
CLIP = [False, False, True, True]   # per target: clip test inputs to the training range before the linear models
SHRINK_TAU = 0.3                    # soft limit on vcd/tcd log-deviation from the population curve
CFG = dict(
    feats=[BASE + ["cum_bd", "snc_tile", "ly0"],
           BASE + ["cum_bd", "snc_tile", "ly0"],
           ["y0", "dvol", "t", "since", "lsnc", "snc", "snc_tile", "un_cum", "low", "fb_now", "un_sum"],
           BASE + ["cum", "since", "snc_tile", "fb_now", "un_early", "nsmall", "un_next"]],
    alpha=[3, 3, 10, 3],
    fuse=[(10000, False), (10000, False), None, (30, True)],   # (lam, per-day scaling) or None = independent days
    rw=[1.0, 1.0, 1.0, 1.0],                 # ridge weight vs weighted median of training curves
    gfeats=[BASE, BASE, None, BASE + ["un_sum", "fbf"]],
    gmix=[0.15, 0.15, 0.0, 0.4],             # boosting share
    drop=[True, True, False, True],
    robust_delta=[0.04, 0.07, None, 0.15],          # leave one-off end jumps out of fitting
)

# Only the byproduct head is changed; keep VCD, TCD and substrate intact.
BYPRODUCT_FEATS = [nm for nm in CFG["feats"][3] if nm not in ("cum", "snc_tile", "un_next")] + ["sub_ctrl", "fb_signal", "u_fut3"]
BYPRODUCT_INITIAL_POWER = 3.0
BYPRODUCT_BOOST_SHARE = 0.30



# ---------------------------------------------------------------- data
def prepare(path):
    d = pd.read_csv(path)
    test_ids = set(d.loc[d.to_predict, "batch_id"])
    batches = []
    for bid, g in d.groupby("batch_id", sort=False):
        g = g.sort_values("elapsed_time")
        m = g.dropna(subset=TARGETS)
        q = g[g.to_predict] if bid in test_ids else m.iloc[1:]
        t = q.elapsed_time.to_numpy()
        init = m.iloc[0]
        cum = np.array([[g.loc[g[f].notna() & (g.elapsed_time < x), f].sum() for f in FEEDS] for x in t])
        fs = g[g.feed_substrate.notna()]
        sub_times = fs.elapsed_time.to_numpy()
        since = np.array([x - sub_times[sub_times < x].max() if (sub_times < x).any() else 15.0 for x in t])
        y0 = init[TARGETS].to_numpy(float)
        v0 = float(init.volume)

        vt = g.dropna(subset=["volume"])
        vgrid, vvals = vt.elapsed_time.to_numpy(), vt.volume.to_numpy()
        vol_at = lambda x: np.interp(x, vgrid, vvals)

        # substrate controller decisions
        early = fs[fs.elapsed_time < 10.5]
        trig_day = int(np.floor(early.elapsed_time.iloc[0])) if len(early) else 0

        def low_flag(lo, hi):
            e = fs[(fs.elapsed_time > lo) & (fs.elapsed_time < hi)]
            if len(e) == 0:
                return np.nan
            return float(e.feed_substrate.iloc[0] / vol_at(e.elapsed_time.iloc[0]) > 0.005)

        # feed_b relative to volume, per sampling interval
        fb = g[g.feed_b.notna()]
        fb_day = np.searchsorted(vgrid, fb.elapsed_time.to_numpy(), "right") - 1
        fb_rate = fb.feed_b.to_numpy() / vol_at(fb.elapsed_time.to_numpy())

        # irregular small feed_c events
        fc = g[g.feed_c.notna()]
        small_t = fc.loc[fc.feed_c < 0.0013, "elapsed_time"].to_numpy()

        # volume change not explained by recorded feeds, per interval
        allfeed = g[FEEDS].fillna(0).sum(1).to_numpy()
        et = g.elapsed_time.to_numpy()
        un = np.array([vvals[i + 1] - vvals[i] - allfeed[(et >= vgrid[i]) & (et < vgrid[i + 1])].sum()
                       for i in range(len(vgrid) - 1)])

        batches.append(dict(
            id=bid, test=bid in test_ids, t=t, y0=y0, v0=v0, vol=q.volume.to_numpy(), cum=cum, since=since,
            snc=(y0[2] * v0 + (cum * C_FEED).sum(1)) / q.volume.to_numpy(),
            trig_day=trig_day, low11=low_flag(11, 12), low12=low_flag(12, 13),
            fb_day=fb_day, fb_rate=fb_rate, small_t=small_t, un=un, vgrid=vgrid,
            y=None if bid in test_ids else q[TARGETS].to_numpy(float),
            ref_t=m.elapsed_time.to_numpy(), ref_y=m[TARGETS].to_numpy(float)))

    # feed_b deviation from the usual schedule (log ratio to the median over all batches)
    med = {k: np.median([b["fb_rate"][b["fb_day"] == k][0] for b in batches if (b["fb_day"] == k).any()])
           for k in range(1, 13)}
    for b in batches:
        dev = np.full(13, np.nan)
        for k in range(1, 13):
            r = b["fb_rate"][b["fb_day"] == k]
            if len(r):
                dev[k] = np.log(r[0] / med[k])
        b["fb_dev"] = dev
        b["blk"] = feature_blocks(b)
    return batches


def feature_blocks(b):
    n = len(b["t"])
    tile = lambda v: np.tile(np.atleast_1d(v), (n, 1))
    t = b["t"]
    k = np.round(t).astype(int)
    dev, un = b["fb_dev"], b["un"]
    l11 = 0.5 if np.isnan(b["low11"]) else b["low11"]
    l12 = 0.5 if np.isnan(b["low12"]) else b["low12"]
    return dict(
        y0=tile(b["y0"]), ly0=tile(np.log(b["y0"])), v0=tile(b["v0"]),
        dvol=(b["vol"] - b["v0"])[:, None], t=t[:, None],
        cum=b["cum"], cum_bd=b["cum"][:, [1, 3]], since=b["since"][:, None],
        snc=b["snc"][:, None], lsnc=np.log(b["snc"])[:, None], snc_tile=tile(b["snc"]),
        low=tile(np.array([l11, l12])),
        fbf=tile(np.array([np.nanmean(dev[1:6]), np.nanmean(dev[6:13]), np.nanmean(dev[1:13])])),
        fb_now=np.array([dev[kk] if 1 <= kk <= 12 else np.nanmean(dev[8:13]) for kk in k])[:, None],
        un_cum=np.interp(t, b["vgrid"], np.concatenate([[0], np.cumsum(un)]))[:, None],
        un_early=tile(un[1:5]),
        un_next=np.r_[un[1:], un[-1]][:, None],
        un_sum=tile(np.array([un[1:5].sum(), un[5:].sum()])),
        nsmall=np.column_stack([[np.sum(b["small_t"] < x) for x in t], np.full(n, len(b["small_t"]))]).astype(float),
        # Byproduct-specific indicators: all are measured/exogenous, not unknown test targets.
        sub_ctrl=tile(np.array([b["trig_day"], l11, l12])),
        fb_signal=np.where(np.isfinite(dev[1:]), dev[1:], 0.0)[np.clip(k - 1, 0, 11), None],
        u_fut3=(un[np.clip(k, 0, 12)] + un[np.clip(k + 1, 0, 12)]
                + un[np.clip(k + 2, 0, 12)])[:, None] / 3,
    )


def X_of(b, names):
    return np.column_stack([b["blk"][nm] for nm in names])


# ---------------------------------------------------------------- models
def weighted_median(b, refs):
    prof = np.stack([np.column_stack([np.interp(b["t"], r["ref_t"], r["ref_y"][:, j]) for j in range(4)])
                     for r in refs])
    change = prof - np.stack([r["ref_y"][0] for r in refs])[:, None, :]
    w = 1 / np.maximum(np.abs(prof), 1e-8)
    order = np.argsort(change, 0)
    cw = np.take_along_axis(w, order, 0).cumsum(0)
    idx = (cw >= cw[-1] / 2).argmax(0)
    return b["y0"] + np.take_along_axis(np.take_along_axis(change, order, 0), idx[None], 0)[0]


def jump_mask(train, j):
    """training points that look like one-off end jumps; only excluded from fitting"""
    M = np.zeros((len(train), 13), bool)
    for i, b in enumerate(train):
        if j == 3 and b["y"][-1, 3] / b["y"][-2, 3] > 1.5:
            M[i, -1] = True
        if j in (0, 1):
            y = np.log(b["y"][:, j])
            if y[-1] - (2 * y[-2] - y[-3]) > 0.4:
                M[i, -1] = True
    return M


def ridge_per_step(train, test, j, names, alpha, drop=None):
    X = np.stack([X_of(b, names) for b in train])
    Xt = np.stack([X_of(b, names) for b in test])
    Y = np.log(np.stack([b["y"][:, j] for b in train]))
    if CLIP[j]:
        Xt = np.clip(Xt, X.min(0), X.max(0))
    P = np.zeros((len(test), X.shape[1]))
    for s in range(X.shape[1]):
        ok = np.ones(len(train), bool) if drop is None else ~drop[:, s]
        c, sd = X[ok, s].mean(0), X[ok, s].std(0)
        sd[sd < 1e-8] = 1
        model = Ridge(alpha=alpha).fit((X[ok, s] - c) / sd, Y[ok, s])
        P[:, s] = np.exp(model.predict((Xt[:, s] - c) / sd))
    return P


def fused_ridge(train, test, j, names, alpha, lam, drop=None, day_scale=False):
    """sum_k ||y_k - a_k - X_k b_k||^2 + alpha*sum_k ||b_k||^2 + lam*sum_k ||b_k - b_{k+1}||^2  (y = log target)"""
    X = np.stack([X_of(b, names) for b in train])
    Xt = np.stack([X_of(b, names) for b in test])
    Y = np.log(np.stack([b["y"][:, j] for b in train]))
    if CLIP[j]:
        Xt = np.clip(Xt, X.min(0), X.max(0))
    n, S, p = X.shape
    W = np.ones((n, S)) if drop is None else (~drop).astype(float)
    if day_scale:
        mu = (X * W[..., None]).sum(0) / W.sum(0)[:, None]
        sd = np.sqrt(((X - mu) ** 2 * W[..., None]).sum(0) / W.sum(0)[:, None])
    else:
        mu = (X * W[..., None]).sum((0, 1)) / W.sum()
        sd = np.sqrt(((X - mu) ** 2 * W[..., None]).sum((0, 1)) / W.sum())
    sd = np.where(sd < 1e-8, 1, sd)
    Z, Zt = (X - mu) / sd, (Xt - mu) / sd
    A = np.zeros((S * p, S * p)); rhs = np.zeros(S * p)
    zbar = np.zeros((S, p)); ybar = np.zeros(S)
    for s in range(S):
        w = W[:, s]
        zbar[s] = (Z[:, s] * w[:, None]).sum(0) / w.sum(); ybar[s] = (Y[:, s] * w).sum() / w.sum()
        Zc = (Z[:, s] - zbar[s]) * np.sqrt(w)[:, None]; yc = (Y[:, s] - ybar[s]) * np.sqrt(w)
        blk = slice(s * p, (s + 1) * p)
        A[blk, blk] += Zc.T @ Zc + alpha * np.eye(p); rhs[blk] = Zc.T @ yc
    I = lam * np.eye(p)
    for s in range(S - 1):
        a, b_ = slice(s * p, (s + 1) * p), slice((s + 1) * p, (s + 2) * p)
        A[a, a] += I; A[b_, b_] += I; A[a, b_] -= I; A[b_, a] -= I
    B = np.linalg.solve(A, rhs).reshape(S, p)
    return np.exp(ybar[None] + np.einsum("nsp,sp->ns", Zt - zbar[None], B))


def boosting(train, test, j, names, drop=None, seeds=(0, 1, 2)):
    X = np.concatenate([X_of(b, names) for b in train])
    Xt = np.concatenate([X_of(b, names) for b in test])
    Y = np.log(np.concatenate([b["y"][:, j] for b in train]))
    ok = np.ones(len(Y), bool) if drop is None else ~drop.ravel()
    mu, sd = X.mean(0), X.std(0) + 1e-9
    preds = []
    for seed in seeds:   # average a few seeds, steadier predictions
        model = GradientBoostingRegressor(n_estimators=400, learning_rate=0.03, max_depth=3,
                                          subsample=0.8, random_state=seed)
        model.fit(((X - mu) / sd)[ok], Y[ok])
        preds.append(model.predict((Xt - mu) / sd))
    return np.exp(np.mean(preds, 0)).reshape(len(test), -1)


# Huber IRLS: downweight large log-errors while keeping temporal coefficient fusion.
def robust_fused(train, test, j, names, alpha, lam, delta, iterations=4, day_scale=False, clip_inputs=None):
 X=np.stack([X_of(b,names) for b in train]);Xt=np.stack([X_of(b,names) for b in test]);Y=np.log(np.stack([b['y'][:,j] for b in train]));n,S,k=X.shape
 if CLIP[j] and clip_inputs is not False:
  Xt=np.clip(Xt,X.min(0),X.max(0))
 M=jump_mask(train,j)
 W=(~M).astype(float)
 if day_scale:
  mu=(X*W[:,:,None]).sum(0)/W.sum(0)[:,None]
  sd=np.sqrt(((X-mu)**2*W[:,:,None]).sum(0)/W.sum(0)[:,None])
 else:
  mu=(X*W[:,:,None]).sum((0,1))/W.sum();sd=np.sqrt(((X-mu)**2*W[:,:,None]).sum((0,1))/W.sum())
 sd=np.where(sd<1e-8,1,sd);Z=(X-mu)/sd;Zt=(Xt-mu)/sd
 for it in range(iterations):
  A=np.zeros((S*k,S*k));rhs=np.zeros(S*k);ybar=np.zeros(S);zbar=np.zeros((S,k))
  for s in range(S):
   w=W[:,s]; zbar[s]=(Z[:,s]*w[:,None]).sum(0)/w.sum();ybar[s]=(Y[:,s]*w).sum()/w.sum()
   Zc=(Z[:,s]-zbar[s])*np.sqrt(w)[:,None];yc=(Y[:,s]-ybar[s])*np.sqrt(w)
   sl=slice(s*k,(s+1)*k);A[sl,sl]+=Zc.T@Zc+alpha*np.eye(k);rhs[sl]=Zc.T@yc
  I=lam*np.eye(k)
  for s in range(S-1):
   a=slice(s*k,(s+1)*k);b=slice((s+1)*k,(s+2)*k)
   A[a,a]+=I;A[b,b]+=I;A[a,b]-=I;A[b,a]-=I
  B=np.linalg.solve(A,rhs).reshape(S,k)
  if it<iterations-1:
   err=Y-ybar[None]-np.einsum('nsp,sp->ns',Z-zbar[None],B)
   ww=np.minimum(1.,delta/np.maximum(abs(err),1.e-8))
   W=(~M)*ww
 return np.exp(ybar[None]+np.einsum('nsp,sp->ns',Zt-zbar[None],B))


def ridge_part(train, test, j, med):
    drop = jump_mask(train, j) if CFG["drop"][j] else None
    # Sparse feed-C event counts add a small, repeatable signal for cells.
    # Everything else, including substrate and byproduct, keeps the original heads.
    names = CFG["feats"][j] + (["nsmall"] if j in (0, 1) else [])
    if CFG["fuse"][j]:
        lam, day_scale = CFG["fuse"][j]
        if CFG["robust_delta"][j] is not None:
            r = robust_fused(train, test, j, names, CFG["alpha"][j], lam,
                             CFG["robust_delta"][j], iterations=4, day_scale=day_scale)
        else:
            r = fused_ridge(train, test, j, names, CFG["alpha"][j], lam, drop, day_scale)
    else:
        r = ridge_per_step(train, test, j, names, CFG["alpha"][j], drop)
    if CFG["rw"][j] < 1:
        r = CFG["rw"][j] * r + (1 - CFG["rw"][j]) * med
    return np.maximum(r, 1e-6)


# ---------------------------------------------------------------- substrate conditioning
def constraints(b):
    """(step index, threshold, True if substrate was below it) from the substrate controller"""
    out = []
    td = b["trig_day"]
    for day in (7, 8, 9):
        if td == 0 or day < td:
            out.append((day - 1, S_TRIG, False))
        elif day == td:
            out.append((day - 1, S_TRIG, True))
    for day, flag in ((11, b["low11"]), (12, b["low12"])):
        if not np.isnan(flag):
            out.append((day - 1, S_LOW, bool(flag)))
    return out


def substrate_cov(train, folds=5):
    """log-residual covariance of the substrate ridge, from an inner CV"""
    n = len(train)
    R = np.zeros((n, 13))
    for f in range(folds):
        vi = np.arange(f, n, folds)
        fi = np.setdiff1d(np.arange(n), vi)
        tr, va = [train[i] for i in fi], [train[i] for i in vi]
        med = np.stack([weighted_median(b, tr) for b in va])[..., 2]
        R[vi] = np.log(np.stack([b["y"][:, 2] for b in va]) / ridge_part(tr, va, 2, med))
    return LedoitWolf().fit(R).covariance_   # shrunk: only ~40 batches for a 13x13 covariance


def conditioned(p, cons, cov, rng, n=100000):
    """sample log residuals, keep those consistent with the controller, take the 1/y-weighted median (MAPE-optimal)"""
    L = np.linalg.cholesky(cov + 1e-9 * np.eye(len(p)))
    r = rng.standard_normal((n, len(p))) @ L.T
    ok = np.ones(n, bool)
    for k, thr, below in cons:
        c = np.log(thr / p[k])
        ok &= (r[:, k] < c) if below else (r[:, k] >= c)
    if ok.sum() < 200:
        ok[:] = True
    y = p[None] * np.exp(r[ok])
    order = np.argsort(y, 0)
    ys = np.take_along_axis(y, order, 0)
    cw = (1 / ys).cumsum(0)
    idx = (cw >= cw[-1] / 2).argmax(0)
    return ys[idx, np.arange(len(p))]


# ---------------------------------------------------------------- predict / cv / submit
def predict_v9(train, test):
    med = np.stack([weighted_median(b, train) for b in test])
    P = np.zeros_like(med)
    for j in range(4):
        drop = jump_mask(train, j) if CFG["drop"][j] else None
        r = ridge_part(train, test, j, med[..., j])
        if CFG["gmix"][j] > 0:
            r = (1 - CFG["gmix"][j]) * r + CFG["gmix"][j] * boosting(train, test, j, CFG["gfeats"][j], drop)
        P[..., j] = r
    cov = 1.5 * substrate_cov(train)
    rng = np.random.default_rng(123)
    for i, b in enumerate(test):
        P[i, :, 2] = conditioned(P[i, :, 2], constraints(b), cov, rng)
    for j in (0, 1):   # soft-limit extreme vcd/tcd deviations from the population curve
        pop = np.exp(np.log(np.stack([b["y"][:, j] for b in train])).mean(0))
        d = np.log(P[..., j] / pop)
        if SHRINK_TAU:
            d = SHRINK_TAU * np.tanh(d / SHRINK_TAU)
        P[..., j] = pop * np.exp(d)
    P[..., 1] = np.maximum(P[..., 1], P[..., 0])   # tcd >= vcd
    return P


def improved_byproduct(train, test):
    """Byproduct-only revision. Uses supplied future feed/volume profiles.

    Fit log(y_t / y0**power), regularized across adjacent days with Huber IRLS.
    Never use unknown output measurements from forecast batches.
    """
    transformed = []
    for batch in train:
        altered = batch.copy()
        altered["y"] = batch["y"].copy()
        altered["y"][:, 3] /= batch["y0"][3] ** BYPRODUCT_INITIAL_POWER
        transformed.append(altered)
    reg = robust_fused(
        transformed, test, 3, BYPRODUCT_FEATS, alpha=3, lam=30,
        delta=0.15, iterations=4, day_scale=True, clip_inputs=False
    )
    reg *= np.array([b["y0"][3] ** BYPRODUCT_INITIAL_POWER for b in test])[:, None]
    tree = boosting(train, test, 3, CFG["gfeats"][3], jump_mask(train, 3))
    return np.maximum((1 - BYPRODUCT_BOOST_SHARE) * reg + BYPRODUCT_BOOST_SHARE * tree, 1e-6)


def predict(train, test):
    """Blend v9 and RBF-SVR for VCD and TCD."""
    predictions = predict_v9(train, test)
    truth = np.stack([batch["y"][:, :2] for batch in train])
    population = np.exp(np.log(truth).mean(axis=0))
    svr = np.empty_like(predictions[..., :2])

    for target in (0, 1):
        # More compact SVR: retain *current* substrate proxy instead of the
        # entire 13-day profile, and include contemporaneous Feed-B deviation.
        # Lower C means more regularization for only 49 independent batches.
        names = ["snc" if nm == "snc_tile" else nm for nm in CFG["feats"][target]] + ["fb_now"]
        train_x = np.concatenate([X_of(batch, names) for batch in train])
        test_x = np.concatenate([X_of(batch, names) for batch in test])
        scaler = StandardScaler().fit(train_x)
        keep = ~jump_mask(train, target).ravel()
        log_deviation = np.log(truth[..., target] / population[:, target])
        model = SVR(kernel="rbf", C=0.1, epsilon=0.025, gamma="scale")
        model.fit(scaler.transform(train_x)[keep], log_deviation.ravel()[keep])
        deviation = model.predict(scaler.transform(test_x)).reshape(predictions.shape[:2])
        svr[..., target] = population[:, target] * np.exp(
            0.30 * np.tanh(deviation / 0.30)
        )

    svr[..., 1] = np.maximum(svr[..., 1], svr[..., 0])
    predictions[..., :2] = 0.5 * (predictions[..., :2] + svr)
    # Keep the collaborator's VCD/TCD and substrate forecasts unchanged.
    # A pre-specified stage-gated byproduct ensemble: the original model had
    # a small but consistent edge on the first 1-3 sample days in batch CV.
    byproduct_original = predictions[..., 3].copy()
    byproduct_v12 = improved_byproduct(train, test)
    improved_byproduct_fallback = byproduct_v12.copy()
    byproduct_v12[:, 0] = byproduct_original[:, 0]
    byproduct_v12[:, 1:3] = (
        0.5 * byproduct_v12[:, 1:3] + 0.5 * byproduct_original[:, 1:3]
    )
    # Hidden batches with initial byproduct beyond the training range are OOD.
    # Do not extrapolate the early-stage blend in this regime; revert to v12.
    upper_training_initial = max(batch["y0"][3] for batch in train)
    beyond_training_range = np.array([
        batch["y0"][3] > upper_training_initial for batch in test
    ])
    byproduct_v12[beyond_training_range, :3] = (
        improved_byproduct_fallback[beyond_training_range, :3]
    )
    predictions[..., 3] = byproduct_v12
    return predictions


def cross_validate(batches, seeds=(0, 1, 2)):
    train = [b for b in batches if not b["test"]]
    errs = []
    for seed in seeds:
        order = np.random.default_rng(seed).permutation(len(train))
        for f in range(5):
            val_idx = set(order[f::5])
            fit = [b for i, b in enumerate(train) if i not in val_idx]
            val = [b for i, b in enumerate(train) if i in val_idx]
            P = predict(fit, val)
            Y = np.stack([b["y"] for b in val])
            errs.append((np.abs(P - Y) / Y * 100).reshape(-1, 4))
    e = np.concatenate(errs).mean(0)
    print("CV MAPE  " + "  ".join(f"{n} {v:.2f}" for n, v in zip(TARGETS, e)) + f"  mean {e.mean():.2f}")


def make_submission(batches, template_path, out_path):
    train = [b for b in batches if not b["test"]]
    test = [b for b in batches if b["test"]]
    P = predict(train, test)
    tmpl = pd.read_csv(template_path, dtype={"batch_id": str})
    lookup = {b["id"]: (b["t"], p) for b, p in zip(test, P)}
    rows = []
    for _, r in tmpl.iterrows():
        t, p = lookup[r.batch_id]
        k = np.abs(t - float(r.elapsed_time)).argmin()
        assert abs(t[k] - float(r.elapsed_time)) < 1e-6, f"time mismatch in {r.batch_id}"
        rows.append(p[k])
    tmpl[TARGETS] = np.array(rows)
    assert tmpl.shape == (156, 6) and np.isfinite(tmpl[TARGETS].to_numpy(float)).all()
    assert (tmpl[TARGETS].to_numpy(float) > 0).all()
    tmpl.to_csv(out_path, index=False)
    print(f"saved {len(tmpl)} predictions to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bioreactor v9 + SVR forecast")
    parser.add_argument("--data", default=DATA)
    parser.add_argument("--template", default=TEMPLATE)
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--cv", action="store_true")
    parser.add_argument("--cv-only", action="store_true", help="Only evaluate; do not create a submission CSV")
    args = parser.parse_args()

    batches = prepare(args.data)
    if args.cv:
        cross_validate(batches)
    if not args.cv_only:
        make_submission(batches, args.template, args.output)

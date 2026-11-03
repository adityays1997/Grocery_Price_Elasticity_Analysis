"""Helper functions for the grocery price elasticity project (numpy / pandas / scipy only)."""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar

from simulate_sales import simulate


def ols(y, X, lags=4):
    """OLS with Newey-West standard errors (the errors are autocorrelated, so plain SEs are too small)."""
    y, X = np.asarray(y, float), np.asarray(X, float)
    n, k = X.shape
    bread = np.linalg.inv(X.T @ X)
    beta = bread @ X.T @ y
    resid = y - X @ beta
    g = X * resid[:, None]
    meat = g.T @ g
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        c = g[l:].T @ g[:-l]
        meat += w * (c + c.T)
    se = np.sqrt(np.diag(bread @ meat @ bread * n / (n - k)))
    t = stats.t.ppf(0.975, n - k)
    return {"beta": beta, "se": se, "lo": beta - t * se, "hi": beta + t * se, "resid": resid}


def design(d, controls):
    cols = [np.ones(len(d)), np.log(d["price"].values)]
    if controls:
        w = d["week"].values
        cols += [np.log(d["competitor_price"].values), d["holiday_week"].values,
                 np.sin(2 * np.pi * w / 52), np.cos(2 * np.pi * w / 52)]
    return np.column_stack(cols)


def estimate(df, controls):
    out = []
    for sku, d in df.groupby("sku"):
        r = ols(np.log(d["units"].values), design(d, controls))
        out.append({"sku": sku, "elasticity": r["beta"][1], "se": r["se"][1], "ci_low": r["lo"][1], "ci_high": r["hi"][1]})
    return pd.DataFrame(out)


def compare(df, catalog):
    a = estimate(df, False).rename(columns={c: c + "_naive" for c in ["elasticity", "se", "ci_low", "ci_high"]})
    b = estimate(df, True).rename(columns={c: c + "_ctrl" for c in ["elasticity", "se", "ci_low", "ci_high"]})
    m = catalog[["sku", "product_name", "category", "true_elasticity"]].merge(a, on="sku").merge(b, on="sku")
    m["bias_naive"] = m.elasticity_naive - m.true_elasticity
    m["bias_ctrl"] = m.elasticity_ctrl - m.true_elasticity
    m["covered"] = (m.ci_low_ctrl <= m.true_elasticity) & (m.true_elasticity <= m.ci_high_ctrl)
    return m


def replicate(seeds):
    rows = []
    for s in seeds:
        df, cat = simulate(s)
        m = compare(df, cat)
        rows.append({"seed": s, "abs_bias_naive": m.bias_naive.abs().mean(), "abs_bias_ctrl": m.bias_ctrl.abs().mean(),
                     "signed_bias_naive": m.bias_naive.mean(), "signed_bias_ctrl": m.bias_ctrl.mean(),
                     "coverage": m.covered.mean()})
    r = pd.DataFrame(rows)
    r["ctrl_better"] = r.abs_bias_ctrl < r.abs_bias_naive
    return r


def profit(p, cost, e, A):
    return (p - cost) * A * p ** e


def optimise(df, catalog, est, col="elasticity_ctrl", ceiling=1.25):
    """Best price under q = A * p^e, searched between cost and ceiling * list price."""
    rows = []
    for _, c in catalog.iterrows():
        d = df[df.sku == c.sku]
        e = float(est.loc[est.sku == c.sku, col].iloc[0])
        p0, q0 = d.price.mean(), d.units.mean()
        A = q0 / p0 ** e
        hi = ceiling * c.list_price
        if e >= -1:
            p_best = hi          # profit keeps rising with price, so the cap decides
        else:
            p_best = minimize_scalar(lambda p: -profit(p, c.cost, e, A), bounds=(c.cost * 1.01, hi), method="bounded").x
        A_true = q0 / p0 ** c.true_elasticity
        rows.append({
            "sku": c.sku, "product_name": c.product_name, "est_elasticity": e, "price_now": p0, "price_best": p_best,
            "cap": hi, "at_cap": p_best >= hi * 0.99,
            "lift_model_pct": 100 * (profit(p_best, c.cost, e, A) / profit(p0, c.cost, e, A) - 1),
            "lift_true_pct": 100 * (profit(p_best, c.cost, c.true_elasticity, A_true) / profit(p0, c.cost, c.true_elasticity, A_true) - 1),
        })
    return pd.DataFrame(rows)


def cap_sensitivity(df, catalog, est, caps=(1.0, 1.1, 1.25, 1.4, 1.6, 2.0)):
    out = []
    for c in caps:
        o = optimise(df, catalog, est, ceiling=c)
        out.append({"cap_x_list_price": c, "n_at_cap": int(o.at_cap.sum()), "n_interior": int((~o.at_cap).sum()),
                    "mean_lift_model_pct": o.lift_model_pct.mean(), "mean_lift_true_pct": o.lift_true_pct.mean()})
    return pd.DataFrame(out)

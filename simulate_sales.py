"""
Simulates weekly sales for a small grocery chain: 12 products in 3 categories,
130 weeks (Jan 2021 to Jun 2023).

I simulate the data (instead of using a real dataset) because that lets me know the
true price elasticity of every product and check how far off my estimates are.

How demand is generated for product i in week t:

    log(units) = log(base_units)
                 + elasticity * log(price / list_price)
                 + cross * log(competitor_price / list_price)
                 + season_amp * sin(2*pi*week/52 + phase)
                 + holiday_lift * holiday_week
                 + shock

`shock` is an AR(1) process, so a bad week tends to be followed by another bad week.
The store manager reacts to a bad week by running a bigger discount the next week.
That makes price depend on the shock, which is the endogeneity problem the
analysis is about.

Run:  python simulate_sales.py --seed 2024 --out data
"""
import argparse
import os

import numpy as np
import pandas as pd

N_WEEKS = 130
START = pd.Timestamp("2021-01-04")
EASTER = [pd.Timestamp(d) for d in ("2021-04-04", "2022-04-17", "2023-04-09")]

# name, category, list price, cost as share of price, typical weekly units
PRODUCTS = [
    ("Whole Milk 2L",        "Dairy",     3.20, 0.72, 900),
    ("Greek Yogurt 500g",    "Dairy",     4.80, 0.62, 420),
    ("Cheddar Block 400g",   "Dairy",     7.50, 0.68, 310),
    ("Butter 250g",          "Dairy",     5.40, 0.70, 350),
    ("Potato Chips 200g",    "Snacks",    4.50, 0.52, 560),
    ("Dark Chocolate Bar",   "Snacks",    3.90, 0.50, 480),
    ("Mixed Nuts 300g",      "Snacks",    8.90, 0.60, 230),
    ("Granola Bars 6pk",     "Snacks",    5.70, 0.55, 300),
    ("Orange Juice 1.5L",    "Beverages", 4.60, 0.58, 520),
    ("Cold Brew Coffee 1L",  "Beverages", 6.80, 0.55, 260),
    ("Sparkling Water 12pk", "Beverages", 9.90, 0.60, 340),
    ("Green Tea 20 bags",    "Beverages", 4.20, 0.45, 280),
]
# plausible elasticity range per category (staples are less price sensitive)
ELASTICITY_RANGE = {"Dairy": (-1.7, -0.9), "Snacks": (-3.0, -1.8), "Beverages": (-2.4, -1.3)}


def holiday_flags(dates):
    flags = []
    for d in dates:
        end = d + pd.Timedelta(days=6)
        xmas = (d.month == 12 and d.day >= 18) or (d.month == 1 and d.day <= 3)
        easter = any(d <= e <= end for e in EASTER)
        flags.append(int(xmas or easter))
    return np.array(flags)


def simulate(seed=2024):
    rng = np.random.default_rng(seed)
    dates = [START + pd.Timedelta(weeks=int(w)) for w in range(N_WEEKS)]
    holiday = holiday_flags(dates)

    catalog, rows = [], []
    for k, (name, cat, price, cost_share, units0) in enumerate(PRODUCTS, start=1):
        lo, hi = ELASTICITY_RANGE[cat]
        info = {
            "sku": f"SKU{k:02d}", "product_name": name, "category": cat,
            "list_price": price, "cost": round(price * cost_share, 2),
            "base_units": units0,
            "true_elasticity": round(rng.uniform(lo, hi), 3),
            "cross_elasticity": round(rng.uniform(0.15, 0.7), 3),
            "season_amp": round(rng.uniform(0.05, 0.25), 3),
            "season_phase": round(rng.uniform(0, 2 * np.pi), 3),
            "holiday_lift": round(rng.uniform(0.10, 0.45), 3),
        }
        catalog.append(info)

        # competitor price drifts around our list price
        log_comp = np.zeros(N_WEEKS)
        for t in range(1, N_WEEKS):
            log_comp[t] = 0.8 * log_comp[t - 1] + rng.normal(0, 0.035)
        comp = price * np.exp(log_comp)

        # persistent demand shock
        rho, sd = 0.65, 0.14
        shock = np.zeros(N_WEEKS)
        shock[0] = rng.normal(0, sd)
        for t in range(1, N_WEEKS):
            shock[t] = rho * shock[t - 1] + rng.normal(0, sd * np.sqrt(1 - rho ** 2))

        for t in range(N_WEEKS):
            prev = shock[t - 1] if t > 0 else 0.0
            # manager reacts to last week's weak sales, plus holiday offers and ordinary promo noise
            disc = 0.10 - 0.12 * prev + 0.05 * holiday[t] + rng.normal(0, 0.065)
            disc = float(np.clip(disc, 0, 0.40))
            p = price * (1 - disc)
            log_u = (np.log(units0)
                     + info["true_elasticity"] * np.log(p / price)
                     + info["cross_elasticity"] * np.log(comp[t] / price)
                     + info["season_amp"] * np.sin(2 * np.pi * t / 52 + info["season_phase"])
                     + info["holiday_lift"] * holiday[t]
                     + shock[t])
            units = max(int(round(np.exp(log_u))), 1)
            rows.append({
                "week": t, "week_start": dates[t].date(), "sku": info["sku"],
                "product_name": name, "category": cat,
                "price": round(p, 2), "list_price": price, "cost": info["cost"],
                "competitor_price": round(comp[t], 2),
                "discount_pct": round(disc * 100, 2),
                "holiday_week": int(holiday[t]), "on_promo": int(disc >= 0.15),
                "units": units, "revenue": round(p * units, 2),
                "gross_profit": round((p - info["cost"]) * units, 2),
            })
    return pd.DataFrame(rows), pd.DataFrame(catalog)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    sales, catalog = simulate(a.seed)
    os.makedirs(a.out, exist_ok=True)
    sales.to_csv(os.path.join(a.out, "weekly_sales.csv"), index=False)
    catalog.to_csv(os.path.join(a.out, "product_catalog_true_params.csv"), index=False)
    print(f"{len(sales)} rows: {sales.sku.nunique()} products x {sales.week.nunique()} weeks -> {a.out}/")

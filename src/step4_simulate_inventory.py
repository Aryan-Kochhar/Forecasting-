"""STEP 4 - Use the forecasts to run the store, and count the money.

A forecast is only useful if it leads to a better decision. Here every method
gets to restock the same store over the same 26 weeks of real demand, and we
measure two things that pull against each other:
  fill rate       : share of customer demand we could actually serve
  inventory value : how much money is sitting on the shelves on an average day

Run:  python src/step4_simulate_inventory.py
"""

import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (DASHBOARD_DATA, FIGURES, HOLDING_RATE, LEAD_TIME, MARGIN,
                    PROCESSED, REVIEW_DAYS, SERVICE_LEVELS, STORE)
from simulator import simulate

LABELS = {"normal": "normal formula", "history": "past-year quantile", "lgbm": "LightGBM quantiles"}

forecasts = pd.read_parquet(PROCESSED / "test_forecasts.parquet")
sales = pd.read_parquet(PROCESSED / "sales.parquet")
prices = pd.read_parquet(PROCESSED / "prices.parquet")

# The simulation starts the day after the first forecast and runs to the end of the data.
order_days = sorted(forecasts["date"].unique())
sim_days = sales.index[sales.index > order_days[0]]

# Only keep products that were on sale for the whole test period.
counts = forecasts.groupby("item_id")["date"].count()
products = counts[counts == len(order_days)].index
demand = sales.loc[sim_days, products].to_numpy()
price = prices.loc[sim_days, products].to_numpy()
unit_cost = price * (1 - MARGIN)          # what the store paid for one unit

rows = []
for method in ["naive", "normal", "history", "lgbm"]:
    for level in SERVICE_LEVELS:
        # table of order-up-to levels: one row per ordering day, one column per product
        targets = forecasts.pivot(index="date", columns="item_id", values=f"{method}_{level}")
        result = simulate(demand, targets[products].to_numpy(), REVIEW_DAYS, LEAD_TIME)

        lost_profit = (result["lost"] * price * MARGIN).sum()
        holding_cost = (result["stock"] * unit_cost * HOLDING_RATE / 365).sum()
        rows.append(
            {
                "method": method,
                "service_level": level,
                "fill_rate": result["sold"].sum() / demand.sum(),
                "avg_inventory_value": (result["stock"] * unit_cost).sum(axis=1).mean(),
                "lost_profit": lost_profit,
                "holding_cost": holding_cost,
                "total_cost": lost_profit + holding_cost,
            }
        )
        if method == "naive":
            break   # naive has no safety margin, so every service level is identical

summary = pd.DataFrame(rows)
summary.to_csv(PROCESSED / "inventory_summary.csv", index=False)

pd.set_option("display.width", 200)
print(f"Store {STORE}: {len(products)} products, {len(sim_days)} days, {len(order_days)} weekly orders\n")
print(summary.round({"fill_rate": 4, "avg_inventory_value": 0, "lost_profit": 0,
                     "holding_cost": 0, "total_cost": 0}).to_string(index=False))

# ---------- fair comparison: same fill rate, how much inventory does each need? ----------
# The methods never land on exactly the same fill rate, so we read the
# inventory off each curve at a common fill rate (straight line between points).
print("\nInventory needed to reach the same fill rate:")
matched = []
for goal in [0.90, 0.95, 0.97]:
    need = {"fill_rate": goal}
    for method in LABELS:
        part = summary[summary["method"] == method]
        need[method] = np.interp(goal, part["fill_rate"], part["avg_inventory_value"])
    need["saving_vs_normal"] = 1 - need["lgbm"] / need["normal"]
    need["saving_vs_history"] = 1 - need["lgbm"] / need["history"]
    matched.append(need)
    print(f"  {goal:.0%}: normal ${need['normal']:,.0f} | history ${need['history']:,.0f} | lgbm ${need['lgbm']:,.0f}"
          f" | lgbm saves {need['saving_vs_normal']:.1%} vs normal, {need['saving_vs_history']:.1%} vs history")
pd.DataFrame(matched).to_csv(PROCESSED / "matched_comparison.csv", index=False)

# ---------- chart: the trade-off curve ----------
# Every point is one policy. Up = fewer empty shelves. Left = less money tied up.
# A better method gives a curve that sits up and to the left.
FIGURES.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(7, 5))
for method, label in LABELS.items():
    part = summary[summary["method"] == method]
    ax.plot(part["avg_inventory_value"], part["fill_rate"] * 100, "o-", label=label)
naive = summary[summary["method"] == "naive"]
ax.plot(naive["avg_inventory_value"], naive["fill_rate"] * 100, "ks", label="naive (no safety stock)")
ax.set_xlabel("average inventory value on the shelves ($)")
ax.set_ylabel("fill rate (% of demand served)")
ax.set_title(f"Service vs. inventory, 26-week backtest, store {STORE}")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "tradeoff.png", dpi=150)

# ---------- save small files for the dashboard ----------
# Only the test period is needed, so these are small enough to keep in git.
# That lets the dashboard run online without the raw Walmart files.
out = DASHBOARD_DATA / STORE
out.mkdir(parents=True, exist_ok=True)
sales.loc[sim_days, products].to_parquet(out / "sales.parquet")
prices.loc[sim_days, products].to_parquet(out / "prices.parquet")
small = forecasts[forecasts["item_id"].isin(products)].copy()
number_cols = small.columns.drop(["date", "item_id"])
small[number_cols] = small[number_cols].round(1).astype("float32")
small.to_parquet(out / "test_forecasts.parquet", index=False)
for name in ["inventory_summary.csv", "forecast_scores.csv", "matched_comparison.csv"]:
    shutil.copy(PROCESSED / name, out / name)

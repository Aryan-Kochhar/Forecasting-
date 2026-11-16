"""STEP 4 - Use the forecasts to run the store, and count the money.

A forecast is only useful if it leads to a better decision. Here every method
gets to restock the same store over the same 26 weeks of real demand, and we
measure two things that pull against each other:
  fill rate       : share of customer demand we could actually serve
  inventory value : how much money is sitting on the shelves on an average day

Run:  python src/step4_simulate_inventory.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (FIGURES, HOLDING_RATE, LEAD_TIME, MARGIN, PROCESSED,
                    REVIEW_DAYS, SERVICE_LEVELS)
from simulator import simulate

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
for method in ["naive", "normal", "lgbm"]:
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
print(f"{len(products)} products, {len(sim_days)} days, {len(order_days)} weekly orders\n")
print(summary.round({"fill_rate": 4, "avg_inventory_value": 0, "lost_profit": 0,
                     "holding_cost": 0, "total_cost": 0}).to_string(index=False))

# ---------- fair comparison: same fill rate, how much inventory does each need? ----------
# The two methods never land on exactly the same fill rate, so we read the
# inventory off each curve at a common fill rate (straight line between points).
print("\nInventory needed to reach the same fill rate:")
matched = []
for goal in [0.90, 0.95, 0.97]:
    need = {}
    for method in ["normal", "lgbm"]:
        part = summary[summary["method"] == method]
        need[method] = np.interp(goal, part["fill_rate"], part["avg_inventory_value"])
    saving = 1 - need["lgbm"] / need["normal"]
    matched.append({"fill_rate": goal, "normal": need["normal"], "lgbm": need["lgbm"], "saving": saving})
    print(f"  {goal:.0%} fill rate: normal ${need['normal']:,.0f} | lgbm ${need['lgbm']:,.0f} | {saving:.1%} less")
pd.DataFrame(matched).to_csv(PROCESSED / "matched_comparison.csv", index=False)

# ---------- chart: the trade-off curve ----------
# Every point is one policy. Up = fewer empty shelves. Left = less money tied up.
# A better method gives a curve that sits up and to the left.
fig, ax = plt.subplots(figsize=(7, 5))
for method, label in [("normal", "normal formula"), ("lgbm", "LightGBM quantiles")]:
    part = summary[summary["method"] == method]
    ax.plot(part["avg_inventory_value"], part["fill_rate"] * 100, "o-", label=label)
naive = summary[summary["method"] == "naive"]
ax.plot(naive["avg_inventory_value"], naive["fill_rate"] * 100, "ks", label="naive (no safety stock)")
ax.set_xlabel("average inventory value on the shelves ($)")
ax.set_ylabel("fill rate (% of demand served)")
ax.set_title("Service vs. inventory, 26-week backtest")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "tradeoff.png", dpi=150)

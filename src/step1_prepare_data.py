"""STEP 1 - Turn the raw Walmart files into two tidy tables for one store.

Output (in data/processed):
  sales.parquet   one row per day, one column per product, value = units sold
  prices.parquet  same shape, value = selling price that day
  calendar.parquet  one row per day: is it a SNAP day? is there an event?
  items.parquet   one row per product: which department / category it is in

Run:  python src/step1_prepare_data.py
"""

import pandas as pd

from config import PROCESSED, RAW, STORE

PROCESSED.mkdir(parents=True, exist_ok=True)

# ---------- sales ----------
# The raw file has one ROW per product and one COLUMN per day (d_1, d_2, ...).
raw = pd.read_csv(RAW / "sales_train_evaluation.csv")
raw = raw[raw["store_id"] == STORE]
day_cols = [c for c in raw.columns if c.startswith("d_")]

items = raw[["item_id", "dept_id", "cat_id"]].reset_index(drop=True)

# ---------- calendar ----------
# calendar.csv has one row per day in order, so row 1 is "d_1", row 2 is "d_2"...
# It gives the real date plus events and SNAP days.
# SNAP = US food-stamp programme; on SNAP days people have money to buy food.
cal = pd.read_csv(RAW / "calendar.csv", parse_dates=["date"])
cal = cal.iloc[: len(day_cols)]
calendar = pd.DataFrame(
    {
        "snap": cal[f"snap_{STORE[:2]}"].to_numpy(),   # SNAP days differ by state (CA / TX / WI)
        "event": cal["event_name_1"].notna().astype(int).to_numpy(),
    },
    index=pd.DatetimeIndex(cal["date"], name="date"),
)

# Flip sales so that days are rows and products are columns.
sales = raw.set_index("item_id")[day_cols].T
sales.index = calendar.index
sales = sales.astype("float32")

# ---------- prices ----------
# Prices are stored per WEEK. We copy each week's price onto its 7 days.
p = pd.read_csv(RAW / "sell_prices.csv")
p = p[p["store_id"] == STORE]
weekly = p.pivot(index="wm_yr_wk", columns="item_id", values="sell_price")
prices = weekly.loc[cal["wm_yr_wk"].to_numpy(), sales.columns]
prices.index = calendar.index
prices = prices.astype("float32")

# A product has no price before it was launched. In the raw sales those days
# show up as 0, which is misleading: it wasn't "zero demand", the product
# simply wasn't there. So we blank those days out.
sales = sales.where(prices.notna())

sales.to_parquet(PROCESSED / "sales.parquet")
prices.to_parquet(PROCESSED / "prices.parquet")
calendar.to_parquet(PROCESSED / "calendar.parquet")
items.to_parquet(PROCESSED / "items.parquet")

print(f"Store {STORE}: {sales.shape[1]} products, {sales.shape[0]} days")
print(f"From {sales.index.min().date()} to {sales.index.max().date()}")
print(f"Share of product-days with zero sales: {(sales == 0).sum().sum() / sales.notna().sum().sum():.0%}")

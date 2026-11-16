"""STEP 2 - Build the table the model learns from.

One row = "one product, on one day". For that row we store:
  target    : how many units were sold in the NEXT 10 days (what we want to predict)
  features  : things we already know on that day (recent sales, price, calendar)

Golden rule: a feature may only use information available on that day.
Using anything from the future (other than plans we really would know, like
our own prices and the calendar) is called "leakage" and makes results fake.

Run:  python src/step2_build_features.py
"""

import pandas as pd

from config import HORIZON, PROCESSED, TEST_WEEKS, TRAIN_YEARS, VALID_WEEKS

sales = pd.read_parquet(PROCESSED / "sales.parquet")
prices = pd.read_parquet(PROCESSED / "prices.parquet")
calendar = pd.read_parquet(PROCESSED / "calendar.parquet")
items = pd.read_parquet(PROCESSED / "items.parquet")


def next_days_sum(table):
    """For every day: the total over the NEXT `HORIZON` days (not including today)."""
    return table.rolling(HORIZON).sum().shift(-HORIZON)


# ---------- target ----------
target = next_days_sum(sales)

# ---------- features (each one is a days x products table) ----------
feats = {}

# Recent sales level: average units per day over the last week / month / quarter / year
for window in [7, 28, 91, 364]:
    feats[f"mean_{window}"] = sales.rolling(window, min_periods=7).mean()

# How bumpy sales are, and how often the product sells nothing at all
feats["std_91"] = sales.rolling(91, min_periods=14).std()
feats["zero_share_28"] = (sales == 0).where(sales.notna()).rolling(28, min_periods=7).mean()

# What happened in the same 10-day window one year ago (captures yearly seasonality)
feats["same_window_last_year"] = target.shift(364)

# Price: today's price, is it cheaper than usual, and is a price change planned
feats["price"] = prices
feats["price_vs_usual"] = prices / prices.rolling(91, min_periods=7).mean()
feats["planned_price_change"] = (next_days_sum(prices) / HORIZON) / prices

# ---------- train only on recent history + the test period ----------
last_day = sales.index.max()
last_forecast_day = last_day - pd.Timedelta(days=HORIZON)
test_days = pd.date_range(end=last_forecast_day, periods=TEST_WEEKS, freq="7D")
test_start = test_days[0]
valid_start = test_start - pd.Timedelta(days=HORIZON + VALID_WEEKS * 7)
train_start = test_start - pd.Timedelta(days=365 * TRAIN_YEARS)

keep = sales.index >= train_start


def to_long(table, name):
    """days x products table  ->  one long column indexed by (date, item_id)."""
    return table.loc[keep].stack(future_stack=True).rename(name)


data = pd.concat(
    [to_long(target, "target")] + [to_long(t, name) for name, t in feats.items()],
    axis=1,
).reset_index()

# Calendar features are the same for every product, so we join them on by date.
cal_feats = pd.DataFrame(
    {
        "snap_days_ahead": next_days_sum(calendar["snap"]),
        "event_days_ahead": next_days_sum(calendar["event"]),
        "month": calendar.index.month,
        "weekday": calendar.index.day_of_week,
    },
    index=calendar.index,
)
data = data.merge(cal_feats, on="date").merge(items[["item_id", "dept_id"]], on="item_id")
data["dept_id"] = data["dept_id"].astype("category")

# Drop rows where the product wasn't on sale yet
data = data.dropna(subset=["price", "mean_28"])

# ---------- label each row: train / valid / test ----------
# Split by TIME. The test rows are the 26 weekly ordering days at the end.
# Training rows stop HORIZON days before the test starts, so that the 10-day
# window they predict never reaches into the test period.
data["split"] = "unused"
data.loc[data["date"] <= test_start - pd.Timedelta(days=HORIZON), "split"] = "train"
data.loc[(data["split"] == "train") & (data["date"] > valid_start), "split"] = "valid"
data.loc[data["date"].isin(test_days), "split"] = "test"
data = data[data["split"] != "unused"].dropna(subset=["target"])

data.to_parquet(PROCESSED / "model_data.parquet", index=False)

print(data["split"].value_counts().to_string())
print(f"Test ordering days: {test_days[0].date()} to {test_days[-1].date()}")
print(f"Columns: {list(data.columns)}")

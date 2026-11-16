"""STEP 3 - Train the forecasting model and check it against simple baselines.

We do NOT predict one number. For each product we predict several
"quantiles" of the next-10-day demand. Example for the 0.90 quantile:
"there is a 90% chance demand will be at or below this number".
That is exactly what a store needs to decide how much to stock.

Four methods are compared:
  naive   : recent daily average x 10 days (no safety margin)
  normal  : textbook formula = average + z * standard deviation (bell-curve assumption)
  history : look at every 10-day total from the past year and take its quantile
            (no bell-curve assumption, so a tougher baseline to beat)
  lgbm    : LightGBM model trained to predict each quantile directly

Run:  python src/step3_train_forecasts.py
"""

from statistics import NormalDist

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FIGURES, HORIZON, PROCESSED, SERVICE_LEVELS

METHODS = ["naive", "normal", "history", "lgbm"]

FIGURES.mkdir(parents=True, exist_ok=True)

data = pd.read_parquet(PROCESSED / "model_data.parquet")
FEATURES = [c for c in data.columns if c not in ["date", "item_id", "target", "split"]]

train = data[data["split"] == "train"]
valid = data[data["split"] == "valid"]
test = data[data["split"] == "test"].copy()
print(f"train {len(train):,} rows | valid {len(valid):,} | test {len(test):,} | {len(FEATURES)} features")


def pinball_loss(actual, predicted, q):
    """The score for a quantile forecast (lower is better).
    Being too low costs q per unit, being too high costs (1 - q) per unit.
    So for q = 0.9, under-forecasting is punished 9x more than over-forecasting."""
    diff = actual - predicted
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


# ---------- baselines ----------
avg_demand = (test["mean_28"] * HORIZON).to_numpy()
spread = (test["std_91"].fillna(0) * np.sqrt(HORIZON)).to_numpy()

preds = {}
for q in SERVICE_LEVELS:
    z = NormalDist().inv_cdf(q)          # e.g. q = 0.95 -> z = 1.64
    preds[f"naive_{q}"] = avg_demand
    preds[f"normal_{q}"] = np.maximum(avg_demand + z * spread, 0)

# "history" baseline: for each product, list the 10-day sales totals ending on
# each of the last 364 days, and read the quantile straight off that list.
sales = pd.read_parquet(PROCESSED / "sales.parquet")
past_totals = sales.rolling(HORIZON).sum().iloc[-800:]   # only the recent part is needed
for q in SERVICE_LEVELS:
    table = past_totals.rolling(364, min_periods=28).quantile(q)
    long = table.stack().rename("value").reset_index()
    long.columns = ["date", "item_id", "value"]
    merged = test[["date", "item_id"]].merge(long, on=["date", "item_id"], how="left")
    preds[f"history_{q}"] = merged["value"].fillna(0).to_numpy()

# ---------- LightGBM: one model per quantile ----------
importance = None
for q in SERVICE_LEVELS:
    model = lgb.LGBMRegressor(
        objective="quantile",   # train with the pinball loss instead of squared error
        alpha=q,                # which quantile
        n_estimators=1500,      # maximum number of trees
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        verbose=-1,
    )
    model.fit(
        train[FEATURES],
        train["target"],
        eval_X=valid[FEATURES],
        eval_y=valid["target"],
        # stop adding trees once the validation score stops improving
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    preds[f"lgbm_{q}"] = np.maximum(model.predict(test[FEATURES]), 0)
    print(f"  quantile {q}: {model.best_iteration_} trees")
    if q == 0.9:
        importance = pd.Series(
            model.booster_.feature_importance(importance_type="gain"), index=FEATURES
        ).sort_values(ascending=False)

# Each quantile has its own model, so occasionally the 0.90 prediction comes
# out a bit above the 0.95 one. Sorting each row fixes that.
lgbm_cols = [f"lgbm_{q}" for q in SERVICE_LEVELS]
sorted_preds = np.sort(np.column_stack([preds[c] for c in lgbm_cols]), axis=1)
for i, c in enumerate(lgbm_cols):
    preds[c] = sorted_preds[:, i]

# ---------- score everything on the test period ----------
actual = test["target"].to_numpy()
rows = []
for q in SERVICE_LEVELS:
    for method in METHODS:
        p = preds[f"{method}_{q}"]
        rows.append(
            {
                "quantile": q,
                "method": method,
                "pinball_loss": pinball_loss(actual, p, q),
                # coverage: how often was real demand at or below our number?
                # A perfect 0.90 forecast has coverage 0.90.
                "coverage": float(np.mean(actual <= p)),
            }
        )
scores = pd.DataFrame(rows)
scores.to_csv(PROCESSED / "forecast_scores.csv", index=False)

print("\nPinball loss (lower is better):")
print(scores.pivot(index="quantile", columns="method", values="pinball_loss").round(3).to_string())
print("\nCoverage (should be close to the quantile):")
print(scores.pivot(index="quantile", columns="method", values="coverage").round(3).to_string())
print("\nMost useful features (0.9 model):")
print((importance / importance.sum()).round(3).head(8).to_string())

# ---------- save predictions for the inventory simulation ----------
out = test[["date", "item_id", "target"]].rename(columns={"target": "actual"})
for name, p in preds.items():
    out[name] = p
out.to_parquet(PROCESSED / "test_forecasts.parquet", index=False)
(importance / importance.sum()).rename("share_of_gain").to_csv(PROCESSED / "feature_importance.csv")

# ---------- chart: are the quantiles honest? ----------
cov = scores.pivot(index="quantile", columns="method", values="coverage")
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot([0.45, 1], [0.45, 1], "k--", label="perfect")
ax.plot(cov.index, cov["normal"], "o-", label="normal formula")
ax.plot(cov.index, cov["history"], "o-", label="past-year quantile")
ax.plot(cov.index, cov["lgbm"], "o-", label="LightGBM")
ax.set_xlabel("quantile we asked for")
ax.set_ylabel("how often real demand was below the forecast")
ax.set_title("Calibration on the test period")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "calibration.png", dpi=150)

"""All the settings for the project live here, so there are no magic numbers
hidden inside the other scripts."""

import os
from pathlib import Path

# ---- which slice of Walmart we work on ----
# One store at a time (3,049 products). CA_1 is the default; run_all_stores.py
# reruns the pipeline for other stores by setting the STORE environment variable.
STORE = os.environ.get("STORE", "CA_1")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed" / STORE
FIGURES = ROOT / "reports" / "figures" / STORE
DASHBOARD_DATA = ROOT / "dashboard" / "data"   # small result files the dashboard reads

# ---- how the store restocks ----
REVIEW_DAYS = 7           # we place an order once a week
LEAD_TIME = 3             # an order arrives 3 days after we place it
# An order placed today has to last until the NEXT order arrives,
# which is REVIEW_DAYS + LEAD_TIME days from now. That is how far ahead
# we need to forecast.
HORIZON = REVIEW_DAYS + LEAD_TIME   # 10 days

# ---- train / test split (by time, never random) ----
TEST_WEEKS = 26           # the last 26 weekly orders are the test period
VALID_WEEKS = 8           # the 8 weeks before that are used for early stopping
TRAIN_YEARS = 3           # how much history the model learns from

# ---- service levels we try ----
# 0.90 means "stock enough that we would cover demand 90% of the time"
SERVICE_LEVELS = [0.50, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99]

# ---- money assumptions (M5 only gives selling prices, so these are assumed) ----
MARGIN = 0.30             # profit is 30% of the selling price
HOLDING_RATE = 0.25       # keeping an item on the shelf for a year costs 25% of what we paid

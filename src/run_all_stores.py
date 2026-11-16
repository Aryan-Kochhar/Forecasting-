"""Run the whole pipeline for several stores, to check the result is not a
one-store fluke. Each store gets its own model, trained only on its own data.

Run:  python src/run_all_stores.py
"""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

STORES = ["CA_1", "TX_1", "WI_1"]     # one store from each state in the dataset
STEPS = ["step1_prepare_data.py", "step2_build_features.py",
         "step3_train_forecasts.py", "step4_simulate_inventory.py"]

src = Path(__file__).resolve().parent
root = src.parent

for store in STORES:
    print(f"\n===== {store} =====")
    for step in STEPS:
        # same scripts, just told to use a different store
        subprocess.run([sys.executable, str(src / step)], env={**os.environ, "STORE": store},
                       check=True, stdout=subprocess.DEVNULL)
        print(f"  finished {step}")

# ---------- put the stores side by side ----------
tables = []
for store in STORES:
    folder = root / "data" / "processed" / store
    matched = pd.read_csv(folder / "matched_comparison.csv")
    scores = pd.read_csv(folder / "forecast_scores.csv")
    coverage = scores[scores["quantile"] == 0.95].set_index("method")["coverage"]
    matched.insert(0, "store", store)
    matched["coverage_95_lgbm"] = coverage["lgbm"]
    matched["coverage_95_normal"] = coverage["normal"]
    matched["coverage_95_history"] = coverage["history"]
    tables.append(matched)

result = pd.concat(tables)
result.to_csv(root / "reports" / "store_comparison.csv", index=False)
pd.set_option("display.width", 250)
print("\n" + result.round(3).to_string(index=False))

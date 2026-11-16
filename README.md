# Walmart demand forecasting for inventory decisions

Most forecasting projects stop at "my error is X". This one asks the question a store
manager actually cares about: **how much of each product should I order this week?**

It forecasts demand for 3,049 products in one Walmart store (M5 dataset), turns the
forecasts into weekly orders, and replays 26 weeks of real demand to see how each
ordering method would have done.

## Results

26-week backtest on data the model never saw (Nov 2015 to May 2016).

**Same service to customers, less money sitting on shelves:**

| Fill rate reached | Textbook formula | LightGBM quantiles | Inventory saved |
|---|---|---|---|
| 90% | $67,362 | $64,619 | 4.1% |
| 95% | $91,990 | $86,003 | 6.5% |
| 97% | $112,895 | $100,402 | 11.1% |

(average value of stock on the shelves; fill rate = share of demand that could be served)

**The forecasts are honest about uncertainty.** When asked for a number that demand
should stay under 95% of the time, the model was right 94.6% of the time. The textbook
bell-curve formula managed only 86.3%.

| | |
|---|---|
| ![tradeoff](reports/figures/tradeoff.png) | ![calibration](reports/figures/calibration.png) |

## How it works

```
raw M5 files
   |  step1_prepare_data.py     pick one store, make a days x products table
   v
   |  step2_build_features.py   15 features per product per day + the target
   v
   |  step3_train_forecasts.py  LightGBM quantile models vs 2 simple baselines
   v
   |  step4_simulate_inventory.py  replay 26 weeks of weekly ordering
   v
dashboard/app.py                look at the results, product by product
```

- **What is predicted:** total units sold over the next 10 days (7 days between orders +
  3 days delivery time), because that is how long one order has to last.
- **Why quantiles:** a store doesn't want the average, it wants "a level I'll only exceed
  5% of the time". The model predicts that level directly.
- **Baselines:** (1) recent average with no safety margin, (2) the textbook
  safety-stock formula `average + z x standard deviation`.
- **Simulation:** every week, top the stock back up to the forecast level; orders take
  3 days; if the shelf is empty the sale is lost.

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Download the M5 data ([Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data))
and put `sales_train_evaluation.csv`, `calendar.csv`, `sell_prices.csv` in `data/raw/`. Then:

```bash
python src/step1_prepare_data.py
python src/step2_build_features.py
python src/step3_train_forecasts.py
python src/step4_simulate_inventory.py
streamlit run dashboard/app.py
```

The whole pipeline takes a few minutes on a laptop. `pytest` runs the simulator checks.

## Honest limitations

- The data records **sales, not demand**. If Walmart's shelf was empty on a day, the
  data shows 0 even if customers wanted the product. So true demand is a bit higher
  than what is simulated here.
- Profit margin (30%) and holding cost (25% a year) are assumptions; M5 only gives
  selling prices. The inventory-vs-fill-rate comparison does not depend on them.
- One store, fixed 3-day delivery, no pack sizes, no shelf-space limits, no spoilage.
- Next week's prices are used as a feature, assuming the store knows its own planned
  promotions.

More detail and explanations of every concept: [docs/LEARNING_GUIDE.md](docs/LEARNING_GUIDE.md)

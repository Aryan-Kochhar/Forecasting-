# Learning guide

Everything in this project, explained from zero. Read it in order once, then open each
script next to its section and read the code. Every script is under 150 lines.

---

## 1. The problem in plain words

A store has thousands of products. Once a week it orders more of each. Order too
little and the shelf goes empty (you lose the sale). Order too much and money sits on
the shelf doing nothing. To order well you need to know how much will sell, and how
unsure you are about that.

So the project has two halves:

1. **Forecast**: how much of each product will sell in the next 10 days?
2. **Decide**: given that forecast, how much to stock? Then check with a simulation
   whether the decision was any good.

The second half is what makes this more than a forecasting exercise.

## 2. The data (M5, Walmart)

- 1,941 days of daily unit sales (Jan 2011 to May 2016).
- We use one store, `CA_1`, with 3,049 products in 3 categories: FOODS, HOUSEHOLD, HOBBIES.
- Weekly selling prices, and a calendar with events and SNAP days (days when US food
  benefits are paid out, so food sales jump).

Two things to remember about this data:

- **55% of all product-days have zero sales.** Most products sell 0, 1 or 2 units a day.
  This is called *intermittent demand* and it is why simple formulas struggle.
- **It records sales, not demand.** If the shelf was empty, the data says 0, even if
  ten people wanted the product. Nobody can see lost demand in this dataset.

## 3. Step 1: preparing the data (`step1_prepare_data.py`)

The raw file has one row per product and 1,941 day columns. We flip it so each row is
a day and each column is a product. This "wide" shape makes the next step easy:
`sales.rolling(28).mean()` gives the 28-day average for every product at once.

Prices come per week, so we copy each week's price onto its seven days.

One cleaning decision: before a product was launched it has no price, but the sales
file shows 0. Those zeros are fake (the product did not exist), so we blank them out.
Otherwise the model would learn "this product never sells" from days it wasn't on sale.

## 4. Step 2: features and target (`step2_build_features.py`)

**Target** = units sold over the next 10 days. Why 10? We order every 7 days and
delivery takes 3. An order placed today must last until the *next* order arrives,
which is 7 + 3 = 10 days from now. In inventory language this is the *protection
interval*: review period + lead time.

**Features** = what we know on the day we order. There are 15:

| Feature | Meaning |
|---|---|
| `mean_7`, `mean_28`, `mean_91`, `mean_364` | average daily sales over the last week / month / quarter / year |
| `std_91` | how much daily sales jump around |
| `zero_share_28` | share of the last 28 days with no sales |
| `same_window_last_year` | sales in the same 10 days a year ago (yearly seasonality) |
| `price` | current price |
| `price_vs_usual` | price divided by its 91-day average (below 1 = on promotion) |
| `planned_price_change` | average price over the next 10 days divided by today's price |
| `snap_days_ahead`, `event_days_ahead` | number of SNAP days / event days in the next 10 days |
| `month`, `weekday` | calendar position of the ordering day |
| `dept_id` | which department the product belongs to |

### Leakage (the most common interview topic)

**Leakage** means the model sees information during training that it would not have
in real life. Results then look great but are fake.

How this project avoids it:

- Every sales feature only looks backwards (`rolling` uses today and earlier).
- The "ahead" features use only things genuinely known in advance: the calendar, and
  the store's own planned prices. (If you think planned prices are unfair, drop that
  one feature. It carries very little weight in the model.)
- **The split is by time, never random.** Train on the past, test on the future. With
  a random split, the model would train on Wednesday and be tested on Tuesday of the
  same week, which is cheating.
- Training rows stop 10 days before the test period starts. A training row's target
  covers the *next* 10 days, so without this gap, training targets would include test
  period sales.

The split:

| Part | Rows | What for |
|---|---|---|
| train | 2.81 million | the model learns from this (about 3 years) |
| valid | 170 thousand | the 8 weeks before the test: decides when to stop training |
| test | 79 thousand | 26 weekly ordering days, never touched until the end |

## 5. Step 3: the forecast (`step3_train_forecasts.py`)

### Quantiles instead of one number

A normal forecast says "you'll sell 20". Reality might be 12 or 35. For stocking you
need to know the range.

A **quantile** forecast says: "there is a 90% chance you'll sell 28 or fewer". That is
the 0.90 quantile. If you stock 28, you cover demand 9 times out of 10. The 0.5
quantile is the median, the middle guess.

We predict seven quantiles: 0.5, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99.

### Pinball loss: how a model learns a quantile

For the average you minimise squared error. For a quantile `q` you minimise the
**pinball loss**:

- forecast too low by 1 unit: penalty `q`
- forecast too high by 1 unit: penalty `1 - q`

For q = 0.9: being too low costs 0.9, being too high costs 0.1. Under-forecasting is
9 times more expensive, so the model pushes its answer up until it is only too low
about 10% of the time. That is exactly the 0.9 quantile. Lower pinball loss = better.

### LightGBM in one paragraph

LightGBM builds **decision trees** one after another. A decision tree is a flowchart
of yes/no questions ("is the 28-day average above 3?") ending in a prediction. Each
new tree is trained to fix the mistakes left by all the trees before it. This is
called **gradient boosting**. Add up a few hundred small trees and you get a strong
model. It is the standard tool for table-shaped data: fast, handles missing values,
needs no feature scaling. Setting `objective="quantile"` makes it use the pinball loss.

Settings used, in plain words:

- `learning_rate=0.05`: each tree only fixes a small part of the error (slow and steady).
- `num_leaves=63`: how complicated one tree may get.
- `min_child_samples=100`: a rule must apply to at least 100 rows (prevents memorising).
- `subsample`, `colsample_bytree = 0.8`: each tree sees a random 80% of rows and
  columns, which makes the trees differ and the total more robust.
- **Early stopping**: after each tree, score the validation set. If it hasn't improved
  for 50 trees, stop. This prevents **overfitting** (memorising the training data
  instead of learning patterns).

One model is trained per quantile. Because they are separate models, the 0.90 answer
is occasionally a hair above the 0.95 answer ("quantile crossing"). We fix that by
sorting each row's seven predictions.

### The baselines

A model result means nothing without something to compare against.

- **Naive**: last 28 days' daily average x 10. No safety margin.
- **Normal formula** (what textbooks teach): `average x 10 + z x std x sqrt(10)`.
  It assumes demand follows a bell curve. `z` comes from the service level
  (95% gives z = 1.64). The `sqrt(10)` is because the uncertainty of a sum of 10
  independent days grows with the square root of the number of days.
- **Past-year quantile** (called `history` in the code): take the product's 10-day
  sales totals ending on each of the last 364 days, and read the quantile straight off
  that list. No bell-curve assumption, so it handles skewed demand. Its weakness: it
  treats a whole year as equally relevant, so it reacts slowly when a product's sales
  level changes.

### Results

Pinball loss (lower is better):

(store CA_1)

| Quantile | Naive | Normal formula | Past-year quantile | LightGBM |
|---|---|---|---|---|
| 0.90 | 2.832 | 1.524 | 1.649 | 1.207 |
| 0.95 | 2.877 | 1.113 | 1.038 | 0.766 |
| 0.99 | 2.912 | 0.602 | 0.345 | 0.236 |

**Calibration** asks: when you say 95%, does it happen 95% of the time?

| Asked for | Normal formula | Past-year quantile | LightGBM |
|---|---|---|---|
| 0.90 | 0.814 | 0.879 | 0.896 |
| 0.95 | 0.863 | 0.927 | 0.946 |
| 0.99 | 0.921 | 0.971 | 0.988 |

LightGBM is almost exactly right. The normal formula promises 95% and delivers 86%.
The past-year quantile is in between.

**Why does the normal formula fail?** A bell curve is symmetric. Real demand for a slow
product is not: mostly 0s and 1s, with an occasional 15. That long right tail is where
high quantiles live, and the bell curve underestimates it. LightGBM makes no shape
assumption, it just learns where the 95% line is from data.

**Which features mattered?** `mean_28` (63% of the gain) and `mean_7` (17%). Recent
sales level dominates. This is normal for retail and worth saying honestly: the fancy
features add a little, the recent average does most of the work.

## 6. Step 4: the inventory simulation (`simulator.py`, `step4_simulate_inventory.py`)

### The ordering rule

An **order-up-to policy**, the simplest standard rule:

1. Once a week, count the stock (on the shelf + already ordered but not yet arrived).
2. Order enough to bring it back up to a target level.
3. The order arrives 3 days later.
4. If a customer comes and the shelf is empty, the sale is lost.

The target level is just the quantile forecast. "0.95 service level" means the target
is the 0.95 quantile of the next 10 days' demand. So the three methods differ only in
where the target comes from.

The simulator is a loop over days: deliveries arrive, sell what you can, and on
ordering days top up. Read `simulator.py`, it is 40 lines.

### What we measure

- **Fill rate**: units sold / units demanded. 95% means 5 in 100 units were missed.
- **Average inventory value**: money sitting on shelves on a typical day.

These fight each other. More stock means higher fill rate but more money tied up. So
you can't compare methods with one number. You draw the **trade-off curve**: for each
service level, plot (inventory, fill rate). A better method is higher and further left.

### Results

Store CA_1, average inventory value needed:

| To reach this fill rate | Normal formula | Past-year quantile | LightGBM | Saved vs the better baseline |
|---|---|---|---|---|
| 90% | $67,362 | $74,157 | $64,619 | 4.1% |
| 95% | $91,990 | $99,710 | $86,003 | 6.5% |
| 97% | $112,895 | $118,937 | $100,402 | 11.1% |

The naive method (no safety stock) only reaches an 83% fill rate.

**Does it hold in other stores?** `run_all_stores.py` repeats everything for one store
in each state, each with its own model:

| Store | 90% | 95% | 97% | LightGBM coverage at 0.95 |
|---|---|---|---|---|
| CA_1 | 4.1% less | 6.5% less | 11.1% less | 94.6% |
| TX_1 | 5.9% less | 7.1% less | 7.2% less | 94.7% |
| WI_1 | 6.4% less | 8.7% less | 10.6% less | 94.5% |

(always compared with whichever baseline did better at that fill rate)

**A surprise worth understanding.** The past-year quantile is *better calibrated* than
the normal formula (92.7% vs 86.3%), yet in CA_1 and TX_1 it needs *more* inventory for
the same fill rate. How? Calibration only says "right on average across all products".
The past-year method is slow: for a product whose sales have dropped it still stocks for
last year's level (wasted inventory), and for a growing product it stocks too little.
Those errors cancel in the coverage number but both hurt in the store. The lesson:
a good quantile forecast must be calibrated **and** sharp (specific to the current
situation). Pinball loss measures both at once, which is why it is the main score.

The gap widens at high service levels, which makes sense: that is where knowing the
*tail* of demand matters, and the tail is exactly what the bell curve gets wrong.

Note "service level" and "fill rate" are different things. Service level 0.95 = in 95%
of weeks we don't run out at all. Fill rate = share of units served. That's why the
0.95 policy gives a 98% fill rate: even in a bad week you still sell most units.

### The money assumptions

M5 has no cost data, so `config.py` assumes a 30% margin and 25% yearly holding cost.
These are used for the "lost profit" and "holding cost" columns only. The table above
(inventory needed for the same fill rate) does not depend on them, which is why it is
the headline result.

## 7. Limitations (say these before the interviewer does)

1. **Sales are not demand.** Empty-shelf days look like zero demand. The simulation
   replays recorded sales, so true lost sales are underestimated for every method.
2. **Costs are assumed**, not real.
3. **Three stores, one 26-week period.** Not tested on other time periods, and there
   are 10 stores in the dataset.
4. **Simplified store**: fixed 3-day delivery, no minimum pack sizes, no shelf limits,
   no food going off.
5. **The baselines are classical but not exhaustive.** Methods built for intermittent
   demand (Croston, for example) were not tried.
6. The improvement is modest (4 to 11%), not magic. That is realistic.

## 8. Interview questions and answers

**Why not just predict the average?**
Stocking the average means running out about half the time. The decision needs a high
quantile, so I predict that directly.

**Why LightGBM and not LSTM / ARIMA / Prophet?**
3,049 products with mostly-zero sales. ARIMA/Prophet fit one model per series and
struggle with intermittent data. One LightGBM model learns from all products at once,
trains in minutes, and tree models won the M5 competition. I'd need a reason to use
something more complex.

**How did you prevent leakage?**
Time-based split, backward-looking features only, and a 10-day gap between train and
test because the target looks 10 days ahead.

**What is pinball loss?**
The loss for quantiles. Too low costs q per unit, too high costs 1-q. Minimising it
gives the q-th quantile.

**How do you know the quantiles are right?**
Calibration: on the test set, the 0.95 forecast was above actual demand 94.6% of the
time.

**Why 10 days?**
Review period (7) + lead time (3). An order has to cover demand until the next order
arrives.

**Why did the normal formula lose?**
Demand is skewed with many zeros, not bell-shaped. It under-covers: 86% instead of 95%.

**Your most important feature is just the 28-day average. Is the model doing anything?**
Yes: at the same fill rate it needs 4 to 11% less stock, and its quantiles are
calibrated where the formula's are not. Most of the signal is recent sales, and the
model's gain is in how it shapes the uncertainty around it.

**What is the biggest weakness?**
Censored demand. I only see sales. Next step would be to estimate true demand on
stockout days.

**Did you try a stronger baseline than the textbook formula?**
Yes, the quantile of each product's own 10-day totals over the past year. It was
better calibrated than the formula but needed more inventory in two of three stores,
because it adapts slowly. LightGBM beat the better of the two by 4 to 11% everywhere.

**Does it generalise?**
I ran three stores in three states with separate models. Savings were 4 to 11% in all
of them and coverage at the 95% quantile was 94.5 to 94.7%.

**Can a forecast be calibrated and still bad?**
Yes. Calibration is an average over products. A method can over-stock some products
and under-stock others and still look calibrated. You also need sharpness; pinball
loss captures both.

**What would you do next?**
Handle censored demand, run all 10 stores and more time periods, try Croston-type
baselines, add pack sizes and spoilage for food.

**What is overfitting and how did you control it?**
Memorising training data. Controlled with early stopping on a validation period,
minimum 100 rows per leaf, and row/column subsampling.

**Fill rate vs service level?**
Service level: share of weeks with no stockout. Fill rate: share of units served.

## 9. How to describe it

**In 30 seconds:**
"I built a system that decides how much of each product a Walmart store should order
each week. It forecasts a range of demand for about 3,000 products per store using
LightGBM quantile regression, turns that into order quantities, and I backtested it by
simulating 26 weeks of ordering in three stores. Compared with the better of two
classical methods it reached the same fill rate with 4 to 11% less inventory, because
its uncertainty estimates were both calibrated and specific to each product."

**Resume bullets:**

- Built an end-to-end demand forecasting and inventory pipeline for ~3,000 products in
  each of 3 Walmart stores (M5 dataset, 2.8M training rows per store) using LightGBM
  quantile regression.
- Achieved well-calibrated forecasts (94.6% actual coverage at the 95% quantile vs
  86.3% for the standard safety-stock formula) and 31% lower pinball loss.
- Wrote an inventory simulator and backtested weekly ordering over 26 weeks: same fill
  rate with 4-11% less inventory than the better of two classical baselines, in all 3 stores.
- Built a Streamlit dashboard to explore forecasts and simulated stock levels per product.

## 10. Make it yours

You'll understand it far better (and have honest answers in interviews) if you change
things and see what happens. Each takes minutes:

1. Add a fourth store (say `CA_3`) to `STORES` in `run_all_stores.py`. Do the results hold?
2. Change `LEAD_TIME` to 7. What happens to the inventory needed?
3. Remove `planned_price_change` from the features. Does anything change?
4. Remove `mean_28`. How much worse does it get?
5. Add a feature of your own (for example the max daily sales in the last 28 days).

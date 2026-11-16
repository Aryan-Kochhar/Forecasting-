"""Dashboard to look at the results.

It reads the small result files in dashboard/data (written by step 4), so it
works without the raw Walmart data.

Run:  streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import DASHBOARD_DATA, LEAD_TIME, MARGIN, REVIEW_DAYS, SERVICE_LEVELS
from simulator import simulate

st.set_page_config(page_title="Walmart inventory forecasting", layout="wide")
NAMES = {"naive": "Naive", "normal": "Normal formula", "history": "Past-year quantile", "lgbm": "LightGBM"}
SAFETY_METHODS = ["normal", "history", "lgbm"]     # the ones that have a service level


@st.cache_data
def load(store):
    folder = DASHBOARD_DATA / store
    return (
        pd.read_parquet(folder / "sales.parquet"),
        pd.read_parquet(folder / "prices.parquet"),
        pd.read_parquet(folder / "test_forecasts.parquet"),
        pd.read_csv(folder / "inventory_summary.csv"),
        pd.read_csv(folder / "forecast_scores.csv"),
        pd.read_csv(folder / "matched_comparison.csv"),
    )


stores = sorted(p.name for p in DASHBOARD_DATA.iterdir() if p.is_dir())
store = st.sidebar.selectbox("Store", stores)
st.sidebar.caption("Each store has its own model, trained only on that store's history.")
sales, prices, forecasts, summary, scores, matched = load(store)

st.title(f"Walmart store {store}: forecast demand, then decide how much to stock")
tab_results, tab_product = st.tabs(["Overall results", "Look at one product"])

# ------------------------------------------------------------------ results
with tab_results:
    st.caption("26-week backtest on data the model never saw. Same store, same demand, "
               "four ways of deciding the stock level.")
    cols = st.columns(len(matched))
    for col, row in zip(cols, matched.itertuples()):
        best_baseline = min(row.normal, row.history)   # compare against whichever baseline did better
        col.metric(
            f"Inventory needed for {row.fill_rate:.0%} fill rate",
            f"${row.lgbm:,.0f}",
            f"{row.lgbm / best_baseline - 1:.1%} vs best baseline (${best_baseline:,.0f})",
            delta_color="inverse",
        )

    left, right = st.columns(2)

    fig = go.Figure()
    for method in SAFETY_METHODS:
        part = summary[summary["method"] == method]
        fig.add_scatter(x=part["avg_inventory_value"], y=part["fill_rate"] * 100, mode="lines+markers",
                        name=NAMES[method], text=part["service_level"],
                        hovertemplate="service level %{text}<br>$%{x:,.0f}<br>%{y:.1f}%")
    naive = summary[summary["method"] == "naive"]
    fig.add_scatter(x=naive["avg_inventory_value"], y=naive["fill_rate"] * 100, mode="markers",
                    name="Naive (no safety stock)", marker=dict(symbol="square", size=10))
    fig.update_layout(title="Service vs. money on the shelves (up and left is better)",
                      xaxis_title="average inventory value ($)", yaxis_title="fill rate (%)")
    left.plotly_chart(fig, width="stretch")

    cov = scores.pivot(index="quantile", columns="method", values="coverage")
    fig = go.Figure()
    fig.add_scatter(x=[0.5, 1], y=[0.5, 1], mode="lines", name="Perfect", line=dict(dash="dash", color="gray"))
    for method in SAFETY_METHODS:
        fig.add_scatter(x=cov.index, y=cov[method], mode="lines+markers", name=NAMES[method])
    fig.update_layout(title="Are the forecasts honest? (calibration)",
                      xaxis_title="quantile we asked for", yaxis_title="how often demand was below it")
    right.plotly_chart(fig, width="stretch")

    st.subheader("Every policy that was simulated")
    st.dataframe(summary.assign(method=summary["method"].map(NAMES)), hide_index=True, width="stretch")

# ------------------------------------------------------------------ one product
with tab_product:
    c1, c2, c3 = st.columns(3)
    top_sellers = forecasts.groupby("item_id")["actual"].sum().sort_values(ascending=False).index
    item = c1.selectbox("Product (best sellers first)", top_sellers)
    method = c2.selectbox("Who decides the stock level", ["lgbm", "history", "normal", "naive"],
                          format_func=NAMES.get)
    level = c3.select_slider("Service level", SERVICE_LEVELS, value=0.95)

    f = forecasts[forecasts["item_id"] == item].sort_values("date")
    days = sales.index
    demand = sales[[item]].to_numpy()
    r = simulate(demand, f[[f"{method}_{level}"]].to_numpy(), REVIEW_DAYS, LEAD_TIME)

    price = float(prices[item].mean())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Fill rate", f"{r['sold'].sum() / max(demand.sum(), 1):.1%}")
    m2.metric("Units lost to empty shelf", f"{r['lost'].sum():.0f}")
    m3.metric("Average units on shelf", f"{r['stock'].mean():.0f}")
    m4.metric("Lost profit", f"${r['lost'].sum() * price * MARGIN:,.0f}")

    fig = go.Figure()
    fig.add_scatter(x=days, y=r["stock"][:, 0], name="Stock at end of day", fill="tozeroy")
    fig.add_bar(x=days, y=demand[:, 0], name="Customer demand")
    fig.add_bar(x=days, y=r["lost"][:, 0], name="Lost sales", marker_color="red")
    fig.update_layout(title="The simulated shelf, day by day", barmode="overlay", yaxis_title="units")
    st.plotly_chart(fig, width="stretch")

    fig = go.Figure()
    fig.add_scatter(x=f["date"], y=f["actual"], name="What really sold in the next 10 days", mode="lines+markers")
    fig.add_scatter(x=f["date"], y=f["lgbm_0.5"], name="LightGBM middle guess (0.5)")
    for m in SAFETY_METHODS:
        fig.add_scatter(x=f["date"], y=f[f"{m}_{level}"], name=f"{NAMES[m]} {level}", line=dict(dash="dot"))
    fig.update_layout(title="Forecast made on each ordering day vs. what happened", yaxis_title="units in next 10 days")
    st.plotly_chart(fig, width="stretch")

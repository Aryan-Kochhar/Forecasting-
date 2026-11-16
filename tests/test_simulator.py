"""Small checks that the simulator behaves like a real stockroom.
Run:  pytest
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from simulator import simulate


def test_units_are_never_created_or_destroyed():
    rng = np.random.default_rng(0)
    demand = rng.poisson(3, size=(28, 5)).astype(float)
    targets = np.full((4, 5), 30.0)
    r = simulate(demand, targets, review_days=7, lead_time=3)

    # every unit of demand is either sold or lost
    assert np.allclose(r["sold"] + r["lost"], demand)
    # stock is never negative
    assert (r["stock"] >= 0).all()
    # starting stock + deliveries that arrived - sales = final stock
    arrived = r["ordered"][: 28 - 3 - 1].sum(axis=0)
    assert np.allclose(30 + arrived - r["sold"].sum(axis=0), r["stock"][-1])


def test_huge_target_means_no_lost_sales():
    demand = np.full((14, 1), 5.0)
    r = simulate(demand, np.full((2, 1), 1000.0), review_days=7, lead_time=3)
    assert r["lost"].sum() == 0


def test_order_arrives_after_lead_time():
    demand = np.zeros((14, 1))
    demand[0] = 10                      # sell 10 on day 0, nothing afterwards
    r = simulate(demand, np.full((2, 1), 10.0), review_days=7, lead_time=3)
    assert r["ordered"][6, 0] == 10     # reordered at the end of the first week
    assert r["stock"][9, 0] == 0        # not here yet on day 9
    assert r["stock"][10, 0] == 10      # arrives on the morning of day 10

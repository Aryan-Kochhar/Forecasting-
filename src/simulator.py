"""A tiny inventory simulator.

The rule we simulate is called an "order-up-to" policy:
  - once a week we look at how much stock we have (on the shelf + already ordered)
  - we order enough to bring that back up to a target level
  - the order arrives LEAD_TIME days later
  - if a customer wants a product and the shelf is empty, that sale is lost

The only thing that differs between policies is how the target level is chosen.
"""

import numpy as np


def simulate(demand, targets, review_days, lead_time):
    """Play the store forward one day at a time.

    demand  : array [days, products]   - units customers wanted each day
    targets : array [weeks, products]  - order-up-to level decided each week.
              targets[0] is the level we start with; targets[w] is used for the
              order placed at the end of week w.
    Returns daily arrays: stock left at end of day, units sold, units lost, units ordered.
    """
    n_days, n_products = demand.shape
    on_hand = np.ceil(targets[0]).astype(float)       # start with full shelves
    arriving = np.zeros((n_days + lead_time + 1, n_products))  # orders on their way, by arrival day

    stock = np.zeros_like(demand, dtype=float)
    sold = np.zeros_like(demand, dtype=float)
    lost = np.zeros_like(demand, dtype=float)
    ordered = np.zeros_like(demand, dtype=float)

    for day in range(n_days):
        # 1. morning: deliveries arrive
        on_hand += arriving[day]

        # 2. during the day: sell what we can
        sold[day] = np.minimum(on_hand, demand[day])
        lost[day] = demand[day] - sold[day]
        on_hand -= sold[day]
        stock[day] = on_hand

        # 3. evening of an ordering day: top up to the target level
        week = (day + 1) // review_days
        if (day + 1) % review_days == 0 and week < len(targets):
            on_order = arriving[day + 1 :].sum(axis=0)
            order = np.maximum(np.ceil(targets[week]) - on_hand - on_order, 0)
            arriving[day + 1 + lead_time] += order
            ordered[day] = order

    return {"stock": stock, "sold": sold, "lost": lost, "ordered": ordered}

import micropip
await micropip.install("fiqua")

import json
import numpy as np
from fiqua.equities import Stock, EuropeanOption, portfolio_surface
from fiqua.core import Portfolio


def price_portfolio(positions, spot, r, sigma, T):
    try:
        stock = Stock(spot=spot, rate=r, volatility=sigma)
        legs = [
            (
                EuropeanOption(
                    underlying=stock,
                    strike=p["strike"],
                    maturity=T,
                    option_type=p["type"],
                ),
                p["quantity"],
            )
            for p in positions
        ]
        portfolio = Portfolio(positions=legs)
        result = portfolio_surface(portfolio)
    except (ValueError, TypeError) as e:
        return {"success": False, "error": str(e)}

    if not result.solved:
        return {"success": False, "error": result.error_msg}

    grid_S = result.grid_x
    grid_t_full = result.grid_t
    S_max = float(grid_S[-1])

    # Payoff is pure arithmetic (no PDE involved), so evaluate it on a much
    # finer grid than the PDE's -- the PDE grid (m=100 by default) is coarse
    # enough that payoff's kinks look jagged, unlike the value curve, which
    # is smooth by construction and doesn't need this.
    fine_grid_S = np.linspace(0.0, S_max, 400)
    payoff_curve = [
        sum(pos.quantity * pos.instrument.payoff(float(s)) for pos in portfolio.positions)
        for s in fine_grid_S
    ]

    # Subsample time steps for the slider: keeps the UI usable and the
    # payload small even when N is large; the solve itself still ran at
    # full resolution, this only thins which columns get shipped to the
    # browser.
    max_frames = 50
    n_t = len(grid_t_full)
    if n_t <= max_frames:
        idx = list(range(n_t))
    else:
        idx = sorted(set(round(i * (n_t - 1) / (max_frames - 1)) for i in range(max_frames)))

    grid_t = [float(grid_t_full[j]) for j in idx]
    value_grid = [result.solution[:, j].tolist() for j in idx]

    spot_value = float(np.interp(spot, grid_S, result.solution[:, 0]))
    spot_payoff = sum(pos.quantity * pos.instrument.payoff(spot) for pos in portfolio.positions)

    return {
        "success": True,
        "grid_S": grid_S.tolist(),
        "fine_grid_S": fine_grid_S.tolist(),
        "payoff_curve": payoff_curve,
        "grid_t": grid_t,
        "value_grid": value_grid,
        "S_max": S_max,
        "spot_value": spot_value,
        "spot_payoff": spot_payoff,
    }

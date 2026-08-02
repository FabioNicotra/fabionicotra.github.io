import micropip
await micropip.install("fiqua==0.2.0")

import json
import numpy as np
from fiqua.core import Portfolio
from fiqua.equities import BlackScholesPDEEngine, EuropeanOption, MarketData, Metric, PricingRequest, Stock, StockQuote

UNDERLYING_SYMBOL = "UNDERLYING"


def price_portfolio(positions, spot, r, sigma, T):
    try:
        stock = Stock(UNDERLYING_SYMBOL)
        market = MarketData(rate=r, quotes={UNDERLYING_SYMBOL: StockQuote(spot=spot, volatility=sigma)})
        engine = BlackScholesPDEEngine(market=market)
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
        engine.add([PricingRequest(instrument=portfolio, metrics=[Metric.PV])])
        priced = engine.run()[0]
    except (ValueError, TypeError) as e:
        return {"success": False, "error": str(e)}

    pv_metadata = priced.metadata.get("pv", {})
    if not priced.priced or "parabolic_result" not in pv_metadata:
        return {
            "success": False,
            "error": priced.error_msg or "Portfolio does not support a shared pricing surface.",
        }

    result = pv_metadata["parabolic_result"]

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

    # fiqua 0.2.0 doesn't expose a delta metric yet, but the PDE already
    # solved the full value surface over S -- differentiating that surface
    # w.r.t. S gives delta directly, no extra pricing call needed. Swap
    # this for the engine's native Metric.DELTA once that ships.
    delta_surface = np.gradient(result.solution, grid_S, axis=0)
    delta_grid = [delta_surface[:, j].tolist() for j in idx]

    spot_value = float(np.interp(spot, grid_S, result.solution[:, 0]))
    spot_delta = float(np.interp(spot, grid_S, delta_surface[:, 0]))
    spot_payoff = sum(pos.quantity * pos.instrument.payoff(spot) for pos in portfolio.positions)

    return {
        "success": True,
        "grid_S": grid_S.tolist(),
        "fine_grid_S": fine_grid_S.tolist(),
        "payoff_curve": payoff_curve,
        "grid_t": grid_t,
        "value_grid": value_grid,
        "delta_grid": delta_grid,
        "S_max": S_max,
        "spot_value": spot_value,
        "spot_delta": spot_delta,
        "spot_payoff": spot_payoff,
    }

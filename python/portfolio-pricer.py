import micropip
await micropip.install("fiqua==0.2.3")

import json
import numpy as np
from fiqua.core import Portfolio
from fiqua.equities import (
    BlackScholesPDEEngine,
    EuropeanOption,
    MarketData,
    Metric,
    PDEGrids,
    PDESolverSettings,
    PricingRequest,
    Stock,
    StockQuote,
)

UNDERLYING_SYMBOL = "UNDERLYING"


def price_portfolio(positions, spot, r, sigma, T, m=100, N=50, method="backward-difference", align_grid_to_strikes=True):
    try:
        stock = Stock(UNDERLYING_SYMBOL)
        market = MarketData(rate=r, quotes={UNDERLYING_SYMBOL: StockQuote(spot=spot, volatility=sigma)})
        settings = PDESolverSettings(
            m=int(m), N=int(N), method=method, align_grid_to_strikes=bool(align_grid_to_strikes)
        )
        engine = BlackScholesPDEEngine(market=market, settings=settings)
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
        engine.add([PricingRequest(instrument=portfolio, metrics=[Metric.PV, Metric.DELTA, Metric.GAMMA, Metric.THETA])])
        priced = engine.run()[0]
    except (ValueError, TypeError) as e:
        return {"success": False, "error": str(e)}

    pv_metadata = priced.metadata.get("pv", {})
    if not priced.priced or "parabolic_pde_result" not in pv_metadata:
        return {
            "success": False,
            "error": priced.error_msg or "Portfolio does not support a shared pricing surface.",
        }

    result = pv_metadata["parabolic_pde_result"]

    grid_S = result.grid_x
    grid_t_full = result.grid_t
    S_max = float(grid_S[-1])

    # Payoff is pure arithmetic (no PDE involved), so evaluate it on a much
    # finer grid than the PDE's -- the PDE grid (m=100 by default) is coarse
    # enough that payoff's kinks look jagged, unlike the value curve, which
    # is smooth by construction and doesn't need this. Strikes are folded
    # into the grid explicitly (union1d sorts + dedupes) so each kink lands
    # exactly on an evaluated point instead of being rounded off to
    # whichever linspace point happens to land nearby.
    strikes = [p["strike"] for p in positions]
    fine_grid_S = np.union1d(np.linspace(0.0, S_max, 400), strikes)
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

    # fiqua's PDEGrids (0.2.3+) hands back the already-solved value surface
    # as a smooth cubic spline per time slice, with exact derivatives (no
    # bump-and-reprice, no np.gradient approximation) -- value/delta/gamma
    # curves all come from the same pv_curve. Theta has no curve builder of
    # its own on PDEGrids, so it's a forward difference between adjacent
    # *full-resolution* solved time columns (fiqua's own _build_theta_curve
    # does the same thing, just for t=0 only) -- backward difference on the
    # last time step, since there's no t+1 to diff against there.
    grids = PDEGrids(priced)
    value_grid, delta_grid, gamma_grid, theta_grid = [], [], [], []
    for k in idx:
        pv_curve = grids.pv(float(grid_t_full[k]))
        value_grid.append(pv_curve(grid_S).tolist())
        delta_grid.append(pv_curve.derivative(grid_S, order=1).tolist())
        gamma_grid.append(pv_curve.derivative(grid_S, order=2).tolist())

        if k + 1 < n_t:
            pv_other = grids.pv(float(grid_t_full[k + 1]))
            dt = float(grid_t_full[k + 1] - grid_t_full[k])
            theta_row = (pv_other(grid_S) - pv_curve(grid_S)) / dt
        else:
            pv_other = grids.pv(float(grid_t_full[k - 1]))
            dt = float(grid_t_full[k] - grid_t_full[k - 1])
            theta_row = (pv_curve(grid_S) - pv_other(grid_S)) / dt
        theta_grid.append(theta_row.tolist())

    # Spot Greeks come straight from fiqua's own metrics rather than this
    # file re-deriving them from the grid -- more accurate (fiqua spline-
    # interpolates PV and derives delta/gamma/theta at the exact spot) and
    # the whole point of requesting them above.
    spot_value = float(priced.values[Metric.PV])
    spot_delta = float(priced.values[Metric.DELTA])
    spot_gamma = float(priced.values[Metric.GAMMA])
    spot_theta = float(priced.values[Metric.THETA])
    spot_payoff = sum(pos.quantity * pos.instrument.payoff(spot) for pos in portfolio.positions)

    return {
        "success": True,
        "grid_S": grid_S.tolist(),
        "fine_grid_S": fine_grid_S.tolist(),
        "payoff_curve": payoff_curve,
        "grid_t": grid_t,
        "value_grid": value_grid,
        "delta_grid": delta_grid,
        "gamma_grid": gamma_grid,
        "theta_grid": theta_grid,
        "S_max": S_max,
        "spot_value": spot_value,
        "spot_delta": spot_delta,
        "spot_gamma": spot_gamma,
        "spot_theta": spot_theta,
        "spot_payoff": spot_payoff,
    }

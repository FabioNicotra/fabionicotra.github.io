import micropip
await micropip.install("fiqua==0.3.0")

import json
import numpy as np
from fiqua.core import (
    CalculationEngine,
    CalculationRequest,
    Metric,
    PricingEngineRegistry,
    Trade,
)
from fiqua.equities import (
    EQUITY_OPTION_PRODUCT_TYPE,
    EQUITY_STRATEGY_PRODUCT_TYPE,
    BlackScholesPDEEngine,
    MarketData,
    PDESolverSettings,
    SolvedGrid,
    StockQuote,
)

UNDERLYING_SYMBOL = "UNDERLYING"
TRADE_ID = "portfolio"
METRICS = [Metric.PV, Metric.DELTA, Metric.GAMMA, Metric.THETA]


def _calculation_engine(market, settings):
    """A CalculationEngine that prices a multi-leg equity strategy on the
    finite-difference engine, solving with `settings`.

    Its own registry rather than fiqua's shared one: this page's grid
    controls *are* a PDESolverSettings, and only a registration carries
    constructor kwargs through to the engine CalculationEngine builds for
    itself -- the shared registration would take the PDE engine's defaults.
    """
    engines = PricingEngineRegistry()
    engines.register(
        EQUITY_STRATEGY_PRODUCT_TYPE, BlackScholesPDEEngine, default=True, settings=settings
    )
    return CalculationEngine(market, engines=engines)


def _portfolio_trade(positions, T):
    """The whole portfolio booked as one multi-leg trade, in the shape
    fiqua's product resolver turns into a CompositeInstrument -- so this
    file never constructs an instrument itself."""
    return Trade(
        trade_id=TRADE_ID,
        product_type=EQUITY_STRATEGY_PRODUCT_TYPE,
        quantity=1,
        terms={
            "legs": [
                {
                    "product_type": EQUITY_OPTION_PRODUCT_TYPE,
                    "quantity": p["quantity"],
                    "terms": {
                        "underlying": UNDERLYING_SYMBOL,
                        "strike": p["strike"],
                        "maturity": T,
                        "option_type": p["type"],
                    },
                }
                for p in positions
            ]
        },
    )


def _run(engine, trade, metrics):
    """`metrics` for `trade`, through CalculationEngine's queue-then-run
    surface -- the one entry point every request on this page goes through."""
    engine.add([CalculationRequest(trade=trade, metrics=metrics)])
    return engine.run()[TRADE_ID]


def price_portfolio(positions, spot, r, sigma, T, m=200, N=100, method="backward-difference", align_grid_to_strikes=True):
    try:
        market = MarketData(
            rate=r, quotes={UNDERLYING_SYMBOL: StockQuote(spot=spot, volatility=sigma)}
        )
        trade = _portfolio_trade(positions, T)
        engine = _calculation_engine(
            market,
            PDESolverSettings(
                m=int(m), N=int(N), method=method, align_grid_to_strikes=bool(align_grid_to_strikes)
            ),
        )
        priced = _run(engine, trade, METRICS)

        # fiqua isolates per-metric failures: a result can come back priced
        # with some of the requested metrics missing, reported in
        # metadata["failed_metrics"]. Every metric here backs a curve the
        # page draws, so a partial answer is no answer.
        failed = priced.metadata.get("failed_metrics", {})
        if not priced.priced or failed:
            return {"success": False, "error": priced.error_msg or "; ".join(failed.values())}

        # The solved surface every curve below is read off, dug out of the
        # result's own metadata -- raises if this result carries no grid
        # (e.g. legs that can't share one mesh), caught alongside the rest.
        grid = SolvedGrid.from_result(priced)
    except (ValueError, TypeError) as e:
        return {"success": False, "error": str(e)}

    grid_S = grid.raw.grid_x
    grid_t_full = grid.times
    S_max = float(grid_S[-1])

    # Payoff is pure arithmetic (no PDE involved), so evaluate it on a much
    # finer grid than the PDE's -- the PDE grid (m=200 by default) is coarse
    # enough that payoff's kinks look jagged, unlike the value curve, which
    # is smooth by construction and doesn't need this. Strikes are folded
    # into the grid explicitly (union1d sorts + dedupes) so each kink lands
    # exactly on an evaluated point instead of being rounded off to
    # whichever linspace point happens to land nearby.
    #
    # The legs come back as per-leg attribution on the priced result, so the
    # payoff is summed over the same instruments fiqua resolved and priced,
    # not a second reading of the raw position dicts.
    legs = priced.metadata[Metric.PV.value]["positions"]
    strikes = [p["strike"] for p in positions]
    fine_grid_S = np.union1d(np.linspace(0.0, S_max, 400), strikes)
    payoff_curve = [
        sum(pos.quantity * pos.instrument.payoff(float(s)) for pos, _ in legs)
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

    # SolvedGrid owns every read off the solved surface: value, delta, gamma
    # and theta all come from the one grid, each an exact spline derivative
    # rather than a bump-and-reprice or a finite difference this file would
    # otherwise have to roll itself.
    value_grid, delta_grid, gamma_grid, theta_grid = [], [], [], []
    for k in idx:
        t = float(grid_t_full[k])
        value_grid.append(grid.pv(t)(grid_S).tolist())
        delta_grid.append(grid.delta(t)(grid_S).tolist())
        gamma_grid.append(grid.gamma(t)(grid_S).tolist())
        theta_grid.append(grid.theta(t)(grid_S).tolist())

    # Spot Greeks come straight from fiqua's own metrics rather than this
    # file re-deriving them from the grid -- more accurate (fiqua spline-
    # interpolates PV and derives delta/gamma/theta at the exact spot) and
    # the whole point of requesting them above.
    spot_payoff = sum(pos.quantity * pos.instrument.payoff(spot) for pos, _ in legs)

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
        "spot_value": float(priced.values[Metric.PV]),
        "spot_delta": float(priced.values[Metric.DELTA]),
        "spot_gamma": float(priced.values[Metric.GAMMA]),
        "spot_theta": float(priced.values[Metric.THETA]),
        "spot_payoff": spot_payoff,
    }

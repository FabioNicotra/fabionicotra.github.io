import micropip
await micropip.install("fiqua==0.3.0")

import json
import numpy as np
from fiqua.core import (
    CalculationEngine,
    CalculationRequest,
    Metric,
    MetricUnit,
    PricingEngineRegistry,
    Trade,
)
from fiqua.equities import (
    EQUITY_OPTION_PRODUCT_TYPE,
    EQUITY_STRATEGY_PRODUCT_TYPE,
    BlackScholesPDEEngine,
    BumpDirection,
    MarketData,
    MarketDataBump,
    PDESolverSettings,
    SolvedGrid,
    StockQuote,
)
from numanlib.pdes import SpatiotemporalDomain1D

UNDERLYING_SYMBOL = "UNDERLYING"
TRADE_ID = "portfolio"
METRICS = [Metric.PV, Metric.DELTA, Metric.GAMMA, Metric.THETA, Metric.VEGA, Metric.RHO]

# Desk quoting conventions rather than the units fiqua computes in: theta
# per year is meaningless to read next to a one-year option, and vega/rho
# per unit of vol/rate are a hundred times the move anyone quotes. Applied
# to the spot numbers via PricerResult.in_units() and to the curves via the
# same MetricUnit.scale, so the axis and the headline never disagree.
DISPLAY_UNITS = {
    Metric.THETA: MetricUnit.PER_CALENDAR_DAY,
    Metric.VEGA: MetricUnit.PER_PERCENT,
    Metric.RHO: MetricUnit.PER_PERCENT,
}

# Which market factor each bumped metric differentiates against, and the
# step to bump it by -- both matching fiqua's own bump-and-revalue defaults,
# so a curve here lands on the number fiqua reports at spot.
BUMPED_METRICS = (
    (Metric.VEGA, lambda h, d: MarketDataBump.volatility(UNDERLYING_SYMBOL, h, direction=d), 1e-4),
    (Metric.RHO, lambda h, d: MarketDataBump.rate(h, direction=d), 1e-4),
)


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


def _sensitivity_surface(engine, trade, market, bump_for, h):
    """dV/dx over the whole solved surface, as a central difference of two
    bumped solves.

    SolvedGrid offers no vega/rho curve: fiqua reaches those by bumping and
    repricing, which answers at spot rather than across it. Differencing
    entire surfaces generalizes the same idea to every spot at once.
    `engine` must be pinned to an explicit domain -- an auto-sized mesh
    moves under a vol bump, and the two surfaces would stop lining up node
    for node, making the subtraction meaningless.
    """
    solutions = []
    for direction in (BumpDirection.ABOVE, BumpDirection.BELOW):
        bumped = _run(
            engine.replace(new_market=market.bump([bump_for(h, direction)])), trade, [Metric.PV]
        )
        if not bumped.priced:
            raise ValueError(bumped.error_msg)
        solutions.append(SolvedGrid.from_result(bumped).raw.solution)
    return (solutions[0] - solutions[1]) / (2 * h)


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
        # with some requested metrics missing, each explained in
        # metadata["failed_metrics"]. Those are reported per metric rather
        # than failing the page -- vega and rho are the ones that drop out
        # first, on a grid too coarse to bump and reprice against. PV is the
        # exception: without it there is no surface and no payoff to draw.
        failed = priced.metadata.get("failed_metrics", {})
        if not priced.priced or Metric.PV not in priced.values:
            return {"success": False, "error": priced.error_msg or failed[Metric.PV.value]}

        # The solved surface every curve below is read off, dug out of the
        # result's own metadata -- raises if this result carries no grid
        # (e.g. legs that can't share one mesh), caught alongside the rest.
        grid = SolvedGrid.from_result(priced)
        grid_S = grid.raw.grid_x
        grid_t_full = grid.times
        S_max = float(grid_S[-1])

        # vega and rho have no curve on the solved grid, so each one costs a
        # pair of bumped solves. Pinned to the base mesh by handing it an
        # explicit domain, and skipped entirely for a metric that already
        # failed above, since nothing would read the result.
        pinned = _calculation_engine(
            market,
            PDESolverSettings(
                domain=SpatiotemporalDomain1D(l=S_max, T=T),
                m=len(grid_S) - 1,
                N=len(grid_t_full) - 1,
                method=method,
            ),
        )
        surfaces = {
            metric: _sensitivity_surface(pinned, trade, market, bump_for, h)
            for metric, bump_for, h in BUMPED_METRICS
            if metric in priced.values
        }
    except (ValueError, TypeError) as e:
        return {"success": False, "error": str(e)}

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
    # otherwise have to roll itself. vega and rho are the two it has no
    # curve for, and come from the bumped surfaces above instead.
    curve_sources = {
        Metric.PV: lambda t, k: grid.pv(t)(grid_S),
        Metric.DELTA: lambda t, k: grid.delta(t)(grid_S),
        Metric.GAMMA: lambda t, k: grid.gamma(t)(grid_S),
        Metric.THETA: lambda t, k: grid.theta(t)(grid_S),
        Metric.VEGA: lambda t, k: surfaces[Metric.VEGA][:, k],
        Metric.RHO: lambda t, k: surfaces[Metric.RHO][:, k],
    }
    grids = {
        metric.value: [
            (curve(float(grid_t_full[k]), k) * DISPLAY_UNITS.get(metric, MetricUnit.NATIVE).scale).tolist()
            for k in idx
        ]
        for metric, curve in curve_sources.items()
        if metric in priced.values
    }

    # Spot Greeks come straight from fiqua's own metrics rather than this
    # file re-deriving them from the grid -- more accurate (fiqua spline-
    # interpolates PV and derives delta/gamma/theta at the exact spot) and
    # the whole point of requesting them above. in_units() converts without
    # touching the values the engine computed, so nothing scaled ever gets
    # fed back into a request.
    spot_metrics = {
        metric.value: metric_value.value
        for metric, metric_value in priced.in_units(DISPLAY_UNITS).items()
    }

    return {
        "success": True,
        "grid_S": grid_S.tolist(),
        "fine_grid_S": fine_grid_S.tolist(),
        "payoff_curve": payoff_curve,
        "grid_t": grid_t,
        "grids": grids,
        "spot": spot_metrics,
        "S_max": S_max,
    }

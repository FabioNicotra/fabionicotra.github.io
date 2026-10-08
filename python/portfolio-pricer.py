import micropip
await micropip.install("fiqua==0.5.0")

import json
from datetime import date, timedelta

import numpy as np
from fiqua.core import (
    CalculationEngine,
    CompoundingFrequency,
    CompositeInstrument,
    CurveId,
    Extrapolation,
    Metric,
    MetricSpec,
    MetricUnit,
    Position,
    Tenor,
    ValuationQuery,
    ZeroCurve,
)
from fiqua.core.curve_id import DISC, ZERO
from fiqua.equities import BlackScholesPDEEngine, EuropeanOption, PDESolverSettings, SolvedGrid, Stock, StockQuote
from fiqua.market import DummyMarketDataProvider, FetchedValue, Identifier, MarketDataProviderRegistry
from numanlib.pdes import SpatiotemporalDomain1D

UNDERLYING_SYMBOL = "UNDERLYING"
CURRENCY = "USD"
ZERO_CURVE_ID = CurveId(CURRENCY, ZERO, CurveId(CURRENCY, "FLAT"))
DISCOUNT_CURVE_ID = CurveId(CURRENCY, DISC, ZERO_CURVE_ID)
# Fixed once at module load (this file runs once per page load; price_portfolio
# is called repeatedly against the same valuation date) -- purely a pricing
# "as of" reference, since every curve point below is quoted at the same flat
# rate regardless of tenor.
VALUATION_DATE = date.today()
# The registry's default engine for an option is the closed form, so every
# metric names the PDE engine: the page draws the solved grid.
PDE_ENGINE = BlackScholesPDEEngine.name
METRICS = [
    MetricSpec(metric, engine=PDE_ENGINE)
    for metric in (Metric.PV, Metric.DELTA, Metric.GAMMA, Metric.THETA, Metric.VEGA, Metric.RHO)
]

# Desk quoting conventions, not the units fiqua computes in: theta per
# year is meaningless to read next to a one-year option, and vega/rho per
# unit of vol/rate are a hundred times the move anyone quotes. Applied to
# the spot numbers via PricerResult.convert_to_units() and to the curves via
# the same MetricUnit.scale, so the axis and the headline never disagree.
DISPLAY_UNITS = {
    Metric.THETA: MetricUnit.PER_CALENDAR_DAY,
    Metric.VEGA: MetricUnit.PER_PERCENT,
    Metric.RHO: MetricUnit.PER_PERCENT,
}

# The step vega and rho are bumped by, matching fiqua's own bump-and-revalue
# default, so a curve here lands on the number fiqua reports at spot.
BUMP_SIZE = 1e-4


def _price(instrument, metrics, settings, spot, sigma, r):
    """The single PricerResult for `instrument` valued on a market of one
    stock at `spot` and `sigma`, discounted off a zero curve flat at `r`.

    A CalculationEngine holds no market of its own, so each call builds its
    own provider and engine; a bumped market is another call with the bumped
    input.
    """
    zero_curve = ZeroCurve(
        curve_id=ZERO_CURVE_ID,
        points={tenor: r for tenor in (Tenor.M1, Tenor.M3, Tenor.Y1, Tenor.Y5, Tenor.Y10, Tenor.Y30)},
        as_of=VALUATION_DATE,
        compounding=CompoundingFrequency.CONTINUOUS,
        extrapolation=Extrapolation.FLAT,
    )
    mdp = MarketDataProviderRegistry()
    mdp.register(
        DummyMarketDataProvider(
            {
                Identifier.stock_quote(UNDERLYING_SYMBOL): FetchedValue(StockQuote(spot=spot, currency=CURRENCY)),
                Identifier.volatility(UNDERLYING_SYMBOL): FetchedValue(sigma),
                Identifier.curve(ZERO_CURVE_ID): FetchedValue(zero_curve),
            },
            VALUATION_DATE,
        )
    )
    engine = CalculationEngine(VALUATION_DATE, mdp=mdp)
    engine.add(
        [
            ValuationQuery(
                instrument=instrument,
                metrics=metrics,
                discount_curve=DISCOUNT_CURVE_ID,
                engine_config=settings,
            )
        ]
    )
    return engine.run()[0]


def _sensitivity_surface(settings, instrument, spot, sigma, r, bump):
    """dV/dx over the whole solved surface, as a central difference of two
    bumped solves.

    SolvedGrid offers no vega/rho curve: fiqua reaches those by bumping and
    repricing, which answers only at spot. Differencing entire surfaces
    generalizes the same idea to every spot at once. `bump` is "volatility"
    or "rate", the input moved by +/-BUMP_SIZE. `settings` must pin an
    explicit domain -- an auto-sized mesh moves under a vol bump, and the
    two surfaces would stop lining up node for node, making the subtraction
    meaningless.
    """
    h = BUMP_SIZE
    solutions = []
    for sign in (1, -1):
        bumped_sigma = sigma + sign * h if bump == "volatility" else sigma
        bumped_r = r + sign * h if bump == "rate" else r
        bumped = _price(instrument, [MetricSpec(Metric.PV, engine=PDE_ENGINE)], settings, spot, bumped_sigma, bumped_r)

        if not bumped.priced:
            raise ValueError(bumped.error_msg)
        solutions.append(SolvedGrid.from_result(bumped).raw.solution)
    return (solutions[0] - solutions[1]) / (2 * h)


def price_portfolio(positions, spot, r, sigma, T, m=200, N=100, method="backward-difference", align_grid_to_strikes=True):
    try:
        expiry_date = VALUATION_DATE + timedelta(days=round(T * 365))
        stock = Stock(UNDERLYING_SYMBOL, currency=CURRENCY)
        legs = [
            Position(EuropeanOption(stock, p["strike"], expiry_date, p["type"]), p["quantity"])
            for p in positions
        ]
        strategy = CompositeInstrument(positions=legs)

        settings = PDESolverSettings(
            m=int(m), N=int(N), method=method, align_grid_to_strikes=bool(align_grid_to_strikes)
        )
        priced = _price(strategy, METRICS, settings, spot, sigma, r)

        # fiqua isolates per-metric failures: a result can come back priced
        # with some requested metrics missing, each listed in
        # priced.failures. Those are reported per metric rather than failing
        # the page -- vega and rho are the ones that drop out first, on a
        # grid too coarse to bump and reprice against. PV is the exception:
        # without it there is no surface and no payoff to draw.
        priced_metrics = {value.metric for value in priced.values}
        if not priced.priced or Metric.PV not in priced_metrics:
            failures = {failure.metric: failure.error_msg for failure in priced.failures}
            return {"success": False, "error": priced.error_msg or failures[Metric.PV]}

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
        pinned_settings = PDESolverSettings(
            domain=SpatiotemporalDomain1D(l=S_max, T=T),
            m=len(grid_S) - 1,
            N=len(grid_t_full) - 1,
            method=method,
        )
        surfaces = {
            metric: _sensitivity_surface(pinned_settings, strategy, spot, sigma, r, bump)
            for metric, bump in ((Metric.VEGA, "volatility"), (Metric.RHO, "rate"))
            if metric in priced_metrics
        }
    except (ValueError, TypeError) as e:
        return {"success": False, "error": str(e)}

    # Payoff is pure arithmetic (no PDE involved), so evaluate it on a much
    # finer grid than the PDE's -- the PDE grid (m=200 by default) is coarse
    # enough that payoff's kinks look jagged, unlike the value curve, which
    # is smooth by construction and doesn't need this. Strikes are folded
    # into the grid explicitly (union1d sorts + dedupes) so each kink lands
    # exactly on an evaluated point, never rounded off to whichever linspace
    # point happens to land nearby.
    strikes = [p["strike"] for p in positions]
    fine_grid_S = np.union1d(np.linspace(0.0, S_max, 400), strikes)
    payoff_curve = [
        sum(leg.quantity * leg.instrument.payoff(float(s)) for leg in legs)
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
    # and theta all come from the one grid, each an exact spline derivative.
    # vega and rho are the two it has no curve for, and come from the
    # bumped surfaces above.
    curve_sources = {
        Metric.PV: lambda t, k: grid.get_pv(t)(grid_S),
        Metric.DELTA: lambda t, k: grid.get_delta(t)(grid_S),
        Metric.GAMMA: lambda t, k: grid.get_gamma(t)(grid_S),
        Metric.THETA: lambda t, k: grid.get_theta(t)(grid_S),
        Metric.VEGA: lambda t, k: surfaces[Metric.VEGA][:, k],
        Metric.RHO: lambda t, k: surfaces[Metric.RHO][:, k],
    }
    grids = {
        metric.value: [
            (curve(float(grid_t_full[k]), k) * DISPLAY_UNITS.get(metric, MetricUnit.NATIVE).scale).tolist()
            for k in idx
        ]
        for metric, curve in curve_sources.items()
        if metric in priced_metrics
    }

    # Spot Greeks come straight from fiqua's own metrics: fiqua spline-
    # interpolates PV and derives delta/gamma/theta at the exact spot, more
    # accurate than this file re-deriving them from the grid.
    # convert_to_units() changes only how a value is read, so nothing scaled
    # ever gets fed back into a request.
    spot_metrics = {
        value.metric.value: value.get_number()
        for value in priced.convert_to_units(DISPLAY_UNITS).values
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

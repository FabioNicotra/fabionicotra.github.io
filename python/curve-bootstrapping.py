import micropip
await micropip.install(["fiqua==0.5.0", "requests", "pyodide-http"])

# fiqua's mirror providers fetch with `requests`; patching routes it through
# the browser's own XMLHttpRequest, the only network access Pyodide has.
import pyodide_http

pyodide_http.patch_all()

from dataclasses import replace
from datetime import date

import numpy as np
import requests

# Browsers refuse to let a page set these two headers, and log an error for
# every request that tries to; `requests` sends both by default.
_DEFAULT_HEADERS = requests.utils.default_headers()
for _name in ("Accept-Encoding", "Connection"):
    del _DEFAULT_HEADERS[_name]
requests.sessions.default_headers = lambda: _DEFAULT_HEADERS.copy()

from fiqua.core import CalculationEngine, CurveId, Interpolator, Tenor
from fiqua.market import (
    BondList,
    BondListEntry,
    BondQuote,
    BondTerms,
    DummyMarketDataProvider,
    FetchedValue,
    Identifier,
    MarketDataProviderRegistry,
    MirrorMarketDataProvider,
    MirrorReferenceDataProvider,
    PriceSide,
    ReferenceDataProvider,
    TreasurySecurityType,
)
from fiqua.rates import BootstrappedCurveRequest, ParYieldCurveRequest
from fiqua.rates.schedule import RollConvention

QUOTES_URL = "https://raw.githubusercontent.com/FabioNicotra/fiqua-market-snapshots/main/treasury_bond_quotes.json"
CURRENCY = "USD"
UNIVERSE = "UST-OTR"
CURVE_ID = CurveId(CURRENCY, UNIVERSE)
SIDES = {"bid": PriceSide.BID, "ask": PriceSide.ASK, "mid": PriceSide.MID}
CURVE_TYPES = (TreasurySecurityType.BILL, TreasurySecurityType.NOTE, TreasurySecurityType.BOND)

# A par yield is read at a standard tenor only when the curve reaches it, up
# to this fraction past its last node: an on-the-run security matures a
# little short of its nominal term once it has aged since its auction, such
# as a 30-year bond a few months after it was issued.
PAR_TENOR_SLACK = 0.05

# Each snapshot is fetched once per page load: the reference snapshot is
# undated, and the market snapshot is read through one provider per
# valuation date, since a provider is bound to its date.
_REFERENCE = MirrorReferenceDataProvider()
_MARKET_BY_DATE = {}
_TERMS = {}


class _TableReferenceDataProvider(ReferenceDataProvider):
    """Serves the contract terms of the rows in the table, keyed by row id."""

    name = "table"
    entities = frozenset({"bond_terms"})

    def __init__(self, terms):
        self._terms = terms

    def fetch(self, identifiers):
        """The terms of every requested row id the table holds."""
        return {
            identifier: FetchedValue(self._terms[identifier.symbol])
            for identifier in identifiers
            if identifier.symbol in self._terms
        }


def _get_market(valuation_date):
    if valuation_date not in _MARKET_BY_DATE:
        _MARKET_BY_DATE[valuation_date] = MirrorMarketDataProvider(valuation_date, entities=["bond_list"])
    return _MARKET_BY_DATE[valuation_date]


def _fetch_bond_list(valuation_date, universe):
    """The mirror's `universe` bond list priced on `valuation_date`.

    Raises:
        ValueError: If the mirror has no list priced on that day.
    """
    identifier = Identifier.bond_list(universe)
    fetched = _get_market(valuation_date).fetch([identifier])
    if identifier not in fetched or fetched[identifier].value.as_of != valuation_date:
        raise ValueError(f"The snapshot has no Treasury prices for {valuation_date}.")
    return fetched[identifier].value


def _fetch_terms(cusips):
    """The terms snapshot's `BondTerms` for each of `cusips` it holds, keyed
    by CUSIP."""
    missing = [cusip for cusip in cusips if cusip not in _TERMS]
    if missing:
        fetched = _REFERENCE.fetch([Identifier.bond_terms(cusip) for cusip in missing])
        _TERMS.update({identifier.symbol: value.value for identifier, value in fetched.items()})
    return {cusip: _TERMS[cusip] for cusip in cusips if cusip in _TERMS}


def _to_row(entry, terms, kind):
    quote = entry.quote
    return {
        "id": entry.cusip,
        "kind": kind,
        "tenor": None if entry.tenor is None else entry.tenor.label,
        "type": terms.security_type.value,
        "coupon": None if terms.coupon_rate is None else round(float(terms.coupon_rate) * 100, 6),
        "maturity": terms.maturity_date.isoformat(),
        "bid": None if quote is None else quote.bid,
        "ask": None if quote is None else quote.ask,
        "last": None if quote is None else quote.last_price,
    }


def list_valuation_dates():
    """The priced days in the bond-quotes snapshot as ISO dates, newest
    first."""
    try:
        response = requests.get(QUOTES_URL, timeout=30)
        response.raise_for_status()
        days = sorted(response.json()["days"], reverse=True)
    except (requests.RequestException, ValueError, KeyError) as exc:
        return {"success": False, "error": f"Couldn't read the Treasury snapshot: {exc}"}
    return {"success": True, "dates": days}


def load_day(valuation_date):
    """One row per on-the-run Treasury priced on `valuation_date` (an ISO
    date), sorted by maturity, with its snapshot terms and quote."""
    try:
        as_of = date.fromisoformat(valuation_date)
        bond_list = _fetch_bond_list(as_of, UNIVERSE)
        terms = _fetch_terms([entry.cusip for entry in bond_list.bonds])
    except Exception as exc:
        return {"success": False, "error": str(exc)}
    rows = [
        _to_row(entry, terms[entry.cusip], "otr")
        for entry in sorted(bond_list.bonds, key=lambda entry: entry.maturity_date)
        if entry.cusip in terms
    ]
    return {"success": True, "rows": rows}


def lookup_cusip(cusip, valuation_date):
    """The row for the Treasury `cusip`, with its quote on `valuation_date`
    when the snapshot prices it that day and empty prices otherwise."""
    cusip = cusip.strip().upper()
    try:
        as_of = date.fromisoformat(valuation_date)
        terms = _fetch_terms([cusip]).get(cusip)
        if terms is None:
            return {"success": False, "error": f"{cusip} isn't a Treasury in the terms snapshot."}
        if terms.security_type not in CURVE_TYPES:
            return {
                "success": False,
                "error": f"{cusip} is a {terms.security_type.value.upper()}; only bills, notes and bonds are curve nodes.",
            }
        if terms.maturity_date <= as_of:
            return {"success": False, "error": f"{cusip} matured on {terms.maturity_date}."}
        listed = {entry.cusip: entry for entry in _fetch_bond_list(as_of, "UST").bonds}
    except Exception as exc:
        return {"success": False, "error": str(exc)}
    entry = listed.get(cusip) or BondListEntry(
        cusip=cusip,
        security_type=terms.security_type,
        coupon_rate=terms.coupon_rate or 0.0,
        maturity_date=terms.maturity_date,
    )
    return {"success": True, "row": _to_row(entry, terms, "cusip")}


def _build_custom_terms(row, as_of, security_type, maturity, coupon_rate):
    """The `BondTerms` of a hypothetical Treasury: a bill issued on `as_of`,
    or a note or bond paying semiannual coupons on a regular schedule rolled
    back from `maturity`, accruing since its last coupon date on or before
    `as_of`."""
    if security_type is TreasurySecurityType.BILL:
        return BondTerms(row["id"], security_type, issue_date=as_of, maturity_date=maturity)
    roll = RollConvention.from_anchor(maturity)
    months_back = 6
    dated_date = roll.get_roll_date(maturity, -months_back)
    while dated_date > as_of:
        months_back += 6
        dated_date = roll.get_roll_date(maturity, -months_back)
    return BondTerms(
        row["id"],
        security_type,
        issue_date=dated_date,
        maturity_date=maturity,
        dated_date=dated_date,
        coupon_rate=coupon_rate,
        coupons_per_year=2,
    )


def _build_entry(row, as_of):
    """The `BondListEntry` and `BondTerms` a table row describes.

    Raises:
        ValueError, TypeError: If the row's values don't make a valid
            Treasury, quote or schedule.
    """
    maturity = date.fromisoformat(row["maturity"])
    if maturity <= as_of:
        raise ValueError(f"matures on {maturity}, not after the valuation date")
    security_type = TreasurySecurityType(row["type"])
    is_bill = security_type is TreasurySecurityType.BILL
    if not is_bill and row["coupon"] is None:
        raise ValueError("needs a coupon")
    coupon_rate = 0.0 if is_bill else row["coupon"] / 100
    if row["kind"] == "custom":
        terms = _build_custom_terms(row, as_of, security_type, maturity, coupon_rate)
    else:
        terms = replace(
            _TERMS[row["id"]],
            maturity_date=maturity,
            coupon_rate=None if is_bill else coupon_rate,
        )
    prices = {field: row[key] for field, key in (("bid", "bid"), ("ask", "ask"), ("last_price", "last"))}
    has_price = any(value is not None for value in prices.values())
    quote = BondQuote(currency=CURRENCY, as_of=as_of, **prices) if has_price else None
    entry = BondListEntry(
        cusip=row["id"],
        security_type=security_type,
        coupon_rate=coupon_rate,
        maturity_date=maturity,
        quote=quote,
    )
    return entry, terms


def _get_curve_grid(node_times):
    """Plot times from one day to the last node, with each node time and a
    point just before it, so a forward that jumps at a node draws as a
    vertical step."""
    inner = node_times[:-1]
    edges = inner + [t - 1e-6 for t in inner]
    return np.unique(np.concatenate([np.linspace(1 / 365, node_times[-1], 600), edges]))


def _find_priced(entries, price_side):
    """The entries with a `price_side` price."""
    priced = []
    for entry in entries:
        try:
            entry.quote.resolve(price_side)
        except (AttributeError, ValueError):
            continue
        priced.append(entry)
    return priced


def _find_shared_maturity(entries):
    """A message naming two of `entries` that mature on the same day, or
    `None` when every maturity is distinct."""
    seen = {}
    for entry in entries:
        if entry.maturity_date in seen:
            return f"{seen[entry.maturity_date]} and {entry.cusip} both mature on {entry.maturity_date}; a curve node needs its own maturity."
        seen[entry.maturity_date] = entry.cusip
    return None


def _get_par_tenors(priced, as_of):
    """The standard tenors no further than `PAR_TENOR_SLACK` past the
    longest of `priced`."""
    if not priced:
        return ()
    longest = (max(entry.maturity_date for entry in priced) - as_of).days / 365
    return tuple(tenor for tenor in Tenor if int(tenor) / 12 <= longest * (1 + PAR_TENOR_SLACK))


def bootstrap(rows, valuation_date, side):
    """The Treasury curve bootstrapped from `rows` priced on `side` ("bid",
    "ask" or "mid") as of `valuation_date`: zero rates and instantaneous
    forwards on a plot grid, each node with its zero rate and residual, and
    constant-maturity par yields at the standard tenors the curve reaches.

    Rates are in percent: zero rates and forwards continuously compounded,
    par yields semiannual. A row without a `side` price is left out of the
    curve, and its id is missing from `nodes`."""
    try:
        as_of = date.fromisoformat(valuation_date)
        price_side = SIDES[side]
        entries, terms = [], {}
        for row in rows:
            try:
                entry, row_terms = _build_entry(row, as_of)
            except (ValueError, TypeError) as exc:
                return {"success": False, "error": f"{row['id']}: {str(exc).rstrip('.')}."}
            entries.append(entry)
            terms[row["id"]] = row_terms
    except Exception as exc:
        return {"success": False, "error": str(exc)}
    priced = _find_priced(entries, price_side)
    shared = _find_shared_maturity(priced)
    if shared:
        return {"success": False, "error": shared}

    # The edited table stands in for the mirror: fiqua's own curve builders
    # fetch the bond list and terms from these providers exactly as they
    # would from the snapshot.
    mdp = MarketDataProviderRegistry()
    mdp.register(
        DummyMarketDataProvider(
            {Identifier.bond_list(UNIVERSE): FetchedValue(BondList(UNIVERSE, as_of, tuple(entries)))}, as_of
        )
    )
    engine = CalculationEngine(as_of, mdp=mdp, rdp=_TableReferenceDataProvider(terms))
    curve_request = BootstrappedCurveRequest(CURVE_ID, price_side)
    tenors = _get_par_tenors(priced, as_of)
    # A cubic spline needs three points; a curve of bills alone may reach only
    # two standard tenors, which are joined linearly.
    par_request = None
    if len(tenors) >= 2:
        par_request = ParYieldCurveRequest(
            source=CURVE_ID,
            tenors=tenors,
            interpolator=Interpolator.CUBIC_SPLINE if len(tenors) >= 3 else Interpolator.LINEAR,
            price_side=price_side,
        )
    engine.add([curve_request] + ([par_request] if par_request else []))
    engine.run()

    built = engine.market_object_results[curve_request.get_market_object_id()]
    if not built.built:
        return {"success": False, "error": built.error_msg}
    bootstrapped = built.market_object
    curve = bootstrapped.curve
    node_times = [node.maturity for node in bootstrapped.nodes]
    grid = _get_curve_grid(node_times)
    result = {
        "success": True,
        "nodes": [
            {
                "id": node.label,
                "t": node.maturity,
                "zero": curve.get_zero_rate(node.maturity).value * 100,
                "residual": residual,
            }
            for node, residual in zip(bootstrapped.nodes, bootstrapped.residuals)
        ],
        "grid": grid.tolist(),
        "zero": [curve.get_zero_rate(float(t)).value * 100 for t in grid],
        "forward": [curve.get_instantaneous_forward(float(t)).value * 100 for t in grid],
        "par": None,
        "par_error": None,
    }

    if par_request is None:
        result["par_error"] = "The curve reaches fewer than two standard tenors, too few for a par yield curve."
        return result
    par_built = engine.market_object_results[par_request.get_market_object_id()]
    if not par_built.built:
        result["par_error"] = par_built.error_msg
        return result
    par_curve = par_built.market_object
    par_times = [span.value for span in par_curve.points]
    par_grid = np.linspace(par_times[0], par_times[-1], 400)
    result["par"] = {
        "points": [
            {"tenor": tenor.label, "t": t, "value": value * 100}
            for tenor, t, value in zip(par_request.tenors, par_times, par_curve.points.values())
        ],
        "grid": par_grid.tolist(),
        "values": [par_curve.get_par_yield(float(t)).value * 100 for t in par_grid],
    }
    return result

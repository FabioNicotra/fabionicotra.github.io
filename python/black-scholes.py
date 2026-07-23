import micropip
await micropip.install("numanlib")

import json
import math
import numpy as np
from numanlib.pdes import SpatiotemporalDomain1D, ParabolicBCs, BlackScholesEquationSolver


def _ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _closed_form_price(S, K, tau, r, sigma, is_call):
    if tau <= 1e-12:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)
    if S <= 0:
        return 0.0 if is_call else K * math.exp(-r * tau)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * tau) / (sigma * math.sqrt(tau))
    d2 = d1 - sigma * math.sqrt(tau)
    if is_call:
        return S * _ncdf(d1) - K * math.exp(-r * tau) * _ncdf(d2)
    return K * math.exp(-r * tau) * _ncdf(-d2) - S * _ncdf(-d1)


def solve_black_scholes(S, K, T, r, sigma, option_type, m, N, method):
    try:
        S, K, T, r, sigma = float(S), float(K), float(T), float(r), float(sigma)
        m, N = int(m), int(N)
        is_call = option_type == "call"
        S_max = 4.0 * K
        domain = SpatiotemporalDomain1D(l=S_max, T=T)

        if is_call:
            terminal = lambda s: max(s - K, 0.0)
            g0 = lambda t: 0.0
            g1 = lambda t: S_max - K * math.exp(-r * (T - t))
        else:
            terminal = lambda s: max(K - s, 0.0)
            g0 = lambda t: K * math.exp(-r * (T - t))
            g1 = lambda t: 0.0

        solver = BlackScholesEquationSolver(
            domain=domain,
            r=r,
            sigma=sigma,
            terminal_condition=terminal,
            boundary_conditions=ParabolicBCs(g0=g0, g1=g1),
            m=m,
            N=N,
            method=method,
        )
        result = solver.solve()
        if not result.solved:
            return {"success": False, "error": result.error_msg}

        grid_S = result.grid_x
        grid_t_full = result.grid_t
        payoff_curve = [terminal(float(s)) for s in grid_S]
        spot_in_domain = 0.0 <= S <= S_max

        # Subsample time steps for the slider: keep the UI usable and the
        # payload small even when N is large; the solve itself still ran
        # at the full N resolution, this only thins which columns we ship.
        max_frames = 50
        n_t = len(grid_t_full)
        if n_t <= max_frames:
            idx = list(range(n_t))
        else:
            idx = sorted(set(round(i * (n_t - 1) / (max_frames - 1)) for i in range(max_frames)))

        grid_t = [float(grid_t_full[j]) for j in idx]
        fd_grid = [result.solution[:, j].tolist() for j in idx]
        closed_form_grid = [
            [_closed_form_price(float(s), K, T - grid_t_full[j], r, sigma, is_call) for s in grid_S]
            for j in idx
        ]

        return {
            "success": True,
            "grid_S": grid_S.tolist(),
            "grid_t": grid_t,
            "payoff_curve": payoff_curve,
            "fd_grid": fd_grid,
            "closed_form_grid": closed_form_grid,
            "S_max": S_max,
            "spot_in_domain": spot_in_domain,
            "spot_fd_price": float(np.interp(S, grid_S, fd_grid[0])) if spot_in_domain else None,
            "spot_closed_form_price": _closed_form_price(S, K, T, r, sigma, is_call) if spot_in_domain else None,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

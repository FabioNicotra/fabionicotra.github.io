import micropip
await micropip.install("numanlib")

import json
import numpy as np
from numanlib.interpolation import CubicSpline

current_spline = None


def fit_spline(x_values, y_values):
    global current_spline
    try:
        x = np.array(x_values, dtype=np.float64)
        y = np.array(y_values, dtype=np.float64)

        if len(x) < 3:
            return {"success": False, "error": "Need at least 3 points for cubic spline"}

        if len(np.unique(x)) != len(x):
            return {"success": False, "error": "X values must be distinct"}

        # Sort by x
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        y = y[sort_idx]

        spline = CubicSpline(x, y)
        current_spline = spline.interpolate()
        return {
            "success": True,
            "min_x": float(x.min()),
            "max_x": float(x.max()),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def interpolate_curve():
    if current_spline is None:
        return {"error": "No spline fitted"}

    min_x = current_spline.x.min()
    max_x = current_spline.x.max()
    span = max_x - min_x
    dense_x = np.linspace(min_x, max_x, 400) if span > 0 else np.array([min_x])
    interpolated_y = current_spline(dense_x)

    return {
        "success": True,
        "dense_x": dense_x.tolist(),
        "interpolated_y": interpolated_y.tolist(),
        "min_x": float(min_x),
        "max_x": float(max_x),
    }

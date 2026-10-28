"""Validated grid loading and quantitative structural-closure calculations."""
from __future__ import annotations

import argparse
import heapq
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

COLUMNS = ("x_m", "y_m", "depth_m")
_NEIGHBORHOOD_8 = np.ones((3, 3), dtype=int)


def _regular_axis(values: np.ndarray, name: str) -> float:
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError(f"{name} coordinates must be a finite one-dimensional axis with at least two values")
    delta = np.diff(values)
    if not np.all(delta > 0):
        raise ValueError(f"{name} coordinates must be strictly increasing")
    if not np.allclose(delta, delta[0], rtol=1e-7, atol=max(1e-10, abs(delta[0]) * 1e-9)):
        raise ValueError(f"{name} coordinates must be regularly spaced; irregular grids are unsupported")
    return float(delta[0])


def _validate_arrays(xs: np.ndarray, ys: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    xs, ys, grid = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), np.asarray(grid, dtype=float)
    dx, dy = _regular_axis(xs, "x"), _regular_axis(ys, "y")
    if grid.ndim != 2 or grid.shape != (len(ys), len(xs)):
        raise ValueError("depth grid shape must be (len(y), len(x))")
    if np.isinf(grid).any() or not np.isfinite(grid).any():
        raise ValueError("depth values must include at least one finite value and no infinities")
    return xs, ys, grid, dx, dy


def load_grid(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a complete, axis-aligned regular grid; NaN depth cells are allowed."""
    df = pd.read_csv(path)
    if not set(COLUMNS).issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {', '.join(COLUMNS)}")
    if df.empty:
        raise ValueError("input grid is empty")
    for col in COLUMNS[:2]:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.isna().any():
            raise ValueError(f"coordinate column {col} must contain finite numeric values")
        df[col] = converted
    depth = pd.to_numeric(df["depth_m"], errors="coerce")
    if (df["depth_m"].notna() & depth.isna()).any():
        raise ValueError("depth_m must contain numeric values or blank/NaN cells")
    df["depth_m"] = depth
    if df.duplicated(["x_m", "y_m"]).any():
        raise ValueError("duplicate (x_m, y_m) coordinates found")
    xs, ys = np.sort(df.x_m.unique()), np.sort(df.y_m.unique())
    if len(xs) * len(ys) != len(df):
        raise ValueError("coordinates must form a complete rectangular grid; missing coordinate pairs are unsupported")
    dx, dy = _regular_axis(xs, "x"), _regular_axis(ys, "y")
    grid = df.pivot(index="y_m", columns="x_m", values="depth_m").reindex(index=ys, columns=xs).to_numpy(float)
    _validate_arrays(xs, ys, grid)
    return xs, ys, grid


def _normalized_smooth(grid: np.ndarray, sigma: float | tuple[float, float]) -> np.ndarray:
    valid = np.isfinite(grid)
    if np.isscalar(sigma):
        sigmas = (float(sigma), float(sigma))
    else:
        sigmas = tuple(float(v) for v in sigma)
    if len(sigmas) != 2 or any(not np.isfinite(v) or v < 0 for v in sigmas):
        raise ValueError("smoothing sigma must be a nonnegative finite scalar or (sigma_y, sigma_x)")
    if not any(sigmas):
        return grid.copy()
    weights = ndimage.gaussian_filter(valid.astype(float), sigma=sigmas, mode="nearest")
    totals = ndimage.gaussian_filter(np.where(valid, grid, 0.0), sigma=sigmas, mode="nearest")
    smoothed = np.full(grid.shape, np.nan, dtype=float)
    smoothed[valid] = totals[valid] / weights[valid]
    return smoothed


def _triangle_area(coords: np.ndarray) -> float:
    v1 = coords[1] - coords[0]
    v2 = coords[2] - coords[0]
    return abs(float(v1[0] * v2[1] - v1[1] * v2[0])) / 2.0


def _clipped_triangle_area(coords: np.ndarray, values: np.ndarray, threshold: float) -> float:
    """Area above threshold for a linearly interpolated triangle."""
    polygon = [(float(x), float(y), float(z)) for (x, y), z in zip(coords, values)]
    clipped: list[tuple[float, float, float]] = []
    previous = polygon[-1]
    prev_inside = previous[2] >= threshold
    for current in polygon:
        curr_inside = current[2] >= threshold
        if curr_inside != prev_inside:
            fraction = (threshold - previous[2]) / (current[2] - previous[2])
            clipped.append((previous[0] + fraction * (current[0] - previous[0]),
                            previous[1] + fraction * (current[1] - previous[1]), threshold))
        if curr_inside:
            clipped.append(current)
        previous, prev_inside = current, curr_inside
    if len(clipped) < 3:
        return 0.0
    xy = np.asarray([(p[0], p[1]) for p in clipped])
    return abs(float(np.dot(xy[:, 0], np.roll(xy[:, 1], -1)) - np.dot(xy[:, 1], np.roll(xy[:, 0], -1)))) / 2.0


def _mesh_areas(xs: np.ndarray, ys: np.ndarray, surface: np.ndarray, labels: np.ndarray,
                component: int, threshold: float) -> tuple[float, float]:
    """Integrate piecewise-linear triangles; omit any triangle touching a null node."""
    closure_area = valid_area = 0.0
    ny, nx = surface.shape
    for row in range(ny - 1):
        for col in range(nx - 1):
            coords = np.asarray(((xs[col], ys[row]), (xs[col + 1], ys[row]),
                                 (xs[col + 1], ys[row + 1]), (xs[col], ys[row + 1])), dtype=float)
            values = np.asarray((surface[row, col], surface[row, col + 1],
                                 surface[row + 1, col + 1], surface[row + 1, col]), dtype=float)
            comps = np.asarray((labels[row, col], labels[row, col + 1],
                                labels[row + 1, col + 1], labels[row + 1, col]))
            # Fixed diagonal from upper-left to lower-right; shared consistently across the grid.
            for indices in ((0, 1, 2), (0, 2, 3)):
                tri_values = values[list(indices)]
                if not np.isfinite(tri_values).all():
                    continue
                tri_coords = coords[list(indices)]
                valid_area += _triangle_area(tri_coords)
                tri_labels = comps[list(indices)]
                if component and np.any(tri_labels == component):
                    closure_area += _clipped_triangle_area(tri_coords, tri_values, threshold)
    return closure_area, valid_area


def _grid_spill(xs: np.ndarray, ys: np.ndarray, surface: np.ndarray,
                crest: tuple[int, int]) -> dict[str, float | int | bool]:
    """Find the grid-exit saddle as the maximum-bottleneck 8-neighbor path from crest to edge."""
    ny, nx = surface.shape
    start = crest
    capacity = np.full(surface.shape, -np.inf, dtype=float)
    parent_r = np.full(surface.shape, -1, dtype=int)
    parent_c = np.full(surface.shape, -1, dtype=int)
    capacity[start] = float(surface[start])
    queue: list[tuple[float, int, int]] = [(-float(capacity[start]), start[0], start[1])]
    exit_cell: tuple[int, int] | None = None
    while queue:
        neg_cap, row, col = heapq.heappop(queue)
        cap = -neg_cap
        if cap < capacity[row, col]:
            continue
        if row in (0, ny - 1) or col in (0, nx - 1):
            exit_cell = (row, col)
            break
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = row + dr, col + dc
                if (dr == 0 and dc == 0) or rr < 0 or rr >= ny or cc < 0 or cc >= nx:
                    continue
                if not np.isfinite(surface[rr, cc]):
                    continue
                candidate = min(cap, float(surface[rr, cc]))
                if candidate > capacity[rr, cc]:
                    capacity[rr, cc] = candidate
                    parent_r[rr, cc], parent_c[rr, cc] = row, col
                    heapq.heappush(queue, (-candidate, rr, cc))
    if exit_cell is None:
        return {"spill_x_m": np.nan, "spill_y_m": np.nan, "spill_depth_m": np.nan,
                "spill_row": -1, "spill_col": -1, "grid_exit_reachable": False}
    path = [exit_cell]
    while path[-1] != start:
        row, col = path[-1]
        pr, pc = int(parent_r[row, col]), int(parent_c[row, col])
        if pr < 0:
            raise RuntimeError("could not reconstruct maximum-bottleneck path")
        path.append((pr, pc))
    saddle = min(path, key=lambda rc: (surface[rc], rc[0], rc[1]))
    return {"spill_x_m": float(xs[saddle[1]]), "spill_y_m": float(ys[saddle[0]]),
            "spill_depth_m": float(capacity[exit_cell]), "spill_row": saddle[0],
            "spill_col": saddle[1], "grid_exit_reachable": True}


def analyze_grid(xs: np.ndarray, ys: np.ndarray, grid: np.ndarray, *, spill_depth_m: float,
                 smoothing_sigma: float | tuple[float, float] = 1.0,
                 min_area_km2: float = 0.0) -> dict:
    """Analyze a selected spill threshold on a regular grid.

    Depth convention: larger numerical values are shallower. The crest is the
    maximum of the unsmoothed valid grid. Threshold regions use normalized
    Gaussian smoothing but retain the original null mask. The grid-exit spill
    point is the lowest point along the maximum-bottleneck 8-neighbor route
    from the smoothed crest to any edge node; it is a domain-dependent digital
    saddle, not an independently interpreted hydrocarbon contact.
    """
    xs, ys, grid, dx, dy = _validate_arrays(xs, ys, grid)
    if not np.isfinite(spill_depth_m):
        raise ValueError("spill depth must be finite")
    if not np.isfinite(min_area_km2) or min_area_km2 < 0:
        raise ValueError("minimum area must be a nonnegative finite value")
    smooth = _normalized_smooth(grid, smoothing_sigma)
    valid = np.isfinite(grid)
    crest_row, crest_col = np.unravel_index(np.nanargmax(grid), grid.shape)
    crest = (int(crest_row), int(crest_col))
    crest_depth = float(grid[crest])
    crest_smoothed = float(smooth[crest])
    mask = valid & (smooth >= spill_depth_m)
    labels, _ = ndimage.label(mask, structure=_NEIGHBORHOOD_8)
    component = int(labels[crest])
    closure = labels == component if component else np.zeros_like(mask)
    area_m2, valid_mesh_area_m2 = _mesh_areas(xs, ys, smooth, labels, component, float(spill_depth_m))
    domain_area_m2 = float((xs[-1] - xs[0]) * (ys[-1] - ys[0]))
    boundary_touched = bool(component and (np.any(closure[0, :]) or np.any(closure[-1, :]) or
                                            np.any(closure[:, 0]) or np.any(closure[:, -1])))
    valid_count = int(valid.sum())
    total_count = int(grid.size)
    status = "closed" if component and area_m2 > 0 and not boundary_touched else (
        "boundary_limited" if component and area_m2 > 0 else "no_closure")
    if component and area_m2 < min_area_km2 * 1e6:
        status = "below_min_area"
    spill_crest = tuple(map(int, np.unravel_index(np.nanargmax(smooth), smooth.shape)))
    spill = _grid_spill(xs, ys, smooth, spill_crest)
    relief = crest_depth - float(spill_depth_m)
    return {
        "crest_x_m": float(xs[crest_col]), "crest_y_m": float(ys[crest_row]),
        "crest_depth_m": crest_depth, "crest_smoothed_depth_m": crest_smoothed,
        "spill_threshold_m": float(spill_depth_m), "relief_m": float(relief),
        "spill_x_m": spill["spill_x_m"], "spill_y_m": spill["spill_y_m"],
        "grid_spill_depth_m": spill["spill_depth_m"],
        "grid_exit_reachable": bool(spill["grid_exit_reachable"]),
        "closure_area_m2": float(area_m2), "closure_area_km2": float(area_m2 / 1e6),
        "has_closure": bool(component and area_m2 > 0), "closed_closure": bool(status == "closed"),
        "closure_status": status, "boundary_touched": boundary_touched,
        "closure_node_count": int(closure.sum()), "grid_nx": int(len(xs)), "grid_ny": int(len(ys)),
        "grid_spacing_x_m": dx, "grid_spacing_y_m": dy,
        "valid_node_count": valid_count, "total_node_count": total_count,
        "valid_data_coverage_pct": float(valid_count / total_count * 100),
        "valid_mesh_coverage_pct": float(valid_mesh_area_m2 / domain_area_m2 * 100),
        "missing_node_count": int(total_count - valid_count), "smoothing_sigma_cells": smoothing_sigma,
        "xs": xs, "ys": ys, "grid": grid, "smoothed": smooth, "closure": closure,
        "labels": labels, "crest_row": crest_row, "crest_col": crest_col,
        "spill_row": int(spill["spill_row"]), "spill_col": int(spill["spill_col"]),
        "valid_mesh_area_m2": valid_mesh_area_m2, "domain_area_m2": domain_area_m2,
    }


def write_outputs(result: dict, output_dir: str | Path, *, stem: str = "synthetic_surface",
                  contour_step_m: float = 20.0) -> tuple[Path, Path]:
    """Write the legacy contour map and its compact metrics row."""
    if not np.isfinite(contour_step_m) or contour_step_m <= 0:
        raise ValueError("contour_step_m must be positive and finite")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    xs, ys, grid = result["xs"], result["ys"], result["grid"]
    valid_values = grid[np.isfinite(grid)]
    levels = np.arange(np.floor(valid_values.min() / contour_step_m) * contour_step_m,
                       np.ceil(valid_values.max() / contour_step_m) * contour_step_m + contour_step_m,
                       contour_step_m)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    filled = ax.contourf(xs, ys, np.ma.masked_invalid(grid), levels=levels, cmap="viridis", extend="both")
    ax.contour(xs, ys, result["smoothed"], levels=levels, colors="black", linewidths=0.45)
    if result["has_closure"]:
        ax.contour(xs, ys, result["closure"].astype(float), levels=[0.5], colors="white", linewidths=1.4)
    ax.scatter(result["crest_x_m"], result["crest_y_m"], marker="*", s=65, color="red", label="Synthetic crest")
    ax.set(title="Fictional synthetic structural surface", xlabel="Synthetic x (m)", ylabel="Synthetic y (m)")
    ax.set_aspect("equal")
    ax.legend(loc="best")
    fig.colorbar(filled, ax=ax, label="Synthetic depth (m; larger is shallower)")
    map_path = out / f"{stem}_contours.png"
    fig.savefig(map_path, dpi=160)
    plt.close(fig)
    legacy_metrics = {"crest_x_m": result["crest_x_m"], "crest_y_m": result["crest_y_m"],
                      "crest_depth_m": result["crest_depth_m"], "spill_depth_m": result["spill_threshold_m"],
                      "relief_m": result["relief_m"], "closure_area_km2": result["closure_area_km2"],
                      "has_closure": result["has_closure"]}
    result_path = out / f"{stem}_metrics.csv"
    pd.DataFrame([legacy_metrics]).to_csv(result_path, index=False)
    return map_path, result_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="synthetic or authorized CSV with x_m,y_m,depth_m columns")
    parser.add_argument("--spill-depth-m", type=float, required=True, help="selected threshold; larger values are shallower")
    parser.add_argument("--output-dir", type=Path, default=Path("examples/output"))
    parser.add_argument("--contour-step-m", type=float, default=20.0)
    parser.add_argument("--smoothing-sigma", type=float, default=1.0)
    parser.add_argument("--min-area-km2", type=float, default=0.0)
    args = parser.parse_args()
    xs, ys, grid = load_grid(args.input)
    result = analyze_grid(xs, ys, grid, spill_depth_m=args.spill_depth_m,
                          smoothing_sigma=args.smoothing_sigma, min_area_km2=args.min_area_km2)
    map_path, table_path = write_outputs(result, args.output_dir, stem=args.input.stem,
                                         contour_step_m=args.contour_step_m)
    print(f"Synthetic or user-supplied grid analyzed: {args.input}")
    print(f"Map: {map_path}\nMetrics: {table_path}")


if __name__ == "__main__":
    main()

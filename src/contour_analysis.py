"""Reusable analysis functions for regular x/y/depth CSV grids."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt

COLUMNS = ("x_m", "y_m", "depth_m")


def load_grid(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a complete regular grid from CSV columns x_m,y_m,depth_m."""
    df = pd.read_csv(path)
    if not set(COLUMNS).issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {', '.join(COLUMNS)}")
    if df.empty:
        raise ValueError("input grid is empty")
    if df[list(COLUMNS[:2])].isna().any().any():
        raise ValueError("grid coordinates cannot be missing")
    if df.duplicated(["x_m", "y_m"]).any():
        raise ValueError("duplicate grid coordinates found")
    xs, ys = np.sort(df.x_m.unique()), np.sort(df.y_m.unique())
    if len(xs) < 2 or len(ys) < 2 or len(xs) * len(ys) != len(df):
        raise ValueError("coordinates must form a complete rectangular grid")
    grid = df.pivot(index="y_m", columns="x_m", values="depth_m").reindex(index=ys, columns=xs).to_numpy(float)
    if np.isinf(grid).any() or np.isnan(grid).all():
        raise ValueError("depth values must contain at least one finite value and no infinities")
    dx, dy = np.diff(xs), np.diff(ys)
    if not np.allclose(dx, dx[0]) or not np.allclose(dy, dy[0]):
        raise ValueError("grid coordinates must be regularly spaced")
    return xs, ys, grid


def analyze_grid(xs: np.ndarray, ys: np.ndarray, grid: np.ndarray, *, spill_depth_m: float,
                 smoothing_sigma: float = 1.0, min_area_km2: float = 0.0) -> dict:
    """Find the connected spill-level region containing the shallowest crest."""
    if not np.isfinite(spill_depth_m) or smoothing_sigma < 0 or min_area_km2 < 0:
        raise ValueError("spill depth must be finite; smoothing and minimum area must be nonnegative")
    valid = np.isfinite(grid)
    if not valid.any() or grid.ndim != 2 or grid.shape != (len(ys), len(xs)):
        raise ValueError("grid shape or valid depth cells are invalid")
    dx = float(np.diff(xs).mean())
    dy = float(np.diff(ys).mean())
    filled = np.where(valid, grid, float(np.nanmean(grid)))
    smoothed = ndimage.gaussian_filter(filled, sigma=smoothing_sigma) if smoothing_sigma else filled.copy()
    crest_row, crest_col = np.unravel_index(np.nanargmax(grid), grid.shape)
    mask = (smoothed >= spill_depth_m) & valid
    labels, _ = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    crest_label = int(labels[crest_row, crest_col])
    closure = labels == crest_label if crest_label else np.zeros_like(mask)
    area_km2 = float(closure.sum() * dx * dy / 1e6)
    result = {
        "crest_x_m": float(xs[crest_col]), "crest_y_m": float(ys[crest_row]),
        "crest_depth_m": float(grid[crest_row, crest_col]), "spill_depth_m": float(spill_depth_m),
        "relief_m": float(grid[crest_row, crest_col] - spill_depth_m),
        "closure_area_km2": area_km2 if area_km2 >= min_area_km2 else 0.0,
        "has_closure": bool(crest_label and area_km2 >= min_area_km2),
    }
    result.update(xs=xs, ys=ys, grid=grid, smoothed=smoothed, closure=closure)
    return result


def write_outputs(result: dict, output_dir: str | Path, *, stem: str = "synthetic_surface",
                  contour_step_m: float = 20.0) -> tuple[Path, Path]:
    """Write one contour PNG and one single-row metrics CSV."""
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
    fig.colorbar(filled, ax=ax, label="Synthetic depth (m)")
    map_path = out / f"{stem}_contours.png"
    fig.savefig(map_path, dpi=160)
    plt.close(fig)
    result_path = out / f"{stem}_metrics.csv"
    pd.DataFrame([{k: v for k, v in result.items() if k not in {"xs", "ys", "grid", "smoothed", "closure"}}]).to_csv(result_path, index=False)
    return map_path, result_path

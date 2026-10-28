"""Generate a fictional structural surface on a regular rectangular grid."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def make_synthetic_grid(*, nx: int = 101, ny: int = 91, seed: int = 20261026,
                        extent_m: tuple[float, float, float, float] = (0.0, 10_000.0, 0.0, 9_000.0),
                        noise_std_m: float = 1.5, null_fraction: float = 0.02) -> pd.DataFrame:
    """Return synthetic x/y/depth rows; no external reservoir data is used.

    The fictional surface is a smooth elliptical structural high (a Gaussian
    dome) over a planar regional trend, plus small seeded Gaussian noise. A
    seeded circular patch is marked missing to demonstrate null handling.
    Depth values are illustrative metre values with no relation to any real
    field or project. Larger numerical values represent shallower structure.
    """
    if nx < 3 or ny < 3:
        raise ValueError("nx and ny must each be at least 3")
    if noise_std_m < 0 or not 0 <= null_fraction < 0.5:
        raise ValueError("noise_std_m must be nonnegative and null_fraction in [0, 0.5)")
    xmin, xmax, ymin, ymax = map(float, extent_m)
    if not (xmin < xmax and ymin < ymax):
        raise ValueError("extent_m must have increasing x and y bounds")
    rng = np.random.default_rng(seed)
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(x, y)
    u = (xx - (xmin + xmax) / 2) / ((xmax - xmin) * 0.24)
    v = (yy - (ymin + ymax) / 2) / ((ymax - ymin) * 0.31)
    dome = 165.0 * np.exp(-0.5 * (u * u + v * v))
    regional = -0.012 * (xx - xmin) + 0.006 * (yy - ymin)
    depth = -2_400.0 + dome + regional + rng.normal(0.0, noise_std_m, size=(ny, nx))
    if null_fraction:
        radius = np.sqrt(((xx - (xmin + 0.18 * (xmax - xmin))) / ((xmax - xmin) * 0.20)) ** 2 +
                         ((yy - (ymin + 0.72 * (ymax - ymin))) / ((ymax - ymin) * 0.20)) ** 2)
        cutoff = np.quantile(radius, null_fraction)
        depth[radius <= cutoff] = np.nan
    return pd.DataFrame({"x_m": xx.ravel(), "y_m": yy.ravel(), "depth_m": depth.ravel()})


def generate_csv(output: str | Path, **kwargs) -> Path:
    """Write a reproducible CSV generated from the fictional surface model."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    make_synthetic_grid(**kwargs).to_csv(path, index=False, float_format="%.6f")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic/synthetic_surface.csv"))
    parser.add_argument("--nx", type=int, default=101)
    parser.add_argument("--ny", type=int, default=91)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--noise-std-m", type=float, default=1.5)
    parser.add_argument("--null-fraction", type=float, default=0.02)
    args = parser.parse_args()
    path = generate_csv(args.output, nx=args.nx, ny=args.ny, seed=args.seed,
                        noise_std_m=args.noise_std_m, null_fraction=args.null_fraction)
    print(f"Wrote fictional synthetic grid: {path}")


if __name__ == "__main__":
    main()

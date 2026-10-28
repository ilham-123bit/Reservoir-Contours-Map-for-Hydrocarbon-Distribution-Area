from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from synthetic_grid import make_synthetic_grid, generate_csv
from contour_analysis import load_grid, analyze_grid, write_outputs


def test_seed_reproduces_fictional_grid():
    a = make_synthetic_grid(nx=21, ny=19, seed=13)
    b = make_synthetic_grid(nx=21, ny=19, seed=13)
    pd.testing.assert_frame_equal(a, b)


def test_csv_round_trip_and_null_cells(tmp_path):
    path = generate_csv(tmp_path / "grid.csv", nx=21, ny=19, seed=4)
    xs, ys, grid = load_grid(path)
    assert grid.shape == (19, 21)
    assert np.isnan(grid).any()
    assert np.isfinite(grid).any()
    assert len(xs) == 21 and len(ys) == 19


def test_closure_and_outputs_are_created(tmp_path):
    df = make_synthetic_grid(nx=41, ny=37, seed=5, null_fraction=0.0)
    csv = tmp_path / "surface.csv"
    df.to_csv(csv, index=False)
    xs, ys, grid = load_grid(csv)
    result = analyze_grid(xs, ys, grid, spill_depth_m=float(np.nanmax(grid) - 70))
    assert result["has_closure"]
    assert result["closure_area_km2"] > 0
    map_path, table_path = write_outputs(result, tmp_path / "out", stem="synthetic", contour_step_m=10)
    assert map_path.is_file() and map_path.stat().st_size > 1000
    table = pd.read_csv(table_path)
    assert list(table.columns) == ["crest_x_m", "crest_y_m", "crest_depth_m", "spill_depth_m", "relief_m", "closure_area_km2", "has_closure"]


def test_irregular_or_duplicate_input_is_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"x_m": [0, 1, 0], "y_m": [0, 0, 1], "depth_m": [1, 2, 3]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_grid(path)

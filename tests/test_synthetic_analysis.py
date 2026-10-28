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


def test_piecewise_linear_area_relief_and_boundary_flags():
    from contour_analysis import analyze_grid

    xs = np.array([0.0, 1.0])
    ys = np.array([0.0, 1.0])
    grid = np.array([[1.0, 0.0], [0.0, 0.0]])
    result = analyze_grid(xs, ys, grid, spill_depth_m=0.5, smoothing_sigma=0.0)
    assert result["crest_x_m"] == 0.0
    assert result["crest_y_m"] == 0.0
    assert result["crest_depth_m"] == 1.0
    assert result["relief_m"] == pytest.approx(0.5)
    # A fixed-diagonal linear triangle mesh clips exactly 0.25 m² above this threshold.
    assert result["closure_area_m2"] == pytest.approx(0.25)
    assert result["closure_area_km2"] == pytest.approx(2.5e-7)
    assert result["boundary_touched"]
    assert result["closure_status"] == "boundary_limited"


def test_grid_exit_saddle_and_closed_region_identification():
    from contour_analysis import analyze_grid

    xs = np.arange(5.0)
    ys = np.arange(5.0)
    grid = np.zeros((5, 5), dtype=float)
    grid[2, 2] = 10.0
    result = analyze_grid(xs, ys, grid, spill_depth_m=5.0, smoothing_sigma=0.0)
    assert result["grid_spill_depth_m"] == 0.0
    assert result["spill_x_m"] in (0.0, 4.0) or result["spill_y_m"] in (0.0, 4.0)
    assert result["has_closure"]
    assert result["closed_closure"]
    assert not result["boundary_touched"]
    assert result["closure_area_m2"] > 0


def test_grid_exit_path_starts_at_smoothed_surface_maximum(monkeypatch):
    import contour_analysis

    xs = np.arange(7.0)
    ys = np.arange(7.0)
    grid = np.zeros((7, 7), dtype=float)
    grid[2:5, 3:6] = 8.0
    grid[1, 1] = 10.0  # Raw-grid maximum is narrower than the broader smoothed high.
    observed = {}
    original = contour_analysis._grid_spill

    def capture_start(x_axis, y_axis, surface, start_node):
        observed["start_node"] = start_node
        observed["surface"] = surface.copy()
        return original(x_axis, y_axis, surface, start_node)

    monkeypatch.setattr(contour_analysis, "_grid_spill", capture_start)
    result = contour_analysis.analyze_grid(xs, ys, grid, spill_depth_m=-1.0, smoothing_sigma=1.0)
    smoothed_maximum = np.unravel_index(np.nanargmax(result["smoothed"]), result["smoothed"].shape)
    raw_crest = np.unravel_index(np.nanargmax(grid), grid.shape)

    assert raw_crest != smoothed_maximum
    assert observed["start_node"] == smoothed_maximum
    np.testing.assert_array_equal(observed["surface"], result["smoothed"])


def test_missing_nodes_are_not_filled_and_mesh_coverage_is_reported():
    from contour_analysis import analyze_grid

    xs = np.array([0.0, 1.0])
    ys = np.array([0.0, 1.0])
    grid = np.array([[1.0, np.nan], [0.0, 0.0]])
    result = analyze_grid(xs, ys, grid, spill_depth_m=0.5, smoothing_sigma=1.0)
    assert result["missing_node_count"] == 1
    assert result["valid_data_coverage_pct"] == pytest.approx(75.0)
    assert result["valid_mesh_coverage_pct"] == pytest.approx(50.0)
    assert np.isnan(result["smoothed"][0, 1])


def test_sensitivity_thresholds_are_consistent_and_report_boundary_cases():
    from geoscience_toolkit import sensitivity_analysis

    xs = np.array([0.0, 1.0])
    ys = np.array([0.0, 1.0])
    grid = np.array([[1.0, 0.0], [0.0, 0.0]])
    table = sensitivity_analysis(xs, ys, grid, thresholds_m=[0.5, 0.75, 1.5], smoothing_sigma=0.0)
    assert table.spill_depth_threshold_m.tolist() == [0.5, 0.75, 1.5]
    assert table.relief_m.tolist() == pytest.approx([0.5, 0.25, -0.5])
    assert table.closure_area_m2.tolist() == pytest.approx([0.25, 0.0625, 0.0])
    assert table.closure_status.tolist() == ["boundary_limited", "boundary_limited", "no_closure"]


def test_full_toolkit_writes_summary_sensitivity_and_visuals(tmp_path):
    from geoscience_toolkit import run_analysis

    source = generate_csv(tmp_path / "surface.csv", nx=21, ny=19, seed=13, null_fraction=0.0)
    output = tmp_path / "analysis"
    output.mkdir()
    (output / "robustness_scenarios.csv").write_text("stale prior-run data\n", encoding="utf-8")
    (output / "robustness_comparison.png").write_bytes(b"stale prior-run figure")
    run_analysis(source, output, spill_depth_m=-2330.0,
                 thresholds_m=[-2290.0, -2310.0, -2330.0],
                 smoothing_sigma=1.0, include_robustness=False)
    expected = ["metrics_summary.csv", "spill_sensitivity.csv", "analysis_report.md",
                "structural_contours.png", "surface_3d.png", "cross_sections.png",
                "closure_area_sensitivity.png", "structural_relief_sensitivity.png",
                "closure_scenarios.png"]
    assert all((output / name).is_file() and (output / name).stat().st_size > 0 for name in expected)
    metrics = pd.read_csv(output / "metrics_summary.csv")
    assert metrics.loc[0, "closure_area_m2"] >= 0
    assert metrics.loc[0, "grid_spacing_x_m"] > 0
    assert metrics.loc[0, "closure_area_km2"] == pytest.approx(metrics.loc[0, "closure_area_m2"] / 1e6)
    assert len(pd.read_csv(output / "spill_sensitivity.csv")) == 3
    assert not (output / "robustness_scenarios.csv").exists()
    assert not (output / "robustness_comparison.png").exists()
    report = (output / "analysis_report.md").read_text(encoding="utf-8")
    assert "## Synthetic geometric results at the selected threshold" in report
    assert "raw-grid crest" in report
    assert "smoothed-surface maximum" in report
    assert "grid-exit saddle" in report
    assert "Robustness analysis was skipped for this run" in report
    assert "robustness_scenarios.csv" not in report
    assert "robustness_comparison.png" not in report


def test_full_toolkit_reports_robustness_outputs_when_enabled(tmp_path):
    from geoscience_toolkit import run_analysis

    source = generate_csv(tmp_path / "surface.csv", nx=11, ny=9, seed=17, null_fraction=0.0)
    output = tmp_path / "analysis"
    run_analysis(source, output, spill_depth_m=-2330.0,
                 thresholds_m=[-2290.0, -2310.0, -2330.0],
                 smoothing_sigma=1.0, include_robustness=True)

    assert (output / "robustness_scenarios.csv").is_file()
    assert (output / "robustness_comparison.png").is_file()
    report = (output / "analysis_report.md").read_text(encoding="utf-8")
    assert "Robustness analysis was skipped" not in report
    assert "- Missing data:" in report
    assert "- Missing Data:" not in report
    assert "robustness_scenarios.csv" in report
    assert "robustness_comparison.png" in report



def test_irregular_coordinate_spacing_and_infinite_depth_are_rejected(tmp_path):
    from contour_analysis import load_grid

    irregular = pd.DataFrame({
        "x_m": [0, 1, 3, 0, 1, 3],
        "y_m": [0, 0, 0, 1, 1, 1],
        "depth_m": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })
    path = tmp_path / "irregular.csv"
    irregular.to_csv(path, index=False)
    with pytest.raises(ValueError, match="regularly spaced"):
        load_grid(path)
    irregular["x_m"] = [0, 1, 2, 0, 1, 2]
    irregular.loc[0, "depth_m"] = np.inf
    irregular.to_csv(path, index=False)
    with pytest.raises(ValueError, match="infinities"):
        load_grid(path)


def test_sensitivity_is_reproducible_for_fixed_surface_and_thresholds():
    from geoscience_toolkit import sensitivity_analysis

    xs = np.arange(5.0)
    ys = np.arange(5.0)
    grid = np.zeros((5, 5), dtype=float)
    grid[2, 2] = 10.0
    first = sensitivity_analysis(xs, ys, grid, thresholds_m=[2.0, 5.0, 8.0], smoothing_sigma=0.0)
    second = sensitivity_analysis(xs, ys, grid, thresholds_m=[2.0, 5.0, 8.0], smoothing_sigma=0.0)
    pd.testing.assert_frame_equal(first, second)

"""Reproducible synthetic geoscience analysis, sensitivity and figure generation."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from contour_analysis import analyze_grid, load_grid, write_outputs
from synthetic_grid import make_synthetic_grid

SCALAR_FIELDS = (
    "crest_x_m", "crest_y_m", "crest_depth_m", "crest_smoothed_depth_m",
    "spill_threshold_m", "relief_m", "spill_x_m", "spill_y_m", "grid_spill_depth_m",
    "grid_exit_reachable", "closure_area_m2", "closure_area_km2", "has_closure",
    "closed_closure", "closure_status", "boundary_touched", "closure_node_count",
    "grid_nx", "grid_ny", "grid_spacing_x_m", "grid_spacing_y_m", "valid_node_count",
    "total_node_count", "valid_data_coverage_pct", "valid_mesh_coverage_pct", "missing_node_count",
)


def _flat_record(result: dict, **extra) -> dict:
    row = {key: result[key] for key in SCALAR_FIELDS}
    sigma = result["smoothing_sigma_cells"]
    if np.isscalar(sigma):
        row["smoothing_sigma_y_cells"] = float(sigma)
        row["smoothing_sigma_x_cells"] = float(sigma)
    else:
        row["smoothing_sigma_y_cells"], row["smoothing_sigma_x_cells"] = map(float, sigma)
    row.update(extra)
    return row


def sensitivity_analysis(xs: np.ndarray, ys: np.ndarray, grid: np.ndarray, *,
                         thresholds_m: list[float] | np.ndarray | None = None,
                         smoothing_sigma: float | tuple[float, float] = 1.0,
                         min_area_km2: float = 0.0) -> pd.DataFrame:
    """Calculate geometric metrics across selected depth thresholds.

    ``thresholds_m`` retains its public name for compatibility. Larger values
    are shallower under the required input convention.
    """
    reference = analyze_grid(xs, ys, grid, spill_depth_m=float(np.nanmax(grid) - 60.0),
                             smoothing_sigma=smoothing_sigma, min_area_km2=min_area_km2)
    if thresholds_m is None:
        top = reference["crest_depth_m"] - 20.0
        bottom = reference["grid_spill_depth_m"] + 5.0
        if not np.isfinite(bottom) or bottom >= top:
            bottom, top = reference["crest_depth_m"] - 40.0, reference["crest_depth_m"] - 5.0
        thresholds_m = np.linspace(top, bottom, 8)
    values = np.asarray(thresholds_m, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("sensitivity analysis requires at least two finite selected depth thresholds")
    rows = []
    for threshold in values:
        result = analyze_grid(xs, ys, grid, spill_depth_m=float(threshold),
                              smoothing_sigma=smoothing_sigma, min_area_km2=min_area_km2)
        row = _flat_record(result)
        row["spill_depth_threshold_m"] = float(threshold)
        rows.append(row)
    return pd.DataFrame(rows)


def robustness_analysis(*, seed: int = 20261026) -> pd.DataFrame:
    """Run controlled smoothing, missing-data and resolution experiments on synthetic surfaces."""
    base = make_synthetic_grid(seed=seed, null_fraction=0.02, noise_std_m=1.5)
    xs, ys = np.sort(base.x_m.unique()), np.sort(base.y_m.unique())
    grid = base.pivot(index="y_m", columns="x_m", values="depth_m").reindex(index=ys, columns=xs).to_numpy(float)
    fixed_threshold = float(np.nanmax(grid) - 60.0)
    rows: list[dict] = []

    for sigma in (0.0, 1.0, 2.0):
        result = analyze_grid(xs, ys, grid, spill_depth_m=fixed_threshold, smoothing_sigma=sigma)
        rows.append(_flat_record(result, experiment="smoothing", scenario=f"{sigma:g} cells",
                                 parameter_value=sigma, fixed_threshold_m=fixed_threshold))

    for fraction in (0.0, 0.02, 0.10):
        frame = make_synthetic_grid(seed=seed, null_fraction=fraction, noise_std_m=1.5)
        gx, gy = np.sort(frame.x_m.unique()), np.sort(frame.y_m.unique())
        g = frame.pivot(index="y_m", columns="x_m", values="depth_m").reindex(index=gy, columns=gx).to_numpy(float)
        result = analyze_grid(gx, gy, g, spill_depth_m=fixed_threshold, smoothing_sigma=1.0)
        rows.append(_flat_record(result, experiment="missing_data", scenario=f"{fraction:.0%} null nodes",
                                 parameter_value=fraction, fixed_threshold_m=fixed_threshold))

    # Zero noise isolates sampling resolution from a different random-noise realization.
    ideal_base = make_synthetic_grid(seed=seed, noise_std_m=0.0, null_fraction=0.0)
    ix, iy = np.sort(ideal_base.x_m.unique()), np.sort(ideal_base.y_m.unique())
    ig = ideal_base.pivot(index="y_m", columns="x_m", values="depth_m").reindex(index=iy, columns=ix).to_numpy(float)
    ideal_threshold = float(np.nanmax(ig) - 60.0)
    for nx, ny in ((51, 46), (101, 91), (201, 181)):
        frame = make_synthetic_grid(nx=nx, ny=ny, seed=seed, noise_std_m=0.0, null_fraction=0.0)
        gx, gy = np.sort(frame.x_m.unique()), np.sort(frame.y_m.unique())
        g = frame.pivot(index="y_m", columns="x_m", values="depth_m").reindex(index=gy, columns=gx).to_numpy(float)
        dx, dy = float(np.diff(gx).mean()), float(np.diff(gy).mean())
        # Keep smoothing length constant at 100 m in physical coordinates.
        sigma = (100.0 / dy, 100.0 / dx)
        result = analyze_grid(gx, gy, g, spill_depth_m=ideal_threshold, smoothing_sigma=sigma)
        rows.append(_flat_record(result, experiment="resolution", scenario=f"{nx} x {ny} nodes",
                                 parameter_value=nx, fixed_threshold_m=ideal_threshold))
    return pd.DataFrame(rows)


def plot_structural_map(result: dict, path: str | Path, *, contour_step_m: float = 20.0) -> Path:
    xs, ys, grid = result["xs"], result["ys"], result["grid"]
    values = grid[np.isfinite(grid)]
    levels = np.arange(np.floor(values.min() / contour_step_m) * contour_step_m,
                       np.ceil(values.max() / contour_step_m) * contour_step_m + contour_step_m,
                       contour_step_m)
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    filled = ax.contourf(xs, ys, np.ma.masked_invalid(grid), levels=levels, cmap="viridis", extend="both")
    ax.contour(xs, ys, np.ma.masked_invalid(result["smoothed"]), levels=levels,
               colors="black", linewidths=0.5, alpha=0.75)
    threshold_drawn = result["has_closure"] and values.min() < result["spill_threshold_m"] < values.max()
    if threshold_drawn:
        ax.contour(xs, ys, np.ma.masked_invalid(result["smoothed"]),
                   levels=[result["spill_threshold_m"]], colors="#ffbe0b", linewidths=2.2, linestyles="--")
    if result["has_closure"]:
        boundary = np.ma.array(result["closure"].astype(float), mask=~np.isfinite(grid))
        ax.contour(xs, ys, boundary, levels=[0.5], colors=["#e63946"], linewidths=2.0)
    ax.scatter(result["crest_x_m"], result["crest_y_m"], marker="*", s=170,
               color="#ffcf33", edgecolor="black", label="Raw-grid crest", zorder=5)
    if result["grid_exit_reachable"]:
        ax.scatter(result["spill_x_m"], result["spill_y_m"], marker="X", s=100,
                   color="#ff4d4d", edgecolor="black", label="Grid-exit saddle", zorder=5)
    handles, labels = ax.get_legend_handles_labels()
    if threshold_drawn:
        handles.append(Line2D([0], [0], color="#ffbe0b", linewidth=2.2, linestyle="--"))
        labels.append("Selected threshold contour")
    if result["has_closure"]:
        handles.append(Line2D([0], [0], color="#e63946", linewidth=2.0))
        labels.append("Crest-connected threshold-region boundary")
    ax.set(title=f"Synthetic structural contours — {result['closure_status'].replace('_', ' ')}",
           xlabel="Synthetic x (m)", ylabel="Synthetic y (m)")
    ax.set_aspect("equal")
    ax.legend(handles, labels, loc="best")
    fig.colorbar(filled, ax=ax, label="Synthetic depth (m; larger values are shallower)")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path

def plot_surface_3d(result: dict, path: str | Path) -> Path:
    xs, ys, grid = result["xs"], result["ys"], result["grid"]
    xx, yy = np.meshgrid(xs, ys)
    fig = plt.figure(figsize=(10, 7), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(xx, yy, np.ma.masked_invalid(grid), cmap="viridis", linewidth=0,
                              antialiased=True, rstride=max(1, len(ys) // 100), cstride=max(1, len(xs) // 100))
    ax.scatter(result["crest_x_m"], result["crest_y_m"], result["crest_depth_m"],
               color="#ffcf33", edgecolor="black", marker="*", s=100, label="Synthetic crest")
    ax.set(title="Synthetic structural surface (3D)", xlabel="x (m)", ylabel="y (m)",
           zlabel="Depth (m; larger values are shallower)")
    ax.legend(loc="best")
    fig.colorbar(surface, ax=ax, shrink=0.62, pad=0.12,
                 label="Synthetic depth (m; larger values are shallower)")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_cross_sections(result: dict, path: str | Path) -> Path:
    xs, ys = result["xs"], result["ys"]
    row, col = result["crest_row"], result["crest_col"]
    raw = result["grid"]
    smoothed = result["smoothed"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    axes[0].plot(xs, raw[row, :], color="#4267ac", marker=".", label="Input grid")
    axes[0].plot(xs, smoothed[row, :], color="#d34a42", label="Normalized Gaussian-smoothed surface")
    axes[0].axhline(result["spill_threshold_m"], color="black", linestyle="--", label="Selected threshold")
    axes[0].set(title=f"E–W section at y = {ys[row]:g} m", xlabel="x (m)", ylabel="Depth (m; larger is shallower)")
    axes[1].plot(ys, raw[:, col], color="#4267ac", marker=".", label="Input grid")
    axes[1].plot(ys, smoothed[:, col], color="#d34a42", label="Normalized Gaussian-smoothed surface")
    axes[1].axhline(result["spill_threshold_m"], color="black", linestyle="--", label="Selected threshold")
    axes[1].set(title=f"N–S section at x = {xs[col]:g} m", xlabel="y (m)", ylabel="Depth (m; larger is shallower)")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(loc="best")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_sensitivity(sensitivity: pd.DataFrame, output_dir: str | Path,
                    xs: np.ndarray, ys: np.ndarray, grid: np.ndarray,
                    smoothing_sigma: float | tuple[float, float]) -> tuple[Path, Path, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ordered = sensitivity.sort_values("spill_depth_threshold_m")
    area_path, relief_path, scenarios_path = (out / "closure_area_sensitivity.png",
                                                out / "structural_relief_sensitivity.png",
                                                out / "closure_scenarios.png")
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(ordered.spill_depth_threshold_m, ordered.closure_area_km2, marker="o", color="#2878a5")
    ax.set(title="Closure area sensitivity", xlabel="Selected depth threshold (m; larger is shallower)",
           ylabel="Estimated closure area (km²)")
    ax.grid(alpha=0.25)
    fig.savefig(area_path, dpi=220)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(ordered.spill_depth_threshold_m, ordered.relief_m, marker="o", color="#bc5b45")
    ax.set(title="Structural relief sensitivity", xlabel="Selected depth threshold (m; larger is shallower)",
           ylabel="Raw-crest relief (m)")
    ax.grid(alpha=0.25)
    fig.savefig(relief_path, dpi=220)
    plt.close(fig)
    picks = sorted(set((0, len(ordered) // 2, len(ordered) - 1)))
    fig, axes = plt.subplots(1, len(picks), figsize=(5.2 * len(picks), 5), constrained_layout=True, squeeze=False)
    raw_values = grid[np.isfinite(grid)]
    base_min, base_max = float(raw_values.min()), float(raw_values.max())
    if np.isclose(base_min, base_max):
        padding = max(abs(base_min) * 0.01, 1.0)
        base_min, base_max = base_min - padding, base_max + padding
    base_levels = np.linspace(base_min, base_max, 16)
    for ax, index in zip(axes[0], picks):
        row = ordered.iloc[index]
        scenario = analyze_grid(xs, ys, grid, spill_depth_m=float(row.spill_depth_threshold_m),
                                smoothing_sigma=smoothing_sigma)
        ax.contourf(xs, ys, np.ma.masked_invalid(grid), levels=base_levels, cmap="Greys", alpha=0.35)
        if scenario["has_closure"]:
            boundary = np.ma.array(scenario["closure"].astype(float), mask=~np.isfinite(grid))
            ax.contour(xs, ys, boundary, levels=[0.5], colors=["#c9473e"], linewidths=2.0)
        ax.scatter(scenario["crest_x_m"], scenario["crest_y_m"], marker="*",
                   color="#ffcf33", edgecolor="black", s=110, zorder=4)
        ax.set(title=f"Threshold {row.spill_depth_threshold_m:.1f} m\nArea {row.closure_area_km2:.2f} km²; {row.closure_status}",
               xlabel="x (m)", ylabel="y (m)")
        ax.set_aspect("equal")
    fig.suptitle("Selected-threshold region comparison")
    fig.savefig(scenarios_path, dpi=220)
    plt.close(fig)
    return area_path, relief_path, scenarios_path

def plot_robustness(robustness: pd.DataFrame, path: str | Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    for ax, group_name in zip(axes, ("smoothing", "missing_data", "resolution")):
        subset = robustness[robustness.experiment == group_name]
        ax.plot(np.arange(len(subset)), subset.closure_area_km2, marker="o", color="#347c74")
        ax.set_xticks(np.arange(len(subset)), subset.scenario, rotation=20, ha="right")
        ax.set(title=group_name.replace("_", " ").capitalize(), ylabel="Estimated closure area (km²)")
        ax.grid(alpha=0.2)
    fig.suptitle("Controlled synthetic robustness experiments")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def write_report(metrics: dict, sensitivity: pd.DataFrame, robustness: pd.DataFrame,
                 out_dir: str | Path, input_path: str | Path) -> Path:
    smoothed_row, smoothed_col = np.unravel_index(
        np.nanargmax(metrics["smoothed"]), metrics["smoothed"].shape)
    smoothed_x = float(metrics["xs"][smoothed_col])
    smoothed_y = float(metrics["ys"][smoothed_row])
    smoothed_depth = float(metrics["smoothed"][smoothed_row, smoothed_col])
    out = Path(out_dir)
    closed = sensitivity[sensitivity.closure_status == "closed"]
    areas = (f"{closed.closure_area_km2.min():.3f}–{closed.closure_area_km2.max():.3f} km²"
             if not closed.empty else "no closed region in the tested thresholds")
    robust_groups = []
    for name, subset in robustness.groupby("experiment", sort=False):
        robust_groups.append(f"- {name.replace('_', ' ').capitalize()}: estimated closure area ranges "
                             f"{subset.closure_area_km2.min():.3f}–{subset.closure_area_km2.max():.3f} km² "
                             f"across the tested scenarios.")
    if robustness.empty:
        robustness_summary = "Robustness analysis was skipped for this run. No robustness results or figures were generated."
        robustness_output = ""
        robustness_machine_output = ""
    else:
        robustness_summary = (f"{chr(10).join(robust_groups)} The resolution experiment uses a zero-noise surface to isolate sampling. "
                              "Missing-data and smoothing experiments use fixed seeds; valid-mesh coverage can change because triangles "
                              "touching missing depth nodes are excluded. See `robustness_scenarios.csv` and `robustness_comparison.png`.")
        robustness_output = "- `robustness_comparison.png`: estimated closure-area comparison across the controlled experiments.\n"
        robustness_machine_output = ", `robustness_scenarios.csv`"
    text = f"""# Synthetic structural-contour analysis report

## Dataset and quality

Input: `{Path(input_path).as_posix()}`. This is a fictional synthetic depth surface generated by this project. The grid has {metrics['grid_nx']} x-nodes by {metrics['grid_ny']} y-nodes, spacing {metrics['grid_spacing_x_m']:.3f} m (x) and {metrics['grid_spacing_y_m']:.3f} m (y), {metrics['valid_data_coverage_pct']:.2f}% valid nodes, and {metrics['valid_mesh_coverage_pct']:.2f}% triangulated mesh coverage. Missing depth nodes remain excluded from contour regions; mesh triangles with any missing vertex are omitted.

## Synthetic geometric results at the selected threshold

The analysis assumes larger numerical depth values are shallower. The synthetic input happens to use negative metre values; negative values are not required for other supported inputs. The raw-grid crest, the maximum valid value in the unsmoothed input, is at ({metrics['crest_x_m']:.1f}, {metrics['crest_y_m']:.1f}) m with depth {metrics['crest_depth_m']:.3f} m.

The selected depth threshold is {metrics['spill_threshold_m']:.3f} m. It is a scenario parameter used to identify the 8-connected threshold region on the normalized smoothed surface at the raw-grid crest location; it is not necessarily a geologically interpreted spill point. Raw-crest relief is the arithmetic difference between the raw-grid crest depth and the selected threshold ({metrics['relief_m']:.3f} m).

The smoothed-surface maximum is at ({smoothed_x:.1f}, {smoothed_y:.1f}) m with depth {smoothed_depth:.3f} m. It may differ from the raw-grid crest and is the starting node for the grid-exit saddle calculation. The grid-exit saddle is at ({metrics['spill_x_m']:.1f}, {metrics['spill_y_m']:.1f}) m with smoothed depth {metrics['grid_spill_depth_m']:.3f} m. It is the bottleneck point along the maximum-bottleneck 8-neighbour path from the smoothed-surface maximum to the model-domain edge. This computed saddle is domain dependent; it is not an independently interpreted geological spill point or hydrocarbon contact. The selected threshold contour and grid-exit saddle are distinct quantities.

Area is integrated over a piecewise-linear triangular mesh constructed from the normalized smoothed surface. Each triangle is clipped at the selected depth threshold, and clipped areas are summed only for valid triangles associated with the crest-connected threshold region. Triangles containing any missing depth node are excluded. The resulting area is an approximation based on grid resolution, smoothing, threshold choice and missing-data coverage, not a polygonized geological boundary. The computed area is {metrics['closure_area_m2']:.1f} m² ({metrics['closure_area_km2']:.6f} km²); geometric status: **{metrics['closure_status']}**.

## Threshold sensitivity

Tested {len(sensitivity)} selected depth thresholds. Across scenarios classified as closed, estimated area spans {areas}. Rows marked `boundary_limited` reach the model boundary and do not represent fully enclosed regions. See `spill_sensitivity.csv`, `closure_area_sensitivity.png`, `structural_relief_sensitivity.png`, and `closure_scenarios.png`.

## Controlled robustness experiments

{robustness_summary}

## Figures and outputs

- `structural_contours.png`: input contours, selected threshold and crest-connected boundary, raw crest and grid-exit saddle.
- `{Path(input_path).stem}_contours.png` and `{Path(input_path).stem}_metrics.csv`: compatibility outputs from the original single-threshold workflow.
- `surface_3d.png`: 3D synthetic surface.
- `cross_sections.png`: E–W and N–S sections through the raw crest node.
- `closure_area_sensitivity.png`, `structural_relief_sensitivity.png`: threshold response.
- `closure_scenarios.png`: selected threshold-contour comparison.
- `metrics_summary.csv`, `spill_sensitivity.csv`{robustness_machine_output}: machine-readable results.
{robustness_output.rstrip()}

## Assumptions and limitations

All values and figures in this report are synthetic. The raw-grid crest is the maximum valid input node; the smoothed-surface maximum is a separate point used to start the grid-exit saddle path. Threshold regions use the normalized smoothed surface and 8-neighbour connectivity while retaining the original missing-depth mask. Area is an approximation from fixed-diagonal, piecewise-linear triangles clipped at the selected threshold; triangles with missing depth nodes are excluded. Area depends on grid resolution, smoothing, threshold choice and missing-data coverage. Boundary-limited regions are not fully enclosed closures. The grid-exit saddle depends on model extent, sampling and smoothing. Neither a closure-shaped high nor any calculated metric establishes hydrocarbon charge, seal integrity, reservoir quality, accumulation, reserves or commercial viability.
"""
    path = out / "analysis_report.md"
    path.write_text(text, encoding="utf-8")
    return path


def run_analysis(input_path: str | Path, output_dir: str | Path, *,
                 spill_depth_m: float | None = None,
                 thresholds_m: list[float] | None = None,
                 smoothing_sigma: float = 1.0,
                 min_area_km2: float = 0.0,
                 seed: int = 20261026,
                 include_robustness: bool = True) -> dict:
    """Run the synthetic or a conforming user-supplied grid through the workflow.

    ``spill_depth_m`` retains its public name for compatibility and supplies the
    selected depth threshold; it is distinct from the calculated grid-exit saddle.
    """
    xs, ys, grid = load_grid(input_path)
    raw_crest = float(np.nanmax(grid))
    selected_threshold = raw_crest - 60.0 if spill_depth_m is None else float(spill_depth_m)
    result = analyze_grid(xs, ys, grid, spill_depth_m=selected_threshold,
                          smoothing_sigma=smoothing_sigma, min_area_km2=min_area_km2)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sensitivity = sensitivity_analysis(xs, ys, grid, thresholds_m=thresholds_m,
                                       smoothing_sigma=smoothing_sigma, min_area_km2=min_area_km2)
    robustness = robustness_analysis(seed=seed) if include_robustness else pd.DataFrame(
        columns=["experiment", "scenario", "closure_area_km2"])
    legacy_map, legacy_metrics = write_outputs(result, out, stem=Path(input_path).stem)
    files = {
        "legacy_map": legacy_map, "legacy_metrics": legacy_metrics,
        "structural_map": plot_structural_map(result, out / "structural_contours.png"),
        "surface_3d": plot_surface_3d(result, out / "surface_3d.png"),
        "cross_sections": plot_cross_sections(result, out / "cross_sections.png"),
    }
    sensitivity.to_csv(out / "spill_sensitivity.csv", index=False)
    files["area_sensitivity"], files["relief_sensitivity"], files["closure_scenarios"] = plot_sensitivity(sensitivity, out, xs, ys, grid, smoothing_sigma)
    if include_robustness:
        robustness.to_csv(out / "robustness_scenarios.csv", index=False)
        files["robustness"] = plot_robustness(robustness, out / "robustness_comparison.png")
    else:
        # Do not leave prior-run artifacts beside a report that says this run skipped robustness.
        for stale_name in ("robustness_scenarios.csv", "robustness_comparison.png"):
            stale_path = out / stale_name
            if stale_path.exists():
                stale_path.unlink()
    metrics_row = _flat_record(result)
    metrics_row["input_file"] = Path(input_path).name
    pd.DataFrame([metrics_row]).to_csv(out / "metrics_summary.csv", index=False)
    files["report"] = write_report(result, sensitivity, robustness, out, input_path)
    return {"result": result, "sensitivity": sensitivity, "robustness": robustness,
            "files": files, "metrics_path": out / "metrics_summary.csv",
            "sensitivity_path": out / "spill_sensitivity.csv",
            "robustness_path": out / "robustness_scenarios.csv" if include_robustness else None,
            "report_path": files["report"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/synthetic/synthetic_surface.csv"),
                        help="synthetic CSV or user-supplied regular-grid CSV meeting the documented requirements")
    parser.add_argument("--output-dir", type=Path, default=Path("examples/output"))
    parser.add_argument("--spill-depth-m", type=float, default=None,
                        help="selected depth threshold (scenario parameter); defaults to 60 m below the raw-grid crest")
    parser.add_argument("--spill-depths-m", type=float, nargs="+", default=None,
                        help="at least two selected depth thresholds for sensitivity; larger values are shallower")
    parser.add_argument("--smoothing-sigma", type=float, default=1.0,
                        help="Gaussian sigma in grid cells; normalized smoothing preserves null mask")
    parser.add_argument("--min-area-km2", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--skip-robustness", action="store_true")
    args = parser.parse_args()
    outputs = run_analysis(args.input, args.output_dir, spill_depth_m=args.spill_depth_m,
                           thresholds_m=args.spill_depths_m, smoothing_sigma=args.smoothing_sigma,
                           min_area_km2=args.min_area_km2, seed=args.seed,
                           include_robustness=not args.skip_robustness)
    result = outputs["result"]
    print(f"Synthetic structural analysis: {result['closure_status']}")
    print(f"Raw-grid crest: ({result['crest_x_m']:.1f}, {result['crest_y_m']:.1f}) m at {result['crest_depth_m']:.3f} m")
    print(f"Selected depth threshold: {result['spill_threshold_m']:.3f} m; raw-crest relief: {result['relief_m']:.3f} m")
    print(f"Grid-exit saddle: ({result['spill_x_m']:.1f}, {result['spill_y_m']:.1f}) m at {result['grid_spill_depth_m']:.3f} m")
    print(f"Estimated closure area: {result['closure_area_m2']:.1f} m² ({result['closure_area_km2']:.6f} km²)")
    for name, path in outputs["files"].items():
        print(f"{name}: {path}")
    print(f"metrics: {outputs['metrics_path']}")
    print(f"sensitivity: {outputs['sensitivity_path']}")
    if outputs["robustness_path"]:
        print(f"robustness: {outputs['robustness_path']}")


if __name__ == "__main__":
    main()

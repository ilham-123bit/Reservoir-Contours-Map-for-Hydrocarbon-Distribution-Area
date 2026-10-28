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
    """Calculate closure metrics across thresholds; higher values are shallower."""
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
        raise ValueError("sensitivity analysis requires at least two finite spill-depth thresholds")
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
        labels.append("Crest-connected boundary")
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
           zlabel="Depth (m; more negative is deeper)")
    ax.legend(loc="best")
    fig.colorbar(surface, ax=ax, shrink=0.62, pad=0.12, label="Synthetic depth (m)")
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
    axes[0].plot(xs, smoothed[row, :], color="#d34a42", label="Normalized Gaussian smooth")
    axes[0].axhline(result["spill_threshold_m"], color="black", linestyle="--", label="Selected threshold")
    axes[0].set(title=f"E–W section at y = {ys[row]:g} m", xlabel="x (m)", ylabel="Depth (m; larger is shallower)")
    axes[1].plot(ys, raw[:, col], color="#4267ac", marker=".", label="Input grid")
    axes[1].plot(ys, smoothed[:, col], color="#d34a42", label="Normalized Gaussian smooth")
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
    ax.set(title="Closure area sensitivity", xlabel="Selected spill-depth threshold (m; larger is shallower)",
           ylabel="Triangulated closure area (km²)")
    ax.grid(alpha=0.25)
    fig.savefig(area_path, dpi=220)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(ordered.spill_depth_threshold_m, ordered.relief_m, marker="o", color="#bc5b45")
    ax.set(title="Structural relief sensitivity", xlabel="Selected spill-depth threshold (m; larger is shallower)",
           ylabel="Relief from raw-grid crest (m)")
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
    fig.suptitle("Crest-connected closure-region comparison")
    fig.savefig(scenarios_path, dpi=220)
    plt.close(fig)
    return area_path, relief_path, scenarios_path

def plot_robustness(robustness: pd.DataFrame, path: str | Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    for ax, group_name in zip(axes, ("smoothing", "missing_data", "resolution")):
        subset = robustness[robustness.experiment == group_name]
        ax.plot(np.arange(len(subset)), subset.closure_area_km2, marker="o", color="#347c74")
        ax.set_xticks(np.arange(len(subset)), subset.scenario, rotation=20, ha="right")
        ax.set(title=group_name.replace("_", " ").title(), ylabel="Closure area (km²)")
        ax.grid(alpha=0.2)
    fig.suptitle("Controlled synthetic robustness experiments")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def write_report(metrics: dict, sensitivity: pd.DataFrame, robustness: pd.DataFrame,
                 out_dir: str | Path, input_path: str | Path) -> Path:
    out = Path(out_dir)
    closed = sensitivity[sensitivity.closure_status == "closed"]
    areas = (f"{closed.closure_area_km2.min():.3f}–{closed.closure_area_km2.max():.3f} km²"
             if not closed.empty else "no closed region in the tested thresholds")
    robust_groups = []
    for name, subset in robustness.groupby("experiment", sort=False):
        robust_groups.append(f"- {name.replace('_', ' ').title()}: closure area ranges "
                             f"{subset.closure_area_km2.min():.3f}–{subset.closure_area_km2.max():.3f} km² "
                             f"across the tested scenarios.")
    if robustness.empty:
        robustness_summary = "Robustness analysis was skipped for this run. No robustness results or figures were generated."
        robustness_output = ""
        robustness_machine_output = ""
    else:
        robustness_summary = (f"{chr(10).join(robust_groups)} The resolution experiment uses a zero-noise surface to isolate sampling. "
                              "Missingness and smoothing experiments use fixed seeds; missing-data coverage changes because triangles "
                              "touching null nodes are not integrated. See `robustness_scenarios.csv` and `robustness_comparison.png`.")
        robustness_output = "- `robustness_comparison.png`: closure-area comparison across the controlled experiments.\n"
        robustness_machine_output = ", `robustness_scenarios.csv`"
    text = f"""# Synthetic structural-contour analysis report

## Dataset and quality

Input: `{Path(input_path).as_posix()}`. This is a fictional synthetic depth surface generated by this project. The grid has {metrics['grid_nx']} x-nodes by {metrics['grid_ny']} y-nodes, spacing {metrics['grid_spacing_x_m']:.3f} m (x) and {metrics['grid_spacing_y_m']:.3f} m (y), {metrics['valid_data_coverage_pct']:.2f}% valid nodes, and {metrics['valid_mesh_coverage_pct']:.2f}% triangulated mesh coverage. Missing depth nodes remain excluded from contour regions; mesh triangles with any missing vertex are omitted.

## Structural interpretation at the selected threshold

The depth convention is larger numerical value = shallower structure (the synthetic depths are negative metres). The raw-grid crest is at ({metrics['crest_x_m']:.1f}, {metrics['crest_y_m']:.1f}) m with depth {metrics['crest_depth_m']:.3f} m. The selected threshold is {metrics['spill_threshold_m']:.3f} m, giving raw-crest relief {metrics['relief_m']:.3f} m. The computed region area is {metrics['closure_area_m2']:.1f} m² ({metrics['closure_area_km2']:.6f} km²); status: **{metrics['closure_status']}**.

The reported grid-exit saddle is ({metrics['spill_x_m']:.1f}, {metrics['spill_y_m']:.1f}) m at smoothed depth {metrics['grid_spill_depth_m']:.3f} m. It is the bottleneck node on the maximum-bottleneck 8-neighbor route from the maximum of the normalized smoothed surface to the model-domain edge. This smoothed crest may differ from the raw-grid crest reported above and used to identify the threshold closure component. The saddle is domain dependent and is not an independently interpreted geologic spill point. The selected threshold contour and grid-exit saddle are distinct quantities.

## Threshold sensitivity

Tested {len(sensitivity)} thresholds. Across scenarios classified as closed, area spans {areas}. Rows marked `boundary_limited` reach the model boundary and should not be interpreted as a fully enclosed closure. See `spill_sensitivity.csv`, `closure_area_sensitivity.png`, `structural_relief_sensitivity.png`, and `closure_scenarios.png`.

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
{robustness_output}

## Assumptions and limitations

The surface is synthetic and has no real reservoir meaning. Crest depth is the maximum valid raw-grid value. Threshold regions use normalized Gaussian smoothing and 8-neighbor node connectivity. Area is integrated over a fixed-diagonal piecewise-linear triangular mesh by clipping each valid triangle at the selected threshold; it is an approximation to a continuous surface and is limited by grid resolution and missing cells. Boundary-touching regions are flagged. The digital grid-exit saddle depends on model extent, sampling and smoothing. Smoothing can shift or suppress local highs; the raw crest is retained for reporting. Neither a closure-shaped high nor any calculated metric proves hydrocarbon charge, seal, reservoir quality or accumulation.
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
    parser.add_argument("--input", type=Path, default=Path("data/synthetic/synthetic_surface.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("examples/output"))
    parser.add_argument("--spill-depth-m", type=float, default=None,
                        help="selected threshold; defaults to 60 m below the raw crest")
    parser.add_argument("--spill-depths-m", type=float, nargs="+", default=None,
                        help="at least two thresholds for sensitivity; larger values are shallower")
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
    print(f"Crest: ({result['crest_x_m']:.1f}, {result['crest_y_m']:.1f}) m at {result['crest_depth_m']:.3f} m")
    print(f"Selected threshold: {result['spill_threshold_m']:.3f} m; relief: {result['relief_m']:.3f} m")
    print(f"Grid-exit saddle: ({result['spill_x_m']:.1f}, {result['spill_y_m']:.1f}) m at {result['grid_spill_depth_m']:.3f} m")
    print(f"Area: {result['closure_area_m2']:.1f} m2 ({result['closure_area_km2']:.6f} km2)")
    for name, path in outputs["files"].items():
        print(f"{name}: {path}")
    print(f"metrics: {outputs['metrics_path']}")
    print(f"sensitivity: {outputs['sensitivity_path']}")
    if outputs["robustness_path"]:
        print(f"robustness: {outputs['robustness_path']}")


if __name__ == "__main__":
    main()

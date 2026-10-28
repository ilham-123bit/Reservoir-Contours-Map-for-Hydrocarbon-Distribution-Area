# Synthetic Reservoir Modelling & Contour Analysis

**Geoscience data analysis portfolio demonstration.** Every input, calculation, figure and reported result in this public demo uses the project’s fictional synthetic surface. Nothing represents a real reservoir, operator, field, prospect or project.

**Branch note:** The implementation described below is currently on `feature/notebook-upload`. Switch to that branch to access the scripts, synthetic dataset, notebook, tests and generated outputs referenced here. This README update on `main` documents that work; it does not merge those files into `main`.

## What the workflow does

1. Generate or load a regular x/y/depth grid and report grid spacing, missing-node coverage and usable mesh coverage.
2. Validate coordinate completeness, uniqueness, regular spacing and finite depth values. Irregular or incomplete coordinate grids are rejected with an explanatory error; missing depth cells are allowed and remain masked.
3. Interpret the largest numerical depth value as the shallowest structural crest. The example uses negative depth values in metres, so more negative values are deeper.
4. Apply normalized Gaussian smoothing to valid nodes without filling null locations. Identify the 8-connected region at a selected spill-depth threshold that contains the raw-grid crest.
5. Estimate area by clipping valid triangles in a fixed-diagonal, piecewise-linear mesh at the selected threshold. Report area in m² and km², relief, node counts, boundary contact and coverage.
6. Calculate a domain-dependent digital grid-exit saddle using the maximum-bottleneck 8-neighbour path from the raw-grid crest location across the normalized smoothed grid to an edge node. The saddle elevation is read from that smoothed grid.
7. Repeat the closure measurements over spill thresholds and controlled smoothing, missing-data and resolution scenarios. Export CSVs, figures and a concise report.

The grid-exit saddle is an algorithmic reference level for this finite synthetic model; it is not a claim of a geologic spill point. The path starts at the raw-grid crest cell, while path elevations are evaluated on the normalized smoothed surface. A threshold region marked `boundary_limited` reaches the model edge and is not a fully enclosed closure. A structural high or closure-shaped contour is not evidence of hydrocarbons, seal, reservoir quality or an accumulation.

## Installation

Use Python 3.10 or newer. Install the listed runtime, plotting, notebook and test dependencies from the repository root:

```powershell
python -m pip install -r requirements.txt
```

## Reproduce the complete analysis

Run these commands from the repository root. The first regenerates the deterministic 101 x 91 synthetic surface using seed `20261026`; it writes only the synthetic CSV. The second loads and validates that grid, calculates the selected closure and threshold sweep, runs the fixed-seed robustness scenarios, then writes the report, tables and figures to `examples/output/`:

```powershell
python src/generate_synthetic_data.py --output data/synthetic/synthetic_surface.csv --seed 20261026
python src/geoscience_toolkit.py --input data/synthetic/synthetic_surface.csv --output-dir examples/output --spill-depths-m -2290 -2310 -2330 -2350 -2370 -2390
```

Threshold values use the dataset’s depth convention: **larger values are shallower**. Omit `--spill-depths-m` to use a reproducible default sweep; omit `--spill-depth-m` to set the primary analysis threshold to 60 m below the raw crest. Use `--smoothing-sigma` to choose Gaussian sigma in grid cells, `--seed` for robustness scenarios, or `--skip-robustness` to omit the robustness experiment and its outputs for a faster run. Run `python src/geoscience_toolkit.py --help` for options.

The original single-threshold command remains available for compatibility:

```powershell
python src/analyze_grid.py --input data/synthetic/synthetic_surface.csv --spill-depth-m -2321.589585 --output-dir examples/output --contour-step-m 20
```

Run the automated tests with:

```powershell
python -m pytest -q
```

Open `notebooks/synthetic_contour_demo.ipynb` from the repository root in JupyterLab for the guided workflow: inspect the data and quality checks, view the surface, quantify closure, compare spill thresholds, inspect cross-sections and 3D geometry, run robustness scenarios, and review the generated summary and limitations.

## Generated outputs

- `metrics_summary.csv`: one-row dataset, quality, crest, grid-saddle, relief and closure metrics.
- `spill_sensitivity.csv`: one row per requested threshold, including area, relief, spill reference, connectivity and edge flags.
- `robustness_scenarios.csv`: controlled smoothing, null-fraction and grid-resolution experiments.
- `analysis_report.md`: calculated summary, interpretation, assumptions, limitations and output references.
- `structural_contours.png`: annotated synthetic depth map with crest, threshold and crest-connected boundary, and grid-exit saddle.
- `surface_3d.png`: 3D view of the synthetic surface.
- `cross_sections.png`: E–W and N–S sections through the raw crest node.
- `closure_area_sensitivity.png` and `structural_relief_sensitivity.png`: response to spill-depth threshold.
- `closure_scenarios.png`: connected closure-boundary comparison at selected thresholds.
- `robustness_comparison.png`: closure-area comparison across the controlled experiments.
- `synthetic_surface_contours.png` and `synthetic_surface_metrics.csv`: compatible outputs from the original single-threshold workflow, refreshed by the complete CLI.

## Method and limitations

Grid coordinates are node locations. Each rectangle is split along a consistent diagonal into two triangles; threshold-clipped triangle areas are summed only where all three depth nodes are valid. The resulting piecewise-linear area respects x/y cell dimensions and the mapped coordinate bounds. It is an approximation at the supplied grid resolution, not a polygonized geologic boundary. Node coverage and valid-triangle mesh coverage are reported separately.

Smoothing can move or reduce highs. The raw-grid crest is retained for crest reporting; threshold connectivity and the digital grid-exit saddle use the normalized smoothed surface. Missing nodes are excluded from smoothing output and closure cells. Triangles touching any missing node are omitted from area integration, so substantial or clustered missingness can reduce the valid mesh and bias area. Edge-contact flags identify results limited by the model boundary.

The synthetic surface contains a seeded elliptical high, a regional trend, seeded noise and a seeded null patch. Sensitivity experiments are illustrative: the resolution comparison uses a zero-noise model to isolate sampling, while the missing-data experiments use a fixed seed and vary the null fraction. These calculations are for software demonstration and learning only; they are not a validated geological interpretation, volumetric estimate or real asset evaluation.

# Synthetic Reservoir Modelling & Contour Analysis

**Geoscience data analysis portfolio demonstration.** Every input, calculation, figure and reported result in this public demo uses the project’s fictional synthetic surface. Nothing represents a real reservoir, operator, field, prospect or project.

## What the workflow does

1. Generate or load a regular x/y/depth grid and report grid spacing, missing-node coverage and usable mesh coverage.
2. Validate coordinate completeness, uniqueness, regular spacing and finite depth values. Irregular or incomplete coordinate grids are rejected with an explanatory error; missing depth cells are allowed and remain masked.
3. Treat the maximum valid input value as the raw-grid crest. Larger depth values must represent shallower structure.
4. Apply normalized Gaussian smoothing without filling missing depth nodes. Identify the 8-connected threshold region on the smoothed surface that contains the raw-grid crest location.
5. Calculate an approximate closure area by clipping the smoothed surface over a fixed-diagonal, piecewise-linear triangular mesh at the selected depth threshold. Report area in m² and km², raw-crest relief, node counts, boundary contact and coverage.
6. Calculate a domain-dependent grid-exit saddle along the maximum-bottleneck 8-neighbour path from the smoothed-surface maximum to a model-domain edge.
7. Repeat the geometric measurements across selected depth thresholds and controlled smoothing, missing-data and resolution scenarios. Export CSVs, figures and a concise report.

The selected depth threshold is a scenario parameter, not necessarily a geologically interpreted spill point. The computed grid-exit saddle is a separate, domain-dependent digital result; it is not an independently interpreted geological spill point or hydrocarbon contact. Its path starts at the maximum of the normalized smoothed surface, which may differ from the raw-grid crest used for threshold connectivity and raw-crest relief. A region marked `boundary_limited` reaches the model edge and is not fully enclosed. Structural geometry alone does not establish hydrocarbon charge, seal integrity, reservoir quality, accumulation, reserves or commercial viability.

## Input requirements

The CSV must contain numeric `x_m`, `y_m` and `depth_m` columns. Coordinates must be in metres and form a complete, unique, axis-aligned rectangular grid with strictly increasing, evenly spaced x and y values. Missing coordinate pairs are unsupported. Blank or `NaN` depth cells are allowed and remain masked; infinite or nonnumeric values are rejected. User-supplied data are supported only when they meet these requirements.

**Input depth convention:** The analysis assumes that larger numerical values represent shallower structure. If your source data use positive-down depths, convert them to the supported convention before running the analysis. The tool does not automatically infer or correct the depth sign convention. Input depths do not have to be negative; the included synthetic dataset happens to use negative metre values.

## Installation

From a fresh clone, open a terminal and run these commands. Use Python 3.10 or newer; the install command uses the repository’s declared runtime, plotting, notebook and test dependencies:

```powershell
git clone https://github.com/ilham-123bit/Reservoir-Contours-Map-for-Hydrocarbon-Distribution-Area.git
cd Reservoir-Contours-Map-for-Hydrocarbon-Distribution-Area
python -m pip install -r requirements.txt
```

## Reproduce the complete analysis

Run these commands from the repository root. The first regenerates the deterministic 101 x 91 synthetic surface using seed `20261026`; it writes only the synthetic CSV. The second loads and validates that grid, calculates the selected closure and threshold sweep, runs the fixed-seed robustness scenarios, then writes the report, tables and figures to `examples/output/`:

```powershell
python src/generate_synthetic_data.py --output data/synthetic/synthetic_surface.csv --seed 20261026
python src/geoscience_toolkit.py --input data/synthetic/synthetic_surface.csv --output-dir examples/output --spill-depths-m -2290 -2310 -2330 -2350 -2370 -2390
```

The CLI retains `--spill-depth-m` and `--spill-depths-m` for compatibility; both specify selected depth thresholds, not grid-exit saddles. Larger threshold values are shallower under the required input convention. Omit `--spill-depths-m` to use a reproducible sensitivity sweep; omit `--spill-depth-m` to set the primary threshold to 60 m below the raw-grid crest. Use `--smoothing-sigma` to choose Gaussian sigma in grid cells, `--seed` for robustness scenarios, or `--skip-robustness` to omit the robustness experiment and its outputs for a faster run. Run `python src/geoscience_toolkit.py --help` for options.

The original single-threshold command remains available for compatibility:

```powershell
python src/analyze_grid.py --input data/synthetic/synthetic_surface.csv --spill-depth-m -2321.589585 --output-dir examples/output --contour-step-m 20
```

Run the automated tests with:

```powershell
python -m pytest -q
```

Open `notebooks/synthetic_contour_demo.ipynb` from the repository root in JupyterLab for the guided workflow: inspect the data and quality checks, view the surface, quantify a selected-threshold region, compare selected depth thresholds, inspect cross-sections and 3D geometry, run robustness scenarios, and review the generated summary and limitations.

## Generated outputs

- `metrics_summary.csv`: one-row dataset, quality, raw-grid crest, selected threshold, grid-exit saddle, relief and closure metrics.
- `spill_sensitivity.csv`: one row per requested selected threshold, including area, raw-crest relief, grid-exit saddle fields, connectivity and edge flags. Legacy column names are retained for compatibility.
- `robustness_scenarios.csv`: controlled smoothing, null-fraction and grid-resolution experiments.
- `analysis_report.md`: calculated summary, interpretation, assumptions, limitations and output references.
- `structural_contours.png`: annotated synthetic depth map with the raw-grid crest, selected threshold, crest-connected threshold-region boundary and grid-exit saddle.
- `surface_3d.png`: 3D view of the synthetic surface.
- `cross_sections.png`: E–W and N–S sections through the raw-grid crest node.
- `closure_area_sensitivity.png` and `structural_relief_sensitivity.png`: response to the selected depth threshold.
- `closure_scenarios.png`: crest-connected threshold-region comparison at selected thresholds.
- `robustness_comparison.png`: closure-area comparison across the controlled experiments.
- `synthetic_surface_contours.png` and `synthetic_surface_metrics.csv`: compatible outputs from the original single-threshold workflow, refreshed by the complete CLI.

## Method and limitations

Grid coordinates are node locations. Each rectangle is split along the same diagonal into two triangles. Threshold regions are identified from the normalized smoothed surface while retaining the original missing-data mask; the 8-connected component is selected using the raw-grid crest’s grid location. Area is integrated over a piecewise-linear triangular mesh constructed from the normalized smoothed surface. Each triangle is clipped at the selected depth threshold, and clipped areas are summed only for valid triangles associated with the crest-connected threshold region. Triangles containing any missing depth node are excluded. The resulting area is an approximation based on grid resolution, smoothing, threshold choice and missing-data coverage; it is not a polygonized geological boundary. Node coverage and valid-triangle mesh coverage are reported separately.

Smoothing can move or reduce highs. The raw-grid crest is the maximum valid value in the unsmoothed input grid; it is used to select the threshold component and calculate raw-crest relief. The smoothed-surface maximum is the maximum valid value after normalized smoothing and is the starting node for the grid-exit saddle calculation. These points need not coincide. Missing depth nodes remain missing after smoothing and are excluded from threshold regions and area triangles. Substantial or clustered missingness can reduce valid mesh coverage and bias the area approximation. Edge-contact flags identify regions limited by the model boundary; such regions should not be described as fully enclosed closures.

The synthetic surface contains a seeded elliptical high, a regional trend, seeded noise and a seeded missing-data patch. Sensitivity experiments are illustrative: the resolution comparison uses a zero-noise model to isolate sampling, while the missing-data experiments use a fixed seed and vary the null fraction. These calculations are for software demonstration and learning only. They are not a validated geological interpretation, volumetric estimate, reserve estimate or real asset evaluation.


# Synthetic contour analysis: methods and assumptions

This project is a self-contained synthetic geoscience demonstration. `src/synthetic_grid.py` generates its surface from an invented mathematical model; the supplied CSV is reproducible with seed `20261026`. No real reservoir files are loaded by the demo.

## Synthetic surface

The generated grid represents a smooth elliptical structural high over a planar regional trend, with seeded Gaussian noise and a seeded circular missing-data patch:

`z(x,y) = -2400 + 165 exp(-0.5(u² + v²)) - 0.012(x - xmin) + 0.006(y - ymin) + ε`.

The x and y coordinates are in metres. The included synthetic depth values happen to be negative metres; a larger numerical value is shallower and a smaller value is deeper. The maximum valid value in the unsmoothed input is the **raw-grid crest**. It is a grid-node maximum, not a sub-grid estimate.

## Input checks and data quality

`load_grid` requires numeric `x_m`, `y_m`, and `depth_m` columns; unique coordinate pairs; a complete rectangular set of coordinate pairs; at least two finite, strictly increasing, evenly spaced coordinates on each axis; at least one finite depth; and no infinite depth values. Irregular spacing and missing coordinate pairs are unsupported and rejected. Blank/NaN depths are allowed and reported; malformed text and infinities are rejected.

Coordinate values must be supplied in metres for the reported coordinate and area units to be correct. User-supplied datasets are supported only when they satisfy the grid and value requirements above.

> **Input depth convention:** The analysis assumes that larger numerical values represent shallower structure. If your source data use positive-down depths, convert them to the supported convention before running the analysis. The tool does not automatically infer or correct the depth sign convention. Input depths do not have to be negative; the included synthetic dataset happens to use negative metre values.

The report gives node coverage (`finite depth nodes / total nodes`) and mesh coverage (`area of triangles with three finite depth nodes / mapped x-y domain area`). Each grid rectangle is split along the same fixed diagonal. Triangles that include a missing depth node do not contribute to mesh coverage or closure area. This avoids silently filling gaps but can understate mapped area near missing data; mesh coverage makes that limitation visible.

## Smoothing, crest, and selected-threshold closure

A normalized Gaussian filter smooths the depth values using the weighted sum of valid nodes divided by the filtered validity weights. Missing nodes remain missing in the returned smoothed array and are excluded from threshold regions. This reduces the edge bias caused by replacing null values with a global mean, but smoothing still shifts contours and can attenuate narrow highs. The default sigma is 1 grid cell; the robustness resolution experiment converts a fixed 100 m smoothing length to cell units.

The closure analysis identifies the raw-grid crest cell, thresholds the normalized smoothed surface using `smoothed_depth >= selected_threshold`, and selects the 8-connected component containing the raw-grid crest location. The original missing-depth mask is retained, so missing nodes do not enter threshold regions. The selected depth threshold is an explicit scenario parameter, not a discovered geologic contact. Raw-crest relief is calculated as `raw-grid crest depth - selected depth threshold`; its sign follows the input depth convention. The connected component and any grid-edge contact are recorded in each result.

Area is integrated over a piecewise-linear triangular mesh constructed from the normalized smoothed surface. Each triangle is clipped at the selected depth threshold, and clipped areas are summed only for valid triangles associated with the crest-connected threshold region. Triangles containing any missing depth node are excluded. Output is in square metres and square kilometres. This method accounts for coordinate spacing and fractional boundary triangles. The resulting area is an approximation based on grid resolution, smoothing, threshold choice and missing-data coverage; it is not a polygonized geological boundary.

## Digital grid-exit saddle

The calculated **grid-exit saddle** is based on the normalized smoothed grid. Its maximum valid node, the **smoothed-surface maximum**, starts the path and may differ from the raw-grid crest used for reporting and threshold-component selection. A maximum-bottleneck path search explores 8-neighbour connections to any valid node on the outer grid edge. For each path, its bottleneck is the minimum smoothed depth along that path; the selected path maximizes this minimum. The grid-exit saddle depth is that bottleneck, and the reported saddle point is the path node attaining it. Tie-breaking is deterministic from array order.

This definition is reproducible for a finite, sampled grid and provides a geometric comparison. It is domain dependent: extending the grid or changing its trend can change the grid-exit saddle. It is not an independently interpreted geological spill point or hydrocarbon contact, and does not establish an accumulation. A selected-threshold region that touches the grid edge is flagged `boundary_limited`; its area should not be presented as a fully enclosed closure.

## Sensitivity and robustness experiments

The threshold-sensitivity table records selected depth thresholds, raw-crest relief, raw-grid crest metrics, grid-exit saddle metrics, area, connected-node count, edge contact, data coverage and region status. Its legacy column names, including `spill_depth_threshold_m` and `grid_spill_depth_m`, are retained for compatibility; the first is a selected threshold and the second is the computed grid-exit saddle depth. Higher threshold values are shallower under the required depth convention. A region can be absent if the threshold exceeds the smoothed-surface values, closed if bounded within the domain, or boundary limited if it touches the grid edge.

The robustness table changes one controlled input family at a time:

- Smoothing sigma: 0, 1 and 2 cells on the supplied synthetic grid.
- Missing data: 0%, 2% and 10% null nodes using the same seed and model.
- Grid resolution: 51 x 46, 101 x 91 and 201 x 181 nodes on the same zero-noise model; smoothing is held near a 100 m physical length.

The experiments report outcomes rather than claiming an optimal parameter choice. Missing data can reduce valid-triangle coverage, and resolution comparisons remain approximations of a continuous synthetic function.

## Outputs and execution

From the repository root, install `requirements.txt`, regenerate the input if desired, and run the full analysis:

```powershell
python -m pip install -r requirements.txt
python src/generate_synthetic_data.py --output data/synthetic/synthetic_surface.csv --seed 20261026
python src/geoscience_toolkit.py --input data/synthetic/synthetic_surface.csv --output-dir examples/output --spill-depths-m -2290 -2310 -2330 -2350 -2370 -2390
python -m pytest -q
```

With robustness enabled (the default), the CLI writes machine-readable `metrics_summary.csv`, `spill_sensitivity.csv`, and `robustness_scenarios.csv`; a concise `analysis_report.md`; annotated structural contours, a 3D surface, raw-grid-crest cross-sections, two selected-threshold response plots, a threshold-region comparison, a robustness comparison; and compatibility outputs from the original single-threshold workflow. The selected threshold contour and computed grid-exit saddle are shown as separate quantities. When `--skip-robustness` is used, the report omits robustness output references and stale robustness CSV/figure files from a prior run are removed. The notebook follows the same workflow interactively.

All values and visuals are synthetic. Structural closure is a geometric description of a selected-threshold region on the smoothed surface; it does not establish hydrocarbon charge, seal integrity, reservoir quality, accumulation, reserves or commercial viability. The grid-exit saddle is domain dependent and is not an independently interpreted geological spill point. This workflow is educational, not a validated geological interpretation or real reservoir evaluation.

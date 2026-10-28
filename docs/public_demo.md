# Synthetic contour analysis: methods and assumptions

This project is a self-contained synthetic geoscience demonstration. `src/synthetic_grid.py` generates its surface from an invented mathematical model; the supplied CSV is reproducible with seed `20261026`. No real reservoir files are loaded by the demo.

## Synthetic surface

The generated grid represents a smooth elliptical structural high over a planar regional trend, with seeded Gaussian noise and a seeded circular missing-data patch:

`z(x,y) = -2400 + 165 exp(-0.5(u² + v²)) - 0.012(x - xmin) + 0.006(y - ymin) + ε`.

The x and y coordinates are in metres. The synthetic depth values are negative metres; a larger numerical value is shallower and a smaller/more negative value is deeper. The maximum valid raw-grid depth is reported as the structural crest. This is the grid maximum, not a sub-grid estimate.

## Input checks and data quality

`load_grid` requires numeric `x_m`, `y_m`, and `depth_m` columns; unique coordinate pairs; a complete rectangular set of coordinate pairs; at least two finite, strictly increasing, evenly spaced coordinates on each axis; at least one finite depth; and no infinite depth values. Irregular spacing and missing coordinate pairs are unsupported and rejected. Blank/NaN depths are allowed and reported; malformed text and infinities are rejected.

The report gives node coverage (`finite depth nodes / total nodes`) and mesh coverage (`area of triangles with three finite nodes / mapped x-y domain area`). A rectangle is triangulated along a fixed diagonal. Triangles that include a missing node do not contribute area. This avoids silently filling gaps but can understate mapped area near missing data; mesh coverage makes that limitation visible.

## Smoothing, crest, and selected-threshold closure

A normalized Gaussian filter smooths the depth values using the weighted sum of valid nodes divided by the filtered validity weights. Missing nodes remain missing in the returned smoothed array and are excluded from threshold regions. This reduces the edge bias caused by replacing null values with a global mean, but smoothing still shifts contours and can attenuate narrow highs. The default sigma is 1 grid cell; the robustness resolution experiment converts a fixed 100 m smoothing length to cell units.

The closure analysis identifies the raw-grid crest cell, thresholds the smoothed surface using `smoothed_depth >= selected_threshold`, and selects the 8-connected component containing that crest cell. The selected threshold is an explicit scenario input, not a discovered geologic contact. Relief is `raw crest depth - selected threshold`; it is positive when the chosen threshold is deeper (more negative) than the crest. The connected component and any grid-edge contact are recorded in each result.

Area is integrated over a piecewise-linear triangular mesh. Each triangle’s linearly interpolated depth polygon is clipped at the selected threshold; clipped triangle areas are summed only when all triangle vertices have valid data and the triangle belongs to the crest-connected above-threshold region. Output is in square metres and square kilometres. This method accounts for coordinate spacing and fractional boundary triangles; it remains grid-resolution dependent and is not a geological polygon measurement.

## Digital grid-exit saddle

The reported grid-exit saddle is calculated from the normalized smoothed grid. Starting at its shallowest valid node, a maximum-bottleneck path search explores 8-neighbour connections to any valid node on the outer grid edge. For each path, its bottleneck is the minimum smoothed depth along that path; the selected path maximizes this minimum. The spill reference depth is that bottleneck, and the reported point is the path node attaining it. Tie-breaking is deterministic from array order.

This definition is reproducible for a finite, sampled grid and provides a useful topographic comparison. It is domain dependent: extending the synthetic grid or changing its trend can change the grid-exit saddle. It is not an interpretation of a real spill point. A selected threshold region that touches the grid edge is flagged `boundary_limited`; its area should not be presented as a fully enclosed closure.

## Sensitivity and robustness experiments

The threshold sensitivity table records selected thresholds, relief, crest and digital grid-saddle values, area, connected-node count, edge contact, data coverage and region status. Higher threshold values are shallower under this project’s depth convention. A region can be absent if the threshold exceeds the crest, closed if bounded within the domain, or boundary limited if it touches the grid edge.

The robustness table changes one controlled input family at a time:

- Smoothing sigma: 0, 1 and 2 cells on the supplied synthetic grid.
- Missing data: 0%, 2% and 10% null nodes using the same seed and model.
- Grid resolution: 51 x 46, 101 x 91 and 201 x 181 nodes on the same zero-noise model; smoothing is held near a 100 m physical length.

The experiments report outcomes rather than claiming an optimal parameter choice. Missingness can reduce valid triangle coverage, and resolution comparisons remain approximations of a continuous synthetic function.

## Outputs and execution

From the repository root, install `requirements.txt`, regenerate the input if desired, and run the full analysis:

```powershell
python -m pip install -r requirements.txt
python src/generate_synthetic_data.py --output data/synthetic/synthetic_surface.csv --seed 20261026
python src/geoscience_toolkit.py --input data/synthetic/synthetic_surface.csv --output-dir examples/output --spill-depths-m -2290 -2310 -2330 -2350 -2370 -2390
python -m pytest -q
```

The CLI writes machine-readable `metrics_summary.csv`, `spill_sensitivity.csv`, and `robustness_scenarios.csv`; a concise `analysis_report.md`; annotated structural contours, a 3D surface, crest cross-sections, two threshold-response plots, a threshold-boundary comparison, a robustness comparison; and compatibility outputs from the original single-threshold workflow. The notebook follows the same sequence interactively.

All values and visuals are synthetic. Structural closure is a geometric description of a thresholded surface and does not establish hydrocarbon charge, seal, reservoir quality, recoverable volumes or an accumulation. This workflow is educational and is not a validated geological interpretation or real reservoir evaluation.

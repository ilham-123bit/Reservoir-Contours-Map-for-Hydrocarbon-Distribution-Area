# Synthetic contour demonstration

The demonstration uses only a fictional surface created by `src/synthetic_grid.py`. It does not load real reservoir files.

## Mathematical model

A rectangular, evenly spaced coordinate grid is assigned a synthetic depth surface

`z(x,y) = -2400 + 165 exp(-0.5(u^2 + v^2)) - 0.012(x - xmin) + 0.006(y - ymin) + epsilon`,

where `u` and `v` are centered, dimensionless coordinates scaled to make an elliptical dome, and `epsilon` is seeded Gaussian noise. A small seeded circular region is set to missing to demonstrate null handling. The constants are invented solely for this example and do not come from field data or project analysis. The random seed makes generation repeatable.

## Files and execution

Run from the repository root. Follow installation and command examples in `README.md`. The generator writes a regular CSV grid; the analyzer accepts any compatible regular x/y/depth CSV, a selected spill threshold, smoothing, area cutoff, contour interval, and output directory.

The example is illustrative. Mean-filling missing cells before smoothing can influence contours near null areas, cell-count area is approximate, and connectivity and threshold choices affect closure boundaries. Outputs are not geological interpretations or estimates for any real asset.

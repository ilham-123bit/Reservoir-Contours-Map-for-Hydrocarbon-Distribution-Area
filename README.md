# Synthetic Reservoir Contouring Demonstration

This repository includes a **fictional, synthetic-data-only** contouring portfolio. The demonstration surface is generated from an independently defined mathematical model with a configurable random seed; its values, outputs, and maps do not represent a real reservoir, operator, field, or project.

## Workflow

1. Generate a regular rectangular grid as CSV with `x_m`, `y_m`, and `depth_m` columns.
2. Load and validate the grid, smooth valid depth values after filling null cells with the grid mean, and identify the connected spill-threshold region containing the shallowest synthetic crest.
3. Write a contour PNG and a one-row CSV containing the synthetic crest and closure metrics.
4. Explore the same workflow in `notebooks/synthetic_contour_demo.ipynb`.

The surface model and its assumptions are described in [docs/public_demo.md](docs/public_demo.md).

## Installation

Use Python 3.10 or newer and install dependencies from the repository root:

```powershell
python -m pip install -r requirements.txt
```

## Generate and analyze the example

```powershell
python src/generate_synthetic_data.py --output data/synthetic/synthetic_surface.csv --seed 20261026
python src/analyze_grid.py --input data/synthetic/synthetic_surface.csv --spill-depth-m -2321.589585 --output-dir examples/output --contour-step-m 20
```

The spill threshold is an example setting for the fictional generated surface. Change it to suit a different input grid. Generator options include `--nx`, `--ny`, `--noise-std-m`, and `--null-fraction`; analysis options include `--smoothing-sigma`, `--min-area-km2`, and `--contour-step-m`. Run either command with `--help` to see all arguments.

## Expected outputs

- `data/synthetic/synthetic_surface.csv`: generated synthetic grid.
- `examples/output/synthetic_surface_contours.png`: contour map for the chosen input and parameters.
- `examples/output/synthetic_surface_metrics.csv`: crest and closure summary.

The demonstration notebook writes equivalent synthetic-only outputs under `examples/output/` and contains saved results generated only from its synthetic dataset.

## Notebook and tests

Open `notebooks/synthetic_contour_demo.ipynb` in Jupyter and run from the repository root. To run the synthetic-only test suite:

```powershell
python -m pytest -q
```

## Limitations

Mean-filling null cells before smoothing may affect contours near missing regions. The closure area uses grid-cell counts and is an approximation; eight-neighbor connectivity and the chosen spill threshold affect the region. This workflow is an educational demonstration, not a validated geological interpretation or reserve estimate. Use only authorized data when adapting it to other work.

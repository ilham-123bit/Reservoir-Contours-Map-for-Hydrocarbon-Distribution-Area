"""Analyze a user-supplied regular-grid CSV and write a map and metrics table."""
from __future__ import annotations
import argparse
from pathlib import Path
from contour_analysis import analyze_grid, load_grid, write_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="CSV with x_m, y_m and depth_m columns that meets the documented grid requirements")
    parser.add_argument("--spill-depth-m", type=float, required=True,
                        help="selected depth threshold (scenario parameter; larger values are shallower)")
    parser.add_argument("--output-dir", type=Path, default=Path("examples/output"))
    parser.add_argument("--contour-step-m", type=float, default=20.0)
    parser.add_argument("--smoothing-sigma", type=float, default=1.0)
    parser.add_argument("--min-area-km2", type=float, default=0.0)
    args = parser.parse_args()
    xs, ys, grid = load_grid(args.input)
    result = analyze_grid(xs, ys, grid, spill_depth_m=args.spill_depth_m,
                          smoothing_sigma=args.smoothing_sigma, min_area_km2=args.min_area_km2)
    map_path, table_path = write_outputs(result, args.output_dir, stem=args.input.stem,
                                         contour_step_m=args.contour_step_m)
    print(f"Grid analysis complete: {args.input}")
    print(f"Map: {map_path}\nMetrics: {table_path}")


if __name__ == "__main__":
    main()

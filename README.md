# Zarr Explorer

**Version 0.3** — A browser-based viewer and comparator for Zarr and NetCDF files, built with Dash/Plotly.

## Features

### Explorer tab
- Browse variables in a tree view; search/filter by name
- Plot arrays as line, heatmap, histogram, or scatter; overlay a second variable
- Dimension slicing with lazy loading (only reads the requested slice from disk)
- CF convention decoding: `scale_factor`, `add_offset`, `_FillValue`
- Log scale and TAI-2000 → UTC time axis conversion
- Persistent per-variable plot settings (plot type, colorscale, axis limits)
- Table view with full-precision values (respects dtype: int → integer, float32 → 8 sig. digits, float64 → 15 sig. digits)
- Per-variable statistics: min, max, mean, std, NaN count
- Attribute viewer

### Compare tab
- Load two Zarr or NetCDF files and auto-match variables by name/path
- Manual match override via in-table dropdown
- Tolerance-aware comparison: `exact` (integers/flags), `abs:<value>` (scaled), `rel:<value>` (floats)
  - Auto-detected from variable metadata (EOPF logical dtype, scale_factor)
  - Editable per variable in the table
- Color-coded results: green (perfect), yellow (within tolerance), orange (outside tolerance)
- Per-variable counts: # perfect, # within tolerance, # outside tolerance
- Time dimension slicing: apply independent `start:stop:step` slices to file A and B before comparison (applied only on dimensions named `time`, `record`, `epoch`, etc.)
- Detail panel on click: overlay plot, difference plot, attribute diff, value table
- Warnings column: NaN mismatches, comparison errors, slice ignored (no time dimension found)
- Download HTML report

## Requirements

```
dash>=2.0
dash-bootstrap-components>=2.0
plotly
numpy
zarr>=3.0
netCDF4          # for NetCDF support
xarray           # for NetCDF support
```

## Installation

```bash
pip install dash dash-bootstrap-components plotly numpy zarr netCDF4 xarray
```

## Usage

```bash
python zarr_explorer.py                        # open empty, load files from UI
python zarr_explorer.py path/to/file.zarr      # open with a file pre-loaded
```

Then open your browser at [http://127.0.0.1:8050](http://127.0.0.1:8050).

## Tolerance syntax

| String | Meaning |
|--------|---------|
| `exact` | All values must be bit-identical |
| `abs:1e-3` | Absolute difference ≤ 1e-3 |
| `rel:1e-7` | Relative difference ≤ 1e-7 |

Auto-detection rules:
- Integer type (byte, short, int, long, …) without scale_factor → `exact`
- Any type with scale_factor → `abs:{scale_factor}`
- Floating-point without scale_factor → `rel:1e-7`
- Variables with `flag` in the name → `exact`

## Time slice syntax

Standard Python slice notation applied to the time/record dimension:

| Slice | Meaning |
|-------|---------|
| `0:800` | First 800 records |
| `100:` | From record 100 onwards |
| `::2` | Every other record (decimation ×2) |
| `0:800:2` | First 800 records, every other one |

The slice is applied only on dimensions named `time`, `record`, `epoch`, `tai`, `utc`, or similar. Variables without a matching dimension are compared as-is. A warning is shown if a slice was specified but no time dimension was found.

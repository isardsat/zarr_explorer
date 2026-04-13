# Zarr Explorer

**Version 0.5** — A browser-based viewer and comparator for Zarr and NetCDF files, built with Dash/Plotly. Also usable as a CLI tool.

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
- Time dimension slicing: apply independent `start:stop:step` slices to file A and B before comparison
- Detail panel on click: overlay plot, difference plot, attribute diff, value table
- Warnings column: NaN mismatches, comparison errors, slice ignored
- Export/import variable mapping as JSON — minimal format (path_a, path_b, status, tolerance, slices)
- Download HTML or CSV comparison report (includes unmatched variables from both files)
- Compare zarr `other_metadata` scalar fields as virtual variables (`group/.meta/other_metadata/key`)

### CLI

```bash
# Explore a file
python zarr_explorer.py explore path/to/file.zarr
python zarr_explorer.py explore path/to/file.zarr --var measurements/waveform          # full attrs + other_metadata
python zarr_explorer.py explore path/to/file.zarr --var measurements/waveform --values
python zarr_explorer.py explore path/to/file.zarr --var measurements/waveform --values --slice 0:10 --raw

# Compare two files using a mapping JSON (exported from the Compare tab)
python zarr_explorer.py compare file_a.zarr file_b.nc --mapping mapping.json
python zarr_explorer.py compare file_a.zarr file_b.nc --mapping mapping.json --html report.html
python zarr_explorer.py compare file_a.zarr file_b.nc --mapping mapping.json --csv report.csv
python zarr_explorer.py compare file_a.zarr file_b.nc --mapping mapping.json --html --csv

# Convert between zarr and NetCDF using a mapping JSON
python zarr_explorer.py convert file_a.zarr --target-sample file_b.nc --mapping mapping.json --output output.nc
python zarr_explorer.py convert file_b.nc --target-sample file_a.zarr --mapping mapping.json --output output.zarr
python zarr_explorer.py convert file_b.nc --target-sample file_a.zarr --mapping mapping.json --output output.zarr --zarr-format 2
```

CLI compare output includes unmatched variables from both files. Defaults to HTML if neither `--html` nor `--csv` is given.

### Convert

Convert variables between zarr and NetCDF formats. Requires a mapping JSON (exported from the Compare tab) and a **target sample** file — an existing file in the target format that provides the encoding template (dtype, `scale_factor`, `add_offset`, `_FillValue`, dimension names).

The conversion direction is detected automatically from the source and target-sample file formats.

**How it works:**
1. Source data is decoded (CF scale/offset applied, FillValue → NaN)
2. Re-encoded using the target sample's attributes (reverse scale/offset, NaN → FillValue, cast to target dtype)
3. Written to the output file with the target's structure and encoding

Only confirmed pairs in the mapping are converted. Unmatched or skipped variables produce a warning.

When converting to zarr, `--zarr-format 2` or `--zarr-format 3` (default) controls the output store version.

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

## Slice syntax

Standard Python slice notation:

| Slice | Meaning |
|-------|---------|
| `0:800` | First 800 elements |
| `100:` | From element 100 onwards |
| `::2` | Every other element |
| `1:7:2` | Elements 1, 3, 5 |

In the Compare tab, slices are applied to the time/record dimension of each file independently. Variables without a matching dimension are compared as-is.

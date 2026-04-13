# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Zarr Explorer — a single-file (`zarr_explorer.py`) Dash/Plotly browser-based viewer and comparator for Zarr and NetCDF files. Also usable as a CLI tool. MIT license, by isardSAT SL.

No package structure, no build system — just one Python file and its dependencies.

## Running

```bash
# Web UI
python zarr_explorer.py                        # open empty, load files from UI
python zarr_explorer.py path/to/file.zarr      # pre-load a file

# CLI
python zarr_explorer.py explore path/to/file.zarr
python zarr_explorer.py explore path/to/file.zarr --var measurements/waveform
python zarr_explorer.py explore path/to/file.zarr --var measurements/waveform --values --slice 0:10

python zarr_explorer.py compare file_a.zarr file_b.nc --mapping mapping.json
python zarr_explorer.py compare file_a.zarr file_b.nc --mapping mapping.json --html --csv
```

## Dependencies

```
dash>=2.0
dash-bootstrap-components>=2.0
plotly
numpy
zarr>=3.0
netCDF4          # NetCDF support
xarray           # NetCDF support
```

```bash
pip install dash dash-bootstrap-components plotly numpy zarr netCDF4 xarray
```

## Code Style

- Single-file — all logic lives in `zarr_explorer.py`
- Line length: ~120 characters (follow existing style)
- No formal formatter enforced — match the surrounding code
- Python 3.11+, uses `X | Y` union types

## Architecture

### Top-level sections (in file order)

| Section | Approx. lines | Purpose |
|---|---|---|
| TAI-2000 helpers | 37–68 | UTC time conversion for EOPF/satellite timestamps |
| Global stores | 70–105 | `dict[str, zarr.Group]` — open files; `open_zarr_file`, `close_zarr_file` |
| Tree builder | 112–192 | `build_tree()` — variable tree as Dash HTML, group collapse, search filter |
| Helpers | 196–315 | `collect_arrays`, `find_dim_coords`, `_parse_slice_text` |
| Data loading | 318–351 | `_load_data_sliced()` — lazy zarr reads, slice at zarr level |
| Figure generation | 354–630 | `_generate_figure()` — Line/Scatter/Bar/Histogram/Heatmap/Contour, CF decoding, B-var ops |
| Table builder | 633–772 | `_build_table()` — DataTable with slicing, CF decoding, x-var |
| Compare helpers | 775–1328 | File var inventory, auto-match, tolerance detection, `_compare_pair`, diff figures |
| Compare layout | 1331–1526 | `_build_compare_layout()` |
| App layout | 1529–2238 | `make_layout()` — Navbar, Explorer tab |
| Dash callbacks | 2249–4145 | All `@app.callback` functions |
| CLI | 4150–4283 | `_cli_explore`, `_cli_compare` |
| Entry point | 4285–4324 | argparse dispatch or web server |

### Key patterns

- **Shared `open-files` store** — one `dcc.Store` feeds both Explore and Compare tab file dropdowns
- **Virtual `other_metadata` paths** — EOPF group attrs exposed as `group/.meta/other_metadata/key` pseudo-variables for comparison
- **Lazy loading** — zarr reads only the requested slice; 5M-cell hard cap on heatmaps (`_HEATMAP_CELL_CAP`)
- **Auto-tolerance detection** — integer/flag vars → `exact`, scaled → `abs:{scale_factor}`, float → `rel:1e-7`
- **Per-variable preferences** — `var-prefs` store persists plot type, colorscale, axis settings per variable path

## Working Guidelines

- **Plan mode for implementation** — Enter plan mode for non-trivial tasks (3+ steps or architectural decisions). Write detailed specs upfront to reduce ambiguity. Use plan mode for verification steps, not just building. If something goes sideways, stop and re-plan immediately.
- **Subagents for parallelism** — Offload research, exploration, and parallel analysis to subagents. One task per subagent for focused execution.
- **Self-improvement** — After any correction, update `MEMORY.md` with the pattern. Review lessons at session start.
- **Verify before done** — Never mark a task complete without proving it works. Diff behaviour between main and your changes when relevant. Ask yourself: "Would a staff engineer approve this?" Run tests, check logs, demonstrate correctness.
- **Simplicity first** — Make every change as simple as possible. Find root causes, no temporary fixes. Senior developer standards. Changes should only touch what's necessary.
- **Demand elegance (balanced)** — For non-trivial changes, pause and ask "is there a more elegant way?" If a fix feels hacky: "Knowing everything I know now, implement the elegant solution." Challenge your own work before presenting it. Skip this for simple, obvious fixes.
- **Track progress** — Plan first with checkable items, check in before implementing, explain changes at each step.

## Glossary

| Term | Meaning |
|------|---------|
| Zarr | Chunked, compressed N-dimensional array storage format |
| NetCDF | Network Common Data Form — scientific array file format (.nc, .h5) |
| CF conventions | Climate and Forecast metadata conventions (`scale_factor`, `add_offset`, `_FillValue`, `_ARRAY_DIMENSIONS`) |
| TAI-2000 | International Atomic Time seconds since 2000-01-01 00:00:00 — used by EOPF/satellite products |
| EOPF | Earth Observation Processing Framework — defines `_eopf_attrs`, `other_metadata`, logical dtypes |
| `other_metadata` | EOPF group attribute holding scalar fields as `{"data": val, "dims": [], "attrs": {...}}` |
| logical dtype | EOPF dtype string (e.g. `byte`, `short`, `int`) stored in `_eopf_attrs` or `dtype` attr |
| `_ARRAY_DIMENSIONS` | zarr/CF attribute listing dimension names for an array |
| tolerance | Comparison mode: `exact`, `abs:<value>`, or `rel:<value>` |
| mapping JSON | Exported variable pairing from the Compare tab — minimal format with `path_a`, `path_b`, `status`, `tolerance`, `slices` |

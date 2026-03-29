# SPDX-License-Identifier: MIT
# Copyright (c) 2026 isardSAT SL
"""
Zarr Explorer — Dash/Plotly viewer and comparator for Zarr and NetCDF files.

Web UI:
    python zarr_explorer.py [path/to/file.zarr]

CLI:
    python zarr_explorer.py explore <path> [--var VAR_PATH] [--values] [--slice SLICE]
    python zarr_explorer.py compare <file_a> <file_b> --mapping mapping.json [--output report.html]
"""

__version__ = "0.4"

import datetime
import difflib
import os
import re
import sys
import threading
import warnings
import webbrowser  # noqa: F401

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import zarr
from dash import ALL, MATCH, Input, Output, State, ctx, dash_table, dcc, html

# ---------------------------------------------------------------------------
# TAI-2000 -> UTC conversion helpers
# ---------------------------------------------------------------------------

_TAI_UTC_TABLE = [
    (0.0, 32),
    ((datetime.datetime(2006, 1, 1) - datetime.datetime(2000, 1, 1)).total_seconds(), 33),
    ((datetime.datetime(2009, 1, 1) - datetime.datetime(2000, 1, 1)).total_seconds(), 34),
    ((datetime.datetime(2012, 7, 1) - datetime.datetime(2000, 1, 1)).total_seconds(), 35),
    ((datetime.datetime(2015, 7, 1) - datetime.datetime(2000, 1, 1)).total_seconds(), 36),
    ((datetime.datetime(2017, 1, 1) - datetime.datetime(2000, 1, 1)).total_seconds(), 37),
]

_TAI2000_TO_UNIX_OFFSET = (
    datetime.datetime(2000, 1, 1, tzinfo=datetime.timezone.utc)
    - datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
).total_seconds()


def tai2000_to_unix(tai: np.ndarray) -> np.ndarray:
    """Convert TAI seconds since 2000-01-01 00:00:00 to Unix UTC timestamps."""
    offsets = np.full_like(tai, float(_TAI_UTC_TABLE[0][1]), dtype=float)
    approx_utc = tai - offsets
    for threshold, off in _TAI_UTC_TABLE[1:]:
        offsets = np.where(approx_utc >= threshold, float(off), offsets)
    return tai - offsets + _TAI2000_TO_UNIX_OFFSET


def tai2000_to_utc_strings(tai: np.ndarray) -> list:
    """Convert TAI-2000 array to ISO UTC strings for Plotly datetime axis."""
    unix_ts = tai2000_to_unix(tai)
    return [
        datetime.datetime.utcfromtimestamp(float(t)).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        for t in unix_ts
    ]


# ---------------------------------------------------------------------------
# Global stores (single-user local tool)
# ---------------------------------------------------------------------------

stores: dict[str, zarr.Group] = {}


def get_store(file_id: str) -> zarr.Group | None:
    return stores.get(file_id)


def open_zarr_file(path: str) -> zarr.Group:
    """Open a zarr file and register it in the global stores dict."""
    s = zarr.open(path, mode="r")
    stores[path] = s
    return s


def close_zarr_file(path: str) -> None:
    stores.pop(path, None)


def _get_dir_size_mb(path: str) -> float:
    """Get approximate directory size in MB."""
    total = 0
    try:
        for dirpath, _dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
    except OSError:
        pass
    return total / (1024 * 1024)


# ---------------------------------------------------------------------------
# Tree builder
# ---------------------------------------------------------------------------

def _build_metadata_node(node_attrs: dict, depth: int, search_filter: str, group_path: str) -> list:
    """Build a virtual [metadata] collapsible node from EOPF other_metadata scalars."""
    other_metadata = node_attrs.get("other_metadata", {})
    if not isinstance(other_metadata, dict):
        return []
    sf = search_filter.lower().strip() if search_filter else ""
    meta_items = []
    for key, val in other_metadata.items():
        if not (isinstance(val, dict) and "data" in val and val.get("dims") == []):
            continue
        if sf and sf not in key.lower():
            continue
        data_val = val["data"]
        dtype = val.get("dtype", "")
        units = val.get("attrs", {}).get("units", "")
        suffix = "  [" + ", ".join(x for x in [dtype, units] if x) + "]" if (dtype or units) else ""
        meta_items.append(
            html.Div(
                html.Span(
                    f"{key}: {data_val}{suffix}",
                    style={"fontSize": "11px", "fontFamily": "monospace",
                           "color": "#6c757d", "padding": "2px 6px", "display": "block"},
                ),
                style={"paddingLeft": f"{(depth + 1) * 14}px", "borderBottom": "1px solid #dee2e6"},
            )
        )
    if not meta_items:
        return []
    meta_id = f"__meta__{group_path}"
    return [html.Div([
        dbc.Button(
            [
                html.Span("\u25BE ", id={"type": "group-arrow", "index": meta_id},
                          style={"fontSize": "10px", "display": "inline-block",
                                 "transition": "transform 0.15s"}),
                "[metadata]",
            ],
            id={"type": "group-toggle", "index": meta_id},
            color="link", size="sm", class_name="text-start w-100",
            style={
                "fontSize": "11px", "fontFamily": "monospace", "fontStyle": "italic",
                "color": "#6c757d",
                "padding": f"3px 6px 3px {depth * 14 + 6}px",
                "background": "#f8f9fa",
                "borderBottom": "1px solid #dee2e6",
                "borderRadius": "0",
            },
        ),
        dbc.Collapse(meta_items, id={"type": "group-collapse", "index": meta_id}, is_open=True),
    ])]


def build_tree(node: zarr.Group | zarr.Array, path: str = "", depth: int = 0,
               search_filter: str = "") -> list:
    """Build tree of variable buttons. If search_filter is set, only show matching arrays."""
    items = []
    sf = search_filter.lower().strip() if search_filter else ""

    if isinstance(node, zarr.Array):
        name = path.split("/")[-1]
        if sf and sf not in path.lower() and sf not in name.lower():
            return items
        items.append(
            html.Div(
                dbc.Button(
                    [html.Span("  ", style={"color": "#593196"}), f"{name}  {node.shape}  [{node.dtype}]"],
                    id={"type": "var-btn", "index": path},
                    color="link",
                    size="sm",
                    class_name="text-start w-100 text-dark",
                    style={"fontSize": "12px", "fontFamily": "monospace", "padding": "3px 6px"},
                ),
                style={"paddingLeft": f"{depth * 14}px", "borderBottom": "1px solid #dee2e6"},
            )
        )
        return items

    for name, child in node.members():
        child_path = f"{path}/{name}" if path else name
        if isinstance(child, zarr.Array):
            if sf and sf not in child_path.lower() and sf not in name.lower():
                continue
            items.append(
                html.Div(
                    dbc.Button(
                        [html.Span("  ", style={"color": "#593196"}), f"{name}  {child.shape}  [{child.dtype}]"],
                        id={"type": "var-btn", "index": child_path},
                        color="link",
                        size="sm",
                        class_name="text-start w-100 text-dark",
                        style={"fontSize": "12px", "fontFamily": "monospace", "padding": "3px 6px"},
                    ),
                    style={"paddingLeft": f"{depth * 14}px", "borderBottom": "1px solid #dee2e6"},
                )
            )
        else:
            # Recurse into group - if filtering, only show group if it has matching children
            child_items = build_tree(child, child_path, depth + 1, search_filter=sf)
            if sf and not child_items:
                continue
            items.append(html.Div([
                dbc.Button(
                    [
                        html.Span(
                            "\u25BE ",
                            id={"type": "group-arrow", "index": child_path},
                            style={"fontSize": "10px", "display": "inline-block",
                                   "transition": "transform 0.15s"},
                        ),
                        f"{name}/",
                    ],
                    id={"type": "group-toggle", "index": child_path},
                    color="link",
                    size="sm",
                    class_name="text-start w-100 fw-semibold",
                    style={
                        "fontSize": "12px",
                        "fontFamily": "monospace",
                        "color": "#593196",
                        "padding": f"4px 6px 3px {depth * 14 + 6}px",
                        "background": "#f4f0fa",
                        "borderBottom": "1px solid #d6c9f0",
                        "borderRadius": "0",
                    },
                ),
                dbc.Collapse(
                    child_items,
                    id={"type": "group-collapse", "index": child_path},
                    is_open=True,  # always start open (user can collapse manually)
                ),
            ]))

    return items


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_coord_range(store: zarr.Group, dim_name: str, dim_size: int) -> str | None:
    """Try to find a coordinate array matching dim_name and return a 'first..last' summary."""
    # Search for an array whose name matches the dimension name
    def _find_coord(node: zarr.Group, name: str, prefix: str = "") -> zarr.Array | None:
        for child_name, child in node.members():
            child_path = f"{prefix}/{child_name}" if prefix else child_name
            if isinstance(child, zarr.Array):
                if child_name == name and child.shape and child.shape[0] == dim_size:
                    return child
            elif isinstance(child, zarr.Group):
                found = _find_coord(child, name, child_path)
                if found is not None:
                    return found
        return None

    try:
        coord = _find_coord(store, dim_name)
        if coord is None:
            return None
        first = coord[0]
        last = coord[-1]
        # Format nicely
        if hasattr(first, 'item'):
            first = first.item()
        if hasattr(last, 'item'):
            last = last.item()
        # Truncate long values
        s_first = f"{first:.6g}" if isinstance(first, float) else str(first)
        s_last = f"{last:.6g}" if isinstance(last, float) else str(last)
        return f"{s_first} .. {s_last}"
    except Exception:
        return None


def collect_arrays(node: zarr.Group | zarr.Array, path: str = "") -> list[dict]:
    """Return list of {label, value} for all arrays in the store."""
    result = []
    if isinstance(node, zarr.Array):
        result.append({"label": f"{path}  {node.shape}", "value": path})
        return result
    for name, child in node.members():
        child_path = f"{path}/{name}" if path else name
        result.extend(collect_arrays(child, child_path))
    return result


def collect_groups(node: zarr.Group, path: str = "") -> list[str]:
    """Return list of group paths in the store."""
    result = []
    if isinstance(node, zarr.Array):
        return result
    if path:
        result.append(path)
    for name, child in node.members():
        child_path = f"{path}/{name}" if path else name
        if not isinstance(child, zarr.Array):
            result.append(child_path)
            result.extend(collect_groups(child, child_path))
    return result


def find_dim_coords(store: zarr.Group, var_path: str) -> dict[str, str]:
    """Return {dim_name: array_path} for dimension coordinates of a variable."""
    try:
        node = store[var_path]
    except KeyError:
        return {}
    if not isinstance(node, zarr.Array):
        return {}
    dim_names: list[str] = dict(node.attrs).get("_ARRAY_DIMENSIONS", [])
    if not dim_names:
        return {}
    all_arrays = collect_arrays(store)
    coords: dict[str, str] = {}
    for dim in dim_names:
        for arr in all_arrays:
            if arr["value"].split("/")[-1] == dim:
                coords[dim] = arr["value"]
                break
    return coords


def _parse_slice_text(text: str | None, ndim: int) -> tuple:
    """Parse numpy-style slice string into an index tuple.

    Examples:  '5, :, 3:10'  ->  (5, slice(None), slice(3,10))
               '::2, 0'      ->  (slice(None,None,2), 0)
    """
    if not text or not text.strip():
        return tuple(slice(None) for _ in range(ndim))
    tokens = [t.strip() for t in text.split(",")]
    while len(tokens) < ndim:
        tokens.append(":")
    index = []
    for token in tokens[:ndim]:
        if token in ("", ":"):
            index.append(slice(None))
        elif ":" in token:
            parts = token.split(":")

            def _int_or_none(s: str):
                s = s.strip()
                return int(s) if s else None

            start = _int_or_none(parts[0])
            stop = _int_or_none(parts[1]) if len(parts) > 1 else None
            step = _int_or_none(parts[2]) if len(parts) > 2 else None
            index.append(slice(start, stop, step))
        else:
            try:
                index.append(int(token))
            except ValueError:
                index.append(slice(None))
    return tuple(index)


# ---------------------------------------------------------------------------
# Data loading with lazy-read and size capping
# ---------------------------------------------------------------------------

_HEATMAP_CELL_WARN = 1_000_000  # warn if > 1M cells for heatmap
_HEATMAP_CELL_CAP = 5_000_000   # hard cap


def _load_data_sliced(node: zarr.Array, slice_text: str | None) -> tuple[np.ndarray, str]:
    """Load array data with slicing. Returns (data, warning_message).

    Applies slicing at zarr read level when possible to avoid loading full array.
    """
    warning = ""
    raw_shape = node.shape
    ndim = len(raw_shape)

    # Parse slice
    idx = _parse_slice_text(slice_text, ndim)

    # Try to read only the needed slice from zarr (lazy read)
    try:
        data = np.squeeze(node[idx])
    except Exception:
        # Fallback: read full, then slice in numpy
        data = np.squeeze(node[:])
        if slice_text and slice_text.strip():
            try:
                idx2 = _parse_slice_text(slice_text, data.ndim)
                data = np.squeeze(data[idx2])
            except Exception:
                pass

    return data, warning


# ---------------------------------------------------------------------------
# Figure generation helper
# ---------------------------------------------------------------------------

def _compute_stats(data: np.ndarray) -> dict:
    """Compute summary statistics on flattened numeric data."""
    try:
        flat = data.ravel().astype(float)
    except (ValueError, TypeError):
        return {
            "shape": str(data.shape), "dtype": str(data.dtype),
            "min": "--", "max": "--", "mean": "--", "std": "--", "nan": "--",
        }
    nan_count = int(np.isnan(flat).sum())
    valid = flat[~np.isnan(flat)]
    if valid.size == 0:
        return {
            "shape": str(data.shape), "dtype": str(data.dtype),
            "min": "--", "max": "--", "mean": "--", "std": "--", "nan": nan_count,
        }

    def _fmt(v):
        return f"{v:.6g}"

    return {
        "shape": str(data.shape),
        "dtype": str(data.dtype),
        "min": _fmt(valid.min()),
        "max": _fmt(valid.max()),
        "mean": _fmt(valid.mean()),
        "std": _fmt(valid.std()),
        "nan": nan_count,
        "_valid": valid,  # kept for histogram; not serialised
    }


def _generate_figure(
    var_path: str,
    x_var: str | None,
    b_var: str | None,
    op: str | None,
    plot_type: str,
    colorscale: str,
    cmin,
    cmax,
    xmin,
    xmax,
    ymin,
    ymax,
    log_scale: bool,
    apply_scale: bool,
    tai_to_utc: bool,
    slice_text: str | None,
    active_store: zarr.Group | None = None,
) -> tuple[go.Figure, dict, str] | tuple[None, None, str]:
    """Generate a Plotly figure and stats. Returns (fig, stats, warning) or (None, None, warning)."""
    s = active_store
    if not var_path or s is None:
        return None, None, ""
    try:
        node = s[var_path]
    except KeyError:
        return None, None, f"Variable not found: {var_path}"
    if not isinstance(node, zarr.Array):
        return None, None, "Selected item is a group, not an array."

    # Lazy-load with slicing
    data, load_warn = _load_data_sliced(node, slice_text)
    ndim = data.ndim
    _node_attrs = dict(node.attrs)
    _units = _node_attrs.get("units", "")
    _var_label = var_path.split("/")[-1] + (f"  [{_units}]" if _units else "")

    # Apply CF scale_factor / add_offset / _FillValue if requested
    if apply_scale:
        attrs = dict(node.attrs)
        fill_value = attrs.get("_FillValue", attrs.get("missing_value", None))
        scale_factor = attrs.get("scale_factor", 1.0)
        add_offset = attrs.get("add_offset", 0.0)
        data = data.astype(float)
        if fill_value is not None:
            data = np.where(data == float(fill_value), np.nan, data)
        data = data * float(scale_factor) + float(add_offset)

    # Apply array operation (A op B) if requested
    op_desc = ""
    overlay_data = None  # set when op == "Overlay"
    overlay_label = ""
    if b_var and op and s is not None:
        try:
            b_node = s[b_var]
            if isinstance(b_node, zarr.Array):
                b_data, _ = _load_data_sliced(b_node, slice_text)
                if op == "Overlay":
                    overlay_data = b_data.astype(float)
                    overlay_label = b_var.split("/")[-1]
                    op_desc = f"  vs  {overlay_label}"
                else:
                    a_flat = data.ravel().astype(float)
                    b_flat = b_data.ravel().astype(float)
                    min_len = min(len(a_flat), len(b_flat))
                    a_flat, b_flat = a_flat[:min_len], b_flat[:min_len]
                    if op == "A-B":
                        result = a_flat - b_flat
                    elif op == "B-A":
                        result = b_flat - a_flat
                    elif op == "(A+B)/2":
                        result = (a_flat + b_flat) / 2.0
                    elif op == "A*B":
                        result = a_flat * b_flat
                    elif op == "A/B":
                        with np.errstate(divide="ignore", invalid="ignore"):
                            result = np.where(b_flat != 0, a_flat / b_flat, np.nan)
                    else:
                        result = a_flat
                    data = result.reshape(a_flat.shape)
                    op_desc = f"  {op}  {b_var.split('/')[-1]}"
        except Exception:
            pass

    slice_desc = ""
    if slice_text and slice_text.strip() and slice_text.replace(",", "").replace(":", "").replace(" ", ""):
        slice_desc = f"  [{slice_text}]"
    title = var_path + slice_desc + op_desc

    fig = go.Figure()

    # Determine effective type: Auto resolves based on ndim
    eff_type = plot_type or "Auto"
    if eff_type == "Auto":
        eff_type = "Line" if ndim <= 1 else "Heatmap"

    warning = load_warn

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if ndim == 0:
            fig.add_annotation(
                text=str(float(data)), x=0.5, y=0.5,
                xref="paper", yref="paper", showarrow=False,
                font=dict(size=22),
            )

        elif eff_type == "Histogram":
            flat = data.ravel().astype(float)
            flat = flat[~np.isnan(flat)]
            fig.add_trace(go.Histogram(x=flat.tolist(), nbinsx=80,
                                       marker_color="#593196", opacity=0.85))
            fig.update_layout(xaxis_title=_var_label, yaxis_title="count")

        elif eff_type in ("Line", "Scatter", "Bar"):
            y = data.ravel().astype(float)
            if log_scale:
                y = np.log10(np.where(y > 0, y, np.nan))
            yaxis = dict(title=_var_label)
            if cmin is not None or cmax is not None:
                yaxis["range"] = [cmin, cmax]
            x_data = None
            x_label = "index"
            if x_var and s is not None:
                try:
                    x_node = s[x_var]
                    if isinstance(x_node, zarr.Array):
                        x_raw = x_node[:].ravel().astype(float)
                        x_label = x_var.split("/")[-1]
                        min_len = min(len(x_raw), len(y))
                        x_raw = x_raw[:min_len]
                        y = y[:min_len]
                        if tai_to_utc:
                            x_data = tai2000_to_utc_strings(x_raw)
                            x_label += " (UTC)"
                        else:
                            x_data = x_raw.tolist()
                except Exception:
                    x_data = None
            a_name = var_path.split("/")[-1]
            if eff_type == "Line":
                fig.add_trace(go.Scatter(x=x_data, y=y.tolist(), mode="lines", line=dict(width=1.5),
                                         name=a_name))
            elif eff_type == "Scatter":
                fig.add_trace(go.Scatter(x=x_data, y=y.tolist(), mode="markers", marker=dict(size=3),
                                         name=a_name))
            else:
                fig.add_trace(go.Bar(x=x_data, y=y.tolist(), name=a_name))
            # Overlay B variable as second trace
            if overlay_data is not None:
                b_y = overlay_data.ravel().astype(float)
                if log_scale:
                    b_y = np.log10(np.where(b_y > 0, b_y, np.nan))
                min_len = min(len(y), len(b_y))
                b_y = b_y[:min_len]
                b_x = x_data[:min_len] if x_data is not None and len(x_data) >= min_len else None
                if eff_type == "Line":
                    fig.add_trace(go.Scatter(x=b_x, y=b_y.tolist(), mode="lines",
                                             line=dict(width=1.5, dash="dash"), name=overlay_label))
                elif eff_type == "Scatter":
                    fig.add_trace(go.Scatter(x=b_x, y=b_y.tolist(), mode="markers",
                                             marker=dict(size=3), name=overlay_label))
                else:
                    fig.add_trace(go.Bar(x=b_x, y=b_y.tolist(), name=overlay_label, opacity=0.6))
                fig.update_layout(showlegend=True)
            fig.update_layout(xaxis_title=x_label, yaxis=yaxis)

        else:  # Heatmap or Contour
            def _prepare_heatmap(arr):
                if arr.ndim == 1:
                    return arr.reshape(1, -1).astype(float)
                elif arr.ndim == 2:
                    return arr.astype(float)
                return arr.reshape(-1, arr.shape[-1]).astype(float)

            plot_data = _prepare_heatmap(data)

            total_cells = plot_data.shape[0] * plot_data.shape[1]
            if total_cells > _HEATMAP_CELL_CAP:
                max_rows = _HEATMAP_CELL_CAP // plot_data.shape[1]
                step = max(1, plot_data.shape[0] // max_rows)
                plot_data = plot_data[::step, :]
                warning = (f"Downsampled from {total_cells:,} to "
                           f"{plot_data.shape[0] * plot_data.shape[1]:,} cells for display.")
            elif total_cells > _HEATMAP_CELL_WARN:
                warning = f"Large heatmap: {total_cells:,} cells. Consider slicing for better performance."

            if log_scale:
                plot_data = np.log10(np.where(plot_data > 0, plot_data, np.nan))
            cs = colorscale or "Viridis"
            yr = "record" if ndim == 2 else f"record (flattened from {data.shape})"

            # 2D overlay: side-by-side subplots
            if overlay_data is not None and overlay_data.ndim >= 1:
                a_name = var_path.split("/")[-1]
                b_plot = _prepare_heatmap(overlay_data)
                if log_scale:
                    b_plot = np.log10(np.where(b_plot > 0, b_plot, np.nan))
                fig = make_subplots(rows=1, cols=2, subplot_titles=[a_name, overlay_label],
                                    shared_yaxes=True, horizontal_spacing=0.05)
                trace_cls = go.Contour if eff_type == "Contour" else go.Heatmap
                fig.add_trace(trace_cls(z=plot_data.tolist(), colorscale=cs, showscale=False,
                                        zmin=cmin, zmax=cmax), row=1, col=1)
                fig.add_trace(trace_cls(z=b_plot.tolist(), colorscale=cs, showscale=True,
                                        zmin=cmin, zmax=cmax), row=1, col=2)
                fig.update_xaxes(title_text="sample", row=1, col=1)
                fig.update_xaxes(title_text="sample", row=1, col=2)
                fig.update_yaxes(title_text=yr, row=1, col=1)
            else:
                if eff_type == "Contour":
                    fig.add_trace(go.Contour(
                        z=plot_data.tolist(), colorscale=cs, showscale=True,
                        zmin=cmin, zmax=cmax,
                    ))
                else:
                    fig.add_trace(go.Heatmap(
                        z=plot_data.tolist(), colorscale=cs, showscale=True,
                        zmin=cmin, zmax=cmax,
                    ))
                fig.update_layout(
                    xaxis_title="sample", yaxis_title=yr,
                    coloraxis_colorbar_title_text=_var_label if _units else "",
                )

    axis_updates: dict = {}
    if xmin is not None or xmax is not None:
        axis_updates["xaxis_range"] = [xmin, xmax]
    if ymin is not None or ymax is not None:
        axis_updates["yaxis_range"] = [ymin, ymax]

    fig.update_layout(
        title=dict(text=title, font=dict(size=12)),
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=60, r=20, t=45, b=50),
        **axis_updates,
    )
    stats = _compute_stats(data)
    return fig, stats, warning


_TABLE_ROW_LIMIT = 2000


def _build_table(
    var_path: str, active_store: zarr.Group | None, slice_text: str | None = None,
    x_var: str | None = None, tai_to_utc: bool = False,
    apply_scale: bool = True, log_scale: bool = False,
) -> html.Div | dash_table.DataTable:
    """Build a DataTable for the given array variable (with slicing support)."""
    if not var_path or active_store is None:
        return html.Div("No data.", className="text-muted p-4 text-center")
    try:
        node = active_store[var_path]
    except KeyError:
        return html.Div(f"Variable not found: {var_path}", className="text-muted p-4 text-center")
    if not isinstance(node, zarr.Array):
        return html.Div("Not an array.", className="text-muted p-4 text-center")

    # Load with slicing (lazy)
    data, _ = _load_data_sliced(node, slice_text)

    # Apply CF scale/offset if requested
    if apply_scale:
        attrs = dict(node.attrs)
        scale = attrs.get("scale_factor", 1.0)
        offset = attrs.get("add_offset", 0.0)
        fill = attrs.get("_FillValue", None)
        if scale != 1.0 or offset != 0.0 or fill is not None:
            data = data.astype(float)
            if fill is not None:
                data[data == fill] = float("nan")
            data = data * scale + offset

    # Apply log10
    if log_scale:
        data = np.log10(np.abs(data) + 1e-30)
    truncated = False

    # Resolve x-axis variable (replaces index column when set)
    x_col_values: list | None = None
    x_col_name = "index"
    if x_var and active_store is not None:
        try:
            x_node = active_store[x_var]
            if isinstance(x_node, zarr.Array):
                x_raw = x_node[:].ravel().astype(float)
                x_col_name = x_var.split("/")[-1]
                if tai_to_utc:
                    x_col_values = tai2000_to_utc_strings(x_raw)
                    x_col_name += " (UTC)"
                else:
                    x_col_values = x_raw.tolist()
        except Exception:
            x_col_values = None

    # Get dimension names for column headers
    dim_names = dict(node.attrs).get("_ARRAY_DIMENSIONS", [])

    if data.ndim == 0:
        df_rows = [{"value": str(float(data))}]
        columns = [{"name": "value", "id": "value"}]
    elif data.ndim == 1:
        arr = data.tolist()
        if len(arr) > _TABLE_ROW_LIMIT:
            arr = arr[:_TABLE_ROW_LIMIT]
            truncated = True
        var_name = var_path.split("/")[-1]
        if x_col_values is not None:
            x_vals = x_col_values[:len(arr)]
            df_rows = [{x_col_name: x_vals[i] if i < len(x_vals) else i, var_name: v} for i, v in enumerate(arr)]
            columns = [{"name": x_col_name, "id": x_col_name}, {"name": var_name, "id": var_name}]
        else:
            df_rows = [{"index": i, var_name: v} for i, v in enumerate(arr)]
            columns = [{"name": "index", "id": "index"}, {"name": var_name, "id": var_name}]
    else:
        if data.ndim > 2:
            data_2d = data.reshape(-1, data.shape[-1])
        else:
            data_2d = data
        if data_2d.shape[0] > _TABLE_ROW_LIMIT:
            data_2d = data_2d[:_TABLE_ROW_LIMIT]
            truncated = True

        # Use dimension names for last dim if available
        last_dim_name = dim_names[-1] if len(dim_names) == len(node.shape) else None
        col_names = [
            f"{last_dim_name}_{j}" if last_dim_name else f"col_{j}"
            for j in range(data_2d.shape[1])
        ]
        # First column: x variable or row index
        if x_col_values is not None:
            first_col_id = "_xvar"
            columns = [{"name": x_col_name, "id": first_col_id}] + [{"name": c, "id": c} for c in col_names]
        else:
            first_col_id = "_row"
            columns = [{"name": "row", "id": first_col_id}] + [{"name": c, "id": c} for c in col_names]
        df_rows = []
        for i, row in enumerate(data_2d.tolist()):
            if x_col_values is not None:
                row_dict: dict = {first_col_id: x_col_values[i] if i < len(x_col_values) else i}
            else:
                row_dict = {first_col_id: i}
            for c, v in zip(col_names, row):
                row_dict[c] = v
            df_rows.append(row_dict)

    total_rows = data.shape[0] if data.ndim >= 1 else 1
    note = (
        html.Div(
            f"Showing first {_TABLE_ROW_LIMIT} rows of {total_rows}.",
            className="text-warning small px-2 pt-1",
        )
        if truncated
        else None
    )

    table = dash_table.DataTable(
        data=df_rows,
        columns=columns,
        page_action="none",
        sort_action="native",
        filter_action="native",
        fixed_rows={"headers": True},
        style_table={
            "overflowX": "auto",
            "overflowY": "auto",
            "height": "calc(100vh - 530px)",
            "minHeight": "200px",
        },
        style_cell={"fontSize": "11px", "fontFamily": "monospace", "padding": "3px 8px"},
        style_header={
            "fontWeight": "bold",
            "background": "#f4f0fa",
            "fontSize": "11px",
            "cursor": "pointer",
        },
        style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#fafafa"}],
    )

    return html.Div([c for c in [note, table] if c is not None])


# ---------------------------------------------------------------------------
# Comparison tool helpers
# ---------------------------------------------------------------------------

_CMP_SIZE_CAP = 5_000_000  # max elements to load per variable for comparison plots


def _detect_format(path: str) -> str:
    """Return 'zarr' if path is a directory, 'netcdf' for file extensions."""
    if os.path.isdir(path):
        return "zarr"
    ext = os.path.splitext(path)[1].lower()
    if ext in (".nc", ".h5", ".he5", ".hdf5", ".hdf", ".nc4"):
        return "netcdf"
    return "zarr"


def _get_file_vars(path: str) -> list[dict]:
    """Return variable inventory from a Zarr or NetCDF file."""
    fmt = _detect_format(path)
    if fmt == "zarr":
        s = zarr.open(path, mode="r")
        result = []
        for arr_info in collect_arrays(s):
            node = s[arr_info["value"]]
            attrs = dict(node.attrs)
            eopf = attrs.get("_eopf_attrs", {}) if isinstance(attrs.get("_eopf_attrs"), dict) else {}
            result.append({
                "path": arr_info["value"],
                "name": arr_info["value"].split("/")[-1],
                "shape": str(node.shape),
                "dtype": str(node.dtype),
                "logical_dtype": attrs.get("dtype", eopf.get("dtype", "")),
                "units": attrs.get("units", ""),
                "long_name": attrs.get("long_name", attrs.get("standard_name", "")),
                "scale_factor": attrs.get("scale_factor", eopf.get("scale_factor", None)),
            })
        # Also collect other_metadata scalar fields from all groups
        def _collect_meta_scalars(node: zarr.Group, grp_path: str = "") -> None:
            if not isinstance(node, zarr.Group):
                return
            other_meta = dict(node.attrs).get("other_metadata", {})
            if isinstance(other_meta, dict):
                for key, val in other_meta.items():
                    if isinstance(val, dict) and "data" in val and val.get("dims") == []:
                        meta_path = f"{grp_path}/.meta/other_metadata/{key}" if grp_path else f".meta/other_metadata/{key}"
                        meta_attrs = val.get("attrs", {})
                        result.append({
                            "path": meta_path,
                            "name": key,
                            "shape": "()",
                            "dtype": val.get("dtype", ""),
                            "logical_dtype": "",
                            "units": meta_attrs.get("units", ""),
                            "long_name": meta_attrs.get("long_name", ""),
                            "scale_factor": None,
                        })
            for name, child in node.members():
                child_path = f"{grp_path}/{name}" if grp_path else name
                if isinstance(child, zarr.Group):
                    _collect_meta_scalars(child, child_path)
        _collect_meta_scalars(s)
        return result
    else:
        import netCDF4 as _nc4
        result = []

        def _collect_nc_vars(grp, prefix=""):
            for name, var in grp.variables.items():
                path = f"{prefix}/{name}" if prefix else name
                attrs = {k: var.getncattr(k) for k in var.ncattrs()}
                result.append({
                    "path": path,
                    "name": name,
                    "shape": str(tuple(var.shape)),
                    "dtype": str(var.dtype),
                    "units": attrs.get("units", ""),
                    "long_name": attrs.get("long_name", attrs.get("standard_name", "")),
                    "scale_factor": attrs.get("scale_factor", None),
                })
            for grp_name, child in grp.groups.items():
                child_prefix = f"{prefix}/{grp_name}" if prefix else grp_name
                _collect_nc_vars(child, child_prefix)

        ds = _nc4.Dataset(path)
        _collect_nc_vars(ds)
        ds.close()
        return result


_EOPF_INT_DTYPES = {"byte", "short", "int", "long", "ubyte", "ushort", "uint", "ulong",
                    "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"}


def _is_integer_dtype(var: dict) -> bool:
    """Return True if the variable has an integer logical type, checking attrs dtype first."""
    # Prefer attrs logical dtype (EOPF stores this as e.g. 'byte', 'short')
    logical = var.get("logical_dtype", "") or var.get("dtype", "")
    if logical.lower() in _EOPF_INT_DTYPES:
        return True
    try:
        return bool(np.issubdtype(np.dtype(logical), np.integer))
    except Exception:
        return "int" in logical.lower()


def _auto_detect_tolerance(var: dict) -> str:
    """Return a default tolerance string for a variable dict from _get_file_vars."""
    name = var.get("name", "").lower()
    scale = var.get("scale_factor", None)
    # Flag or integer variables → exact match
    if "flag" in name or (_is_integer_dtype(var) and scale is None):
        return "exact"
    # Scaled variable → absolute tolerance = 1× scale_factor
    if scale is not None:
        try:
            return f"abs:{abs(float(scale)):.6g}"
        except (TypeError, ValueError):
            pass
    # Default floating-point relative tolerance
    return "rel:1e-7"


def _auto_match(vars_a: list[dict], vars_b: list[dict]) -> list[dict]:
    """Auto-match variables from file A to file B using name + shape scoring."""
    def _chain_segs(path: str) -> frozenset[str]:
        """Return short alphanumeric path segments that look like chain IDs (ka, ku1, ku2 …)."""
        parts = path.lower().split("/")[:-1]  # exclude variable name
        return frozenset(p for p in parts if p and len(p) <= 4 and p.isalnum())

    def _score(a: dict, b: dict) -> float:
        if a["path"] == b["path"]:
            return 1.0  # identical full path (same file or mirrored layout)
        name_a = a["name"].lower()
        name_b = b["name"].lower()
        path_b = b["path"].lower()
        if name_a == name_b:
            score = 0.95
        elif name_a in path_b.split("/"):
            score = 0.90
        else:
            ratio = difflib.SequenceMatcher(None, name_a, name_b).ratio()
            shape_bonus = 0.05 if a["shape"] == b["shape"] else 0.0
            score = min(ratio + shape_bonus, 0.84)  # cap below high-confidence threshold
        # Chain-aware adjustment: boost same-chain, block cross-chain
        chains_a = _chain_segs(a["path"])
        chains_b = _chain_segs(b["path"])
        if chains_a and chains_b:
            if chains_a & chains_b:  # share at least one chain segment (e.g. both ku1)
                score = min(score + 0.04, 1.0)
            else:  # different chain IDs (e.g. ku1 vs ka) → prevent match
                score = 0.0
        return score

    # Compute best match for each A var
    candidates = []
    for a in vars_a:
        best_score, best_b = 0.0, None
        for b in vars_b:
            s = _score(a, b)
            if s > best_score:
                best_score, best_b = s, b
        candidates.append((a, best_b, best_score))

    # Greedy assignment (highest score first)
    candidates.sort(key=lambda x: x[2], reverse=True)
    used_b: set[str] = set()
    assigned: dict[str, dict] = {}
    for a, best_b, best_score in candidates:
        if best_b is not None and best_b["path"] not in used_b and best_score > 0.3:
            used_b.add(best_b["path"])
            confidence = "high" if best_score >= 0.9 else ("medium" if best_score >= 0.6 else "low")
            assigned[a["path"]] = {
                "path_a": a["path"], "name_a": a["name"], "shape_a": a["shape"],
                "path_b": best_b["path"], "name_b": best_b["name"], "shape_b": best_b["shape"],
                "confidence": confidence, "status": "pending",
                "tolerance": _auto_detect_tolerance(a),
            }
        else:
            assigned[a["path"]] = {
                "path_a": a["path"], "name_a": a["name"], "shape_a": a["shape"],
                "path_b": "", "name_b": "", "shape_b": "",
                "confidence": "none", "status": "pending",
                "tolerance": _auto_detect_tolerance(a),
            }

    rows = [assigned[a["path"]] for a in vars_a]

    # Append unmatched B vars
    for b in vars_b:
        if b["path"] not in used_b:
            rows.append({
                "path_a": "", "name_a": "", "shape_a": "",
                "path_b": b["path"], "name_b": b["name"], "shape_b": b["shape"],
                "confidence": "none", "status": "pending",
            })

    return rows


_TIME_DIM_PATTERNS = re.compile(
    r"^(time|times|epoch|record|records|n_record|n_records|timestamp|tai|utc)s?$", re.IGNORECASE
)


def _parse_slice(s: str) -> slice | None:
    """Parse a slice string like '0:800', '::2', '100:' into a slice object. Returns None if empty."""
    if not s or not s.strip():
        return None
    parts = s.strip().split(":")
    if len(parts) < 2 or len(parts) > 3:
        return None
    def _int_or_none(v: str) -> int | None:
        v = v.strip()
        return int(v) if v else None
    try:
        if len(parts) == 2:
            return slice(_int_or_none(parts[0]), _int_or_none(parts[1]))
        return slice(_int_or_none(parts[0]), _int_or_none(parts[1]), _int_or_none(parts[2]))
    except ValueError:
        return None


def _detect_time_axis(node: zarr.Array, attrs: dict) -> int | None:
    """Return the axis index of the time/record dimension, or None if not found."""
    dim_names: list[str] | None = None
    # zarr v3 native dimension names
    if hasattr(node, "metadata") and hasattr(node.metadata, "dimension_names"):
        dim_names = node.metadata.dimension_names
    elif hasattr(node, "dims"):
        dim_names = list(node.dims)
    # Fallback: _ARRAY_DIMENSIONS attribute (zarr/CF convention)
    if not dim_names:
        dim_names = attrs.get("_ARRAY_DIMENSIONS") or []
    if dim_names:
        for i, name in enumerate(dim_names):
            if name and _TIME_DIM_PATTERNS.match(str(name)):
                return i
    return None


def _apply_time_slice(data: np.ndarray, axis: int | None, sl: slice | None) -> np.ndarray:
    """Apply sl along axis of data. No-op if axis or sl is None."""
    if axis is None or sl is None:
        return data
    return np.take(data, range(*sl.indices(data.shape[axis])), axis=axis)


def _load_var_data(file_path: str, var_path: str, apply_scale: bool = True,
                   time_slice: slice | None = None) -> tuple[np.ndarray, dict]:
    """Load a variable as a numpy array. CF decoding applied when apply_scale=True.
    time_slice is applied on the detected time axis before returning. Returns (data, attrs)."""
    fmt = _detect_format(file_path)
    # Handle virtual other_metadata scalar paths: "group/.meta/other_metadata/key"
    if "/.meta/" in var_path or var_path.startswith(".meta/"):
        if fmt != "zarr":
            raise ValueError("other_metadata scalars are only supported in Zarr files")
        if var_path.startswith(".meta/"):
            rest = var_path[len(".meta/"):]
            grp_path = ""
        else:
            grp_path, rest = var_path.split("/.meta/", 1)
        # rest is "other_metadata/key" or legacy "key"
        rest_parts = rest.split("/", 1)
        meta_source, key = (rest_parts[0], rest_parts[1]) if len(rest_parts) == 2 else ("other_metadata", rest_parts[0])
        s = zarr.open(file_path, mode="r")
        grp = s[grp_path] if grp_path else s
        other_meta = dict(grp.attrs).get(meta_source, {})
        entry = other_meta.get(key, {})
        if not isinstance(entry, dict) or "data" not in entry:
            raise KeyError(f"other_metadata key not found: {key}")
        data = np.array([float(entry["data"])])
        meta_attrs = entry.get("attrs", {})
        meta_attrs["_detected_time_axis"] = None
        return data, meta_attrs
    if fmt == "zarr":
        s = zarr.open(file_path, mode="r")
        node = s[var_path]
        attrs = dict(node.attrs)
        data = node[:].astype(float)
        if apply_scale:
            fill_value = attrs.get("_FillValue", attrs.get("missing_value", None))
            if fill_value is not None:
                data = np.where(data == float(fill_value), np.nan, data)
            data = data * float(attrs.get("scale_factor", 1.0)) + float(attrs.get("add_offset", 0.0))
        axis = _detect_time_axis(node, attrs)
        attrs["_detected_time_axis"] = axis  # propagate for caller
        data = _apply_time_slice(data, axis, time_slice)
    else:
        import xarray as xr
        # var_path may include a group prefix, e.g. "ku1/cal1_power_waveform"
        parts = var_path.split("/")
        if len(parts) > 1:
            group = "/".join(parts[:-1])
            var_name = parts[-1]
            ds = xr.open_dataset(file_path, engine="netcdf4", group=group,
                                  mask_and_scale=bool(apply_scale))
        else:
            var_name = var_path
            ds = xr.open_dataset(file_path, engine="netcdf4",
                                  mask_and_scale=bool(apply_scale))
        var = ds[var_name]
        # detect time dim by name from xarray dimension names
        axis: int | None = None
        for i, dim in enumerate(var.dims):
            if _TIME_DIM_PATTERNS.match(str(dim)):
                axis = i
                break
        raw = var.values
        if np.issubdtype(raw.dtype, np.datetime64):
            # xarray decoded CF time to datetime64 (ns since 1970-01-01)
            # convert to seconds since 2000-01-01 to match EOPF/TAI convention
            _TAI2000_NS = np.datetime64("2000-01-01T00:00:00", "ns").astype(np.int64)
            data = (raw.astype(np.int64) - _TAI2000_NS) / 1e9
        else:
            data = raw.astype(float)
        attrs = dict(var.attrs)
        attrs["_detected_time_axis"] = axis  # propagate for caller
        ds.close()
        data = _apply_time_slice(data, axis, time_slice)
    return data, attrs


def _build_warnings(res: dict) -> str:
    """Build a human-readable warnings string from a comparison result dict."""
    parts = []
    if res.get("error"):
        parts.append(f"Error: {res['error']}")
    nan_a = res.get("nan_a")
    nan_b = res.get("nan_b")
    if nan_a is not None and nan_b is not None:
        if nan_a > 0 and nan_b == 0:
            parts.append(f"NaN in A only ({nan_a} pts)")
        elif nan_b > 0 and nan_a == 0:
            parts.append(f"NaN in B only ({nan_b} pts)")
        elif nan_a > 0 and nan_b > 0 and nan_a != nan_b:
            parts.append(f"NaN count mismatch (A={nan_a}, B={nan_b})")
    if not res.get("error") and res.get("shape_match") and res.get("rmse") is None:
        parts.append("No valid points to compare (all NaN)")
    if res.get("slice_ignored_a") and res.get("slice_ignored_b"):
        parts.append("Time slice ignored (no time dimension found in A or B)")
    elif res.get("slice_ignored_a"):
        parts.append("A time slice ignored (no time dimension found)")
    elif res.get("slice_ignored_b"):
        parts.append("B time slice ignored (no time dimension found)")
    return "; ".join(parts)


def _parse_tolerance(tol_str: str) -> tuple[str, float]:
    """Parse tolerance string into (mode, value). mode: 'exact'|'abs'|'rel'."""
    if not tol_str or tol_str.strip().lower() == "exact":
        return "exact", 0.0
    s = tol_str.strip().lower()
    if s.startswith("abs:"):
        return "abs", float(s[4:])
    if s.startswith("rel:"):
        return "rel", float(s[4:])
    # bare number → absolute
    return "abs", float(s)


def _compare_pair(file_a: str, path_a: str, file_b: str, path_b: str,
                  tolerance: str = "rel:1e-7",
                  slice_a: slice | None = None, slice_b: slice | None = None) -> dict:
    """Compute comparison statistics between two variables."""
    try:
        data_a, attrs_a = _load_var_data(file_a, path_a, time_slice=slice_a)
        data_b, attrs_b = _load_var_data(file_b, path_b, time_slice=slice_b)

        # Track if a slice was requested but no time dim was found
        slice_ignored_a = slice_a is not None and attrs_a.get("_detected_time_axis") is None
        slice_ignored_b = slice_b is not None and attrs_b.get("_detected_time_axis") is None

        shape_match = data_a.shape == data_b.shape
        units_a = attrs_a.get("units", "")
        units_b = attrs_b.get("units", "")
        nan_a = int(np.isnan(data_a).sum())
        nan_b = int(np.isnan(data_b).sum())

        rmse = max_abs_diff = None
        tol_status = "—"
        n_perfect = n_within = n_outside = None

        if shape_match:
            diff = data_a - data_b
            valid = ~(np.isnan(data_a) | np.isnan(data_b))
            if valid.any():
                abs_diff = np.abs(diff[valid])
                rmse = float(np.sqrt(np.mean(abs_diff ** 2)))
                max_abs_diff = float(np.max(abs_diff))

                # Tolerance classification
                tol_mode, tol_val = _parse_tolerance(tolerance)
                n_total = int(valid.sum())
                n_perfect = int((diff[valid] == 0).sum())
                if tol_mode == "exact":
                    n_within = 0
                    n_outside = n_total - n_perfect
                elif tol_mode == "abs":
                    within_mask = abs_diff <= tol_val
                    n_within = int(within_mask.sum()) - n_perfect
                    n_outside = n_total - int(within_mask.sum())
                else:  # rel
                    denom = np.maximum(np.abs(data_a[valid]), np.abs(data_b[valid]))
                    denom = np.where(denom == 0, 1.0, denom)
                    rel_diff = abs_diff / denom
                    within_mask = rel_diff <= tol_val
                    n_within = int(within_mask.sum()) - n_perfect
                    n_outside = n_total - int(within_mask.sum())

                if n_outside == 0 and n_within == 0:
                    tol_status = "perfect"
                elif n_outside == 0:
                    tol_status = "within"
                else:
                    tol_status = "outside"

        return {
            "path_a": path_a, "path_b": path_b,
            "shape_a": str(data_a.shape), "shape_b": str(data_b.shape),
            "shape_match": shape_match,
            "dtype_a": str(data_a.dtype), "dtype_b": str(data_b.dtype),
            "units_a": units_a, "units_b": units_b,
            "units_match": units_a == units_b,
            "rmse": round(rmse, 6) if rmse is not None else None,
            "max_abs_diff": round(max_abs_diff, 6) if max_abs_diff is not None else None,
            "nan_a": nan_a, "nan_b": nan_b, "nan_delta": nan_a - nan_b,
            "tolerance": tolerance,
            "tol_status": tol_status,
            "n_perfect": n_perfect, "n_within": n_within, "n_outside": n_outside,
            "slice_ignored_a": slice_ignored_a, "slice_ignored_b": slice_ignored_b,
            "error": None,
        }
    except Exception as exc:
        return {
            "path_a": path_a, "path_b": path_b,
            "shape_a": "", "shape_b": "", "shape_match": False,
            "dtype_a": "", "dtype_b": "", "units_a": "", "units_b": "", "units_match": False,
            "rmse": None, "max_abs_diff": None, "nan_a": None, "nan_b": None, "nan_delta": None,
            "tolerance": tolerance, "tol_status": "—",
            "n_perfect": None, "n_within": None, "n_outside": None,
            "slice_ignored_a": False, "slice_ignored_b": False,
            "error": str(exc),
        }


def _make_compare_figures(
    file_a: str, path_a: str, file_b: str, path_b: str,
    slice_a: slice | None = None, slice_b: slice | None = None,
) -> tuple[go.Figure | None, go.Figure | None, go.Figure | None, str]:
    """Generate side-by-side and diff figures for a variable pair. Returns (fig_a, fig_b, fig_diff, warning)."""
    try:
        data_a, _ = _load_var_data(file_a, path_a, time_slice=slice_a)
        data_b, _ = _load_var_data(file_b, path_b, time_slice=slice_b)
    except Exception as exc:
        return None, None, None, str(exc)

    warning = ""
    label_a = path_a.split("/")[-1]
    label_b = path_b.split("/")[-1]

    def _cap(arr: np.ndarray) -> np.ndarray:
        if arr.size > _CMP_SIZE_CAP:
            step = max(1, arr.size // _CMP_SIZE_CAP)
            return arr.ravel()[::step]
        return arr

    sq_a = np.squeeze(data_a)
    sq_b = np.squeeze(data_b)
    _layout = dict(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",
                   plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=40, r=10, t=35, b=40))

    # ---- Overlay figure ----
    if sq_a.ndim <= 1:
        # 1D: two lines on the same axes
        fig_overlay = go.Figure()
        ya = _cap(sq_a).ravel()
        yb = _cap(sq_b).ravel()
        fig_overlay.add_trace(go.Scatter(y=ya.tolist(), mode="lines",
                                         line=dict(width=1.2), name=label_a))
        fig_overlay.add_trace(go.Scatter(y=yb.tolist(), mode="lines",
                                         line=dict(width=1.2, dash="dash"), name=label_b))
        fig_overlay.update_layout(title=dict(text=f"{label_a}  vs  {label_b}", font=dict(size=11)),
                                  showlegend=True, **_layout)
    else:
        # 2D: side-by-side subplots
        pa = sq_a.reshape(-1, sq_a.shape[-1]).astype(float)
        pb = sq_b.reshape(-1, sq_b.shape[-1]).astype(float)
        if pa.size > _CMP_SIZE_CAP:
            step = max(1, pa.shape[0] * pa.shape[1] // _CMP_SIZE_CAP)
            pa = pa[::step, :]
            pb = pb[::step, :]
            warning = "Data downsampled for display."
        fig_overlay = make_subplots(rows=1, cols=2,
                                    subplot_titles=[label_a, label_b],
                                    shared_yaxes=True, horizontal_spacing=0.05)
        fig_overlay.add_trace(go.Heatmap(z=pa.tolist(), colorscale="Viridis",
                                         showscale=False), row=1, col=1)
        fig_overlay.add_trace(go.Heatmap(z=pb.tolist(), colorscale="Viridis",
                                         showscale=True), row=1, col=2)
        fig_overlay.update_layout(**_layout)

    # ---- Diff figure ----
    fig_diff = go.Figure()
    if data_a.shape == data_b.shape:
        diff = data_a - data_b
        sq_diff = np.squeeze(diff)
        if sq_diff.ndim <= 1:
            y = _cap(sq_diff).ravel()
            fig_diff.add_trace(go.Scatter(y=y.tolist(), mode="lines",
                                          line=dict(width=1.2, color="#593196"), name="A − B"))
        else:
            pd_ = sq_diff.reshape(-1, sq_diff.shape[-1])
            if pd_.size > _CMP_SIZE_CAP:
                step = max(1, pd_.shape[0] * pd_.shape[1] // _CMP_SIZE_CAP)
                pd_ = pd_[::step, :]
            abs_max = float(np.nanmax(np.abs(pd_))) if pd_.size > 0 else 1.0
            fig_diff.add_trace(go.Heatmap(z=pd_.tolist(), colorscale="RdBu",
                                           zmin=-abs_max, zmax=abs_max, showscale=True))
        fig_diff.update_layout(title=dict(text="A − B", font=dict(size=11)), **_layout)
    else:
        fig_diff.add_annotation(text="Shape mismatch — diff not available",
                                x=0.5, y=0.5, xref="paper", yref="paper",
                                showarrow=False, font=dict(size=13, color="#888"))
        fig_diff.update_layout(**_layout)

    return fig_overlay, fig_diff, warning


_CONF_COLORS = {
    "high": "#198754",    # green
    "medium": "#fd7e14",  # orange
    "low": "#dc3545",     # red
    "none": "#adb5bd",    # grey
}
_STATUS_COLORS = {
    "confirmed": "#198754",
    "skipped": "#6c757d",
    "pending": "#0d6efd",
}


def _build_compare_layout() -> html.Div:
    """Build the Compare Files tab layout."""
    return html.Div([
        dbc.Container([

            # ---- File pickers ----
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                dbc.Button("Open file", id="cmp-open-a-btn", color="secondary",
                                           size="sm", outline=True),
                                html.Span("File A", className="fw-semibold text-muted mx-2",
                                          style={"fontSize": "12px", "whiteSpace": "nowrap"}),
                                dcc.Dropdown(
                                    id="cmp-file-a-select",
                                    options=[],
                                    value=None,
                                    clearable=True,
                                    placeholder="No file selected",
                                    style={"flex": "1", "minWidth": "200px", "fontSize": "12px"},
                                ),
                                dbc.Button(
                                    "ⓘ", id="cmp-info-a-btn", color="link", size="sm",
                                    style={"fontSize": "14px", "padding": "2px 4px", "color": "#888",
                                           "textDecoration": "none"},
                                ),
                                dbc.Tooltip(id="cmp-info-a-tooltip", target="cmp-info-a-btn",
                                            placement="bottom"),
                            ], style={"display": "flex", "alignItems": "center", "gap": "0px"}),
                            html.Div([
                                dcc.Input(id="cmp-path-a-input", type="text",
                                          placeholder="Paste path to .zarr or NetCDF file…",
                                          debounce=False,
                                          style={"flex": "1", "fontSize": "12px", "height": "31px",
                                                 "padding": "4px 8px", "border": "1px solid #ced4da",
                                                 "borderRadius": "4px"}),
                                dbc.Button("Load", id="cmp-path-a-go-btn", color="primary", size="sm",
                                           style={"marginLeft": "4px"}),
                                dbc.Button("×", id="cmp-path-a-dismiss-btn", color="link", size="sm",
                                           style={"marginLeft": "2px", "color": "#adb5bd",
                                                  "fontSize": "16px", "padding": "0 4px",
                                                  "textDecoration": "none"}),
                            ], id="cmp-path-a-row",
                               style={"display": "none", "alignItems": "center", "marginTop": "4px"}),
                        ], width=6, class_name="pe-2"),
                        dbc.Col([
                            html.Div([
                                dbc.Button("Open file", id="cmp-open-b-btn", color="secondary",
                                           size="sm", outline=True),
                                html.Span("File B", className="fw-semibold text-muted mx-2",
                                          style={"fontSize": "12px", "whiteSpace": "nowrap"}),
                                dcc.Dropdown(
                                    id="cmp-file-b-select",
                                    options=[],
                                    value=None,
                                    clearable=True,
                                    placeholder="No file selected",
                                    style={"flex": "1", "minWidth": "200px", "fontSize": "12px"},
                                ),
                                dbc.Button(
                                    "ⓘ", id="cmp-info-b-btn", color="link", size="sm",
                                    style={"fontSize": "14px", "padding": "2px 4px", "color": "#888",
                                           "textDecoration": "none"},
                                ),
                                dbc.Tooltip(id="cmp-info-b-tooltip", target="cmp-info-b-btn",
                                            placement="bottom"),
                            ], style={"display": "flex", "alignItems": "center", "gap": "0px"}),
                            html.Div([
                                dcc.Input(id="cmp-path-b-input", type="text",
                                          placeholder="Paste path to .zarr or NetCDF file…",
                                          debounce=False,
                                          style={"flex": "1", "fontSize": "12px", "height": "31px",
                                                 "padding": "4px 8px", "border": "1px solid #ced4da",
                                                 "borderRadius": "4px"}),
                                dbc.Button("Load", id="cmp-path-b-go-btn", color="primary", size="sm",
                                           style={"marginLeft": "4px"}),
                                dbc.Button("×", id="cmp-path-b-dismiss-btn", color="link", size="sm",
                                           style={"marginLeft": "2px", "color": "#adb5bd",
                                                  "fontSize": "16px", "padding": "0 4px",
                                                  "textDecoration": "none"}),
                            ], id="cmp-path-b-row",
                               style={"display": "none", "alignItems": "center", "marginTop": "4px"}),
                        ], width=6, class_name="ps-2"),
                    ], class_name="g-0"),
                ], class_name="py-2 px-3"),
            ], class_name="border-0 shadow-sm mb-2 mt-2"),

            # ---- Unified mapping + results table ----
            dbc.Card([
                dbc.CardHeader(html.Div([
                    html.Small("Variables", className="fw-semibold text-muted me-3"),
                    dbc.Button("Auto-match", id="cmp-automatch-btn", color="primary",
                               size="sm", style={"fontSize": "11px"}),
                    dbc.Button("Confirm high confidence", id="cmp-bulk-confirm-btn",
                               color="success", size="sm", outline=True, disabled=True,
                               style={"fontSize": "11px", "marginLeft": "6px"}),
                    dbc.Button("Confirm all", id="cmp-confirm-all-btn",
                               color="success", size="sm", disabled=True,
                               style={"fontSize": "11px", "marginLeft": "6px"}),
                    dbc.RadioItems(
                        id="cmp-mapping-filter",
                        options=[
                            {"label": "All", "value": "all"},
                            {"label": "Pending", "value": "pending"},
                            {"label": "Confirmed", "value": "confirmed"},
                            {"label": "Skipped", "value": "skipped"},
                        ],
                        value="all", inline=True,
                        style={"fontSize": "11px", "marginLeft": "16px"},
                        input_class_name="me-1", label_class_name="me-2",
                    ),
                ], style={"display": "flex", "alignItems": "center", "flexWrap": "wrap", "gap": "4px"})),
                dbc.CardBody(
                    dcc.Loading(
                        html.Div(id="cmp-mapping-area",
                                 children=html.Span("Load both files and click Auto-match.",
                                                    className="text-muted small")),
                        type="circle", color="#0d6efd", delay_show=200,
                    ),
                    class_name="p-2",
                ),
                dbc.CardFooter(html.Div([
                    dbc.Button("Run Comparison", id="cmp-run-btn", color="primary", size="sm"),
                    html.Div([
                        dbc.Label("A time slice:", className="text-muted me-1",
                                  style={"fontSize": "11px", "marginBottom": 0, "whiteSpace": "nowrap"}),
                        dbc.Input(id="cmp-slice-a", placeholder="e.g. 0:800 or ::2", size="sm",
                                  debounce=True, style={"width": "120px", "fontSize": "11px"}),
                    ], style={"display": "flex", "alignItems": "center", "gap": "4px", "marginLeft": "12px"}),
                    html.Div([
                        dbc.Label("B time slice:", className="text-muted me-1",
                                  style={"fontSize": "11px", "marginBottom": 0, "whiteSpace": "nowrap"}),
                        dbc.Input(id="cmp-slice-b", placeholder="e.g. 0:800 or ::2", size="sm",
                                  debounce=True, style={"width": "120px", "fontSize": "11px"}),
                    ], style={"display": "flex", "alignItems": "center", "gap": "4px", "marginLeft": "8px"}),
                    dcc.Loading(
                        html.Span(id="cmp-run-status", className="text-muted small ms-3",
                                  style={"fontSize": "11px"}),
                        type="circle", color="#0d6efd", delay_show=200,
                        style={"marginLeft": "8px"},
                    ),
                    html.Span("│", style={"color": "#dee2e6", "margin": "0 8px"}),
                    dbc.Button("Export mapping", id="cmp-export-mapping-btn",
                               color="link", size="sm",
                               style={"fontSize": "11px", "padding": "0"}),
                    dcc.Upload(
                        id="cmp-import-mapping-upload",
                        children=dbc.Button("Import mapping", color="link", size="sm",
                                            style={"fontSize": "11px", "padding": "0"}),
                        accept=".json",
                        multiple=False,
                        style={"display": "inline-flex", "alignItems": "center",
                               "marginLeft": "8px"},
                    ),
                    html.Span(id="cmp-import-status", className="text-muted",
                              style={"fontSize": "10px", "marginLeft": "6px"}),
                    dbc.Button("Download HTML Report", id="cmp-report-btn",
                               color="link", size="sm",
                               style={"fontSize": "11px", "padding": "0", "marginLeft": "auto"}),
                    dbc.Button("Download CSV", id="cmp-csv-btn",
                               color="link", size="sm",
                               style={"fontSize": "11px", "padding": "0", "marginLeft": "8px"}),
                ], style={"display": "flex", "alignItems": "center", "flexWrap": "wrap"}),
                    class_name="py-2 px-3"),
            ], class_name="border-0 shadow-sm mb-2"),

            # ---- Detail controls bar (shown when a row is clicked) ----
            html.Div([
                dbc.RadioItems(
                    id="cmp-detail-mode",
                    options=[{"label": "Plot", "value": "plot"}, {"label": "Table", "value": "table"}],
                    value="plot", inline=True,
                    input_class_name="btn-check",
                    label_class_name="btn btn-outline-secondary btn-sm",
                    label_checked_class_name="btn btn-secondary btn-sm active",
                ),
                html.Span("│", style={"color": "#dee2e6", "margin": "0 10px"}),
                html.Div([
                    dbc.Label("Log", style={"fontSize": "10px", "marginRight": "4px", "marginBottom": 0}),
                    dbc.Switch(id="cmp-log-scale", value=False, style={"fontSize": "11px"}),
                ], style={"display": "flex", "alignItems": "center"}),
                html.Div([
                    dbc.Label("CF scale", style={"fontSize": "10px", "marginRight": "4px", "marginBottom": 0}),
                    dbc.Switch(id="cmp-apply-scale", value=True, style={"fontSize": "11px"}),
                ], style={"display": "flex", "alignItems": "center", "marginLeft": "10px"}),
            ], id="cmp-detail-controls",
               style={"display": "none", "alignItems": "center",
                      "padding": "8px 0 4px 0", "borderTop": "1px solid #dee2e6", "marginTop": "8px"}),

            # ---- Detail panel ----
            html.Div(id="cmp-detail-panel"),

        ], fluid=True),
    ])


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

def make_layout(initial_path: str | None = None) -> html.Div:
    tree_content: list = []
    x_axis_opts: list = []
    open_files_init: list = []
    active_file_init: str = ""

    if initial_path:
        try:
            open_zarr_file(initial_path)
            s = get_store(initial_path)
            tree_content = build_tree(s)
            x_axis_opts = collect_arrays(s)
            label = initial_path.split("/")[-1]
            size_mb = _get_dir_size_mb(initial_path)
            label_with_size = f"{label}  ({size_mb:.1f} MB)"
            open_files_init = [{"id": initial_path, "label": label_with_size}]
            active_file_init = initial_path
        except Exception as e:
            tree_content = [dbc.Alert(f"Could not open zarr file: {e}", color="danger")]

    file_opts = [{"label": f["label"], "value": f["id"]} for f in open_files_init]

    return html.Div([
        # ---- Stores (MUST be in static layout) ----------------------------
        dcc.Store(id="var-tabs", data=[]),
        dcc.Store(id="active-var-tab", data=""),
        dcc.Store(id="render-trigger", data=0),
        dcc.Store(id="dim-coords-store", data={}),
        dcc.Store(id="view-mode", data="plot"),
        dcc.Store(id="open-files", data=open_files_init),
        dcc.Store(id="active-file", data=active_file_init),
        dcc.Store(id="var-prefs", data={}),
        # Animation stores
        dcc.Store(id="play-state", data={"playing": False, "dim": 0, "frame": 0, "max_frame": 0}),
        dcc.Interval(id="play-interval", interval=500, n_intervals=0, disabled=True),
        dcc.Download(id="download-csv"),
        # ---- Comparison stores (must live outside tabs) --------------------
        dcc.Store(id="cmp-file-a-path", data=""),
        dcc.Store(id="cmp-file-b-path", data=""),
        dcc.Store(id="cmp-file-a-vars", data=[]),
        dcc.Store(id="cmp-file-b-vars", data=[]),
        dcc.Store(id="cmp-mapping", data=[]),
        dcc.Store(id="cmp-results", data=[]),
        dcc.Download(id="cmp-download-report"),
        dcc.Download(id="cmp-download-csv"),
        dcc.Download(id="cmp-download-mapping"),
        dcc.Store(id="app-mode", data="explore"),

        # ---- Navbar --------------------------------------------------------
        dbc.Navbar(
            dbc.Container([
                dbc.NavbarBrand(f"Zarr Explorer v{__version__}", class_name="fw-bold me-3"),
                dbc.RadioItems(
                    id="app-mode-selector",
                    options=[
                        {"label": "Explore", "value": "explore"},
                        {"label": "Compare", "value": "compare"},
                    ],
                    value="explore",
                    inline=True,
                    input_class_name="btn-check",
                    label_class_name="btn btn-outline-light btn-sm",
                    label_checked_class_name="btn btn-light btn-sm active",
                    class_name="me-4",
                ),
            ], fluid=True),
            color="dark", dark=True, class_name="mb-0 py-2",
        ),

        # ---- Explorer content ----------------------------------------------
        html.Div(id="explorer-content", children=[

        # File bar
        dbc.Container(html.Div([
            html.Div([
                dbc.Button("Open file", id="open-btn", color="secondary", size="sm", outline=True),
                dcc.Dropdown(
                    id="file-select",
                    options=file_opts,
                    value=active_file_init or None,
                    clearable=True,
                    placeholder="No file open",
                    style={"flex": "1", "minWidth": "200px", "fontSize": "12px"},
                ),
                dbc.Button(
                    "ⓘ", id="file-info-btn", color="link", size="sm",
                    style={"fontSize": "14px", "padding": "2px 4px", "color": "#888",
                           "textDecoration": "none"},
                ),
                dbc.Tooltip(id="file-info-tooltip", target="file-info-btn", placement="bottom"),
            ], style={"display": "flex", "alignItems": "center", "gap": "8px"}),
            html.Div([
                dcc.Input(
                    id="path-input",
                    type="text",
                    placeholder="Paste path to .zarr directory or NetCDF file...",
                    debounce=False,
                    style={"flex": "1", "fontSize": "12px", "height": "31px",
                           "padding": "4px 8px", "border": "1px solid #ced4da",
                           "borderRadius": "4px"},
                ),
                dbc.Button("Load", id="path-go-btn", color="primary", size="sm",
                           style={"marginLeft": "4px"}),
                dbc.Button("×", id="path-dismiss-btn", color="link", size="sm",
                           style={"marginLeft": "2px", "color": "#adb5bd",
                                  "fontSize": "16px", "padding": "0 4px",
                                  "textDecoration": "none"}),
            ], id="path-input-row", style={"display": "none", "alignItems": "center", "marginTop": "4px"}),
            html.Div(id="open-error", className="text-danger small"),
        ], style={"display": "flex", "flexDirection": "column", "padding": "8px 0", "maxWidth": "50%"}),
        fluid=True, class_name="border-bottom mb-2 pb-1"),

        dbc.Container(
            dbc.Row([

                # Left: variable tree with search
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader([
                            html.Small("Variables", className="fw-semibold text-muted"),
                        ]),
                        dbc.CardBody([
                            # Search input at top
                            dbc.Input(
                                id="var-search",
                                type="text",
                                placeholder="Search variables...",
                                debounce=False,
                                size="sm",
                                style={"fontSize": "11px", "borderRadius": "0",
                                       "borderLeft": "none", "borderRight": "none",
                                       "borderTop": "none", "marginBottom": "4px"},
                            ),
                            html.Div(
                                id="var-tree",
                                children=tree_content,
                                style={"overflowY": "auto", "height": "calc(100vh - 210px)"},
                            ),
                        ], class_name="p-0"),
                    ], class_name="h-100 border-0 shadow-sm"),
                    width=3, class_name="pe-1",
                ),

                # Right column
                dbc.Col([

                    # Variable tab bar
                    html.Div(
                        html.Div([
                            html.Div(
                                id="var-tab-bar",
                                style={
                                    "display": "flex",
                                    "flexWrap": "nowrap",
                                    "alignItems": "center",
                                    "gap": "2px",
                                    "flex": "1",
                                },
                            ),
                            dbc.Button(
                                "Close all",
                                id="close-all-tabs-btn",
                                color="link",
                                size="sm",
                                className="text-muted flex-shrink-0",
                                style={"fontSize": "10px", "padding": "2px 6px"},
                            ),
                        ], style={"display": "flex", "alignItems": "center"}),
                        id="var-tab-row",
                        style={
                            "display": "none",
                            "borderBottom": "2px solid #593196",
                            "marginBottom": "6px",
                            "overflowX": "auto",
                        },
                    ),

                    # Attributes (left) + Slices & Options (right)
                    dbc.Row([
                        dbc.Col(
                            dbc.Card([
                                dbc.CardHeader(html.Small("Attributes", className="fw-semibold text-muted")),
                                dbc.CardBody(
                                    html.Pre(
                                        id="attrs-display",
                                        children="Select a variable.",
                                        style={
                                            "fontSize": "11px", "margin": 0,
                                            "overflowY": "auto", "maxHeight": "150px",
                                            "color": "#2c3e50", "background": "transparent",
                                        },
                                    ),
                                    class_name="py-2 px-3",
                                ),
                            ], class_name="h-100 border-0 shadow-sm"),
                            width=4, class_name="pe-1",
                        ),
                        dbc.Col(
                            dbc.Card([
                                dbc.CardHeader(
                                    html.Small("Dimension slices & Plot options", className="fw-semibold text-muted"),
                                ),
                                dbc.CardBody(
                                    dbc.Row([
                                        # Left: dimension slices + animation
                                        dbc.Col([
                                            html.Div([
                                                html.Small("Dimension slices", className="text-muted"),
                                                html.Div([
                                                    dbc.Label(
                                                        "Show coords",
                                                        style={"fontSize": "10px", "marginRight": "2px",
                                                               "marginBottom": "0", "whiteSpace": "nowrap"},
                                                    ),
                                                    dbc.Switch(
                                                        id="show-coords",
                                                        value=False,
                                                        style={"fontSize": "11px"},
                                                    ),
                                                ], style={"display": "flex", "alignItems": "center",
                                                          "marginLeft": "12px"}),
                                            ], style={"display": "flex", "alignItems": "center"}),
                                            html.Div([
                                                html.Small(
                                                    id="slice-hint",
                                                    children="Select an array variable.",
                                                    className="text-muted",
                                                    style={
                                                        "fontFamily": "monospace",
                                                        "fontSize": "10px",
                                                        "display": "block",
                                                        "marginBottom": "4px",
                                                    },
                                                ),
                                                html.Div([
                                                    html.Span(
                                                        "[ ",
                                                        style={
                                                            "fontFamily": "monospace",
                                                            "color": "#888",
                                                            "fontSize": "13px",
                                                        },
                                                    ),
                                                    dbc.Input(
                                                        id="slice-text",
                                                        type="text",
                                                        value="",
                                                        debounce=True,
                                                        style={
                                                            "fontFamily": "monospace",
                                                            "fontSize": "12px",
                                                            "width": "320px",
                                                        },
                                                        size="sm",
                                                    ),
                                                    html.Span(
                                                        " ]",
                                                        style={
                                                            "fontFamily": "monospace",
                                                            "color": "#888",
                                                            "fontSize": "13px",
                                                        },
                                                    ),
                                                ], id="slice-input-row",
                                                    style={
                                                        "display": "none",
                                                        "alignItems": "center",
                                                        "gap": "4px",
                                                    }),
                                                html.Small(
                                                    ": full  |  5 fix  |  3:10 range  |  ::2 step",
                                                    id="slice-legend",
                                                    className="text-muted",
                                                    style={
                                                        "display": "none",
                                                        "fontSize": "10px",
                                                        "marginTop": "3px",
                                                    },
                                                ),
                                            ], id="slice-controls", className="mt-1"),
                                        ], width="auto"),
                                        # Divider
                                        dbc.Col(
                                            html.Div(style={
                                                "borderLeft": "1px solid #dee2e6",
                                                "height": "100%",
                                                "minHeight": "60px",
                                            }),
                                            width="auto",
                                            class_name="px-2",
                                        ),
                                        # Right: compact plot options
                                        dbc.Col([
                                            html.Div([
                                                html.Small("Plot options", className="text-muted"),
                                                dbc.Button(
                                                    "Reset",
                                                    id="reset-opts-btn",
                                                    color="link",
                                                    size="sm",
                                                    style={"fontSize": "10px", "padding": "0 4px",
                                                           "marginLeft": "8px", "color": "#888"},
                                                ),
                                            ], style={"display": "flex", "alignItems": "center"}),
                                            html.Div([
                                                dcc.Dropdown(
                                                    id="option-selector",
                                                    options=[
                                                        {"label": "Plot type", "value": "plot-type"},
                                                        {"label": "X axis", "value": "x-axis"},
                                                        {"label": "Axis range", "value": "axis-range"},
                                                        {"label": "Colorscale / Range", "value": "colorscale"},
                                                        {"label": "B variable", "value": "b-var"},
                                                        {"label": "Play animation", "value": "play"},
                                                    ],
                                                    value=None,
                                                    placeholder="Select option...",
                                                    clearable=True,
                                                    style={
                                                        "fontSize": "12px",
                                                        "width": "170px",
                                                        "flexShrink": "0",
                                                    },
                                                ),
                                                # Group: Plot type
                                                html.Div([
                                                    dbc.Label(
                                                        "Type",
                                                        style={"fontSize": "10px", "marginBottom": "2px",
                                                               "marginRight": "4px"},
                                                    ),
                                                    dcc.Dropdown(
                                                        id="plot-type-select",
                                                        options=[
                                                            {"label": t, "value": t}
                                                            for t in [
                                                                "Auto", "Line", "Scatter", "Bar",
                                                                "Heatmap", "Contour", "Histogram",
                                                            ]
                                                        ],
                                                        value="Auto",
                                                        clearable=False,
                                                        style={"fontSize": "12px", "width": "120px"},
                                                    ),
                                                ], id="opts-plot-type",
                                                    style={
                                                        "display": "none",
                                                        "alignItems": "center",
                                                        "gap": "4px",
                                                    }),
                                                # Group: X axis
                                                html.Div([
                                                    html.Div([
                                                        dbc.Label(
                                                            "Dim coord",
                                                            style={"fontSize": "10px", "marginRight": "4px",
                                                                   "whiteSpace": "nowrap"},
                                                        ),
                                                        dcc.Dropdown(
                                                            id="x-dim-select",
                                                            options=[{"label": "Index", "value": ""}],
                                                            value="",
                                                            clearable=False,
                                                            style={"fontSize": "11px", "width": "130px",
                                                                   "minWidth": "90px"},
                                                        ),
                                                    ], style={"display": "flex", "alignItems": "center",
                                                              "gap": "2px"}),
                                                    html.Div([
                                                        dbc.Label(
                                                            "or array",
                                                            style={"fontSize": "10px", "marginRight": "4px",
                                                                   "whiteSpace": "nowrap"},
                                                        ),
                                                        dcc.Dropdown(
                                                            id="x-axis-select",
                                                            options=x_axis_opts,
                                                            value=None,
                                                            placeholder="none",
                                                            clearable=True,
                                                            style={"fontSize": "11px", "width": "140px",
                                                                   "minWidth": "90px"},
                                                        ),
                                                    ], style={"display": "flex", "alignItems": "center",
                                                              "gap": "2px"}),
                                                    html.Div([
                                                        dbc.Label(
                                                            "TAI->UTC",
                                                            style={"fontSize": "10px", "marginRight": "2px",
                                                                   "whiteSpace": "nowrap"},
                                                        ),
                                                        dbc.Switch(
                                                            id="tai-to-utc",
                                                            value=False,
                                                            style={"fontSize": "11px"},
                                                        ),
                                                    ], style={
                                                        "display": "flex",
                                                        "alignItems": "center",
                                                    }),
                                                ], id="opts-x-axis",
                                                    style={
                                                        "display": "none",
                                                        "alignItems": "center",
                                                        "gap": "4px",
                                                    }),
                                                # Group: Colorscale + range
                                                html.Div([
                                                    dcc.Dropdown(
                                                        id="colorscale-select",
                                                        options=[
                                                            {"label": c, "value": c}
                                                            for c in [
                                                                "Viridis", "Plasma", "Inferno", "Magma", "Hot",
                                                                "Jet", "RdBu", "RdYlBu", "Spectral",
                                                                "Greys", "Blues", "Reds", "YlOrRd",
                                                            ]
                                                        ],
                                                        value="Viridis",
                                                        clearable=False,
                                                        style={"fontSize": "12px", "width": "110px"},
                                                    ),
                                                    dbc.Label(
                                                        "Min",
                                                        style={
                                                            "fontSize": "10px",
                                                            "marginLeft": "4px",
                                                            "marginRight": "2px",
                                                        },
                                                    ),
                                                    dbc.Input(
                                                        id="cmin-input",
                                                        type="number",
                                                        placeholder="auto",
                                                        size="sm",
                                                        style={"width": "60px", "fontSize": "11px"},
                                                        debounce=True,
                                                    ),
                                                    dbc.Label(
                                                        "Max",
                                                        style={
                                                            "fontSize": "10px",
                                                            "marginLeft": "4px",
                                                            "marginRight": "2px",
                                                        },
                                                    ),
                                                    dbc.Input(
                                                        id="cmax-input",
                                                        type="number",
                                                        placeholder="auto",
                                                        size="sm",
                                                        style={"width": "60px", "fontSize": "11px"},
                                                        debounce=True,
                                                    ),
                                                ], id="opts-colorscale",
                                                    style={
                                                        "display": "none",
                                                        "alignItems": "center",
                                                        "gap": "2px",
                                                    }),
                                                # Group: B variable + operation
                                                html.Div([
                                                    dcc.Dropdown(
                                                        id="b-var-select",
                                                        options=x_axis_opts,
                                                        value=None,
                                                        placeholder="B variable...",
                                                        clearable=True,
                                                        style={"fontSize": "12px", "width": "180px"},
                                                    ),
                                                    dcc.Dropdown(
                                                        id="op-select",
                                                        options=[
                                                            {"label": o, "value": o}
                                                            for o in ["Overlay", "A-B", "B-A", "(A+B)/2", "A*B", "A/B"]
                                                        ],
                                                        value=None,
                                                        placeholder="op...",
                                                        clearable=True,
                                                        style={"fontSize": "12px", "width": "110px"},
                                                    ),
                                                ], id="opts-b-var",
                                                    style={
                                                        "display": "none",
                                                        "alignItems": "center",
                                                        "gap": "6px",
                                                    }),
                                                # Group: Axis range
                                                html.Div([
                                                    dbc.Label("X", style={"fontSize": "10px", "marginRight": "4px"}),
                                                    dbc.Input(
                                                        id="xmin-input", type="number", placeholder="min",
                                                        size="sm", style={"width": "70px"}, debounce=True,
                                                    ),
                                                    dbc.Input(
                                                        id="xmax-input", type="number", placeholder="max",
                                                        size="sm", style={"width": "70px"}, debounce=True,
                                                    ),
                                                    dbc.Label(
                                                        "Y",
                                                        style={
                                                            "fontSize": "10px",
                                                            "marginLeft": "10px",
                                                            "marginRight": "4px",
                                                        },
                                                    ),
                                                    dbc.Input(
                                                        id="ymin-input", type="number", placeholder="min",
                                                        size="sm", style={"width": "70px"}, debounce=True,
                                                    ),
                                                    dbc.Input(
                                                        id="ymax-input", type="number", placeholder="max",
                                                        size="sm", style={"width": "70px"}, debounce=True,
                                                    ),
                                                ], id="opts-axis-range",
                                                    style={
                                                        "display": "none",
                                                        "alignItems": "center",
                                                        "gap": "4px",
                                                    }),
                                            ], style={
                                                "marginTop": "4px",
                                            }),
                                            # Animation controls (play/pause) — shown via option-selector
                                            html.Div([
                                                html.Div([
                                                    dbc.Button(
                                                        "\u25B6",
                                                        id="play-btn",
                                                        color="primary",
                                                        outline=True,
                                                        size="sm",
                                                        style={
                                                            "fontSize": "11px",
                                                            "padding": "2px 8px",
                                                            "marginRight": "4px",
                                                        },
                                                        title="Play/pause animation through slices",
                                                    ),
                                                    dbc.Label(
                                                        "Dim:",
                                                        style={
                                                            "fontSize": "10px",
                                                            "marginRight": "2px",
                                                            "marginBottom": "0",
                                                        },
                                                    ),
                                                    dbc.Input(
                                                        id="play-dim-input",
                                                        type="number",
                                                        value=0,
                                                        min=0,
                                                        size="sm",
                                                        style={"width": "50px", "fontSize": "11px"},
                                                    ),
                                                    dbc.Label(
                                                        "Speed:",
                                                        style={
                                                            "fontSize": "10px",
                                                            "marginLeft": "8px",
                                                            "marginRight": "2px",
                                                            "marginBottom": "0",
                                                        },
                                                    ),
                                                    dcc.Dropdown(
                                                        id="play-speed",
                                                        options=[
                                                            {"label": "0.2s", "value": 200},
                                                            {"label": "0.5s", "value": 500},
                                                            {"label": "1s", "value": 1000},
                                                            {"label": "2s", "value": 2000},
                                                        ],
                                                        value=500,
                                                        clearable=False,
                                                        style={"width": "75px", "fontSize": "11px"},
                                                    ),
                                                    html.Span(
                                                        id="play-frame-label",
                                                        children="",
                                                        style={"fontSize": "10px", "marginLeft": "8px",
                                                               "fontFamily": "monospace", "color": "#593196"},
                                                    ),
                                                ], id="anim-controls",
                                                    style={
                                                        "display": "flex",
                                                        "alignItems": "center",
                                                        "gap": "2px",
                                                        "flexWrap": "nowrap",
                                                    }),
                                                html.Div(
                                                    dcc.Slider(
                                                        id="frame-slider",
                                                        min=0,
                                                        max=1,
                                                        step=1,
                                                        value=0,
                                                        marks=None,
                                                        tooltip={"placement": "bottom", "always_visible": False},
                                                    ),
                                                    id="frame-slider-row",
                                                    style={"marginTop": "4px"},
                                                ),
                                            ], id="opts-play",
                                                style={"display": "none"}),
                                        ], width="auto",
                                            style={"overflow": "hidden", "maxWidth": "100%"}),
                                    ], class_name="g-0 align-items-start",
                                        style={"flexWrap": "nowrap", "overflow": "hidden"}),
                                    class_name="py-2 px-3",
                                ),
                                dbc.CardFooter(
                                    html.Div([
                                        dbc.Button("Refresh", id="refresh-btn", color="primary", size="sm"),
                                        dbc.RadioItems(
                                            id="view-toggle",
                                            options=[
                                                {"label": "Plot", "value": "plot"},
                                                {"label": "Table", "value": "table"},
                                            ],
                                            value="plot",
                                            inline=True,
                                            class_name="ms-3",
                                            input_class_name="btn-check",
                                            label_class_name="btn btn-outline-secondary btn-sm",
                                            label_checked_class_name="btn btn-secondary btn-sm active",
                                        ),
                                        html.Span("│", style={"color": "#dee2e6", "margin": "0 10px"}),
                                        html.Div([
                                            dbc.Label("Log", style={"fontSize": "10px", "marginRight": "4px", "marginBottom": 0}),
                                            dbc.Switch(id="log-scale", value=False, style={"fontSize": "11px"}),
                                        ], style={"display": "flex", "alignItems": "center"}),
                                        html.Div([
                                            dbc.Label("CF scale", style={"fontSize": "10px", "marginRight": "4px", "marginBottom": 0}),
                                            dbc.Switch(id="apply-scale", value=True, style={"fontSize": "11px"}),
                                        ], style={"display": "flex", "alignItems": "center", "marginLeft": "10px"}),
                                    ], style={"display": "flex", "alignItems": "center"}),
                                    class_name="py-2 px-3",
                                ),
                            ], class_name="h-100 border-0 shadow-sm"),
                            width=8,
                        ),
                    ], class_name="g-2 mb-2 align-items-stretch"),

                    # Plot/table area
                    dbc.Card(
                        dbc.CardBody(
                            dcc.Loading(
                                html.Div(
                                    id="plot-graph-area",
                                    children=html.Div(
                                        "Select a variable from the left panel.",
                                        className="text-muted p-4 text-center",
                                    ),
                                ),
                                type="circle", color="#0d6efd", delay_show=200,
                            ),
                            class_name="p-1",
                        ),
                        class_name="border-0 shadow-sm",
                    ),

                ], width=9, class_name="ps-1"),

            ], class_name="g-2 pt-2"),
            fluid=True,
        ),
        ]),  # end explorer-content

        # ---- Compare content -----------------------------------------------
        html.Div(id="compare-content", children=[_build_compare_layout()],
                 style={"display": "none"}),
    ])


app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.PULSE, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True,
)
app.title = f"Zarr Explorer v{__version__}"
app.index_string = app.index_string.replace(
    "</head>",
    """<style>
#cmp-unified-table td[data-dash-column="path_b"] { cursor: pointer; padding-right: 20px !important; }
#cmp-unified-table td[data-dash-column="path_b"] .dash-cell-value {
    display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; position: relative;
}
#cmp-unified-table td[data-dash-column="path_b"] .dash-cell-value::after {
    content: "▾"; color: #adb5bd; font-size: 10px; position: absolute; right: -14px; top: 0;
}
#cmp-unified-table td[data-dash-column="status"] { cursor: pointer; padding-right: 20px !important; }
#cmp-unified-table td[data-dash-column="status"] .dash-cell-value {
    display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; position: relative;
}
#cmp-unified-table td[data-dash-column="status"] .dash-cell-value::after {
    content: "▾"; color: #adb5bd; font-size: 10px; position: absolute; right: -14px; top: 0;
}
/* Allow dropdown menus in the compare table to escape the card overflow container */
#cmp-mapping-area { overflow: visible !important; }
#cmp-mapping-area .dash-table-container { overflow: visible !important; }
#cmp-mapping-area .dash-spreadsheet-container { overflow: auto; max-height: 420px; }
/* Wider tooltips for file info */
.tooltip-inner { max-width: 500px !important; }
/* Align dcc.Upload wrapper with sibling flex items */
#cmp-import-mapping-upload { display: inline-flex !important; align-items: center !important; line-height: 1 !important; }
</style></head>""",
)
app.layout = make_layout(sys.argv[1] if len(sys.argv) > 1 else None)


# ---------------------------------------------------------------------------
# Native folder picker (tkinter subprocess)
# ---------------------------------------------------------------------------

def _open_folder_dialog() -> str | None:
    """Open a native file/folder picker. Returns selected path, '' if cancelled, None if failed."""
    import subprocess as _sp
    import platform as _platform

    initialdir = os.path.dirname(os.path.abspath(sys.argv[0]))

    if _platform.system() == "Darwin":
        # osascript choose file or folder handles both NetCDF files and Zarr directories
        safe_dir = initialdir.replace('"', '\\"')
        script = (
            f'set defaultDir to POSIX file "{safe_dir}"\n'
            "try\n"
            "    set chosen to choose file or folder with prompt "
            '"Select a Zarr directory or NetCDF file" default location defaultDir\n'
            "    POSIX path of chosen\n"
            "on error\n"
            '    ""\n'
            "end try"
        )
        try:
            result = _sp.run(["osascript", "-e", script], capture_output=True, text=True, timeout=30)
            path = result.stdout.strip()
            if path:
                return path
            if result.returncode != 0:
                return None  # osascript failed
            return ""  # user cancelled
        except Exception:
            return None

    # Non-macOS: file dialog first (NetCDF), then directory dialog (Zarr) if cancelled
    _picker_script = (
        "import tkinter as tk\n"
        "from tkinter import filedialog\n"
        "root = tk.Tk()\n"
        "root.withdraw()\n"
        "root.call('wm', 'attributes', '.', '-topmost', True)\n"
        "root.lift()\n"
        "root.focus_force()\n"
        f"initialdir = {repr(initialdir)}\n"
        "path = filedialog.askopenfilename(\n"
        "    title='Select NetCDF file', initialdir=initialdir,\n"
        "    filetypes=[('NetCDF / HDF5', '*.nc *.h5 *.hdf5 *.he5 *.nc4'), ('All files', '*')])\n"
        "if not path:\n"
        "    path = filedialog.askdirectory(title='Select Zarr directory', initialdir=initialdir)\n"
        "print(path or '')\n"
    )
    try:
        result = _sp.run(
            [sys.executable, "-c", _picker_script],
            capture_output=True, text=True, timeout=30,
        )
        path = result.stdout.strip()
        if path:
            return path
        if result.returncode != 0 or result.stderr.strip():
            return None
        return ""
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

# ---- File management ----

@app.callback(
    Output("open-files", "data"),
    Output("active-file", "data"),
    Output("open-error", "children"),
    Output("file-select", "options"),
    Output("file-select", "value"),
    Output("cmp-file-a-select", "options"),
    Output("cmp-file-a-select", "value"),
    Output("cmp-file-b-select", "options"),
    Output("cmp-file-b-select", "value"),
    Output("path-input-row", "style"),
    Output("path-input", "value"),
    Output("cmp-path-a-row", "style"),
    Output("cmp-path-b-row", "style"),
    Output("cmp-path-a-input", "value"),
    Output("cmp-path-b-input", "value"),
    Input("open-btn", "n_clicks"),
    Input("file-select", "value"),
    Input("path-go-btn", "n_clicks"),
    Input("cmp-open-a-btn", "n_clicks"),
    Input("cmp-open-b-btn", "n_clicks"),
    Input("cmp-path-a-go-btn", "n_clicks"),
    Input("cmp-path-b-go-btn", "n_clicks"),
    State("path-input", "value"),
    State("cmp-path-a-input", "value"),
    State("cmp-path-b-input", "value"),
    State("open-files", "data"),
    State("active-file", "data"),
    State("cmp-file-a-select", "value"),
    State("cmp-file-b-select", "value"),
    prevent_initial_call=True,
)
def manage_files(_open, _switch, _go,
                 _cmp_open_a, _cmp_open_b, _cmp_go_a, _cmp_go_b,
                 path_input_val, path_cmp_a, path_cmp_b,
                 open_files, active_file, cmp_a_val, cmp_b_val):
    triggered = ctx.triggered_id
    no = dash.no_update
    _path_hidden = {"display": "none"}
    _path_shown = {"display": "flex", "alignItems": "center", "marginTop": "4px"}
    _cmp_row_hidden = {"display": "none"}
    _cmp_row_shown = {"display": "flex", "alignItems": "center", "marginTop": "4px"}
    # 15 outputs: open-files, active-file, open-error,
    #   file-select opts, file-select value,
    #   cmp-a-select opts, cmp-a-select value,
    #   cmp-b-select opts, cmp-b-select value,
    #   path-input-row style, path-input value,
    #   cmp-path-a-row style, cmp-path-b-row style, cmp-path-a-input value, cmp-path-b-input value

    def _opts(files):
        return [{"label": f["label"], "value": f["id"]} for f in (files or [])]

    def _do_open(path: str):
        """Add file to open_files list. Returns (new_files, error_str | None)."""
        if not path:
            return None, "No path provided"
        if path in [f["id"] for f in open_files]:
            return open_files, None  # already open
        try:
            fmt = _detect_format(path)
            if fmt == "zarr":
                open_zarr_file(path)  # cache in stores; not needed for NetCDF
            label = os.path.basename(path.rstrip("/\\"))
            size_mb = _get_dir_size_mb(path) if os.path.isdir(path) else os.path.getsize(path) / (1024 * 1024)
            new_files = open_files + [{"id": path, "label": f"{label}  ({size_mb:.1f} MB)"}]
            return new_files, None
        except Exception as e:
            return None, str(e)

    if triggered == "open-btn":
        picked = _open_folder_dialog()
        if picked:
            new_files, err = _do_open(picked)
            if err:
                return no, no, f"Failed to open: {err}", no, no, no, no, no, no, no, no, no, no, no, no
            opts = _opts(new_files)
            return new_files, picked, "", opts, picked, opts, no, opts, no, _path_hidden, "", no, no, no, no
        # dialog failed or cancelled → show path input fallback
        return no, no, "", no, no, no, no, no, no, _path_shown, "", no, no, no, no

    if triggered == "path-go-btn":
        path = (path_input_val or "").strip()
        if not path:
            return no, no, "", no, no, no, no, no, no, no, no, no, no, no, no
        new_files, err = _do_open(path)
        if err:
            return no, no, f"Failed to open: {err}", no, no, no, no, no, no, no, no, no, no, no, no
        opts = _opts(new_files)
        return new_files, path, "", opts, path, opts, no, opts, no, _path_hidden, "", no, no, no, no

    if triggered == "file-select":
        if _switch is None:  # user cleared the dropdown → close active file
            if not active_file:
                return no, no, "", no, no, no, no, no, no, no, no, no, no, no, no
            close_zarr_file(active_file)
            new_files = [f for f in open_files if f["id"] != active_file]
            new_active = new_files[-1]["id"] if new_files else None
            opts = _opts(new_files)
            new_cmp_a = None if cmp_a_val == active_file else no
            new_cmp_b = None if cmp_b_val == active_file else no
            return new_files, new_active or "", "", opts, new_active, opts, new_cmp_a, opts, new_cmp_b, no, no, no, no, no, no
        return no, _switch, "", no, no, no, no, no, no, no, no, no, no, no, no

    if triggered == "cmp-open-a-btn":
        picked = _open_folder_dialog()
        if picked:
            new_files, err = _do_open(picked)
            if err:
                return no, no, f"Failed to open: {err}", no, no, no, no, no, no, no, no, no, no, no, no
            opts = _opts(new_files)
            return new_files, no, "", opts, no, opts, picked, opts, no, no, no, _cmp_row_hidden, no, "", no
        # dialog failed or cancelled → show inline path input
        return no, no, "", no, no, no, no, no, no, no, no, _cmp_row_shown, no, no, no

    if triggered == "cmp-open-b-btn":
        picked = _open_folder_dialog()
        if picked:
            new_files, err = _do_open(picked)
            if err:
                return no, no, f"Failed to open: {err}", no, no, no, no, no, no, no, no, no, no, no, no
            opts = _opts(new_files)
            return new_files, no, "", opts, no, opts, no, opts, picked, no, no, no, _cmp_row_hidden, no, ""
        # dialog failed or cancelled → show inline path input
        return no, no, "", no, no, no, no, no, no, no, no, no, _cmp_row_shown, no, no

    if triggered == "cmp-path-a-go-btn":
        path = (path_cmp_a or "").strip()
        if not path:
            return no, no, "", no, no, no, no, no, no, no, no, no, no, no, no
        new_files, err = _do_open(path)
        if err:
            return no, no, f"Failed to open: {err}", no, no, no, no, no, no, no, no, no, no, no, no
        opts = _opts(new_files)
        return new_files, no, "", opts, no, opts, path, opts, no, no, no, _cmp_row_hidden, no, "", no

    if triggered == "cmp-path-b-go-btn":
        path = (path_cmp_b or "").strip()
        if not path:
            return no, no, "", no, no, no, no, no, no, no, no, no, no, no, no
        new_files, err = _do_open(path)
        if err:
            return no, no, f"Failed to open: {err}", no, no, no, no, no, no, no, no, no, no, no, no
        opts = _opts(new_files)
        return new_files, no, "", opts, no, opts, no, opts, path, no, no, no, _cmp_row_hidden, no, ""

    return no, no, "", no, no, no, no, no, no, no, no, no


# ---- Tree rendering with search ----

@app.callback(
    Output("var-tree", "children"),
    Output("x-axis-select", "options"),
    Output("b-var-select", "options"),
    Input("active-file", "data"),
    Input("var-search", "value"),
)
def render_tree(active_file, search_text):
    if not active_file:
        return [html.Span("Open a .zarr file.", className="text-muted small p-2")], [], []
    s = get_store(active_file)
    if s is None:
        return [html.Span("File not loaded.", className="text-muted small p-2")], [], []
    opts = collect_arrays(s)
    children = build_tree(s, search_filter=search_text or "")
    if not children and search_text:
        children = [html.Span(f'No variables matching "{search_text}".', className="text-muted small p-2")]

    # Wrap everything in a root-level collapsible group showing the file name
    root_label = os.path.basename(active_file.rstrip("/\\"))
    root_attrs = dict(s.attrs)
    # Build a short summary line from key metadata
    meta_keys = ["product_type", "product_name", "sensing_start", "sensing_stop",
                 "title", "source", "orbit_number"]
    meta_bits = []
    for k in meta_keys:
        if k in root_attrs:
            meta_bits.append(f"{k}={root_attrs[k]}")
    meta_summary = "  |  ".join(meta_bits[:3]) if meta_bits else ""

    root_header = html.Div([
        dbc.Button(
            [
                html.Span(
                    "\u25BE ",
                    id={"type": "group-arrow", "index": "__root__"},
                    style={"fontSize": "10px", "display": "inline-block",
                           "transition": "transform 0.15s"},
                ),
                html.Strong(root_label, style={"fontSize": "12px"}),
                html.Span(
                    f"  {meta_summary}" if meta_summary else "",
                    style={"fontSize": "10px", "color": "#888", "marginLeft": "8px"},
                ),
            ],
            id={"type": "group-toggle", "index": "__root__"},
            color="link",
            size="sm",
            class_name="text-start w-100",
            style={"fontSize": "12px", "fontFamily": "monospace", "padding": "4px 6px",
                   "background": "#f4f0fa"},
        ),
    ], style={"borderBottom": "2px solid #593196"})

    root_children = html.Div(children, id="root-tree-children")

    tree = [root_header, root_children]
    return tree, opts, opts


# ---- Variable selection (from tree click) ----

@app.callback(
    Output("var-tabs", "data"),
    Output("active-var-tab", "data"),
    Output("render-trigger", "data"),
    Input({"type": "var-btn", "index": ALL}, "n_clicks"),
    State("var-tabs", "data"),
    State("render-trigger", "data"),
    prevent_initial_call=True,
)
def select_variable(_, var_tabs, trigger):
    if not ctx.triggered_id:
        return dash.no_update, dash.no_update, dash.no_update
    if not ctx.triggered or not ctx.triggered[0].get("value"):
        return dash.no_update, dash.no_update, dash.no_update
    path = ctx.triggered_id["index"]
    label = path.split("/")[-1]
    if not any(t["path"] == path for t in var_tabs):
        var_tabs = var_tabs + [{"path": path, "label": label}]
    return var_tabs, path, (trigger or 0) + 1


# ---- Tab management (select / close / close-all) ----

@app.callback(
    Output("var-tabs", "data", allow_duplicate=True),
    Output("active-var-tab", "data", allow_duplicate=True),
    Output("render-trigger", "data", allow_duplicate=True),
    Input({"type": "select-var-tab", "index": ALL}, "n_clicks"),
    Input({"type": "close-var-tab", "index": ALL}, "n_clicks"),
    Input("close-all-tabs-btn", "n_clicks"),
    State("var-tabs", "data"),
    State("active-var-tab", "data"),
    State("render-trigger", "data"),
    prevent_initial_call=True,
)
def manage_var_tabs(_sel, _close, _close_all, var_tabs, active, trigger):
    t = ctx.triggered_id
    if not t:
        return dash.no_update, dash.no_update, dash.no_update

    # Close all tabs
    if t == "close-all-tabs-btn":
        return [], "", (trigger or 0) + 1

    # Pattern-matching guard
    if not ctx.triggered or not ctx.triggered[0].get("value"):
        return dash.no_update, dash.no_update, dash.no_update

    if isinstance(t, dict) and t.get("type") == "select-var-tab":
        path = t["index"]
        return dash.no_update, path, (trigger or 0) + 1

    if isinstance(t, dict) and t.get("type") == "close-var-tab":
        path = t["index"]
        new_tabs = [v for v in var_tabs if v["path"] != path]
        if not new_tabs:
            return new_tabs, "", (trigger or 0) + 1
        new_active = active if active != path else new_tabs[-1]["path"]
        return new_tabs, new_active, (trigger or 0) + 1

    return dash.no_update, dash.no_update, dash.no_update


# ---- Tab bar rendering ----

@app.callback(
    Output("var-tab-bar", "children"),
    Output("var-tab-row", "style"),
    Input("var-tabs", "data"),
    Input("active-var-tab", "data"),
)
def render_var_tab_bar(var_tabs, active):
    hidden = {
        "display": "none",
        "borderBottom": "2px solid #593196",
        "marginBottom": "6px",
        "overflowX": "auto",
    }
    visible = {
        "display": "flex",
        "borderBottom": "2px solid #593196",
        "marginBottom": "6px",
        "overflowX": "auto",
        "flexWrap": "nowrap",
    }
    if not var_tabs:
        return [], hidden
    tabs = []
    for v in var_tabs:
        is_active = v["path"] == active
        tabs.append(
            html.Div([
                dbc.Button(
                    v["label"],
                    id={"type": "select-var-tab", "index": v["path"]},
                    color="link",
                    size="sm",
                    style={
                        "fontSize": "12px",
                        "padding": "4px 6px",
                        "fontWeight": "700" if is_active else "400",
                        "color": "#593196" if is_active else "#555",
                        "borderBottom": "2px solid #593196" if is_active else "2px solid transparent",
                        "borderRadius": "0",
                    },
                ),
                dbc.Button(
                    "\u00d7",
                    id={"type": "close-var-tab", "index": v["path"]},
                    color="link",
                    size="sm",
                    className="text-muted p-0",
                    style={"fontSize": "13px", "lineHeight": "1"},
                ),
            ], style={"display": "flex", "alignItems": "center", "gap": "1px", "marginRight": "4px"})
        )
    return tabs, visible


# ---- Group toggle ----

@app.callback(
    Output({"type": "group-collapse", "index": MATCH}, "is_open"),
    Input({"type": "group-toggle", "index": MATCH}, "n_clicks"),
    State({"type": "group-collapse", "index": MATCH}, "is_open"),
    prevent_initial_call=True,
)
def toggle_group(_, is_open):
    return not is_open


# ---- View mode ----

@app.callback(
    Output("view-mode", "data"),
    Input("view-toggle", "value"),
    prevent_initial_call=True,
)
def update_view_mode(toggle_val):
    return toggle_val


# ---- Dimension coords update ----

@app.callback(
    Output("dim-coords-store", "data"),
    Output("x-dim-select", "options"),
    Output("x-dim-select", "value"),
    Input("active-var-tab", "data"),
    State("active-file", "data"),
)
def update_dim_coords(var_path, active_file):
    s = get_store(active_file or "")
    if not var_path or s is None:
        opts = [{"label": "Index", "value": ""}]
        return {}, opts, ""
    coords = find_dim_coords(s, var_path)
    opts = [{"label": "Index", "value": ""}] + [
        {"label": f"{dim}  ({path.split('/')[-1]})", "value": path}
        for dim, path in coords.items()
    ]
    return coords, opts, ""


# ---- Option panel visibility ----

_SHOW = {"display": "flex", "alignItems": "center", "gap": "4px", "flexWrap": "wrap", "marginTop": "6px"}
_HIDE = {"display": "none"}


@app.callback(
    Output("opts-plot-type", "style"),
    Output("opts-x-axis", "style"),
    Output("opts-axis-range", "style"),
    Output("opts-colorscale", "style"),
    Output("opts-b-var", "style"),
    Output("opts-play", "style"),
    Input("option-selector", "value"),
)
def show_option_group(selected):
    return (
        _SHOW if selected == "plot-type" else _HIDE,
        _SHOW if selected == "x-axis" else _HIDE,
        _SHOW if selected == "axis-range" else _HIDE,
        _SHOW if selected == "colorscale" else _HIDE,
        _SHOW if selected == "b-var" else _HIDE,
        {"display": "block"} if selected == "play" else _HIDE,
    )


# ---- Attributes display (arrays AND groups) ----

@app.callback(
    Output("attrs-display", "children"),
    Output("plot-graph-area", "children", allow_duplicate=True),
    Input("active-var-tab", "data"),
    Input({"type": "group-toggle", "index": ALL}, "n_clicks"),
    State("active-file", "data"),
    prevent_initial_call=True,
)
def show_attrs(var_path, _group_clicks, active_file):
    s = get_store(active_file or "")
    no_plot = dash.no_update
    if s is None:
        return "Select a variable to see its attributes.", no_plot

    # Check if a group header was clicked
    display_path = var_path
    is_group_click = False
    if ctx.triggered_id and isinstance(ctx.triggered_id, dict) and ctx.triggered_id.get("type") == "group-toggle":
        display_path = ctx.triggered_id["index"]
        is_group_click = True

    if not display_path:
        return "Select a variable to see its attributes.", no_plot

    try:
        if display_path == "__root__":
            node = s
            display_path = os.path.basename((active_file or "").rstrip("/\\")) or "/"
        else:
            node = s[display_path]
    except KeyError:
        return f"Not found: {display_path}", no_plot

    lines = [f"path  : {display_path}"]
    if isinstance(node, zarr.Array):
        lines += [f"shape : {node.shape}", f"dtype : {node.dtype}", f"chunks: {node.chunks}"]
    else:
        lines.append("type  : group")
        n_arrays = 0
        n_groups = 0
        for _name, child in node.members():
            if isinstance(child, zarr.Array):
                n_arrays += 1
            else:
                n_groups += 1
        lines.append(f"children: {n_arrays} arrays, {n_groups} groups")

    attrs = dict(node.attrs)
    if attrs:
        lines.append("\n--- attributes ---")
        for k, v in attrs.items():
            if isinstance(v, dict):
                lines.append(f"{k}:")
                for sk, sv in v.items():
                    # EOPF scalar variable pattern: {data: <value>, dtype: ..., attrs: {units: ...}, dims: []}
                    if isinstance(sv, dict) and "data" in sv and sv.get("dims") == []:
                        data_val = sv["data"]
                        dtype = sv.get("dtype", "")
                        units = sv.get("attrs", {}).get("units", "")
                        suffix = "  [" + ", ".join(x for x in [dtype, units] if x) + "]" if (dtype or units) else ""
                        lines.append(f"  {sk}: {data_val}{suffix}")
                    else:
                        lines.append(f"  {sk}: {sv}")
            else:
                lines.append(f"{k}: {v}")
    else:
        lines.append("\n(no attributes)")

    # Clear the plot area when a group is clicked
    clear_plot = no_plot
    if is_group_click and not isinstance(node, zarr.Array):
        clear_plot = html.Div(
            f"Group: {display_path}", className="text-muted p-4 text-center",
        )

    return "\n".join(lines), clear_plot


# ---- Slice controls ----

_HIDDEN = {"display": "none"}
_FLEX = {"display": "flex", "alignItems": "center", "gap": "4px"}
_BLOCK = {"display": "block", "fontSize": "10px", "marginTop": "3px"}
_ANIM_FLEX = {"display": "flex", "alignItems": "center", "gap": "2px",
              "marginTop": "6px", "flexWrap": "nowrap"}


@app.callback(
    Output("slice-hint", "children"),
    Output("slice-text", "value"),
    Output("slice-input-row", "style"),
    Output("slice-legend", "style"),
    Output("frame-slider", "max"),
    Input("active-var-tab", "data"),
    Input("show-coords", "value"),
    State("active-file", "data"),
)
def build_slice_controls(var_path, show_coords, active_file):
    s = get_store(active_file or "")
    if not var_path or s is None:
        return "Select an array variable.", "", _HIDDEN, _HIDDEN, 1
    try:
        node = s[var_path]
    except KeyError:
        return "", "", _HIDDEN, _HIDDEN, 1
    if not isinstance(node, zarr.Array):
        return "Groups cannot be plotted.", "", _HIDDEN, _HIDDEN, 1

    squeezed_shape = tuple(sz for sz in node.shape if sz != 1)
    if not squeezed_shape:
        return "Scalar -- no slicing needed.", "", _HIDDEN, _HIDDEN, 1

    dim_names = dict(node.attrs).get("_ARRAY_DIMENSIONS", [])
    surviving_indices = [i for i, sz in enumerate(node.shape) if sz != 1]
    surviving_names = [
        (dim_names[i] if i < len(dim_names) else f"dim{i}")
        for i in surviving_indices
    ]
    default_slice = ", ".join(":" for _ in squeezed_shape)

    # Build hint with optional coordinate ranges
    parts = []
    for name, sz, orig_i in zip(surviving_names, squeezed_shape, surviving_indices):
        label = f"{name}({sz})"
        if show_coords and s is not None:
            coord_info = _get_coord_range(s, name, sz)
            if coord_info:
                label += f" [{coord_info}]"
        parts.append(label)
    hint = "  x  ".join(parts)

    max_frame = max(squeezed_shape[0] - 1, 0) if squeezed_shape else 0

    return hint, default_slice, _FLEX, _BLOCK, max_frame


# ---- Per-variable preference persistence ----

_DEFAULT_PREFS = {
    "plot_type": "Auto",
    "colorscale": "Viridis",
    "x_axis": None,
    "b_var": None,
    "op": None,
    "cmin": None,
    "cmax": None,
    "xmin": None,
    "xmax": None,
    "ymin": None,
    "ymax": None,
    "log_scale": False,
    "apply_scale": True,
    "option_selector": None,
}


@app.callback(
    Output("var-prefs", "data"),
    Input("plot-type-select", "value"),
    Input("colorscale-select", "value"),
    Input("x-axis-select", "value"),
    Input("b-var-select", "value"),
    Input("op-select", "value"),
    Input("cmin-input", "value"),
    Input("cmax-input", "value"),
    Input("xmin-input", "value"),
    Input("xmax-input", "value"),
    Input("ymin-input", "value"),
    Input("ymax-input", "value"),
    Input("log-scale", "value"),
    Input("apply-scale", "value"),
    Input("option-selector", "value"),
    State("active-var-tab", "data"),
    State("var-prefs", "data"),
    prevent_initial_call=True,
)
def save_var_prefs(plot_type, colorscale, x_axis, b_var, op,
                   cmin, cmax, xmin, xmax, ymin, ymax,
                   log_scale, apply_scale, option_sel,
                   var_path, all_prefs):
    if not var_path:
        return dash.no_update
    all_prefs = all_prefs or {}
    all_prefs[var_path] = {
        "plot_type": plot_type or "Auto",
        "colorscale": colorscale or "Viridis",
        "x_axis": x_axis,
        "b_var": b_var,
        "op": op,
        "cmin": cmin,
        "cmax": cmax,
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "log_scale": bool(log_scale),
        "apply_scale": bool(apply_scale),
        "option_selector": option_sel,
    }
    return all_prefs


@app.callback(
    Output("plot-type-select", "value"),
    Output("colorscale-select", "value"),
    Output("x-axis-select", "value"),
    Output("b-var-select", "value"),
    Output("op-select", "value"),
    Output("cmin-input", "value"),
    Output("cmax-input", "value"),
    Output("xmin-input", "value"),
    Output("xmax-input", "value"),
    Output("ymin-input", "value"),
    Output("ymax-input", "value"),
    Output("log-scale", "value"),
    Output("apply-scale", "value"),
    Output("option-selector", "value"),
    Output("tai-to-utc", "value"),
    Input("active-var-tab", "data"),
    Input("reset-opts-btn", "n_clicks"),
    State("var-prefs", "data"),
    State("active-var-tab", "data"),
    prevent_initial_call=True,
)
def restore_var_prefs(var_path_input, _reset, all_prefs, var_path_state):
    triggered = ctx.triggered_id
    all_prefs = all_prefs or {}

    if triggered == "reset-opts-btn":
        # Reset current variable to defaults
        p = _DEFAULT_PREFS
    else:
        # Switching variable — restore saved prefs or defaults
        var_path = var_path_input
        if not var_path:
            p = _DEFAULT_PREFS
        else:
            p = all_prefs.get(var_path, _DEFAULT_PREFS)

    return (
        p.get("plot_type", "Auto"),
        p.get("colorscale", "Viridis"),
        p.get("x_axis"),
        p.get("b_var"),
        p.get("op"),
        p.get("cmin"),
        p.get("cmax"),
        p.get("xmin"),
        p.get("xmax"),
        p.get("ymin"),
        p.get("ymax"),
        p.get("log_scale", False),
        p.get("apply_scale", True),
        p.get("option_selector"),
        False,  # tai-to-utc always resets to off
    )


# ---- Animation: play/pause ----

@app.callback(
    Output("play-state", "data"),
    Output("play-interval", "disabled"),
    Output("play-btn", "children"),
    Input("play-btn", "n_clicks"),
    State("play-state", "data"),
    State("frame-slider", "max"),
    State("play-dim-input", "value"),
    prevent_initial_call=True,
)
def toggle_play(n_clicks, play_state, max_frame, play_dim):
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update
    playing = not play_state.get("playing", False)
    new_state = {
        "playing": playing,
        "dim": int(play_dim or 0),
        "frame": play_state.get("frame", 0),
        "max_frame": max_frame or 0,
    }
    btn_label = "\u23F8" if playing else "\u25B6"  # pause or play icon
    return new_state, not playing, btn_label


# ---- Animation: update speed ----

@app.callback(
    Output("play-interval", "interval"),
    Input("play-speed", "value"),
)
def update_play_speed(speed):
    return speed or 500


# ---- Animation: advance frame on interval tick ----

@app.callback(
    Output("slice-text", "value", allow_duplicate=True),
    Output("play-state", "data", allow_duplicate=True),
    Output("play-frame-label", "children"),
    Output("frame-slider", "value"),
    Output("play-interval", "disabled", allow_duplicate=True),
    Output("play-btn", "children", allow_duplicate=True),
    Input("play-interval", "n_intervals"),
    State("play-state", "data"),
    State("slice-text", "value"),
    State("active-var-tab", "data"),
    State("active-file", "data"),
    prevent_initial_call=True,
)
def advance_frame(_n, play_state, slice_text, var_path, active_file):
    if not play_state or not play_state.get("playing"):
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    s = get_store(active_file or "")
    if not var_path or s is None:
        return dash.no_update, dash.no_update, "", dash.no_update, dash.no_update, dash.no_update

    try:
        node = s[var_path]
    except KeyError:
        return dash.no_update, dash.no_update, "", dash.no_update, dash.no_update, dash.no_update
    if not isinstance(node, zarr.Array):
        return dash.no_update, dash.no_update, "", dash.no_update, dash.no_update, dash.no_update

    squeezed_shape = tuple(sz for sz in node.shape if sz != 1)
    dim_idx = play_state.get("dim", 0)
    if dim_idx >= len(squeezed_shape):
        dim_idx = 0

    frame = play_state.get("frame", 0) + 1
    max_frame = squeezed_shape[dim_idx] - 1

    # Stop at end
    if frame > max_frame:
        new_state = {**play_state, "playing": False, "frame": 0}
        return dash.no_update, new_state, f"Done ({max_frame + 1} frames)", 0, True, "\u25B6"

    # Build slice text: fix the animated dimension at `frame`, leave others as ":"
    tokens = []
    for i in range(len(squeezed_shape)):
        if i == dim_idx:
            tokens.append(str(frame))
        else:
            tokens.append(":")
    new_slice = ", ".join(tokens)

    new_state = {**play_state, "frame": frame, "max_frame": max_frame}
    label = f"Frame {frame}/{max_frame}"

    return new_slice, new_state, label, frame, dash.no_update, dash.no_update


# ---- Frame slider manual scrub ----

@app.callback(
    Output("slice-text", "value", allow_duplicate=True),
    Output("play-frame-label", "children", allow_duplicate=True),
    Input("frame-slider", "value"),
    State("play-state", "data"),
    State("active-var-tab", "data"),
    State("active-file", "data"),
    prevent_initial_call=True,
)
def scrub_frame(frame_val, play_state, var_path, active_file):
    # Only respond to user drag, not programmatic updates
    if not ctx.triggered_id or ctx.triggered_id != "frame-slider":
        return dash.no_update, dash.no_update

    # If currently playing, don't interfere
    if play_state and play_state.get("playing"):
        return dash.no_update, dash.no_update

    s = get_store(active_file or "")
    if not var_path or s is None:
        return dash.no_update, ""

    try:
        node = s[var_path]
    except KeyError:
        return dash.no_update, ""
    if not isinstance(node, zarr.Array):
        return dash.no_update, ""

    squeezed_shape = tuple(sz for sz in node.shape if sz != 1)
    dim_idx = play_state.get("dim", 0) if play_state else 0
    if dim_idx >= len(squeezed_shape):
        dim_idx = 0

    frame = int(frame_val or 0)
    max_frame = squeezed_shape[dim_idx] - 1

    tokens = []
    for i in range(len(squeezed_shape)):
        if i == dim_idx:
            tokens.append(str(frame))
        else:
            tokens.append(":")
    new_slice = ", ".join(tokens)
    label = f"Frame {frame}/{max_frame}"
    return new_slice, label


# ---- Stats bar builder ----

def _make_stats_bar(stats: dict, show_histogram: bool = True) -> html.Div:
    stat_items = [
        ("shape", stats.get("shape", "--")),
        ("dtype", stats.get("dtype", "--")),
        ("min", stats.get("min", "--")),
        ("max", stats.get("max", "--")),
        ("mean", stats.get("mean", "--")),
        ("std", stats.get("std", "--")),
        ("NaN", str(stats.get("nan", "--"))),
    ]
    spans = [
        html.Span([
            html.Span(k + ": ", style={"color": "#888", "fontSize": "10px"}),
            html.Span(v, style={"fontFamily": "monospace", "fontSize": "11px", "marginRight": "16px"}),
        ])
        for k, v in stat_items
    ]

    # Mini histogram sparkline
    hist_fig = None
    valid = stats.get("_valid")
    if show_histogram and valid is not None and len(valid) > 1:
        try:
            counts, bin_edges = np.histogram(valid, bins=40)
            centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            hist_fig = go.Figure(go.Bar(
                x=centers.tolist(), y=counts.tolist(),
                marker_color="#593196", opacity=0.7,
            ))
            hist_fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                height=30, width=120,
                bargap=0.05,
            )
        except Exception:
            hist_fig = None

    hist_component = (
        dcc.Graph(
            figure=hist_fig,
            config={"displayModeBar": False},
            style={"height": "30px", "width": "120px", "display": "inline-block",
                   "verticalAlign": "middle", "marginLeft": "8px"},
        )
        if hist_fig
        else html.Span()
    )

    return html.Div(
        html.Div([
            html.Div(spans, style={"flex": "1", "overflowX": "auto", "whiteSpace": "nowrap"}),
            hist_component,
            dbc.Button(
                "CSV",
                id="download-csv-btn",
                color="link",
                size="sm",
                className="text-primary p-0 flex-shrink-0 ms-3",
                style={"fontSize": "11px"},
            ),
            dbc.Button(
                "PNG",
                id="export-png-btn",
                color="link",
                size="sm",
                className="text-primary p-0 flex-shrink-0 ms-2",
                style={"fontSize": "11px"},
            ),
        ], style={"display": "flex", "alignItems": "center"}),
        style={"padding": "3px 8px", "background": "#f8f9fa", "borderBottom": "1px solid #dee2e6"},
    )


# ---- Main render ----

@app.callback(
    Output("plot-graph-area", "children"),
    Input("render-trigger", "data"),
    Input("refresh-btn", "n_clicks"),
    Input("reset-opts-btn", "n_clicks"),
    Input("view-mode", "data"),
    Input("slice-text", "value"),
    State("active-var-tab", "data"),
    State("x-dim-select", "value"),
    State("x-axis-select", "value"),
    State("tai-to-utc", "value"),
    State("b-var-select", "value"),
    State("op-select", "value"),
    State("plot-type-select", "value"),
    State("colorscale-select", "value"),
    State("cmin-input", "value"),
    State("cmax-input", "value"),
    State("xmin-input", "value"),
    State("xmax-input", "value"),
    State("ymin-input", "value"),
    State("ymax-input", "value"),
    Input("log-scale", "value"),
    Input("apply-scale", "value"),
    State("active-file", "data"),
    prevent_initial_call=True,
)
def render_figure_area(
    _trigger, _refresh, _reset, view_mode, slice_text,
    var_path, x_dim, x_var, tai_to_utc, b_var, op, plot_type, colorscale, cmin, cmax,
    xmin, xmax, ymin, ymax, log_scale, apply_scale,
    active_file,
):
    x_var = x_dim if x_dim else x_var
    if not var_path:
        return html.Div("Select a variable from the left panel.", className="text-muted p-4 text-center")

    s = get_store(active_file or "")
    label = var_path.split("/")[-1]

    if view_mode == "table":
        stats_bar = html.Div()
        content = _build_table(var_path, s, slice_text=slice_text,
                               x_var=x_var, tai_to_utc=bool(tai_to_utc),
                               apply_scale=bool(apply_scale), log_scale=bool(log_scale))
    else:
        fig, stats, warning = _generate_figure(
            var_path, x_var, b_var, op, plot_type, colorscale, cmin, cmax,
            xmin, xmax, ymin, ymax, log_scale, apply_scale, bool(tai_to_utc),
            slice_text, s,
        )
        if fig is None:
            msg = warning or "Could not render variable."
            return html.Div(msg, className="text-danger p-4 text-center")
        stats_bar = _make_stats_bar(stats or {})
        warning_div = (
            html.Div(warning, className="text-warning small px-2 py-1")
            if warning
            else None
        )
        content = html.Div([
            c for c in [
                warning_div,
                dcc.Graph(
                    figure=fig,
                    style={"height": "calc(100vh - 430px)", "minHeight": "300px"},
                    config={
                        "displaylogo": False,
                        "modeBarButtonsToAdd": ["toImage"],
                        "toImageButtonOptions": {"format": "svg", "filename": label},
                    },
                ),
            ]
            if c is not None
        ])

    return html.Div([stats_bar, content])


# ---- CSV download ----

@app.callback(
    Output("download-csv", "data"),
    Input("download-csv-btn", "n_clicks"),
    State("active-var-tab", "data"),
    State("active-file", "data"),
    State("slice-text", "value"),
    prevent_initial_call=True,
)
def download_csv(n_clicks, var_path, active_file, slice_text):
    if not n_clicks:
        return dash.no_update
    s = get_store(active_file or "")
    if not var_path or s is None:
        return dash.no_update
    try:
        node = s[var_path]
        if not isinstance(node, zarr.Array):
            return dash.no_update
        data, _ = _load_data_sliced(node, slice_text)
        data = data.ravel()
        import io

        buf = io.StringIO()
        buf.write(f"# variable: {var_path}\n")
        buf.write(f"# shape: {node.shape}  dtype: {node.dtype}\n")
        if slice_text:
            buf.write(f"# slice: [{slice_text}]\n")
        buf.write("index,value\n")
        for i, v in enumerate(data):
            buf.write(f"{i},{v}\n")
        filename = var_path.replace("/", "_") + ".csv"
        return dcc.send_string(buf.getvalue(), filename)
    except Exception:
        return dash.no_update


# Dismiss path input rows without loading
app.clientside_callback(
    "function(n) { return n ? {display: 'none'} : window.dash_clientside.no_update; }",
    Output("path-input-row", "style", allow_duplicate=True),
    Input("path-dismiss-btn", "n_clicks"),
    prevent_initial_call=True,
)
app.clientside_callback(
    "function(n) { return n ? {display: 'none'} : window.dash_clientside.no_update; }",
    Output("cmp-path-a-row", "style", allow_duplicate=True),
    Input("cmp-path-a-dismiss-btn", "n_clicks"),
    prevent_initial_call=True,
)
app.clientside_callback(
    "function(n) { return n ? {display: 'none'} : window.dash_clientside.no_update; }",
    Output("cmp-path-b-row", "style", allow_duplicate=True),
    Input("cmp-path-b-dismiss-btn", "n_clicks"),
    prevent_initial_call=True,
)

# PNG export via clientside JS (uses Plotly's toImage)
app.clientside_callback(
    """
    function(n_clicks) {
        if (!n_clicks) return window.dash_clientside.no_update;
        var graphs = document.querySelectorAll('.js-plotly-plot');
        if (graphs.length === 0) return window.dash_clientside.no_update;
        var gd = graphs[graphs.length - 1];
        Plotly.downloadImage(gd, {format: 'png', width: 1400, height: 700, filename: 'zarr_plot'});
        return window.dash_clientside.no_update;
    }
    """,
    Output("export-png-btn", "className"),
    Input("export-png-btn", "n_clicks"),
    prevent_initial_call=True,
)



# ---------------------------------------------------------------------------
# Mode toggle (Explore / Compare)
# ---------------------------------------------------------------------------

@app.callback(
    Output("explorer-content", "style"),
    Output("compare-content", "style"),
    Input("app-mode-selector", "value"),
)
def toggle_mode(mode):
    show = {"display": "block"}
    hide = {"display": "none"}
    return (show, hide) if mode == "explore" else (hide, show)


# ---------------------------------------------------------------------------
# Comparison callbacks
# ---------------------------------------------------------------------------

def _file_load_callback(path: str) -> tuple[str, list, str]:
    """Shared logic for loading either file in the comparison tool."""
    path = (path or "").strip()
    if not path:
        return "", [], ""
    try:
        fmt = _detect_format(path)
        vars_list = _get_file_vars(path)
        label = os.path.basename(path.rstrip("/\\"))
        size_mb = _get_dir_size_mb(path) if fmt == "zarr" else os.path.getsize(path) / (1024 * 1024)
        status = f"{label}  ({size_mb:.1f} MB)  —  {len(vars_list)} variables  [{fmt}]"
        return path, vars_list, status
    except Exception as exc:
        return "", [], f"Error: {exc}"


@app.callback(
    Output("cmp-file-a-path", "data"),
    Output("cmp-file-a-vars", "data"),
    Input("cmp-file-a-select", "value"),
    prevent_initial_call=True,
)
def update_cmp_file_a(path):
    if not path:
        return "", []
    try:
        return path, _get_file_vars(path)
    except Exception:
        return "", []


@app.callback(
    Output("cmp-file-b-path", "data"),
    Output("cmp-file-b-vars", "data"),
    Input("cmp-file-b-select", "value"),
    prevent_initial_call=True,
)
def update_cmp_file_b(path):
    if not path:
        return "", []
    try:
        return path, _get_file_vars(path)
    except Exception:
        return "", []


@app.callback(
    Output("file-info-tooltip", "children"),
    Input("active-file", "data"),
)
def update_file_info_tooltip(path):
    if not path:
        return "No file open"
    try:
        fmt = "Zarr" if _detect_format(path) == "zarr" else "NetCDF"
        size_mb = _get_dir_size_mb(path) if os.path.isdir(path) else os.path.getsize(path) / (1024 * 1024)
        return f"Type: {fmt}  ·  {size_mb:.1f} MB\n{path}"
    except Exception:
        return path


@app.callback(
    Output("cmp-info-a-tooltip", "children"),
    Input("cmp-file-a-select", "value"),
)
def update_cmp_info_a(path):
    if not path:
        return "No file selected"
    try:
        fmt = "Zarr" if _detect_format(path) == "zarr" else "NetCDF"
        size_mb = _get_dir_size_mb(path) if os.path.isdir(path) else os.path.getsize(path) / (1024 * 1024)
        return f"Type: {fmt}  ·  {size_mb:.1f} MB\n{path}"
    except Exception:
        return path or "No file selected"


@app.callback(
    Output("cmp-info-b-tooltip", "children"),
    Input("cmp-file-b-select", "value"),
)
def update_cmp_info_b(path):
    if not path:
        return "No file selected"
    try:
        fmt = "Zarr" if _detect_format(path) == "zarr" else "NetCDF"
        size_mb = _get_dir_size_mb(path) if os.path.isdir(path) else os.path.getsize(path) / (1024 * 1024)
        return f"Type: {fmt}  ·  {size_mb:.1f} MB\n{path}"
    except Exception:
        return path or "No file selected"


@app.callback(
    Output("cmp-mapping", "data"),
    Input("cmp-automatch-btn", "n_clicks"),
    State("cmp-file-a-vars", "data"),
    State("cmp-file-b-vars", "data"),
    prevent_initial_call=True,
)
def auto_match(_n, vars_a, vars_b):
    if not vars_a or not vars_b:
        return dash.no_update
    return _auto_match(vars_a, vars_b)


@app.callback(
    Output("cmp-bulk-confirm-btn", "disabled"),
    Output("cmp-confirm-all-btn", "disabled"),
    Input("cmp-mapping", "data"),
)
def toggle_confirm_buttons(mapping):
    disabled = not bool(mapping)
    return disabled, disabled


@app.callback(
    Output("cmp-mapping", "data", allow_duplicate=True),
    Input("cmp-bulk-confirm-btn", "n_clicks"),
    State("cmp-mapping", "data"),
    prevent_initial_call=True,
)
def bulk_confirm_high(_n, mapping):
    if not mapping:
        return dash.no_update
    updated = []
    for row in mapping:
        if row.get("confidence") == "high" and row.get("path_a") and row.get("path_b"):
            updated.append({**row, "status": "confirmed"})
        else:
            updated.append(row)
    return updated


@app.callback(
    Output("cmp-mapping", "data", allow_duplicate=True),
    Input("cmp-confirm-all-btn", "n_clicks"),
    State("cmp-mapping", "data"),
    prevent_initial_call=True,
)
def confirm_all(_n, mapping):
    if not mapping:
        return dash.no_update
    return [{**row, "status": "confirmed"} if row.get("path_a") and row.get("path_b")
            else row for row in mapping]


@app.callback(
    Output("cmp-mapping", "data", allow_duplicate=True),
    Input("cmp-unified-table", "data"),
    State("cmp-mapping", "data"),
    prevent_initial_call=True,
)
def save_mapping_edits(table_data, current_mapping):
    if not table_data or not current_mapping:
        return dash.no_update
    # Update path_b and status from table edits
    table_by_path_a = {r["path_a"]: r for r in table_data if r.get("path_a")}
    updated = []
    for row in current_mapping:
        key = row.get("path_a", "")
        if key and key in table_by_path_a:
            t = table_by_path_a[key]
            updated.append({
                **row,
                "path_b": t.get("path_b", row.get("path_b", "")),
                "status": t.get("status", row.get("status", "pending")),
                "tolerance": t.get("tolerance", row.get("tolerance", "rel:1e-7")),
            })
        else:
            updated.append(row)
    return updated


@app.callback(
    Output("cmp-mapping-area", "children"),
    Input("cmp-mapping", "data"),
    Input("cmp-results", "data"),
    Input("cmp-mapping-filter", "value"),
    State("cmp-file-b-vars", "data"),
    prevent_initial_call=True,
)
def render_unified_table(mapping, results, filt, vars_b):
    if not mapping:
        return html.Span("Load both files and click Auto-match.", className="text-muted small")

    # Build results lookup by (path_a, path_b)
    results_by_key: dict = {}
    if results:
        for r in results:
            results_by_key[(r["path_a"], r["path_b"])] = r

    b_path_opts = [{"label": v["path"], "value": v["path"]} for v in (vars_b or [])]
    b_path_opts = [{"label": "— unmatched —", "value": ""}] + b_path_opts

    rows = mapping
    if filt and filt != "all":
        rows = [r for r in mapping if r.get("status") == filt]
    if not rows:
        return html.Span(f"No rows with status '{filt}'.", className="text-muted small")

    def _fmt(v):
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.4g}"
        if isinstance(v, bool):
            return "✓" if v else "✗"
        return str(v)

    table_data = []
    for r in rows:
        key = (r.get("path_a", ""), r.get("path_b", ""))
        res = results_by_key.get(key)
        table_data.append({
            "path_a": r.get("path_a", ""),
            "shape_a": r.get("shape_a", ""),
            "path_b": r.get("path_b", ""),
            "shape_b": r.get("shape_b", ""),
            "confidence": r.get("confidence", "none"),
            "status": r.get("status", "pending"),
            "tolerance": r.get("tolerance", ""),
            "shape_match": _fmt(res["shape_match"]) if res else "",
            "units_match": _fmt(res["units_match"]) if res else "",
            "rmse": _fmt(res["rmse"]) if res else "",
            "max_abs_diff": _fmt(res["max_abs_diff"]) if res else "",
            "nan_delta": _fmt(res["nan_delta"]) if res else "",
            "tol_status": (res.get("tol_status") or "") if res else "",
            "n_perfect": _fmt(res["n_perfect"]) if res else "",
            "n_within": _fmt(res["n_within"]) if res else "",
            "n_outside": _fmt(res["n_outside"]) if res else "",
            "warnings": _build_warnings(res) if res else "",
        })

    return dash_table.DataTable(
        id="cmp-unified-table",
        data=table_data,
        columns=[
            {"name": "Variable A", "id": "path_a", "editable": False},
            {"name": "Shape A", "id": "shape_a", "editable": False},
            {"name": "Variable B", "id": "path_b", "editable": True,
             "presentation": "dropdown"},
            {"name": "Shape B", "id": "shape_b", "editable": False},
            {"name": "Confidence", "id": "confidence", "editable": False},
            {"name": "Status", "id": "status", "editable": True,
             "presentation": "dropdown"},
            {"name": "Tolerance", "id": "tolerance", "editable": True},
            {"name": "Shape ✓", "id": "shape_match", "editable": False},
            {"name": "Units ✓", "id": "units_match", "editable": False},
            {"name": "RMSE", "id": "rmse", "editable": False},
            {"name": "Max |diff|", "id": "max_abs_diff", "editable": False},
            {"name": "NaN Δ", "id": "nan_delta", "editable": False},
            {"name": "# perfect", "id": "n_perfect", "editable": False},
            {"name": "# within", "id": "n_within", "editable": False},
            {"name": "# outside", "id": "n_outside", "editable": False},
            {"name": "Tol. status", "id": "tol_status", "editable": False},
            {"name": "Warnings", "id": "warnings", "editable": False},
        ],
        dropdown={
            "path_b": {"options": b_path_opts, "clearable": True},
            "status": {"options": [
                {"label": "pending", "value": "pending"},
                {"label": "confirmed", "value": "confirmed"},
                {"label": "skipped", "value": "skipped"},
            ]},
        },
        style_table={"overflowX": "auto", "overflowY": "auto",
                     "maxHeight": "420px", "fontSize": "12px"},
        fixed_rows={"headers": True},
        style_cell={"fontFamily": "monospace", "fontSize": "11px",
                    "padding": "4px 8px", "textAlign": "left", "minWidth": "60px"},
        style_cell_conditional=[
            {"if": {"column_id": "path_a"}, "minWidth": "160px", "maxWidth": "280px"},
            {"if": {"column_id": "path_b"}, "minWidth": "160px", "maxWidth": "280px"},
            {"if": {"column_id": "shape_a"}, "minWidth": "70px", "maxWidth": "100px"},
            {"if": {"column_id": "shape_b"}, "minWidth": "70px", "maxWidth": "100px"},
            {"if": {"column_id": "confidence"}, "minWidth": "80px", "maxWidth": "90px"},
            {"if": {"column_id": "status"}, "minWidth": "90px", "maxWidth": "110px"},
            {"if": {"column_id": "tolerance"}, "minWidth": "90px", "maxWidth": "130px"},
            {"if": {"column_id": "shape_match"}, "minWidth": "65px", "maxWidth": "75px",
             "textAlign": "center"},
            {"if": {"column_id": "units_match"}, "minWidth": "65px", "maxWidth": "75px",
             "textAlign": "center"},
            {"if": {"column_id": "rmse"}, "minWidth": "80px", "maxWidth": "110px"},
            {"if": {"column_id": "max_abs_diff"}, "minWidth": "90px", "maxWidth": "120px"},
            {"if": {"column_id": "nan_delta"}, "minWidth": "70px", "maxWidth": "90px"},
            {"if": {"column_id": "tol_status"}, "minWidth": "80px", "maxWidth": "90px",
             "textAlign": "center"},
            {"if": {"column_id": "n_perfect"}, "minWidth": "70px", "maxWidth": "90px",
             "textAlign": "right"},
            {"if": {"column_id": "n_within"}, "minWidth": "70px", "maxWidth": "90px",
             "textAlign": "right"},
            {"if": {"column_id": "n_outside"}, "minWidth": "70px", "maxWidth": "90px",
             "textAlign": "right"},
            {"if": {"column_id": "warnings"}, "minWidth": "120px"},
        ],
        style_header={"fontWeight": "bold", "background": "#f4f0fa", "fontSize": "11px"},
        style_data_conditional=(
            [{"if": {"filter_query": f'{{confidence}} = "{k}"', "column_id": "confidence"},
              "color": v, "fontWeight": "700"} for k, v in _CONF_COLORS.items()]
            + [{"if": {"filter_query": f'{{status}} = "{k}"', "column_id": "status"},
                "color": v, "fontWeight": "700"} for k, v in _STATUS_COLORS.items()]
            + [{"if": {"filter_query": '{tol_status} = "perfect"'}, "backgroundColor": "#d1f5e0"},  # green
               {"if": {"filter_query": '{tol_status} = "within"'}, "backgroundColor": "#fff9c4"},   # yellow
               {"if": {"filter_query": '{tol_status} = "outside"'}, "backgroundColor": "#ffe0b2"},  # orange
               {"if": {"filter_query": '{shape_match} = "✗"'}, "backgroundColor": "#ffd0a0"},       # darker orange — shape mismatch
               {"if": {"filter_query": '{warnings} != ""'}, "backgroundColor": "#ffb74d",              # dark orange — error
                "color": "#5d2b00"},
               # tol_status text colour
               {"if": {"filter_query": '{tol_status} = "perfect"', "column_id": "tol_status"},
                "color": "#198754", "fontWeight": "700"},
               {"if": {"filter_query": '{tol_status} = "within"', "column_id": "tol_status"},
                "color": "#b38600", "fontWeight": "700"},
               {"if": {"filter_query": '{tol_status} = "outside"', "column_id": "tol_status"},
                "color": "#dc3545", "fontWeight": "700"},
               {"if": {"row_index": "odd", "filter_query": '{tol_status} = ""'},
                "backgroundColor": "#fafafa"},
               {"if": {"state": "active", "column_id": "path_b"},
                "backgroundColor": "#e3f2fd", "border": "1px solid #90caf9"},
               {"if": {"state": "active", "column_id": "status"},
                "backgroundColor": "#e3f2fd", "border": "1px solid #90caf9"},
               {"if": {"state": "active", "column_id": "tolerance"},
                "backgroundColor": "#e3f2fd", "border": "1px solid #90caf9"}]
        ),
        page_action="none",
        sort_action="native",
    )


@app.callback(
    Output("cmp-results", "data"),
    Output("cmp-run-status", "children"),
    Input("cmp-run-btn", "n_clicks"),
    State("cmp-mapping", "data"),
    State("cmp-file-a-path", "data"),
    State("cmp-file-b-path", "data"),
    State("cmp-slice-a", "value"),
    State("cmp-slice-b", "value"),
    prevent_initial_call=True,
)
def run_comparison(_n, mapping, file_a, file_b, slice_a_str, slice_b_str):
    if not mapping or not file_a or not file_b:
        return dash.no_update, "Load both files and confirm variable pairs first."
    confirmed = [r for r in mapping if r.get("status") == "confirmed"
                 and r.get("path_a") and r.get("path_b")]
    if not confirmed:
        return [], "No confirmed pairs to compare. Confirm pairs in the mapping table."
    sl_a = _parse_slice(slice_a_str or "")
    sl_b = _parse_slice(slice_b_str or "")
    results = [_compare_pair(file_a, r["path_a"], file_b, r["path_b"],
                             tolerance=r.get("tolerance", "rel:1e-7"),
                             slice_a=sl_a, slice_b=sl_b) for r in confirmed]
    n_ok = sum(1 for r in results if r["error"] is None and r["shape_match"])
    n_err = sum(1 for r in results if r["error"] is not None)
    slice_note = ""
    if sl_a or sl_b:
        slice_note = f" (A slice: {slice_a_str or '—'}, B slice: {slice_b_str or '—'})"
    return results, f"Done — {len(results)} pairs compared, {n_ok} shape-matched, {n_err} errors.{slice_note}"


@app.callback(
    Output("cmp-detail-panel", "children"),
    Output("cmp-detail-controls", "style"),
    Input("cmp-unified-table", "active_cell"),
    Input("cmp-log-scale", "value"),
    Input("cmp-apply-scale", "value"),
    Input("cmp-detail-mode", "value"),
    State("cmp-unified-table", "data"),
    State("cmp-results", "data"),
    State("cmp-file-a-path", "data"),
    State("cmp-file-b-path", "data"),
    State("cmp-slice-a", "value"),
    State("cmp-slice-b", "value"),
    prevent_initial_call=True,
)
def render_detail_panel(active_cell, log_scale, apply_scale, detail_mode, table_data, results, file_a, file_b,
                        slice_a_str, slice_b_str):
    _hide = {"display": "none"}
    _show = {"display": "flex", "alignItems": "center",
             "padding": "8px 0 4px 0", "borderTop": "1px solid #dee2e6", "marginTop": "8px"}
    if not active_cell or not results or not table_data:
        return html.Div(), _hide
    # Look up the clicked row's (path_a, path_b) and find in results
    clicked = table_data[active_cell["row"]]
    key = (clicked.get("path_a", ""), clicked.get("path_b", ""))
    row = next((r for r in results if r["path_a"] == key[0] and r["path_b"] == key[1]), None)
    if row is None:
        return html.Div(), _hide  # row not yet compared
    if row.get("error"):
        return dbc.Alert(f"Error for this pair: {row['error']}", color="danger", class_name="mt-2"), _show

    sl_a = _parse_slice(slice_a_str or "")
    sl_b = _parse_slice(slice_b_str or "")
    fig_overlay, fig_diff, warn = _make_compare_figures(
        file_a, row["path_a"], file_b, row["path_b"], slice_a=sl_a, slice_b=sl_b)

    # Attribute diff table
    try:
        _, attrs_a = _load_var_data(file_a, row["path_a"])
        _, attrs_b = _load_var_data(file_b, row["path_b"])
    except Exception:
        attrs_a, attrs_b = {}, {}

    all_keys = sorted(set(attrs_a) | set(attrs_b))
    attr_rows = [{
        "attr": k,
        "value_a": str(attrs_a.get(k, "—")),
        "value_b": str(attrs_b.get(k, "—")),
        "match": "✓" if attrs_a.get(k) == attrs_b.get(k) else "✗",
    } for k in all_keys]

    label_a = os.path.basename(file_a.rstrip("/\\"))
    label_b = os.path.basename(file_b.rstrip("/\\"))

    stats_rows = [
        ("Shape A / B", f"{row['shape_a']}  /  {row['shape_b']}"),
        ("RMSE", str(row["rmse"]) if row["rmse"] is not None else "—"),
        ("Max |diff|", str(row["max_abs_diff"]) if row["max_abs_diff"] is not None else "—"),
        ("NaN A / B", f"{row['nan_a']}  /  {row['nan_b']}"),
        ("Units A / B", f"{row['units_a'] or '—'}  /  {row['units_b'] or '—'}"),
    ]

    # Build data table view (values side-by-side)
    try:
        data_a, _ = _load_var_data(file_a, row["path_a"], apply_scale=bool(apply_scale), time_slice=sl_a)
        data_b, _ = _load_var_data(file_b, row["path_b"], apply_scale=bool(apply_scale), time_slice=sl_b)
        if log_scale:
            data_a = np.log10(np.abs(data_a) + 1e-30)
            data_b = np.log10(np.abs(data_b) + 1e-30)
        flat_a = data_a.ravel()
        flat_b = data_b.ravel() if data_b.shape == data_a.shape else None
        _MAX_ROWS = 500
        data_table_rows = []
        dtype_a = data_a.dtype
        def _fmtv(v: float, dtype: np.dtype) -> str:
            if v != v:  # NaN
                return "NaN"
            if np.issubdtype(dtype, np.integer):
                return str(int(v))
            if dtype == np.float32:
                return f"{v:.8g}"
            return f"{v:.15g}"  # float64: full precision
        for i in range(min(len(flat_a), _MAX_ROWS)):
            va = float(flat_a[i])
            vb = float(flat_b[i]) if flat_b is not None else None
            diff = (va - vb) if (vb is not None and not (va != va or vb != vb)) else None
            data_table_rows.append({
                "idx": str(i),
                "val_a": _fmtv(va, dtype_a),
                "val_b": _fmtv(vb, dtype_a) if vb is not None else "—",
                "diff": f"{diff:.6g}" if diff is not None else "—",
            })
    except Exception:
        data_table_rows = []

    data_table = dash_table.DataTable(
        data=data_table_rows,
        columns=[
            {"name": "Index", "id": "idx"},
            {"name": f"A: {row['path_a']}", "id": "val_a"},
            {"name": f"B: {row['path_b']}", "id": "val_b"},
            {"name": "A − B", "id": "diff"},
        ],
        style_table={"overflowX": "auto", "overflowY": "auto", "maxHeight": "320px"},
        style_cell={"fontFamily": "monospace", "fontSize": "11px", "padding": "3px 8px"},
        style_header={"fontWeight": "bold", "background": "#f4f0fa", "fontSize": "11px"},
        style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#fafafa"}],
        page_action="none",
        fixed_rows={"headers": True},
    ) if data_table_rows else html.Span("Data not available.", className="text-muted small")

    return dbc.Card([
        dbc.CardHeader(html.Small(
            f"Detail: {row['path_a']}  vs  {row['path_b']}",
            className="fw-semibold text-muted",
        )),
        dbc.CardBody([
            html.Div([
                html.Span([
                    html.Span(f"{k}: ", style={"color": "#888", "fontSize": "10px"}),
                    html.Span(v, style={"fontFamily": "monospace", "fontSize": "11px",
                                        "marginRight": "20px"}),
                ]) for k, v in stats_rows
            ], style={"marginBottom": "10px"}),
            warn and dbc.Alert(warn, color="warning", class_name="py-1 small") or html.Div(),
            # Plot view
            html.Div([
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig_overlay, style={"height": "300px"},
                                      config={"displaylogo": False}) if fig_overlay else html.Div(),
                            width=7),
                    dbc.Col(dcc.Graph(figure=fig_diff, style={"height": "300px"},
                                      config={"displaylogo": False}) if fig_diff else html.Div(),
                            width=5),
                ], class_name="g-1 mb-2"),
                html.Div([
                    html.Small(f"A = {label_a} / {row['path_a']}",
                               style={"color": "#888", "fontFamily": "monospace", "fontSize": "10px"}),
                    html.Br(),
                    html.Small(f"B = {label_b} / {row['path_b']}",
                               style={"color": "#888", "fontFamily": "monospace", "fontSize": "10px"}),
                ], className="mb-2"),
                dash_table.DataTable(
                    data=attr_rows,
                    columns=[
                        {"name": "Attribute", "id": "attr"},
                        {"name": f"A ({label_a})", "id": "value_a"},
                        {"name": f"B ({label_b})", "id": "value_b"},
                        {"name": "Match", "id": "match"},
                    ],
                    style_table={"overflowX": "auto", "maxHeight": "200px", "overflowY": "auto"},
                    style_cell={"fontFamily": "monospace", "fontSize": "10px", "padding": "3px 6px"},
                    style_header={"fontWeight": "bold", "background": "#f4f0fa", "fontSize": "10px"},
                    style_data_conditional=[
                        {"if": {"filter_query": '{match} = "✗"'}, "backgroundColor": "#fff3cd"},
                        {"if": {"row_index": "odd"}, "backgroundColor": "#fafafa"},
                    ],
                    page_action="none",
                    fixed_rows={"headers": True},
                ) if attr_rows else html.Span("No attributes.", className="text-muted small"),
            ], id="cmp-detail-plot-view", style={"display": "" if detail_mode == "plot" else "none"}),
            # Table view
            html.Div(data_table, id="cmp-detail-table-view", style={"display": "" if detail_mode == "table" else "none"}),
        ], class_name="py-2 px-3"),
    ], class_name="border-0 shadow-sm mt-2"), _show


@app.callback(
    Output("cmp-download-report", "data"),
    Input("cmp-report-btn", "n_clicks"),
    State("cmp-results", "data"),
    State("cmp-mapping", "data"),
    State("cmp-file-a-path", "data"),
    State("cmp-file-b-path", "data"),
    State("cmp-file-b-vars", "data"),
    prevent_initial_call=True,
)
def download_html_report(_n_clicks, results, mapping, file_a, file_b, file_b_vars):
    if not results:
        return dash.no_update
    label_a = os.path.basename((file_a or "").rstrip("/\\"))
    label_b = os.path.basename((file_b or "").rstrip("/\\"))
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"comparison_{label_a}_vs_{label_b}_{ts}.html"
    ua, ub = _compute_unmatched(mapping, file_b_vars)
    return dcc.send_string(_build_html_report(results, file_a, file_b, ua, ub), filename)


@app.callback(
    Output("cmp-download-csv", "data"),
    Input("cmp-csv-btn", "n_clicks"),
    State("cmp-results", "data"),
    State("cmp-mapping", "data"),
    State("cmp-file-a-path", "data"),
    State("cmp-file-b-path", "data"),
    State("cmp-file-b-vars", "data"),
    prevent_initial_call=True,
)
def download_csv_report(_n_clicks, results, mapping, file_a, file_b, file_b_vars):
    if not results:
        return dash.no_update
    label_a = os.path.basename((file_a or "").rstrip("/\\"))
    label_b = os.path.basename((file_b or "").rstrip("/\\"))
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"comparison_{label_a}_vs_{label_b}_{ts}.csv"
    ua, ub = _compute_unmatched(mapping, file_b_vars)
    return dcc.send_string(_build_csv_report(results, file_a, file_b, ua, ub), filename)


def _compute_unmatched(mapping: list, file_b_vars: list) -> tuple[list[str], list[str]]:
    """Return (unmatched_a, unmatched_b) variable path lists from a mapping + file B var list."""
    confirmed_a: set[str] = set()
    confirmed_b: set[str] = set()
    for row in (mapping or []):
        if row.get("status") == "confirmed" and row.get("path_b"):
            confirmed_a.add(row["path_a"])
            confirmed_b.add(row["path_b"])
    unmatched_a = [row["path_a"] for row in (mapping or [])
                   if row.get("path_a") and row["path_a"] not in confirmed_a]
    b_paths = {v["path"] for v in (file_b_vars or [])}
    unmatched_b = sorted(b_paths - confirmed_b)
    return unmatched_a, unmatched_b


def _build_csv_report(results: list, file_a: str, file_b: str,
                      unmatched_a: list | None = None, unmatched_b: list | None = None) -> str:
    """Build a CSV comparison report string from a list of _compare_pair result dicts."""
    import csv as _csv
    import io as _io
    out = _io.StringIO()
    writer = _csv.writer(out)
    writer.writerow(["file_a", "file_b", "generated"])
    writer.writerow([file_a, file_b, datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")])
    writer.writerow([])
    writer.writerow(["variable_a", "variable_b", "shape_a", "shape_b", "shape_match",
                     "units_a", "units_b", "units_match",
                     "rmse", "max_abs_diff", "nan_delta",
                     "n_perfect", "n_within", "n_outside", "tol_status", "warnings"])

    def _fmt(v):
        return "" if v is None else (f"{v:.6g}" if isinstance(v, float) else str(v))

    for r in results:
        writer.writerow([
            r["path_a"], r["path_b"],
            r.get("shape_a", ""), r.get("shape_b", ""),
            "yes" if r.get("shape_match") else "no",
            r.get("units_a", ""), r.get("units_b", ""),
            "yes" if r.get("units_match") else "no",
            _fmt(r.get("rmse")), _fmt(r.get("max_abs_diff")), _fmt(r.get("nan_delta")),
            _fmt(r.get("n_perfect")), _fmt(r.get("n_within")), _fmt(r.get("n_outside")),
            r.get("tol_status", ""), _build_warnings(r),
        ])

    for path in (unmatched_a or []):
        writer.writerow([path, "", "", "", "", "", "", "", "", "", "", "", "", "", "unmatched_a", ""])
    for path in (unmatched_b or []):
        writer.writerow(["", path, "", "", "", "", "", "", "", "", "", "", "", "", "unmatched_b", ""])

    return out.getvalue()


def _build_html_report(results: list, file_a: str, file_b: str,
                       unmatched_a: list | None = None, unmatched_b: list | None = None) -> str:
    """Build the HTML comparison report string from a list of _compare_pair result dicts."""
    label_a = os.path.basename((file_a or "").rstrip("/\\"))
    label_b = os.path.basename((file_b or "").rstrip("/\\"))
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    parts = [
        "<!DOCTYPE html><html><head>",
        "<meta charset='utf-8'>",
        f"<title>Comparison: {label_a} vs {label_b}</title>",
        "<style>body{font-family:monospace;margin:24px;background:#fff}"
        "h1{font-size:16px}h2{font-size:13px;margin-top:28px;border-bottom:1px solid #dee2e6}"
        "table{border-collapse:collapse;font-size:11px;width:100%}"
        "th,td{border:1px solid #dee2e6;padding:4px 8px;text-align:left}"
        "th{background:#f4f0fa}.ok{color:#198754}.warn{color:#fd7e14}.err{color:#dc3545}"
        ".unmatched{color:#6c757d}"
        "</style></head><body>",
        "<h1>File comparison report</h1>",
        f"<p><b>File A:</b> {file_a}<br><b>File B:</b> {file_b}<br>"
        f"<b>Generated:</b> {ts} — {len(results)} pairs compared</p>",
        "<table><thead><tr><th>Variable A</th><th>Variable B</th><th>Shape match</th>"
        "<th>Units match</th><th>RMSE</th><th>Max |diff|</th><th>NaN Δ</th><th>Tol. status</th><th>Warnings</th></tr></thead><tbody>",
    ]

    def _fmt(v):
        return "—" if v is None else (f"{v:.6g}" if isinstance(v, float) else str(v))

    for r in results:
        cls = "err" if r.get("error") else ("ok" if r["shape_match"] else "warn")
        parts.append(
            f"<tr class='{cls}'>"
            f"<td>{r['path_a']}</td><td>{r['path_b']}</td>"
            f"<td>{'✓' if r['shape_match'] else '✗'}</td>"
            f"<td>{'✓' if r['units_match'] else '✗'}</td>"
            f"<td>{_fmt(r['rmse'])}</td><td>{_fmt(r['max_abs_diff'])}</td>"
            f"<td>{_fmt(r['nan_delta'])}</td>"
            f"<td>{r.get('tol_status', '—')}</td>"
            f"<td>{_build_warnings(r)}</td></tr>"
        )
    parts.append("</tbody></table>")

    if unmatched_a:
        parts.append(f"<h2>Unmatched in File A ({len(unmatched_a)})</h2>"
                     "<table><thead><tr><th>Variable</th></tr></thead><tbody>")
        for p in unmatched_a:
            parts.append(f"<tr class='unmatched'><td>{p}</td></tr>")
        parts.append("</tbody></table>")

    if unmatched_b:
        parts.append(f"<h2>Unmatched in File B ({len(unmatched_b)})</h2>"
                     "<table><thead><tr><th>Variable</th></tr></thead><tbody>")
        for p in unmatched_b:
            parts.append(f"<tr class='unmatched'><td>{p}</td></tr>")
        parts.append("</tbody></table>")

    first_fig = True
    for r in results:
        if r.get("error") or not r["shape_match"]:
            continue
        parts.append(f"<h2>{r['path_a']}  vs  {r['path_b']}</h2>")
        try:
            fig_overlay, fig_diff, _ = _make_compare_figures(file_a, r["path_a"], file_b, r["path_b"])
            for fig, title in [(fig_overlay, f"{r['path_a']}  vs  {r['path_b']}"),
                               (fig_diff, "A − B")]:
                if fig:
                    fig.update_layout(title=dict(text=title, font=dict(size=12)),
                                      height=300, margin=dict(l=40, r=10, t=40, b=40))
                    include_js = "cdn" if first_fig else False
                    parts.append(pio.to_html(fig, full_html=False, include_plotlyjs=include_js))
                    first_fig = False
        except Exception as exc:
            parts.append(f"<p class='err'>Could not generate figures: {exc}</p>")

    parts.append("</body></html>")
    return "".join(parts)


# ---- Mapping export ----

@app.callback(
    Output("cmp-download-mapping", "data"),
    Input("cmp-export-mapping-btn", "n_clicks"),
    State("cmp-mapping", "data"),
    State("cmp-slice-a", "value"),
    State("cmp-slice-b", "value"),
    prevent_initial_call=True,
)
def export_mapping(_n, mapping, slice_a, slice_b):
    if not mapping:
        return dash.no_update
    import json as _json
    payload = {"version": "1", "mapping": mapping,
               "slice_a": slice_a or "", "slice_b": slice_b or ""}
    filename = f"variable_mapping_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    return dcc.send_string(_json.dumps(payload, indent=2), filename)


# ---- Mapping import ----

@app.callback(
    Output("cmp-mapping", "data", allow_duplicate=True),
    Output("cmp-import-status", "children"),
    Output("cmp-slice-a", "value"),
    Output("cmp-slice-b", "value"),
    Input("cmp-import-mapping-upload", "contents"),
    State("cmp-file-a-vars", "data"),
    State("cmp-file-b-vars", "data"),
    prevent_initial_call=True,
)
def import_mapping(contents, vars_a, vars_b):
    no = dash.no_update
    if not contents:
        return no, "", no, no
    import json as _json
    import base64 as _b64
    try:
        _header, _data = contents.split(",", 1)
        payload = _json.loads(_b64.b64decode(_data).decode("utf-8"))
        mapping = payload.get("mapping") if isinstance(payload, dict) else payload
        if not isinstance(mapping, list):
            return no, "Invalid file.", no, no
    except Exception as exc:
        return no, f"Error: {exc}", no, no

    slice_a = payload.get("slice_a", "") if isinstance(payload, dict) else ""
    slice_b = payload.get("slice_b", "") if isinstance(payload, dict) else ""

    # Warn about unmatched paths
    a_paths = {v["path"] for v in (vars_a or [])}
    b_paths = {v["path"] for v in (vars_b or [])}
    missing = sum(
        1 for r in mapping
        if (r.get("path_a") and r["path_a"] not in a_paths)
        or (r.get("path_b") and r["path_b"] not in b_paths)
    )
    status = f"Loaded {len(mapping)} rows." + (f" {missing} paths not found in current files." if missing else "")
    return mapping, status, slice_a or no, slice_b or no


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _cli_explore(args: "argparse.Namespace") -> None:
    """CLI: print variable tree and optionally values for a zarr/NC file."""
    import json as _json
    path = args.path
    fmt = _detect_format(path)
    vars_info = _get_file_vars(path)

    if args.var:
        # Single variable
        matches = [v for v in vars_info if v["path"] == args.var]
        if not matches:
            print(f"Variable not found: {args.var}", file=sys.stderr)
            sys.exit(1)
        v = matches[0]
        print(f"path:   {v['path']}")
        print(f"shape:  {v['shape']}")
        print(f"dtype:  {v['dtype']}")
        if v.get("units"):
            print(f"units:  {v['units']}")
        if v.get("long_name"):
            print(f"long_name: {v['long_name']}")
        if args.values:
            sl = _parse_slice(args.slice or "") if args.slice else None
            data, _ = _load_var_data(path, args.var, apply_scale=not args.raw, time_slice=sl)
            print(f"values:\n{data}")
    else:
        # Full tree
        col_w = max((len(v["path"]) for v in vars_info), default=20)
        header = f"{'path':<{col_w}}  {'shape':<18}  {'dtype':<10}  units"
        print(header)
        print("─" * len(header))
        for v in vars_info:
            print(f"{v['path']:<{col_w}}  {v['shape']:<18}  {v['dtype']:<10}  {v.get('units', '')}")


def _cli_compare(args: "argparse.Namespace") -> None:
    """CLI: run comparison using a mapping JSON and produce an HTML report."""
    import json as _json

    # Load mapping
    with open(args.mapping, "r", encoding="utf-8") as fh:
        payload = _json.load(fh)
    mapping = payload.get("mapping") if isinstance(payload, dict) else payload
    slice_a_str = payload.get("slice_a", "") if isinstance(payload, dict) else ""
    slice_b_str = payload.get("slice_b", "") if isinstance(payload, dict) else ""
    sl_a = _parse_slice(slice_a_str)
    sl_b = _parse_slice(slice_b_str)

    confirmed = [r for r in mapping if r.get("status") == "confirmed" and r.get("path_b")]
    if not confirmed:
        print("No confirmed pairs in mapping — nothing to compare.", file=sys.stderr)
        sys.exit(1)

    print(f"Comparing {len(confirmed)} confirmed pairs…")
    results = []
    for i, row in enumerate(confirmed, 1):
        tol = row.get("tolerance", "rel:1e-7") or "rel:1e-7"
        r = _compare_pair(args.file_a, row["path_a"], args.file_b, row["path_b"],
                          tolerance=tol, slice_a=sl_a, slice_b=sl_b)
        results.append(r)
        status = r.get("tol_status", "—")
        warn = _build_warnings(r)
        print(f"  [{i:>3}/{len(confirmed)}] {row['path_a']!s:<50}  {status}" +
              (f"  ⚠ {warn}" if warn else ""))

    file_b_vars = _get_file_vars(args.file_b)
    unmatched_a, unmatched_b = _compute_unmatched(mapping, file_b_vars)
    if unmatched_a or unmatched_b:
        print(f"  Unmatched: {len(unmatched_a)} in A, {len(unmatched_b)} in B")

    label_a = os.path.basename(args.file_a.rstrip("/\\"))
    label_b = os.path.basename(args.file_b.rstrip("/\\"))
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")

    n_perfect = sum(1 for r in results if r.get("tol_status") == "perfect")
    n_within = sum(1 for r in results if r.get("tol_status") == "within")
    n_outside = sum(1 for r in results if r.get("tol_status") == "outside")
    n_err = sum(1 for r in results if r.get("error"))
    print(f"\nDone: {n_perfect} perfect  {n_within} within  {n_outside} outside  {n_err} errors")

    # Default to HTML if neither flag given
    write_html = args.html is not None or (args.csv is None)
    write_csv = args.csv is not None

    if write_html:
        html_path = (args.html if args.html else None) or f"comparison_{label_a}_vs_{label_b}_{ts}.html"
        with open(html_path, "w", encoding="utf-8") as fh:
            fh.write(_build_html_report(results, args.file_a, args.file_b, unmatched_a, unmatched_b))
        print(f"HTML report saved to: {html_path}")

    if write_csv:
        csv_path = args.csv if args.csv else f"comparison_{label_a}_vs_{label_b}_{ts}.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as fh:
            fh.write(_build_csv_report(results, args.file_a, args.file_b, unmatched_a, unmatched_b))
        print(f"CSV  report saved to: {csv_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="zarr_explorer.py",
        description=f"Zarr Explorer v{__version__} — viewer and comparator for Zarr/NetCDF files",
    )
    sub = parser.add_subparsers(dest="command")

    # ---- explore subcommand ----
    p_explore = sub.add_parser("explore", help="Print variable tree for a file")
    p_explore.add_argument("path", help="Path to .zarr directory or NetCDF file")
    p_explore.add_argument("--var", metavar="VAR_PATH", help="Focus on a specific variable path")
    p_explore.add_argument("--values", action="store_true", help="Print array values (requires --var)")
    p_explore.add_argument("--slice", metavar="SLICE", help="Slice to apply when printing values, e.g. '0:10' or '::4'")
    p_explore.add_argument("--raw", action="store_true", help="Skip scaling (no scale_factor/add_offset/FillValue applied)")

    # ---- compare subcommand ----
    p_compare = sub.add_parser("compare", help="Run comparison using a mapping JSON file")
    p_compare.add_argument("file_a", help="Path to File A (.zarr or NetCDF)")
    p_compare.add_argument("file_b", help="Path to File B (.zarr or NetCDF)")
    p_compare.add_argument("--mapping", required=True, metavar="FILE", help="Mapping JSON (exported from the Compare tab)")
    p_compare.add_argument("--html", metavar="FILE", help="Write HTML report (auto-named if no path given)", nargs="?", const="")
    p_compare.add_argument("--csv", metavar="FILE", help="Write CSV report (auto-named if no path given)", nargs="?", const="")

    args = parser.parse_args()

    if args.command == "explore":
        _cli_explore(args)
    elif args.command == "compare":
        _cli_compare(args)
    else:
        # No subcommand → launch web app
        print(f"Starting Zarr Explorer v{__version__}...")
        print("Open your browser at: http://127.0.0.1:8050")
        try:
            threading.Timer(1.2, lambda: webbrowser.open("http://127.0.0.1:8050")).start()
        except Exception:
            pass
        app.run(debug=False, host="0.0.0.0", port=8050)

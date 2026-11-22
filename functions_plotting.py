import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# various plot levels for reactive layers of plots.
def _normalize_series_list(
    data: pd.DataFrame | list[pd.DataFrame],
    y_col: str,
    series_col: str,
    time_col: str,
) -> list[pd.DataFrame]:
    """Validates and splits input data into sorted per-series/run DataFrames."""
    if y_col not in ("value", "cumulative_value"):
        raise ValueError(f"y_col must be 'value' or 'cumulative_value', got {y_col!r}.")

    if isinstance(data, pd.DataFrame):
        if data.empty:
            raise ValueError("'data' DataFrame is empty.")
        group_col = (
            "sim_id"
            if "sim_id" in data.columns
            else (series_col if series_col in data.columns else None)
        )
        if group_col and data[group_col].nunique() > 1:
            dfs = [
                g.sort_values(time_col).copy()
                for _, g in data.groupby(group_col, sort=False)
            ]
        else:
            dfs = [data.sort_values(time_col).copy()]
    else:
        dfs = [df.sort_values(time_col).copy() for df in data if not df.empty]
        if not dfs:
            raise ValueError("'data' must contain at least one non-empty DataFrame.")

    for i, df in enumerate(dfs):
        if y_col not in df.columns:
            raise KeyError(f"Column {y_col!r} not found in DataFrame at index {i}.")

    return dfs


# set plot text
def _style_figure_typography(
    fig: go.Figure,
    format_facet_titles: bool = False,
    top_margin: int = 80,
) -> go.Figure:
    """Increases font sizes across axes, facets, and point text, and adds title spacing."""
    # Global font, title size, and top margin spacing
    fig.update_layout(
        font={"size": 14},
        title={"font": {"size": 20}, "y": 0.97, "yanchor": "top"},
        margin={"t": top_margin},
    )

    # Enlarge axis titles and tick labels
    fig.update_xaxes(
        title_font={"size": 16},
        tickfont={"size": 13},
    )
    fig.update_yaxes(
        title_font={"size": 16},
        tickfont={"size": 13},
    )

    # Enlarge point text labels
    fig.for_each_trace(
        lambda trace: (
            trace.update(textfont={"size": 13})
            if hasattr(trace, "textfont") and trace.mode and "text" in trace.mode
            else None
        )
    )

    # Enlarge facet subplot titles
    def _update_facet_annotation(ann):
        text = ann.text or ""
        if format_facet_titles:
            clean_text = text.split("=")[-1].replace("_", " ").title()
            ann.update(text=clean_text)
        ann.update(font={"size": 15})

    fig.for_each_annotation(_update_facet_annotation)
    return fig


# create base time series figure
def create_base_timeseries_fig(
    dfs: list[pd.DataFrame],
    y_col: str = "value",
    series_col: str = "series_name",
    time_col: str = "time",
) -> tuple[go.Figure, int, int]:
    """
    Builds the base faceted time series `go.Figure` without overlays or `fig.show()`.
    Returns `(fig, nrow, ncol)` so overlay functions can target the exact subplot grid.
    """
    n_plots = len(dfs)
    ncol = int(np.ceil(np.sqrt(n_plots)))
    nrow = int(np.ceil(n_plots / ncol))
    titles = [
        str(df[series_col].iloc[0])
        if series_col in df.columns
        else f"series_{i + 1:02d}"
        for i, df in enumerate(dfs)
    ]

    fig = make_subplots(
        rows=nrow,
        cols=ncol,
        subplot_titles=titles,
        vertical_spacing=0.12 if nrow > 1 else 0.05,
        horizontal_spacing=0.05,
    )

    for i, df_slice in enumerate(dfs):
        row, col = (i // ncol) + 1, (i % ncol) + 1

        # Break line segments at cycle boundaries when plotting cumulative_value
        if y_col == "cumulative_value" and "cycle_number" in df_slice.columns:
            x_vals, y_vals = [], []
            for _, cyc_grp in df_slice.groupby("cycle_number", sort=False):
                x_vals.extend(cyc_grp[time_col].tolist() + [None])
                y_vals.extend(cyc_grp[y_col].tolist() + [None])
        else:
            x_vals = df_slice[time_col].tolist()
            y_vals = df_slice[y_col].tolist()

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines+markers",
                name=titles[i],
                showlegend=False,
                connectgaps=False,
                line={"color": "#1f77b4", "width": 1},
                marker={"color": "#1f77b4", "size": 4},
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

        if row == nrow:
            fig.update_xaxes(title_text="Time", row=row, col=col)
        if col == 1:
            fig.update_yaxes(
                title_text="Cumulative Value"
                if y_col == "cumulative_value"
                else "Value",
                row=row,
                col=col,
            )

    axis_border_style = {
        "showline": True,
        "linewidth": 1,
        "linecolor": "#CCCCCC",
        "mirror": True,
        "showgrid": True,
        "gridcolor": "#F0F0F0",
    }
    fig.update_annotations(font={"size": 12})
    fig.update_xaxes(**axis_border_style)
    fig.update_yaxes(**axis_border_style)

    fig.update_layout(
        template="plotly_white",
        title=None,
        hovermode="closest",
        height=max(390, 310 * nrow),
        margin={"t": 40, "l": 55, "r": 110, "b": 45},
        legend={
            "x": 1.02,
            "y": 1.0,
            "xanchor": "left",
            "yanchor": "top",
        },
    )
    return fig, nrow, ncol


## add labelled points for identified peaks and troughs.
def add_extrema_layer(
    fig: go.Figure,
    dfs: list[pd.DataFrame],
    ncol: int,
    y_col: str = "value",
    extrema_df: pd.DataFrame | None = None,
    series_col: str = "series_name",
    time_col: str = "time",
) -> go.Figure:
    """Adds peak and trough triangle markers, hover tooltips, and explanatory subtext."""
    styles = {
        "peak": {
            "color": "crimson",
            "symbol": "triangle-up",
            "pos": "top center",
            "dist_label": "Weeks since last trough",
        },
        "trough": {
            "color": "royalblue",
            "symbol": "triangle-down",
            "pos": "bottom center",
            "dist_label": "Cycle length (weeks)",
        },
    }

    for i, df_slice in enumerate(dfs):
        row, col = (i // ncol) + 1, (i % ncol) + 1
        s_name = (
            df_slice[series_col].iloc[0] if series_col in df_slice.columns else None
        )

        if extrema_df is not None:
            pts = (
                extrema_df[extrema_df[series_col] == s_name].copy()
                if s_name is not None and series_col in extrema_df.columns
                else extrema_df.copy()
            )
            pts = pts.drop(columns=[y_col], errors="ignore").merge(
                df_slice[[time_col, y_col]], on=time_col, how="inner"
            )
        elif "type" in df_slice.columns:
            pts = df_slice[df_slice["type"].isin(["peak", "trough"])].copy()
        else:
            continue

        if pts.empty:
            continue

        pts = pts.sort_values(time_col).reset_index(drop=True)
        last_trough = pts[time_col].where(pts["type"] == "trough").ffill()
        pts["label_val"] = np.where(
            pts["type"] == "trough",
            pts["dist_from_prev"] if "dist_from_prev" in pts.columns else np.nan,
            pts[time_col] - last_trough,
        )

        for ext_type, style in styles.items():
            sub = pts[pts["type"] == ext_type]
            if sub.empty:
                continue

            labels = np.where(
                sub["label_val"].notna(),
                sub["label_val"].fillna(0).astype(int).astype(str),
                "",
            )
            hover_dists = np.where(
                sub["label_val"].notna(),
                sub["label_val"].fillna(0).astype(int).astype(str) + " wks",
                "N/A (first point)",
            )

            # Extrema marker trace with hover tooltip
            fig.add_trace(
                go.Scatter(
                    x=sub[time_col],
                    y=sub[y_col],
                    mode="markers+text",
                    name=ext_type.title(),
                    showlegend=(i == 0),
                    marker={
                        "color": style["color"],
                        "symbol": style["symbol"],
                        "size": 9,
                    },
                    text=labels,
                    textposition=style["pos"],
                    textfont={"size": 10, "color": style["color"]},
                    customdata=hover_dists,
                    hovertemplate=(
                        f"<b>{ext_type.title()}</b><br>"
                        "Time: %{x}<br>"
                        f"{y_col}: %{{y}}<br>"
                        f"{style['dist_label']}: %{{customdata}}"
                        "<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )

    return fig


# main plot time series function
def plot_timeseries(
    data: pd.DataFrame | list[pd.DataFrame],
    y_col: str = "value",
    show_extrema: bool = False,
    extrema_df: pd.DataFrame | None = None,
    series_col: str = "series_name",
    time_col: str = "time",
    show: bool = True,
) -> go.Figure:
    """High-level wrapper that composes the base plot, optional extrema overlay, and footer notes."""
    dfs = _normalize_series_list(
        data, y_col=y_col, series_col=series_col, time_col=time_col
    )
    fig, _, ncol = create_base_timeseries_fig(
        dfs, y_col=y_col, series_col=series_col, time_col=time_col
    )

    if show_extrema:
        fig = add_extrema_layer(
            fig,
            dfs,
            ncol=ncol,
            y_col=y_col,
            extrema_df=extrema_df,
            series_col=series_col,
            time_col=time_col,
        )
    if show:
        fig.show()
    fig = _style_figure_typography(fig, format_facet_titles=False, top_margin=75)
    return fig

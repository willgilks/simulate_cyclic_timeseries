import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget

from constants import (
    DEFAULT_SIM_PARAMS,
    GREY_THEME_CSS,
    INT_PARAMS,
    TIMESERIES_PLOT_SUBTEXT,
)
from functions_analysis import (
    assign_trough_to_trough_cycles,
    enhance_ts_summary_info,
    find_extrema_summary,
)
from functions_plotting import plot_timeseries
from functions_shiny import _build_param_controls, _sync_figure_widget, debounce
from functions_sim import simulate_cyclic_timeseries

# ui ####
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.div(
            ui.h5("Simulation Parameters", class_="mb-2"),
            ui.input_action_button(
                "reset_params",
                "Reset to Defaults",
                class_="btn-sm btn-outline-secondary w-100",
            ),
            class_="p-3 border-bottom flex-shrink-0",
        ),
        ui.div(
            *_build_param_controls(),
            class_="p-3 overflow-y-auto flex-grow-1",
        ),
        width=320,
        padding="0px",
        class_="d-flex flex-column h-100",
    ),
    ui.tags.style(GREY_THEME_CSS),
    ui.navset_card_tab(
        ui.nav_panel(
            "Simulated Time Series",
            output_widget("main_ts_plot"),
            ui.output_ui("ts_plot_caption"),
            ui.hr(class_="my-2"),
            ui.layout_columns(
                ui.input_radio_buttons(
                    "y_col",
                    "Y-Axis Metric",
                    choices={
                        "value": "Raw Value",
                        "cumulative_value": "Cumulative Value",
                    },
                    selected="value",
                    inline=True,
                ),
                ui.div(
                    ui.input_checkbox(
                        "show_extrema", "Show Peaks & Troughs", value=True
                    ),
                    class_="pt-4",
                ),
                col_widths=[6, 6],
            ),
        ),
        ui.nav_panel(
            "Data",
            ui.div(
                ui.div(
                    ui.input_select(
                        "selected_table",
                        "Select Table",
                        choices={
                            "enhanced": "Enhanced Simulated Time Series",
                            "raw": "Raw Simulated Time Series",
                            "extrema": "Detected Extrema",
                        },
                        selected="enhanced",
                    ),
                    style="min-width: 280px; margin-bottom: -15px;",
                ),
                ui.download_button(
                    "download_csv", "Download CSV", class_="btn-outline-primary"
                ),
                class_="d-flex align-items-end gap-3 mb-2",
            ),
            ui.output_data_frame("data_table"),
        ),
    ),
    title="Time Series Simulation with periodicity and noise",
    fillable=True,
)


# Server ###
def server(input, output, session):

    # make persistent FigureWidgets
    main_fig_widget = go.FigureWidget()

    @render.ui
    def ts_plot_caption():
        text = TIMESERIES_PLOT_SUBTEXT
        return ui.p(text, class_="text-muted fs-6 mt-2 mb-0")

    # Bind persistent widgets to ui output slots
    @render_widget
    def main_ts_plot():
        return main_fig_widget

    # Reset sliders
    @reactive.effect
    @reactive.event(input.reset_params)
    def _reset_sliders():
        for name, (default_val, *_) in DEFAULT_SIM_PARAMS.items():
            ui.update_slider(f"param_{name}", value=default_val)

    # Collect active parameters from ui sliders
    @debounce(0.25)
    @reactive.calc
    def active_sim_params() -> dict:
        return {
            name: [
                int(getattr(input, f"param_{name}")())
                if name in INT_PARAMS
                else float(getattr(input, f"param_{name}")()),
                low,
                high,
            ]
            for name, (_, low, high, _) in DEFAULT_SIM_PARAMS.items()
        }

    # Simulate time series
    @reactive.calc
    def simulated_ts() -> pd.DataFrame:
        return simulate_cyclic_timeseries(sim_params=active_sim_params())

    # Run post-analysis
    @reactive.calc
    def analysis_bundle() -> dict:
        raw_df = simulated_ts()
        try:
            extrema_df = find_extrema_summary(raw_df)
            if extrema_df.empty or set(extrema_df["type"]) != {"peak", "trough"}:
                return {"extrema": None, "enhanced": None, "estimates": None}

            with_cycles_df = assign_trough_to_trough_cycles(
                ts_df=raw_df,
                extrema_df=extrema_df,
                series_col="series_name",
                time_col="time",
                value_col="value",
            )
            enhanced_df = enhance_ts_summary_info(
                ts_with_cycles_df=with_cycles_df,
                series_col="series_name",
                time_col="time",
                value_col="value",
            )
            return {
                "extrema": extrema_df,
                "enhanced": enhanced_df,
            }

        except (
            ValueError,
            KeyError,
            IndexError,
            ZeroDivisionError,
            np.linalg.LinAlgError,
        ):
            return {"extrema": None, "enhanced": None}

    # Update Main Time Series FigureWidget
    @reactive.effect
    def _update_main_ts_plot():
        raw_df = simulated_ts()
        bundle = analysis_bundle()

        enhanced_df = bundle["enhanced"]
        plot_data = (
            enhanced_df if enhanced_df is not None and not enhanced_df.empty else raw_df
        )

        chosen_y = input.y_col() if input.y_col() in plot_data.columns else "value"
        extrema_df = bundle["extrema"]
        can_show_extrema = bool(input.show_extrema()) and (
            (extrema_df is not None and not extrema_df.empty)
            or "type" in plot_data.columns
        )

        new_fig = plot_timeseries(
            data=plot_data,
            y_col=chosen_y,
            show_extrema=can_show_extrema,
            extrema_df=extrema_df,
            series_col="series_name",
            time_col="time",
            show=False,
        )
        _sync_figure_widget(main_fig_widget, new_fig)

    # Data Tab
    # view and download selected table
    @reactive.calc
    def active_preview_df() -> pd.DataFrame:
        choice = input.selected_table()
        raw_df = simulated_ts()
        if choice == "raw":
            return raw_df

        bundle = analysis_bundle()
        selected = bundle.get(choice)
        if selected is not None and not selected.empty:
            return selected

        return raw_df

    # Render table
    @render.data_frame
    def data_table():
        return render.DataGrid(active_preview_df(), filters=True)

    # Export table as csv
    @render.download_button(
        filename=lambda: f"seasonal_sim_{input.selected_table()}.csv"
    )
    def download_csv():
        yield active_preview_df().to_csv(index=False)


app = App(app_ui, server)

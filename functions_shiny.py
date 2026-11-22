import time

import numpy as np
import plotly.graph_objects as go
from shiny import reactive, ui

from constants import DEFAULT_SIM_PARAMS, INT_PARAMS


# clean json seq. type conversion in shiny widgets
def _clean_json_seq(seq):
    """Converts any float NaN values in a trace array to None (JSON null) for shinywidgets."""
    if seq is None:
        return None
    return [None if (isinstance(v, float) and np.isnan(v)) else v for v in seq]


# sync figure widget. maintains style consistency during transition
def _sync_figure_widget(target_widget: go.FigureWidget, source_fig: go.Figure) -> None:
    """Updates a persistent FigureWidget in-place without flashing or trace style mismatch."""
    for trace in source_fig.data:
        trace.x = _clean_json_seq(trace.x)
        trace.y = _clean_json_seq(trace.y)

    with target_widget.batch_update():
        same_trace_count = (
            len(target_widget.data) == len(source_fig.data) and len(source_fig.data) > 0
        )
        same_annotations = len(target_widget.layout.annotations) == len(
            source_fig.layout.annotations
        )

        if same_trace_count and same_annotations:
            for old_trace, new_trace in zip(target_widget.data, source_fig.data):
                old_trace.x = new_trace.x
                old_trace.y = new_trace.y
                old_trace.mode = new_trace.mode
                if hasattr(new_trace, "line") and new_trace.line is not None:
                    old_trace.line = new_trace.line
                if hasattr(new_trace, "marker") and new_trace.marker is not None:
                    old_trace.marker = new_trace.marker
                if hasattr(new_trace, "text") and new_trace.text is not None:
                    old_trace.text = list(new_trace.text)
            if source_fig.layout.yaxis.title.text:
                target_widget.layout.yaxis.title.text = (
                    source_fig.layout.yaxis.title.text
                )
        else:
            target_widget.data = []
            target_widget.layout = source_fig.layout
            target_widget.add_traces(source_fig.data)


# delay transition to allow other calculations to catch up
def debounce(delay_secs: float):
    """Delays downstream invalidation until `delay_secs` have passed without changes."""

    def wrapper(f):
        deadline = reactive.Value(None)
        trigger = reactive.Value(0)

        @reactive.calc
        def cached():
            return f()

        @reactive.effect(priority=102)
        def _primer():
            cached()
            deadline.set(time.time() + delay_secs)

        @reactive.effect(priority=101)
        def _timer():
            dl = deadline()
            if dl is None:
                return
            remaining = dl - time.time()
            if remaining <= 0:
                deadline.set(None)
                with reactive.isolate():
                    trigger.set(trigger() + 1)
            else:
                reactive.invalidate_later(remaining)

        @reactive.calc
        def debounced():
            trigger()
            with reactive.isolate():
                return cached()

        return debounced

    return wrapper


# build sliders in a loop
def _build_param_controls() -> list:
    """Dynamically builds Shiny sliders with hover tooltips from DEFAULT_SIM_PARAMS."""
    controls = []
    for name, (default_val, min_val, max_val, desc) in DEFAULT_SIM_PARAMS.items():
        is_int = name in INT_PARAMS
        step = 1 if is_int else round((max_val - min_val) / 200, 4)
        label = name.replace("_", " ").title()

        slider = ui.input_slider(
            id=f"param_{name}",
            label=label,
            min=min_val,
            max=max_val,
            value=default_val,
            step=step,
            ticks=True,
        )
        controls.append(ui.tooltip(slider, desc, placement="right"))
    return controls

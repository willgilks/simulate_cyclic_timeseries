POINTS_PER_CYCLE = 52

INT_PARAMS = {"n_cycles", "standard_high_time", "standard_low_time", "seed"}


TIMESERIES_PLOT_SUBTEXT = (
    "Simulated weekly series generated from 52-week cycles. "
    + " Numbers at extrema (peaks and troughs) indicate weeks since last low point. "
    + "Peaks and troughs are identified by anchoring the first prominent extremum via "
    + "Python::scipy.signal.find_peaks and then finding the extrema within a 40-week search window, "
    + "which is adaptive to the observed cycle length."
)


# default parameters for the ui and simulation
DEFAULT_SIM_PARAMS: dict[str, tuple[int | float, int | float, int | float, str]] = {
    "n_cycles": (6, 1, 20, "Total number of seasonal cycles to simulate."),
    "standard_high_time": (
        1,
        1,
        52,
        "High week",
    ),
    "standard_low_time": (
        26,
        1,
        52,
        "Low week",
    ),
    "peak_roundness": (
        0.1,
        -3.0,
        3.0,
        "0 = straight, >0 = broad, <0 = narrow",
    ),
    "trough_roundness": (
        2.0,
        -3.0,
        3.0,
        "0 = straight, >0 = broad, <0 = narrow",
    ),
    "amplitude_noise": (
        0.1,
        0.0,
        0.2,
        "Random variation in height.",
    ),
    "timing_noise": (
        0.25,
        0.0,
        2.0,
        "Random variation in timing",
    ),
    "cycle_reversion": (
        0.0,
        0.0,
        1.0,
        "Strength at which deviations return to the previous cycle.",
    ),
    "amplitude_drift": (
        0.0,
        -0.1,
        0.1,
        "Trend in height over time.",
    ),
    "timing_drift": (
        0.0,
        -0.1,
        0.1,
        "Trend in cycle length over time.",
    ),
    "seed": (42, 1, 1000, "Random number generator seed for reproducible simulations."),
}


GREY_THEME_CSS: str = """
:root {
    --bs-primary: #8c969e;
    --bs-primary-rgb: 140, 150, 158;
    --bs-link-color: #6c757d;
    --bs-link-hover-color: #495057;
}

/* Sliders (ion.rangeSlider): selected value badge, circle handle, and active left bar */
.irs--shiny .irs-bar {
    background-color: #9aa3ab !important;
    border-top-color: #9aa3ab !important;
    border-bottom-color: #9aa3ab !important;
}
.irs--shiny .irs-single,
.irs--shiny .irs-from,
.irs--shiny .irs-to {
    background-color: #8c969e !important;
    color: #ffffff !important;
}
.irs--shiny .irs-handle {
    background-color: #8c969e !important;
    border-color: #78828a !important;
}
.irs--shiny .irs-handle:hover,
.irs--shiny .irs-handle.state_hover {
    background-color: #78828a !important;
}

/* Radio buttons & checkboxes */
.form-check-input:checked {
    background-color: #8c969e !important;
    border-color: #8c969e !important;
}
.form-check-input:focus {
    border-color: #9aa3ab !important;
    box-shadow: 0 0 0 0.2rem rgba(140, 150, 158, 0.25) !important;
}

/* Non-selected tab titles */
.nav-tabs .nav-link:not(.active),
.card-header-tabs .nav-link:not(.active) {
    color: #6c757d !important;
}
.nav-tabs .nav-link:not(.active):hover,
.card-header-tabs .nav-link:not(.active):hover {
    color: #495057 !important;
}
"""

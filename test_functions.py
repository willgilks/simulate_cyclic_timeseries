from constants import DEFAULT_SIM_PARAMS
from functions_analysis import (
    assign_trough_to_trough_cycles,
    enhance_ts_summary_info,
    find_extrema_summary,
)
from functions_plotting import plot_timeseries
from functions_sim import simulate_cyclic_timeseries

## change simulation parameters
updated_sim_params = {k: list(v) for k, v in DEFAULT_SIM_PARAMS.items()}
updated_sim_params["n_cycles"][0] = 10
updated_sim_params["timing_noise"][0] = 0.05
updated_sim_params["amplitude_noise"][0] = 0.05
updated_sim_params["cycle_reversion"][0] = 0.025
updated_sim_params["amplitude_drift"][0] = 0.0025
updated_sim_params["timing_drift"][0] = -0.0025


# simulate time series using sim parameters
simulated_ts_df = simulate_cyclic_timeseries(sim_params=updated_sim_params)

# Do post-analysis
# find high and low points.
peak_and_trough_output = find_extrema_summary(simulated_ts_df)

# assign a cycle number to the raw data based on the trough-to-trough ranges
simulated_ts_with_cycles = assign_trough_to_trough_cycles(
    ts_df=simulated_ts_df,
    extrema_df=peak_and_trough_output,
    series_col="series_name",
    time_col="time",
    value_col="value",
)

# add columns with extra info on the time series
simulated_ts_enhanced = enhance_ts_summary_info(
    ts_with_cycles_df=simulated_ts_with_cycles,
    series_col="series_name",
    time_col="time",
    value_col="value",
)


# time series plot
plot_timeseries(
    data=simulated_ts_enhanced,
    y_col="value",
    show_extrema=True,
    extrema_df=peak_and_trough_output,
    series_col="series_name",
    time_col="time",
)

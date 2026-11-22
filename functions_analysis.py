## analysis of simulated time series
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


# Find high and low points within windows
# take into account what the location of the previous peak and low point was,
# then search 20 time points either side of that for next peak and low.
# use  dist_window=20, then the actual min and max dist_window variables
# are calculated depending on the previous peak.
def find_extrema_summary(
    df: pd.DataFrame,
    expected_period: int = 52,
    dist_window: int = 20,
    prominence: float | None = None,
    series_col: str = "series_name",
    time_col: str = "time",
    value_col: str = "value",
) -> pd.DataFrame:
    """
    Finds peaks and troughs per series by stepping forward from each previous extremum
    and searching within `± dist_window` of the expected next location.
    """
    records = []

    for series_name, group in df.groupby(series_col, sort=False):
        g = group.sort_values(time_col).reset_index(drop=True)
        vals = g[value_col].interpolate().to_numpy()
        times = g[time_col].to_numpy()
        prom = prominence if prominence is not None else 0.1 * np.ptp(vals)

        prom = prominence if prominence is not None else 0.1 * np.ptp(vals)
        if prom <= 1e-9:
            continue

        for extremum_type, sign in [("peak", 1), ("trough", -1)]:
            # Seed the first extremum via `find_peaks`, then step forward by `current_period`
            min_dist = max(1, expected_period - dist_window)
            first_cands, _ = find_peaks(sign * vals, prominence=prom, distance=min_dist)

            if len(first_cands) == 0:
                continue

            indices = [int(first_cands[0])]
            period = float(expected_period)

            for _ in range(len(vals)):
                # prevent loop freeze
                win_start = max(
                    indices[-1] + 1, int(indices[-1] + period - dist_window)
                )
                if win_start >= len(vals):
                    break
                win_end = min(
                    len(vals),
                    max(win_start + 1, int(indices[-1] + period + dist_window) + 1),
                )

                best_idx = win_start + int(np.argmax(sign * vals[win_start:win_end]))
                period = max(2.0, 0.7 * period + 0.3 * (best_idx - indices[-1]))
                indices.append(best_idx)

            # Prepend NaN to np.diff so distances align 1-to-1 with indices
            dists = np.insert(np.diff(indices).astype(float), 0, np.nan)
            records.extend(
                {
                    series_col: series_name,
                    "type": extremum_type,
                    "row_index": idx,
                    time_col: times[idx],
                    value_col: vals[idx],
                    "dist_from_prev": dist,
                    "in_window": pd.notna(dist),
                }
                for idx, dist in zip(indices, dists)
            )

    return pd.DataFrame(records)


# Assign a cycle number to the raw data based on the trough-to-trough ranges
def assign_trough_to_trough_cycles(
    ts_df: pd.DataFrame,
    extrema_df: pd.DataFrame,
    series_col: str = "series_name",
    time_col: str = "time",
    value_col: str = "value",
) -> pd.DataFrame:
    """
    Joins the raw time series with detected extrema and assigns a `cycle_number`
    starting at 1, incrementing at every detected trough.
    """
    # Drop `value_col` from extrema_df dynamically so it doesn't collide during merge
    ext_subset = extrema_df.drop(columns=[value_col], errors="ignore").drop_duplicates(
        subset=[series_col, time_col]
    )

    df = ts_df.sort_values([series_col, time_col]).merge(
        ext_subset, on=[series_col, time_col], how="left"
    )

    # Vectorized cumsum per series (much faster than groupby().transform(lambda ...))
    df["_is_trough"] = df["type"].eq("trough")
    df["cycle_number"] = df.groupby(series_col, sort=False)["_is_trough"].cumsum() + 1
    df["cumulative_value"] = df.groupby([series_col, "cycle_number"], sort=False)[
        value_col
    ].cumsum()

    return df.drop(columns=["_is_trough"])


# main function to enhance ts with summary cycle level info
def enhance_ts_summary_info(
    ts_with_cycles_df: pd.DataFrame,
    series_col: str = "series_name",
    time_col: str = "time",
    value_col: str = "value",
) -> pd.DataFrame:
    """
    Enhances cycle data with cycle-level total values, filled `dist_from_prev`,
    `cycle_length`, and relative rise/fall durations (`start_to_peak_dur`, `peak_to_end`).
    """
    # Keep only the required columns
    cols_to_keep = [
        series_col,
        "date",
        time_col,
        value_col,
        "final_sim_params",
        "type",
        "dist_from_prev",
        "cycle_number",
        "cumulative_value",
    ]
    df = ts_with_cycles_df[cols_to_keep].copy()

    # Group by series and cycle
    cycle_groups = df.groupby([series_col, "cycle_number"], sort=False)

    # Normalize value within each cycle to [0, 1]
    cycle_min = cycle_groups[value_col].transform("min")
    cycle_max = cycle_groups[value_col].transform("max")
    norm_val = (df[value_col] - cycle_min) / (cycle_max - cycle_min).replace(0, np.nan)

    # Count steps spent in the high state (> 0.5) vs low state (<= 0.5) per cycle
    df["_is_high"] = (norm_val > 0.5).astype(float)
    df["_is_low"] = (norm_val <= 0.5).astype(float)

    # Total value per cycle
    df["cycle_total_value"] = cycle_groups[value_col].transform("sum")

    # Fill dist_from_prev across all rows in each cycle
    df["dist_from_prev"] = cycle_groups["dist_from_prev"].transform("first")

    # Calculate cycle_length (distance between consecutive troughs)
    cycle_meta = df.groupby([series_col, "cycle_number"], as_index=False)[
        "dist_from_prev"
    ].first()
    cycle_meta["cycle_length"] = cycle_meta.groupby(series_col)["dist_from_prev"].shift(
        -1
    )

    df = df.merge(
        cycle_meta[[series_col, "cycle_number", "cycle_length"]],
        on=[series_col, "cycle_number"],
        how="left",
    )

    # Isolate timestamps where troughs and peaks occur so they can be broadcast across each cycle
    df["_trough_time"] = df[time_col].where(df["type"] == "trough")
    df["_peak_time"] = df[time_col].where(df["type"] == "peak")

    # Re-group after adding the new timestamp columns so pandas includes them, then broadcast start/peak times to every row in the cycle
    updated_cycle_groups = df.groupby([series_col, "cycle_number"], sort=False)
    trough_time = updated_cycle_groups["_trough_time"].transform("first")
    peak_time = updated_cycle_groups["_peak_time"].transform("first")

    # Build a cycle-level lookup table and shift the next cycle's starting trough backward to get each cycle's ending timestamp
    next_trough_meta = (
        df.groupby([series_col, "cycle_number"], as_index=False)["_trough_time"]
        .first()
        .rename(columns={"_trough_time": "_cycle_start_time"})
    )
    next_trough_meta["_cycle_end_time"] = next_trough_meta.groupby(series_col)[
        "_cycle_start_time"
    ].shift(-1)

    # Attach each cycle's end timestamp back onto the main time series DataFrame
    df = df.merge(
        next_trough_meta[[series_col, "cycle_number", "_cycle_end_time"]],
        on=[series_col, "cycle_number"],
        how="left",
    )

    # Compute relative rise (trough-to-peak) and fall (peak-to-end) durations, then clean up helper columns
    df["start_to_peak_dur"] = peak_time - trough_time
    df["peak_to_end"] = df["_cycle_end_time"] - peak_time
    df = df.drop(columns=["_trough_time", "_peak_time", "_cycle_end_time"])

    return df

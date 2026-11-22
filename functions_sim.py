from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from constants import DEFAULT_SIM_PARAMS, POINTS_PER_CYCLE


# Clean and standardise a value according to inputs
# Strict means error if input is out of bounds
def _clean_param(
    val: object,
    name: str,
    default: float | None,
    low: float | None = None,
    high: float | None = None,
    as_int: bool = False,
    allow_none: bool = False,
    strict: bool = True,
) -> int | float | None:
    """Coerces and clamps a parameter, or raises ValueError when `strict=True`."""
    if allow_none and val is None:
        return None

    # Validate numeric type and finiteness
    num: float | None = None
    if val is not None and not isinstance(val, bool):
        try:
            parsed = float(val)
            if np.isfinite(parsed):
                num = parsed
        except TypeError, ValueError:
            pass

    if num is None:
        if strict:
            raise ValueError(f"'{name}' must be a valid finite number, got {val!r}.")
        print(f"Note: '{name}' adjusted from {val!r} to {default!r}.")
        return default

    # Check bounds
    out_of_range = (low is not None and num < low) or (high is not None and num > high)
    if out_of_range and strict:
        low_str = "-inf" if low is None else str(low)
        high_str = "inf" if high is None else str(high)
        raise ValueError(
            f"'{name}' value {val!r} is out of range [{low_str}, {high_str}]."
        )

    clamped = num
    if low is not None:
        clamped = max(clamped, low)
    if high is not None:
        clamped = min(clamped, high)

    cleaned: int | float = round(clamped) if as_int else float(clamped)

    if cleaned != val:
        print(f"Note: '{name}' adjusted from {val!r} to {cleaned!r}.")

    return cleaned


# Generate alternating cycle jitter (AR(1) with negative autocorrelation)
def _alternating_jitter(
    rng: np.random.Generator,
    max_jitter: float | np.ndarray,
    reversion: float,
) -> np.ndarray:
    """Generates bounded jitter where positive deviations tend to be followed by negative ones."""
    n = np.size(max_jitter)
    raw_shocks = rng.uniform(-1.0, 1.0, size=n)
    state = raw_shocks.copy()

    phi = -reversion
    innovation_scale = np.sqrt(max(1.0 - phi**2, 0.0))
    for i in range(1, n):
        state[i] = phi * state[i - 1] + innovation_scale * raw_shocks[i]

    return np.clip(state, -1.0, 1.0) * max_jitter


# Build chronological peak/trough anchor points padded at both ends
# extrema are peaks and troughs
def _build_padded_extrema(
    n_cycles: int,
    points_per_cycle: int,
    high_time: int,
    low_time: int,
    timing_noise: float,
    amplitude_noise: float,
    amplitude_drift: float,
    cycle_reversion: float,
    rng: np.random.Generator,
    cycle_starts: np.ndarray,
    cycle_scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates chronological (time, height) peak & trough anchors with mirrored end padding."""
    cycles = np.arange(n_cycles)

    # Peak and trough heights
    max_amp_jitter = 100.0 * amplitude_noise
    drift_offset = cycles * (100.0 * amplitude_drift)

    peak_jitter = _alternating_jitter(
        rng, np.full(n_cycles, max_amp_jitter), cycle_reversion
    )
    trough_jitter = _alternating_jitter(
        rng, np.full(n_cycles, max_amp_jitter), cycle_reversion
    )

    peak_heights = np.maximum(
        (100.0 - max_amp_jitter) + drift_offset + peak_jitter, 0.0
    )
    trough_heights = np.maximum(max_amp_jitter + drift_offset + trough_jitter, 0.0)

    # Peak and trough timing
    gap = abs(low_time - high_time)
    min_gap = min(gap, points_per_cycle - gap)
    max_t_jitter = (0.45 * min_gap * timing_noise) * cycle_scales

    high_t_jitter = _alternating_jitter(rng, max_t_jitter, cycle_reversion)
    low_t_jitter = _alternating_jitter(rng, max_t_jitter, cycle_reversion)

    t_high = cycle_starts + high_time * cycle_scales + high_t_jitter
    t_low = cycle_starts + low_time * cycle_scales + low_t_jitter

    # Interleave chronologically
    t_core = np.empty(2 * n_cycles)
    h_core = np.empty(2 * n_cycles)
    if high_time < low_time:
        t_core[0::2], t_core[1::2] = t_high, t_low
        h_core[0::2], h_core[1::2] = peak_heights, trough_heights
    else:
        t_core[0::2], t_core[1::2] = t_low, t_high
        h_core[0::2], h_core[1::2] = trough_heights, peak_heights

    # Pad both ends so linear interpolation covers t = 1 through the end of the final cycle
    t0, t1, *_ = t_core
    _, h1, *_ = h_core
    t_pts = np.concatenate([[2 * t0 - t1], t_core, [2 * t_core[-1] - t_core[-2]]])
    h_pts = np.concatenate([[h1], h_core, [h_core[-2]]])
    return t_pts, h_pts


# helper function for rounded interpolation with skewed wave.
def _interp_rounded_extrema(
    t_grid: np.ndarray,
    t_pts: np.ndarray,
    h_pts: np.ndarray,
    peak_roundness: float = 0.0,
    trough_roundness: float = 0.0,
) -> np.ndarray:
    """
    Interpolate between peak and trough anchor points. 'Monotonic piecewise power'.
    roundness of zero mean straight triangular lines.
    Roundness greater than zero means broader
    roundness less than zero means spikey
    """

    # if both peak and trough roundness is zero, then use basic linear interpolation
    if np.isclose(peak_roundness, 0.0) and np.isclose(trough_roundness, 0.0):
        return np.interp(t_grid, t_pts, h_pts)

    # find consequetive pairs of low-high
    idx = np.clip(np.searchsorted(t_pts, t_grid, side="right") - 1, 0, len(t_pts) - 2)
    t0, t1 = t_pts[idx], t_pts[idx + 1]
    h0, h1 = h_pts[idx], h_pts[idx + 1]

    # normalise time between peak and trough
    dt = np.maximum(t1 - t0, 1e-12)
    u = np.clip((t_grid - t0) / dt, 0.0, 1.0)

    # convert roundness into exponents
    p_peak = 2.0 ** float(np.clip(peak_roundness, -4.0, 4.0))
    p_trough = 2.0 ** float(np.clip(trough_roundness, -4.0, 4.0))

    rising = h1 >= h0

    # define curves around peak and trough areas
    trough_curve = u**p_trough
    peak_curve = 1.0 - (1.0 - u) ** p_peak

    # join the two power curves
    rising_u = (1.0 - u) * trough_curve + u * peak_curve
    falling_u = (1.0 - u) * (1.0 - peak_curve) + u * (1.0 - trough_curve)

    # for falling segments, h0 is peak and h1 is trough
    # Near the top (u ≈ 0), peak_fall = u ** p_peak controls how slowly the curve leaves the peak
    # Near the bottom (u ≈ 1), trough_fall = 1 - (1 - u) ** p_trough controls
    # how gently it settles into the next trough.
    peak_fall = u**p_peak
    trough_fall = 1.0 - (1.0 - u) ** p_trough
    falling_u = (1.0 - u) * peak_fall + u * trough_fall

    # map back to actual heights
    eased_u = np.where(rising, rising_u, falling_u)
    return h0 + (h1 - h0) * eased_u


# Main simulate cyclic timeseries function
def simulate_cyclic_timeseries(
    sim_params: dict | None = None,
    series_name: str = "series_01",
    strict: bool = True,
) -> pd.DataFrame:
    """
    Simulates a cyclic time series from a parameter dictionary.
    Each entry in `sim_params` can be a `[value, min, max]` list/tuple or a single scalar value.
    """
    sim_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    active_params = {k: list(v) for k, v in DEFAULT_SIM_PARAMS.items()}

    if sim_params is not None:
        for k, v in sim_params.items():
            if k not in active_params:
                raise KeyError(f"Unknown simulation parameter: {k!r}")
            # Support updating via either [val, min, max] or a plain scalar value
            if isinstance(v, (list, tuple)):
                active_params[k] = list(v)
            else:
                active_params[k][0] = v

    # Helper to extract and validate a parameter using its [val, min, max] spec
    def get_clean(name: str, as_int: bool = False, allow_none: bool = False):
        val, low, high = active_params[name][:3]
        default = DEFAULT_SIM_PARAMS[name]
        return _clean_param(
            val=val,
            name=name,
            default=default,
            low=low,
            high=high,
            as_int=as_int,
            allow_none=allow_none,
            strict=strict,
        )

    n_cycles = int(get_clean("n_cycles", as_int=True))
    standard_high_time = int(get_clean("standard_high_time", as_int=True))
    standard_low_time = int(get_clean("standard_low_time", as_int=True))

    peak_roundness = float(get_clean("peak_roundness"))
    trough_roundness = float(get_clean("trough_roundness"))

    timing_noise = float(get_clean("timing_noise"))
    timing_drift = float(get_clean("timing_drift"))
    amplitude_noise = float(get_clean("amplitude_noise"))
    amplitude_drift = float(get_clean("amplitude_drift"))
    cycle_reversion = float(get_clean("cycle_reversion"))
    seed = get_clean("seed", as_int=True, allow_none=True)

    if standard_high_time == standard_low_time:
        old_low = standard_low_time
        standard_low_time = 2 if standard_high_time == 1 else standard_high_time - 1
        print(
            f"Note: 'standard_low_time' adjusted from {old_low!r} to {standard_low_time!r} to differ from 'standard_high_time'."
        )

    clean_name = (
        str(series_name).strip()
        if series_name is not None and str(series_name).strip()
        else "series_01"
    )

    param_str = (
        f"n_cycles={n_cycles};points_per_cycle={POINTS_PER_CYCLE};"
        f"standard_high_time={standard_high_time};standard_low_time={standard_low_time};"
        f"peak_roundness={peak_roundness};trough_roundness={trough_roundness};"
        f"timing_noise={timing_noise};timing_drift={timing_drift};"
        f"amplitude_noise={amplitude_noise};amplitude_drift={amplitude_drift};"
        f"cycle_reversion={cycle_reversion};seed={seed}"
    )

    rng = np.random.default_rng(seed)

    cycles = np.arange(n_cycles)
    cycle_scales = np.clip(1.0 + cycles * timing_drift, 0.25, 3.0)
    cycle_lengths = POINTS_PER_CYCLE * cycle_scales
    cycle_starts = np.concatenate([[0.0], np.cumsum(cycle_lengths[:-1])])

    t_pts, h_pts = _build_padded_extrema(
        n_cycles=n_cycles,
        points_per_cycle=POINTS_PER_CYCLE,
        high_time=standard_high_time,
        low_time=standard_low_time,
        timing_noise=timing_noise,
        amplitude_noise=amplitude_noise,
        amplitude_drift=amplitude_drift,
        cycle_reversion=cycle_reversion,
        rng=rng,
        cycle_starts=cycle_starts,
        cycle_scales=cycle_scales,
    )

    # Ensure t_pts is strictly increasing and sanitize any NaNs before .astype(int)
    t_grid = np.arange(1, max(2, round(np.sum(cycle_lengths))) + 1)

    interp_vals = _interp_rounded_extrema(
        t_grid=t_grid,
        t_pts=t_pts,
        h_pts=h_pts,
        peak_roundness=peak_roundness,
        trough_roundness=trough_roundness,
    )
    values = np.round(np.nan_to_num(interp_vals, nan=0.0)).astype(int)

    today = datetime.now().date()
    last_sunday = pd.Timestamp(today - timedelta(days=(today.weekday() + 1) % 7))
    step_offsets = (len(t_grid) - 1 - np.arange(len(t_grid))) * 7

    return pd.DataFrame(
        {
            "series_name": clean_name,
            "date": last_sunday - pd.to_timedelta(step_offsets, unit="D"),
            "time": t_grid,
            "value": values,
            "final_sim_params": param_str,
            "sim_timestamp": sim_timestamp,
        }
    )

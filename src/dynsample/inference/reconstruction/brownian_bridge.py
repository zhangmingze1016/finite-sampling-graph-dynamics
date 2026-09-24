from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from dynsample.core.state import State
from dynsample.core.trajectory import Trajectory

def brownian_bridge_step(
        left_state: State,
        right_state: State,
        time: float,
        volatility: float,
        rng: np.random.Generator,
) -> State:

    # The requested time must be a finite quantity.
    if not np.isfinite(time):
        raise ValueError(
            "time must be finite"
        )

    # The two boundary states must describe the same state space.
    if left_state.values.shape != right_state.values.shape:
        raise ValueError(
            "left_state and right_state must have the same shape"
        )

    # The right boundary must stand later in time than the left.
    if right_state.time <= left_state.time:
        raise ValueError(
            "right_state.time must be greater than left_state.time"
        )

    # A bridge point must lie strictly within its two boundaries.
    if not left_state.time < time < right_state.time:
        raise ValueError(
            "time must lie strictly between left_state.time and right_state.time"
        )

    # Volatility may vanish, but it cannot be negative or non-finite.
    if not np.isfinite(volatility) or volatility < 0:
        raise ValueError(
            "volatility must be finite and non-negative"
        )

    alpha = (
        (time - left_state.time)
        / (right_state.time - left_state.time)
    )

    mean = (
        left_state.values 
        + alpha
        * (right_state.values - left_state.values)
    )

    left_dt = time - left_state.time
    right_dt = right_state.time - time
    total_dt = left_dt + right_dt

    std = volatility * np.sqrt(left_dt*right_dt/(total_dt))
    noise = rng.standard_normal(
        size = left_state.values.shape
    )

    bridge_value = mean + std * noise
    bridge_state = State(
        time = time,
        values = bridge_value
    )

    return bridge_state
    
def brownian_bridge(
    left_state: State,
    right_state: State,
    times: NDArray[np.float64],
    volatility: float,
    rng: np.random.Generator,
) -> Trajectory:
    """
    Construct a Brownian bridge through several intermediate times.

    The bridge is drawn from the middle outward. Each state once
    drawn becomes a boundary for the intervals remaining on either
    side thereof.
    """

    times = np.asarray(
        times,
        dtype=np.float64,
    )

    # The requested times must form one finite and ordered sequence.
    if times.ndim != 1:
        raise ValueError(
            "times must be one-dimensional"
        )

    if times.size == 0:
        raise ValueError(
            "times must contain at least one time point"
        )

    if not np.all(np.isfinite(times)):
        raise ValueError(
            "times must contain only finite values"
        )

    if np.any(np.diff(times) <= 0):
        raise ValueError(
            "times must be strictly increasing"
        )

    # The two boundary states must describe the same state space.
    if left_state.values.shape != right_state.values.shape:
        raise ValueError(
            "left_state and right_state must have the same shape"
        )

    # The right boundary must stand later in time than the left.
    if right_state.time <= left_state.time:
        raise ValueError(
            "right_state.time must be greater than left_state.time"
        )

    # Every requested time must lie strictly within the boundaries.
    if (
        left_state.time >= times[0]
        or right_state.time <= times[-1]
    ):
        raise ValueError(
            "times must lie strictly between "
            "left_state.time and right_state.time"
        )

    # Volatility may vanish, but it cannot be negative or non-finite.
    if not np.isfinite(volatility) or volatility < 0:
        raise ValueError(
            "volatility must be finite and non-negative"
        )

    values = np.empty(
        (
            len(times),
            left_state.n_nodes,
            left_state.n_features,
        ),
        dtype = np.float64,
    )

    intervals = [
        (
            0,
            len(times),
            left_state,
            right_state,
        ),
    ]

    while intervals:

        start, end, left, right = intervals.pop(0)

        if start >= end:
            continue

        target_time = (
            left.time + right.time
        ) / 2

        mid = start + int(
            np.argmin(
                np.abs(
                    times[start:end]
                    - target_time
                )
            )
        )

        middle_state = brownian_bridge_step(
            left_state = left,
            right_state = right,
            time = times[mid],
            volatility = volatility,
            rng = rng,
        )

        values[mid] = middle_state.values

        if start < mid:
            intervals.append(
                (
                    start,
                    mid,
                    left,
                    middle_state,
                )
            )

        if mid + 1 < end:
            intervals.append(
                (
                    mid + 1,
                    end,
                    middle_state,
                    right,
                )
            )

    return Trajectory(
        times = times,
        values = values
    )
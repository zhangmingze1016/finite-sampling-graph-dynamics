from __future__ import annotations
import numpy as np
from dynsample.core.state import State
import math

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
    

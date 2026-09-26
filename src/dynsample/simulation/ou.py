import numpy as np
from dynsample.core.state import State
from dynsample.core.trajectory import Trajectory
from numpy.typing import NDArray

def ou_transition(
    dt: float,
    mean_reversion: float,
    long_run_mean: float,
    volatility: float,
) -> tuple[float, float, float]:
    """Return the scalar OU transition (F, offset, variance)."""
    parameters = (dt, mean_reversion, long_run_mean, volatility)
    if not np.all(np.isfinite(parameters)):
        raise ValueError(
            "all parameters must be finite"
        )

    if dt < 0:
        raise ValueError(
            "dt must be positive"
        )
    if mean_reversion < 0:
        raise ValueError(
            "mean_reversion must be non-negative"
        )

    if volatility < 0:
        raise ValueError(
            "volatility must be non-negative"
        )

    if dt == 0:
        return 1.0, 0.0, 0.0

    if mean_reversion == 0:
        return 1.0, 0.0, float(volatility ** 2 * dt)

    scaled_time = mean_reversion * dt
    transition = np.exp(-scaled_time)
    offset = -long_run_mean * np.expm1(-scaled_time)
    variance = (
        volatility ** 2
        * (-np.expm1(-2.0 * scaled_time))
        / (2.0 * mean_reversion)
    )

    return float(transition), float(offset), float(variance)

def ou_step(
    state: State,
    next_time: float,
    mean_reversion: float,
    long_run_mean: float,
    volatility: float,
    rng: np.random.Generator
) -> State:
    """Advance independent OU compoent to a later time."""
    
    if not np.isfinite(next_time):
        raise ValueError("next_time must be finite")

    if next_time <= state.time:
        raise ValueError(
            "next_time must be greater than the current state time"
        )
    
    dt = next_time - state.time

    transition, offset, variance = ou_transition(
        dt = dt,
        mean_reversion = mean_reversion,
        long_run_mean = long_run_mean,
        volatility = volatility
    )

    mean = transition * state.values + offset
    noise = rng.standard_normal(size = state.values.shape)
    next_values = mean + np.sqrt(variance) * noise

    return State(
        time = next_time,
        values = next_values
    )

def simulate_ou(
    initial_state: State,
    times: NDArray[np.float64],
    mean_reversion: float,
    long_run_mean: float,
    volatility: float,
    rng: np.random.Generator,
) -> Trajectory:
    
    times = np.asarray(
        times,
        dtype= np.float64
    )

    if times.ndim != 1:
        raise ValueError(
            "times must have shape (n_steps,)"
        )

    if times.size == 0:
        raise ValueError(
            "times must contain at least one time"
        )

    if not np.all(np.isfinite(times)):
        raise ValueError(
            "time must contain only finite values"
        )

    if times[0] != initial_state.time:
        raise ValueError(
            "the first time must equal initial_state.time"
        )

    if np.any(np.diff(times) <= 0):
        raise ValueError(
            "time must be strictly increasing"
        )

    ou_transition(
        dt=0.0,
        mean_reversion=mean_reversion,
        long_run_mean=long_run_mean,
        volatility=volatility,
    )

    values = np.empty(
        (
            times.shape[0],
            initial_state.n_nodes,
            initial_state.n_features,
        ),
        dtype = np.float64,
    )

    values[0] = initial_state.values

    current_state = initial_state
    
    for i in range(1, times.shape[0]):
        current_state = ou_step(
            state = current_state,
            next_time = times[i],
            volatility = volatility,
            mean_reversion = mean_reversion,
            long_run_mean = long_run_mean,
            rng = rng,
        )

        values[i] = current_state.values

    return Trajectory(
        times = times,
        values = values,
    )



    
    
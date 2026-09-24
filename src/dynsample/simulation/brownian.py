import numpy as np
from numpy.typing import NDArray

from dynsample.core.state import State
from dynsample.core.trajectory import Trajectory

def brownian_step(
        state: State,
        next_time: float,
        volatility: float,
        rng: np.random.Generator,
) -> State:
    """
    Advance a state from its present time unto a later time according
    to independent Brownian motion.

    Parameters
    ----------
    state:
        The state from which the transition shall begin.

    next_time:
        The time unto which the state shall be advanced.

    volatility:
        The common volatility assigned to every node and feature.

        If sigma denotes this quantity and dt the elapsed time, then
        every component of the Brownian increment hath variance

            sigma^2 * dt.

    rng:
        The NumPy random number generator from which the several
        standard normal increments shall be drawn.

    Returns
    -------
    State
        The state obtained at ``next_time``.

    Notes
    -----
    The transition obeys

        X(t + dt) = X(t) + sigma * sqrt(dt) * Z,

    where every entry of Z is drawn independently from N(0, 1).
    """    
    # The next state must belong to a time strictly later than that
    # of the state from which we proceed.
    if not np.isfinite(next_time):
        raise ValueError(
            "next_time must be finite"
        )

    if next_time <= state.time:
        raise ValueError(
            "next_time must be greater than the current state time"
        )

    # Volatility represents a standard-deviation scale and therefore
    # cannot properly take a negative value.

    if not np.isfinite(volatility) or volatility < 0:
        raise ValueError(
            "volatility must be finite and non-negative"
        )

    # The elapsed time may be irregular; nothing here supposes that
    # successive observations stand at equal intervals.

    dt = next_time - state.time

    # Draw one independent standard normal increment for every
    # node-feature pair contained in the state.
    #
    # If state.values hath shape
    #
    #     (n_nodes, n_features),
    #
    # then noise possesses that same shape.
    noise = rng.standard_normal(
        size = state.values.shape
    )

    # Brownian motion scales with sqrt(dt), rather than dt itself.
    #
    # Since
    #
    #     Z ~ N(0, 1),
    #
    # we have
    #
    #     sigma * sqrt(dt) * Z
    #
    # distributed as
    #
    #     N(0, sigma^2 * dt).

    increment = (
        volatility
        * np.sqrt(dt)
        * noise
    )

    # The old state is not altered.  A new State is returned at the
    # appointed time.
    return State(
        time = next_time,
        values = state.values + increment
    )

def simulate_brownian(
        initial_state: State,
        times: NDArray[np.float64],
        volatility: float,
        rng: np.random.Generator,
) -> Trajectory:
    """
        Simulate Brownian dynamics at a prescribed collection of times.
    
        Parameters
        ----------
        initial_state:
            The state from which the trajectory shall begin.
    
        times:
            A one-dimensional array containing the times at which states
            are required.
    
            The first time must equal ``initial_state.time``, and all
            subsequent times must be strictly increasing.
    
            Equal spacing is neither required nor presumed.
    
        volatility:
            The common volatility assigned to every node and feature.
    
        rng:
            The NumPy random number generator used for all stochastic
            increments.
    
        Returns
        -------
        Trajectory
            The simulated states at precisely the times supplied by the
            caller.
    
        Notes
        -----
        The simulation does not construct its own time grid.
    
        The caller determines the times at which states are required, and
        the Brownian dynamics determine only how the state changes between
        each successive pair of those times.
        """
    # Reduce the supplied times to the numerical representation used
    # throughout the package.
    times = np.asarray(
        times,
        dtype = np.float64
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

    if not np.isclose(
        times[0],
        initial_state.time,
    ):
        raise ValueError(
            "the first time must equal initial_state.time"
        )

    if np.any(np.diff(times) <= 0):
        raise ValueError(
            "time must be strictly increasing"
        )

    if not np.isfinite(volatility) or volatility < 0:
        raise ValueError(
            "volatility must be finite and non-negative"
        )

    # Reserve the complete trajectory in advance.
    #
    # Its form is
    #
    #     time x node x feature
    #
    # or
    #
    #     (n_steps, n_nodes, n_features).

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
        current_state = brownian_step(
            state = current_state,
            next_time = times[i],
            volatility = volatility,
            rng = rng,
        )

        values[i] = current_state.values

    return Trajectory(
        times = times,
        values = values,
    )
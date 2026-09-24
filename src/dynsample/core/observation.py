from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

@dataclass
class Observation:
    """
    An observation of a dynamical system at a single point in time.

    Unlike a State, which contains the complete condition of the system,
    an Observation records only those quantities which have in fact
    been seen.

    Parameters
    ----------
    time: float
        The time belonging to this observation.

    values:
        The several measured values of the system.
    
        Shape: (N, d)
        Type: NDArray[np.float64]

        Axis 0:
            Node.

        Axis 1:
            Feature.

        values[i, k] is the recorded value of feature k belonging
        to node i.

    mask:
        A declaration of which entries have been observed.

        Shape: (N, d)
        Type: NDArray[np.bool_]

        mask[i, k] is True when values[i, k] hath been observed,
        and False otherwise
    """

    time: float

    # Shape: (N, d)
    # Axis 0: node
    # Axis 1: feature
    #values[i, k] = recorded value of feature k belonging to node i.
    values: NDArray[np.float64]

    # Shape:(N, d)
    # Axis 0: node
    # Axis 1: feature
    # values[i, k] = recorded value of feature k belonging to node i.
    mask: NDArray[np.bool_]

    def __post_init__(self) -> None:
        # Let the values and mask be reduced to their proper numerical
        # forms, that subsequent computations may proceed uniformly.
        self.values = np.asarray(
            self.values,
            dtype = np.float64
        )

        self.mask = np.asarray(
            self.mask,
            dtype = np.bool_,
        )

        # As with State, the values ought to form a matrix whose rows
        # distinguish nodes and whose columns distinguish features.
        if self.values.ndim != 2:
            raise ValueError(
                "values must have shape (n_nodes, n_features)"
            )

        # The mask must possess the same two-dimensional constitution.
        if self.mask.ndim != 2:
            raise ValueError(
                "mask must have shape (n_nodes, n_features)"
            )

        # Every entry of the state must have one corresponding declaration
        # telling us whether that quantity was observed.
        if self.values.shape != self.mask.shape:
            raise ValueError(
                "values and mask must have the same shape"
            )

        # The appointed time must be a finite quantity.
        if not np.isfinite(self.time):
            raise ValueError(
                "time must be finite"
            )

    @property
    def n_nodes(self) -> int:
        """Return the number of nodes belonging to the observation."""
        return self.values.shape[0]

    @property
    def n_features(self) -> int:
        """Return the number of features belonging to each node."""
        return self.values.shape[1]

    @property   
    def n_observed(self) -> int:
        """Return the number of entries which have been observed."""
        return int(np.sum(self.mask))
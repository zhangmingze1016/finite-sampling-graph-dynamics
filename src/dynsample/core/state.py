from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

@dataclass
class State:
    """
    State of a dynamical system at a single point in time.

    Parameters
    -------------
    time: float
       The time at which the present state is observed

    values: np.ndarray
        A matrix containing the several values which constitute the state,
        having shape (n_nodes, n_features).

        Each row represents one node; each column, one feature thereof.
    """

    time: float
    values: NDArray[np.float64]

    def __post_init__(self) -> None:
        #Convert array-like input into a NumPy float64 array.
        self.values = np.asarray(self.values, dtype = np.float64)

        #A state shall be represented in the following form:
        #(number of nodes, number of features)
        #
        #
        #Thus, for example, the state
        #
        #[[0.01, 0.20],
        # [0.02, 0.15],
        # [-0.01, 0.30]]
        #
        #consists of three nodes, each possessing two features

        if self.values.ndim != 2:
            raise ValueError(
                "values must be represented in the form (n_nodes, n_features)"
            )

        if not np.isfinite(self.time):
            raise ValueError("time must be finite")

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the system."""
        return self.values.shape[0]

    @property
    def n_features(self) -> int:
        """Number of features in each node."""
        return self.values.shape[1]

    


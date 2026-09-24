import numpy as np
import pytest

from dynsample.core.state import State
from dynsample.inference.reconstruction.brownian_bridge import (
    brownian_bridge_step,
)


def test_brownian_bridge_step_returns_state() -> None:
    left_state = State(
        time=0.0,
        values=np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        ),
    )

    right_state = State(
        time=10.0,
        values=np.array(
            [
                [5.0, 6.0],
                [7.0, 8.0],
            ]
        ),
    )

    rng = np.random.default_rng(42)

    result = brownian_bridge_step(
        left_state=left_state,
        right_state=right_state,
        time=4.0,
        volatility=1.0,
        rng=rng,
    )

    assert isinstance(result, State)
    assert result.time == 4.0
    assert result.values.shape == left_state.values.shape


def test_brownian_bridge_step_zero_volatility() -> None:
    left_state = State(
        time=0.0,
        values=np.array(
            [
                [10.0, 20.0],
                [30.0, 40.0],
            ]
        ),
    )

    right_state = State(
        time=10.0,
        values=np.array(
            [
                [30.0, 40.0],
                [50.0, 60.0],
            ]
        ),
    )

    rng = np.random.default_rng(42)

    result = brownian_bridge_step(
        left_state=left_state,
        right_state=right_state,
        time=2.0,
        volatility=0.0,
        rng=rng,
    )

    alpha = 2.0 / 10.0

    expected = (
        left_state.values
        + alpha
        * (right_state.values - left_state.values)
    )

    assert np.allclose(result.values, expected)


def test_brownian_bridge_step_reproducible() -> None:
    left_state = State(
        time=0.0,
        values=np.array([[1.0, 2.0]]),
    )

    right_state = State(
        time=10.0,
        values=np.array([[5.0, 8.0]]),
    )

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    result1 = brownian_bridge_step(
        left_state=left_state,
        right_state=right_state,
        time=4.0,
        volatility=1.5,
        rng=rng1,
    )

    result2 = brownian_bridge_step(
        left_state=left_state,
        right_state=right_state,
        time=4.0,
        volatility=1.5,
        rng=rng2,
    )

    assert np.allclose(result1.values, result2.values)


def test_brownian_bridge_step_rejects_left_endpoint() -> None:
    left_state = State(
        time=0.0,
        values=np.array([[1.0]]),
    )

    right_state = State(
        time=10.0,
        values=np.array([[2.0]]),
    )

    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        brownian_bridge_step(
            left_state=left_state,
            right_state=right_state,
            time=0.0,
            volatility=1.0,
            rng=rng,
        )


def test_brownian_bridge_step_rejects_right_endpoint() -> None:
    left_state = State(
        time=0.0,
        values=np.array([[1.0]]),
    )

    right_state = State(
        time=10.0,
        values=np.array([[2.0]]),
    )

    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        brownian_bridge_step(
            left_state=left_state,
            right_state=right_state,
            time=10.0,
            volatility=1.0,
            rng=rng,
        )


def test_brownian_bridge_step_rejects_time_before_interval() -> None:
    left_state = State(
        time=0.0,
        values=np.array([[1.0]]),
    )

    right_state = State(
        time=10.0,
        values=np.array([[2.0]]),
    )

    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        brownian_bridge_step(
            left_state=left_state,
            right_state=right_state,
            time=-1.0,
            volatility=1.0,
            rng=rng,
        )


def test_brownian_bridge_step_rejects_time_after_interval() -> None:
    left_state = State(
        time=0.0,
        values=np.array([[1.0]]),
    )

    right_state = State(
        time=10.0,
        values=np.array([[2.0]]),
    )

    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        brownian_bridge_step(
            left_state=left_state,
            right_state=right_state,
            time=11.0,
            volatility=1.0,
            rng=rng,
        )


def test_brownian_bridge_step_rejects_invalid_state_order() -> None:
    left_state = State(
        time=10.0,
        values=np.array([[1.0]]),
    )

    right_state = State(
        time=5.0,
        values=np.array([[2.0]]),
    )

    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        brownian_bridge_step(
            left_state=left_state,
            right_state=right_state,
            time=7.0,
            volatility=1.0,
            rng=rng,
        )


def test_brownian_bridge_step_rejects_shape_mismatch() -> None:
    left_state = State(
        time=0.0,
        values=np.ones((2, 3)),
    )

    right_state = State(
        time=10.0,
        values=np.ones((3, 3)),
    )

    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        brownian_bridge_step(
            left_state=left_state,
            right_state=right_state,
            time=5.0,
            volatility=1.0,
            rng=rng,
        )


def test_brownian_bridge_step_rejects_negative_volatility() -> None:
    left_state = State(
        time=0.0,
        values=np.array([[1.0]]),
    )

    right_state = State(
        time=10.0,
        values=np.array([[2.0]]),
    )

    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        brownian_bridge_step(
            left_state=left_state,
            right_state=right_state,
            time=5.0,
            volatility=-0.5,
            rng=rng,
        )


@pytest.mark.parametrize(
    "time",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_brownian_bridge_step_rejects_nonfinite_time(
    time: float,
) -> None:
    left_state = State(
        time=0.0,
        values=np.array([[1.0]]),
    )

    right_state = State(
        time=10.0,
        values=np.array([[2.0]]),
    )

    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        brownian_bridge_step(
            left_state=left_state,
            right_state=right_state,
            time=time,
            volatility=1.0,
            rng=rng,
        )


@pytest.mark.parametrize(
    "volatility",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_brownian_bridge_step_rejects_nonfinite_volatility(
    volatility: float,
) -> None:
    left_state = State(
        time=0.0,
        values=np.array([[1.0]]),
    )

    right_state = State(
        time=10.0,
        values=np.array([[2.0]]),
    )

    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        brownian_bridge_step(
            left_state=left_state,
            right_state=right_state,
            time=5.0,
            volatility=volatility,
            rng=rng,
        )


def test_brownian_bridge_step_empirical_mean_and_variance() -> None:
    left_state = State(
        time=0.0,
        values=np.array([[10.0]]),
    )

    right_state = State(
        time=10.0,
        values=np.array([[30.0]]),
    )

    time = 2.0
    volatility = 2.0
    n_samples = 100_000

    rng = np.random.default_rng(42)

    samples = np.empty(n_samples)

    for i in range(n_samples):
        result = brownian_bridge_step(
            left_state=left_state,
            right_state=right_state,
            time=time,
            volatility=volatility,
            rng=rng,
        )

        samples[i] = result.values[0, 0]

    alpha = (
        (time - left_state.time)
        / (right_state.time - left_state.time)
    )

    expected_mean = (
        left_state.values[0, 0]
        + alpha
        * (
            right_state.values[0, 0]
            - left_state.values[0, 0]
        )
    )

    expected_variance = (
        volatility**2
        * (time - left_state.time)
        * (right_state.time - time)
        / (right_state.time - left_state.time)
    )

    assert np.isclose(
        samples.mean(),
        expected_mean,
        atol=0.03,
    )

    assert np.isclose(
        samples.var(),
        expected_variance,
        atol=0.08,
    )
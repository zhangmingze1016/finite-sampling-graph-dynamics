import numpy as np
import matplotlib.pyplot as plt

from dynsample.core.state import State
from dynsample.inference.reconstruction.brownian_bridge import (
    brownian_bridge,
)


# ---------------------------------------------------------------------------
# Experimental setting
# ---------------------------------------------------------------------------

# The two states here serve as the known boundaries of the bridge.
# Their values are deliberately separated, that the conditional
# tendency of the path may be plainly seen.
left_state = State(
    time=0.0,
    values=np.array([
        [2.0],
    ]),
)

right_state = State(
    time=10.0,
    values=np.array([
        [8.0],
    ]),
)


# These intermediate times are deliberately irregular.
# The bridge must therefore attend to time itself, rather than
# merely taking the middle index of the remaining array.
times = np.array([
    0.20,
    0.55,
    0.90,
    1.25,
    1.60,
    1.95,
    2.35,
    2.80,
    3.15,
    3.60,
    4.05,
    4.45,
    4.85,
    5.20,
    5.65,
    6.05,
    6.55,
    6.90,
    7.35,
    7.80,
    8.15,
    8.55,
    8.90,
    9.35,
    9.75,
])


volatility = 1.5
seed = 42

rng = np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Trace the order in which the bridge intervals are divided
# ---------------------------------------------------------------------------

def trace_bridge_construction(
    left_state: State,
    right_state: State,
    times: np.ndarray,
) -> None:
    """
    Display the order in which the several bridge points are chosen.

    This function reproduces only the interval-selection logic.
    It draws no random values and therefore alters not the random
    generator employed by the experiment itself.
    """

    intervals = [
        (
            0,
            len(times),
            left_state.time,
            right_state.time,
        ),
    ]

    step = 1

    print("=" * 60)
    print("BROWNIAN BRIDGE CONSTRUCTION")
    print("=" * 60)
    print()

    while intervals:

        start, end, left_time, right_time = intervals.pop(0)

        if start >= end:
            continue

        target_time = (
            left_time + right_time
        ) / 2

        mid = start + int(
            np.argmin(
                np.abs(
                    times[start:end]
                    - target_time
                )
            )
        )

        selected_time = times[mid]

        print(f"Step {step}")
        print("-" * 40)

        print(
            f"Boundary interval : "
            f"[{left_time:.2f}, {right_time:.2f}]"
        )

        print(
            f"Candidate indices : "
            f"[{start}, {end})"
        )

        print(
            f"Candidate times   : "
            f"{times[start:end]}"
        )

        print(
            f"Temporal midpoint : "
            f"{target_time:.2f}"
        )

        print(
            f"Selected index    : "
            f"{mid}"
        )

        print(
            f"Selected time     : "
            f"{selected_time:.2f}"
        )

        if start < mid:
            intervals.append(
                (
                    start,
                    mid,
                    left_time,
                    selected_time,
                )
            )

        if mid + 1 < end:
            intervals.append(
                (
                    mid + 1,
                    end,
                    selected_time,
                    right_time,
                )
            )

        print()

        print("Queue after division:")

        if intervals:
            for interval in intervals:
                (
                    queued_start,
                    queued_end,
                    queued_left,
                    queued_right,
                ) = interval

                print(
                    "    "
                    f"indices [{queued_start}, {queued_end}) "
                    f"bounded by "
                    f"[{queued_left:.2f}, {queued_right:.2f}]"
                )
        else:
            print("    empty")

        print()
        step += 1


# ---------------------------------------------------------------------------
# Run the actual Brownian bridge
# ---------------------------------------------------------------------------

trace_bridge_construction(
    left_state=left_state,
    right_state=right_state,
    times=times,
)


result = brownian_bridge(
    left_state=left_state,
    right_state=right_state,
    times=times,
    volatility=volatility,
    rng=rng,
)


# ---------------------------------------------------------------------------
# Compute the conditional mean
# ---------------------------------------------------------------------------

alpha = (
    (times - left_state.time)
    / (right_state.time - left_state.time)
)

conditional_mean = (
    left_state.values[0, 0]
    + alpha
    * (
        right_state.values[0, 0]
        - left_state.values[0, 0]
    )
)


# ---------------------------------------------------------------------------
# Display the reconstructed states
# ---------------------------------------------------------------------------

print("=" * 60)
print("FINAL BROWNIAN BRIDGE")
print("=" * 60)
print()

print(
    f"Left boundary  : "
    f"t = {left_state.time:.2f}, "
    f"X = {left_state.values[0, 0]:.4f}"
)

print(
    f"Right boundary : "
    f"t = {right_state.time:.2f}, "
    f"X = {right_state.values[0, 0]:.4f}"
)

print(
    f"Volatility     : {volatility}"
)

print(
    f"Random seed    : {seed}"
)

print()

print(
    f"{'Time':>10}"
    f"{'Mean':>15}"
    f"{'Sample':>15}"
)

print("-" * 40)

for i in range(len(times)):
    print(
        f"{times[i]:>10.2f}"
        f"{conditional_mean[i]:>15.4f}"
        f"{result.values[i, 0, 0]:>15.4f}"
    )


# ---------------------------------------------------------------------------
# Prepare the complete path for plotting
# ---------------------------------------------------------------------------

plot_times = np.concatenate(
    (
        [left_state.time],
        result.times,
        [right_state.time],
    )
)

plot_values = np.concatenate(
    (
        [left_state.values[0, 0]],
        result.values[:, 0, 0],
        [right_state.values[0, 0]],
    )
)


mean_times = np.linspace(
    left_state.time,
    right_state.time,
    200,
)

mean_values = (
    left_state.values[0, 0]
    + (
        (mean_times - left_state.time)
        / (right_state.time - left_state.time)
    )
    * (
        right_state.values[0, 0]
        - left_state.values[0, 0]
    )
)


# ---------------------------------------------------------------------------
# Plot the experiment
# ---------------------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    plot_times,
    plot_values,
    marker="o",
    label="Sampled Brownian bridge",
)

plt.plot(
    mean_times,
    mean_values,
    linestyle="--",
    label="Conditional mean",
)

plt.scatter(
    [
        left_state.time,
        right_state.time,
    ],
    [
        left_state.values[0, 0],
        right_state.values[0, 0],
    ],
    s=100,
    label="Known boundary states",
)

plt.xlabel("Time")
plt.ylabel("State value")

plt.title(
    "Brownian Bridge Reconstruction "
    "with Irregular Intermediate Times"
)

plt.legend()
plt.grid(
    alpha=0.25
)

plt.tight_layout()
plt.show()
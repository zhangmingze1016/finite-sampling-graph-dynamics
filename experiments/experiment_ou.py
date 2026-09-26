import numpy as np
import matplotlib.pyplot as plt

from dynsample.core.state import State
from dynsample.simulation.ou import simulate_ou

initial_state = State(
    time=0.0,
    values=np.array([[14.0]]),
)

times = np.array(
    [
        0.00,
        0.03,
        0.12,
        0.40,
        0.45,
        0.90,
        1.35,
        1.38,
        2.10,
        2.80,
        3.00,
        3.85,
        4.00,
        4.08,
        5.20,
        6.50,
        6.55,
        7.30,
        8.60,
        9.10,
        9.95,
        10.00,
    ],
    dtype=np.float64,
)
mean_reversion = 0.7
long_run_mean = 10.0
volatility = 1.5
rng = np.random.default_rng(42)

result = simulate_ou(
    initial_state=initial_state,
    times=times,
    mean_reversion=mean_reversion,
    long_run_mean=long_run_mean,
    volatility=volatility,
    rng=rng,
)

elapsed = times - initial_state.time
initial_value = initial_state.values[0, 0]

conditional_mean = long_run_mean + np.exp(
    -mean_reversion * elapsed
) * (initial_value - long_run_mean)

if mean_reversion == 0.0:
    conditional_variance = volatility**2 * elapsed
else:
    conditional_variance = (
        volatility**2
        * (-np.expm1(-2.0 * mean_reversion * elapsed))
        / (2.0 * mean_reversion)
    )

conditional_std = np.sqrt(conditional_variance)
lower = conditional_mean - 1.96 * conditional_std
upper = conditional_mean + 1.96 * conditional_std

print("OU SIMULATION")
print(f"Initial state: {initial_value:.4f}")
print(f"Mean reversion: {mean_reversion:.4f}")
print(f"Long-run mean: {long_run_mean:.4f}")
print(f"Volatility: {volatility:.4f}")
print()

print(
    f"{'Time':>10}"
    f"{'Mean':>12}"
    f"{'Variance':>12}"
    f"{'Sample':>12}"
)

for i, time in enumerate(result.times):
    print(
        f"{time:>10.2f}"
        f"{conditional_mean[i]:>12.4f}"
        f"{conditional_variance[i]:>12.4f}"
        f"{result.values[i, 0, 0]:>12.4f}"
    )

plt.figure(figsize=(10, 6))

plt.fill_between(
    times,
    lower,
    upper,
    color="tab:blue",
    alpha=0.15,
    label="Pointwise 95% conditional interval",
)

plt.plot(
    result.times,
    result.values[:, 0, 0],
    color="tab:blue",
    label="Sampled OU path",
    marker="o",
    markersize=4,
)

plt.plot(
    times,
    conditional_mean,
    color="tab:orange",
    linestyle="--",
    label="Conditional mean",
)

plt.axhline(
    long_run_mean,
    color="black",
    linestyle=":",
    label="Long-run mean",
)

plt.scatter(
    [initial_state.time],
    [initial_value],
    color="red",
    s=60,
    zorder=5,
    label="Initial state",
)

plt.xlabel("Time")
plt.ylabel("State value")
plt.title("OU Simulation")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()
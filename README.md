# Finite-Sampling Graph Dynamics

A Python framework for studying dynamical systems on graphs when only a finite, irregular, and partial collection of observations is available.

The project is concerned with the general problem

$$
\text{finite observations}
\;\longrightarrow\;
\text{stochastic dynamical model}
\;\longrightarrow\;
\text{conditional reconstruction and prediction}.
$$

Rather than supposing the complete state of a system to be known at every instant, the framework distinguishes the underlying dynamical state from the quantities that have actually been observed.

Observation times are externally specified and need not be equally spaced. The dynamical model does not determine when observations occur; rather, it governs how latent states may evolve between the times at which the system has been sampled.

Observed quantities act as constraints upon inference, while unobserved quantities may be reconstructed conditionally under a chosen dynamical model.

## Project Goals

The project is being developed to support experiments involving:

* stochastic dynamical systems;
* graph-coupled dynamics;
* finite and irregular sampling;
* partial and asynchronous observations;
* observation and sampling policies;
* conditional reconstruction of unobserved states;
* uncertainty-aware inference of latent trajectories;
* forecasting from incomplete information;
* comparison of sampling and estimation methods under limited observation budgets.

A principal question is how much of the evolution of a dynamical system may be recovered when it is not possible to observe every node, feature, and time point.

A further question is how observations ought to be allocated when the number of measurements is itself constrained.

## Core Representation

For a system of \(N\) nodes, each possessing \(d\) features, the state at time \(t\) is represented by

$$
X(t)\in\mathbb{R}^{N\times d}.
$$

A trajectory recorded at a finite ordered collection of \(T\) time points

$$
t_0<t_1<\cdots<t_{T-1}
$$

is represented by

$$
X\in\mathbb{R}^{T\times N\times d}.
$$

The intervals

$$
t_{k+1}-t_k
$$

need not be equal.

Relations among nodes are represented by a weighted adjacency matrix

$$
A\in\mathbb{R}^{N\times N}.
$$

Partial observations are represented by a value matrix together with a Boolean mask declaring which quantities were observed.

The distinction between state and observation is fundamental:

$$
\text{state}
\neq
\text{observation}.
$$

A state describes the system at a particular time. An observation describes that portion of the state which was actually measured.

## Stochastic Reconstruction

Suppose that a latent trajectory \(X\) is governed by a stochastic dynamical model, while only some portion of that trajectory has been observed.

The reconstruction problem may be expressed generally as inference of

$$
p\!\left(
X_{\mathrm{missing}}
\mid
X_{\mathrm{observed}},
\text{dynamics},
G
\right),
$$

where \(G\) denotes the graph structure when relations among nodes are available.

The purpose of reconstruction is therefore not merely to assign a single value to every missing quantity. Whenever possible, the framework shall preserve the uncertainty associated with states which were never directly observed.

For example, under Brownian dynamics,

$$
dX_t=\sigma\,dW_t,
$$

observations at two times may constrain the conditional distribution of states lying between them. This leads naturally to Brownian bridges and, more generally, to conditional stochastic processes.

Future models may include mean-reverting processes, correlated multivariate dynamics, state-space models, and graph-coupled stochastic differential equations.

## Current Components

The present core implementation contains four principal data structures.

### `State`

Represents the state of the dynamical system at one point in time.

```text
time:   float
values: (N, d)
```

A `State` is not required to have been directly observed. It may represent an underlying, simulated, reconstructed, or otherwise specified state of the system.

### `Trajectory`

Represents the state of the system at a finite ordered collection of time points.

```text
times:  (T,)
values: (T, N, d)
```

The recorded times must be strictly increasing, but they need not be equally spaced.

Thus a trajectory may be recorded at times such as

```text
[0.0, 0.13, 0.47, 1.20, 2.85]
```

without introducing an artificial fixed time step.

### `Observation`

Represents the portion of a state that has actually been observed.

```text
time:   float
values: (N, d)
mask:   (N, d)
```

For each node \(i\) and feature \(k\),

```text
mask[i, k] = True
```

means that `values[i, k]` was observed.

Different nodes and features may therefore possess different observation patterns at the same time.

For example,

```text
             t0      t1      t2      t3

Node A       yes      no     yes      no
Node B       yes     yes      no      no
Node C        no     yes     yes     yes
```

is a valid finite observation pattern.

### `Graph`

Represents weighted relations among the nodes.

```text
adjacency: (N, N)
```

where

```text
adjacency[i, j]
```

is the weight of the connection from node \(i\) to node \(j\).

The graph representation presently provides:

* node count;
* weighted degree;
* degree matrix;
* graph Laplacian \(L=D-A\);
* detection of directed and undirected adjacency matrices.

The graph is intended eventually to provide spatial or relational information for reconstruction when the state of one node bears information about the states of its neighbors.

## Design Principle

Sampling and dynamics are treated as separate concerns.

The sampling mechanism determines

$$
\boxed{\text{where and when information is observed}},
$$

while the dynamical model determines

$$
\boxed{\text{how the underlying system may evolve}}.
$$

Consequently, a stochastic model should not be required to generate the observation times.

Given two successive sampling times \(t_i\) and \(t_{i+1}\), a dynamical model instead governs the transition

$$
X(t_i)\longrightarrow X(t_{i+1}).
$$

For Brownian dynamics, for example,

$$
W_{t_{i+1}}-W_{t_i}
\sim
\mathcal{N}(0,t_{i+1}-t_i),
$$

so irregular sampling intervals arise naturally without requiring a constant `dt`.

When observations exist on both sides of a missing state, those observations may jointly constrain its conditional distribution.

When no future observation is available, the same dynamical model may instead be used for forward simulation or forecasting.

## Project Structure

```text
finite-sampling-graph-dynamics/
│
├── src/
│   └── dynsample/
│       ├── core/
│       │   ├── state.py
│       │   ├── trajectory.py
│       │   ├── observation.py
│       │   └── graph.py
│       │
│       ├── simulation/
│       ├── sampling/
│       ├── estimation/
│       ├── metrics/
│       └── data/
│
├── tests/
├── examples/
├── experiments/
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Development Roadmap

The present development path is

$$
\text{Core representations}
\rightarrow
\text{Stochastic dynamics}
\rightarrow
\text{Finite observations}
\rightarrow
\text{Conditional reconstruction}
\rightarrow
\text{Graph-aware inference}
\rightarrow
\text{Forecasting}
\rightarrow
\text{Evaluation}.
$$

The first stochastic reconstruction model will use Brownian dynamics.

This provides a simple setting in which to distinguish three related tasks:

1. forward stochastic simulation;
2. conditional reconstruction between observed states;
3. forecasting beyond the final observation.

For missing states lying between known observations, Brownian bridge methods provide the first conditional reconstruction model.

Subsequent development may include:

* Brownian motion and Brownian bridges;
* Ornstein-Uhlenbeck dynamics and mean reversion;
* multivariate and correlated stochastic processes;
* conditional Gaussian inference;
* state-space models;
* Kalman filtering and smoothing;
* graph-coupled stochastic differential equations;
* graph-aware reconstruction;
* uncertainty quantification;
* finite-observation sampling policies;
* quantitative reconstruction and forecasting metrics.

The longer-term objective is to study graph-aware stochastic inference for irregularly sampled dynamical systems.

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Testing

Run the complete test suite with:

```bash
python -m pytest -v
```

The tests presently cover the core state, trajectory, observation, and graph representations.

## Status

**V001 — Core structures**

Implemented:

* `State`
* `Trajectory`
* `Observation`
* `Graph`
* core validation
* irregular trajectory times
* unit tests

**V003 — Stochastic dynamics and conditional reconstruction**

Next:

* Brownian transition model;
* Brownian bridge reconstruction;
* reconstruction of missing observations;
* uncertainty-preserving output;
* tests for irregularly sampled stochastic trajectories.

# Finite-Sampling Graph Dynamics

A Python framework for studying dynamical systems on graphs when only a finite and partial collection of observations is available.

The project is concerned with the general problem

$$
\text{dynamics}
\;\longrightarrow\;
\text{finite observations}
\;\longrightarrow\;
\text{reconstruction and prediction}.
$$

Rather than supposing the complete state of a system to be known at every instant, the framework distinguishes the true underlying trajectory from the quantities that have actually been observed.

## Project Goals

The project is being developed to support experiments involving:

* stochastic dynamical systems;
* graph-coupled dynamics;
* finite and partial observations;
* observation and sampling policies;
* reconstruction of unobserved states;
* forecasting from incomplete information;
* comparison of sampling and estimation methods under limited observation budgets.

A principal question is how observations ought to be allocated when it is not possible to observe every node, feature, and time point.

## Core Representation

For a system of \(N\) nodes, each possessing \(d\) features, the state at time \(t\) is represented by

$$
X(t)\in\mathbb{R}^{N\times d}.
$$

A complete trajectory observed at \(T\) time points is represented by

$$
X\in\mathbb{R}^{T\times N\times d}.
$$

Relations among nodes are represented by a weighted adjacency matrix

$$
A\in\mathbb{R}^{N\times N}.
$$

Partial observations are represented by a value matrix together with a Boolean mask declaring which quantities were observed.

## Current Components

The present core implementation contains four principal data structures.

### `State`

Represents the complete state of the dynamical system at one point in time.

```text
time:   float
values: (N, d)
```

### `Trajectory`

Represents the complete evolution of the system across several time points.

```text
times:  (T,)
values: (T, N, d)
```

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

The first development path is

$$
\text{Graph}
\rightarrow
\text{Simulation}
\rightarrow
\text{Trajectory}
\rightarrow
\text{Sampling}
\rightarrow
\text{Observation}
\rightarrow
\text{Reconstruction}
\rightarrow
\text{Evaluation}.
$$

Planned additions include stochastic simulators, graph-coupled dynamics, finite-observation samplers, reconstruction methods, and quantitative error metrics.

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
* unit tests

Next:

* stochastic simulation

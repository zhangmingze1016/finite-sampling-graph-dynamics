# Finite-Sampling Graph Dynamics

A Python framework for reconstructing, forecasting, and quantifying uncertainty in dynamical systems from finite, irregular, and partial observations.

The project is built around a practical problem:

> Given an incomplete record of a dynamical system, what can be reconstructed, how uncertain is that reconstruction, and where would an additional observation be most valuable?

The general inference pipeline is

$$
\text{observations}
\longrightarrow
\text{parameter estimation}
\longrightarrow
\text{dynamical inference}
\longrightarrow
\text{reconstruction / forecasting}
\longrightarrow
\text{uncertainty and evaluation}.
$$

Observation times are externally supplied and need not be equally spaced.

The framework is intended to operate directly upon irregular observations rather than requiring data to be resampled onto an artificial uniform time grid.

---

## Project Goals

The project studies dynamical systems when only a finite and incomplete collection of observations is available.

The principal objectives are to support:

- finite and irregular observation times;
- partial and asynchronous observations;
- stochastic dynamical systems;
- graph-coupled dynamics;
- reconstruction of unobserved states;
- forecasting from incomplete histories;
- uncertainty-aware inference;
- quantitative evaluation against withheld observations;
- comparison of competing dynamical structures;
- adaptive allocation of computational resolution;
- active selection of future observations.

A central question is

$$
\boxed{
\text{How much of a dynamical system may be recovered from limited observations?}
}
$$

A second question arises when observations themselves are costly:

$$
\boxed{
\text{Where should the next observation be made?}
}
$$

The long-term objective is therefore not merely to reconstruct missing values, but to develop a general framework connecting

$$
\text{observation}
\rightarrow
\text{inference}
\rightarrow
\text{uncertainty}
\rightarrow
\text{evaluation}
\rightarrow
\text{observation design}.
$$

---

## Development Objectives

The project is intended to grow not merely as a collection of stochastic-process implementations, but as a practical inference framework.

Development is guided by the following principles.

### Native Irregular-Time Support

Real observations need not occur on a uniform time grid.

For example,

```text
[0.00, 0.13, 0.47, 3.82, 8.91, 9.04]
```

should be accepted directly.

Algorithms should account for the actual temporal distances between observations rather than requiring a constant `dt`.

### Reconstruction from Incomplete Information

Missing information may arise from:

- unobserved time points;
- unobserved nodes;
- unobserved features;
- contiguous missing intervals;
- asynchronous observation schedules.

Whenever the model permits, reconstruction should return both an estimate and its uncertainty.

### Learn Unknown Parameters from Data

Parameters required for inference should, where possible, be estimated from observations rather than always supplied manually.

For example, Brownian volatility should eventually be estimated from irregular observations rather than requiring

```python
volatility = 1.5
```

to be known beforehand.

### Use Structure Only When It Helps

Graph, feature, and other relational information should not be assumed useful merely because they are available.

Additional structure should be evaluated against withheld or future observations.

A more complicated model should be retained only when it provides measurable improvement in reconstruction, forecasting, uncertainty calibration, or another declared objective.

### Evaluate Against Hidden Ground Truth

Reconstruction should be tested by deliberately withholding known observations.

For example,

```text
Complete:    1  2  3  4  5  6  7  8  9
Observed:    1     3     5     7     9
Withheld:       2     4     6     8
```

allows the withheld states to serve as ground truth.

Evaluation should eventually include:

- alternating holdout;
- random missingness;
- contiguous missing blocks;
- irregular missingness;
- partial node and feature observations;
- varying observation densities.

Metrics should measure both point accuracy and probabilistic calibration.

### Separate Information from Resolution

Introducing additional latent reconstruction points does not create new observational information.

Adaptive refinement may improve numerical resolution and computational allocation, but uncertainty should decrease only when justified by additional observations or by the statistical model.

Thus,

$$
\boxed{
\text{computational refinement}
\neq
\text{new statistical information}.
}
$$

### Keep the User Interface Simple

The internal mathematical machinery may involve stochastic processes, graph operators, conditional Gaussian distributions, state-space models, and Bayesian inference.

Ordinary workflows should nevertheless move toward interfaces such as

```python
model.fit(data)

result = model.reconstruct()

forecast = model.forecast(horizon=5)

next_observations = model.suggest_observations(
    budget=10
)
```

The framework should eventually accept common scientific data representations including:

- NumPy arrays;
- pandas DataFrames;
- CSV files.

### Preserve Reproducibility and Testability

Major inference methods should be accompanied by:

- explicit mathematical assumptions;
- deterministic unit tests where appropriate;
- statistical tests for stochastic algorithms;
- reproducible random-number handling;
- experiments against known or simulated ground truth;
- quantitative comparison with simpler baselines.

A method should not be retained merely because it produces a plausible-looking trajectory.

### Grow by Solving Concrete Limitations

Each substantial addition should answer three questions:

1. What practical limitation does it address?
2. How can a user invoke it through a clear interface?
3. How can its benefit be measured?

The project therefore favors justified additions over the accumulation of algorithms for their own sake.

---

## Core Representation

For a system of $N$ nodes, each possessing $d$ features, the state at time $t$ is represented by

$$
X(t)\in\mathbb{R}^{N\times d}.
$$

A trajectory evaluated at a finite ordered collection of $T$ times,

$$
t_0<t_1<\cdots<t_{T-1},
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

Partial observations are represented by values together with a Boolean mask declaring which quantities were observed.

A fundamental distinction of the framework is

$$
\boxed{
\text{state}
\neq
\text{observation}.
}
$$

A state describes the system at a particular time.

An observation describes that portion of the state which was actually measured.

---

## Core Data Structures

### `State`

Represents the state of a dynamical system at one point in time.

```text
time:   float
values: (N, d)
```

A `State` need not have been directly observed.

It may represent an underlying, simulated, reconstructed, or otherwise specified state.

---

### `Trajectory`

Represents states at a finite ordered collection of times.

```text
times:  (T,)
values: (T, N, d)
```

Times must be strictly increasing but need not be equally spaced.

For example,

```text
[0.0, 0.13, 0.47, 1.20, 2.85]
```

is a valid trajectory time grid.

---

### `Observation`

Represents the portion of a state that has actually been observed.

```text
time:   float
values: (N, d)
mask:   (N, d)
```

For node $i$ and feature $k$,

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

---

### `Graph`

Represents weighted relations among nodes.

```text
adjacency: (N, N)
```

where

```text
adjacency[i, j]
```

is the weight of the connection from node $i$ to node $j$.

The graph representation presently provides:

- node count;
- weighted degree;
- degree matrix;
- graph Laplacian

$$
L=D-A;
$$

- detection of directed and undirected adjacency matrices.

The graph stores relational structure.

The dynamical model, rather than the `Graph` object itself, determines how that structure affects the evolution of the state.

---

## Stochastic Dynamics

The first reference dynamics implemented by the project is Brownian motion,

$$
dX_t=\sigma\,dW_t.
$$

For two times $t_0<t_1$,

$$
X_{t_1}
=
X_{t_0}
+
\sigma\sqrt{t_1-t_0}\,Z,
$$

where

$$
Z\sim\mathcal N(0,I).
$$

Equivalently,

$$
X_{t_1}-X_{t_0}
\sim
\mathcal N
\left(
0,
\sigma^2(t_1-t_0)
\right).
$$

Because the variance depends directly upon the actual time interval, irregular sampling arises naturally.

The current Brownian implementation provides:

- one-step Brownian transitions;
- simulation across multiple time points;
- arbitrary strictly increasing time grids;
- reproducible simulation through NumPy random generators;
- scalar volatility;
- validation of temporal and volatility constraints.

Brownian motion serves as the first reference model upon which the inference architecture may be tested before more structured dynamics are introduced.

---

## Conditional Reconstruction

Suppose a latent trajectory is governed by a stochastic dynamical model while only part of the trajectory is observed.

The reconstruction problem may be written generally as

$$
p\left(
X_{\mathrm{missing}}
\mid
X_{\mathrm{observed}},
\text{dynamics},
G,
\theta
\right),
$$

where $G$ denotes relational structure and $\theta$ denotes model parameters.

The purpose of reconstruction is not merely to assign one value to every missing quantity.

Whenever possible, the framework should preserve the uncertainty associated with states which were never directly observed.

---

## Brownian Bridge Reconstruction

For Brownian dynamics, suppose two boundary states are known:

$$
X(t_L)=X_L,
\qquad
X(t_R)=X_R.
$$

For an intermediate time

$$
t_L<t<t_R,
$$

the conditional state follows a Brownian bridge distribution.

Its mean is

$$
\mathbb E[X_t\mid X_L,X_R]
=
X_L
+
\frac{t-t_L}{t_R-t_L}
(X_R-X_L),
$$

and its conditional variance is

$$
\operatorname{Var}(X_t\mid X_L,X_R)
=
\sigma^2
\frac{
(t-t_L)(t_R-t)
}{
t_R-t_L
}.
$$

The current implementation supports:

- single-point conditional reconstruction;
- multi-point conditional reconstruction;
- irregular target times;
- recursive temporal subdivision;
- reproducible stochastic reconstruction;
- empirical verification of conditional means;
- empirical verification of conditional variances;
- empirical verification of joint covariance.

Multi-point reconstruction is performed conditionally.

Intermediate states are not sampled independently from the two original endpoints.

Once a state is sampled, it becomes a boundary for the remaining subintervals, preserving the joint dependence of the Brownian bridge.

---

## Irregular-Time Reconstruction

Target times need not be uniformly spaced.

For a current interval

$$
[t_L,t_R],
$$

the reconstruction algorithm considers the temporal midpoint

$$
t_M=\frac{t_L+t_R}{2}
$$

and selects the available target time nearest to that midpoint.

The selected state is reconstructed conditionally and then divides the original problem into smaller conditional reconstruction problems.

For example,

```text
                         4.8
                       /     \
                    2.3       7.2
                   /          /  \
                1.1         5.2   9.4
               /
             0.4
```

may arise from an irregular collection of target times.

This subdivision is consistent with Brownian bridge uncertainty, whose conditional variance is greatest at the temporal midpoint of an interval.

---

## Evaluation

A reconstruction method should be judged against observations which were not supplied to it.

Given a complete trajectory,

$$
X_1,X_2,\ldots,X_9,
$$

one experiment may retain

$$
X_1,X_3,X_5,X_7,X_9
$$

while withholding

$$
X_2,X_4,X_6,X_8.
$$

The model reconstructs the withheld states, which may then be compared with their known values.

Future evaluation will examine reconstruction under different observation geometries and densities.

Point-error metrics will include quantities such as

$$
\operatorname{RMSE}
=
\sqrt{
\frac{1}{m}
\sum_{i=1}^{m}
(\hat X_i-X_i)^2
}
$$

and

$$
\operatorname{MAE}
=
\frac{1}{m}
\sum_{i=1}^{m}
|\hat X_i-X_i|.
$$

Because reconstruction is probabilistic, evaluation should also consider:

- negative log-likelihood;
- predictive interval coverage;
- uncertainty calibration.

This common evaluation framework will allow future dynamical models to be compared under the same withheld-observation experiments.

---

## Design Principles

### Sampling and Dynamics Are Separate

The observation mechanism determines

$$
\boxed{
\text{where and when information is observed},
}
$$

while the dynamical model determines

$$
\boxed{
\text{how the underlying system may evolve}.
}
$$

A dynamical model therefore need not generate observation times.

Given two successive observation times $t_i$ and $t_{i+1}$, the model governs the transition

$$
X(t_i)\longrightarrow X(t_{i+1}).
$$

---

### Graph and Dynamics Are Separate

A graph describes relationships among nodes.

It does not prescribe how those relationships affect the dynamics.

For example, a future model may use the graph Laplacian through

$$
dX_t
=
-\alpha L X_t\,dt
+
\sigma\,dW_t,
$$

but another dynamical model may use the same graph differently.

Structural representation and dynamical assumptions therefore remain separate.

---

### Reconstruction and Forecasting Are Distinct

When observations exist on both sides of an unknown state, the task is conditional reconstruction or smoothing.

When no future observation is available, the task becomes forecasting.

In probabilistic terms,

$$
p(X_t\mid Y_{\le T}),
\qquad t<T,
$$

describes smoothing, whereas

$$
p(X_{T+h}\mid Y_{\le T})
$$

describes forecasting.

Both may eventually be supported by a common dynamical model.

---

## Project Structure

The project is evolving toward the following organization:

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
│       │   └── brownian.py
│       │
│       ├── inference/
│       │   └── reconstruction/
│       │       └── brownian_bridge.py
│       │
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

---

# Development Roadmap

Each development stage introduces new machinery to address a limitation exposed by the preceding stage.

## V1 — Finite-Observation Inference

Establish a complete reconstruction and evaluation pipeline using Brownian dynamics as the reference model.

### Core Methods

- Brownian motion simulation;
- Brownian bridge reconstruction;
- maximum-likelihood estimation of volatility;
- reconstruction across multiple observed intervals;
- partial-observation handling;
- held-out reconstruction experiments;
- RMSE and MAE;
- likelihood-based evaluation;
- uncertainty calibration and interval coverage;
- high-level reconstruction interfaces;
- common scientific data adapters.

### Problem Addressed

> Can unobserved states be reconstructed reliably from finite, irregular, and incomplete observations?

Brownian dynamics provides a mathematically transparent baseline upon which the general inference architecture can be established.

---

## V2 — Structured Graph Dynamics

Introduce relational information among nodes.

A basic graph-coupled stochastic model may take the form

$$
dX_t
=
-\alpha L X_t\,dt
+
\sigma\,dW_t.
$$

### Core Methods

- graph Laplacians;
- graph diffusion;
- linear stochastic differential equations;
- matrix exponentials;
- structured covariance;
- graph-aware reconstruction;
- node-dependent information sharing.

### Problem Addressed

> Can observations from related nodes improve reconstruction of a partially observed system?

This version begins to distinguish the project from purely temporal interpolation and reconstruction methods.

---

## V3 — State-Space Inference and Forecasting

Separate latent states from noisy measurements.

A general state-space representation takes the form

$$
X_{t+\Delta}
=
F_\Delta X_t+w_t,
$$

$$
Y_t
=
H_tX_t+v_t.
$$

### Core Methods

- state-space models;
- Kalman filtering;
- Kalman smoothing;
- irregular-time transitions;
- measurement noise;
- probabilistic forecasting.

### Problem Addressed

> How should latent states be inferred when observations themselves are noisy, and how can the same model forecast future states?

This version unifies reconstruction, filtering, smoothing, and forecasting.

---

## V4 — Structure and Model Selection

Relational information should be evaluated rather than trusted automatically.

Candidate models may include

$$
M_1=\{\text{time}\},
$$

$$
M_2=\{\text{time, node graph}\},
$$

$$
M_3=\{\text{time, feature structure}\},
$$

and

$$
M_4=\{\text{time, node graph, feature structure}\}.
$$

### Core Methods

- ablation studies;
- held-out reconstruction;
- cross-validation;
- walk-forward validation;
- model comparison;
- structure selection.

### Problem Addressed

> Which supplied structures actually improve inference on unseen observations?

More complicated structure should survive only when supported by out-of-sample evidence.

---

## V5 — Adaptive Irregular-Time Inference

Irregular observation patterns may contain both densely constrained and sparsely constrained intervals.

Computational resolution should therefore be allocated adaptively.

For Brownian bridges, the greatest conditional variance inside an interval occurs at its midpoint:

$$
V_{\max}
=
\sigma^2
\frac{t_R-t_L}{4}.
$$

An interval may therefore be refined while

$$
V_{\max}>\varepsilon.
$$

### Core Methods

- adaptive interval refinement;
- uncertainty-based refinement;
- dyadic subdivision;
- explicit stopping tolerances;
- native irregular-time computation.

### Problem Addressed

> Where should additional computational resolution be placed when observation times are highly irregular?

Adaptive reconstruction improves numerical representation without pretending that computationally generated latent points constitute new observations.

---

## V6 — Active Observation

Move from inference under a fixed observation pattern to the design of future measurements.

Given a limited observation budget, the framework should eventually identify observations expected to provide the greatest value.

A possible objective is

$$
t^*
=
\arg\max_t
\operatorname{InformationGain}(t).
$$

### Core Methods

- posterior uncertainty;
- expected information gain;
- active sampling;
- observation-budget optimization;
- experimental design.

The resulting loop becomes

```text
Observe
   ↓
Infer
   ↓
Measure uncertainty
   ↓
Select the next observation
   ↓
Observe again
   ↺
```

### Problem Addressed

> If only a limited number of additional observations may be collected, where should they be placed?

---

## V7 — Bayesian Uncertainty

Earlier stages may estimate a parameter such as volatility by a single value

$$
\hat\sigma.
$$

When observations are limited, however, the parameter itself may be uncertain.

Bayesian inference instead considers

$$
p(\sigma\mid Y).
$$

The posterior predictive distribution then becomes

$$
p(X_{\mathrm{missing}}\mid Y)
=
\int
p(X_{\mathrm{missing}}\mid Y,\sigma)
p(\sigma\mid Y)
\,d\sigma.
$$

### Core Methods

- posterior parameter inference;
- posterior predictive distributions;
- parameter uncertainty;
- uncertainty-aware reconstruction;
- uncertainty-aware forecasting;
- eventual uncertainty over structural assumptions.

### Problem Addressed

> How should inference account for uncertainty in the dynamical model itself?

The final uncertainty should reflect both

$$
\boxed{
\text{state uncertainty}
+
\text{parameter uncertainty}.
}
$$

---

## Long-Term Architecture

The intended architecture may ultimately be summarized as

```text
Raw finite observations
          ↓
Data validation and representation
          ↓
Temporal geometry + node/feature structure
          ↓
Parameter estimation
          ↓
Dynamical model
          ↓
Reconstruction / Filtering / Forecasting
          ↓
Uncertainty quantification
          ↓
Held-out evaluation
          ↓
Structure selection
          ↓
Adaptive observation design
          ↺
```

The versions may therefore be understood as answering progressively broader questions:

$$
\text{V1: Can we reconstruct what we did not observe?}
$$

$$
\text{V2--V4: What structure should we use to reconstruct it?}
$$

$$
\text{V5--V7: What should we infer, trust, compute, and observe next?}
$$

---

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

---

## Testing

Run the complete test suite with:

```bash
python -m pytest -v
```

The current test suite covers:

- core state representation;
- trajectories;
- observations;
- graphs;
- Brownian simulation;
- Brownian bridge reconstruction;
- invalid temporal configurations;
- invalid volatility parameters;
- reproducibility;
- empirical Brownian bridge means;
- empirical Brownian bridge variances;
- empirical multi-point covariance.

---

## Status

### V1 — Finite-Observation Inference

#### Completed

- [x] `State`;
- [x] `Trajectory`;
- [x] `Observation`;
- [x] `Graph`;
- [x] irregular trajectory times;
- [x] Brownian one-step simulation;
- [x] Brownian trajectory simulation;
- [x] single-point Brownian bridge reconstruction;
- [x] multi-point Brownian bridge reconstruction;
- [x] irregular reconstruction times;
- [x] recursive conditional bridge construction;
- [x] empirical validation of bridge mean;
- [x] empirical validation of bridge variance;
- [x] empirical validation of multi-point covariance;
- [x] Brownian bridge reconstruction experiment.

#### Next

- [ ] estimate Brownian volatility from irregular observations;
- [ ] reconstruct across multiple observed intervals;
- [ ] introduce held-out reconstruction benchmarks;
- [ ] implement RMSE and MAE;
- [ ] implement likelihood and coverage diagnostics;
- [ ] reconstruct partial node and feature observations;
- [ ] introduce a high-level user-facing inference API;
- [ ] support NumPy, pandas, and CSV data adapters.

The immediate development path is therefore

$$
\boxed{
\text{Brownian bridge}
\rightarrow
\text{volatility estimation}
\rightarrow
\text{multi-observation reconstruction}
\rightarrow
\text{held-out evaluation}.
}
$$

---

## Development Principle

The project shall grow by solving concrete limitations rather than by accumulating algorithms.

For every substantial addition, the governing questions are:

> **What problem does it solve?**

> **How does the user use it?**

> **How do we demonstrate that it helps?**

The intended result is a framework combining rigorous stochastic inference, reproducible scientific software, and a simple practical interface.
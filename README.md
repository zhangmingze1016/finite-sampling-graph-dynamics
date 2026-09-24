# Finite-Sampling Graph Dynamics

A Python framework for reconstructing, forecasting, and quantifying uncertainty in dynamical systems from finite, irregular, and partial observations.

The primary objective is to support both **engineering applications and scientific research**: a usable inference tool with explicit assumptions, reproducible experiments, and quantitative evidence of reliability. Each release should deliver working software together with validation of its behavior and limitations.

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
X_{t_1} =
X_{t_0} +
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
\mathbb E[X_t\mid X_L,X_R] =
X_L +
\frac{t-t_L}{t_R-t_L}
(X_R-X_L),
$$

and its conditional variance is

$$
\operatorname{Var}(X_t\mid X_L,X_R) =
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
\operatorname{RMSE} =
\sqrt{
\frac{1}{m}
\sum_{i=1}^{m}
(\hat X_i-X_i)^2
}
$$

and

$$
\operatorname{MAE} =
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
dX_t =
-\alpha L X_t\,dt +
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

The following plan separates changes to existing code, near-term additions, and future capabilities. All unchecked work is planned, not implemented. The current implementation status is recorded in [Status](#status).

The V1--V7 labels below identify capability areas rather than a strict implementation order. In particular, basic Bayesian inference should be introduced early, and state-space inference should provide the foundation for graph dynamics.

## Changes to Existing Code

Priorities: **P0** establishes correctness and installability; **P1** completes a usable inference workflow; **P2** follows demonstrated needs and measured benefits.

| Priority | Change | Scope and acceptance criteria |
| --- | --- | --- |
| P0 | Repair test collection | Move the three nested trajectory tests to module scope, correct `np.zeroes`, and verify that the intended tests are collected and executed. |
| P0 | Correct time validation | Replace default relative-tolerance comparisons of initial timestamps with an explicit time precision policy; document time units. |
| P0 | Validate numerical values | Require finite values in complete states and trajectories. For observations, validate entries marked as observed while allowing NaN in unobserved entries. |
| P0 | Define boundary behavior | Specify behavior for zero volatility, empty inputs, no usable increments, and missing left or right observations. Explicitly define whether unequal endpoints at zero volatility are rejected or treated as a zero-noise bridge limit. |
| P0 | Complete package configuration | Populate `pyproject.toml` with package metadata, dependencies, supported Python versions, and a standard installation workflow verified in a clean environment. |
| P1 | Separate inference from sampling | Make clear that existing bridge functions return random posterior samples; expose point estimates, uncertainty, and sampling through distinct operations. |
| P1 | Specify array ownership | Define copying and sharing rules so that external array mutations cannot unexpectedly change states, observations, trajectories, or graphs. |
| P1 | Specify graph conventions | Document edge direction, weight constraints, and scale conventions; let each dynamical model validate the graph assumptions it requires. |
| P1 | Refactor experiment scripts | Separate configuration, execution, evaluation, and plotting, and avoid running experiments on module import. |
| P1 | Keep documentation aligned | Distinguish implemented behavior, the next deliverable, and long-term plans; label proposed APIs as examples rather than available features. |
| P2 | Improve bridge sampling performance | Replace front-removal from Python lists with a queue and use ordered-time searches where useful; benchmark improvements while preserving the joint distribution. |

## Near-Term Additions

The first usable release should complete Brownian inference and evaluation. A basic Bayesian extension, V7-A, should follow that closed loop before more complex models are introduced.

| Addition | Initial scope | Acceptance criteria |
| --- | --- | --- |
| Analytical posterior | Brownian conditional means, marginal variances, and covariance available on demand. | Match analytical formulas; deterministic predictions require no random generator. |
| Posterior result object | Means, variances, intervals, and joint path sampling. | Keep output semantics consistent and preserve temporal dependence in samples without requiring a dense covariance matrix for every query. |
| Multi-observation reconstruction | Locate neighboring real observations and reconstruct across multiple intervals. | Preserve exact observations and define behavior outside observed boundaries. |
| Partial and asynchronous observations | Give each node-feature component its own observation times under the independent Brownian model. | Never use masked values; require an appropriate prior or report a limitation when observations do not identify the requested result. |
| Volatility estimation | Maximum-likelihood estimation for driftless Brownian motion without measurement noise. | Verify estimation error in simulations and handle insufficient observations explicitly. |
| Basic forecasting | Propagate a Brownian distribution beyond the latest available observation. | Use only past information and recover the model's analytical mean and variance. |
| Basic Bayesian inference: V7-A | An inverse-gamma prior and posterior for variance, Student-t posterior prediction, and joint path sampling. | Match analytical and Monte Carlo results; assess prior sensitivity and small-sample calibration. |
| Simple user interface | A short workflow for fitting, posterior queries, and forecasting. | A complete example takes observations through inference and evaluation without exposing internal machinery. |
| Observation splitting | Random missingness, contiguous gaps, asynchronous observations, and node holdouts. | Keep training, validation, and test information separate, including parameter fitting and graph construction. |
| Evaluation metrics | RMSE, MAE, marginal predictive scores, interval coverage, and interval width. | Score genuinely withheld observations and distinguish marginal scores from joint trajectory likelihoods. |
| Reproducible experiment records | Save configuration, random seeds, parameters, software revision, and metrics. | Another user can reproduce the experiment and identify its assumptions. |
| Automated checks and documentation | Installation checks, tests, basic static checks, a quick start, and a complete example. | Run successfully in a clean environment and keep examples consistent with available APIs. |
| Data adapters | NumPy first, followed by DataFrame and CSV support. | Preserve timestamps, node identities, feature identities, and observation masks. |

Key correctness checks should include:

- Adding query times does not change the posterior distribution at existing query times; identical random samples across different grids are not required.
- Generated latent states never count as additional observations or tighten the parameter posterior merely because the query grid is denser.
- With fixed parameters in a linear Gaussian model, adding real observations does not increase posterior covariance in the positive-semidefinite ordering.
- A Bayesian path sample draws shared parameters once, then draws the entire path conditionally; it does not draw unrelated parameters at each query time.
- Scale, variance, and interval semantics are explicit, including cases where posterior moments do not exist.

The common benchmark suite should distinguish three questions:

1. **Known parameters, correct model:** is the inference implementation correct?
2. **Unknown parameters, correct model:** how much error comes from parameter estimation, and what changes when parameter uncertainty is included?
3. **Misspecified model:** how reliable are predictions and uncertainty estimates when assumptions fail?

## Future Capabilities and Dependencies

| Order | Capability | Dependency and intended evidence |
| --- | --- | --- |
| 1 | Ornstein--Uhlenbeck dynamics and exact irregular-time transitions | Complete the Brownian workflow first; introduce mean reversion and validate a second reference model. |
| 2 | Measurement noise, Kalman filtering, and RTS smoothing | Verify transition and observation models; distinguish latent process variation from measurement error. |
| 3 | Graph-coupled linear SDEs and correlated process noise | Retain a reliable independent temporal baseline and validate joint covariance. |
| 4 | Graph ablations and structure selection | Compare no graph, a correct graph, and perturbed graphs under the same evaluation protocol. |
| 5 | One complete real application | Secure suitable data and define domain constraints and metrics; deliver an end-to-end example and report. |
| 6 | Joint Bayesian parameter inference: V7-B | Validate the likelihood and investigate identifiability before inferring dynamical and noise parameters jointly. |
| 7 | Model diagnostics | Check residuals, calibration, misspecification, outliers, and prior sensitivity using posterior predictions. |
| 8 | Hierarchical Bayesian models | Introduce partial parameter pooling across nodes, features, or trajectories when data support heterogeneous behavior. |
| 9 | Budget-constrained observation design | Require credible uncertainty estimates and explicit measurement costs and target risks. |
| 10 | Computational scaling | Use profiling to justify sparse methods, low-rank approximations, caching, and on-demand computation; retain a small exact reference implementation. |
| 11 | Graph and model uncertainty: V7-C | Start with a small set of candidate structures after fixed-structure inference is reliable. |
| 12 | Nonlinear, time-varying, jump, and event dynamics | Add only when a concrete problem demonstrates that the existing models are insufficient. |

For linear Gaussian models, a shared transition interface should expose the quantities in

$$
X_{t+\Delta}=F_\Delta X_t+b_\Delta+w_t,
\qquad w_t\sim\mathcal N(0,Q_\Delta).
$$

Brownian, OU, and graph models can then share inference machinery. Extract common interfaces when at least two working models need them; do not force future event or nonlinear models into an unsuitable abstraction.

Keep the architecture in three cooperating parts:

- **Inference core:** data representations, dynamics, parameter estimation, and posterior results.
- **Experiment system:** observation splits, baselines, metrics, and reproducible reports.
- **Application adapters:** domain data, assumptions, constraints, and evaluation criteria.

## Development Workflow and Release Gates

Develop one complete workflow at a time:

1. Define the practical limitation, supported inputs, and excluded cases.
2. State the mathematical assumptions and expected behavior, including relevant limiting cases.
3. Write a minimal user-facing example before expanding the API.
4. Connect input handling, computation, results, and one runnable example.
5. Validate against analytical results, an independent implementation, or meaningful statistical experiments; measure performance when it is part of the objective.
6. Update documentation and deliver a focused commit or pull request describing behavior, evidence, and limitations.

Every release has two acceptance gates:

| Engineering acceptance | Research acceptance |
| --- | --- |
| Clear input/output contracts and explicit failure behavior. | Explicit assumptions, estimands, and evaluation objectives. |
| A short, usable API and an end-to-end example. | Simple, fair baselines and analytical or independent reference checks. |
| Reproducible installation, execution, and result records. | Separate fitting, model selection, and test data. |
| Measured runtime and memory for the intended scale. | Report uncertainty, failure cases, and model misspecification, not only successful examples. |
| Documentation matches implemented capabilities. | Experiments and their conclusions can be reproduced. |

Keep fast deterministic checks separate from heavier Monte Carlo validation, with statistical tolerances justified by sampling error. An implemented method is not complete merely because it produces a plausible plot.

The first deliverable is an installable Python package, a complete usage example, automated tests, and a reproducible accuracy and uncertainty report.

The recommended implementation sequence is:

```text
Correct existing implementation and tests
    -> Analytical posterior and multi-observation reconstruction
    -> Parameter estimation and held-out evaluation
    -> Basic Bayesian inference (V7-A)
    -> OU and noisy state-space inference
    -> Graph dynamics and graph misspecification experiments
    -> One complete real application
    -> Advanced Bayesian inference, observation design, and scaling
```

Defer simultaneous development of multiple application domains, unrestricted graph learning, large collections of deep-learning or sampling frameworks, GPU/distributed execution without a measured bottleneck, and plugin systems for models that do not yet exist.

## Capability Areas: V1--V7

Each capability area introduces machinery to address a concrete limitation. The descriptions below retain the original roadmap labels; the dependency order above determines implementation priority.

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
dX_t =
-\alpha L X_t\,dt +
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
X_{t+\Delta} =
F_\Delta X_t+w_t,
$$

$$
Y_t =
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
V_{\max} =
\sigma^2
\frac{t_R-t_L}{4}.
$$

For conditional path generation, this quantity may guide subdivision while

$$
V_{\max}>\varepsilon.
$$

This is uncertainty conditional on the current path boundaries, which may include sampled latent states. After integrating out those states, the posterior uncertainty conditional on the original observations is unchanged. Use numerical or downstream task error to justify computational stopping tolerances; do not interpret subdivision as new evidence or as a reduction of observational uncertainty.

### Core Methods

- adaptive interval refinement;
- conditional path refinement and task-specific numerical error control;
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
t^* =
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

Bayesian inference is a staged capability, not work deferred until all other roadmap areas are complete. State inference with fixed parameters, parameter uncertainty, and structural uncertainty are separate levels.

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
p(X_{\mathrm{missing}}\mid Y) =
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

### Staged Delivery

| Stage | Timing | Scope |
| --- | --- | --- |
| V7-A | After the Brownian reconstruction and evaluation workflow | Conjugate inference for a shared Brownian variance under explicit independence and noiseless-observation assumptions; Student-t predictions; joint path sampling; prior sensitivity and small-sample calibration. |
| V7-B | After reliable state-space and graph inference | Joint inference for coupling, mean reversion, process noise, and measurement noise; use the validated filtering likelihood where applicable, with identifiability and computational diagnostics. |
| V7-C | After fixed-structure inference and model comparison | Uncertainty over a limited set of candidate graphs and models, model averaging, and observation design that accounts for parameter or structural uncertainty. |

Point-estimate and Bayesian workflows should share result conventions while recording whether parameter uncertainty has been integrated out. Under the basic noiseless Brownian bridge model, learning the variance changes the predictive distribution, not the linear conditional mean. Bayesian inference does not automatically improve point accuracy or guarantee calibration under misspecification.

### Problem Addressed

> How should inference account for uncertainty in the dynamical model itself?

The final uncertainty should reflect both

$$
\boxed{
\text{state uncertainty} +
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

- [ ] repair trajectory test collection, timestamp comparisons, numerical validation, and package configuration;
- [ ] define boundary behavior, array ownership, and posterior result semantics;
- [ ] expose analytical posterior means, variances, and joint sampling;
- [ ] estimate Brownian volatility from irregular observations;
- [ ] reconstruct across multiple observed intervals;
- [ ] introduce held-out reconstruction benchmarks;
- [ ] implement RMSE and MAE;
- [ ] implement likelihood and coverage diagnostics;
- [ ] reconstruct partial node and feature observations;
- [ ] support basic forecasting using only available past observations;
- [ ] introduce a high-level user-facing inference API;
- [ ] support NumPy, pandas, and CSV data adapters;
- [ ] add a conjugate Bayesian Brownian baseline and posterior predictive evaluation (V7-A);
- [ ] provide automated checks, reproducible experiment records, and a complete usage example.

The immediate target is a usable Brownian inference workflow that estimates parameters, returns posterior results for irregular and partial observations, and reports held-out accuracy and uncertainty. V7-A then extends that same workflow to parameter uncertainty. The implementation sequence and engineering/research acceptance gates above define completion.

---

## Development Principle

The project shall grow by solving concrete limitations rather than by accumulating algorithms.

For every substantial addition, the governing questions are:

> **What problem does it solve?**

> **How does the user use it?**

> **How do we demonstrate that it helps?**

The intended result is a framework combining rigorous stochastic inference, reproducible scientific software, and a simple practical interface.

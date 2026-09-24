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

###

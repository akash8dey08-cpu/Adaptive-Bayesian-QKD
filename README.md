# Adaptive Physical-Layer Stabilization in BB84 Quantum Key Distribution

This repository contains a two-stage Qiskit and statistical simulation framework for adaptive physical-layer drift stabilization in the Bennett-Brassard 1984 (BB84) protocol. It models non-stationary channel birefringence (polarization drift) and demonstrates how active, recursive Bayesian pre-compensation can sustain positive secure key rates under continuous environmental turbulence.

This work was developed by Akash Dey as a research project evaluating finite-shot quantum parameter estimation, Pauli channel characterization, and physical-layer closed-loop control in quantum key distribution.

## Project Architecture

The simulation is partitioned into two distinct stages to mirror the progression from static hardware characterization to dynamic time-series control.

### 1. Stage I: Baseline Simulation (`/baseline_simulation`)
Characterizes the static operating boundaries of BB84 under stationary memoryless noise.
*   **Pauli Depolarization & SPAD Readout Errors:** Maps the 2D secure operating phase space across channel depolarizing probabilities and single-photon avalanche diode (SPAD) detector bit-flip rates.
*   **Amplitude Damping (Non-Unital):** Analyzes the basis-dependent error asymmetry between the computational ($Z$) and diagonal ($X$) bases under energy relaxation.
*   **Key Rate Evaluation:** Computes the asymptotic Shor-Preskill Secret Key Rate (SKR) to locate the 11.0% QBER protocol abort threshold.

### 2. Stage II: Adaptive Bayesian Time-Series (`/adaptive_bayesian`)
Refactors the BB84 protocol into a chronological Markov time-series engine ($T=100$ steps, $N=800$ pulses/step) where channel polarization evolves via a drifted Gaussian random walk ($\sigma_{\text{drift}}, \mu_{\text{drift}}$).
*   **Recursive Bayesian Drift Filter:** Implements a discrete 601-point circular Bayesian filter on $S^1 = [-\pi, \pi)$.
*   **Basis-Conditioned Likelihood:** Eliminates $Z$-basis shot-noise dilution by conditioning the binomial likelihood strictly on $X$-basis test bits, doubling interferometric visibility to $(1 - 2e_0)$.
*   **Active Pre-Compensation:** Bob applies a unitary phase rotation $R_z(-\hat{\theta}_{t-1})$ prior to measurement to continuously stabilize the channel.
*   **Basis-Asymmetric SKR:** Evaluates the decoupled CSS Shor-Preskill key rate via Jensen's inequality to maximize net cryptographic yield.

## Core Results

Evaluated across a 50-seed Monte Carlo ensemble ($4 \times 10^6$ simulated quantum pulses) at a severe aerial drift volatility of $\sigma_{\text{drift}} = 0.04$ rad/step, the recursive Bayesian filter:
*   Achieves a geodesic tracking RMSE of **$0.202 \pm 0.076$ rad**.
*   Clamps mean sifted QBER to **$1.58\% \pm 0.42\%$** (preventing the $12.20\%$ uncompensated protocol abort).
*   Sustains a **$99.96\%$** operational link availability.
*   Yields **$27,693 \pm 1,528$ distilled secret bits** per session (a $2.22\times$ net throughput multiplication over the uncompensated baseline).

## Getting Started

### Prerequisites
*   Python 3.8+
*   IBM Qiskit & Qiskit Aer
*   NumPy, SciPy, Matplotlib

```bash
pip install -r requirements.txt

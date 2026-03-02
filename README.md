# Solar-Power-Plant-Maintenance
Gamma-based Remaining-Useful-Life modelling and binary optimisation of solar cleaning schedules under capacity constraints (Gurobi).

This project models solar panel unit (SPU) efficiency degradation, estimates remaining useful life (RUL) using a parametric distribution, and computes an optimal cleaning schedule under capacity constraints via binary optimisation.

The workflow combines **stochastic modelling** (degradation + RUL sampling) with **constraint-based optimisation** (scheduling).

## Quant/Technical Summary

### 1) Degradation signal processing
- Loads SPU efficiency time series for multiple areas (e.g., West/East).
- Focuses on a single degradation cycle (data restricted to avoid partial cleaning cycles).

### 2) RUL modelling via parametric distribution
- Fits a **Gamma distribution** to degradation / RUL-related quantities.
- Uses the fitted distribution to generate **Monte Carlo samples** of RUL and reports expected RUL.

### 3) Cleaning schedule as a binary optimisation problem
Given:
- Daily costs (power price and cleaning charges)
- An RUL/deadline per SPU (must be cleaned before failure/degradation threshold)
- A daily capacity constraint (max number of cleanings per day)

Decision variables:
- \(x_{i,t} \in \{0,1\}\): clean SPU \(i\) on day \(t\)
- \(y_t \in \{0,1\}\): any cleaning performed on day \(t\) (fixed daily charge)

Objective:
- Minimise total cost:
  - fixed daily charge when any job is scheduled
  - per-job variable charges (unit charge + power price)

Constraints:
- Each SPU cleaned exactly once before its deadline (RUL)
- Daily cleaning capacity limit (e.g., ≤ 3 jobs/day)
- Linking constraints between \(x\) and \(y\)

Solved using **Gurobi**.

## Files / Data
The script expects the following CSV files in the project root:
- `spu_efficiency_West.csv`
- `spu_efficiency_East.csv`
- `cost_cleaning.csv`
- `RUL_North.csv`

## How to run

1) Install dependencies:
```bash
pip install -r requirements.txt

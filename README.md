# ConsensusEpidemicTrajectory

This repository is dedicated to the data availability for the paper [**Estimating Consensus Epidemic Trajectories via a Constrained Power Fr\'echet Mean with Functional Registration**](https://doi.org/10.48550/arXiv.2605.10069)

# How to conduct simulations

The scripts in this repository requires Python 3.9 or later.

## Package Installation

```bash
pip install requirements.txt
```

## Simulation with Parameters under Uncertainty

```bash
./simulation_mc.sh
```

## Summarizing Trajectories using Parameters from Multiple Research Groups

```bash
./realdata_literature.py --K 60
```

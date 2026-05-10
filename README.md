# ConsensusEpidemicTrajectory

This repository is dedicated to the data availability for the paper **Estimating Consensus Epidemic Trajectories across Compartmental Models via a Constrained Power Fr\'echet Mean with Functional Registration**.

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

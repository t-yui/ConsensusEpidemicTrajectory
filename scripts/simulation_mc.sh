#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

# J=5, K=30, R=1
python3 simulation_mc.py --J 5 --K 30 --R 1 >  logs/sim_mc_J5_K30_R1.log 2>&1

# J=10, K=30, R=1
python3 simulation_mc.py --J 10 --K 30 --R 1 > logs/sim_mc_J10_K30_R1.log 2>&1

# J=50, K=30, R=1
python3 simulation_mc.py --J 50 --K 30 --R 1 > logs/sim_mc_J50_K30_R1.log 2>&1

# J=10, K=30, R=2
python3 simulation_mc.py --J 10 --K 30 --R 2 > logs/sim_mc_J10_K30_R2.log 2>&1

# J=10, K=30, R=10
python3 simulation_mc.py --J 10 --K 30 --R 10 > logs/sim_mc_J10_K30_R10.log 2>&1

# J=10, K=15, R=1
python3 simulation_mc.py --J 10 --K 15 --R 1 >  logs/sim_mc_J10_K15_R1.log 2>&1

# J=10, K=60, R=1
python3 simulation_mc.py --J 10 --K 60 --R 1 > logs/sim_mc_J50_K60_R1.log 2>&1


#!/usr/bin/env bash



srun python GraphDecompositionBO/main.py --objective ising1_0.01 --n_eval 150 --random_seed_config $1
srun python GraphDecompositionBO/main.py --objective ising1_0.0001 --n_eval 150 --random_seed_config $1
srun python GraphDecompositionBO/main.py --objective ising1_0.0 --n_eval 150 --random_seed_config $1
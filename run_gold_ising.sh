#!/usr/bin/env bash



srun python GraphDecompositionBO/main.py --objective ising --lamda 0.01   --n_eval 150 --random_seed_config $1
srun python GraphDecompositionBO/main.py --objective ising --lamda 0.0001 --n_eval 150 --random_seed_config $1
srun python GraphDecompositionBO/main.py --objective ising --lamda 0.0    --n_eval 150 --random_seed_config $1
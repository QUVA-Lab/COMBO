#!/usr/bin/env bash



srun -t 0 python GraphDecompositionBO/main.py --parallel --objective ising --lamda 0.01   --n_eval 150 --no_graph_learning --random_seed_config $1
srun -t 0 python GraphDecompositionBO/main.py --parallel --objective ising --lamda 0.0001 --n_eval 150 --no_graph_learning --random_seed_config $1
srun -t 0 python GraphDecompositionBO/main.py --parallel --objective ising --lamda 0.0    --n_eval 150 --no_graph_learning --random_seed_config $1
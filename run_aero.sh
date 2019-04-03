#!/usr/bin/env bash



srun python GraphDecompositionBO/main.py --objective aerostruct1_0.01 --n_eval 250 --random_seed_config $1
srun python GraphDecompositionBO/main.py --objective aerostruct2_0.01 --n_eval 250 --random_seed_config $1
srun python GraphDecompositionBO/main.py --objective aerostruct3_0.01 --n_eval 250 --random_seed_config $1

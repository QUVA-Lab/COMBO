#!/usr/bin/env bash



srun python CombinatorialBO/main.py --objective contamination1_0.01 --n_eval 250 --random_seed_config $1
srun python CombinatorialBO/main.py --objective contamination1_0.0001 --n_eval 250 --random_seed_config $1
srun python CombinatorialBO/main.py --objective contamination1_0.0 --n_eval 250 --random_seed_config $1
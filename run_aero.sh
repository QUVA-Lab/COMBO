#!/usr/bin/env bash



srun python CombinatorialBO/main.py --objective aerostruct1_0.01 --n_eval 250 --random_seed_config $1
srun python CombinatorialBO/main.py --objective aerostruct2_0.01 --n_eval 250 --random_seed_config $1
srun python CombinatorialBO/main.py --objective aerostruct3_0.01 --n_eval 250 --random_seed_config $1

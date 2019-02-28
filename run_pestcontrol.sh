#!/usr/bin/env bash



srun python CombinatorialBO/main.py --objective pestcontrol --n_eval 250 --random_seed_config $1

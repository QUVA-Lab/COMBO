#!/usr/bin/env bash



srun python GraphDecompositionBO/main.py --objective centroid --n_eval 200 --random_seed_config $1
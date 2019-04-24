#!/usr/bin/env bash

python GraphDecompositionBO/experiments/GP_ablation/GP_regression.py --data_type 2 --train_data_scale 3
python GraphDecompositionBO/experiments/GP_ablation/GP_regression.py --data_type 2 --train_data_scale 3 --learn_decomposition

python GraphDecompositionBO/experiments/GP_ablation/GP_regression.py --data_type 2 --train_data_scale 4
python GraphDecompositionBO/experiments/GP_ablation/GP_regression.py --data_type 2 --train_data_scale 4 --learn_decomposition

python GraphDecompositionBO/experiments/GP_ablation/GP_regression.py --data_type 2 --train_data_scale 5
python GraphDecompositionBO/experiments/GP_ablation/GP_regression.py --data_type 2 --train_data_scale 5 --learn_decomposition

python GraphDecompositionBO/experiments/GP_ablation/GP_regression.py --data_type 3 --train_data_scale 1
python GraphDecompositionBO/experiments/GP_ablation/GP_regression.py --data_type 3 --train_data_scale 1 --learn_decomposition

python GraphDecompositionBO/experiments/GP_ablation/GP_regression.py --data_type 3 --train_data_scale 2
python GraphDecompositionBO/experiments/GP_ablation/GP_regression.py --data_type 3 --train_data_scale 2 --learn_decomposition

python GraphDecompositionBO/experiments/GP_ablation/GP_regression.py --data_type 3 --train_data_scale 3
python GraphDecompositionBO/experiments/GP_ablation/GP_regression.py --data_type 3 --train_data_scale 3 --learn_decomposition

python GraphDecompositionBO/experiments/GP_ablation/GP_regression.py --data_type 3 --train_data_scale 4
python GraphDecompositionBO/experiments/GP_ablation/GP_regression.py --data_type 3 --train_data_scale 4 --learn_decomposition

python GraphDecompositionBO/experiments/GP_ablation/GP_regression.py --data_type 3 --train_data_scale 5
python GraphDecompositionBO/experiments/GP_ablation/GP_regression.py --data_type 3 --train_data_scale 5 --learn_decomposition
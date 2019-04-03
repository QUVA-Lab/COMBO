#!/bin/bash

VENV_ROOT_DIR="`which python | xargs dirname | xargs dirname`/envs/GraphDecompositionBO"
if [ -d "$VENV_ROOT_DIR" ]; then
	cd "$VENV_ROOT_DIR"
	if [ ! -d "$VENV_ROOT_DIR/.git" ]; then
		echo "Data in GraphDecompositionBO is moved to virtual environment root directory."
  		mv GraphDecompositionBO GraphDecompositionBO_TBR
  		cp -a GraphDecompositionBO_TBR/. ./
  		rm -rf GraphDecompositionBO_TBR
	else
		echo "Data has been moved"
	fi
	source activate GraphDecompositionBO
	conda install --yes pytorch torchvision -c soumith -n GraphDecompositionBO
	pip install -r requirements.txt
else
	echo "Already in virtual environment"
	conda install --yes pytorch torchvision -c soumith -n GraphDecompositionBO
	pip install -r requirements.tx
fi

#!/bin/bash

VENV_ROOT_DIR="`which python | xargs dirname | xargs dirname`/envs/CombinatorialBO"
if [ -d "$VENV_ROOT_DIR" ]; then
	cd "$VENV_ROOT_DIR"
	if [ ! -d "$VENV_ROOT_DIR/.git" ]; then
		echo "Data in CombinatorialBO is moved to virtual environment root directory."
  		mv CombinatorialBO CombinatorialBO_TBR
  		cp -a CombinatorialBO_TBR/. ./
  		rm -rf CombinatorialBO_TBR
	else
		echo "Data has been moved"
	fi
	source activate CombinatorialBO 
	conda install --yes pytorch torchvision -c soumith -n CombinatorialBO
	pip install -r requirements.txt
else
	echo "Already in virtual environment"
	conda install --yes pytorch torchvision -c soumith -n CombinatorialBO
	pip install -r requirements.tx
fi

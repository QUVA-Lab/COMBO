#!/usr/bin/env bash

if [[ ":$PYTHONPATH:" != *":$(pwd):"* ]]; then
    echo "$(pwd) is not in PYTHONPATH, This will be added."
	export PYTHONPATH=$(pwd):$PYTHONPATH
else
    echo "$(pwd) already PYTHONPATH, No change in PYTHONPATH"
fi
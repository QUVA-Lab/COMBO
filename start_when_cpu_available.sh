#!/bin/bash

cpu_usage=100
while [ $cpu_usage -gt 50 ]; do
	echo "CPUs are heavily used now"
	date +%Y%m%d-%H:%M:%S
	cpu_usage=$(python -c "import psutil;print(int(sum(psutil.cpu_percent(interval=30.0,percpu=True))/psutil.cpu_count()))" 2>&1)
	echo $cpu_usage
done

echo $@
eval $@

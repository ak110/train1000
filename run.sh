#!/bin/bash -eux
GPU=$(nvidia-smi --list-gpus | wc -l)
mpirun -np $GPU python3 train1000.py $*

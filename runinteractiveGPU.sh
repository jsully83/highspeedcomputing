#!/bin/bash

srun --partition=gpu --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 --gres=gpu:v100-sxm2:1 --mem=64Gb --time=08:00:00 --export=ALL --pty /bin/bash


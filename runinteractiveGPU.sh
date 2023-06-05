#!/bin/bash

srun --partition=gpu --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --gres=gpu:v100-sxm2:1 --mem=8Gb --time=00:30:00 --export=ALL --pty /bin/bash

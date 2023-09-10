#!/bin/bash

srun --partition=gpu --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --gres=gpu:p100:1 --mem=8Gb --time=08:00:00 --export=ALL --pty /bin/bash

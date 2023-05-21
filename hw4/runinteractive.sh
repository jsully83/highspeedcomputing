#!/bin/bash

srun --partition=short --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --mem=64Gb --time=04:00:00 --export=ALL --pty /bin/bash


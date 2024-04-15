#!/bin/bash

# calculate step latency with 10 rps
python code/train_mdqn.py --episodes 300 --increment_action 50 --init_resources 1000 --rps 100
python code/train_mdqn.py --episodes 300 --increment_action 25 --init_resources 500 --rps 50 --alpha 0.6

#!/bin/bash

python3 run_experiment.py cifar10 FedGMM --n_learners 3 --n_gmm 3 --n_rounds 200 --bz 128 --lr 0.01 --lr_scheduler multi_step --log_freq 5 --device cuda --optimizer sgd --seed 1234 --logs_dir ./logs --verbose 1